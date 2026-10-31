#!/usr/bin/env python3
"""Dodo single-support experiment: pose search + contact-aware inverse dynamics.

Simulation only. No changes to the loaded XML, no support wrench, no welded feet.
Designed for build_model_v4.py (10 direct torque motors; floating base).

DEFAULT --mode balance deliberately initializes the robot in a searched,
collision-free one-foot pose at t=0. It tests balance, NOT the transition from
two feet. --mode transfer separately tests a restricted flat-foot transfer.
The latter may fail its pose search: that is NOT proof all transfers are impossible.
Optional push trials distinguish uninterrupted recovery from regaining balance
after a swing-foot touchdown. RECOVERED_AFTER_TOUCHDOWN is not a passing result.

Install: python -m pip install 'mujoco>=3.2' numpy scipy 'osqp>=1.0'
Run:     python dodo_single_support.py --model dodo_floating.xml
Mirror:  python dodo_single_support.py --support right --output right_balance
Push:    python dodo_single_support.py --push-newtons .25 --output push_balance
Transfer: python dodo_single_support.py --mode transfer --output transfer_trial
Video:   MUJOCO_GL=egl python dodo_single_support.py --video balance.mp4
         Video additionally needs imageio and imageio-ffmpeg.

Outputs PREFIX.csv and PREFIX.json. If the search fails before simulation,
only JSON is written. Exit status: 0 = measured success (or a successful
--pose-only check); 2 = experiment/setup failed. Always read the JSON result.
The JSON records parameters, model SHA256, software versions and thresholds.

Method:
  1. Search a crouched, collision-free single-support pose (balance mode), or
     IK continuation poses with both feet fixed followed by a lift (transfer).
  2. Check static force/torque feasibility of the single-support endpoint.
  3. Run a floating-base inverse-dynamics QP at each simulation step. Variables
     are qacc, motor torque and four 3D corner forces per supporting foot.
     Dynamics, motor limits, joint acceleration bounds and conservative Coulomb
     friction pyramids are hard constraints. Sole acceleration is stabilized.
     COM, torso and joint posture are weighted acceleration objectives.
  4. Transfer -> unload -> lift -> hold, with measured gates and timeouts.

QP contact is an ideal rigid-contact approximation; MuJoCo remains the actual
compliant forward dynamics. The predicted forces are NOT injected into MuJoCo.
Success uses measured contacts/forces, not the optimized force predictions.
No hardware robustness, torque-speed/thermal model or trained policy is claimed.
Development validation on the unchanged v4 generator with MuJoCo 3.17.0 and
OSQP 1.1.3: default left/right balance runs passed ten continuous seconds.
This is local nominal balance, not evidence of push recovery or a working gait.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

TYPES = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')
SIDES = ('left', 'right')


def rotation_error(target, current):
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def blend(t):
    x = np.clip(t, 0., 1.)
    return x**3 * (10. - 15.*x + 6.*x*x)


class Robot:
    def __init__(self, path):
        self.m = m = mujoco.MjModel.from_xml_path(str(path))
        if (m.nq, m.nv, m.nu) != (17, 16, 10):
            raise ValueError('Requires the floating-base v4 model: nq=17 nv=16 nu=10')
        if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected free base as the first joint')
        self.d = mujoco.MjData(m)
        self.tmp = mujoco.MjData(m)
        def ident(kind, name):
            value = mujoco.mj_name2id(m, kind, name)
            if value < 0:
                raise ValueError('Missing model object: ' + name)
            return value
        self.torso = ident(mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self.floor = ident(mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.feet = {s: ident(mujoco.mjtObj.mjOBJ_GEOM, s+'_foot_geom') for s in SIDES}
        self.names = [s+'_'+j for s in SIDES for j in TYPES]
        self.jids = np.array([ident(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names])
        if any(m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j] for j in self.jids):
            raise ValueError('Expected ten limited hinge joints')
        if hasattr(m, 'jnt_actfrclimited') and np.any(m.jnt_actfrclimited[self.jids]):
            raise ValueError('Joint-level actuator force limits are not supported; use motor limits')
        self.qa = m.jnt_qposadr[self.jids]
        self.va = m.jnt_dofadr[self.jids]
        self.aids = np.array([ident(mujoco.mjtObj.mjOBJ_ACTUATOR, n+'_motor') for n in self.names])
        for j, a in zip(self.jids, self.aids):
            if (m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or
                m.actuator_trnid[a, 0] != j or
                not np.allclose(m.actuator_gear[a], [1, 0, 0, 0, 0, 0]) or
                m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE or
                m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or
                m.actuator_gainprm[a, 0] != 1 or
                m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE or
                not m.actuator_ctrllimited[a]):
                raise ValueError('Only direct gear=1 torque motors supported')
        self.limits = m.actuator_ctrlrange[self.aids].copy()
        for i, a in enumerate(self.aids):
            if m.actuator_forcelimited[a]:
                self.limits[i, 0] = max(self.limits[i, 0], m.actuator_forcerange[a, 0])
                self.limits[i, 1] = min(self.limits[i, 1], m.actuator_forcerange[a, 1])
        self.S = np.zeros((m.nv, m.nu)); self.S[self.va, np.arange(m.nu)] = 1.
        self.mass = float(m.body_subtreemass[self.torso])
        self.weight = self.mass * abs(m.opt.gravity[2])
        self.dt = float(m.opt.timestep)
        self.ground_z = float(m.geom_pos[self.floor, 2])
        if (m.geom_type[self.floor] != mujoco.mjtGeom.mjGEOM_PLANE or
            not np.allclose(m.geom_quat[self.floor], [1, 0, 0, 0])):
            raise ValueError('This experiment requires a horizontal floor plane')
        for g in self.feet.values():
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError('This experiment requires v4 box feet')
        key = ident(mujoco.mjtObj.mjOBJ_KEY, 'bird_stand')
        mujoco.mj_resetDataKeyframe(m, self.d, key)
        mujoco.mj_forward(m, self.d)
        self.q0 = self.d.qpos.copy()
        self.anchors = {}
        for s, g in self.feet.items():
            p = self.d.geom_xpos[g].copy()
            p[2] = self.ground_z + m.geom_size[g, 2] - .0002
            self.anchors[s] = (p, np.eye(3))
        self.c0 = self.d.subtree_com[self.torso].copy()
        leg_geoms = {s: [ident(mujoco.mjtObj.mjOBJ_GEOM, s+'_'+j+'_geom')
                         for j in ('thigh', 'shank', 'foot')] for s in SIDES}
        self.collision_pairs = [(a, b) for a in leg_geoms['left'] for b in leg_geoms['right']]
        torso_geom = ident(mujoco.mjtObj.mjOBJ_GEOM, 'torso_geom')
        self.collision_pairs += [(torso_geom, g) for s in SIDES for g in leg_geoms[s][1:]]
        self.collision_pairs += [(leg_geoms[s][0], leg_geoms[s][2]) for s in SIDES]
        self.M = np.zeros((m.nv, m.nv))
        self.new_fullM = None

    def full_mass(self, d):
        if self.new_fullM is None:
            try:
                mujoco.mj_fullM(self.m, d, self.M)
                self.new_fullM = True
            except TypeError:
                mujoco.mj_fullM(self.m, self.M, d.qM)
                self.new_fullM = False
        elif self.new_fullM:
            mujoco.mj_fullM(self.m, d, self.M)
        else:
            mujoco.mj_fullM(self.m, self.M, d.qM)
        return self.M

    def kinematics(self, d):
        m = self.m
        jc = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, self.torso)
        jp = np.zeros_like(jc); jt = np.zeros_like(jc)
        mujoco.mj_jacBody(m, d, jp, jt, self.torso)
        feet = {}
        for s, g in self.feet.items():
            jp = np.zeros_like(jc); jr = np.zeros_like(jc)
            mujoco.mj_jacGeom(m, d, jp, jr, g)
            feet[s] = (d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(),
                       np.vstack((jp, jr)))
        return jc, jt, feet

    def corners(self, d, s):
        g = self.feet[s]
        hx, hy, hz = self.m.geom_size[g]
        local = np.array([[x, y, -hz] for x in (-hx, hx) for y in (-hy, hy)])
        return d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T

    def clearance(self, d, s):
        g = self.feet[s]; hx, hy, hz = self.m.geom_size[g]
        vertices = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])
        world = d.geom_xpos[g] + vertices @ d.geom_xmat[g].reshape(3, 3).T
        return float(world[:, 2].min()-self.ground_z)

    def pose_data(self, q):
        d = mujoco.MjData(self.m); d.qpos[:] = q
        mujoco.mj_forward(self.m, d)
        return d


def search_pose(r, seed, com, swing, height):
    """Pose search allows torso roll; fixed stance sole dominates the objective."""
    m = r.m; d = mujoco.MjData(m)
    rot0 = Rotation.from_quat(seed[3:7][[1, 2, 3, 0]]).as_rotvec()
    x0 = np.r_[seed[:3], rot0, seed[r.qa]]
    lo = np.r_[[-.3, -.3, .22], [-.8, -.6, -.5], m.jnt_range[r.jids, 0]+.005]
    hi = np.r_[[.3, .3, .55], [.8, .6, .5], m.jnt_range[r.jids, 1]-.005]
    def fill(x):
        d.qpos[:3] = x[:3]
        d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        d.qpos[r.qa] = x[6:]
        mujoco.mj_forward(m, d)
    def fun(x):
        fill(x)
        residual = []
        for s, g in r.feet.items():
            p, R = r.anchors[s]
            target = p.copy()
            if s == swing:
                target[2] += height
            residual.extend(500. * (d.geom_xpos[g] - target))
            # Both soles flat during transfer; airborne orientation is secondary.
            residual.extend((50. if height == 0 or s != swing else 2.) *
                            rotation_error(R, d.geom_xmat[g].reshape(3, 3)))
        residual.extend(250. * (d.subtree_com[r.torso] - com))
        rv = Rotation.from_matrix(d.xmat[r.torso].reshape(3, 3)).as_rotvec()
        residual.extend([.05*rv[0], 2.*rv[1], 2.*rv[2]])
        residual.extend(.025*x[6:])
        return residual
    result = least_squares(fun, np.clip(x0, lo+1e-8, hi-1e-8), bounds=(lo, hi),
                           max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-8)
    fill(result.x)
    errors = {}
    for s, g in r.feet.items():
        target = r.anchors[s][0].copy(); target[2] += height if s == swing else 0.
        errors[s+'_position_error_m'] = float(np.linalg.norm(d.geom_xpos[g]-target))
        errors[s+'_tilt_deg'] = float(np.rad2deg(np.arccos(np.clip(d.geom_xmat[g].reshape(3, 3)[2, 2], -1, 1))))
    errors['com_error_m'] = float(np.linalg.norm(d.subtree_com[r.torso]-com))
    errors['torso_roll_deg'] = float(np.rad2deg(Rotation.from_matrix(d.xmat[r.torso].reshape(3, 3)).as_euler('xyz')[0]))
    errors['optimizer_success'] = bool(result.success)
    return d.qpos.copy(), errors


def search_balance_pose(r, support, height, clearance):
    """Search COM-centered poses; allow swing leg placement and torso lean.

    This is a local numerical search, not a global feasibility proof. Strong
    stance/COM tasks and explicit geometric distances prevent crossed limbs
    from masquerading as a valid single-support posture.
    """
    m = r.m; d = mujoco.MjData(m)
    swing = 'right' if support == 'left' else 'left'
    sign = 1. if support == 'left' else -1.
    target_com = np.r_[r.anchors[support][0][:2], height]
    lo = np.r_[[-.3, -.3, .19], [-1., -.8, -1.], m.jnt_range[r.jids, 0]+.01]
    hi = np.r_[[.3, .3, .55], [1., .8, 1.], m.jnt_range[r.jids, 1]-.01]
    trials = []
    for swing_x in (.1, -.1):
        x0 = np.r_[r.q0[:3], [0., 0., 0.], r.q0[r.qa]]
        def fill(x):
            d.qpos[:3] = x[:3]
            d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
            d.qpos[r.qa] = x[6:]
            mujoco.mj_forward(m, d)
        def distances():
            return [mujoco.mj_geomDistance(m, d, a, b, .05, None) for a, b in r.collision_pairs]
        def fun(x):
            fill(x)
            g = r.feet[support]; p, R = r.anchors[support]
            residual = list(1000.*(d.geom_xpos[g]-p))
            residual.extend(200.*rotation_error(R, d.geom_xmat[g].reshape(3, 3)))
            g = r.feet[swing]
            residual.extend(d.geom_xpos[g, :2]-[swing_x, -.01*sign])
            residual.append(300.*(r.clearance(d, swing)-clearance))
            residual.extend(.05*rotation_error(np.eye(3), d.geom_xmat[g].reshape(3, 3)))
            residual.extend(500.*(d.subtree_com[r.torso]-target_com))
            residual.extend(.01*x[6:])
            residual.extend(1000.*min(0., distance-.005) for distance in distances())
            return residual
        fit = least_squares(fun, x0, bounds=(lo, hi), max_nfev=300,
                            ftol=1e-9, xtol=1e-9, gtol=1e-8)
        fill(fit.x)
        g = r.feet[support]
        checks = {'com_error_m': float(np.linalg.norm(d.subtree_com[r.torso]-target_com)),
                  'stance_position_error_m': float(np.linalg.norm(d.geom_xpos[g]-r.anchors[support][0])),
                  'stance_tilt_deg': float(np.rad2deg(np.arccos(np.clip(d.geom_xmat[g].reshape(3, 3)[2, 2], -1, 1)))),
                  'swing_clearance_m': r.clearance(d, swing),
                  'minimum_collision_distance_m': float(min(distances())),
                  'optimizer_success': bool(fit.success), 'swing_x_seed_m': swing_x}
        trials.append(checks)
        if (checks['com_error_m'] < .0005 and checks['stance_position_error_m'] < .0005 and
            checks['stance_tilt_deg'] < .1 and checks['swing_clearance_m'] > clearance-.001 and
            checks['minimum_collision_distance_m'] > .003):
            return d.qpos.copy(), target_com, trials
    raise RuntimeError('BALANCE_POSE_SEARCH_FAILED: '+json.dumps(trials))


class WholeBodyQP:
    def __init__(self, r, mu, margin):
        self.r, self.mu, self.margin = r, mu, margin
        self.solvers = {}

    def solve(self, d, active, qref, cref, foot_refs, force_share=None, static=False):
        r = self.r; m = r.m; nv = m.nv; nu = m.nu
        n = nv+nu+12*len(active)
        jc, jt, feet = r.kinematics(d)
        eps = 1e-5
        r.tmp.qpos[:] = d.qpos; r.tmp.qvel[:] = d.qvel
        mujoco.mj_integratePos(m, r.tmp.qpos, d.qvel, eps)
        mujoco.mj_forward(m, r.tmp)
        jc1, jt1, feet1 = r.kinematics(r.tmp)
        v = d.qvel
        tasks = []; targets = []; constraints = []; lower = []; upper = []
        def task(A, b, weight):
            tasks.append(np.sqrt(weight)*A); targets.append(np.sqrt(weight)*np.atleast_1d(b))
        def accel(J, J1, desired, weight):
            A = np.zeros((J.shape[0], n)); A[:, :nv] = J
            task(A, desired - (J1-J) @ v / eps, weight)
        def bound(A, l, u):
            constraints.append(np.atleast_2d(A)); lower.extend(np.broadcast_to(l, (np.atleast_2d(A).shape[0],)))
            upper.extend(np.broadcast_to(u, (np.atleast_2d(A).shape[0],)))
        dyn = np.zeros((nv, n)); dyn[:, :nv] = r.full_mass(d); dyn[:, nv:nv+nu] = -r.S
        Fz = {}
        for k, s in enumerate(active):
            g = r.feet[s]; bid = int(m.geom_bodyid[g]); start = nv+nu+12*k
            Fz[s] = np.zeros(n)
            tx = np.zeros(n); ty = np.zeros(n)
            center = feet[s][0]
            for j, p in enumerate(r.corners(d, s)):
                f = start+3*j
                jp = np.zeros((3, nv)); mujoco.mj_jac(m, d, jp, None, p, bid)
                dyn[:, f:f+3] = -jp.T
                Fz[s][f+2] = 1.
                row = np.zeros(n); row[f+2] = 1.; bound(row, 0., 3*r.weight)
                for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    row = np.zeros(n); row[f:f+3] = [a, b, -self.mu]
                    bound(row, -np.inf, 0.)
                tx[f+2] = p[0]-center[0]; ty[f+2] = p[1]-center[1]
            # Approximate horizontal support rectangle, valid for near-flat soles.
            for moment, half in ((tx, m.geom_size[g, 0]), (ty, m.geom_size[g, 1])):
                for sign in (-1., 1.):
                    bound(sign*moment-(half-self.margin)*Fz[s], -np.inf, 0.)
            pos, R, J = feet[s]; J1 = feet1[s][2]
            pt, Rt = foot_refs[s]
            desired = np.r_[350.*(pt-pos), 200.*rotation_error(Rt, R)]-35.*(J@v)
            A = np.zeros((6, n)); A[:, :nv] = J
            rhs = desired-(J1-J)@v/eps
            if not static:
                bound(A, rhs, rhs)
        bias = d.qfrc_passive-d.qfrc_bias
        bound(dyn, bias, bias)
        A = np.zeros((nu, n)); A[:, nv:nv+nu] = np.eye(nu)
        bound(A, r.limits[:, 0], r.limits[:, 1])
        if static:
            A = np.zeros((nv, n)); A[:, :nv] = np.eye(nv); bound(A, 0., 0.)
        else:
            # Joint stopping constraints over a short predictive horizon.
            A = np.zeros((nu, n)); A[np.arange(nu), r.va] = 1.
            horizon = .10
            q = d.qpos[r.qa]; qd = v[r.va]
            lim = m.jnt_range[r.jids]
            low = np.maximum(-150., 2*(lim[:, 0]+.01-q-horizon*qd)/horizon**2)
            high = np.minimum(150., 2*(lim[:, 1]-.01-q-horizon*qd)/horizon**2)
            if np.any(low > high):
                raise RuntimeError('Joint stopping bounds infeasible')
            bound(A, low, high)
            accel(jc, jc1, 80.*(cref-d.subtree_com[r.torso])-20.*(jc@v), 30000.)
            rd = r.pose_data(qref)
            accel(jt, jt1, 50.*rotation_error(rd.xmat[r.torso].reshape(3, 3),
                  d.xmat[r.torso].reshape(3, 3))-14.*(jt@v), .3)
            J = np.zeros((nu, nv)); J[np.arange(nu), r.va] = 1.
            accel(J, J, 45.*(qref[r.qa]-q)-12.*qd, .003)
            for s in SIDES:
                if s in active: continue
                pos, R, J = feet[s]; pt, Rt = foot_refs[s]
                accel(J[:3], feet1[s][2][:3], 100.*(pt-pos)-20.*(J[:3]@v), 150.)
                accel(J[3:], feet1[s][2][3:], 50.*rotation_error(Rt,R)-14.*(J[3:]@v), 1.)
        if force_share is not None and len(active) == 2:
            for s, share in force_share.items():
                task(Fz[s][None, :], [share*r.weight], .02)
        # Regularization chooses finite accelerations and modest motor/contact forces.
        reg = np.r_[np.full(nv, 1e-5), np.full(nu, 1e-3), np.full(n-nv-nu, 1e-5)]
        P = np.diag(reg); linear = np.zeros(n)
        for A, b in zip(tasks, targets):
            P += A.T@A; linear -= A.T@b
        A = np.vstack(constraints); l = np.asarray(lower); u = np.asarray(upper)
        key = (tuple(active), static, n, len(l))
        # Full fixed sparsity allows numeric updates even as zeros become nonzero.
        pr, pc = np.triu_indices(n)
        Pc = sparse.csc_matrix((P[pr, pc], (pr, pc)), shape=(n, n))
        ar = np.tile(np.arange(len(l)), n); ac = np.repeat(np.arange(n), len(l))
        Ac = sparse.csc_matrix((A.T.ravel(), (ar, ac)), shape=A.shape)
        if key not in self.solvers:
            solver = osqp.OSQP()
            solver.setup(P=Pc, q=linear, A=Ac, l=l, u=u, verbose=False,
                         eps_abs=2e-5, eps_rel=2e-5, max_iter=6000,
                         polishing=False, warm_starting=True)
            self.solvers[key] = solver
        else:
            solver = self.solvers[key]
            solver.update(Px=Pc.data, q=linear, Ax=Ac.data, l=l, u=u)
        result = solver.solve()
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError('QP '+result.info.status)
        x = result.x
        violation = float(max(np.max(l-A@x), np.max(A@x-u), 0.))
        if not np.isfinite(x).all() or violation > .02:
            raise RuntimeError(f'QP constraint violation {violation:.4g}')
        return x[nv:nv+nu], {'qp_iterations': int(result.info.iter),
            'qp_violation': violation, 'qp_solve_ms': 1000*result.info.run_time,
            **{s+'_predicted_fz_N': float(Fz[s]@x) if s in Fz else 0. for s in SIDES}}


def measurements(r, d, support):
    m = r.m; jc, _, feet = r.kinematics(d)
    row = {}
    unwanted = 0
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        allowed = any(pair == {g, r.floor} for g in r.feet.values())
        if not allowed and c.dist < -.0001:
            unwanted += 1
    row['unwanted_contacts'] = unwanted
    for s, g in r.feet.items():
        count = 0; total = np.zeros(3); moment = np.zeros(3)
        center = feet[s][0]; origin = np.r_[center[:2], r.ground_z]
        for i in range(d.ncon):
            c = d.contact[i]; a, b = int(c.geom1), int(c.geom2)
            if {a, b} != {g, r.floor}: continue
            wrench = np.zeros(6); mujoco.mj_contactForce(m, d, i, wrench)
            sign = 1. if b == g else -1.
            R = np.asarray(c.frame).reshape(3, 3)
            force = sign*R.T@wrench[:3]
            total += force
            moment += np.cross(c.pos-origin, force)+sign*R.T@wrench[3:]
            count += 1
        fz = float(total[2]); corners = r.corners(d, s)
        cop = np.array([-moment[1], moment[0]])/fz if fz > .1 else np.array([np.nan, np.nan])
        hx, hy = m.geom_size[g, :2]
        # These margins are evaluated in the sole's local x/y axes.
        cop_world = np.r_[origin[:2]+cop, r.ground_z]
        local = feet[s][1].T@(cop_world-center)
        margin = min(hx-abs(local[0]), hy-abs(local[1])) if fz > .1 else np.nan
        # Track speed of the lowest sole corner; center speed alone misses rocking.
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, None, corners[np.argmin(corners[:, 2])], int(m.geom_bodyid[g]))
        row.update({s+'_fz_N': fz, s+'_contacts': count,
                    s+'_clearance_m': r.clearance(d, s),
                    s+'_sole_speed_m_s': float(np.linalg.norm(jp@d.qvel)),
                    s+'_tilt_deg': float(np.rad2deg(np.arccos(np.clip(feet[s][1][2, 2], -1., 1.)))),
                    s+'_cop_margin_m': float(margin),
                    s+'_drift_m': float(np.linalg.norm(center[:2]-r.anchors[s][0][:2]))})
    com = d.subtree_com[r.torso]; vel = jc@d.qvel
    for i, axis in enumerate('xyz'):
        row['com_'+axis+'_m'] = float(com[i]); row['com_v'+axis+'_m_s'] = float(vel[i])
    g = r.feet[support]; center, R, _ = feet[support]
    projection = R.T@(np.r_[com[:2], r.ground_z]-center)
    row['com_support_margin_m'] = float(min(m.geom_size[g, 0]-abs(projection[0]), m.geom_size[g, 1]-abs(projection[1])))
    row['com_speed_m_s'] = float(np.linalg.norm(vel))
    row['torso_z_m'] = float(d.xpos[r.torso, 2])
    for n, a in zip(r.names, r.aids): row[n+'_tau_Nm'] = float(d.actuator_force[a])
    row['peak_abs_tau_Nm'] = float(np.max(np.abs(d.actuator_force[r.aids])))
    row['torque_utilization'] = float(np.max(np.abs(d.actuator_force[r.aids]) /
                                           np.maximum(.001, np.max(np.abs(r.limits), axis=1))))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='dodo_floating.xml')
    p.add_argument('--support', choices=SIDES, default='left')
    p.add_argument('--mode', choices=('balance', 'transfer'), default='balance',
                   help='balance initializes one-foot pose; transfer starts on two feet')
    p.add_argument('--com-height', type=float, default=.27,
                   help='COM height in metres for balance-mode pose search (default .27)')
    p.add_argument('--output', default='single_support')
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--transfer-seconds', type=float, default=6.)
    p.add_argument('--lift-seconds', type=float, default=2.)
    p.add_argument('--lift-mm', type=float, default=25.)
    p.add_argument('--hold-seconds', type=float, default=10.)
    p.add_argument('--com-height-offset-mm', type=float, default=0.)
    p.add_argument('--mu', type=float, default=.8)
    p.add_argument('--margin-mm', type=float, default=3.)
    p.add_argument('--pose-only', action='store_true')
    p.add_argument('--push-newtons', type=float, default=0.,
                   help='Optional signed external disturbance force; zero disables it')
    p.add_argument('--push-axis', choices=('x', 'y'), default='y')
    p.add_argument('--push-at', type=float, default=3.)
    p.add_argument('--push-seconds', type=float, default=.2)
    p.add_argument('--video', default='')
    p.add_argument('--fps', type=int, default=30)
    args = p.parse_args()
    if min(args.seconds, args.transfer_seconds, args.lift_seconds, args.hold_seconds, args.mu) <= 0:
        p.error('Durations and friction must be positive')
    if not 10 <= args.lift_mm <= 60 or not 0 < args.margin_mm < 10 or args.fps < 1:
        p.error('Use lift-mm 10..60, margin-mm 0..10, fps >= 1')
    if not .20 <= args.com_height <= .36:
        p.error('com-height must be between .20 and .36 metres')
    if args.push_at < 0 or args.push_seconds <= 0 or not np.isfinite(args.push_newtons):
        p.error('Push time must be nonnegative, duration positive, and force finite')
    if args.push_newtons and args.seconds < args.push_at+args.push_seconds+args.hold_seconds:
        p.error('seconds must allow the push and a full post-push hold')
    prefix = Path(args.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {'parameters': vars(args), 'result': 'SETUP_ERROR',
               'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'osqp': osqp.__version__},
               'success_thresholds': {'clearance_m': .008, 'swing_force_fraction_max': .02,
                'support_force_fraction_min': .85, 'com_speed_m_s_max': .025,
                'sole_speed_m_s_max': .01, 'support_tilt_deg_max': 3.,
                'support_drift_m_max': .005, 'cop_margin_m_min': .002,
                'com_margin_m_min': .002, 'torque_utilization_max': .98,
                'unwanted_contacts_max': 0, 'continuous_seconds': args.hold_seconds},
               'initialization': 'searched one-foot pose at t=0' if args.mode == 'balance' else 'bird_stand keyframe',
               'tests_transition_from_two_feet': args.mode == 'transfer'}
    rows = []; writer = renderer = None; started = time.perf_counter(); exit_code = 2
    try:
        summary['model_sha256'] = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
        r = Robot(args.model); d = r.d; m = r.m
        summary['mass_kg'] = r.mass
        summary['timestep_s'] = r.dt
        support = args.support; swing = 'right' if support == 'left' else 'left'
        lift = args.lift_mm/1000.; margin = args.margin_mm/1000.
        if margin >= min(m.geom_size[g, 1] for g in r.feet.values()):
            raise ValueError('Margin exceeds foot width')
        if args.mu > min(m.geom_friction[r.floor, 0], *(m.geom_friction[g, 0] for g in r.feet.values())):
            raise ValueError('QP friction must not exceed model surface friction')
        qp = WholeBodyQP(r, args.mu, margin)
        if args.mode == 'balance':
            print('Searching collision-free single-support pose (balance-only initialization)...', flush=True)
            qlift, cref, checks = search_balance_pose(r, support, args.com_height, lift)
            print('Endpoint pose:', json.dumps(checks[-1]), flush=True)
            summary['pose_search'] = {'trials': checks, 'single_support_qpos': qlift.tolist(),
                                      'com_target_m': cref.tolist()}
            d.qpos[:] = qlift; d.qvel[:] = 0.; mujoco.mj_forward(m, d)
            balance_refs = {s: (d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy())
                            for s, g in r.feet.items()}
            r.anchors[support] = (balance_refs[support][0].copy(), balance_refs[support][1].copy())
            poses = [qlift.copy() for _ in range(11)]
            foot_refs = balance_refs
        else:
            cref = r.c0.copy(); cref[:2] = r.anchors[support][0][:2]
            cref[2] += args.com_height_offset_mm/1000.
            poses = [r.q0.copy()]; checks = []
            print('Searching restricted flat-foot transfer poses...', flush=True)
            for fraction in np.linspace(0., 1., 11)[1:]:
                target = r.c0+fraction*(cref-r.c0)
                q, errors = search_pose(r, poses[-1], target, swing, 0.)
                poses.append(q); checks.append(errors)
            qlift, lift_check = search_pose(r, poses[-1], cref, swing, lift)
            summary['pose_search'] = {'transfer': checks, 'lift': lift_check,
                                      'single_support_qpos': qlift.tolist(), 'com_target_m': cref.tolist()}
            print('Endpoint pose:', json.dumps(lift_check), flush=True)
            if any(c['com_error_m'] > .003 or max(c[s+'_position_error_m'] for s in SIDES) > .003 or
                   max(c[s+'_tilt_deg'] for s in SIDES) > 2. for c in checks):
                raise RuntimeError('TRANSFER_POSE_SEARCH_FAILED: restricted flat-foot path not found; this does not rule out other transfers')
            if (lift_check['com_error_m'] > .003 or max(lift_check[s+'_position_error_m'] for s in SIDES) > .003 or
                lift_check[support+'_tilt_deg'] > 2.):
                raise RuntimeError('TRANSFER_POSE_SEARCH_FAILED: lift target not found')
            foot_refs = {s: (a[0].copy(), a[1].copy()) for s, a in r.anchors.items()}
            foot_refs[swing][0][2] += lift
        static_tau, _ = qp.solve(r.pose_data(qlift), [support], qlift, cref, foot_refs, static=True)
        summary['static_peak_tau_Nm'] = float(np.max(np.abs(static_tau)))
        print(f'Static endpoint feasible; peak torque {np.max(np.abs(static_tau)):.3f} Nm', flush=True)
        if args.pose_only:
            summary['result'] = 'POSE_FEASIBLE_NOT_DYNAMICALLY_TESTED'; exit_code = 0
            return exit_code
        if args.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(m, width=960, height=720)
            writer = imageio.get_writer(args.video, fps=args.fps, codec='libx264', quality=7)
        phase = 'hold' if args.mode == 'balance' else 'settle'
        phase_start = 0.; dwell = 0.; success_time = best_time = 0.
        frame = 0; last_tau = np.zeros(m.nu); log_every = max(1, round(.02/r.dt))
        touchdowns = []; was_touching = False; airborne_established = args.mode == 'balance'
        summary['single_support_touchdowns'] = touchdowns
        summary['result'] = 'TIMEOUT'; events = []; summary['events'] = events
        def transition(new):
            nonlocal phase, phase_start, dwell
            print(f'{d.time:.3f}s: {phase} -> {new}', flush=True)
            events.append({'time_s': float(d.time), 'from': phase, 'to': new})
            phase = new; phase_start = float(d.time); dwell = 0.
        for step in range(int(np.ceil(args.seconds/r.dt))):
            # Optional test disturbance only; never apply an assisting base wrench.
            d.xfrc_applied[:] = 0.
            push_active = bool(args.push_newtons and args.push_at <= d.time < args.push_at+args.push_seconds)
            if push_active:
                d.xfrc_applied[r.torso, 'xyz'.index(args.push_axis)] = args.push_newtons
            # Forward refresh keeps state, contact forces and diagnostics synchronized.
            mujoco.mj_forward(m, d)
            measured = measurements(r, d, support)
            elapsed = float(d.time)-phase_start
            if phase == 'settle':
                good = (measured['com_speed_m_s'] < .02 and all(measured[s+'_fz_N'] > .2*r.weight for s in SIDES))
                dwell = dwell+r.dt if good else 0.
                if elapsed > 1. and dwell > .3: transition('transfer')
            elif phase == 'transfer':
                good = (elapsed >= args.transfer_seconds and np.linalg.norm(
                    d.subtree_com[r.torso]-cref) < .008 and measured['com_speed_m_s'] < .025)
                dwell = dwell+r.dt if good else 0.
                if dwell > .4: transition('unload')
            elif phase == 'unload':
                good = (measured[swing+'_fz_N'] < .05*r.weight and
                        measured['com_support_margin_m'] > .003 and measured['com_speed_m_s'] < .025 and
                        measured[support+'_tilt_deg'] < 3.)
                dwell = dwell+r.dt if good else 0.
                if dwell > .3: transition('lift')
            elif phase == 'lift':
                if elapsed >= args.lift_seconds and measured[swing+'_clearance_m'] > .008 and measured[swing+'_contacts'] == 0:
                    transition('hold')
            elapsed = float(d.time)-phase_start
            if (phase in ('settle', 'unload') and elapsed > 6. or
                phase == 'transfer' and elapsed > args.transfer_seconds+6. or
                phase == 'lift' and elapsed > args.lift_seconds+4.):
                summary['result'] = 'PHASE_TIMEOUT_'+phase.upper(); break
            fraction = 0. if phase == 'settle' else blend(elapsed/args.transfer_seconds) if phase == 'transfer' else 1.
            idx = min(int(fraction*10), 9); alpha = fraction*10-idx
            qref = poses[idx].copy()
            # Integrate tangent difference to interpolate the base quaternion correctly.
            diff = np.zeros(m.nv); mujoco.mj_differentiatePos(m, diff, 1., poses[idx], poses[idx+1])
            mujoco.mj_integratePos(m, qref, diff, alpha)
            com_target = r.c0+fraction*(cref-r.c0)
            foot_refs = {s: (a[0].copy(), a[1].copy()) for s, a in r.anchors.items()}
            active = list(SIDES)
            if phase in ('lift', 'hold'):
                lift_fraction = blend(elapsed/args.lift_seconds) if phase == 'lift' else 1.
                foot_refs[swing][0][2] += lift_fraction*lift
                diff = np.zeros(m.nv); mujoco.mj_differentiatePos(m, diff, 1., poses[-1], qlift)
                qref = poses[-1].copy(); mujoco.mj_integratePos(m, qref, diff, lift_fraction)
                active = [support]
            if args.mode == 'balance':
                qref = qlift; com_target = cref
                foot_refs = balance_refs; active = [support]
            share = .5+.5*fraction
            tau, stats = qp.solve(d, active, qref, com_target, foot_refs,
                                  {support: share, swing: 1.-share})
            # Clamp only solver roundoff; significant violations cause an explicit failure.
            if np.any(tau < r.limits[:, 0]-.01) or np.any(tau > r.limits[:, 1]+.01):
                raise RuntimeError('Motor torque bound violation')
            last_tau = np.clip(tau, r.limits[:, 0], r.limits[:, 1])
            d.ctrl[r.aids] = last_tau
            mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
            measured = measurements(r, d, support)
            if measured[swing+'_contacts'] == 0 and measured[swing+'_clearance_m'] > .008:
                airborne_established = True
            touching = bool(measured[swing+'_contacts'] > 0)
            if airborne_established and touching and not was_touching:
                touchdowns.append({'time_s': float(d.time), 'force_N': measured[swing+'_fz_N'],
                                   'clearance_m': measured[swing+'_clearance_m']})
            was_touching = touching
            good = (phase == 'hold' and measured[swing+'_clearance_m'] >= .008 and
                    measured[swing+'_contacts'] == 0 and abs(measured[swing+'_fz_N']) < .02*r.weight and
                    measured[support+'_fz_N'] > .85*r.weight and measured[support+'_contacts'] >= 2 and
                    measured[support+'_sole_speed_m_s'] < .01 and measured[support+'_tilt_deg'] < 3. and
                    measured[support+'_drift_m'] < .005 and measured[support+'_cop_margin_m'] > .002 and
                    measured['com_support_margin_m'] > .002 and measured['com_speed_m_s'] < .025 and
                    measured['torque_utilization'] < .98 and measured['unwanted_contacts'] == 0 and
                    (not args.push_newtons or d.time >= args.push_at+args.push_seconds))
            success_time = success_time+r.dt if good else 0.
            best_time = max(best_time, success_time)
            row = {'time_s': float(d.time), 'phase': phase, 'support': support,
                   'success_now': int(good), 'success_streak_s': success_time,
                   'push_active': int(push_active), 'push_force_N': args.push_newtons if push_active else 0.,
                   **measured, **stats}
            if step % log_every == 0: rows.append(row)
            if writer is not None and d.time >= frame/args.fps:
                renderer.update_scene(d); writer.append_data(renderer.render()); frame += 1
            if not np.isfinite(d.qpos).all() or measured['torso_z_m'] < .25 or abs(d.qpos[1]) > .25:
                summary['result'] = 'FALL_OR_EXCESSIVE_DRIFT'; rows.append(row); break
            if success_time >= args.hold_seconds:
                if touchdowns:
                    summary['result'] = 'RECOVERED_AFTER_TOUCHDOWN'
                    exit_code = 2
                else:
                    summary['result'] = 'SUCCESS_BALANCE_ONLY' if args.mode == 'balance' else 'SUCCESS_TRANSFER_AND_BALANCE'
                    exit_code = 0
                rows.append(row); break
        summary.update({'simulation_time_s': float(d.time), 'best_success_streak_s': best_time,
                        'last_phase': phase, 'final_measurements': measurements(r, d, support)})
    except Exception as exc:
        summary['result'] = 'ERROR'; summary['error'] = str(exc)
        if 'r' in locals():
            summary['simulation_time_s'] = float(r.d.time)
            summary['last_phase'] = locals().get('phase', 'pose_search')
            summary['best_success_streak_s'] = locals().get('best_time', 0.)
            if 'support' in locals():
                summary['final_measurements'] = measurements(r, r.d, support)
        print('Experiment stopped:', exc, flush=True)
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        summary['wall_time_s'] = time.perf_counter()-started
        summary['logged_samples'] = len(rows)
        if rows:
            summary['logged_peak_abs_tau_Nm'] = max(row['peak_abs_tau_Nm'] for row in rows)
            summary['logged_joint_rms_tau_Nm'] = {n: float(np.sqrt(np.mean([row[n+'_tau_Nm']**2 for row in rows])))
                                                 for n in r.names}
        if rows:
            with prefix.with_suffix('.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        # Strict JSON: NaN diagnostics (e.g. absent foot CoP) become null.
        def clean(x):
            if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list): return [clean(v) for v in x]
            if isinstance(x, float) and not np.isfinite(x): return None
            return x
        prefix.with_suffix('.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+'\n')
        print(summary['result'], '| summary:', prefix.with_suffix('.json'), flush=True)
        if rows: print('CSV:', prefix.with_suffix('.csv'), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
