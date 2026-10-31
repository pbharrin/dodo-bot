#!/usr/bin/env python3
"""Dodo v2: active unloading with measured force feedback and bounded IK updates.

Standalone simulation experiment on the unchanged v4 model. Starts from two
feet and applies motor torques only. No qpos reset after initialization,
external support, modified XML, or injected contact forces.

Run: python dodo_transition_v2.py --model dodo_floating.xml --output transition_v2_left
Video on headless Ubuntu (install libosmesa6, imageio, imageio-ffmpeg):
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python dodo_transition_v2.py \
    --model dodo_floating.xml --output transition_v2_left --video transition_v2_left.mp4

The initial geometric plan is the v1 heel-pivot path. During unloading, a
smooth load ramp and bounded PI feedback set a SOFT QP force objective.
Filtered measured load also requests small lateral COM/posture corrections;
each IK update is checked before acceptance and blended into the command.
An unreachable correction is rejected, recorded, and prevents integrator windup.
The controller never equates predicted QP force with measured unloading.
Contact acceleration tracking is hard by default, matching v1. Optional
--contact-soft-weight 1000000 tests compliant tracking during unloading only.
Anchors remain fixed; measured drift, tilt and force checks govern release and
stopping. Dynamics, joint stopping and motor bounds stay hard in either mode.

CSV/JSON record desired/commanded/predicted/measured force, failed release
gates, joint margins, tracking errors, QP residuals and posture update history.
--disable-active-unloading provides a v1-like control comparison in this file.
--plan-only checks geometric waypoints and endpoint static feasibility only.
Exit 0 means measured success or plan-only completion (read JSON); exit 2 means
failure. A successful plan does not establish a successful transition.
"""

import argparse
import csv
import hashlib
import json
import time
import uuid
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy
from scipy import sparse
from scipy.optimize import least_squares, linprog
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


def collision_distance_gradient(r, state, ga, gb):
    """Gradient of SIGNED geom distance, including shallow penetration.

    The closest-point segment reverses when signed distance becomes negative;
    its Euclidean direction alone then gives the opposite gradient.
    """
    m=r.m;segment=np.zeros(6)
    dist=mujoco.mj_geomDistance(m,state,ga,gb,.05,segment)
    delta=segment[3:]-segment[:3];norm=np.linalg.norm(delta)
    if norm<1e-8:return dist,None
    normal=(1. if dist>=0. else -1.)*delta/norm
    ja=np.zeros((3,m.nv));jb=np.zeros((3,m.nv))
    mujoco.mj_jac(m,state,ja,None,segment[:3],int(m.geom_bodyid[ga]))
    mujoco.mj_jac(m,state,jb,None,segment[3:],int(m.geom_bodyid[gb]))
    return dist,normal@(jb-ja)


class WholeBodyQP:
    def __init__(self, r, mu, margin):
        self.r, self.mu, self.margin = r, mu, margin
        self.solvers = {}

    def solve(self, d, active, qref, cref, foot_refs, force_share=None, static=False, force_weight=.02, contact_soft_weight=None):
        r = self.r; m = r.m; nv = m.nv; nu = m.nu
        n = nv+nu+12*len(active)
        jc, jt, feet = r.kinematics(d)
        eps = 1e-5
        r.tmp.qpos[:] = d.qpos; r.tmp.qvel[:] = d.qvel
        mujoco.mj_integratePos(m, r.tmp.qpos, d.qvel, eps)
        mujoco.mj_forward(m, r.tmp)
        jc1, jt1, feet1 = r.kinematics(r.tmp)
        v = d.qvel
        tasks = []; targets = []; constraints = []; lower = []; upper = []; labels = []
        def task(A, b, weight):
            tasks.append(np.sqrt(weight)*A); targets.append(np.sqrt(weight)*np.atleast_1d(b))
        def accel(J, J1, desired, weight):
            A = np.zeros((J.shape[0], n)); A[:, :nv] = J
            task(A, desired - (J1-J) @ v / eps, weight)
        def bound(A, l, u, label='unspecified'):
            constraints.append(np.atleast_2d(A)); lower.extend(np.broadcast_to(l, (np.atleast_2d(A).shape[0],)))
            upper.extend(np.broadcast_to(u, (np.atleast_2d(A).shape[0],)))
            labels.extend(f'{label}[{i}]' for i in range(np.atleast_2d(A).shape[0]))
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
                row = np.zeros(n); row[f+2] = 1.; bound(row, 0., 3*r.weight if j in active[s] else 0., f'{s}_corner{j}_normal_force')
                for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    row = np.zeros(n); row[f:f+3] = [a, b, -self.mu]
                    bound(row, -np.inf, 0., f'{s}_corner{j}_friction')
                tx[f+2] = p[0]-center[0]; ty[f+2] = p[1]-center[1]
            # Approximate horizontal support rectangle, valid for near-flat soles.
            for moment, half in (((tx, m.geom_size[g, 0]), (ty, m.geom_size[g, 1])) if len(active[s]) == 4 else []):
                for sign in (-1., 1.):
                    bound(sign*moment-(half-self.margin)*Fz[s], -np.inf, 0., f'{s}_cop')
            pos, R, J = feet[s]; J1 = feet1[s][2]
            pt, Rt = foot_refs[s]
            desired = np.r_[2000.*(pt-pos), 500.*rotation_error(Rt, R)]-90.*(J@v)
            A = np.zeros((6, n)); A[:, :nv] = J
            rhs = desired-(J1-J)@v/eps
            if not static:
                if len(active[s]) == 4:
                    if contact_soft_weight is None:
                        bound(A, rhs, rhs, f'{s}_sole_acceleration')
                    else:
                        task(A, rhs, contact_soft_weight)
                else:
                    for corner in active[s]:
                        point = r.corners(d, s)[corner]
                        point1 = r.corners(r.tmp, s)[corner]
                        Jp = np.zeros((3, nv)); Jp1 = np.zeros((3, nv))
                        mujoco.mj_jac(m, d, Jp, None, point, bid)
                        mujoco.mj_jac(m, r.tmp, Jp1, None, point1, bid)
                        local = R.T@(point-pos)
                        target = pt+Rt@local
                        rhs_point = 2000.*(target-point)-90.*(Jp@v)-(Jp1-Jp)@v/eps
                        Ap = np.zeros((3, n)); Ap[:, :nv] = Jp
                        if contact_soft_weight is None:
                            bound(Ap, rhs_point, rhs_point, f'{s}_pivot_acceleration')
                        else:
                            # MuJoCo contacts are compliant. A hard stabilized
                            # pivot can conflict with joint stopping constraints
                            # near joint limits. This cost permits tracking error;
                            # dynamics, joint stopping and torque bounds stay hard.
                            task(Ap, rhs_point, contact_soft_weight)
                    # Rocking contacts leave orientation free; track the planned rotation softly.
                    accel(J[3:], J1[3:], 50.*rotation_error(Rt,R)-14.*(J[3:]@v), 300.)
        bias = d.qfrc_passive-d.qfrc_bias
        bound(dyn, bias, bias, 'dynamics')
        A = np.zeros((nu, n)); A[:, nv:nv+nu] = np.eye(nu)
        bound(A, r.limits[:, 0], r.limits[:, 1], 'motor_torque')
        if static:
            A = np.zeros((nv, n)); A[:, :nv] = np.eye(nv); bound(A, 0., 0., 'static_acceleration')
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
            bound(A, low, high, 'joint_stopping')
            # Distance barriers keep neighboring limbs separated during tracking.
            # Recompute closest points at a directional finite-difference state.
            def distance_gradient(state, ga, gb):
                return collision_distance_gradient(r,state,ga,gb)
            for ga,gb in r.collision_pairs:
                dist,grad=distance_gradient(d,ga,gb)
                if dist>=.03 or grad is None: continue
                _,grad1=distance_gradient(r.tmp,ga,gb)
                if grad1 is None: grad1=grad
                row=np.zeros(n);row[:nv]=grad
                rhs=-400.*(dist-.004)-40.*(grad@v)-(grad1-grad)@v/eps
                bound(row,rhs,np.inf,f'collision_{m.geom(ga).name}_{m.geom(gb).name}')
            accel(jc, jc1, 80.*(cref-d.subtree_com[r.torso])-20.*(jc@v), 30000.)
            rd = r.pose_data(qref)
            accel(jt, jt1, 50.*rotation_error(rd.xmat[r.torso].reshape(3, 3),
                  d.xmat[r.torso].reshape(3, 3))-14.*(jt@v), .3)
            J = np.zeros((nu, nv)); J[np.arange(nu), r.va] = 1.
            accel(J, J, 45.*(qref[r.qa]-q)-12.*qd, 30.)
            for s in SIDES:
                if s in active: continue
                pos, R, J = feet[s]; pt, Rt = foot_refs[s]
                accel(J[:3], feet1[s][2][:3], 100.*(pt-pos)-20.*(J[:3]@v), 150.)
                accel(J[3:], feet1[s][2][3:], 50.*rotation_error(Rt,R)-14.*(J[3:]@v), 1.)
        if force_share is not None and len(active) == 2:
            for s, share in force_share.items():
                task(Fz[s][None, :], [share*r.weight], force_weight)
        # Regularization chooses finite accelerations and modest motor/contact forces.
        reg = np.r_[np.full(nv, 1e-5), np.full(nu, 1e-3), np.full(n-nv-nu, 1e-5)]
        P = np.diag(reg); linear = np.zeros(n)
        for A, b in zip(tasks, targets):
            P += A.T@A; linear -= A.T@b
        P /= 1000.; linear /= 1000.  # Common cost scaling preserves the optimum.
        A = np.vstack(constraints); l = np.asarray(lower); u = np.asarray(upper)
        key = (tuple((s, tuple(ids)) for s, ids in active.items()), static, n, len(l))
        # Full fixed sparsity allows numeric updates even as zeros become nonzero.
        pr, pc = np.triu_indices(n)
        Pc = sparse.csc_matrix((P[pr, pc], (pr, pc)), shape=(n, n))
        ar = np.tile(np.arange(len(l)), n); ac = np.repeat(np.arange(n), len(l))
        Ac = sparse.csc_matrix((A.T.ravel(), (ar, ac)), shape=A.shape)
        if key not in self.solvers:
            solver = osqp.OSQP()
            solver.setup(P=Pc, q=linear, A=Ac, l=l, u=u, verbose=False,
                         eps_abs=2e-6, eps_rel=2e-6, max_iter=12000,
                         polishing=False, warm_starting=True)
            self.solvers[key] = solver
        else:
            solver = self.solvers[key]
            solver.update(Px=Pc.data, q=linear, Ax=Ac.data, l=l, u=u)
        result = solver.solve()
        self.last_attempt = {'qp_status':result.info.status,'qp_iterations':int(result.info.iter),
            'qp_primal_residual':float(result.info.prim_res),'qp_dual_residual':float(result.info.dual_res),
            'qp_solve_ms':1000*float(result.info.run_time)}
        if result.x is None or result.info.status_val not in (1, 2):
            self.last_attempt['constraint_diagnosis'] = constraint_diagnosis(A,l,u,labels)
            raise RuntimeError('QP '+result.info.status)
        x = result.x
        violation = float(max(np.max(l-A@x), np.max(A@x-u), 0.))
        self.last_attempt['qp_violation']=violation
        if not np.isfinite(x).all() or violation > .02:
            raise RuntimeError(f'QP constraint violation {violation:.4g}')
        return x[nv:nv+nu], {'qp_iterations': int(result.info.iter),
            'qp_violation': violation, 'qp_solve_ms': 1000*result.info.run_time,
            'qp_status': result.info.status, 'qp_primal_residual': float(result.info.prim_res),
            'qp_dual_residual': float(result.info.dual_res),
            **{s+'_predicted_fz_N': float(Fz[s]@x) if s in Fz else 0. for s in SIDES}}


def measurements(r, d, support):
    m = r.m; jc, _, feet = r.kinematics(d)
    row = {}
    unwanted = 0
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        allowed = any(pair == {g, r.floor} for g in r.feet.values())
        if not allowed and c.dist < 0. and c.efc_address >= 0:
            unwanted += 1
    row['unwanted_contacts'] = unwanted
    distances=[(float(mujoco.mj_geomDistance(m,d,a,b,.05,None)),a,b) for a,b in r.collision_pairs]
    distance,ga,gb=min(distances)
    row['minimum_body_clearance_m']=distance
    row['closest_body_pair']=m.geom(ga).name+'|'+m.geom(gb).name
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



def clean_json(x):
    if isinstance(x, dict): return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    return x


def interpolate_q(r, q0, q1, alpha):
    q = q0.copy(); tangent = np.zeros(r.m.nv)
    mujoco.mj_differentiatePos(r.m, tangent, 1., q0, q1)
    mujoco.mj_integratePos(r.m, q, tangent, float(alpha))
    return q


def path_pose(r, knots, fraction):
    x = np.clip(fraction, 0., 1.)*(len(knots)-1)
    i = min(int(x), len(knots)-2)
    return interpolate_q(r, knots[i], knots[i+1], x-i)


def pose_targets(r, q):
    d = r.pose_data(q)
    return d.subtree_com[r.torso].copy(), {
        s: (d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy())
        for s, g in r.feet.items()}


def plan_pose(r, seed, support, pivot, anchor, com_target, clearance=None, max_nfev=350):
    """Local constrained IK approximation, with explicit residual verification."""
    m = r.m; d = mujoco.MjData(m)
    swing = 'right' if support == 'left' else 'left'
    x0 = np.r_[seed[:3], Rotation.from_quat(seed[3:7][[1, 2, 3, 0]]).as_rotvec(), seed[r.qa]]
    lo = np.r_[[-.3, -.3, .19], [-1., -.8, -1.], m.jnt_range[r.jids, 0]+.01]
    hi = np.r_[[.3, .3, .55], [1., .8, 1.], m.jnt_range[r.jids, 1]-.01]
    seed_data = r.pose_data(seed)
    swing_xy = seed_data.geom_xpos[r.feet[swing], :2].copy()
    def fill(x):
        d.qpos[:3] = x[:3]
        d.qpos[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        d.qpos[r.qa] = x[6:]
        mujoco.mj_forward(m, d)
    def distances():
        return [mujoco.mj_geomDistance(m, d, a, b, .05, None) for a, b in r.collision_pairs]
    def fun(x):
        fill(x)
        g = r.feet[support]; pt, Rt = r.anchors[support]
        res = list(1000.*(d.geom_xpos[g]-pt))
        res.extend(200.*rotation_error(Rt, d.geom_xmat[g].reshape(3, 3)))
        if clearance is None:
            res.extend(1000.*(r.corners(d, swing)[pivot]-anchor))
            res.append(1000.*min(0., r.clearance(d, swing)+.0005))
        else:
            res.append(500.*(r.clearance(d, swing)-clearance))
            res.extend(.5*(d.geom_xpos[r.feet[swing], :2]-swing_xy))
        res.extend(500.*(d.subtree_com[r.torso]-com_target))
        res.extend(.02*x[6:])
        res.extend(.02*(x[6:]-x0[6:]))
        res.extend(1000.*min(0., dist-.005) for dist in distances())
        return res
    fit = least_squares(fun, np.clip(x0,lo+1e-8,hi-1e-8), bounds=(lo,hi),
                        max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-8)
    fill(fit.x)
    g = r.feet[support]
    checks = {'com_error_m': float(np.linalg.norm(d.subtree_com[r.torso]-com_target)),
              'stance_position_error_m': float(np.linalg.norm(d.geom_xpos[g]-r.anchors[support][0])),
              'stance_tilt_deg': float(np.rad2deg(np.arccos(np.clip(d.geom_xmat[g].reshape(3,3)[2,2],-1,1)))),
              'pivot_error_m': float(np.linalg.norm(r.corners(d,swing)[pivot]-anchor)),
              'swing_clearance_m': r.clearance(d,swing),
              'collision_distance_m': float(min(distances())),
              'optimizer_success': bool(fit.success),
              'min_joint_margin_deg': float(np.rad2deg(np.min(np.minimum(d.qpos[r.qa]-m.jnt_range[r.jids,0],m.jnt_range[r.jids,1]-d.qpos[r.qa])))),
              'max_joint_step_deg': float(np.rad2deg(np.max(np.abs(d.qpos[r.qa]-seed[r.qa]))))}
    return d.qpos.copy(), checks


def build_plan(r, args, summary):
    support = args.support; swing = 'right' if support == 'left' else 'left'
    sign = 1. if support == 'left' else -1.
    # corners order: (-x,-y), (-x,+y), (+x,-y), (+x,+y).
    pivot = (1 if support == 'left' else 0) + (2 if args.pivot == 'toe' else 0)
    anchor = r.corners(r.d,swing)[pivot].copy()
    anchor[2] = r.ground_z-.0002
    release_com = r.c0.copy(); release_com[0] = r.anchors[support][0][0]
    width = r.m.geom_size[r.feet[support],1]
    release_com[1] = r.anchors[support][0][1]-sign*(width-args.release_inset_mm/1000.)
    release_com[2] = args.com_height
    final_com = release_com.copy(); final_com[1] = r.anchors[support][0][1]
    plan = {'pivot_corner': pivot, 'pivot_anchor_m': anchor.tolist(),
            'release_com_requested_m': release_com.tolist(), 'final_com_requested_m': final_com.tolist(),
            'rock_checks': [], 'lift_checks': []}
    summary['plan'] = plan
    rock = [r.q0.copy()]
    print('Planning crouch and pivot-assisted weight transfer...', flush=True)
    for u in np.linspace(0.,1.,args.knots+1)[1:]:
        com = r.c0+u*(release_com-r.c0)
        q, checks = plan_pose(r,rock[-1],support,pivot,anchor,com)
        checks['fraction'] = float(u); plan['rock_checks'].append(checks); rock.append(q)
    actual_com, _ = pose_targets(r,rock[-1])
    plan['release_com_achieved_m'] = actual_com.tolist()
    lift = [rock[-1].copy()]
    print('Planning contact release and foot lift...', flush=True)
    initial_clearance = r.clearance(r.pose_data(rock[-1]),swing)
    for u in np.linspace(0.,1.,args.knots+1)[1:]:
        q, checks = plan_pose(r,lift[-1],support,pivot,anchor,
                             actual_com+u*(final_com-actual_com),
                             initial_clearance+u*(args.lift_mm/1000.-initial_clearance))
        checks['fraction'] = float(u); plan['lift_checks'].append(checks); lift.append(q)
    # Retain the path so failed local searches are inspectable in JSON.
    plan['rock_qpos'] = [q.tolist() for q in rock]
    plan['lift_qpos'] = [q.tolist() for q in lift]
    all_checks = plan['rock_checks']+plan['lift_checks']
    errors = []
    for stage, cs in [('rock',plan['rock_checks']),('lift',plan['lift_checks'])]:
        for i,c in enumerate(cs):
            if (c['com_error_m'] > .004 or c['stance_position_error_m'] > .001 or
                c['stance_tilt_deg'] > 1. or c['collision_distance_m'] < .002 or
                (stage == 'rock' and (c['pivot_error_m'] > .001 or c['swing_clearance_m'] < -.001)) or
                (stage == 'lift' and c['swing_clearance_m'] < -.001)):
                errors.append({'stage':stage,'knot':i+1,**c})
    plan['rejected_knots'] = errors
    plan['max_joint_step_deg'] = max(c['max_joint_step_deg'] for c in all_checks)
    if errors:
        raise RuntimeError(f'PLAN_SEARCH_FAILED: {len(errors)} knots exceed geometric tolerances; see JSON plan. This is a local-search failure, not proof no transition exists.')
    # Check force/torque feasibility of the final single-support endpoint.
    com, refs = pose_targets(r,lift[-1])
    qp = WholeBodyQP(r,args.mu,args.margin_mm/1000.)
    tau, stats = qp.solve(r.pose_data(lift[-1]),{support:[0,1,2,3]},lift[-1],com,refs,static=True)
    plan['static_final_peak_tau_Nm'] = float(np.max(np.abs(tau)))
    plan['static_qp'] = stats
    print(f'Path found; final static peak torque {np.max(np.abs(tau)):.3f} Nm',flush=True)
    return rock,lift,pivot,anchor


def constraint_diagnosis(A, lower, upper, labels):
    """Read-only LP checks of a failed QP; never used as a torque fallback."""
    n=A.shape[1];k=len(lower)
    finite_u=np.isfinite(upper);finite_l=np.isfinite(lower)
    Au=np.vstack([A[finite_u],-A[finite_l]])
    bu=np.r_[upper[finite_u],-lower[finite_l]]
    feasible=linprog(np.zeros(n),A_ub=Au,b_ub=bu,bounds=[(None,None)]*n,method='highs')
    report={'linear_constraints_feasible':bool(feasible.success),'lp_status':feasible.message}
    if feasible.success or feasible.status!=2:return report
    slack=-np.vstack([np.eye(k)[finite_u],np.eye(k)[finite_l]])
    scales=np.maximum(1.,np.max(np.abs(A),axis=1))
    relaxed=linprog(np.r_[np.zeros(n),1./scales],A_ub=np.c_[Au,slack],b_ub=bu,
                    bounds=[(None,None)]*n+[(0,None)]*k,method='highs')
    if relaxed.success:
        report['elastic_relaxations']=[{'constraint':labels[i],'slack_in_constraint_units':float(v)}
            for i,v in enumerate(relaxed.x[n:]) if v>1e-6]
    report['interpretation']='Row-normalized L1 elastic LP: one possible relaxation, not a unique cause. No relaxed solution is applied.'
    return report


def release_gates(r, args, metrics, support):
    """Actual MuJoCo measurements only: predicted QP forces never open this gate."""
    swing = 'right' if support == 'left' else 'left'
    return {
        'swing_load': abs(metrics[swing+'_fz_N']) < args.release_load_fraction*r.weight,
        'support_load': metrics[support+'_fz_N'] > .85*r.weight,
        'com_margin': metrics['com_support_margin_m'] > .002,
        'com_speed': metrics['com_speed_m_s'] < .03,
        'support_tilt': metrics[support+'_tilt_deg'] < 3.,
        'support_drift': metrics[support+'_drift_m'] < .005,
        'support_speed': metrics[support+'_sole_speed_m_s'] < .01,
        'cop_margin': metrics[support+'_cop_margin_m'] > .002,
        'torque_reserve': metrics['torque_utilization'] < .98,
        'no_body_contact': metrics['unwanted_contacts'] == 0,
    }


def diagnostic_row(r, args, d, qref, com, metrics, support, stats):
    swing = 'right' if support == 'left' else 'left'
    q = d.qpos[r.qa]; ranges = r.m.jnt_range[r.jids]
    margins = np.rad2deg(np.minimum(q-ranges[:, 0], ranges[:, 1]-q))
    target_margins = np.rad2deg(np.minimum(qref[r.qa]-ranges[:, 0], ranges[:, 1]-qref[r.qa]))
    tracking = np.rad2deg(qref[r.qa]-q)
    com_error = com-d.subtree_com[r.torso]
    gate = release_gates(r, args, metrics, support)
    near = [n for n, margin in zip(r.names, margins) if margin < 2.]
    observations = []
    if metrics.get('minimum_body_clearance_m',.05)<.004: observations.append('body_clearance_below_4mm')
    if near: observations.append('joints_within_2deg_of_limit')
    if np.max(np.abs(tracking)) > 5.: observations.append('joint_tracking_error_over_5deg')
    if np.linalg.norm(com_error) > .005: observations.append('com_tracking_error_over_5mm')
    if metrics['torque_utilization'] >= .98: observations.append('torque_limit_reached')
    if metrics[swing+'_fz_N'] >= args.release_load_fraction*r.weight: observations.append('swing_foot_still_loaded')
    if metrics['com_support_margin_m'] <= .002: observations.append('insufficient_com_support_margin')
    return {
        'release_gate_open': int(all(gate.values())),
        'release_blockers': '|'.join(k for k, v in gate.items() if not v),
        'diagnostic_observations': '|'.join(observations),
        'near_limit_joints': '|'.join(near),
        'min_joint_margin_deg': float(np.min(margins)),
        'min_target_joint_margin_deg': float(np.min(target_margins)),
        'max_joint_tracking_error_deg': float(np.max(np.abs(tracking))),
        'worst_tracking_joint': r.names[int(np.argmax(np.abs(tracking)))],
        'com_tracking_error_m': float(np.linalg.norm(com_error)),
        'com_lateral_error_m': float(com_error[1]),
        'swing_force_prediction_error_N': float(stats.get(swing+'_predicted_fz_N', np.nan)-metrics[swing+'_fz_N']),
        **{n+'_limit_margin_deg': float(v) for n, v in zip(r.names, margins)},
        **{'gate_'+k: int(v) for k, v in gate.items()},
    }


class ActiveUnloader:
    """Slow measured-load loop over the 500 Hz inverse-dynamics controller.

    A force PI produces a bounded nonnegative soft force target. A separate
    load-to-COM loop changes a geometric posture reference, never qpos itself.
    Failed IK requests freeze the COM integrator until a feasible update returns.
    All feedback time constants use simulation time, not wall time.
    """
    def __init__(self, r, args, support, pivot, anchor, q):
        self.r, self.args, self.support = r, args, support
        self.swing = 'right' if support == 'left' else 'left'
        self.sign = 1. if support == 'left' else -1.
        self.pivot, self.anchor = pivot, anchor
        self.base_com, _ = pose_targets(r, q)
        self.q_from = q.copy(); self.q_to = q.copy(); self.blend_start = 0.
        self.history = []; self.last_update = -np.inf; self.last_accepted = True
        self.force_integral = self.com_integral = 0.
        self.filtered_N = self.initial_N = self.target_N = self.command_N = 0.
        self.correction = self.error_N = 0.

    def start(self, metrics):
        self.filtered_N = self.initial_N = max(0., metrics[self.swing+'_fz_N'])
        self.target_N = self.command_N = self.initial_N

    def reference(self, elapsed):
        return interpolate_q(self.r, self.q_from, self.q_to,
                             blend((elapsed-self.blend_start)/self.args.posture_update_seconds))

    def update(self, metrics, elapsed):
        r, a = self.r, self.args
        self.filtered_N += (1.-np.exp(-r.dt/.1))*(metrics[self.swing+'_fz_N']-self.filtered_N)
        u = blend(elapsed/a.unload_ramp_seconds)
        self.target_N = (1.-u)*self.initial_N+u*a.target_load_fraction*r.weight
        error = self.target_N-self.filtered_N
        raw = self.target_N+a.force_kp*error+self.force_integral
        # Conditional integration: do not wind up against the nonnegative limit.
        if 0. < raw < r.weight or (raw <= 0. and error > 0.) or (raw >= r.weight and error < 0.):
            self.force_integral = float(np.clip(self.force_integral+a.force_ki*error*r.dt, -r.weight, r.weight))
        self.command_N = float(np.clip(self.target_N+a.force_kp*error+self.force_integral, 0., r.weight))
        self.error_N = self.filtered_N-self.target_N
        cap = a.max_com_correction_mm/1000.
        # Coordinate a small COM advance with the load ramp even when force
        # tracking is good; residual load feedback then adds a bounded correction.
        feedforward = min(cap,a.com_feedforward_mm/1000.)*u
        raw_correction = feedforward+a.com_load_gain_mm_per_N/1000.*self.error_N+self.com_integral
        if self.last_accepted and (0.<raw_correction<cap or (raw_correction<=0. and self.error_N>0.) or (raw_correction>=cap and self.error_N<0.)):
            self.com_integral = float(np.clip(self.com_integral+a.com_load_integral_mm_per_N_s/1000.*self.error_N*r.dt, 0., cap))
        self.correction = float(np.clip(feedforward+a.com_load_gain_mm_per_N/1000.*self.error_N+self.com_integral, 0., cap))
        if elapsed-self.last_update >= a.posture_update_seconds and self.correction > .0001:
            current = self.reference(elapsed)
            # Limit each requested COM movement to 1 mm. Recompute IK rather
            # than applying a direct hip-roll offset that may tilt the sole.
            current_com, _ = pose_targets(r, current)
            desired = self.base_com.copy()
            desired[1] += self.sign*self.correction
            desired[1] = current_com[1]+np.clip(desired[1]-current_com[1], -.001, .001)
            q, checks = plan_pose(r, current, self.support, self.pivot, self.anchor, desired, max_nfev=100)
            achieved_com, _ = pose_targets(r, q)
            requested_step = desired[1]-current_com[1]
            achieved_step = achieved_com[1]-current_com[1]
            checks['achieved_com_m'] = achieved_com.tolist()
            checks['lateral_com_step_m'] = float(achieved_step)
            reasons = []
            for name, bad in {
                'com_residual': checks['com_error_m'] > .004,
                'stance_position': checks['stance_position_error_m'] > .001,
                'stance_tilt': checks['stance_tilt_deg'] > 1.,
                'pivot_position': checks['pivot_error_m'] > .001,
                'floor_penetration': checks['swing_clearance_m'] < -.001,
                'collision_clearance': checks['collision_distance_m'] < .002,
                'joint_margin': checks['min_joint_margin_deg'] < .5,
                'joint_step': checks['max_joint_step_deg'] > 5.,
                # Residual tolerances alone can admit a pose that moves the COM
                # opposite to a small request. Require progress toward the request.
                'no_com_progress': requested_step*achieved_step <= 0.,
                'worse_com_residual': np.linalg.norm(achieved_com-desired) > np.linalg.norm(current_com-desired)+1e-5,
            }.items():
                if bad: reasons.append(name)
            self.last_accepted = not reasons
            self.history.append({'elapsed_unload_s':elapsed,'accepted':self.last_accepted,
                                 'reasons':reasons,'requested_com_m':desired.tolist(),**checks})
            if self.last_accepted:
                self.q_from=current;self.q_to=q;self.blend_start=elapsed
            self.last_update=elapsed
        return self.reference(elapsed)

    def empty_telemetry(self):
        return {'unload_target_fz_N':np.nan,'unload_command_fz_N':np.nan,
                'unload_filtered_fz_N':np.nan,'unload_force_error_N':np.nan,
                'unload_com_correction_m':0.,'unload_force_integral_N':0.,
                'unload_posture_accepted':0,'unload_posture_rejected':0}

    def telemetry(self):
        return {'unload_target_fz_N':float(self.target_N),'unload_command_fz_N':float(self.command_N),
                'unload_filtered_fz_N':float(self.filtered_N),'unload_force_error_N':float(self.error_N),
                'unload_com_correction_m':float(self.correction),'unload_force_integral_N':float(self.force_integral),
                'unload_posture_accepted':sum(x['accepted'] for x in self.history),
                'unload_posture_rejected':sum(not x['accepted'] for x in self.history)}


def replan_lift(r, args, q, support, pivot, anchor, summary):
    start_com, _ = pose_targets(r, q)
    goal_com = start_com.copy();goal_com[1] = r.anchors[support][0][1]
    swing = 'right' if support == 'left' else 'left'
    clearance = r.clearance(r.pose_data(q), swing)
    knots = [q.copy()]; checks = []
    summary['release_lift_replan'] = checks
    for u in np.linspace(0., 1., args.knots+1)[1:]:
        q, c = plan_pose(r, knots[-1], support, pivot, anchor,
                         start_com+u*(goal_com-start_com), clearance+u*(args.lift_mm/1000.-clearance))
        checks.append(c)
        if (c['com_error_m']>.004 or c['stance_position_error_m']>.001 or c['stance_tilt_deg']>1.
            or c['collision_distance_m']<.002 or c['swing_clearance_m']<-.001):
            raise RuntimeError('PLAN_SEARCH_FAILED: updated release pose has no verified local lift path')
        knots.append(q)
    summary['release_lift_qpos'] = [q.tolist() for q in knots]
    return knots


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',default='dodo_floating.xml')
    p.add_argument('--output',default='transition_v2_left')
    p.add_argument('--support',choices=SIDES,default='left')
    p.add_argument('--pivot',choices=('heel','toe'),default='heel')
    p.add_argument('--com-height',type=float,default=.25)
    p.add_argument('--release-inset-mm',type=float,default=5.)
    p.add_argument('--transfer-seconds',type=float,default=8.)
    p.add_argument('--lift-seconds',type=float,default=4.)
    p.add_argument('--lift-mm',type=float,default=25.)
    p.add_argument('--hold-seconds',type=float,default=10.)
    p.add_argument('--seconds',type=float,default=40.)
    p.add_argument('--knots',type=int,default=24)
    p.add_argument('--mu',type=float,default=.8)
    p.add_argument('--margin-mm',type=float,default=2.)
    p.add_argument('--release-load-fraction',type=float,default=.05)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--video',default='')
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--unload-seconds',type=float,default=12.)
    p.add_argument('--unload-ramp-seconds',type=float,default=4.)
    p.add_argument('--target-load-fraction',type=float,default=.01)
    p.add_argument('--force-weight',type=float,default=10.)
    p.add_argument('--contact-soft-weight',type=float,default=0.,
                   help='Unloading sole/pivot acceleration cost; 0 retains v1 hard constraints')
    p.add_argument('--force-kp',type=float,default=.5)
    p.add_argument('--force-ki',type=float,default=.3)
    p.add_argument('--com-load-gain-mm-per-N',type=float,default=1.)
    p.add_argument('--com-feedforward-mm',type=float,default=3.)
    p.add_argument('--com-load-integral-mm-per-N-s',type=float,default=.5)
    p.add_argument('--max-com-correction-mm',type=float,default=15.)
    p.add_argument('--posture-update-seconds',type=float,default=.5)
    p.add_argument('--min-body-clearance-mm',type=float,default=.5,
                   help='Stop before body contact when checked geom clearance drops below this margin')
    p.add_argument('--disable-active-unloading',action='store_true')
    args = p.parse_args()
    numbers=[v for v in vars(args).values() if isinstance(v,(int,float))]
    if not all(np.isfinite(v) for v in numbers): p.error('All numerical arguments must be finite')
    if not 0<=args.target_load_fraction<args.release_load_fraction:
        p.error('Target load must be nonnegative and smaller than the release threshold')
    if not .5<=args.unload_ramp_seconds<args.unload_seconds or not .1<=args.posture_update_seconds<=2.:
        p.error('Use .5 <= unload-ramp-seconds < unload-seconds, and posture-update-seconds .1..2')
    if min(args.force_weight,args.contact_soft_weight,args.force_kp,args.force_ki,args.com_load_gain_mm_per_N,args.com_load_integral_mm_per_N_s)<0:
        p.error('Feedback gains and weights must be nonnegative')
    if not 0<=args.max_com_correction_mm<=25 or args.fps<=0 or args.mu<=0:
        p.error('Use max-com-correction-mm 0..25, positive fps and friction')
    if not 0<=args.com_feedforward_mm<=25:p.error('Use com-feedforward-mm 0..25')
    if not 0<=args.min_body_clearance_mm<=2.:p.error('Use min-body-clearance-mm 0..2')
    if not .21<=args.com_height<=.34 or not 2.<=args.release_inset_mm<=15.:
        p.error('Use com-height .21..34 m, release-inset-mm 2..15')
    if not 8<=args.knots<=100 or min(args.transfer_seconds,args.lift_seconds,args.hold_seconds,args.seconds)<=0:
        p.error('Use knots 8..100 and positive durations')
    if not 10<=args.lift_mm<=60 or not 0<args.margin_mm<10 or not 0<args.release_load_fraction<=.2:
        p.error('Invalid lift, margin or release-load-fraction')
    prefix = Path(args.output); prefix.parent.mkdir(parents=True,exist_ok=True)
    summary = {'run_id':uuid.uuid4().hex,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parameters':vars(args),'result':'ERROR','initialization':'bird_stand keyframe on two feet',
               'model_modified':False,'qpos_resets_after_initialization':0,'support_wrenches_applied':False,
               'versions':{'mujoco':mujoco.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'osqp':osqp.__version__},
               'success_thresholds':{'airborne_clearance_m':.008,'com_speed_m_s_max':.025,
                 'support_sole_speed_m_s_max':.01,'support_tilt_deg_max':3.,'support_drift_m_max':.005,
                 'com_margin_m_min':.002,'cop_margin_m_min':.002,'support_force_fraction_min':.85,
                 'swing_force_fraction_max':.02,'torque_utilization_max':.98,'hold_seconds':args.hold_seconds}}
    summary['stop_thresholds']={'minimum_body_clearance_m':args.min_body_clearance_mm/1000.,
                                'support_drift_m':.01,'support_tilt_deg':5.}
    last_qref=None;last_com=None;last_stats={};last_command={};unloader=None;gate={};diagnostic_timeline=[]
    rows=[]; events=[]; touchdowns=[]; summary['events']=events; summary['touchdowns_after_lift']=touchdowns
    started=time.perf_counter(); writer=renderer=None; code=2
    best=streak=dwell=0.; phase='planning'; valid_initial=False; release_verified=False; air_seen=False; contact_previous=True
    try:
        summary['model_sha256']=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
        r=Robot(args.model); m=r.m; d=r.d
        support=args.support; swing='right' if support=='left' else 'left'
        summary['mass_kg']=r.mass; summary['timestep_s']=r.dt
        if args.mu>min(m.geom_friction[r.floor,0],*(m.geom_friction[g,0] for g in r.feet.values())):
            raise ValueError('QP friction exceeds model friction')
        rock,lift,pivot,anchor=build_plan(r,args,summary)
        if args.plan_only:
            summary['result']='PLAN_FOUND_NOT_SIMULATED'; return 0
        qp=WholeBodyQP(r,args.mu,args.margin_mm/1000.)
        unloader=ActiveUnloader(r,args,support,pivot,anchor,rock[-1])
        summary['active_unloading_enabled']=not args.disable_active_unloading
        summary['posture_updates']=unloader.history
        summary['diagnostic_timeline']=diagnostic_timeline
        if args.video:
            import imageio.v2 as imageio
            renderer=mujoco.Renderer(m,width=960,height=720)
            writer=imageio.get_writer(args.video,fps=args.fps,codec='libx264',quality=7)
        phase='settle'; phase_start=0.; frame=0; log_every=max(1,round(.02/r.dt))
        def transition(new):
            nonlocal phase,phase_start,dwell
            events.append({'time_s':float(d.time),'from':phase,'to':new})
            print(f'{d.time:.3f}s: {phase} -> {new}',flush=True)
            phase=new; phase_start=float(d.time); dwell=0.
        summary['result']='TIMEOUT'
        for step in range(int(np.ceil(args.seconds/r.dt))):
            mujoco.mj_forward(m,d)
            metrics=measurements(r,d,support)
            elapsed=float(d.time)-phase_start
            if phase=='settle':
                good=all(metrics[s+'_fz_N']>.25*r.weight and metrics[s+'_contacts']>=2 for s in SIDES) and metrics['com_speed_m_s']<.02
                dwell=dwell+r.dt if good else 0.
                if elapsed>=1. and dwell>=.3:
                    valid_initial=True; transition('rock_transfer')
            elif phase=='rock_transfer':
                if elapsed>=args.transfer_seconds:
                    unloader.start(metrics);transition('unload')
            elif phase=='unload':
                gate=release_gates(r,args,metrics,support)
                good=all(gate.values())
                dwell=dwell+r.dt if good else 0.
                if dwell>=.25:
                    release_verified=True;summary['release_measurements']=metrics
                    # Replan the lift from the CURRENT accepted unloading reference;
                    # do not jump back to the original pre-feedback release pose.
                    lift=replan_lift(r,args,unloader.reference(elapsed),support,pivot,anchor,summary)
                    transition('lift')
            elif phase=='lift':
                if elapsed>=args.lift_seconds and metrics[swing+'_clearance_m']>.008 and metrics[swing+'_contacts']==0:
                    transition('hold')
            elapsed=float(d.time)-phase_start
            if (phase=='settle' and elapsed>6.) or (phase=='unload' and elapsed>args.unload_seconds) or (phase=='lift' and elapsed>args.lift_seconds+5.):
                summary['result']='PHASE_TIMEOUT_'+phase.upper();break
            if phase=='settle': qref=r.q0;active={s:[0,1,2,3] for s in SIDES};fraction=0.
            elif phase=='rock_transfer':
                fraction=blend(elapsed/args.transfer_seconds);qref=path_pose(r,rock,fraction)
                active={support:[0,1,2,3],swing:[pivot]}
            elif phase=='unload':
                qref=unloader.reference(elapsed) if args.disable_active_unloading else unloader.update(metrics,elapsed)
                active={support:[0,1,2,3],swing:[pivot]};fraction=1.
            else:
                fraction=blend(elapsed/args.lift_seconds) if phase=='lift' else 1.
                qref=path_pose(r,lift,fraction);active={support:[0,1,2,3]}
            com,refs=pose_targets(r,qref)
            refs[support]=(r.anchors[support][0].copy(),r.anchors[support][1].copy())
            if phase=='settle': refs[swing]=(r.anchors[swing][0].copy(),r.anchors[swing][1].copy())
            if phase in ('rock_transfer','unload'):
                g=r.feet[swing];hx,hy,hz=m.geom_size[g]
                local=np.array([(-hx if pivot<2 else hx),(-hy if pivot%2==0 else hy),-hz])
                refs[swing]=(anchor-refs[swing][1]@local,refs[swing][1])
            force_share=None;force_weight=.02
            if phase=='unload' and not args.disable_active_unloading:
                force_share={swing:unloader.command_N/r.weight,support:1.-unloader.command_N/r.weight}
                force_weight=args.force_weight*blend(elapsed/1.)
            last_qref=qref.copy();last_com=com.copy()
            last_command=unloader.telemetry() if phase=='unload' and not args.disable_active_unloading else unloader.empty_telemetry()
            # Save pre-solve state so even QP failures retain a useful diagnosis.
            last_stats={}
            soft_weight=args.contact_soft_weight if phase=='unload' and not args.disable_active_unloading and args.contact_soft_weight>0 else None
            tau,stats=qp.solve(d,active,qref,com,refs,force_share=force_share,force_weight=force_weight,contact_soft_weight=soft_weight)
            last_stats=stats
            d.ctrl[r.aids]=np.clip(tau,r.limits[:,0],r.limits[:,1])
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            metrics=measurements(r,d,support)
            if phase in ('lift','hold') and metrics[swing+'_contacts']==0 and metrics[swing+'_clearance_m']>.008:air_seen=True
            touching=metrics[swing+'_contacts']>0
            if air_seen and touching and not contact_previous:touchdowns.append(float(d.time))
            contact_previous=touching
            good=(phase=='hold' and valid_initial and release_verified and air_seen and not touchdowns and
                  metrics[swing+'_clearance_m']>=.008 and metrics[swing+'_contacts']==0 and abs(metrics[swing+'_fz_N'])<.02*r.weight and
                  metrics[support+'_fz_N']>.85*r.weight and metrics[support+'_contacts']>=2 and
                  metrics[support+'_sole_speed_m_s']<.01 and metrics[support+'_tilt_deg']<3. and metrics[support+'_drift_m']<.005 and
                  metrics[support+'_cop_margin_m']>.002 and metrics['com_support_margin_m']>.002 and
                  metrics['com_speed_m_s']<.025 and metrics['torque_utilization']<.98 and metrics['unwanted_contacts']==0)
            streak=streak+r.dt if good else 0.;best=max(best,streak)
            jc,_,_=r.kinematics(d);velocity=jc@d.qvel;omega=np.sqrt(9.81/max(.05,d.subtree_com[r.torso,2]-r.ground_z))
            capture=d.subtree_com[r.torso,:2]+velocity[:2]/omega
            row={'run_id':summary['run_id'],'time_s':float(d.time),'phase':phase,'path_fraction':float(fraction),'support':support,
                 'success_now':int(good),'success_streak_s':streak,'capture_x_m':float(capture[0]),'capture_y_m':float(capture[1]),
                 'target_com_x_m':float(com[0]),'target_com_y_m':float(com[1]),'target_com_z_m':float(com[2]),
                 **{name+'_q_deg':float(np.rad2deg(d.qpos[qa])) for name,qa in zip(r.names,r.qa)},
                 **{name+'_target_deg':float(np.rad2deg(qref[qa])) for name,qa in zip(r.names,r.qa)},
                 'pivot_clearance_m':float(r.corners(d,swing)[pivot,2]-r.ground_z),
                 'pivot_anchor_error_m':float(np.linalg.norm(r.corners(d,swing)[pivot]-anchor)),
                 'commanded_contact_mode':'flat+point' if len(active)==2 and phase!='settle' else 'double_flat' if phase=='settle' else 'single_flat',
                 'contact_constraints_soft':int(soft_weight is not None),
                 'force_objective_enabled':int(force_share is not None),
                 'force_objective_weight':force_weight if force_share is not None else 0.,
                 **metrics,**stats,**last_command,
                 **diagnostic_row(r,args,d,qref,com,metrics,support,stats)}
            if step%max(1,round(.5/r.dt))==0:
                diagnostic_timeline.append({'time_s':float(d.time),'phase':phase,
                    'swing_fz_N':metrics[swing+'_fz_N'],**last_command,
                    **diagnostic_row(r,args,d,qref,com,metrics,support,stats)})
            if phase=='unload' and step%max(1,round(1./r.dt))==0:
                print(f"{d.time:.2f}s unload: measured={metrics[swing+'_fz_N']:.3f}N "
                      f"target={unloader.target_N:.3f}N command={unloader.command_N:.3f}N "
                      f"COM margin={1000*metrics['com_support_margin_m']:.2f}mm "
                      f"blocked={row['release_blockers']}",flush=True)
            if step%log_every==0:rows.append(row)
            if writer is not None and d.time>=frame/args.fps:
                renderer.update_scene(d);writer.append_data(renderer.render());frame+=1
            if not np.isfinite(d.qpos).all() or metrics['torso_z_m']<.20 or abs(d.qpos[1])>.30:
                summary['result']='FALL_OR_EXCESSIVE_DRIFT';rows.append(row);break
            if phase in ('unload','lift','hold') and (metrics[support+'_drift_m']>.01 or metrics[support+'_tilt_deg']>5.):
                summary['result']='SUPPORT_POSE_LIMIT';rows.append(row);break
            if metrics['unwanted_contacts']:
                summary['result']='UNWANTED_BODY_CONTACT';rows.append(row);break
            if metrics['minimum_body_clearance_m']<args.min_body_clearance_mm/1000.:
                summary['result']='BODY_CLEARANCE_LIMIT';rows.append(row);break
            if streak>=args.hold_seconds:
                summary['result']='SUCCESS_TWO_FEET_TO_ONE';code=0;rows.append(row);break
    except Exception as exc:
        summary['result']='PLAN_SEARCH_FAILED' if str(exc).startswith('PLAN_SEARCH_FAILED') else 'QP_FAILURE' if str(exc).startswith('QP ') else 'ERROR'
        if str(exc).startswith('Joint stopping bounds'): summary['result']='JOINT_STOPPING_INFEASIBLE'
        if 'qp' in locals(): last_stats=getattr(qp,'last_attempt',{})
        summary['error']=str(exc);print('Experiment stopped:',exc,flush=True)
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        summary.update({'wall_time_s':time.perf_counter()-started,'last_phase':phase,'initial_two_foot_support_verified':valid_initial,
                        'unloading_verified':release_verified,'airborne_verified':air_seen,'best_success_streak_s':best,'logged_samples':len(rows),'csv_written':bool(rows)})
        if 'r' in locals():
            summary['simulation_time_s']=float(r.d.time)
            summary['final_measurements']=measurements(r,r.d,args.support)
            final=summary['final_measurements']
            if last_qref is not None:
                summary['diagnosis']=diagnostic_row(r,args,r.d,last_qref,last_com,final,args.support,last_stats)
                summary['diagnosis']['interpretation']='Observed blockers, not proof that the model cannot transition.'
                summary['diagnosis']['last_unload_command']=last_command
                summary['diagnosis']['last_qp']=last_stats
                summary['last_target_qpos']=last_qref.tolist()
                summary['last_target_com_m']=last_com.tolist()
                if unloader is not None:
                    summary['diagnosis']['accepted_posture_updates']=sum(x['accepted'] for x in unloader.history)
                    summary['diagnosis']['rejected_posture_updates']=sum(not x['accepted'] for x in unloader.history)
                if rows:
                    tail=[x for x in rows if x['phase']=='unload' and x['time_s']>=r.d.time-2.]
                    if len(tail)>1:
                        swing='right' if args.support=='left' else 'left'
                        summary['diagnosis']['last_2s_swing_force_change_N']=tail[-1][swing+'_fz_N']-tail[0][swing+'_fz_N']
                        summary['diagnosis']['last_2s_span_s']=tail[-1]['time_s']-tail[0]['time_s']
            summary['final_qpos']=r.d.qpos.tolist()
            summary['final_qvel']=r.d.qvel.tolist()
            summary['final_contact_pairs']=[{'geom1':m.geom(int(c.geom1)).name,'geom2':m.geom(int(c.geom2)).name,'distance_m':float(c.dist)} for c in r.d.contact]
        summary['model_hash_unchanged']=('model_sha256' in summary and hashlib.sha256(Path(args.model).read_bytes()).hexdigest()==summary['model_sha256']) if Path(args.model).is_file() else False
        if rows:
            with prefix.with_suffix('.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            summary['logged_peak_tau_Nm']=max(row['peak_abs_tau_Nm'] for row in rows)
            unload_rows=[row for row in rows if row['phase']=='unload']
            if unload_rows:
                swing='right' if args.support=='left' else 'left'
                summary['unload_progress']={
                    'first_logged_swing_fz_N':unload_rows[0][swing+'_fz_N'],
                    'last_logged_swing_fz_N':unload_rows[-1][swing+'_fz_N'],
                    'minimum_logged_swing_fz_N':min(x[swing+'_fz_N'] for x in unload_rows),
                    'release_threshold_N':args.release_load_fraction*r.weight,
                    'maximum_logged_support_drift_m':max(x[args.support+'_drift_m'] for x in unload_rows),
                    'maximum_logged_support_tilt_deg':max(x[args.support+'_tilt_deg'] for x in unload_rows)}
        prefix.with_suffix('.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
        print(summary['result'],'|',prefix.with_suffix('.json'),flush=True)
        if 'diagnosis' in summary and summary['result']!='SUCCESS_TWO_FEET_TO_ONE':
            diag=summary['diagnosis']
            print('Release blockers:',diag['release_blockers'] or 'none',flush=True)
            print('Observed:',diag['diagnostic_observations'] or 'none',flush=True)
            if diag['near_limit_joints']:print('Near joint limits:',diag['near_limit_joints'],flush=True)
        if rows:print('CSV:',prefix.with_suffix('.csv'),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
