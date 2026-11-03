#!/usr/bin/env python3
"""Planning-only Dodo release/path search. No mj_step, torque execution or XML edits.

Requires dodo_transition_v2.py beside this file. Uses multi-start geometric IK,
independent static force/torque LPs, continuation, and dense interpolation checks.
Results describe sampled quasi-static feasibility, never dynamic success or proof
of impossibility. Atomic JSON checkpoints are written after every candidate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import scipy
from scipy.optimize import least_squares, linprog, minimize
from scipy.spatial.transform import Rotation

from dodo_transition_v2 import Robot, clean_json, interpolate_q, rotation_error


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


class Search:
    def __init__(self, args):
        self.a=args;self.r=Robot(args.model);self.m=self.r.m
        self.support=args.support;self.swing='right' if self.support=='left' else 'left'
        self.sign=1 if self.support=='left' else -1
        self.started=time.monotonic();self.deadline=self.started+args.budget_seconds
        self.clearance=args.clearance_mm/1000.;self.joint_margin=np.deg2rad(args.joint_margin_deg)
        self.contact_tol=.0005;self.com_tol=.002;self.tilt_tol=.25
        self.seeds=self.make_seeds()
        self.report={'status':'RUNNING','planning_only':True,'forward_dynamics_steps':0,
            'started_utc':datetime.now(timezone.utc).isoformat(),'parameters':vars(args),
            'model_sha256':digest(args.model),'script_sha256':digest(__file__),
            'controller_dependency_sha256':digest(Path(__file__).with_name('dodo_transition_v2.py')),
            'seed_qpos':{name:q.tolist() for name,q in self.seeds.items()},
            'versions':{'mujoco':mujoco.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
            'thresholds':{'body_clearance_m':self.clearance,'joint_margin_deg':args.joint_margin_deg,
                'contact_position_tolerance_m':self.contact_tol,'support_tilt_deg':self.tilt_tol,
                'com_target_tolerance_m':self.com_tol,'com_support_margin_m':.002,
                'floor_penetration_tolerance_m':.001,'torque_limit_Nm':args.torque_limit,
                'release_swing_force_max_N':.05*self.r.weight,'friction_pyramid_mu':.8},
            'trials':[],'paths':[],'best_geometry':None,
            'limitations':['Finite local search over fixed foot pivots; no global infeasibility claim.',
                'Ideal rigid-contact statics and sampled paths do not establish dynamic trackability.',
                'Model masses, geometry, joint limits, friction and motor limits are unchanged.']}
        self.save()

    def check_time(self):
        if time.monotonic()>self.deadline:raise TimeoutError('Search wall-time budget reached')

    def save(self):
        self.report['wall_time_s']=time.monotonic()-self.started
        self.report['updated_utc']=datetime.now(timezone.utc).isoformat()
        trials=self.report['trials']
        self.report['counts']={'endpoints_tested':len(trials),
            'geometry_passed':sum(t['geometry_passed'] for t in trials),
            'release_statics_passed':sum(t.get('release_feasible',False) for t in trials),
            'paths_tested':len(self.report['paths']),
            'paths_passed':sum(p.get('passed',False) for p in self.report['paths'])}
        self.report['geometry_rejection_counts']=dict(Counter(k for t in trials for k in t['geometry']['reasons']))
        self.report['model_hash_unchanged']=digest(self.a.model)==self.report['model_sha256']
        path=Path(self.a.output);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(clean_json(self.report),indent=2,allow_nan=False,default=json_default)+'\n')
        temporary.replace(path)

    def make_seeds(self):
        r=self.r;seeds={'neutral':r.q0.copy()}
        root=Path(__file__).resolve().parents[1]/'results'
        for name,path,key in [
            ('previous_release',root/'transition_v2/active_unloading.json',('plan','rock_qpos')),
            ('known_balance',root/'single_support_left/single_support_left.json',('pose_search','single_support_qpos'))]:
            if path.is_file() and self.support=='left':
                j=json.loads(path.read_text())
                if j.get('model_sha256')==digest(self.a.model):
                    q=np.array(j[key[0]][key[1]])
                    seeds[name]=q[-1].copy() if q.ndim==2 else q.copy()
        for side in [-1,1]:
            q=r.q0.copy();q[3:7]=Rotation.from_euler('z',side*.5).as_quat()[[3,0,1,2]]
            q[r.qa[[0,5]]]=-side*.4
            q[r.qa[[2,7]]]=.3;q[r.qa[[3,8]]]=-.6;q[r.qa[[4,9]]]=.3
            seeds['yaw_'+str(side)]=q
        return seeds

    def anchor(self,pivot):
        p=self.r.corners(self.r.d,self.swing)[pivot].copy();p[2]=self.r.ground_z-.0002
        return p

    def geometry(self,q,pivot,target=None,lift=None,release=False):
        r=self.r;m=self.m;d=r.pose_data(q);g=r.feet[self.support]
        pos,R=r.anchors[self.support];actual_R=d.geom_xmat[g].reshape(3,3)
        joint_gap=np.minimum(q[r.qa]-m.jnt_range[r.jids,0],m.jnt_range[r.jids,1]-q[r.qa])
        distances=[(float(mujoco.mj_geomDistance(m,d,a,b,.1,None)),m.geom(a).name,m.geom(b).name) for a,b in r.collision_pairs]
        gap,ga,gb=min(distances)
        center=d.geom_xpos[g];local=actual_R.T@(np.r_[d.subtree_com[r.torso,:2],r.ground_z]-center)
        support_margin=float(min(m.geom_size[g,0]-abs(local[0]),m.geom_size[g,1]-abs(local[1])))
        values={'support_position_error_m':float(np.linalg.norm(center-pos)),
            'support_rotation_error_deg':float(np.rad2deg(np.linalg.norm(rotation_error(R,actual_R)))),
            'pivot_error_m':float(np.linalg.norm(r.corners(d,self.swing)[pivot]-self.anchor(pivot))),
            'swing_clearance_m':r.clearance(d,self.swing),'minimum_body_clearance_m':gap,
            'closest_body_pair':[ga,gb],'min_joint_margin_deg':float(np.rad2deg(min(joint_gap))),
            'closest_joint_limit':r.names[int(np.argmin(joint_gap))],
            'com_m':d.subtree_com[r.torso].tolist(),'com_support_margin_m':support_margin,
            'com_target_error_m':float(np.linalg.norm(d.subtree_com[r.torso]-target)) if target is not None else None}
        reasons=[]
        for name,bad in {
            'support_position':values['support_position_error_m']>self.contact_tol,
            'support_orientation':values['support_rotation_error_deg']>self.tilt_tol,
            'pivot_position':lift is None and values['pivot_error_m']>self.contact_tol,
            'swing_height':lift is not None and abs(values['swing_clearance_m']-lift)>.001,
            'floor_penetration':values['swing_clearance_m']<-.001,
            'body_clearance':gap<self.clearance-1e-6,
            'joint_margin':min(joint_gap)<self.joint_margin-1e-7,
            # Endpoint COM targets guide the optimizer; the physical release
            # requirements are support margin and a permissible height, not an
            # arbitrary exact COM coordinate. Continuation targets remain checked.
            'com_target':not release and target is not None and values['com_target_error_m']>self.com_tol,
            'release_height':release and not .21<=d.subtree_com[r.torso,2]<=.32,
            'release_com_margin':release and support_margin<.002,
        }.items():
            if bad:reasons.append(name)
        values['reasons']=reasons
        return values

    def solve_pose(self,seed,pivot,target,hint=None,lift=None):
        self.check_time();r=self.r;m=self.m;d=mujoco.MjData(m)
        x0=np.r_[seed[:3],Rotation.from_quat(seed[3:7][[1,2,3,0]]).as_rotvec(),seed[r.qa]]
        lo=np.r_[[-.35,-.35,.16],[-1.3,-1.3,-2.],m.jnt_range[r.jids,0]+self.joint_margin]
        hi=np.r_[[.35,.35,.55],[1.3,1.3,2.],m.jnt_range[r.jids,1]-self.joint_margin]
        if hint is None:hint=seed
        hint_R=Rotation.from_quat(hint[3:7][[1,2,3,0]]).as_matrix()
        def fill(x):
            d.qpos[:3]=x[:3];d.qpos[3:7]=Rotation.from_rotvec(x[3:6]).as_quat()[[3,0,1,2]]
            d.qpos[r.qa]=x[6:];mujoco.mj_forward(m,d)
        def fun(x):
            self.check_time();fill(x)
            g=r.feet[self.support];p,R=r.anchors[self.support]
            res=list(2000*(d.geom_xpos[g]-p))
            res.extend(300*rotation_error(R,d.geom_xmat[g].reshape(3,3)))
            if lift is None:res.extend(2000*(r.corners(d,self.swing)[pivot]-self.anchor(pivot)))
            else:res.append(2000*(r.clearance(d,self.swing)-lift))
            res.append(2000*min(0.,r.clearance(d,self.swing)+.0002))
            res.extend(500*(d.subtree_com[r.torso]-target))
            for a,b in r.collision_pairs:
                gap=mujoco.mj_geomDistance(m,d,a,b,.1,None)
                res.append(2000*min(0.,gap-self.clearance-.001))
            res.extend(.05*(x[6:]-hint[r.qa]))
            res.extend(.05*rotation_error(hint_R,d.xmat[r.torso].reshape(3,3)))
            return res
        fit=least_squares(fun,np.clip(x0,lo+1e-8,hi-1e-8),bounds=(lo,hi),
            max_nfev=self.a.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)
        fill(fit.x)
        q=d.qpos.copy();info={'optimizer_success':bool(fit.success),'nfev':int(fit.nfev),'cost':float(fit.cost),'optimizer':'least_squares'}
        if lift is not None and self.a.refine_lift:
            # Explicit physical constraints avoid accepting a least-squares
            # compromise that lifts the foot while losing the COM support margin.
            def objective(x):
                self.check_time();fill(x)
                error=(d.subtree_com[r.torso]-target)/.01
                return float(error[0]**2+error[1]**2+.1*error[2]**2+.01*np.sum((x[6:]-x0[6:])**2))
            def equalities(x):
                fill(x);g=r.feet[self.support];p,R=r.anchors[self.support]
                return np.r_[1000*(d.geom_xpos[g]-p),200*rotation_error(R,d.geom_xmat[g].reshape(3,3)),1000*(r.clearance(d,self.swing)-lift)]
            def inequalities(x):
                fill(x);g=r.feet[self.support];p,R=r.anchors[self.support];com=d.subtree_com[r.torso]
                local=R.T@(np.r_[com[:2],r.ground_z]-p)
                hx,hy=m.geom_size[g,:2]-.0021
                vals=[hx-local[0],hx+local[0],hy-local[1],hy+local[1],com[2]-.21,.32-com[2]]
                vals.extend(mujoco.mj_geomDistance(m,d,a,b,.1,None)-self.clearance for a,b in r.collision_pairs)
                return 1000*np.array(vals)
            constrained=minimize(objective,fit.x,method='SLSQP',bounds=list(zip(lo,hi)),
                constraints=[{'type':'eq','fun':equalities},{'type':'ineq','fun':inequalities}],
                options={'maxiter':160,'ftol':1e-9})
            fill(constrained.x);candidate=d.qpos.copy()
            check=self.geometry(candidate,pivot,target,lift,release=True)
            info['constrained_attempt']={'success':bool(constrained.success),'message':str(constrained.message),
                'nfev':int(constrained.nfev),'eq_residual':float(max(abs(equalities(constrained.x)))),
                'minimum_inequality':float(min(inequalities(constrained.x))),'geometry':check}
            if not check['reasons'] and max(abs(equalities(constrained.x)))<.01 and min(inequalities(constrained.x))>=-1e-4:
                q=candidate;info.update(optimizer='SLSQP',optimizer_success=bool(constrained.success),cost=float(constrained.fun))
        return q,info

    def statics(self,q,pivot,single=False,initial=False):
        """Minimize swing Fz subject to exact rigid-body force balance and bounds."""
        r=self.r;m=self.m;d=r.pose_data(q)
        contacts={self.support:[0,1,2,3]}
        if not single:contacts[self.swing]=[0,1,2,3] if initial else [pivot]
        count=sum(len(v) for v in contacts.values());n=m.nu+3*count
        Aeq=np.zeros((m.nv,n));Aeq[:,:m.nu]=r.S
        rows=[];rhs=[];bounds=[(max(-self.a.torque_limit,l),min(self.a.torque_limit,u)) for l,u in r.limits]
        cost=np.zeros(n);fz={};idx=m.nu
        for side,ids in contacts.items():
            g=r.feet[side];center=d.geom_xpos[g];rotation=d.geom_xmat[g].reshape(3,3)
            fz[side]=np.zeros(n);mx=np.zeros(n);my=np.zeros(n)
            for corner in ids:
                p=r.corners(d,side)[corner];J=np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,J,None,p,int(m.geom_bodyid[g]));Aeq[:,idx:idx+3]=J.T
                bounds.extend([(None,None),(None,None),(0.,2*r.weight)])
                fz[side][idx+2]=1.
                if side==self.swing:cost[idx+2]=1.
                for a,b in itertools.product([-1,1],repeat=2):
                    row=np.zeros(n);row[idx:idx+3]=[a,b,-.8];rows.append(row);rhs.append(0.)
                local=rotation.T@(p-center);mx[idx+2]=local[0];my[idx+2]=local[1]
                idx+=3
            if len(ids)==4:
                for moment,half in [(mx,m.geom_size[g,0]),(my,m.geom_size[g,1])]:
                    for sign in [-1,1]:rows.append(sign*moment-(half-.002)*fz[side]);rhs.append(0.)
        A=np.array(rows);b=np.array(rhs);beq=d.qfrc_bias-d.qfrc_passive
        fit=linprog(cost,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
        out={'feasible':bool(fit.success),'status':fit.message,'single_support_only':single}
        if fit.success:
            out.update({'minimum_swing_force_N':float(fz[self.swing]@fit.x) if self.swing in fz else 0.,
                'support_force_N':float(fz[self.support]@fit.x),'peak_torque_Nm':float(max(abs(fit.x[:m.nu]))),
                'torques_Nm':fit.x[:m.nu].tolist(),'contact_force_variables_N':fit.x[m.nu:].tolist(),
                'equality_residual_max':float(max(abs(Aeq@fit.x-beq))),
                'inequality_violation_max':float(max(0.,max(A@fit.x-b)))})
            out['feasible']=out['equality_residual_max']<1e-6 and out['inequality_violation_max']<1e-6
        return out

    def path(self,trial,direction='forward'):
        r=self.r;pivot=trial['pivot'];end=np.array(trial['qpos']);end_com=np.array(trial['geometry']['com_m'])
        q=end.copy() if direction=='reverse' else r.q0.copy()
        path={'endpoint_trial':trial['id'],'search_direction':direction,'passed':False,'stages':[],'qpos':[q.tolist()]}
        self.report['paths'].append(path);self.save()
        # Check start statics with both complete soles, then constrain the
        # unloading foot to the chosen pivot for the quasi-static continuation.
        start_static=self.statics(r.q0,pivot,initial=True);path['initial_statics']=start_static
        if not start_static['feasible']:path['failure']='initial_statics';return
        for stage in ['transfer','lift']:
            stage_start=q.copy();start_com=r.pose_data(q).subtree_com[r.torso].copy()
            goal=end_com.copy() if stage=='transfer' else start_com.copy()
            if stage=='transfer' and direction=='reverse':goal=r.c0.copy()
            if stage=='lift':goal[1]=r.anchors[self.support][0][1]
            clearance=r.clearance(r.pose_data(q),self.swing)
            for k in range(1,self.a.path_knots+1):
                self.check_time();u=k/self.a.path_knots;target=start_com+u*(goal-start_com)
                lift=None if stage=='transfer' else clearance+u*(.025-clearance)
                hint=interpolate_q(r,r.q0,end,u) if stage=='transfer' else q
                if stage=='transfer' and direction=='reverse':hint=interpolate_q(r,end,r.q0,u)
                candidates=[]
                for seed in [q,hint]:
                    if stage=='transfer' and direction=='reverse' and k==self.a.path_knots:
                        candidate=r.q0.copy();opt={'optimizer_success':True,'nfev':0,'cost':0.,'exact_initial_pose':True}
                    else:candidate,opt=self.solve_pose(seed,pivot,target,hint,lift)
                    geom=self.geometry(candidate,pivot,target,lift,release=stage=='lift' or (direction=='forward' and k==self.a.path_knots))
                    candidates.append((candidate,opt,geom))
                    if not geom['reasons']:break
                candidate,opt,geom=min(candidates,key=lambda x:(len(x[2]['reasons']),x[1]['cost']))
                record={'stage':stage,'knot':k,'geometry':geom,**opt};path['stages'].append(record)
                if geom['reasons']:path['failure']='waypoint_geometry';self.save();return
                # Validate interpolation, not only IK knots, at <=1 degree joint
                # increments and at least four samples per segment.
                count=max(4,int(np.ceil(np.max(np.abs(candidate[r.qa]-q[r.qa]))/np.deg2rad(1.))))
                record['interpolation_samples']=count
                for alpha in np.linspace(0.,1.,count+1):
                    self.check_time();sample=interpolate_q(r,q,candidate,alpha)
                    expected_lift=None
                    if stage=='lift':
                        prior=r.clearance(r.pose_data(q),self.swing);expected_lift=prior+alpha*(lift-prior)
                    check=self.geometry(sample,pivot,lift=expected_lift,release=stage=='lift')
                    forces=self.statics(sample,pivot,single=stage=='lift')
                    if check['reasons'] or not forces['feasible']:
                        path['failure']='interpolated_geometry_or_statics'
                        record['failed_sample']={'fraction':float(alpha),'geometry':check,'statics':forces}
                        self.save();return
                record['statics']=self.statics(candidate,pivot,single=stage=='lift')
                q=candidate;path['qpos'].append(q.tolist());self.save()
            if stage=='transfer' and direction=='reverse':
                path['qpos'].reverse();q=end.copy()
        path['passed']=True;path['failure']=None;path['qualification']='Sampled quasi-static path only; dynamic validation still required.'
        self.save()

    def run(self):
        # Seeded permutation distributes an interrupted/budgeted search across
        # pivots, heights, fore-aft targets and seed families rather than one corner.
        configs=list(itertools.product(range(4),[.23,.26,.29],[-.03,0.,.03],[.055,.065],list(self.seeds)))
        rng=np.random.default_rng(self.a.seed);rng.shuffle(configs)
        try:
            for i,(pivot,height,xoffset,y,name) in enumerate(configs[:self.a.max_candidates]):
                self.check_time();target=np.array([self.r.anchors[self.support][0][0]+xoffset,self.sign*y,height])
                q,opt=self.solve_pose(self.seeds[name],pivot,target)
                geom=self.geometry(q,pivot,target,release=True)
                trial={'id':i,'pivot':pivot,'seed':name,'com_target_m':target.tolist(),
                    'geometry_passed':not geom['reasons'],'geometry':geom,**opt,'qpos':q.tolist()}
                if trial['geometry_passed']:
                    trial['statics']=self.statics(q,pivot)
                    trial['release_feasible']=bool(trial['statics']['feasible'] and trial['statics'].get('minimum_swing_force_N',np.inf)<=.05*self.r.weight)
                self.report['trials'].append(trial)
                best=self.report['best_geometry']
                if best is None or (len(geom['reasons']),opt['cost'])<(len(best['geometry']['reasons']),best['cost']):self.report['best_geometry']=trial
                self.save()
                print(f"{time.monotonic()-self.started:.1f}s trial={i} pivot={pivot} seed={name} geometry={trial['geometry_passed']} release={trial.get('release_feasible',False)} reasons={geom['reasons']}",flush=True)
                if trial.get('release_feasible') and len(self.report['paths'])<self.a.max_paths:
                    for direction in ['forward','reverse']:
                        if len(self.report['paths'])>=self.a.max_paths:break
                        self.path(trial,direction)
                        if self.report['paths'][-1]['passed']:break
                    if self.report['paths'][-1]['passed']:
                        self.report['status']='FEASIBLE_SAMPLED_QUASISTATIC_PATH_FOUND';break
            else:self.report['status']='SEARCH_COMPLETED_NO_FEASIBLE_PATH'
        except TimeoutError:
            self.report['status']='NO_FEASIBLE_PATH_FOUND_WITHIN_TIME_BUDGET'
        except Exception as exc:
            self.report['status']='ERROR';self.report['error']=repr(exc);raise
        finally:
            assert self.r.d.time==0.
            self.save()
        print(self.report['status'],self.report['counts'],flush=True)
        return 0 if self.report['counts']['paths_passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='simulation/models/dodo_floating.xml')
    p.add_argument('--output',default='simulation/results/feasibility_search/current.json')
    p.add_argument('--support',choices=['left','right'],default='left')
    p.add_argument('--budget-seconds',type=float,default=600.)
    p.add_argument('--max-candidates',type=int,default=144)
    p.add_argument('--max-paths',type=int,default=6)
    p.add_argument('--path-knots',type=int,default=16)
    p.add_argument('--max-nfev',type=int,default=160)
    p.add_argument('--clearance-mm',type=float,default=5.)
    p.add_argument('--joint-margin-deg',type=float,default=2.)
    p.add_argument('--torque-limit',type=float,default=3.8)
    p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--refine-lift',action='store_true',help='Use constrained optimization for lift samples; allow COM height to adapt within 210..320 mm')
    a=p.parse_args()
    if not all(np.isfinite(v) for v in vars(a).values() if isinstance(v,(int,float))):p.error('Numerical values must be finite')
    if min(a.budget_seconds,a.max_candidates,a.max_paths,a.path_knots,a.max_nfev)<=0:p.error('Budgets and counts must be positive')
    if not 0<=a.clearance_mm<=20 or not 0<=a.joint_margin_deg<=10 or not 0<a.torque_limit<=4:p.error('Invalid clearance, joint margin, or torque limit')
    return Search(a).run()


if __name__=='__main__':raise SystemExit(main())
