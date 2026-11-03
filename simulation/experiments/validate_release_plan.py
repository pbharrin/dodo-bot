#!/usr/bin/env python3
"""Recheck and export ONLY a two-foot-to-zero-load release path, without lifting.

No forward integration. Recomputes geometry and static LPs at dense samples;
does not trust stored pass flags and does not equate release with airborne support.
"""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from dodo_feasibility_search import Search, digest, json_default
from dodo_transition_v2 import clean_json, interpolate_q


def validate(report,model,output):
    source=json.loads(Path(report).read_text())
    if digest(model)!=source['model_sha256']:raise ValueError('Model hash differs from search report')
    a=SimpleNamespace(**source['parameters']);a.model=str(model);a.refine_lift=getattr(a,'refine_lift',False)
    with tempfile.TemporaryDirectory() as temporary:
        a.output=str(Path(temporary)/'internal.json');s=Search(a);r=s.r;n=a.path_knots
        eligible=[(i,p) for i,p in enumerate(source['paths']) if len(p['qpos'])>=n+1 and np.allclose(p['qpos'][0],r.q0,atol=1e-10,rtol=0)]
        if not eligible:raise ValueError('No complete transfer sequence beginning at bird_stand')
        index,path=eligible[0];qpos=[np.array(q) for q in path['qpos'][:n+1]]
        trial=next(t for t in source['trials'] if t['id']==path['endpoint_trial']);pivot=trial['pivot']
        rows=[];failures=[]
        initial=s.statics(qpos[0],pivot,initial=True)
        if not initial['feasible']:failures.append({'reason':'initial_double_support_statics'})
        for segment,(qa,qb) in enumerate(zip(qpos[:-1],qpos[1:])):
            count=max(4,int(np.ceil(np.max(np.abs(qb[r.qa]-qa[r.qa]))/np.deg2rad(1.))))
            for fraction in np.linspace(0.,1.,count+1):
                q=interpolate_q(r,qa,qb,float(fraction));last=segment==n-1 and fraction==1.
                g=s.geometry(q,pivot,release=last)
                f=s.statics(q,pivot,single=last)
                row={'segment':segment,'fraction':float(fraction),'geometry':g,'statics':f}
                rows.append(row)
                if g['reasons'] or not f['feasible']:failures.append(row)
        endpoint=s.geometry(qpos[-1],pivot,release=True)
        endpoint_static=s.statics(qpos[-1],pivot,single=True)
        if not endpoint_static['feasible']:failures.append({'reason':'release_single_support_statics'})
        result={'status':'VALIDATED_SAMPLED_RELEASE_PATH' if not failures else 'VALIDATION_FAILED',
            'planning_only':True,'forward_integration_steps':0,'airborne_transition_validated':False,
            'includes_lift':False,'timing_assigned':False,'assumed_velocity_and_acceleration':'zero for each static check',
            'model_sha256':digest(model),'source_report_sha256':digest(report),
            'validator_sha256':digest(__file__),'search_helper_sha256':digest(Path(__file__).with_name('dodo_feasibility_search.py')),
            'source_report':str(report),'source_path_index':index,'endpoint_trial_id':trial['id'],
            'support':s.support,'pivot_corner':pivot,'pivot_anchor_m':s.anchor(pivot).tolist(),
            'thresholds':source['thresholds'],'qpos':np.array(qpos).tolist(),
            'pose_count':len(qpos),'interpolated_samples_checked':len(rows),
            'minimum_checked_body_clearance_m':min(x['geometry']['minimum_body_clearance_m'] for x in rows),
            'minimum_joint_margin_deg':min(x['geometry']['min_joint_margin_deg'] for x in rows),
            'maximum_support_position_error_m':max(x['geometry']['support_position_error_m'] for x in rows),
            'maximum_support_orientation_error_deg':max(x['geometry']['support_rotation_error_deg'] for x in rows),
            'maximum_pivot_error_m':max(x['geometry']['pivot_error_m'] for x in rows),
            'peak_feasible_solution_torque_Nm':max((x['statics'].get('peak_torque_Nm',0.) for x in rows)),
            'endpoint_geometry':endpoint,'endpoint_single_support_statics':endpoint_static,'failures':failures,
            'limitations':['Only sampled geometry and ideal contact statics were validated.',
                'The swing foot remains at the ground pivot; zero computed load is not airborne clearance.',
                'No time parameterization, velocity/acceleration bounds, or dynamic tracking test is included.',
                'LP minimizes swing load, not peak torque; reported torques are feasible solutions, not actuator sizing minima.']}
        assert r.d.time==0.
        result['model_hash_unchanged']=digest(model)==source['model_sha256']
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(clean_json(result),indent=2,allow_nan=False,default=json_default)+'\n')
        print(result['status'],'poses',len(qpos),'samples',len(rows),'failures',len(failures))
        return 0 if not failures else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',default='simulation/results/feasibility_search/refined.json')
    p.add_argument('--model',default='simulation/models/dodo_floating.xml')
    p.add_argument('--output',default='simulation/results/feasibility_search/release_path.json')
    a=p.parse_args();return validate(a.report,a.model,a.output)


if __name__=='__main__':raise SystemExit(main())
