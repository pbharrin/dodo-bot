#!/usr/bin/env python3
"""Track the validated outer-toe release path using v2 torque/force control.

Release-only experiment: starts in two-foot bird_stand; never commands a lift.
Run from any directory; defaults locate the model/plan relative to this file.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import uuid

import mujoco
import numpy as np
import dodo_transition_v2 as v2

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [plain(v) for v in value]
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_plan(r, path, model_path):
    plan = json.loads(Path(path).read_text())
    if plan.get('status') != 'VALIDATED_SAMPLED_RELEASE_PATH' or plan.get('includes_lift') is not False:
        raise ValueError('Expected a validated release-only plan')
    if plan.get('model_sha256') != sha(model_path):
        raise ValueError('Plan/model SHA256 mismatch')
    if plan.get('support') != 'left' or plan.get('pivot_corner') != 2:
        raise ValueError('This experiment requires the validated left-support/right-outer-toe plan')
    q = np.asarray(plan['qpos'], dtype=float)
    if q.ndim != 2 or q.shape[1] != r.m.nq or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Invalid plan qpos array')
    if not np.allclose(q[0], r.q0, atol=1e-10, rtol=0):
        raise ValueError('Plan must start at bird_stand, without a pose reset')
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1., atol=1e-8):
        raise ValueError('Plan quaternions must be normalized')
    anchor = np.asarray(plan['pivot_anchor_m'], dtype=float)
    expected = r.corners(r.d, 'right')[2].copy(); expected[2] = r.ground_z-.0002
    if anchor.shape != (3,) or not np.allclose(anchor, expected, atol=1e-8, rtol=0):
        raise ValueError('Plan pivot anchor mismatch')
    return q, anchor


def governor(diag, metrics):
    """Smoothly slow the path before a tracking/geometry pause threshold."""
    limits = {
        'joint_tracking': (diag['max_joint_tracking_error_deg'], 8., 15.),
        'com_tracking': (diag['com_tracking_error_m'], .0025, .005),
        'body_clearance': (-metrics['minimum_body_clearance_m'], -.005, -.004),
        'joint_margin': (-diag['min_joint_margin_deg'], -1.5, -.75),
        'support_drift': (metrics['left_drift_m'], .003, .005),
        'support_tilt': (metrics['left_tilt_deg'], 2., 3.),
    }
    scales = {k: float(np.clip((stop-value)/(stop-slow), 0., 1.))
              for k, (value, slow, stop) in limits.items()}
    return min(scales.values()), '|'.join(k for k, v in scales.items() if v < .999)


def gates(r, args, metrics, diag, complete):
    return {**v2.release_gates(r, args, metrics, 'left'),
            'path_complete': bool(complete),
            'body_clearance': metrics['minimum_body_clearance_m'] >= .005,
            'joint_margin': diag['min_joint_margin_deg'] >= .5,
            'support_contacts': metrics['left_contacts'] >= 2}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=str(ROOT/'models/dodo_floating.xml'))
    p.add_argument('--plan', default=str(ROOT/'results/feasibility_search/release_path.json'))
    p.add_argument('--output', default='release_left')
    p.add_argument('--transfer-seconds', type=float, default=12.)
    p.add_argument('--seconds', type=float, default=50.)
    p.add_argument('--stall-seconds', type=float, default=8.)
    p.add_argument('--unload-seconds', type=float, default=12.)
    p.add_argument('--hold-seconds', type=float, default=2.)
    p.add_argument('--force-weight', type=float, default=10.)
    p.add_argument('--force-kp', type=float, default=.5)
    p.add_argument('--force-ki', type=float, default=.3)
    p.add_argument('--unload-ramp-seconds', type=float, default=4.)
    p.add_argument('--disable-active-unloading', action='store_true')
    p.add_argument('--instant-point-contact', action='store_true',
                   help='Diagnostic comparison: remove non-pivot force variables immediately')
    p.add_argument('--video', default='')
    p.add_argument('--fps', type=int, default=30)
    # Keep v2 force PI; disable its off-path IK/COM correction.
    p.set_defaults(release_load_fraction=.05, target_load_fraction=.01,
                   max_com_correction_mm=0., com_feedforward_mm=0.,
                   com_load_gain_mm_per_N=1., com_load_integral_mm_per_N_s=.5,
                   posture_update_seconds=.5)
    return p


def run(args):
    prefix = Path(args.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = {'run_id': uuid.uuid4().hex, 'result': 'RUNNING', 'parameters': vars(args),
        'script_sha256': sha(__file__), 'controller_sha256': sha(v2.__file__),
        'model_sha256': sha(args.model), 'plan_sha256': sha(args.plan),
        'initialization': 'bird_stand on two feet', 'model_modified': False,
        'qpos_resets_after_initialization': 0, 'support_wrenches_applied': False,
        'includes_lift': False, 'active_unloading_enabled': not args.disable_active_unloading,
        'off_path_posture_correction_enabled': False,
        'contact_force_handoff_enabled': not args.instant_point_contact,
        'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__},
        'success_thresholds': {'hold_seconds': args.hold_seconds, 'swing_force_fraction_max': .05,
            'com_margin_m_min': .002, 'cop_margin_m_min': .002, 'body_clearance_m_min': .005,
            'joint_margin_deg_min': .5, 'support_drift_m_max': .005, 'support_tilt_deg_max': 3.,
            'support_speed_m_s_max': .01, 'com_speed_m_s_max': .03,
            'support_load_fraction_min': .85, 'torque_utilization_max': .98},
        'events': [], 'timeline': []}
    rows = []; renderer = writer = None; phase = 'initializing'
    best = streak = clock = stall = dwell = phase_start = 0.; initial = False
    qref = com = None; stats = {}; lastrow = {}; frame = 0
    try:
        r = v2.Robot(args.model); d = r.d; m = r.m
        knots, anchor = load_plan(r, args.plan, args.model)
        qp = v2.WholeBodyQP(r, .8, .002)
        unload = v2.ActiveUnloader(r, args, 'left', 2, anchor, knots[-1])
        phase = 'settle'; qref = r.q0.copy(); summary['mass_kg'] = float(r.mass)
        if args.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(m, width=960, height=720)
            writer = imageio.get_writer(args.video, fps=args.fps, codec='libx264', quality=7)
        def transition(new):
            nonlocal phase, phase_start, dwell
            summary['events'].append({'time_s': float(d.time), 'from': phase, 'to': new})
            print(f'{d.time:.3f}s: {phase} -> {new}', flush=True)
            phase = new; phase_start = float(d.time); dwell = 0.
        for step in range(int(np.ceil(args.seconds/r.dt))):
            mujoco.mj_forward(m, d)
            metrics = v2.measurements(r, d, 'left')
            com, _ = v2.pose_targets(r, qref)
            diag = v2.diagnostic_row(r, args, d, qref, com, metrics, 'left', stats)
            speed, reasons = governor(diag, metrics)
            elapsed = float(d.time)-phase_start
            if phase == 'settle':
                good = all(metrics[s+'_fz_N'] > .25*r.weight and metrics[s+'_contacts'] >= 2 for s in v2.SIDES) and metrics['com_speed_m_s'] < .02
                dwell = dwell+r.dt if good else 0.
                if elapsed >= 1. and dwell >= .3:
                    initial = True; transition('track_release')
                elif elapsed > 6.:
                    summary['result'] = 'INITIAL_SUPPORT_TIMEOUT'; break
            if phase == 'track_release':
                clock = min(args.transfer_seconds, clock+r.dt*speed)
                stall = stall+r.dt if speed < .01 else 0.
                qref = v2.path_pose(r, knots, v2.blend(clock/args.transfer_seconds))
                if stall >= args.stall_seconds:
                    summary['result'] = 'PATH_TRACKING_STALLED'; break
                if clock >= args.transfer_seconds:
                    unload.start(metrics); transition('unload_hold')
            elapsed = float(d.time)-phase_start
            if phase == 'unload_hold':
                qref = unload.reference(elapsed) if args.disable_active_unloading else unload.update(metrics, elapsed)
                if elapsed > args.unload_seconds:
                    summary['result'] = 'RELEASE_HOLD_TIMEOUT'; break
            fraction = v2.blend(clock/args.transfer_seconds)
            com, refs = v2.pose_targets(r, qref)
            refs['left'] = tuple(x.copy() for x in r.anchors['left'])
            active = {'left': [0,1,2,3], 'right': [0,1,2,3] if phase == 'settle' else [2]}
            if phase == 'settle':
                refs['right'] = tuple(x.copy() for x in r.anchors['right'])
            else:
                hx, hy, hz = m.geom_size[r.feet['right']]
                rotation = refs['right'][1]
                refs['right'] = (anchor-rotation@np.array([hx,-hy,-hz]), rotation)
            force_share = None; weight = .02
            force_enabled = phase == 'unload_hold' and not args.disable_active_unloading
            if force_enabled:
                force_share = {'right': unload.command_N/r.weight, 'left': 1.-unload.command_N/r.weight}
                weight = args.force_weight*v2.blend(elapsed)
            stats = {}
            caps = None
            if phase != 'settle' and not args.instant_point_contact:
                # Retain possible forces until measured sole corners leave the
                # compliant floor. This changes the QP contact estimate, not the
                # MuJoCo model or the validated geometric reference.
                heights = r.corners(d, 'right')[:,2]-r.ground_z
                caps = {'right': 3*r.weight*np.exp(-(np.maximum(heights,0.)/.001)**2)}
                caps['right'][2] = 3*r.weight
            tau, stats = qp.solve(d, active, qref, com, refs, force_share=force_share,
                                 force_weight=weight, force_contact_caps=caps)
            d.ctrl[r.aids] = np.clip(tau, r.limits[:,0], r.limits[:,1])
            mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
            metrics = v2.measurements(r, d, 'left')
            diag = v2.diagnostic_row(r, args, d, qref, com, metrics, 'left', stats)
            gate = gates(r, args, metrics, diag, phase == 'unload_hold')
            good = initial and all(gate.values())
            streak = streak+r.dt if good else 0.; best = max(best, streak)
            lastrow = {'time_s': float(d.time), 'phase': phase, 'path_fraction': fraction,
                'path_clock_s': clock, 'governor_speed': speed, 'governor_reasons': reasons,
                'release_blockers_all': '|'.join(k for k,v in gate.items() if not v),
                'success_streak_s': streak, 'force_objective_enabled': int(force_enabled),
                'force_objective_weight': weight if force_enabled else 0.,
                **{f'right_corner{i}_force_cap_N': float(caps['right'][i]) if caps is not None else (3*r.weight if i in active['right'] else 0.) for i in range(4)},
                'pivot_anchor_error_m': float(np.linalg.norm(r.corners(d,'right')[2]-anchor)),
                **metrics, **diag, **stats,
                **(unload.telemetry() if force_enabled else unload.empty_telemetry())}
            if step % max(1, round(.02/r.dt)) == 0: rows.append(lastrow)
            if step % max(1, round(1./r.dt)) == 0:
                summary['timeline'].append(lastrow.copy())
                summary.update(simulation_time_s=float(d.time), last_phase=phase,
                               path_fraction=fraction, best_success_streak_s=best, current=lastrow)
                save_json(prefix.with_suffix('.json'), summary)
                print(f'{d.time:.2f}s {phase} path={fraction:.3f} swing={metrics["right_fz_N"]:.3f}N COMmargin={1000*metrics["com_support_margin_m"]:.2f}mm slowed={reasons}', flush=True)
            if writer is not None and d.time >= frame/args.fps:
                renderer.update_scene(d); writer.append_data(renderer.render()); frame += 1
            stops = {
                'NONFINITE_STATE': not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all(),
                'FALL_OR_EXCESSIVE_DRIFT': metrics['torso_z_m'] < .2 or abs(d.qpos[1]) > .3,
                'UNWANTED_BODY_CONTACT': metrics['unwanted_contacts'] > 0,
                'BODY_CLEARANCE_LIMIT': metrics['minimum_body_clearance_m'] < .002,
                'JOINT_LIMIT': diag['min_joint_margin_deg'] < -.5,
                'SUPPORT_POSE_LIMIT': metrics['left_drift_m'] > .01 or metrics['left_tilt_deg'] > 5.}
            failures = [k for k,v in stops.items() if v]
            if failures: summary['result'] = failures[0]; break
            if streak >= args.hold_seconds:
                summary['result'] = 'SUCCESS_RELEASE_HOLD'; break
        if summary['result'] == 'RUNNING': summary['result'] = 'TIMEOUT'
    except Exception as exc:
        summary['result'] = 'QP_FAILURE' if str(exc).startswith('QP ') else 'ERROR'
        summary['error'] = str(exc)
        if 'qp' in locals(): summary['failed_qp'] = getattr(qp, 'last_attempt', {})
        print('Stopped:', exc, flush=True)
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        if lastrow and (not rows or rows[-1] is not lastrow): rows.append(lastrow)
        summary.update(last_phase=phase, initial_two_foot_support_verified=initial,
            best_success_streak_s=best, wall_time_s=time.perf_counter()-started,
            path_fraction=v2.blend(clock/args.transfer_seconds), final_diagnostics=lastrow,
            model_hash_unchanged=sha(args.model)==summary['model_sha256'])
        if 'r' in locals():
            summary.update(simulation_time_s=float(r.d.time), final_measurements=v2.measurements(r,r.d,'left'),
                final_qpos=r.d.qpos.tolist(), final_qvel=r.d.qvel.tolist())
            if qref is not None:
                summary['last_target_qpos'] = qref.tolist()
                summary['last_target_com_m'] = v2.pose_targets(r,qref)[0].tolist()
        if rows:
            with prefix.with_suffix('.csv').open('w', newline='') as f:
                writer_csv = csv.DictWriter(f, fieldnames=sorted(set().union(*(x.keys() for x in rows))))
                writer_csv.writeheader(); writer_csv.writerows(rows)
            summary['logged_peak_tau_Nm'] = max(x['peak_abs_tau_Nm'] for x in rows)
            summary['logged_min_body_clearance_m'] = min(x['minimum_body_clearance_m'] for x in rows)
        save_json(prefix.with_suffix('.json'), summary)
        print(summary['result'], '|', prefix.with_suffix('.json'), flush=True)
    return 0 if summary['result'] == 'SUCCESS_RELEASE_HOLD' else 2


def main():
    p = parser(); args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, (float, int)) and not np.isfinite(value): p.error(f'{key} must be finite')
    if min(args.transfer_seconds,args.seconds,args.stall_seconds,args.unload_seconds,args.hold_seconds,args.fps) <= 0:
        p.error('Durations and fps must be positive')
    if min(args.force_weight,args.force_kp,args.force_ki) < 0: p.error('Force gains/weight must be nonnegative')
    if not .5 <= args.unload_ramp_seconds < args.unload_seconds: p.error('Use .5 <= unload-ramp-seconds < unload-seconds')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
