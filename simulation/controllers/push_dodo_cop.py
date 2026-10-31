"""Experimental CoP-aware controller for Dodo flat-foot v4.

Uses the original PD+feedforward law; adds a conservative CoP-edge ankle
torque intervention. This is a diagnostic prototype, not proven balance control.

First-pass controller, not a trained balance policy. Run from the same directory as
build_model.py/dodo_floating.xml. Saves CSV and optionally an offscreen MP4.
"""
import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

JOINT_SUFFIXES = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')


def clip(x, bound):
    return float(np.clip(x, -bound, bound))


def contact_diagnostics(model, data, foot_geom_ids, floor_ids):
    """World-frame wrenches acting ON the feet; CoP projected to z=0.

    MuJoCo contact.frame rows are contact axes. mj_contactForce returns a
    6D wrench on geom2 in that frame (normal, tangents, then moments).
    """
    result = {}
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    per_foot = {}
    for side in ('left', 'right'):
        per_foot[side] = dict(count=0, force=np.zeros(3), normal=0.,
                              tangent=0., max_slip=0., cop_x=np.nan,
                              cop_y=np.nan, moment=np.zeros(3))
    for i in range(data.ncon):
        contact = data.contact[i]
        a, b = int(contact.geom1), int(contact.geom2)
        foot_side = None
        foot_is_geom2 = False
        for side, gid in foot_geom_ids.items():
            if (a == gid and b in floor_ids):
                foot_side = side
                foot_is_geom2 = False
                break
            if (b == gid and a in floor_ids):
                foot_side = side
                foot_is_geom2 = True
                break
        if foot_side is None:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local)
        R = np.asarray(contact.frame).reshape(3, 3)
        sign = 1. if foot_is_geom2 else -1.
        force = sign * (R.T @ local[:3])
        torque = sign * (R.T @ local[3:])
        p = np.asarray(contact.pos)
        moment = np.cross(p, force) + torque
        foot = per_foot[foot_side]
        foot['count'] += 1
        foot['force'] += force
        foot['moment'] += moment
        foot['normal'] += max(0., float(force[2]))
        foot['tangent'] += float(np.linalg.norm(force[:2]))
        bid = int(model.geom_bodyid[foot_geom_ids[foot_side]])
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, p, bid)
        v = jacp @ data.qvel
        foot['max_slip'] = max(foot['max_slip'], float(np.linalg.norm(v[:2])))
        total_force += force
        total_moment += moment
    for side, foot in per_foot.items():
        fz = foot['force'][2]
        if fz > 1e-5:
            foot['cop_x'] = -foot['moment'][1] / fz
            foot['cop_y'] = foot['moment'][0] / fz
    fz = total_force[2]
    result['cop_x_m'] = float(-total_moment[1]/fz) if fz > 1e-5 else np.nan
    result['cop_y_m'] = float(total_moment[0]/fz) if fz > 1e-5 else np.nan
    result['ground_fx_N'] = float(total_force[0])
    result['ground_fy_N'] = float(total_force[1])
    result['ground_fz_N'] = float(fz)
    result['foot_ground_contact_count'] = sum(f['count'] for f in per_foot.values())
    for side, foot in per_foot.items():
        prefix = side + '_'
        result[prefix+'ground_contacts'] = foot['count']
        result[prefix+'fx_N'] = float(foot['force'][0])
        result[prefix+'fy_N'] = float(foot['force'][1])
        result[prefix+'fz_N'] = float(foot['force'][2])
        result[prefix+'normal_N'] = foot['normal']
        result[prefix+'tangent_N'] = foot['tangent']
        result[prefix+'slip_speed_m_s'] = foot['max_slip'] if foot['count'] else np.nan
        result[prefix+'cop_x_m'] = float(foot['cop_x'])
        result[prefix+'cop_y_m'] = float(foot['cop_y'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='dodo_floating.xml')
    p.add_argument('--cop-control', action='store_true', help='Enable CoP-edge ankle-torque limiting; otherwise EXACT PD baseline')
    p.add_argument('--cop-margin', type=float, default=0.020, help='Start protecting within this distance (m) of the toe/heel boundary')
    p.add_argument('--cop-max-reduction', type=float, default=0.40, help='Maximum attenuation of destabilizing ankle torque increment, 0..1')
    p.add_argument('--cop-torque-sign', type=int, choices=(-1,1), default=1, help='Assumed ankle torque sign that moves CoP toward +X; validate experimentally')
    p.add_argument('--cop-filter', type=float, default=0.08, help='CoP low-pass time constant in seconds')
    p.add_argument('--force', type=float, default=0.5, help='Signed horizontal push in newtons (+X forward, -X backward)')
    p.add_argument('--push-start', type=float, default=2.0, help='Start time of disturbance (s)')
    p.add_argument('--push-duration', type=float, default=0.1, help='Duration of disturbance (s)')
    p.add_argument('--axis', choices=['x','y'], default='x', help='World-axis push direction')
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--output', default='standing.csv')
    p.add_argument('--video', default='', help='Optional mp4 path; use MUJOCO_GL=osmesa on SSH machine')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=720)
    args = p.parse_args()
    if not (0 <= args.cop_max_reduction <= 1) or args.cop_margin <= 0 or args.cop_filter <= 0:
        p.error('Require 0<=--cop-max-reduction<=1 and positive CoP margin/filter')
    if args.seconds <= 0 or args.push_start < 0 or args.push_duration <= 0 or args.push_start + args.push_duration > args.seconds:
        p.error('Require positive --seconds and --push-duration, with the push window inside the simulation')

    model = mujoco.MjModel.from_xml_path(str(Path(args.model)))
    if model.nq != 17 or model.nu != 10:
        raise SystemExit(f'Expected floating model nq=17, nu=10; got nq={model.nq}, nu={model.nu}')
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'bird_stand')
    if key < 0:
        raise SystemExit('Missing bird_stand keyframe: regenerate XML using latest build_model.py')
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    joints=[]
    for side in ('left','right'):
        for suffix in JOINT_SUFFIXES:
            name = f'{side}_{suffix}'
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name+'_motor')
            if jid < 0 or aid < 0:
                raise SystemExit('Missing joint or actuator: '+name)
            joints.append((name, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), aid))

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    if torso < 0: raise SystemExit('Missing torso body')
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s+'_foot') for s in ('left','right')]
    foot_geom_ids = {}
    for side in ('left', 'right'):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side+'_foot_geom')
        if gid < 0:
            raise SystemExit('Missing '+side+'_foot_geom')
        foot_geom_ids[side] = gid
    floor_ids = {i for i in range(model.ngeom)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').lower() in ('floor','ground')}
    if not floor_ids:
        raise SystemExit('No floor/ground geometry found')
    foot_half_extents = {}
    for side, gid in foot_geom_ids.items():
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise SystemExit('CoP controller requires v4 box feet: '+side)
        foot_half_extents[side] = model.geom_size[gid].copy()
    first_saturation = None
    cop_x_filt = np.nan
    ankle_neutral = {'left': None, 'right': None}
    ankle_samples = {'left': [], 'right': []}
    latest_cop_diag = contact_diagnostics(model, data, foot_geom_ids, floor_ids)
    kp = {'hip_yaw': 10., 'hip_roll': 18., 'hip_pitch': 22., 'knee_pitch': 24., 'ankle_pitch': 18.}
    kd = {'hip_yaw': 0.65, 'hip_roll': 1.0, 'hip_pitch': 1.1, 'knee_pitch': 1.1, 'ankle_pitch': 0.85}
    rows=[]
    frame_idx=0
    writer=None
    renderer=None
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(model, height=args.height, width=args.width)
        writer=imageio.get_writer(args.video, fps=args.fps, codec='libx264', quality=7, macro_block_size=16)
    axis_index = 0 if args.axis == 'x' else 1
    next_frame=0.0
    total_steps=int(np.ceil(args.seconds/model.opt.timestep))
    try:
        for step in range(total_steps):
            push_active = args.push_start <= data.time < args.push_start + args.push_duration
            data.xfrc_applied[:] = 0.0
            if push_active:
                data.xfrc_applied[torso, axis_index] = args.force
            last_contacts = latest_cop_diag['foot_ground_contact_count']
            measured_cop = latest_cop_diag['cop_x_m']
            dt = float(model.opt.timestep)
            if last_contacts and np.isfinite(measured_cop):
                alpha = dt/(args.cop_filter + dt)
                if not np.isfinite(cop_x_filt): cop_x_filt = measured_cop
                else: cop_x_filt += alpha*(measured_cop-cop_x_filt)
            else:
                cop_x_filt = np.nan
            x_min, x_max = np.inf, -np.inf
            for side, gid in foot_geom_ids.items():
                if latest_cop_diag[side+'_ground_contacts'] <= 0:
                    continue
                R = np.asarray(data.geom_xmat[gid]).reshape(3,3)
                center = np.asarray(data.geom_xpos[gid])
                size = foot_half_extents[side]
                span = sum(abs(R[0,j])*size[j] for j in range(3))
                x_min = min(x_min, float(center[0]-span))
                x_max = max(x_max, float(center[0]+span))
            edge_dist = np.nan
            edge_side = 0
            if np.isfinite(cop_x_filt) and x_max > x_min:
                front = x_max - cop_x_filt
                rear = cop_x_filt - x_min
                edge_dist = min(front,rear)
                edge_side = 1 if front < rear else -1
            limit_strength = 0.0
            if args.cop_control and np.isfinite(edge_dist):
                limit_strength = args.cop_max_reduction * np.clip((args.cop_margin-edge_dist)/args.cop_margin,0,1)
            ankle_interventions = {}
            for name, qadr, vadr, aid in joints:
                suffix = name.split('_',1)[1]
                q=float(data.qpos[qadr]); dq=float(data.qvel[vadr])
                tau=kp[suffix]*(-q)-kd[suffix]*dq + float(data.qfrc_bias[vadr])
                if suffix == 'ankle_pitch':
                    side = name.split('_',1)[0]
                    if 1.0 <= data.time < 1.9:
                        ankle_samples[side].append(tau)
                    if data.time >= 1.9 and ankle_neutral[side] is None:
                        ankle_neutral[side] = float(np.mean(ankle_samples[side])) if ankle_samples[side] else tau
                    neutral = ankle_neutral[side] if ankle_neutral[side] is not None else tau
                    delta = tau-neutral
                    if args.cop_control and edge_side*args.cop_torque_sign*delta > 0:
                        tau -= limit_strength*delta
                        ankle_interventions[side] = float(limit_strength*delta)
                    else: ankle_interventions[side] = 0.0
                lo,hi=model.actuator_ctrlrange[aid]
                data.ctrl[aid]=np.clip(tau, lo, hi)
            mujoco.mj_step(model,data)
            t=float(data.time)
            torso_z=float(data.xpos[torso,2])
            com=np.sum(data.xipos*model.body_mass[:,None],axis=0)/sum(model.body_mass)
            row={'time_s':round(t,5),'torso_z_m':torso_z,
                 'base_x_m':float(data.qpos[0]),'base_y_m':float(data.qpos[1]),
                 'com_x_m':float(com[0]),'com_y_m':float(com[1]),
                 'foot_left_z_m':float(data.xpos[feet[0],2]),
                 'foot_right_z_m':float(data.xpos[feet[1],2]),
                 'contacts':int(data.ncon),
                 'push_N':float(args.force if push_active else 0.0),
                 'push_axis':args.axis,
                 'base_vx_m_s':float(data.qvel[0]),
                 'base_vy_m_s':float(data.qvel[1])}
            latest_cop_diag = contact_diagnostics(model, data, foot_geom_ids, floor_ids)
            row.update(latest_cop_diag)
            row.update({'cop_filtered_x_m':float(cop_x_filt),
                        'cop_support_min_x_m':float(x_min) if np.isfinite(x_min) else np.nan,
                        'cop_support_max_x_m':float(x_max) if np.isfinite(x_max) else np.nan,
                        'cop_edge_margin_m':float(edge_dist),
                        'cop_limit_strength':float(limit_strength),
                        'ankle_left_intervention_Nm':ankle_interventions.get('left',0.),
                        'ankle_right_intervention_Nm':ankle_interventions.get('right',0.),
                        'cop_control_enabled':int(args.cop_control)})
            row['torso_pitch_rad'] = float(np.arctan2(data.xmat[torso].reshape(3,3)[0,2], data.xmat[torso].reshape(3,3)[2,2]))
            for name,qadr,vadr,aid in joints:
                lo, hi = model.actuator_ctrlrange[aid]
                if first_saturation is None and (abs(data.ctrl[aid]-lo)<1e-6 or abs(data.ctrl[aid]-hi)<1e-6):
                    first_saturation = (t, name, float(data.ctrl[aid]))
                row[name+'_q_rad']=float(data.qpos[qadr])
                row[name+'_tau_Nm']=float(data.actuator_force[aid])
                row[name+'_power_W']=float(data.actuator_force[aid]*data.qvel[vadr])
            if step%5==0: rows.append(row)
            if renderer is not None and t+1e-9>=next_frame:
                renderer.update_scene(data)
                writer.append_data(renderer.render())
                frame_idx+=1
                next_frame=frame_idx/args.fps
            if not np.isfinite(data.qpos).all():
                print('Stopped: nonfinite state at',t)
                break
            if torso_z<0.17:
                print(f'FALL DETECTED at t={t:.2f} s (torso origin z={torso_z:.3f} m)')
                break
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    if rows:
        with open(args.output,'w',newline='') as f:
            dw=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            dw.writeheader();dw.writerows(rows)
    if first_saturation is None:
        print('First torque command saturation: none')
    else:
        print('First torque command saturation: t=%.4f s, joint=%s, command=%.3f Nm' % first_saturation)
    print(f'CoP-edge control: {args.cop_control}, torque sign: {args.cop_torque_sign}, margin: {args.cop_margin*1000:.1f} mm')
    print(f'Wrote {args.output} ({len(rows)} rows)')
    if args.video: print(f'Wrote {args.video} ({frame_idx} frames)')
    print(f'Final time={data.time:.2f} s, torso origin z={data.xpos[torso,2]:.3f} m, contact count={data.ncon}')
    print(f'Push: {args.force:+.3f} N along world {args.axis.upper()} at t={args.push_start:.3f}–{args.push_start+args.push_duration:.3f} s (impulse={args.force*args.push_duration:.4f} N s)')
    print('NOTE: CoP-edge attenuation is experimental; actual torque-to-CoP direction must be confirmed.')

if __name__=='__main__':
    main()
