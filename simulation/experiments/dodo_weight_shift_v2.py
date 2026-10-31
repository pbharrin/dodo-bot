"""Conservative Dodo weight-transfer experiment for the v4 flat-foot model."""
import argparse
import csv
from pathlib import Path
import numpy as np
import mujoco

JOINT_TYPES=('hip_yaw','hip_roll','hip_pitch','knee_pitch','ankle_pitch')
KP={'hip_yaw':10.,'hip_roll':18.,'hip_pitch':22.,'knee_pitch':24.,'ankle_pitch':18.}
KD={'hip_yaw':.65,'hip_roll':1.,'hip_pitch':1.1,'knee_pitch':1.1,'ankle_pitch':.85}

def smoothstep(x):
    t=float(np.clip(x,0.,1.))
    return t*t*(3.-2.*t)

def refs_for(model):
    refs={}
    for side in ('left','right'):
        for typ in JOINT_TYPES:
            name=f'{side}_{typ}'
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name+'_motor')
            if jid<0 or aid<0:raise ValueError(f'Missing {name} joint/motor')
            refs[name]=(int(model.jnt_qposadr[jid]),int(model.jnt_dofadr[jid]),int(aid),typ)
    feet={s:int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,s+'_foot_geom')) for s in ('left','right')}
    floor=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor'))
    torso=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso'))
    if min(*feet.values(),floor,torso)<0:raise ValueError('Missing foot, floor, or torso geometry')
    return refs,feet,floor,torso

def reset(model,data):
    key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'bird_stand')
    if key<0:raise ValueError('Missing bird_stand keyframe')
    mujoco.mj_resetDataKeyframe(model,data,key)
    mujoco.mj_forward(model,data)

def ground_loads(model,data,feet,floor):
    loads={'left':0.,'right':0.}
    counts={'left':0,'right':0}
    for i in range(data.ncon):
        ct=data.contact[i]
        a,b=int(ct.geom1),int(ct.geom2)
        side=next((s for s,g in feet.items() if (a==g and b==floor) or (b==g and a==floor)),None)
        if side is None:continue
        wrench=np.zeros(6)
        mujoco.mj_contactForce(model,data,i,wrench)
        force_world=(1. if b==feet[side] else -1.)*(np.asarray(ct.frame).reshape(3,3).T@wrench[:3])
        loads[side]+=max(0.,float(force_world[2]))
        counts[side]+=1
    return loads,counts

def motor_step(model,data,refs,targets):
    for name,(qa,va,aid,typ) in refs.items():
        tau=KP[typ]*(targets.get(name,0.)-float(data.qpos[qa]))-KD[typ]*float(data.qvel[va])+float(data.qfrc_bias[va])
        lo,hi=model.actuator_ctrlrange[aid]
        data.ctrl[aid]=float(np.clip(tau,lo,hi))
    mujoco.mj_step(model,data)

def measure_sign(model,refs,feet,floor,support,probe_deg):
    results=[]
    for sign in (+1.,-1.):
        d=mujoco.MjData(model);reset(model,d)
        shares=[]
        for _ in range(int(1.7/model.opt.timestep)):
            t=float(d.time)
            off=sign*np.deg2rad(probe_deg)*smoothstep((t-1.0)/.5)
            motor_step(model,d,refs,{f'{s}_hip_roll':off for s in ('left','right')})
            if 1.45<d.time<1.7:
                loads,_=ground_loads(model,d,feet,floor)
                tot=sum(loads.values())
                if tot>5.:shares.append(loads[support]/tot)
        results.append((sign,float(np.mean(shares)) if shares else float('nan')))
    print('Sign probe (sign, support-foot load fraction):',results)
    good=[r for r in results if np.isfinite(r[1])]
    if len(good)<2:raise RuntimeError('Unable to calibrate sign: no reliable ground contacts')
    sign,share=max(good,key=lambda x:x[1])
    if share<.505:print('WARNING: probe did not identify a reliable transfer direction')
    return sign

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='dodo_floating.xml')
    p.add_argument('--support',choices=('left','right'),default='left')
    p.add_argument('--target-share',type=float,default=.60)
    p.add_argument('--max-roll-deg',type=float,default=2.)
    p.add_argument('--probe-deg',type=float,default=1.)
    p.add_argument('--ramp-seconds',type=float,default=4.)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--output',default='weight_shift_v2.csv')
    p.add_argument('--video',default='')
    p.add_argument('--fps',type=int,default=30)
    args=p.parse_args()
    if not .5<args.target_share<1.:p.error('--target-share must be between 0.5 and 1')
    if args.max_roll_deg<=0 or args.max_roll_deg>5:p.error('--max-roll-deg must be in (0,5]')
    if args.seconds<args.ramp_seconds+2:p.error('--seconds must be at least ramp-seconds + 2')
    model=mujoco.MjModel.from_xml_path(str(Path(args.model)))
    if model.nu!=10 or model.nq!=17:raise ValueError('Expected 10 motors, free base')
    refs,feet,floor,torso=refs_for(model)
    sign=measure_sign(model,refs,feet,floor,args.support,args.probe_deg)
    print(f'Command sign for {args.support} support = {sign:+.0f}')
    d=mujoco.MjData(model);reset(model,d)
    start_y=float(d.qpos[1]);dt=float(model.opt.timestep)
    mass=float(sum(model.body_mass))
    rows=[];peak_tau=0.;status='Completed'
    renderer=writer=None;frame_n=0
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(model,width=960,height=720)
        writer=imageio.get_writer(args.video,fps=args.fps,codec='libx264',quality=7,macro_block_size=16)
    filtered_share=.5
    command_mag=0.
    last_check=0.
    last_check_cmd=0.
    last_check_share=.5
    hold=False
    abort_reason=''
    try:
        for step in range(int(np.ceil(args.seconds/dt))):
            t=float(d.time)
            loads,counts=ground_loads(model,d,feet,floor)
            total=loads['left']+loads['right']
            measured=loads[args.support]/total if total>5. else float('nan')
            if np.isfinite(measured):
                alpha=dt/(0.12+dt)
                filtered_share+=(measured-filtered_share)*alpha
            desired=.5+(args.target_share-.5)*smoothstep((t-1.)/args.ramp_seconds)
            if t>=1. and not hold:
                if filtered_share<desired-.005:
                    command_mag=min(np.deg2rad(args.max_roll_deg),command_mag+np.deg2rad(args.max_roll_deg)*dt/args.ramp_seconds)
                else:
                    command_mag=max(0.,command_mag-np.deg2rad(args.max_roll_deg)*dt/(2*args.ramp_seconds))
            if t>=1.5 and t-last_check>=.25:
                if (command_mag>last_check_cmd+np.deg2rad(.08) and
                    filtered_share<last_check_share-.015):
                    hold=True
                    abort_reason='load share fell despite increasing hip-roll command'
                    print(f'SAFEGUARD at {t:.3f}s: {abort_reason}; returning roll command toward zero')
                last_check=t;last_check_cmd=command_mag;last_check_share=filtered_share
            if hold:
                command_mag=max(0.,command_mag-np.deg2rad(args.max_roll_deg)*dt/1.5)
            roll=float(sign*command_mag)
            targets={f'{s}_hip_roll':roll for s in ('left','right')}
            motor_step(model,d,refs,targets)
            loads,counts=ground_loads(model,d,feet,floor)
            total=loads['left']+loads['right']
            new_share=loads[args.support]/total if total>5. else float('nan')
            com=(model.body_mass[:,None]*d.xipos).sum(axis=0)/mass
            row={'time_s':float(d.time),'phase':'safeguard' if hold else ('transfer' if t>=1 else 'settle'),
                 'support':args.support,'target_share':desired,'measured_load_share':new_share,
                 'filtered_share':filtered_share,'hip_roll_command_deg':np.rad2deg(roll),
                 'left_fz_N':loads['left'],'right_fz_N':loads['right'],
                 'left_contacts':counts['left'],'right_contacts':counts['right'],
                 'com_x_m':float(com[0]),'com_y_m':float(com[1]),'com_z_m':float(com[2]),
                 'base_y_m':float(d.qpos[1]),'base_vy_m_s':float(d.qvel[1]),
                 'torso_z_m':float(d.xpos[torso,2]),'safeguard_active':int(hold)}
            for name,(qa,va,aid,typ) in refs.items():
                row[name+'_tau_Nm']=float(d.actuator_force[aid]);peak_tau=max(peak_tau,abs(row[name+'_tau_Nm']))
            if step%5==0:rows.append(row)
            if renderer is not None and d.time>=frame_n/args.fps-1e-9:
                renderer.update_scene(d);writer.append_data(renderer.render());frame_n+=1
            if (d.xpos[torso,2]<.20 or abs(d.qpos[1]-start_y)>.10 or not np.isfinite(d.qpos).all()):
                status='Stopped: fall / >100 mm lateral drift';break
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
    if rows:
        with open(args.output,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f'{status}; time={d.time:.3f}s; peak joint torque={peak_tau:.3f} Nm')
    print(f'Final filtered support share={filtered_share:.3f}; safeguard={hold}')
    if abort_reason:print(abort_reason)
    print('CSV:',args.output)
    if args.video:print('MP4:',args.video)
    print('NOTE: This is a cautious diagnostic, not a validated single-leg balance controller.')

if __name__=='__main__':main()
