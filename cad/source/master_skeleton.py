#!/usr/bin/env python3
"""Dodo v6 reference CAD, in mm. Not manufacturing geometry.

Read the simulation generator rather than duplicating its joint geometry.
Emit named STEP assemblies, a datum/mass manifest, clearance study and preview.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cadquery as cq
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'simulation/models/build_model_v6.py'
CONFIG = Path(__file__).with_name('envelopes_v6.json')
COLORS = {'module': '#384a63', 'drum': '#e7a33b', 'input': '#d86d36',
          'link': '#91a8b9', 'foot': '#596977', 'axis': '#13a694', 'torso': '#b2cbdc'}


def load_model():
    spec = importlib.util.spec_from_file_location('dodo_v6_cad_source', MODEL)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def vec(text, default=(0, 0, 0)):
    return np.array(default if text is None else [float(v) for v in text.split()], float)


def rotation(axis, degrees):
    axis = np.array(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = math.radians(degrees)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def quat_rotation(text):
    q = vec(text, (1, 0, 0, 0))
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def xml_for_config(model, cfg):
    return model.make_model_with(
        HIP_ROLL_OFFSET=tuple(np.array(cfg['hip_roll_offset_mm']) / 1000),
        HIP_PITCH_OFFSET=tuple(np.array(cfg['hip_pitch_offset_mm']) / 1000))


def kinematics(xml, q_degrees=None):
    """Forward kinematics for this generator's zero-origin hinge bodies.

    q is relative to neutral, NOT absolute anatomical pitch. All outputs mm.
    Reject unsupported joint offsets instead of silently misplacing them.
    """
    q_degrees = q_degrees or {}
    bodies, joints, geoms, sites = {}, {}, [], {}
    def walk(body, parent_R, parent_p):
        p = parent_p + parent_R @ (1000 * vec(body.get('pos')))
        R = parent_R @ quat_rotation(body.get('quat'))
        name = body.get('name')
        for joint in body.findall('joint'):
            if np.linalg.norm(vec(joint.get('pos'))) > 1e-12:
                raise ValueError('Only zero-origin hinges supported')
            jname = joint.get('name')
            axis = vec(joint.get('axis'))
            joints[jname] = {'origin_mm': p.copy(), 'axis': R @ axis,
                            'range_deg': vec(joint.get('range')), 'body': name}
            R = R @ rotation(axis, q_degrees.get(jname, 0))
        bodies[name] = (R.copy(), p.copy())
        for geom in body.findall('geom'):
            g = dict(geom.attrib)
            g['body'] = name
            g['R'], g['p'] = R.copy(), p.copy()
            g['center'] = p + R @ (1000 * vec(g.get('pos')))
            if 'fromto' in g:
                ft = 1000 * vec(g['fromto']).reshape(2, 3)
                g['ends'] = [p + R @ v for v in ft]
                g['center'] = np.mean(g['ends'], axis=0)
            geoms.append(g)
        for site in body.findall('site'):
            sites[site.get('name')] = p + R @ (1000 * vec(site.get('pos')))
        for child in body.findall('body'):
            walk(child, R, p)
    for b in ET.fromstring(xml).find('worldbody').findall('body'):
        walk(b, np.eye(3), np.zeros(3))
    unknown = set(q_degrees) - set(joints)
    if unknown:
        raise ValueError(f'Unknown joints: {unknown}')
    return {'bodies': bodies, 'joints': joints, 'geoms': geoms, 'sites': sites}


def cylinder(center, axis, radius, length):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return cq.Solid.makeCylinder(float(radius), float(length),
        cq.Vector(*(np.asarray(center) - axis * length / 2)), cq.Vector(*axis))


def capsule(a, b, radius):
    a, b = np.asarray(a), np.asarray(b)
    shaft = cylinder((a+b)/2, b-a, radius, np.linalg.norm(b-a))
    return shaft.fuse(cq.Solid.makeSphere(radius, cq.Vector(*a), angleDegrees1=-90),
                      cq.Solid.makeSphere(radius, cq.Vector(*b), angleDegrees1=-90))


def box(center, R, size):
    plane = cq.Plane(origin=cq.Vector(*center), xDir=cq.Vector(*R[:, 0]), normal=cq.Vector(*R[:, 2]))
    return cq.Workplane(plane).box(*size).val()


def parts_for(state, model, cfg, include_datums=True):
    parts = []
    def add(name, shape, kind, body, note=''):
        parts.append({'name': name, 'shape': shape, 'kind': kind, 'body': body, 'note': note})
    diameter = cfg['module_envelope']['diameter_mm']
    length = cfg['module_envelope']['axial_length_mm']
    for g in state['geoms']:
        name, body = g['name'], g['body']
        if name.endswith('_module'):
            axis = g['R'] @ vec(g.get('zaxis'), (0, 0, 1))
            add(name, cylinder(g['center'], axis, diameter/2, length), 'module', body,
                'Aggregate simulation envelope; purchased dimensions unverified')
        elif 'ends' in g:
            add(name, capsule(*g['ends'], float(g['size']) * 1000), 'link', body,
                'Simulation link envelope; not a printable link')
        elif name in ('left_foot', 'right_foot'):
            add(name, box(g['center'], g['R'], 2000 * vec(g['size'])), 'foot', body,
                'Plate plus sole envelope; no attachment details')
    # Distal drum centers sit beside the motor reservation, not inside it.
    drum_width = cfg['distal_drums']['width_mm']
    axial = length/2 + cfg['distal_drums']['axial_gap_mm'] + drum_width/2
    for side, sign in [('left', 1), ('right', -1)]:
        for joint, parent in [('knee_pitch', 'thigh'), ('ankle_pitch', 'shank')]:
            name = f'{side}_{joint}'
            origin = state['joints'][name]['origin_mm']
            R, _ = state['bodies'][f'{side}_{parent}']
            outward = sign * R[:, 1]
            motor = origin + R @ np.array([0, 0, model.CAPSTAN_CENTER_DISTANCE*1000])
            add(f'{name}_output_drum', cylinder(origin + outward*axial, outward,
                model.OUTPUT_DRUM_RADIUS*1000, drum_width), 'drum', state['joints'][name]['body'],
                'Study envelope; width/axial plane unverified')
            add(f'{name}_input_drum', cylinder(motor + outward*axial, outward,
                model.INPUT_DRUM_RADIUS*1000, drum_width), 'input', f'{side}_{parent}',
                'Study envelope; no shaft or rope routing')
    if include_datums:
        for name, j in state['joints'].items():
            add(name+'_axis', cylinder(j['origin_mm'], j['axis'], 1.5, 65), 'axis', j['body'])
        R, p = state['bodies']['torso']
        # Edges as thin cylinders make a visible, non-solid torso reservation cage.
        half = np.array(model.TORSO_HALF)*1000
        for i in range(3):
            others = [k for k in range(3) if k != i]
            for a in [-1, 1]:
                for b in [-1, 1]:
                    v = np.zeros(3); v[others] = half[others] * [a, b]
                    add(f'torso_cage_{i}_{a}_{b}', cylinder(p+R@v, R[:, i], 0.7, 2*half[i]),
                        'torso', 'torso', 'Torso bounding box, not shell surface')
    return parts


def mass_budget(state):
    rows = []
    for g in state['geoms']:
        mass = float(g.get('mass', 0))
        if mass:
            rows.append({'name': g['name'], 'mass_kg': mass, 'center_mm': g['center'].tolist(),
                         'status': 'simulation_estimate'})
    total = sum(r['mass_kg'] for r in rows)
    com = sum(r['mass_kg'] * np.array(r['center_mm']) for r in rows)/total
    return {'total_kg': total, 'center_of_mass_mm': com.tolist(), 'items': rows,
            'note': 'Drums, bearings and driver are already inside module allowance. CAD display volumes are not weighed again. Hock envelope has zero mass in v6.'}


def collision_pairs(parts):
    """Only diagnostic solid checks: module-module, module-nonowner link/foot.

    Owner-link intersections are intentional attachment regions, omitted here.
    Datum cage, axes, and assumed drum interfaces excluded from this pass.
    """
    modules = [p for p in parts if p['kind'] == 'module']
    links = [p for p in parts if p['kind'] in ('link', 'foot')]
    pairs = [(a,b) for i,a in enumerate(modules) for b in modules[i+1:]]
    pairs += [(a,b) for a in modules for b in links if a['body'] != b['body']]
    pairs += [(a,b) for i,a in enumerate(links) for b in links[i+1:]
              if a['name'].startswith('left_') and b['name'].startswith('right_')]
    return pairs


def overlaps(parts):
    hits = []
    for a, b in collision_pairs(parts):
        ba, bb = a['shape'].BoundingBox(), b['shape'].BoundingBox()
        if any(getattr(ba, k+'max') <= getattr(bb, k+'min') or
               getattr(bb, k+'max') <= getattr(ba, k+'min') for k in 'xyz'):
            continue
        volume = a['shape'].intersect(b['shape']).Volume()
        if volume > 0.1:
            hits.append({'a': a['name'], 'b': b['name'], 'intersection_mm3': round(volume, 3)})
    return hits


def candidate_step(model, samples=21):
    """Left swing: +/−20 mm fore/aft and 40 mm lift. Fixed torso/right foot.

    Geometric test trajectory only: no COM transfer, balance or dynamics.
    """
    L1, L2 = model.THIGH_LENGTH*1000, model.SHANK_LENGTH*1000
    h0, k0 = math.radians(model.HIP_PITCH_DEG), math.radians(model.KNEE_PITCH_DEG)
    x0 = -L1*math.sin(h0)-L2*math.sin(h0+k0)
    d0 = L1*math.cos(h0)+L2*math.cos(h0+k0)
    frames = []
    for t in np.linspace(0, 1, samples):
        # Closed swing loop: backward below neutral, then forward, then home.
        x = x0 + 20 * math.sin(2*math.pi*t)
        drop = d0 - 40 * math.sin(math.pi*t)**2
        c = (x*x+drop*drop-L1*L1-L2*L2)/(2*L1*L2)
        if abs(c) > 1:
            raise ValueError('Candidate step is outside leg reach')
        knee = -math.acos(c)
        hip = math.atan2(-x, drop)-math.atan2(L2*math.sin(knee), L1+L2*math.cos(knee))
        ankle = -hip-knee
        q = dict(zip(['left_hip_pitch','left_knee_pitch','left_ankle_pitch'],
                     [math.degrees(hip)-model.HIP_PITCH_DEG,
                      math.degrees(knee)-model.KNEE_PITCH_DEG,
                      math.degrees(ankle)-model.ANKLE_PITCH_DEG]))
        frames.append({'phase': float(t), 'q_degrees': q})
    return frames


def distal_sweeps(xml, model, cfg):
    """5-degree samples over each knee/ankle full window. Not continuous proof."""
    results = []
    for side in ['left', 'right']:
        for joint, child, motor in [('knee_pitch','shank_geom','knee_module'),
                                    ('ankle_pitch','hock_geom','ankle_module')]:
            name = f'{side}_{joint}'
            hits = []
            for q in np.linspace(*model.JOINT_RANGE_DEG[joint], 25):
                st = kinematics(xml, {name: float(q)})
                # Construct only relevant bodies to keep this check inexpensive.
                motor_g = next(g for g in st['geoms'] if g['name'] == f'{side}_{motor}')
                child_g = next(g for g in st['geoms'] if g['name'] == f'{side}_{child}')
                mshape = cylinder(motor_g['center'], motor_g['R'] @ vec(motor_g['zaxis']),
                    cfg['module_envelope']['diameter_mm']/2, cfg['module_envelope']['axial_length_mm'])
                cshape = capsule(*child_g['ends'], 1000*float(child_g['size']))
                v = mshape.intersect(cshape).Volume()
                if v > 0.1:
                    hits.append(float(q))
            results.append({'joint': name, 'sample_step_deg': 5, 'overlap_q_deg': hits,
                            'tested': f'{side}_{motor} vs {side}_{child}'})
    return results


def dump_json(path, data):
    def default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(data, indent=2, default=default)+'\n')


def save_assembly(path, parts):
    assembly = cq.Assembly(name=path.stem)
    for p in parts:
        h = COLORS[p['kind']].lstrip('#')
        assembly.add(p['shape'], name=p['name'], color=cq.Color(*[int(h[i:i+2],16)/255 for i in (0,2,4)]))
    assembly.export(str(path), exportType='STEP', mode='default')
    # Normalize exporter whitespace for reviewable Git diffs.
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def preview(path, parts, state, cfg, mass):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    from matplotlib.collections import PolyCollection
    fig = plt.figure(figsize=(16,10), facecolor='#f4f7fa')
    fig.suptitle('DODO / v6 master skeleton', x=.05, ha='left', fontsize=24, weight='bold', color='#172a40')
    fig.text(.05,.925,'Reference geometry + estimated hardware envelopes  •  millimeters  •  not printable parts',fontsize=12,color='#4d6075')
    ax3 = fig.add_axes([.02,.14,.47,.74], projection='3d')
    side = fig.add_axes([.53,.16,.21,.68]); front = fig.add_axes([.77,.16,.21,.68])
    meshes = []
    for p in parts:
        vertices, triangles = p['shape'].tessellate(1.3, .25)
        pts = np.array([v.toTuple() for v in vertices])
        tris = pts[np.array(triangles)]
        meshes.append((p,tris))
        alpha = .16 if p['kind']=='torso' else (.65 if p['kind']=='link' else .9)
        ax3.add_collection3d(Poly3DCollection(tris, facecolor=COLORS[p['kind']], edgecolor='none', alpha=alpha))
    for ax, horizontal, title in [(side,0,'SIDE / +X forward'),(front,1,'FRONT / +Y left')]:
        for p,tris in sorted(meshes, key=lambda it: 0 if it[0]['kind']=='torso' else 1):
            if horizontal==0 and p['name'].startswith('right_'): continue
            poly = PolyCollection(tris[:,:,[horizontal,2]], facecolor=COLORS[p['kind']], edgecolor='none', alpha=.25 if p['kind'] in ('module','link') else .55)
            ax.add_collection(poly)
        for name,j in state['joints'].items():
            if name.startswith('left_'):
                p=j['origin_mm']; ax.scatter(p[horizontal],p[2],s=15,color='#0b806d',zorder=10)
        ax.axhline(0, color='#738296', linewidth=1)
        ax.set(xlim=(-180,180),ylim=(-15,530),title=title,xlabel=('X' if horizontal==0 else 'Y')+' (mm)')
        ax.set_aspect('equal'); ax.grid(alpha=.16); ax.set_facecolor('#f4f7fa')
        ax.spines[['top','right']].set_visible(False)
    side.set_ylabel('Z (mm)')
    for z,label in [(500,'torso top 500'),(330,'hip pitch 330')]:
        side.axhline(z, color='#71839a', linestyle=':', linewidth=.7)
        side.text(-174,z+6,label,fontsize=8,color='#465c75')
    ax3.set(xlim=(-170,170),ylim=(-180,180),zlim=(0,520),xlabel='X forward',ylabel='Y left',zlabel='Z up')
    ax3.set_box_aspect((340,360,520)); ax3.view_init(elev=17,azim=-55); ax3.set_facecolor('#f4f7fa')
    fig.legend(handles=[Patch(color=COLORS[k],label=t) for k,t in [('module','Module allowance'),('drum','Output drum'),('input','Input drum'),('axis','Joint axis'),('link','Link envelope')]],loc='lower center',bbox_to_anchor=(.5,.065),ncol=5,frameon=False)
    fig.text(.05,.035,f"10 independent joints  |  {mass['total_kg']:.2f} kg simulation budget  |  hip offsets: {cfg['hip_roll_offset_mm']} / {cfg['hip_pitch_offset_mm']} mm",fontsize=10,color='#465c75')
    fig.savefig(path,dpi=150,facecolor=fig.get_facecolor()); plt.close(fig)


def build(config_path=CONFIG, output_dir=ROOT/'cad', checks=True, images=True):
    cfg = json.loads(Path(config_path).read_text()); model = load_model()
    for field in ('diameter_mm','axial_length_mm'):
        if cfg['module_envelope'][field] <= 0: raise ValueError(field+' must be positive')
    if cfg['distal_drums']['width_mm'] <= 0 or cfg['distal_drums']['axial_gap_mm'] < 0:
        raise ValueError('Invalid drum width/gap')
    xml = xml_for_config(model,cfg); state = kinematics(xml)
    parts = parts_for(state,model,cfg); mass = mass_budget(state)
    output_dir = Path(output_dir); assemblies=output_dir/'assemblies'; reports=output_dir/'reports'
    assemblies.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    for p in parts:
        if not p['shape'].isValid() or p['shape'].Volume() <= 0:
            raise ValueError('Invalid solid '+p['name'])
    save_assembly(assemblies/'dodo_v6_master.step',parts)
    save_assembly(assemblies/'dodo_v6_skeleton.step',[p for p in parts if p['kind'] not in ('module','drum','input')])
    save_assembly(assemblies/'dodo_v6_hardware_envelopes.step',[p for p in parts if p['kind'] in ('module','drum','input')])
    modules = [{ 'name':g['name'], 'body':g['body'], 'center_mm':g['center'],
                'axis':g['R']@vec(g['zaxis']), 'diameter_mm':cfg['module_envelope']['diameter_mm'],
                'length_mm':cfg['module_envelope']['axial_length_mm'], 'status':'simulation_estimate'}
               for g in state['geoms'] if g['name'].endswith('_module')]
    manifest = {'schema_version':1,'units':'mm','configuration':cfg,'source_model':str(MODEL.relative_to(ROOT)),
       'source_model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
       'coordinate_frame':'X forward, Y left, Z up; right pitch/yaw signs are NOT negated',
       'joint_zero':'q=0 is neutral; body-frame pitch offsets +25/-40/+15 deg',
       'joints':state['joints'],'sites':state['sites'],'actuator_envelopes':modules,'mass_budget':mass,
       'parts':[dict(name=p['name'],kind=p['kind'],body=p['body'],note=p['note']) for p in parts]}
    dump_json(assemblies/'dodo_v6_manifest.json',manifest)
    report={'status':'not_manufacturing_ready','nominal_overlaps':overlaps(parts) if checks else [],
            'scope':'Module-module, module-nonowner link/foot, opposite-leg link/foot only. No torso-shell, wiring, bearings, shafts, fasteners or detailed drum interface clearance.',
            'distal_travel_sweeps':distal_sweeps(xml,model,cfg) if checks else [],'candidate_step':[]}
    for f in candidate_step(model):
        st=kinematics(xml,f['q_degrees'])
        limits_ok=all(st['joints'][j]['range_deg'][0] <= q <= st['joints'][j]['range_deg'][1] for j,q in f['q_degrees'].items())
        report['candidate_step'].append({**f,'left_sole_mm':st['sites']['left_foot'], 'within_joint_limits':limits_ok,
             'overlaps':overlaps(parts_for(st,model,cfg,False)) if checks else []})
    report['checks_run']=checks
    report['step_note']='21 samples, fixed torso/right foot, 40 mm swing lift, +/-20 mm fore/aft. Geometric diagnostic only; not a balanced or dynamically validated gait. Sampled checks do not prove continuous clearance.'
    dump_json(reports/'dodo_v6_clearances.json',report)
    write_report(reports/'master_skeleton.md',state,model,cfg,mass,parts,report)
    if images: preview(reports/'dodo_v6_master.png',parts,state,cfg,mass)
    print(json.dumps({'parts':len(parts),'mass_kg':mass['total_kg'],'nominal_overlaps':len(report['nominal_overlaps']),
                      'output':str(output_dir)},indent=2))
    return manifest,report


def write_report(path,state,model,cfg,mass,parts,report):
    lines=['# Dodo v6 master skeleton — first CAD study','',
      'Status: editable reference geometry and estimated envelopes. **Not printable or manufacturing-ready parts.**','',
      'Source: `simulation/models/build_model_v6.py`. See manifest for the SHA-256 and exact configuration. Dimensions are in mm. No simulation files are changed.','',
      '## Deliverables','',
      '- `../assemblies/dodo_v6_master.step`: named skeleton, ten actuator-module allowances, knee/ankle drum studies and torso bounding cage.',
      '- `../assemblies/dodo_v6_skeleton.step`: joint axes, links, feet and torso reservation.',
      '- `../assemblies/dodo_v6_hardware_envelopes.step`: ten module allowances and eight distal drum envelopes.',
      '- `../assemblies/dodo_v6_manifest.json`: joint frames/limits, module centers, source provenance and mass budget.',
      '- `dodo_v6_clearances.json`: nominal and sampled travel/step diagnostics.','',
      '![Master skeleton](dodo_v6_master.png)','',
      '## Geometry and conventions','',
      '| Item | Value |','| --- | --- |',
      f'| Torso top | {1000*model.TORSO_TOP_Z:.1f} |',
      f'| Hip pitch height | {state["joints"]["left_hip_pitch"]["origin_mm"][2]:.3f} |',
      f'| Hip pitch spacing | {state["joints"]["left_hip_pitch"]["origin_mm"][1]-state["joints"]["right_hip_pitch"]["origin_mm"][1]:.3f} |',
      f'| Thigh / shank | {1000*model.THIGH_LENGTH:.1f} / {1000*model.SHANK_LENGTH:.1f} |',
      f'| Knee / ankle motor center spacing | {1000*model.CAPSTAN_CENTER_DISTANCE:.1f} |',
      f'| Sole plane at neutral | Z = {state["sites"]["left_foot"][2]:.3f} |',
      f'| Total mass allowance | {mass["total_kg"]:.2f} kg |',
      f'| Whole-body COM | {np.round(mass["center_of_mass_mm"],3).tolist()} |','',
      'The 0.189 mm sole clearance is inherited from v6 rounding; no silent height correction is applied. The torso bounding cage is not a solid shell and may intentionally contain hardware. Joint order is yaw → roll → pitch → knee → ankle. Both hip drives are independent. All q values are relative to neutral; do not add the neutral pitch offsets a second time.','',
      '### Joint datums','', '| Joint | Origin XYZ | Axis XYZ | Relative travel (deg) |', '| --- | --- | --- | --- |']
    for name,j in state['joints'].items():
        lines.append(f'| {name} | {np.round(j["origin_mm"],3).tolist()} | {np.round(j["axis"],6).tolist()} | {j["range_deg"].tolist()} |')
    lines += ['', '## Hardware confidence','',
      f'The ten cylinders are aggregate simulation module allowances: diameter {cfg["module_envelope"]["diameter_mm"]} × axial length {cfg["module_envelope"]["axial_length_mm"]}. They are not measured motor geometry and do not certify that a complete capstan actuator fits. No mounting holes or purchased-part interfaces have been invented.',
      '', f'Distal drum radii come from v6 (40 / 5 mm). The {cfg["distal_drums"]["width_mm"]} mm drum width and {cfg["distal_drums"]["axial_gap_mm"]} mm axial gap are CAD study assumptions. Motors remain at the simulation locations; only drum envelopes occupy the outboard plane. Module mass already includes drums/bearings/driver; this view adds no mass.',
      '', 'Unresolved hardware is listed in `../source/envelopes_v6.json`. Driver/electronics and bearing details remain unplaced until dimensions are verified. The `8308` name is not a verified motor outside diameter.',
      '', '## Clearance findings','',
      'Checks are enabled: '+str(report['checks_run'])+'. The results below concern estimated envelope solids, not actual manufactured collisions.', '',
      f'Nominal overlapping pairs: **{len(report["nominal_overlaps"])}**. Intentional module-to-owner-link attachments and torso cage overlaps are excluded. See the JSON scope for the tested pair categories; drum-to-link, internal interfaces and floor clearance are not covered by this pass.', '',
      '| Envelope A | Envelope B | Intersection (mm³) |','| --- | --- | ---: |']
    for h in report['nominal_overlaps']: lines.append(f'| {h["a"]} | {h["b"]} | {h["intersection_mm3"]:.1f} |')
    lines += ['', '### Distal travel sweep','', 'Each knee/ankle is sampled every 5° from −60° to +60°, all other joints neutral. A module-to-child-link overlap identifies a region needing actual link/shaft-plane design.', '', '| Joint | Sampled relative angles with overlap |', '| --- | --- |']
    for s in report['distal_travel_sweeps']: lines.append(f'| {s["joint"]} | {s["overlap_q_deg"]} |')
    count=sum(bool(f['overlaps']) for f in report['candidate_step'])
    lines += ['', '### Candidate step geometry','', report['step_note'], '',
      f'Frames with reported envelope overlaps: {count}/{len(report["candidate_step"])}. All sampled targets inside joint limits: {all(f["within_joint_limits"] for f in report["candidate_step"])}.',
      '', '## Next decisions','',
      '1. Measure the motor and controller, then replace the aggregate allowance with separate purchased-part geometry.',
      '2. Choose the hip axis spacing and motor orientations. V6 exposes offsets but its nominal committed values are still zero. Keep roll axes at/inboard of ±100 mm and pitch spacing near 200 mm; prefer vertical offsets.',
      '3. Design the distal child-link path around the motor envelope, together with the output shaft/bearings and drum planes.',
      '4. Re-run clearance studies and put any physical axis/mass changes back into simulation before printing structural parts.',
      '', 'A collision-free physical layout is not claimed. This study makes the remaining packaging work measurable.']
    path.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--output-dir',type=Path,default=ROOT/'cad')
    p.add_argument('--skip-clearance',action='store_true')
    p.add_argument('--skip-preview',action='store_true')
    a=p.parse_args(); build(a.config,a.output_dir,not a.skip_clearance,not a.skip_preview)
