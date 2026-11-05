"""Geometric checks against MuJoCo, not merely snapshots of CAD code."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import cadquery as cq
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('skeleton', ROOT/'cad/source/master_skeleton.py')
cad = importlib.util.module_from_spec(spec); spec.loader.exec_module(cad)


class MasterSkeletonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = cad.load_model()
        cls.cfg = json.loads(cad.CONFIG.read_text())

    def assert_matches_mujoco(self, cfg, pose):
        xml = cad.xml_for_config(self.model,cfg)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,0)
        for name, degrees in pose.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = np.deg2rad(degrees)
        mujoco.mj_forward(model,data)
        state = cad.kinematics(xml,pose)
        for name,j in state['joints'].items():
            jid=model.joint(name).id
            np.testing.assert_allclose(j['origin_mm'],1000*data.xanchor[jid],atol=1e-7)
            np.testing.assert_allclose(j['axis'],data.xaxis[jid],atol=1e-9)
        for g in state['geoms']:
            np.testing.assert_allclose(g['center'],1000*data.geom_xpos[model.geom(g['name']).id],atol=1e-7)
        for name,p in state['sites'].items():
            np.testing.assert_allclose(p,1000*data.site_xpos[model.site(name).id],atol=1e-7)
        mass=cad.mass_budget(state)
        self.assertAlmostEqual(mass['total_kg'],model.body_subtreemass[1],places=10)
        np.testing.assert_allclose(mass['center_of_mass_mm'],1000*data.subtree_com[1],atol=1e-7)

    def test_nominal_matches_mujoco(self):
        self.assert_matches_mujoco(self.cfg,{})

    def test_offsets_and_all_axes_match_mujoco(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['hip_roll_offset_mm']=[7,-10,-20]
        cfg['hip_pitch_offset_mm']=[-4,10,-20]
        pose={'left_hip_yaw':14,'left_hip_roll':-10,'left_hip_pitch':18,
              'left_knee_pitch':-22,'left_ankle_pitch':4,'right_hip_yaw':-11,
              'right_hip_roll':8,'right_hip_pitch':-15,'right_knee_pitch':21,'right_ankle_pitch':-6}
        self.assert_matches_mujoco(cfg,pose)

    def test_neutral_datums_and_module_count(self):
        state=cad.kinematics(cad.xml_for_config(self.model,self.cfg))
        self.assertEqual(len(state['joints']),10)
        self.assertEqual(sum(g['name'].endswith('_module') for g in state['geoms']),10)
        for side,sign in [('left',1),('right',-1)]:
            np.testing.assert_allclose(state['joints'][side+'_hip_pitch']['origin_mm'],[0,sign*100,330],atol=1e-7)
            for joint,geom in [('knee_pitch','knee_module'),('ankle_pitch','ankle_module')]:
                origin=state['joints'][side+'_'+joint]['origin_mm']
                motor=next(g for g in state['geoms'] if g['name']==side+'_'+geom)
                self.assertAlmostEqual(np.linalg.norm(motor['center']-origin),48,places=7)
        self.assertAlmostEqual(state['sites']['left_foot'][2],0.188891220742,places=5)

    def test_candidate_step_closes_and_is_level_with_fixed_stance(self):
        xml=cad.xml_for_config(self.model,self.cfg)
        home=cad.kinematics(xml)
        frames=cad.candidate_step(self.model)
        self.assertTrue(all(abs(v)<1e-10 for f in [frames[0],frames[-1]] for v in f['q_degrees'].values()))
        max_lift=0
        for f in frames:
            state=cad.kinematics(xml,f['q_degrees'])
            np.testing.assert_allclose(state['sites']['right_foot'],home['sites']['right_foot'],atol=1e-8)
            np.testing.assert_allclose(state['bodies']['left_foot'][0][:,2],[0,0,1],atol=1e-8)
            for name,q in f['q_degrees'].items():
                lo,hi=state['joints'][name]['range_deg']; self.assertTrue(lo<=q<=hi)
            max_lift=max(max_lift,state['sites']['left_foot'][2]-home['sites']['left_foot'][2])
        self.assertAlmostEqual(max_lift,40,places=5)

    def test_distal_drum_planes_and_gap(self):
        state=cad.kinematics(cad.xml_for_config(self.model,self.cfg))
        parts={p['name']:p for p in cad.parts_for(state,self.model,self.cfg)}
        for side in ['left','right']:
            for j,m in [('knee_pitch','knee_module'),('ankle_pitch','ankle_module')]:
                output=parts[f'{side}_{j}_output_drum']['shape']
                inp=parts[f'{side}_{j}_input_drum']['shape']
                motor=parts[f'{side}_{m}']['shape']
                self.assertAlmostEqual(output.distance(inp),3,places=6)
                self.assertAlmostEqual(output.distance(motor),3,places=6)
                self.assertAlmostEqual(inp.distance(motor),3,places=6)

    def test_step_roundtrip_solids_volume_and_units(self):
        state=cad.kinematics(cad.xml_for_config(self.model,self.cfg))
        parts=cad.parts_for(state,self.model,self.cfg)
        for filename,kinds in [('dodo_v6_master.step',set(cad.COLORS)),
            ('dodo_v6_skeleton.step',{'axis','link','foot','torso'}),
            ('dodo_v6_hardware_envelopes.step',{'module','drum','input'})]:
            expected=[p['shape'] for p in parts if p['kind'] in kinds]
            imported=cq.importers.importStep(str(ROOT/'cad/assemblies'/filename)).val()
            self.assertEqual(len(imported.Solids()),len(expected))
            self.assertTrue(all(s.isValid() for s in imported.Solids()))
            self.assertAlmostEqual(imported.Volume(),sum(s.Volume() for s in expected),delta=.01)
            if filename=='dodo_v6_master.step':
                self.assertAlmostEqual(imported.BoundingBox().zmax,500.7,places=5)


if __name__=='__main__': unittest.main()
