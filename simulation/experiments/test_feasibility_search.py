"""Checks the planning LP independently of the search's IK acceptance."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from dodo_feasibility_search import Search


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        root=Path(__file__).resolve().parents[1]
        a=SimpleNamespace(model=str(root/'models/dodo_floating.xml'),output=str(Path(self.tmp.name)/'result.json'),
            support='left',budget_seconds=60,clearance_mm=5.,joint_margin_deg=2.,torque_limit=3.8,
            max_nfev=100,path_knots=16,max_paths=6,max_candidates=144,seed=20261031)
        self.s=Search(a)
        self.balance=np.array(json.loads((root/'results/single_support_left/single_support_left.json').read_text())['pose_search']['single_support_qpos'])

    def tearDown(self):self.tmp.cleanup()

    def test_known_balance_has_static_force_solution(self):
        f=self.s.statics(self.balance,1,single=True)
        self.assertTrue(f['feasible'],f)
        self.assertAlmostEqual(f['support_force_N'],self.s.r.weight,places=6)
        self.assertEqual(f['minimum_swing_force_N'],0.)
        self.assertLessEqual(f['peak_torque_Nm'],3.8+1e-8)
        self.assertEqual(self.s.r.d.time,0.)

    def test_insufficient_torque_is_rejected(self):
        self.s.a.torque_limit=.01
        self.assertFalse(self.s.statics(self.balance,1,single=True)['feasible'])

    def test_double_support_and_joint_margin_are_separate_checks(self):
        self.assertTrue(self.s.statics(self.s.r.q0,1,initial=True)['feasible'])
        g=self.s.geometry(self.balance,1,lift=.025,release=True)
        self.assertIn('joint_margin',g['reasons'])
        self.assertGreater(g['com_support_margin_m'],.002)

    def test_release_com_target_is_a_search_hint_not_a_physical_constraint(self):
        self.s.joint_margin=np.deg2rad(.5)
        g=self.s.geometry(self.balance,1,target=self.s.r.c0,lift=.025,release=True)
        self.assertGreater(g['com_target_error_m'],.002)
        self.assertNotIn('com_target',g['reasons'])
        self.assertNotIn('release_com_margin',g['reasons'])

    def test_checkpoint_serializes_numpy_booleans(self):
        self.s.report['numpy_boolean']=np.bool_(True)
        self.s.save()
        self.assertTrue(json.loads(Path(self.s.a.output).read_text())['numpy_boolean'])


if __name__=='__main__':unittest.main()
