"""Checks for plan identity, release evidence, and contact handoff bounds."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import dodo_release_test as trial


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.a = trial.parser().parse_args([])
        self.r = trial.v2.Robot(self.a.model)

    def test_plan_rejects_wrong_model_and_noninitial_start(self):
        original = self.r.d.qpos.copy()
        q, anchor = trial.load_plan(self.r, self.a.plan, self.a.model)
        self.assertEqual(q.shape, (65,17))
        np.testing.assert_array_equal(original, self.r.d.qpos)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'plan.json'
            plan = json.loads(Path(self.a.plan).read_text())
            plan['model_sha256'] = 'wrong'; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'SHA256'): trial.load_plan(self.r,path,self.a.model)
            plan['model_sha256'] = trial.sha(self.a.model)
            plan['qpos'][0][0] += .001; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'bird_stand'): trial.load_plan(self.r,path,self.a.model)

    def test_success_requires_actual_load_complete_path_and_clearance(self):
        m = {'right_fz_N':.1,'left_fz_N':20.2,'com_support_margin_m':.003,
             'com_speed_m_s':0.,'left_tilt_deg':0.,'left_drift_m':0.,
             'left_sole_speed_m_s':0.,'left_cop_margin_m':.003,
             'torque_utilization':.5,'unwanted_contacts':0,
             'minimum_body_clearance_m':.006,'left_contacts':4}
        diag = {'min_joint_margin_deg':2.}
        self.assertTrue(all(trial.gates(self.r,self.a,m,diag,True).values()))
        self.assertFalse(all(trial.gates(self.r,self.a,m,diag,False).values()))
        m['right_fz_N'] = 4.; m['right_predicted_fz_N'] = 0.
        self.assertFalse(trial.gates(self.r,self.a,m,diag,True)['swing_load'])
        m['right_fz_N'] = .1; m['minimum_body_clearance_m'] = .0049
        self.assertFalse(trial.gates(self.r,self.a,m,diag,True)['body_clearance'])

    def test_contact_force_caps_reject_negative_or_nonfinite(self):
        com, refs = trial.v2.pose_targets(self.r,self.r.q0)
        for bad in [-1.,float('nan'),4*self.r.weight]:
            with self.assertRaisesRegex(ValueError, 'corner force cap'):
                trial.v2.WholeBodyQP(self.r,.8,.002).solve(
                    self.r.d, {'left':[0,1,2,3],'right':[2]}, self.r.q0,com,refs,
                    force_contact_caps={'right':[bad,0.,self.r.weight,0.]})


if __name__ == '__main__': unittest.main()
