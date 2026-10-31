"""Regression checks for measured release gating and bounded unloading feedback.

Run from the repository root:
  python -m unittest discover -s simulation/experiments -p test_transition_v2.py
"""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import dodo_transition_v2 as trial


class UnloadingTests(unittest.TestCase):
    def setUp(self):
        self.r = trial.Robot(Path(__file__).resolve().parents[1]/'models/dodo_floating.xml')
        self.args = SimpleNamespace(release_load_fraction=.05, target_load_fraction=.01,
            unload_ramp_seconds=4., force_kp=.5, force_ki=.3,
            max_com_correction_mm=0., com_load_gain_mm_per_N=1., com_feedforward_mm=3.,
            com_load_integral_mm_per_N_s=.5, posture_update_seconds=.5)

    def test_release_requires_actual_unloading_and_com_margin(self):
        m = {'right_fz_N':4.4,'left_fz_N':15.9,'com_support_margin_m':.01,
             'com_speed_m_s':0.,'left_tilt_deg':0.,'left_drift_m':0.,
             'left_sole_speed_m_s':0.,'left_cop_margin_m':.01,
             'torque_utilization':.5,'unwanted_contacts':0,
             'right_predicted_fz_N':0.}
        self.assertFalse(all(trial.release_gates(self.r,self.args,m,'left').values()))
        m.update(right_fz_N=.1,left_fz_N=20.2,com_support_margin_m=-.001)
        self.assertFalse(trial.release_gates(self.r,self.args,m,'left')['com_margin'])
        m['com_support_margin_m']=.01
        self.assertTrue(all(trial.release_gates(self.r,self.args,m,'left').values()))
        m['right_fz_N']=float('nan')
        self.assertFalse(trial.release_gates(self.r,self.args,m,'left')['swing_load'])

    def make_unloader(self,support='left'):
        swing='right' if support=='left' else 'left'
        pivot=1 if support=='left' else 0
        return trial.ActiveUnloader(self.r,self.args,support,pivot,
            self.r.corners(self.r.d,swing)[pivot].copy(),self.r.q0)

    def test_force_ramp_saturates_without_integrator_windup(self):
        u=self.make_unloader();m={'right_fz_N':4.4};u.start(m)
        self.assertAlmostEqual(u.command_N,4.4)
        for t in np.arange(0,8,self.r.dt):u.update(m,float(t))
        self.assertAlmostEqual(u.target_N,.01*self.r.weight)
        self.assertEqual(u.command_N,0.)
        integral=u.force_integral
        for t in np.arange(8,12,self.r.dt):u.update(m,float(t))
        self.assertEqual(u.force_integral,integral)
        self.assertEqual(u.filtered_N,4.4)

    def test_posture_request_mirrors_and_preserves_simulated_state(self):
        self.args.max_com_correction_mm=15.
        original=self.r.d.qpos.copy()
        for support,sign in [('left',1),('right',-1)]:
            u=self.make_unloader(support);u.start({u.swing+'_fz_N':4.4})
            seen=[]
            def reject(r,seed,support,pivot,anchor,desired,**kwargs):
                seen.append(desired.copy())
                return seed.copy(),{'com_error_m':.02,'stance_position_error_m':0.,
                    'stance_tilt_deg':0.,'pivot_error_m':0.,'swing_clearance_m':0.,
                    'collision_distance_m':.05,'min_joint_margin_deg':10.,'max_joint_step_deg':0.}
            with patch.object(trial,'plan_pose',side_effect=reject):
                q=u.update({u.swing+'_fz_N':4.4},4.)
                self.assertGreater(sign*(seen[-1][1]-u.base_com[1]),0.)
                self.assertLessEqual(abs(seen[-1][1]-u.base_com[1]),.0010001)
                self.assertFalse(u.history[-1]['accepted'])
                self.assertIn('com_residual',u.history[-1]['reasons'])
                integral=u.com_integral
                u.update({u.swing+'_fz_N':4.4},4.002)
                self.assertEqual(u.com_integral,integral)
                np.testing.assert_allclose(q,self.r.q0)
        np.testing.assert_array_equal(self.r.d.qpos,original)

    def test_signed_distance_gradient_outside_and_inside_contact(self):
        for x in [.23,.19]:
            m=trial.mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
                <geom name="a" type="sphere" size=".1"/>
                <body pos="{x} 0 0"><freejoint/><geom name="b" type="sphere" size=".1"/></body>
                </worldbody></mujoco>''')
            d=trial.mujoco.MjData(m);trial.mujoco.mj_forward(m,d)
            dist,grad=trial.collision_distance_gradient(SimpleNamespace(m=m),d,0,1)
            d.qpos[0]+=1e-7;trial.mujoco.mj_forward(m,d)
            numerical=(trial.mujoco.mj_geomDistance(m,d,0,1,.05,None)-dist)/1e-7
            self.assertAlmostEqual(grad[0],numerical,places=6)
            self.assertGreater(grad[0],0.)

    def test_infeasible_constraint_diagnosis_is_explicit(self):
        report=trial.constraint_diagnosis(np.array([[1.],[1.]]),np.array([1.,-np.inf]),
                                          np.array([np.inf,0.]),['minimum','maximum'])
        self.assertFalse(report['linear_constraints_feasible'])
        self.assertTrue(report['elastic_relaxations'])


if __name__=='__main__':unittest.main()
