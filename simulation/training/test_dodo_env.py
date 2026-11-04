"""CPU checks for the robot models and training environment (MJX JAX backend).

    python -m unittest discover -s simulation/training -p test_dodo_env.py

These run before spending GPU time. They check plumbing and physics sanity, not
policy quality. On the GPU machine also run them against MJX-Warp:

    DODO_TEST_IMPL=warp python -m unittest discover -s simulation/training -p test_dodo_env.py
"""
import os
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dodo_env  # noqa: E402

IMPL = os.environ.get('DODO_TEST_IMPL', 'jax')
ENV = dodo_env.DodoStepInPlace(config_overrides={'impl': IMPL, 'naconmax': 64})
RESET = jax.jit(ENV.reset)
STEP = jax.jit(ENV.step)


class ModelTest(unittest.TestCase):
    def test_mass_and_geometry(self):
        m = ENV.mj_model
        self.assertTrue(6.0 <= m.body_subtreemass[1] <= 10.0)  # plan range
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        for f in dodo_env.FEET:
            self.assertAlmostEqual(d.site(f).xpos[2], 0.0, delta=0.002)
        self.assertEqual(m.nu, 10)

    def test_model_versions(self):
        """v5 and v6 differ only in where the knee/ankle motors sit; old runs map to v5."""
        self.assertEqual(ENV._config.model, dodo_env.DEFAULT_MODEL)
        self.assertTrue(ENV.xml_path.endswith(dodo_env.MODELS[dodo_env.DEFAULT_MODEL]))
        v5 = mujoco.MjModel.from_xml_path(str(dodo_env.model_path('v5')))
        v6 = mujoco.MjModel.from_xml_path(str(dodo_env.model_path('v6')))
        self.assertAlmostEqual(v5.body_subtreemass[1], v6.body_subtreemass[1], places=6)
        self.assertEqual((v5.nq, v5.nu), (v6.nq, v6.nu))
        thigh_len = float(-v6.body('left_shank').pos[2])
        self.assertAlmostEqual(thigh_len + float(v6.geom('left_knee_module').pos[2]), 0.048, places=4)
        self.assertAlmostEqual(thigh_len + float(v5.geom('left_knee_module').pos[2]), 0.091, places=4)
        import evaluate
        old = dodo_env.default_config().to_dict()
        del old['model']
        self.assertEqual(evaluate.saved_env_config(old).model, 'v5')
        self.assertEqual(evaluate.saved_env_config(dodo_env.default_config().to_dict()).model,
                         dodo_env.DEFAULT_MODEL)

    def test_mjx_matches_mujoco_zero_action(self):
        """MJX and C MuJoCo agree on 1 s of passive standing."""
        m = ENV.mj_model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mx = ENV.mjx_model
        dx = jax.jit(lambda q: dodo_env.mjx_env.make_data(m, qpos=q, ctrl=jp.zeros(10), impl=IMPL, naconmax=64))(d.qpos)
        stepper = jax.jit(lambda dd: dodo_env.mjx_env.step(mx, dd, jp.zeros(10), 50))
        for _ in range(5):
            dx = stepper(dx)
        for _ in range(250):
            mujoco.mj_step(m, d)
        np.testing.assert_allclose(np.asarray(dx.qpos), d.qpos, atol=2e-3)


class EnvTest(unittest.TestCase):
    def test_obs_shapes_and_finite(self):
        s = RESET(jax.random.PRNGKey(0))
        self.assertEqual(s.obs['state'].shape, (46,))
        self.assertTrue(bool(jp.all(jp.isfinite(s.obs['privileged_state']))))

    def test_zero_action_stands_without_pushes(self):
        """The home pose is statically stable under PD: 4 s, zero action, no push."""
        env = dodo_env.DodoStepInPlace(config_overrides={
            'impl': IMPL, 'naconmax': 64, 'push_config.enable': False})
        step = jax.jit(env.step)
        s = jax.jit(env.reset)(jax.random.PRNGKey(1))
        for _ in range(200):
            s = step(s, jp.zeros(10))
            self.assertLess(float(s.done), 0.5)
        self.assertGreater(float(s.data.qpos[2]), 0.37)
        self.assertTrue(bool(jp.all(ENV._contact(s.data))))
        self.assertTrue(np.isfinite(float(s.reward)))

    def test_gait_clock(self):
        s = RESET(jax.random.PRNGKey(2))
        info = dict(s.info, stepping=jp.array(1.0), swing_height=jp.array(0.04),
                    phase=jp.array([0.8, 0.3]))
        stance, h = ENV._gait(info)
        np.testing.assert_array_equal(np.asarray(stance), [False, True])
        self.assertAlmostEqual(float(h[0]), 0.04, places=5)   # mid-swing
        self.assertEqual(float(h[1]), 0.0)
        info['stepping'] = jp.array(0.0)
        stance, h = ENV._gait(info)
        self.assertTrue(bool(jp.all(stance)) and float(jp.max(h)) == 0.0)

    def test_solver_override_does_not_leak(self):
        e = dodo_env.DodoStepInPlace(config_overrides={'impl': IMPL, 'naconmax': 64,
                                                       'solver_iterations': 4})
        self.assertEqual(e.mj_model.opt.iterations, 4)
        self.assertEqual(dodo_env.DodoStepInPlace(config_overrides={'impl': IMPL, 'naconmax': 64})
                         .mj_model.opt.iterations, 2)

    def test_walk_commands(self):
        env = dodo_env.DodoStepInPlace(config_overrides={
            'impl': IMPL, 'naconmax': 64, **dodo_env.WALK_OVERRIDES})
        keys = jax.random.split(jax.random.PRNGKey(4), 2000)
        sample = jax.jit(jax.vmap(env.sample_command, in_axes=(0, None)))
        self.assertEqual(float(jp.abs(sample(keys, jp.array(0.0))).max()), 0.0)  # standing: no command
        cmd = np.asarray(sample(keys, jp.array(1.0)))
        zero = np.all(cmd == 0, axis=1).mean()
        self.assertAlmostEqual(zero, dodo_env.WALK_OVERRIDES['zero_command_probability'], delta=0.03)
        self.assertTrue(-0.1 <= cmd[:, 0].min() and cmd[:, 0].max() <= 0.3)

    def test_symmetry_error(self):
        """Zero for a perfectly mirrored half-cycle-shifted gait, positive otherwise."""
        s = RESET(jax.random.PRNGKey(5))
        info = dict(s.info, stepping=jp.array(1.0), gait_freq=jp.array(1.25),
                    command=jp.zeros(3), hist_len=jp.array(dodo_env.HIST_LEN))
        k = int(ENV.half_cycle_steps(info['gait_freq']))
        self.assertEqual(k, 20)
        left_now = jp.array([0.1, 0.05, 0.3, -0.2, 0.1])
        left_past = jp.array([-0.05, 0.02, -0.1, 0.1, 0.05])
        sign = jp.array(dodo_env.MIRROR_SIGN)
        q = jp.concatenate([left_now, sign * left_past])        # right now = M left(t-k)
        past = jp.concatenate([left_past, sign * left_now])     # right(t-k) = M left now
        info['q_hist'] = jp.zeros((dodo_env.HIST_LEN, 10)).at[k - 1].set(past)
        self.assertAlmostEqual(float(ENV._symmetry_error(q, info)), 0.0, places=6)
        q_bad = q.at[9].add(0.2)                                 # right ankle off by 0.2 rad
        self.assertAlmostEqual(float(ENV._symmetry_error(q_bad, info)), 0.04, places=5)
        info['stepping'] = jp.array(0.0)
        self.assertEqual(float(ENV._symmetry_error(q_bad, info)), 0.0)

    def test_action_filter(self):
        """Off by default; with a cutoff, targets approach a held action exponentially."""
        self.assertEqual(ENV._filter_alpha, 1.0)
        env = dodo_env.DodoStepInPlace(config_overrides={
            'impl': IMPL, 'naconmax': 64, 'action_delay_probability': 0.0,
            **dodo_env.SMOOTH_OVERRIDES})
        alpha = env._filter_alpha
        self.assertAlmostEqual(alpha, 1 - np.exp(-2 * np.pi * 6.0 * 0.02), places=6)
        step = jax.jit(env.step)
        s = jax.jit(env.reset)(jax.random.PRNGKey(6))
        a = jp.full(10, 0.2)
        for n in range(1, 4):
            s = step(s, a)
            expect = 0.2 * (1 - (1 - alpha) ** n) * env._config.action_scale
            np.testing.assert_allclose(np.asarray(s.info['motor_targets'] - env._default_pose),
                                       expect, atol=1e-6)
        np.testing.assert_allclose(np.asarray(s.info['last_act']), 0.2)   # obs keeps the raw action

    def test_domain_randomization_batches(self):
        keys = jax.random.split(jax.random.PRNGKey(3), 4)
        model, in_axes = dodo_env.domain_randomize(ENV.mjx_model, keys)
        self.assertEqual(model.body_mass.shape[0], 4)
        self.assertEqual(model.actuator_forcerange.shape, (4, 10, 2))
        self.assertEqual(in_axes.body_mass, 0)



class ReplayTest(unittest.TestCase):
    def test_replay_obs_matches_env(self):
        """replay_mujoco rebuilds the actor observation exactly as the env does."""
        import replay_mujoco
        env = dodo_env.DodoStepInPlace(config_overrides={
            'impl': IMPL, 'naconmax': 64, 'noise_config.level': 0.0, **dodo_env.WALK_OVERRIDES})
        # At reset both sides come straight from a forward pass. (After mj_step, sensors
        # and site frames lag qpos by one substep, identically in MJX and C MuJoCo.)
        s = jax.jit(env.reset)(jax.random.PRNGKey(7))
        self.assertTrue(float(s.info['stepping']) == 1.0 and float(jp.abs(s.info['command']).sum()) > 0)
        r = replay_mujoco.Replay(env, policy=lambda obs, key: (jp.zeros(10), {}))
        m = env.mj_model
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:] = np.asarray(s.data.qpos), np.asarray(s.data.qvel)
        mujoco.mj_forward(m, d)
        info = {k: np.asarray(v) for k, v in s.info.items() if k != 'rng'}
        info['stepping'] = float(info['stepping'])
        got = r.obs(d, info, np.random.default_rng(0))
        np.testing.assert_allclose(got, np.asarray(s.obs['state']), atol=1e-4)


if __name__ == '__main__':
    unittest.main()
