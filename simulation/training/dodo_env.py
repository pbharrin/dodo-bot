"""Dodo v5 stand / step-in-place environment for MuJoCo Playground (MJX or MJX-Warp).

Task: on each episode the robot is either told to STAND (both feet planted) or to
STEP IN PLACE following a two-foot gait clock. The base-velocity command is zero
by default; widen `lin_vel_x` etc. in the config later to turn this into a
walking task without changing the code.

Policy interface (keep identical on hardware, see docs/project-plan.md section 7):
  * 50 Hz policy (ctrl_dt = 0.02 s).
  * action a (10,) -> joint position target q* = q_home + action_scale * a,
    clamped to the joint range by MuJoCo (ctrlrange inherits the joint range).
  * Joint order: left hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch,
    then right in the same order. qpos = 0 is the neutral bird stance.
  * Actor observation, "state" (46,):
      gyro (3, rad/s, IMU frame) | gravity in IMU frame (3) | command (3) |
      joint pos - home (10, rad) | joint vel (10, rad/s) * 0.1 |
      last action (10) | gait clock cos/sin per foot (4) | stepping flag (1) |
      gait frequency (1, Hz) | swing height (1, m) * 10
    No base linear velocity: the real robot cannot measure it directly.
  * Critic observation, "privileged_state": actor obs without noise plus base
    velocity, height, contact, foot heights/velocities, torques and pushes.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'dodo_v5.xml'
FEET = ('left_foot', 'right_foot')
NUM_JOINTS = 10
# Left<->right mirror of one leg's joints (hip yaw, hip roll, hip pitch, knee, ankle):
# all joint axes are identical on both sides, so mirroring flips yaw and roll only.
MIRROR_SIGN = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
HIST_LEN = 32                  # control steps of joint history (covers half a cycle down to ~0.8 Hz)


# Stage 2: walk slowly forward/back, sidestep a little and turn, from the stage-1
# policy (train_ppo.py --task walk --warm-start <stage-1 run>). ESTIMATE ranges:
# 0.3 m/s is a slow walk for a 0.33 m leg (Froude number ~0.03).
WALK_OVERRIDES = {
    'lin_vel_x': [-0.1, 0.3],
    'lin_vel_y': [-0.1, 0.1],
    'ang_vel_yaw': [-0.5, 0.5],
    'zero_command_probability': 0.2,
    'stand_probability': 0.15,
    'reward_config.scales.symmetry': -4.0,
}


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,          # 20 s
        action_repeat=1,
        action_scale=0.5,             # rad per unit action (policy output is tanh-bounded to [-1, 1])
        soft_joint_pos_limit_factor=0.9,
        impl='warp',                  # 'warp' on the GPU box, 'jax' for CPU smoke tests
        naconmax=16 * 8192,           # contact buffer across all worlds (<= 8 used per world)
        njmax=96,                     # constraint rows per world
        solver_iterations=None,       # None = model XML value (2); raise to 3-4 if Warp
        ls_iterations=None,           #   warns 'solver iterations limit reached' often
        # --- task ---
        stand_probability=0.25,       # fraction of episodes that only stand
        gait_frequency=[1.0, 1.5],    # full left+right cycles per second
        swing_height=[0.03, 0.05],    # m, sole clearance target at mid-swing
        duty_factor=0.6,              # stance fraction of each foot's cycle
        lin_vel_x=[0.0, 0.0],         # m/s; widen for walking (stage 2)
        lin_vel_y=[0.0, 0.0],
        ang_vel_yaw=[0.0, 0.0],
        zero_command_probability=0.0, # stepping episodes/segments told to step in place (stage 2)
        command_resample_steps=500,
        base_height_target=0.40,      # torso origin height at home
        min_feet_distance=0.12,       # m between foot centres, lateral
        # --- sim-to-real ---
        action_delay_probability=0.5, # chance an episode uses one control-step delay
        encoder_offset=0.02,          # rad, per-episode calibration error in obs
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.01, joint_vel=0.5, gravity=0.05, gyro=0.1),
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[3.0, 8.0],     # s
            magnitude_range=[0.05, 0.35],  # m/s change of base velocity
        ),
        reward_config=config_dict.create(
            tracking_sigma=0.1,
            feet_phase_sigma=0.002,
            # Symmetry: yaw/roll terms fade out for sideways or turning commands, which
            # legitimately need an asymmetric gait. Scales in m/s and rad/s.
            symmetry_lateral_sigma=0.03,
            symmetry_yaw_sigma=0.1,
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.75,
                lin_vel_z=-1.0,
                ang_vel_xy=-0.15,
                orientation=-5.0,
                base_height=-10.0,
                torques=-1e-3,
                action_rate=-0.05,
                energy=-1e-3,
                dof_pos_limits=-1.0,
                pose=-0.5,
                feet_phase=2.0,
                contact_match=1.0,
                feet_slip=-0.5,
                feet_flat=-2.0,
                feet_distance=-10.0,
                alive=1.0,
                termination=-50.0,
                symmetry=0.0,          # stage 2 uses -4.0 (WALK_OVERRIDES)
            ),
        ),
    )


class DodoStepInPlace(mjx_env.MjxEnv):
    """Stand, or step in place on a gait clock, while staying upright."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        xml_path: Path = MODEL_PATH,
    ):
        # Fresh config per instance: overrides are applied in place, so a shared
        # default would leak one env's overrides into the next.
        config = default_config() if config is None else config.copy_and_resolve_references()
        super().__init__(config, config_overrides)
        self._xml_path = str(xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        if self._config.solver_iterations is not None:
            self._mj_model.opt.iterations = int(self._config.solver_iterations)
        if self._config.ls_iterations is not None:
            self._mj_model.opt.ls_iterations = int(self._config.ls_iterations)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        m = self._mj_model
        self._init_q = jp.array(m.keyframe('home').qpos)
        self._default_pose = jp.array(m.keyframe('home').qpos[7:])
        lowers, uppers = m.jnt_range[1:].T
        c, r = (lowers + uppers) / 2, uppers - lowers
        f = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = jp.array(c - 0.5 * r * f)
        self._soft_uppers = jp.array(c + 0.5 * r * f)
        names = [m.joint(i).name for i in range(1, m.njnt)]
        # Pose weights while stepping: hip yaw/roll held near home, sagittal joints free.
        self._step_pose_weights = jp.array(
            [1.0 if ('yaw' in n or 'roll' in n) else 0.05 for n in names])
        self._imu_site = m.site('imu').id
        self._torso_id = m.body('torso').id
        self._feet_site = np.array([m.site(f).id for f in FEET])
        self._feet_body = np.array([m.body(f).id for f in FEET])
        self._contact_adr = np.array(
            [m.sensor_adr[m.sensor(f'{f}_floor_found').id] for f in FEET])
        adr = []
        for f in FEET:
            s = m.sensor(f'{f}_global_linvel').id
            adr.append(list(range(m.sensor_adr[s], m.sensor_adr[s] + 3)))
        self._foot_linvel_adr = jp.array(adr)

    # ------------------------------------------------------------------ sensors
    def _sensor(self, data, name):
        return mjx_env.get_sensor_data(self._mj_model, data, name)

    def _gravity(self, data):
        return data.site_xmat[self._imu_site].T @ jp.array([0.0, 0.0, -1.0])

    def _contact(self, data):
        return data.sensordata[self._contact_adr] > 0

    # ------------------------------------------------------------------ gait
    def _gait(self, info):
        """Desired stance flags (2,) and swing-foot height targets (2,)."""
        phi = info['phase']                             # in [0, 1) per foot
        d = self._config.duty_factor
        swing = phi >= d
        s = jp.clip((phi - d) / (1.0 - d), 0.0, 1.0)
        height = info['swing_height'] * 0.5 * (1.0 - jp.cos(2.0 * jp.pi * s))
        stepping = info['stepping'] > 0.5
        stance = jp.where(stepping, ~swing, jp.ones(2, dtype=bool))
        height = jp.where(stepping, height * swing, jp.zeros(2))
        return stance, height

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        cfg = self._config
        qpos = self._init_q
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        yaw = jax.random.uniform(k1, (1,), minval=-jp.pi, maxval=jp.pi)
        qpos = qpos.at[3:7].set(math.quat_mul(
            qpos[3:7], math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)))
        qpos = qpos.at[7:].add(jax.random.uniform(k2, (NUM_JOINTS,), minval=-0.05, maxval=0.05))
        qpos = qpos.at[2].add(0.01)                     # drop from 1 cm to avoid penetration
        qvel = jp.zeros(self._mjx_model.nv).at[0:6].set(
            jax.random.uniform(k3, (6,), minval=-0.1, maxval=0.1))
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
            impl=self._mjx_model.impl.value,
            naconmax=cfg.naconmax, njmax=cfg.njmax)
        data = mjx.forward(self._mjx_model, data)

        rng, kf, kh, ks, kc, kd, ko, kp, kph = jax.random.split(rng, 9)
        freq = jax.random.uniform(kf, minval=cfg.gait_frequency[0], maxval=cfg.gait_frequency[1])
        stepping = (jax.random.uniform(ks) > cfg.stand_probability).astype(jp.float32)
        phase0 = jax.random.uniform(kph)
        push_interval = jax.random.uniform(
            kp, minval=cfg.push_config.interval_range[0], maxval=cfg.push_config.interval_range[1])
        info = {
            'rng': rng,
            'step': 0,
            'command': self.sample_command(kc, stepping),
            'stepping': stepping,
            'gait_freq': freq,
            'swing_height': jax.random.uniform(
                kh, minval=cfg.swing_height[0], maxval=cfg.swing_height[1]),
            'phase': jp.mod(jp.array([phase0, phase0 + 0.5]), 1.0),
            'delay': (jax.random.uniform(kd) < cfg.action_delay_probability).astype(jp.float32),
            'encoder_offset': jax.random.uniform(
                ko, (NUM_JOINTS,), minval=-cfg.encoder_offset, maxval=cfg.encoder_offset),
            'last_act': jp.zeros(NUM_JOINTS),
            'motor_targets': jp.zeros(NUM_JOINTS),
            'q_hist': jp.zeros((HIST_LEN, NUM_JOINTS)),   # [0] = previous control step
            'hist_len': 0,
            'push': jp.zeros(2),
            'push_step': 0,
            'push_interval_steps': jp.round(push_interval / self.dt).astype(jp.int32),
        }
        metrics = {f'reward/{k}': jp.zeros(()) for k in cfg.reward_config.scales.keys()}
        metrics['contact_match_frac'] = jp.zeros(())
        metrics['max_swing_height'] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------ step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        info = state.info

        # Random base-velocity push.
        info['rng'], k1, k2 = jax.random.split(info['rng'], 3)
        theta = jax.random.uniform(k1, maxval=2 * jp.pi)
        mag = jax.random.uniform(
            k2, minval=cfg.push_config.magnitude_range[0], maxval=cfg.push_config.magnitude_range[1])
        push = jp.array([jp.cos(theta), jp.sin(theta)]) * mag
        push *= jp.mod(info['push_step'] + 1, info['push_interval_steps']) == 0
        push *= cfg.push_config.enable
        data = state.data.replace(qvel=state.data.qvel.at[:2].add(push))

        # One control-step latency on some episodes.
        applied = jp.where(info['delay'] > 0.5, info['last_act'], action)
        motor_targets = self._default_pose + applied * cfg.action_scale
        data = mjx_env.step(self._mjx_model, data, motor_targets, self.n_substeps)
        info['motor_targets'] = motor_targets

        contact = self._contact(data)
        done = self._get_termination(data)
        rewards = self._get_reward(data, action, info, contact, done)
        rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
        reward = sum(rewards.values()) * self.dt

        # Advance the gait clock; resample command periodically.
        info['phase'] = jp.mod(info['phase'] + info['gait_freq'] * self.dt, 1.0)
        info['push'] = push
        info['push_step'] += 1
        info['step'] += 1
        info['last_act'] = action
        q_now = data.qpos[7:] - self._default_pose
        info['q_hist'] = jp.roll(info['q_hist'], 1, axis=0).at[0].set(q_now)
        info['hist_len'] = jp.minimum(info['hist_len'] + 1, HIST_LEN)
        info['rng'], kc = jax.random.split(info['rng'])
        resample = info['step'] >= cfg.command_resample_steps
        info['command'] = jp.where(resample, self.sample_command(kc, info['stepping']), info['command'])
        info['step'] = jp.where(resample | done, 0, info['step'])

        stance, _ = self._gait(info)
        for k, v in rewards.items():
            state.metrics[f'reward/{k}'] = v
        state.metrics['contact_match_frac'] = jp.mean(stance == contact)
        state.metrics['max_swing_height'] = jp.max(data.site_xpos[self._feet_site, 2])

        obs = self._get_obs(data, info)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _get_termination(self, data):
        up = data.site_xmat[self._imu_site][2, 2]       # cos(tilt)
        fallen = (up < 0.7) | (data.qpos[2] < 0.25)
        return fallen | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    # ------------------------------------------------------------------ obs
    def _noise(self, info, x, scale):
        info['rng'], k = jax.random.split(info['rng'])
        lvl = self._config.noise_config.level
        return x + (2 * jax.random.uniform(k, x.shape) - 1) * lvl * scale

    def _get_obs(self, data, info):
        sc = self._config.noise_config.scales
        gyro = self._sensor(data, 'gyro')
        gravity = self._gravity(data)
        q = data.qpos[7:] - self._default_pose
        dq = data.qvel[6:]
        clock = 2 * jp.pi * info['phase']
        clock = jp.where(info['stepping'] > 0.5,
                         jp.concatenate([jp.cos(clock), jp.sin(clock)]), jp.zeros(4))
        task = jp.hstack([info['stepping'], info['gait_freq'], info['swing_height'] * 10.0])
        state = jp.hstack([
            self._noise(info, gyro, sc.gyro),
            self._noise(info, gravity, sc.gravity),
            info['command'],
            self._noise(info, q + info['encoder_offset'], sc.joint_pos),
            self._noise(info, dq, sc.joint_vel) * 0.1,
            info['last_act'],
            clock,
            task,
        ])
        stance, target_h = self._gait(info)
        privileged = jp.hstack([
            gyro, gravity, info['command'], q, dq * 0.1, info['last_act'], clock, task,
            self._sensor(data, 'local_linvel'),
            data.qpos[2],
            self._contact(data),
            stance,
            data.site_xpos[self._feet_site, 2],
            target_h,
            data.sensordata[self._foot_linvel_adr].ravel(),
            data.actuator_force * 0.1,
            info['push'],
            info['delay'],
        ])
        return {'state': state, 'privileged_state': privileged}

    # ------------------------------------------------------------------ symmetry
    def half_cycle_steps(self, gait_freq):
        """Control steps in half a gait cycle (clipped to the history length)."""
        k = jp.round(0.5 / (gait_freq * self.dt)).astype(jp.int32)
        return jp.clip(k, 1, HIST_LEN)

    def _symmetry_error(self, q, info):
        """Squared error between each leg now and the other leg, mirrored, half a cycle ago.

        In a symmetric gait right(t) = M left(t - T/2) and left(t) = M right(t - T/2).
        Returns (error, valid). Yaw/roll terms are weighted down for sideways or
        turning commands.
        """
        rc = self._config.reward_config
        k = self.half_cycle_steps(info['gait_freq'])
        past = info['q_hist'][k - 1]                      # q at t - k
        sign = jp.array(MIRROR_SIGN)
        cmd = info['command']
        straight = jp.exp(-jp.square(cmd[1] / rc.symmetry_lateral_sigma)
                          - jp.square(cmd[2] / rc.symmetry_yaw_sigma))
        w = jp.array([1.0, 1.0, 0.0, 0.0, 0.0]) * straight + jp.array([0.0, 0.0, 1.0, 1.0, 1.0])
        e_r = q[5:] - sign * past[:5]
        e_l = q[:5] - sign * past[5:]
        err = jp.sum(w * (jp.square(e_r) + jp.square(e_l)))
        valid = (info['stepping'] > 0.5) & (info['hist_len'] >= k)
        return jp.where(valid, err, 0.0)

    # ------------------------------------------------------------------ reward
    def _get_reward(self, data, action, info, contact, done):
        rc = self._config.reward_config
        cmd = info['command']
        linvel = self._sensor(data, 'local_linvel')
        gyro = self._sensor(data, 'gyro')
        gravity = self._gravity(data)
        q = data.qpos[7:]
        dq = data.qvel[6:]
        tau = data.actuator_force
        stance, target_h = self._gait(info)
        foot_z = data.site_xpos[self._feet_site, 2]
        foot_v = data.sensordata[self._foot_linvel_adr]
        foot_zaxis = data.xmat[self._feet_body][:, :, 2]   # (2, 3)

        stepping = info['stepping'] > 0.5
        weights = jp.where(stepping, self._step_pose_weights, jp.ones(NUM_JOINTS))
        lf, rf = data.site_xpos[self._feet_site[0]], data.site_xpos[self._feet_site[1]]
        xm = data.site_xmat[self._imu_site]
        yaw = jp.arctan2(xm[1, 0], xm[0, 0])
        lateral = jp.abs(jp.cos(yaw) * (lf[1] - rf[1]) - jp.sin(yaw) * (lf[0] - rf[0]))

        return {
            'tracking_lin_vel': jp.exp(-jp.sum(jp.square(cmd[:2] - linvel[:2])) / rc.tracking_sigma),
            'tracking_ang_vel': jp.exp(-jp.square(cmd[2] - gyro[2]) / rc.tracking_sigma),
            'lin_vel_z': jp.square(linvel[2]),
            'ang_vel_xy': jp.sum(jp.square(gyro[:2])),
            'orientation': jp.sum(jp.square(gravity[:2])),
            'base_height': jp.square(data.qpos[2] - self._config.base_height_target),
            'torques': jp.sum(jp.square(tau)),
            'action_rate': jp.sum(jp.square(action - info['last_act'])),
            'energy': jp.sum(jp.abs(dq * tau)),
            'dof_pos_limits': jp.sum(jp.clip(self._soft_lowers - q, 0.0) + jp.clip(q - self._soft_uppers, 0.0)),
            'pose': jp.sum(jp.square(q - self._default_pose) * weights),
            'feet_phase': jp.exp(-jp.sum(jp.square(foot_z - target_h)) / rc.feet_phase_sigma),
            'contact_match': jp.mean((stance == contact).astype(jp.float32)),
            'feet_slip': jp.sum(jp.sum(jp.square(foot_v[:, :2]), axis=-1) * contact),
            'feet_flat': jp.sum(jp.sum(jp.square(foot_zaxis[:, :2]), axis=-1) * contact),
            'feet_distance': jp.clip(self._config.min_feet_distance - lateral, 0.0, 0.1),
            'alive': jp.array(1.0),
            'termination': done.astype(jp.float32),
            'symmetry': self._symmetry_error(q - self._default_pose, info),
        }

    def sample_command(self, rng, stepping):
        """Base velocity command [vx, vy, yaw rate] in the body frame.

        Zero for stand episodes (they must stand still) and, with probability
        zero_command_probability, for stepping segments (keeps step-in-place).
        """
        c = self._config
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        cmd = jp.hstack([
            jax.random.uniform(k1, minval=c.lin_vel_x[0], maxval=c.lin_vel_x[1]),
            jax.random.uniform(k2, minval=c.lin_vel_y[0], maxval=c.lin_vel_y[1]),
            jax.random.uniform(k3, minval=c.ang_vel_yaw[0], maxval=c.ang_vel_yaw[1]),
        ])
        moving = (stepping > 0.5) & (jax.random.uniform(k4) >= c.zero_command_probability)
        return jp.where(moving, cmd, jp.zeros(3))

    # ------------------------------------------------------------------ accessors
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


# ---------------------------------------------------------------------- randomization
def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Per-world physical parameter randomization (vmapped over rng)."""
    FLOOR = 0
    TORSO = 1

    @jax.vmap
    def rand(rng):
        ks = jax.random.split(rng, 10)
        u = lambda k, lo, hi, shape=(): jax.random.uniform(k, shape, minval=lo, maxval=hi)
        friction = model.geom_friction.at[FLOOR, 0].set(u(ks[0], 0.4, 1.2))
        mass = model.body_mass * u(ks[1], 0.9, 1.1, (model.nbody,))
        mass = mass.at[TORSO].add(u(ks[2], -0.3, 0.8))            # battery / shell uncertainty
        ipos = model.body_ipos.at[TORSO].add(u(ks[3], -0.02, 0.02, (3,)))
        armature = model.dof_armature.at[6:].multiply(u(ks[4], 0.7, 1.3, (NUM_JOINTS,)))
        damping = model.dof_damping.at[6:].multiply(u(ks[5], 0.5, 2.0, (NUM_JOINTS,)))
        frictionloss = model.dof_frictionloss.at[6:].multiply(u(ks[6], 0.5, 2.0, (NUM_JOINTS,)))
        kp = model.actuator_gainprm[:, 0] * u(ks[7], 0.8, 1.2, (model.nu,))
        kd = -model.actuator_biasprm[:, 2] * u(ks[8], 0.7, 1.3, (model.nu,))
        gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        biasprm = model.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)
        forcerange = model.actuator_forcerange * u(ks[9], 0.8, 1.1, (model.nu, 1))
        return friction, mass, ipos, armature, damping, frictionloss, gainprm, biasprm, forcerange

    out = rand(rng)
    fields = ('geom_friction', 'body_mass', 'body_ipos', 'dof_armature', 'dof_damping',
              'dof_frictionloss', 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange')
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({f: 0 for f in fields})
    model = model.tree_replace(dict(zip(fields, out)))
    return model, in_axes
