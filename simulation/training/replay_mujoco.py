"""Replay a trained policy in plain C MuJoCo under harsher-than-training conditions.

    python simulation/training/replay_mujoco.py simulation/results/local/train/<run>

Training and evaluate.py use MJX with a 2-iteration contact solver, a 4 ms
timestep, at most one control step (20 ms) of delay and a 10 Nm torque limit.
This script runs the same policy with mujoco.MjData / mj_step (no MJX) and a
suite of conditions: the full default solver, a smaller timestep, explicit
sensor-to-actuator latency in milliseconds and lower torque limits. A policy that
only works with MJX's shortcuts, or that needs the 10 Nm spikes, shows up here
before it gets near hardware.

The observation is rebuilt here in NumPy exactly as dodo_env.DodoStepInPlace
builds it (checked by test_dodo_env.ReplayTest). Every condition runs the same
episodes (same seeds: start state, gait, commands, pushes, noise), so the rows
can be compared directly. Episodes use the nominal model (no domain
randomization), the run's own task config (commands, push settings, noise), and
no random one-step delay: latency is set explicitly per condition.

Writes <run>/replay_mujoco.json and prints a summary table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_backend  # noqa: E402  (must run before mujoco is imported)

CAN_RENDER = render_backend.setup()

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import dodo_env  # noqa: E402
import evaluate  # noqa: E402

# name: (solver iterations, ls iterations, sim dt, latency ms, torque limit Nm)
CONDITIONS = {
    'as_trained':         (2,   8,  0.004,  0, 10.0),   # same physics as MJX training
    'full_solver':        (100, 50, 0.004,  0, 10.0),   # MuJoCo default solver effort
    'full_solver_dt2ms':  (100, 50, 0.002,  0, 10.0),   # and half the timestep
    'latency_10ms':       (100, 50, 0.004, 10, 10.0),
    'latency_20ms':       (100, 50, 0.004, 20, 10.0),   # one full control step, every step
    'latency_40ms':       (100, 50, 0.004, 40, 10.0),   # stress
    'torque_8Nm':         (100, 50, 0.004,  0,  8.0),
    'torque_6Nm':         (100, 50, 0.004,  0,  6.0),   # stress: about the p99 demand
    'realistic':          (100, 50, 0.002, 10,  8.0),   # combined, plausible hardware
}


class Replay:
    """One C MuJoCo robot driven by the policy, mirroring DodoStepInPlace."""

    def __init__(self, env: dodo_env.DodoStepInPlace, policy):
        self.env = env
        self.cfg = env._config
        self.policy = jax.jit(policy)
        self.base_model = env.mj_model
        m = self.base_model
        self.default_pose = np.asarray(env._default_pose)
        self.init_q = np.asarray(env._init_q)
        self.imu = m.site('imu').id
        self.feet_site = [m.site(f).id for f in dodo_env.FEET]
        self.contact_adr = [m.sensor_adr[m.sensor(f'{f}_floor_found').id] for f in dodo_env.FEET]
        self.gyro_adr = m.sensor_adr[m.sensor('gyro').id]
        self.linvel_adr = m.sensor_adr[m.sensor('local_linvel').id]
        self.priv_size = env.observation_size['privileged_state'][0]

    # ---------------------------------------------------------------- obs
    def obs(self, d, ep, noise_rng):
        sc = self.cfg.noise_config.scales
        lvl = self.cfg.noise_config.level

        def noisy(x, scale):
            return x + (2 * noise_rng.random(x.shape) - 1) * lvl * scale

        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        q = d.qpos[7:] - self.default_pose
        dq = d.qvel[6:]
        clock = 2 * np.pi * ep['phase']
        clock = np.concatenate([np.cos(clock), np.sin(clock)]) if ep['stepping'] else np.zeros(4)
        task = np.array([ep['stepping'], ep['gait_freq'], ep['swing_height'] * 10.0])
        state = np.concatenate([
            noisy(gyro, sc.gyro), noisy(gravity, sc.gravity), ep['command'],
            noisy(q + ep['encoder_offset'], sc.joint_pos), noisy(dq, sc.joint_vel) * 0.1,
            ep['last_act'], clock, task])
        return state

    def act(self, state):
        obs = {'state': jp.asarray(state, dtype=jp.float32),
               'privileged_state': jp.zeros(self.priv_size)}   # unused by the actor
        a, _ = self.policy(obs, jax.random.PRNGKey(0))
        return np.asarray(a)

    # ---------------------------------------------------------------- episode
    def new_episode(self, seed):
        """Episode randomness drawn exactly once per seed, shared by all conditions."""
        cfg = self.cfg
        r = np.random.default_rng(seed)
        steps = cfg.episode_length
        yaw = r.uniform(-np.pi, np.pi)
        stepping = float(r.random() > cfg.stand_probability)
        ep = {
            'yaw': yaw,
            'joint_noise': r.uniform(-0.05, 0.05, dodo_env.NUM_JOINTS),
            'base_vel': r.uniform(-0.1, 0.1, 6),
            'stepping': stepping,
            'gait_freq': r.uniform(*cfg.gait_frequency),
            'swing_height': r.uniform(*cfg.swing_height),
            'phase0': r.random(),
            'encoder_offset': r.uniform(-cfg.encoder_offset, cfg.encoder_offset, dodo_env.NUM_JOINTS),
            'noise_seed': int(r.integers(2**31)),
        }
        # Command schedule (resampled every command_resample_steps).
        cmds = []
        for _ in range(int(np.ceil(steps / cfg.command_resample_steps))):
            c = np.array([r.uniform(*cfg.lin_vel_x), r.uniform(*cfg.lin_vel_y), r.uniform(*cfg.ang_vel_yaw)])
            moving = stepping and r.random() >= cfg.zero_command_probability
            cmds.append(c if moving else np.zeros(3))
        ep['commands'] = cmds
        # Push schedule.
        interval = int(round(r.uniform(*cfg.push_config.interval_range) / self.env.dt))
        pushes = np.zeros((steps, 2))
        if cfg.push_config.enable:
            for t in range(steps):
                theta, mag = r.uniform(0, 2 * np.pi), r.uniform(*cfg.push_config.magnitude_range)
                if (t + 1) % interval == 0:
                    pushes[t] = mag * np.array([np.cos(theta), np.sin(theta)])
        ep['pushes'] = pushes
        return ep

    def run(self, ep, condition, record_frames=False):
        iters, ls_iters, sim_dt, latency_ms, torque = condition
        m = mujoco.MjModel.from_xml_path(self.env.xml_path)
        m.opt.timestep = sim_dt
        m.opt.iterations = iters
        m.opt.ls_iterations = ls_iters
        m.actuator_forcerange[:] = np.array([-torque, torque])
        d = mujoco.MjData(m)
        cfg = self.cfg
        ctrl_dt = self.env.dt
        n_sub = int(round(ctrl_dt / sim_dt))
        lat_sub = int(round(latency_ms / 1000.0 / sim_dt))

        qpos = self.init_q.copy()
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, qpos[3:7], np.array([np.cos(ep['yaw'] / 2), 0, 0, np.sin(ep['yaw'] / 2)]))
        qpos[3:7] = quat
        qpos[7:] += ep['joint_noise']
        qpos[2] += 0.01
        d.qpos[:] = qpos
        d.qvel[:6] = ep['base_vel']
        d.ctrl[:] = qpos[7:]
        mujoco.mj_forward(m, d)

        noise_rng = np.random.default_rng(ep['noise_seed'])
        st = dict(phase=np.mod(np.array([ep['phase0'], ep['phase0'] + 0.5]), 1.0),
                  last_act=np.zeros(dodo_env.NUM_JOINTS), stepping=ep['stepping'],
                  gait_freq=ep['gait_freq'], swing_height=ep['swing_height'],
                  encoder_offset=ep['encoder_offset'], command=ep['commands'][0])
        state = self.obs(d, st, noise_rng)

        pending = []                       # (substep index when it takes effect, target)
        sub = 0
        rec = {k: [] for k in ('alive', 'match', 'foot_z', 'tau', 'lin_err', 'yaw_err', 'moving')}
        frames = []
        renderer = mujoco.Renderer(m, 480, 640) if record_frames else None
        alive = True
        for t in range(cfg.episode_length):
            a = self.act(state)
            d.qvel[:2] += ep['pushes'][t]
            target = self.default_pose + a * cfg.action_scale
            pending.append((sub + lat_sub, target))
            peak_tau = np.zeros(m.nu)
            for _ in range(n_sub):
                while pending and pending[0][0] <= sub:
                    d.ctrl[:] = pending.pop(0)[1]
                mujoco.mj_step(m, d)
                peak_tau = np.maximum(peak_tau, np.abs(d.actuator_force))
                sub += 1
            st['phase'] = np.mod(st['phase'] + st['gait_freq'] * ctrl_dt, 1.0)
            st['last_act'] = a

            up = d.site_xmat[self.imu][8]
            fallen = up < 0.7 or d.qpos[2] < 0.25 or not np.all(np.isfinite(d.qpos))
            alive = alive and not fallen
            # Stance per the gait clock (as dodo_env._gait).
            swing = st['phase'] >= cfg.duty_factor
            stance = ~swing if st['stepping'] else np.ones(2, bool)
            contact = np.array([d.sensordata[i] > 0 for i in self.contact_adr])
            v = d.sensordata[self.linvel_adr:self.linvel_adr + 3]
            w = d.sensordata[self.gyro_adr + 2]
            cmd = st['command']
            rec['alive'].append(alive)
            rec['match'].append(np.mean(stance == contact))
            rec['foot_z'].append(d.site_xpos[self.feet_site, 2].copy())
            rec['tau'].append(peak_tau)
            rec['lin_err'].append(np.linalg.norm(cmd[:2] - v[:2]))
            rec['yaw_err'].append(abs(cmd[2] - w))
            rec['moving'].append(bool(np.any(cmd != 0)))
            if renderer is not None and t % 2 == 0:
                renderer.update_scene(d, camera='track')
                frames.append(renderer.render())
            if not alive:
                break
            # As in dodo_env.step: the command is resampled before the next observation.
            st['command'] = ep['commands'][min((t + 1) // cfg.command_resample_steps, len(ep['commands']) - 1)]
            state = self.obs(d, st, noise_rng)
        if renderer is not None:
            renderer.close()
        return {k: np.array(v) for k, v in rec.items()}, frames


def summarize(results, limit, dt):
    """results: list of (episode dict, record dict) for one condition."""
    surv = [bool(r['alive'][-1]) and len(r['alive']) == len(e['pushes']) for e, r in results]
    t_alive = [r['alive'].sum() * dt for _, r in results]
    stepping = np.array([e['stepping'] > 0.5 for e, _ in results])

    def cat(key, eps_mask, step_mask=None):
        xs = []
        for (e, r), keep in zip(results, eps_mask):
            if not keep:
                continue
            ok = r['alive'].astype(bool)
            if step_mask is not None:
                ok = ok & step_mask(r)
            xs.append(r[key][ok])
        return np.concatenate(xs) if xs else np.zeros((0,))

    tau = cat('tau', np.ones(len(results), bool))
    out = {
        'episodes': len(results),
        'survival_rate': float(np.mean(surv)),
        'falls': int(len(surv) - sum(surv)),
        'mean_time_alive_s': float(np.mean(t_alive)),
        'stepping_contact_match': float(cat('match', stepping).mean()) if stepping.any() else None,
        'moving_lin_vel_error_mps': None, 'moving_yaw_rate_error_radps': None,
        'peak_torque_Nm': float(tau.max()) if len(tau) else None,
        'torque_p99_Nm': float(np.percentile(tau.max(-1), 99)) if len(tau) else None,
        'saturated_frac': float(np.mean(tau >= 0.98 * limit)) if len(tau) else None,
    }
    lin = cat('lin_err', stepping, lambda r: r['moving'])
    if len(lin):
        out['moving_lin_vel_error_mps'] = float(lin.mean())
        out['moving_yaw_rate_error_radps'] = float(cat('yaw_err', stepping, lambda r: r['moving']).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--video', action='store_true', help='render the first stepping episode of each condition')
    ap.add_argument('--env-override', action='append', default=[], metavar='KEY=JSON')
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / 'config.json').read_text())
    env_cfg = evaluate.saved_env_config(cfg['env'])
    overrides = {'impl': 'jax'}
    for item in args.env_override:
        k, v = item.split('=', 1)
        overrides[k] = json.loads(v)
    env = dodo_env.DodoStepInPlace(config=env_cfg, config_overrides=overrides)
    policy, _ = evaluate.load_policy(args.run_dir, env)
    replay = Replay(env, policy)

    episodes = [replay.new_episode(args.seed * 100_000 + i) for i in range(args.episodes)]
    first_step = next((i for i, e in enumerate(episodes) if e['stepping']), 0)
    summary = {}
    print(f'{"condition":<20}{"iters":>6}{"dt ms":>6}{"lat ms":>7}{"Nm":>5}{"surv":>7}{"falls":>6}'
          f'{"match":>7}{"v err":>7}{"w err":>7}{"peak":>6}{"p99":>6}{"sat %":>7}{"wall s":>7}')
    for name in args.conditions:
        c = CONDITIONS[name]
        t0 = time.monotonic()
        results = []
        for i, ep in enumerate(episodes):
            rec, frames = replay.run(ep, c, record_frames=args.video and CAN_RENDER and i == first_step)
            results.append((ep, rec))
            if frames:
                import mediapy
                path = args.run_dir / f'replay_{name}.mp4'
                mediapy.write_video(str(path), frames, fps=1.0 / env.dt / 2)
        s = summarize(results, c[4], env.dt)
        s['condition'] = dict(zip(('solver_iterations', 'ls_iterations', 'sim_dt', 'latency_ms', 'torque_limit_Nm'), c))
        summary[name] = s
        fmt = lambda x, f: f'{x:{f}}' if x is not None else '-'.rjust(int(f.split('.')[0]))
        print(f'{name:<20}{c[0]:>6}{c[2] * 1000:>6.0f}{c[3]:>7}{c[4]:>5.0f}{s["survival_rate"]:>7.0%}{s["falls"]:>6}'
              f'{fmt(s["stepping_contact_match"], "7.2f")}{fmt(s["moving_lin_vel_error_mps"], "7.3f")}'
              f'{fmt(s["moving_yaw_rate_error_radps"], "7.3f")}{s["peak_torque_Nm"]:>6.1f}{s["torque_p99_Nm"]:>6.1f}'
              f'{100 * s["saturated_frac"]:>7.2f}{time.monotonic() - t0:>7.0f}', flush=True)
    out = args.run_dir / 'replay_mujoco.json'
    out.write_text(json.dumps({'episodes': args.episodes, 'seed': args.seed, 'results': summary}, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
