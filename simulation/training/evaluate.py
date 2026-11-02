"""Evaluate (and render) a trained Dodo policy.

    python simulation/training/evaluate.py simulation/results/local/train/<run-name>

Loads params.pkl + config.json from the run directory, runs deterministic
rollouts on the nominal (non-randomized) model with the training noise, pushes
and latency settings, writes eval.json and rollout videos. Add
--env-override push_config.enable=false etc. to probe specific conditions.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_backend  # noqa: E402  (must run before mujoco is imported)

CAN_RENDER = render_backend.setup()

import jax
import jax.numpy as jp
import numpy as np

import dodo_env  # noqa: E402


def evaluate(env, policy, episode_length, num_episodes, num_videos, outdir, seed=0):
    """Batched deterministic rollouts. policy(obs, key) -> (action, extras)."""
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    act = jax.vmap(policy)
    feet = env._feet_site

    @jax.jit
    def rollout(state, key):
        def body(carry, _):
            s, k, alive = carry
            k, ak = jax.random.split(k)
            a, _ = act(s.obs, jax.random.split(ak, num_episodes))
            s = step(s, a)
            alive = alive & (s.done < 0.5)
            stance, _ = jax.vmap(env._gait)(s.info)
            contact = jax.vmap(env._contact)(s.data)
            rec = dict(qpos=s.data.qpos, alive=alive, stepping=s.info['stepping'],
                       match=jp.mean(stance == contact, axis=-1),
                       foot_z=s.data.site_xpos[:, feet, 2],
                       tau=jp.abs(s.data.actuator_force),              # (N, nu)
                       pushed=jp.linalg.norm(s.info['push'], axis=-1) > 0)
            return (s, k, alive), rec
        alive0 = jp.ones(num_episodes, dtype=bool)
        _, traj = jax.lax.scan(body, (state, key, alive0), None, length=episode_length)
        return traj

    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    state = reset(keys)
    start_xy = np.asarray(state.data.qpos[:, :2])
    traj = jax.device_get(rollout(state, jax.random.PRNGKey(seed + 1)))

    alive = traj['alive']                                   # (T, N)
    survived = alive[-1]
    steps_alive = alive.sum(0)
    stepping = traj['stepping'][0] > 0.5
    match = np.where(alive, traj['match'], np.nan)
    foot_z = traj['foot_z']                                 # (T, N, 2)
    last = np.maximum(steps_alive - 1, 0)
    end_xy = traj['qpos'][last, np.arange(num_episodes), :2]
    drift = np.linalg.norm(end_xy - start_xy, axis=-1)

    def grp(mask):
        if not mask.any():
            return None
        return {
            'episodes': int(mask.sum()),
            'survival_rate': float(survived[mask].mean()),
            'mean_time_alive_s': float(steps_alive[mask].mean() * env.dt),
            'contact_match': float(np.nanmean(match[:, mask])),
            'mean_peak_foot_height_m': float(np.nanmean(np.where(alive[..., None], foot_z, np.nan)[:, mask].max(0))),
            'base_drift_m': float(drift[mask].mean()),
            'peak_joint_torque_Nm': float(np.where(alive, traj['tau'].max(-1), 0)[:, mask].max()),
        }

    stats = {'all': grp(np.ones_like(stepping)), 'stepping': grp(stepping), 'standing': grp(~stepping),
             'episode_length_s': episode_length * env.dt}
    stats['torque'] = torque_stats(env, traj, alive, stepping)
    print_torque_table(stats['torque'])

    if num_videos and CAN_RENDER:
        try:
            _render(env, traj, stepping, episode_length, min(num_videos, num_episodes), outdir)
        except Exception as e:  # never lose the stats over a rendering problem
            print(f'WARNING: video rendering failed ({type(e).__name__}: {e}); stats are unaffected.')
    elif num_videos:
        print('Rendering unavailable (MUJOCO_GL=disable): no videos written.')
    return stats


PUSH_RECOVERY_S = 1.0   # steps within this long after a push count as push recovery
SATURATION_FRAC = 0.98  # |torque| >= 98% of the actuator limit counts as saturated


def torque_stats(env, traj, alive, stepping):
    """Per-joint torque use, split into steady motion vs. the second after a push.

    Answers: is the torque limit hit on every step (gait needs more than the
    actuator has) or only while recovering from pushes (acceptable)?
    """
    m = env.mj_model
    names = [m.actuator(i).name for i in range(m.nu)]
    limit = np.abs(m.actuator_forcerange[:, 1])
    tau = traj['tau']                                        # (T, N, nu)
    T = tau.shape[0]
    t = np.arange(T)[:, None]
    last_push = np.maximum.accumulate(np.where(traj['pushed'], t, -10**9), axis=0)
    recovering = (t - last_push) * env.dt < PUSH_RECOVERY_S  # (T, N)

    def joint_table(mask):                                   # mask (T, N) of steps to include
        if not mask.any():
            return None
        x = tau[mask]                                        # (K, nu)
        sat = x >= SATURATION_FRAC * limit
        return {
            'control_steps': int(mask.sum()),
            'joints': {n: {
                'saturated_frac': round(float(sat[:, j].mean()), 4),
                'p50_Nm': round(float(np.percentile(x[:, j], 50)), 2),
                'p95_Nm': round(float(np.percentile(x[:, j], 95)), 2),
                'p99_Nm': round(float(np.percentile(x[:, j], 99)), 2),
                'max_Nm': round(float(x[:, j].max()), 2),
                'rms_Nm': round(float(np.sqrt(np.mean(x[:, j] ** 2))), 2),
            } for j, n in enumerate(names)},
        }

    out = {'limit_Nm': limit.tolist(), 'saturation_threshold': SATURATION_FRAC,
           'push_recovery_window_s': PUSH_RECOVERY_S}
    for mode, ep in (('stepping', stepping), ('standing', ~stepping)):
        base = alive & ep[None, :]
        out[mode] = {'steady': joint_table(base & ~recovering),
                     'push_recovery': joint_table(base & recovering)}
    return out


def print_torque_table(ts):
    for mode in ('stepping', 'standing'):
        for phase in ('steady', 'push_recovery'):
            tab = ts[mode][phase]
            if tab is None:
                continue
            print(f'\nTorque, {mode} / {phase} ({tab["control_steps"]} control steps, '
                  f'saturated = |tau| >= {ts["saturation_threshold"]:.0%} of limit)')
            print(f'  {"joint":<20}{"sat %":>7}{"p50":>7}{"p95":>7}{"p99":>7}{"max":>7}{"rms":>7}')
            for n, r in tab['joints'].items():
                print(f'  {n:<20}{100 * r["saturated_frac"]:>7.2f}{r["p50_Nm"]:>7.2f}{r["p95_Nm"]:>7.2f}'
                      f'{r["p99_Nm"]:>7.2f}{r["max_Nm"]:>7.2f}{r["rms_Nm"]:>7.2f}')


def _render(env, traj, stepping, episode_length, num_videos, outdir):
    import mediapy
    import mujoco
    m = env.mj_model
    d = mujoco.MjData(m)
    with mujoco.Renderer(m, height=480, width=640) as r:
        for i in range(num_videos):
            frames = []
            for t in range(0, episode_length, 2):
                d.qpos[:] = traj['qpos'][t, i]
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera='track')
                frames.append(r.render())
            mode = 'step' if stepping[i] else 'stand'
            path = Path(outdir) / f'rollout{i}_{mode}.mp4'
            mediapy.write_video(str(path), frames, fps=1.0 / env.dt / 2)
            print('Wrote', path)


def load_policy(run_dir: Path, env):
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    cfg = json.loads((run_dir / 'config.json').read_text())
    net = cfg['ppo']['network_factory']
    for k in ('policy_hidden_layer_sizes', 'value_hidden_layer_sizes'):
        net[k] = tuple(net[k])
    networks = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **net)
    params = brax_model.load_params(str(run_dir / 'params.pkl'))
    return ppo_networks.make_inference_fn(networks)(params, deterministic=True), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--impl', choices=['warp', 'jax'], default=None, help='default: as trained')
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--videos', type=int, default=2)
    ap.add_argument('--env-override', action='append', default=[], metavar='KEY=JSON')
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / 'config.json').read_text())
    overrides = {'impl': args.impl or cfg['env']['impl']}
    for item in args.env_override:
        k, v = item.split('=', 1)
        overrides[k] = json.loads(v)
    env = dodo_env.DodoStepInPlace(config_overrides=overrides)
    policy, _ = load_policy(args.run_dir, env)
    stats = evaluate(env, policy, cfg['ppo']['episode_length'], args.episodes, args.videos, args.run_dir)
    out = args.run_dir / ('eval.json' if not args.env_override else 'eval_override.json')
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != 'torque'}, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
