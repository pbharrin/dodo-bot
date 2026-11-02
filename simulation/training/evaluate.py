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

os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
                       tau=jp.max(jp.abs(s.data.actuator_force), axis=-1))
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
            'peak_joint_torque_Nm': float(np.where(alive, traj['tau'], 0)[:, mask].max()),
        }

    stats = {'all': grp(np.ones_like(stepping)), 'stepping': grp(stepping), 'standing': grp(~stepping),
             'episode_length_s': episode_length * env.dt}

    if num_videos:
        import mediapy
        import mujoco
        m = env.mj_model
        d = mujoco.MjData(m)
        with mujoco.Renderer(m, height=480, width=640) as r:
            for i in range(min(num_videos, num_episodes)):
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
    return stats


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
    (args.run_dir / 'eval.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
