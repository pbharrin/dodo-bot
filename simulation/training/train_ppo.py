"""Train the Dodo v5 stand / step-in-place policy with Brax PPO on MJX-Warp.

GPU (g6e.xlarge or similar):
    python simulation/training/train_ppo.py --run-name stage1

CPU smoke test (tiny, minutes; checks the pipeline end to end, not learning):
    python simulation/training/train_ppo.py --impl jax --smoke

Outputs go to simulation/results/local/train/<run-name>/:
    config.json        env + PPO configuration actually used
    progress.csv       eval reward and reward terms per evaluation
    checkpoints/       Brax/Orbax checkpoints (resume with --restore)
    params.pkl         final policy params (brax.io.model format)
    eval.json          deterministic rollout statistics (falls, contact match ...)
    rollout*.mp4       rendered rollouts (unless --no-video)
"""
import argparse
import csv
import functools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_backend  # noqa: E402  (must run before mujoco is imported)

CAN_RENDER = render_backend.setup()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

import jax  # noqa: E402
from brax.io import model as brax_model  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from ml_collections import config_dict  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402

import dodo_env  # noqa: E402
import evaluate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def ppo_config() -> config_dict.ConfigDict:
    return config_dict.create(
        num_timesteps=150_000_000,
        num_evals=15,
        reward_scaling=1.0,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=5e-3,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        clipping_epsilon=0.2,
        num_resets_per_eval=1,
        num_eval_envs=128,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 128, 64),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key='state',
            value_obs_key='privileged_state',
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--impl', choices=['warp', 'jax'], default='warp')
    ap.add_argument('--num-timesteps', type=int)
    ap.add_argument('--num-envs', type=int)
    ap.add_argument('--num-evals', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-randomization', action='store_true', help='disable domain randomization')
    ap.add_argument('--restore', type=Path, help='checkpoint directory to resume from')
    ap.add_argument('--env-override', action='append', default=[], metavar='KEY=JSON',
                    help='env config override, e.g. stand_probability=0.5 or lin_vel_x=[0,0.3]')
    ap.add_argument('--num-videos', type=int, default=2)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='tiny CPU-sized run to test the pipeline')
    args = ap.parse_args()
    if not CAN_RENDER and not args.no_video:
        print('Rendering unavailable: skipping videos (render later with evaluate.py).')
        args.no_video = True

    run_name = args.run_name or datetime.now().strftime('dodo_step_%Y%m%d_%H%M%S')
    logdir = REPO / 'simulation' / 'results' / 'local' / 'train' / run_name
    logdir.mkdir(parents=True, exist_ok=True)

    overrides = {'impl': args.impl}
    for item in args.env_override:
        key, val = item.split('=', 1)
        overrides[key] = json.loads(val)

    rl = ppo_config()
    if args.smoke:
        rl.update(num_timesteps=40_000, num_envs=64, batch_size=64, num_minibatches=4,
                  num_evals=2, num_eval_envs=8, episode_length=100, unroll_length=10,
                  num_updates_per_batch=1)
        overrides.setdefault('naconmax', 16 * 64)
    for name in ('num_timesteps', 'num_envs', 'num_evals'):
        if getattr(args, name) is not None:
            rl[name] = getattr(args, name)

    env_cfg = dodo_env.default_config()
    env = dodo_env.DodoStepInPlace(config=env_cfg, config_overrides=overrides)
    eval_env = dodo_env.DodoStepInPlace(config=dodo_env.default_config(), config_overrides=overrides)

    with open(logdir / 'config.json', 'w') as f:
        json.dump({'env': env._config.to_dict(), 'ppo': rl.to_dict(), 'model': str(env.xml_path),
                   'randomization': not args.no_randomization, 'jax_devices': [str(d) for d in jax.devices()]},
                  f, indent=2, default=str)
    print(f'Run dir: {logdir}\nDevices: {jax.devices()}\nimpl={args.impl}  envs={rl.num_envs}  steps={rl.num_timesteps:,}')

    params = rl.to_dict()
    net = params.pop('network_factory')
    train_fn = functools.partial(
        ppo.train,
        **params,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net),
        randomization_fn=None if args.no_randomization else dodo_env.domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=args.seed,
        save_checkpoint_path=logdir / 'checkpoints',
        restore_checkpoint_path=args.restore,
    )

    t0 = time.monotonic()
    times = [t0]
    rows = []

    def progress(step, metrics):
        times.append(time.monotonic())
        row = {'step': step, 'wall_s': round(times[-1] - t0, 1)}
        row.update({k: float(v) for k, v in metrics.items() if k.startswith('eval/')})
        rows.append(row)
        with open(logdir / 'progress.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        # Brax sums per-step metrics over the episode; divide by length for a per-step mean.
        length = row.get('eval/avg_episode_length', float('nan'))
        match = row.get('eval/episode_contact_match_frac', float('nan')) / max(length, 1.0)
        print(f"{step:>12,}  reward={row.get('eval/episode_reward', float('nan')):8.2f}  "
              f"len={length:6.1f}  contact_match={match:.3f}  [{row['wall_s']:.0f}s]", flush=True)

    make_inference_fn, policy_params, _ = train_fn(environment=env, eval_env=eval_env, progress_fn=progress)
    if len(times) > 1:
        print(f'JIT/first eval: {times[1] - times[0]:.0f}s  training: {times[-1] - times[1]:.0f}s')

    brax_model.save_params(str(logdir / 'params.pkl'), policy_params)
    print('Saved', logdir / 'params.pkl')

    stats = evaluate.evaluate(
        eval_env, make_inference_fn(policy_params, deterministic=True),
        episode_length=rl.episode_length, num_episodes=min(rl.num_eval_envs, 64),
        num_videos=0 if args.no_video else args.num_videos, outdir=logdir, seed=args.seed + 1)
    with open(logdir / 'eval.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != 'torque'}, indent=2))


if __name__ == '__main__':
    main()
