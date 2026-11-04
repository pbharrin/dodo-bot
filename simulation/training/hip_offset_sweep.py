"""Hip packaging sensitivity: how hip axis offsets change torque and walking.

    python simulation/training/hip_offset_sweep.py            # all variants, ~5 min on a Mac
    python simulation/training/hip_offset_sweep.py --variants baseline roll_axis_out_20

The v6 generator places the hip yaw, roll and pitch axes at one point. Real
bearings, drums and supports need offsets between them. This script builds v6
variants with offsets (build_model_v6.make_model_with), runs a FIXED trained
policy on each in C MuJoCo (replay_mujoco.Replay, same episodes for every
variant) and reports hip and leg torque, survival and tracking.

The policy is not retrained per variant, so the numbers show the cost of an
offset to a policy that doesn't know about it. A retrained policy could
compensate for kinematic effects (foot placement, posture) but not for physics:
the single-support hip-roll moment is set by where the roll axis sits relative
to the centre of mass.

Offsets are in mm: +x forward, +y outward from the midline, +z up; "roll" is
the roll axis relative to the yaw axis, "pitch" the pitch centre relative to
the roll axis. The hip pitch centre stays at 330 mm, so leg length and stance
height don't change.

Writes simulation/results/local/hip_sweep/hip_offset_sweep.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_backend  # noqa: E402  (must run before mujoco is imported)

render_backend.setup()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import dodo_env  # noqa: E402
import evaluate  # noqa: E402
import replay_mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'simulation' / 'models'))
import build_model_v6  # noqa: E402

# name: (roll offset from yaw, pitch offset from roll), mm (x fwd, y out, z up)
VARIANTS = {
    'baseline':             ((0, 0, 0),   (0, 0, 0)),
    # Roll axis moved sideways, pitch centre moved back: feet stay where they are.
    'roll_axis_in_20':      ((0, -20, 0), (0, 20, 0)),
    'roll_axis_in_10':      ((0, -10, 0), (0, 10, 0)),
    'roll_axis_out_10':     ((0, 10, 0),  (0, -10, 0)),
    'roll_axis_out_20':     ((0, 20, 0),  (0, -20, 0)),
    'roll_axis_out_30':     ((0, 30, 0),  (0, -30, 0)),
    # Pitch centre (whole leg) moved outward from the roll axis: wider stance.
    'pitch_out_20':         ((0, 0, 0),   (0, 20, 0)),
    'pitch_out_40':         ((0, 0, 0),   (0, 40, 0)),
    # Vertical stacking: roll axis below yaw; pitch centre below roll axis.
    'roll_below_yaw_20':    ((0, 0, -20), (0, 0, 0)),
    'roll_below_yaw_40':    ((0, 0, -40), (0, 0, 0)),
    'pitch_below_roll_20':  ((0, 0, 0),   (0, 0, -20)),
    'pitch_below_roll_40':  ((0, 0, 0),   (0, 0, -40)),
    # Fore/aft: pitch centre (whole leg) moved relative to the torso.
    'pitch_fwd_20':         ((0, 0, 0),   (20, 0, 0)),
    'pitch_back_20':        ((0, 0, 0),   (-20, 0, 0)),
    # A plausible packaged hip: roll 30 mm below yaw, pitch 25 mm below roll and
    # 15 mm outboard, roll axis 10 mm outboard of the yaw axis.
    'packaged_example':     ((0, 10, -30), (0, 15, -25)),
}
CONDITIONS = ('full_solver', 'realistic')
GROUPS = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')
SHORT = {'hip_yaw': 'yaw p99', 'hip_roll': 'roll p99', 'hip_pitch': 'pitch p99',
         'knee_pitch': 'knee p99', 'ankle_pitch': 'ankle p99'}


def build_xml(name, out_dir: Path) -> Path:
    roll, pitch = VARIANTS[name]
    xml = build_model_v6.make_model_with(
        HIP_ROLL_OFFSET=tuple(v / 1000 for v in roll),
        HIP_PITCH_OFFSET=tuple(v / 1000 for v in pitch))
    path = out_dir / f'dodo_v6_{name}.xml'
    path.write_text(xml)
    return path


def static_hip_roll_moment(m: mujoco.MjModel) -> float:
    """Total weight x lateral distance from the left roll axis to the CoM, home pose
    (the generator's single-support estimate, computed on the actual model)."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    roll_axis_y = d.xanchor[m.joint('left_hip_roll').id][1]
    return float(m.body_subtreemass[1] * 9.81 * abs(roll_axis_y - d.subtree_com[1][1]))


def joint_stats(results):
    """Per joint group, max over left/right of peak-torque p95/p99 (alive steps)."""
    tau = np.concatenate([r['tau'][r['alive'].astype(bool)] for _, r in results])   # (steps, 10)
    out = {}
    for i, g in enumerate(GROUPS):
        lr = tau[:, [i, i + 5]]
        out[g] = {'p95_Nm': float(np.percentile(lr, 95, axis=0).max()),
                  'p99_Nm': float(np.percentile(lr, 99, axis=0).max())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', type=Path, default=REPO / 'simulation' / 'policies' / 'stage2_smooth')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(replay_mujoco.CONDITIONS))
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--out', type=Path, default=REPO / 'simulation' / 'results' / 'local' / 'hip_sweep')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((args.policy / 'config.json').read_text())
    env_cfg = evaluate.saved_env_config(cfg['env'])
    env_cfg.model = 'v6'
    base_env = dodo_env.DodoStepInPlace(config=env_cfg, config_overrides={'impl': 'jax'})
    policy, _ = evaluate.load_policy(args.policy, base_env)

    summary = {}
    print(f'{"variant":<22}{"cond":<12}{"M_roll":>7}{"surv":>6}{"match":>7}{"v err":>7}{"w err":>7}'
          + ''.join(f'{SHORT[g]:>9}' for g in GROUPS)
          + f'{"roll95":>8}{"sat %":>7}')
    for name in args.variants:
        xml = build_xml(name, args.out)
        env = dodo_env.DodoStepInPlace(config=env_cfg, config_overrides={'impl': 'jax'}, xml_path=xml)
        moment = static_hip_roll_moment(env.mj_model)
        replay = replay_mujoco.Replay(env, policy)
        episodes = [replay.new_episode(i) for i in range(args.episodes)]
        summary[name] = {'roll_offset_mm': VARIANTS[name][0], 'pitch_offset_mm': VARIANTS[name][1],
                         'static_hip_roll_moment_Nm': moment, 'conditions': {}}
        for cname in args.conditions:
            c = replay_mujoco.CONDITIONS[cname]
            t0 = time.monotonic()
            results = [(ep, replay.run(ep, c)[0]) for ep in episodes]
            s = replay_mujoco.summarize(results, c[4], env.dt)
            s['joints'] = joint_stats(results)
            summary[name]['conditions'][cname] = s
            fmt = lambda x: f'{x:7.3f}' if x is not None else '      -'
            print(f'{name:<22}{cname:<12}{moment:>7.2f}{s["survival_rate"]:>6.0%}{fmt(s["stepping_contact_match"])}'
                  f'{fmt(s["moving_lin_vel_error_mps"])}{fmt(s["moving_yaw_rate_error_radps"])}'
                  + ''.join(f'{s["joints"][g]["p99_Nm"]:>9.2f}' for g in GROUPS)
                  + f'{s["joints"]["hip_roll"]["p95_Nm"]:>8.2f}{100 * s["saturated_frac"]:>7.2f}'
                  f'   ({time.monotonic() - t0:.0f} s)', flush=True)
    out = args.out / 'hip_offset_sweep.json'
    out.write_text(json.dumps({'policy': str(args.policy), 'episodes': args.episodes,
                               'columns': 'joint values = peak-over-substeps torque p99, max of L/R',
                               'results': summary}, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
