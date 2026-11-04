# Archived policies

Milestone policies, copied out of `simulation/results/local/train/<run>/` (which is gitignored) so they are versioned. Only promote a run here when it becomes the reference for later work, such as hardware export or a new warm start. Training runs themselves stay local.

Each directory holds:

| File | Contents |
| --- | --- |
| `params.pkl` | Brax params `(normalizer, policy, value)`: observation normalizer statistics plus actor and critic weights (`brax.io.model` format, a Python pickle: load only files from this repo) |
| `config.json` | Env and PPO config used for training: observation layout, `action_scale`, `action_filter_hz`, gains, network sizes |
| `eval.json` | `evaluate.py` result (64 episodes, MJX) |
| `replay_mujoco.json` | `replay_mujoco.py` result (24 episodes, C MuJoCo, 9 conditions) |
| `progress.csv` | Training curve |

`params.pkl` and `config.json` belong together: the policy is only valid with the env settings it was trained with, especially the action filter and action scale. Videos aren't archived (`*.mp4` is gitignored); regenerate them with `evaluate.py`.

`evaluate.py` and `replay_mujoco.py` accept a policy directory in place of a run directory, but they write their results into it. Running them on an archived policy overwrites its `eval.json` / `replay_mujoco.json`. Copy the directory to `simulation/results/local/train/` first, or use `--env-override`, which writes `*_override.json` instead.

## `stage2_smooth` (current best, 2026-11-02)

Walks forward/back up to 0.3 m/s, sidesteps and turns on command, stands or steps in place, recovers from pushes. A 6 Hz action low-pass filter removes most PD-servo torque spikes. Results: `simulation/training/README.md`, "Stage 2b result".

- **Lineage:** `stage1_v5` → `stage2_walk` → `stage2_sym` → `stage2_smooth`, each warm-started from the previous one; 150M steps each on a g5.xlarge.
- **Training code:** commit `fd377f0` (`--task walk_smooth --warm-start stage2_sym`).
- **Robot model:** `simulation/models/dodo_v5.xml`, unchanged since `a0ceaec`. sha256 `8cc5d72a83dee37843d057ddc1048416f8912d256b876b5b31966308b6a99217`. The `model` path in `config.json` is the g5's absolute path to the same file. Its env config predates the `model` key, so the tools load it as v5. On v6 (knee/ankle motors at the joints) it runs with torques within ±3%: see `training/README.md`, "Model v6". A policy is tied to its model version: if the model is regenerated with new masses or actuator data, retrain and archive a new policy.
- **params.pkl** sha256: `408e8215365ff8f29f07f84f8e65dcb31d8a92be73dbcc35a17092699ce03c1e`.
- **Interface:** 50 Hz; actor input `state` (46 values, layout in the `dodo_env.py` docstring), normalized with the stored statistics; actor MLP 256-128-64; output 10 actions in [−1, 1] → 6 Hz first-order filter (α = 0.53 per step, state starts at zero) → `q* = q_home + 0.5 rad × a_filtered`. "Last action" in the observation is the raw, unfiltered output.
- **Hardware conditions:** end-to-end latency (sensor sample → target active) at or under 20 ms; at 40 ms it fell in 1 of 24 replay episodes.

Load it:

```python
import json
from pathlib import Path
import dodo_env, evaluate   # from simulation/training

d = Path('simulation/policies/stage2_smooth')
cfg = json.loads((d / 'config.json').read_text())
env = dodo_env.DodoStepInPlace(config=evaluate.saved_env_config(cfg['env']),
                               config_overrides={'impl': 'jax'})
policy, _ = evaluate.load_policy(d, env)   # policy(obs_dict, rng) -> (action, extras)
```
