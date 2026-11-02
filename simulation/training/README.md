# Learned stand / step-in-place policy (stage 1)

Trains a 50 Hz joint-position policy for the [v5 model](../models/build_model_v5.py) with MuJoCo Playground, Brax PPO and MJX-Warp. Each episode is either **stand** (25%) or **step in place** following a two-foot gait clock (75%). The base-velocity command is zero, so a good policy marches on the spot without drifting and recovers from pushes.

Stage 1 is meant to replace the quasi-static two-foot-to-one-foot work. Stepping in place includes weight transfer and single support, but done dynamically.

| File | Purpose |
| --- | --- |
| `dodo_env.py` | `DodoStepInPlace` environment, rewards, observations, domain randomization |
| `train_ppo.py` | PPO training, checkpoints, final evaluation and videos |
| `evaluate.py` | Reload a run, run deterministic rollouts, write `eval.json` and videos |
| `render_backend.py` | Picks EGL for headless video, or falls back to no rendering if the GL libraries are missing |
| `warm_start.py` | Starts stage 2 from the stage-1 policy, adapting the command inputs |
| `test_dodo_env.py` | Pre-flight checks: model mass/geometry, MJX vs MuJoCo, passive standing, gait clock, randomization |

## Status

- **CPU-validated (JAX backend, Apple M2):** all tests pass, the full train → save → evaluate → render pipeline runs, and the zero-action robot stands for 4 s. See "Validation" below for a short CPU learning-signal run.
- **GPU-validated (MJX-Warp, g5.xlarge A10G, 2026-11-02):** all 6 tests (the 7th, a config check, was added afterwards) pass with `DODO_TEST_IMPL=warp`. A 10M-step trial ran at about 35k steps/s. Eval: 100% survival over 20 s with pushes in all 64 episodes, but stepping contact match 0.61 (the never-lift baseline is 0.60) and peak foot height about 1 cm. It learned to stand, not to step.
- **Stage 1 trained (2026-11-02):** 150M steps on the g5.xlarge in 41 min. The policy stands and steps in place with 4 cm foot clearance and survives every evaluation episode with pushes. See "Stage 1 result" below. Not yet checked: C MuJoCo replay, rope compliance, measured actuator limits.
- **Stage 2 trained (2026-11-02):** `stage2_walk` walks, turns and sidesteps on command with 100% survival with pushes, but with a favoured-leg (asymmetric) gait. A symmetry reward has been added for a fine-tuning run (`stage2_sym`), which hasn't been trained yet. See "Stage 2: walking" below.

## Stage 1 result (2026-11-02)

Run `stage1_v5`: 150M env steps, 8192 envs, MJX-Warp on a g5.xlarge (A10G), domain randomization on, default config. Training took 2434 s (about 62k steps/s) plus 46 s of JIT. The run directory isn't in git (`simulation/results/local/` is ignored).

**Learning curve.** Episodes reached the full 20 s by 10M steps. Contact match rose through about 45M steps and then plateaued at about 0.86. Reward plateaued at about 117. More steps at this config are unlikely to help.

| Steps | Eval reward | Episode length | Contact match (all episodes) |
| ---: | ---: | ---: | ---: |
| 0 | -1.0 | 45 | 0.52 |
| 10.8M | 99.8 | 1000 | 0.74 |
| 21.6M | 111.5 | 1000 | 0.82 |
| 43.3M | 115.5 | 1000 | 0.85 |
| 86.5M | 117.0 | 1000 | 0.87 |
| 151.4M | 116.4 | 993 | 0.87 |

**Evaluation.** 64 deterministic 20 s episodes on the nominal model with sensor noise, pushes and random action delay. The figures are from `evaluate.py --impl jax` on a Mac; the GPU evaluation at the end of training gave the same picture (stepping contact match 0.84, foot height 4.2 cm, peak torque 10.0 Nm).

| | Stepping (47 episodes) | Standing (17 episodes) |
| --- | ---: | ---: |
| Survival over 20 s | 100% | 100% |
| Contact match (never-lift baseline 0.60) | 0.84 | 1.00 |
| Mean peak foot height (target 3–5 cm) | 4.2 cm | 0.9 cm |
| Base drift over 20 s | 20 cm | 1 cm |
| Peak joint torque | 9.3 Nm | 4.7 Nm |

**Joint torque while stepping** (N m, steady stepping, 38.8k control steps outside the 1 s after a push):

| Joint | Left p95 | Left p99 | Left max | Left RMS | Right p95 | Right p99 | Right max | Right RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hip yaw | 1.5 | 1.9 | 2.8 | 0.7 | 1.0 | 1.4 | 2.4 | 0.5 |
| hip roll | 4.8 | 5.7 | 7.0 | 2.4 | 4.5 | 5.1 | 8.6 | 2.5 |
| hip pitch | 1.4 | 2.3 | 3.7 | 0.7 | 1.5 | 2.0 | 3.6 | 0.8 |
| knee pitch | 4.8 | 5.9 | 8.0 | 2.5 | 3.2 | 4.1 | 5.8 | 1.8 |
| ankle pitch | 3.8 | 6.4 | 7.8 | 1.7 | 2.0 | 2.8 | 4.3 | 0.9 |

Findings:

- **No joint ever hits the 10 Nm limit** (0% of control steps saturated, stepping or standing).
- **The torque comes from stepping itself, not from pushes.** The push-recovery table is nearly identical to the one above; it adds only the single highest peak (left ankle, 9.3 Nm). Single support needs it: hip roll, knee and ankle reach p95 of 4–5 Nm and p99 of up to 6.4 Nm on every gait cycle, with short peaks of 7–9 Nm. That matches the 6.7 Nm static hip-roll moment of the 6.84 kg model.
- **RMS is at most 2.5 Nm** (hip roll and knee). This is the number to compare with the measured continuous rating; peaks must fit the measured short-term rating (about 9 Nm ≈ 14 A at the plan's 0.625 Nm/A estimate).
- **The gait is asymmetric:** the left knee and ankle carry noticeably more torque than the right (p95 4.8 vs 3.2 Nm and 3.8 vs 2.0 Nm). No symmetry term is in the reward. Worth watching in videos and possibly penalizing.
- **Standing is cheap:** p95 below 2 Nm on every joint.
- **Drift of 20 cm in 20 s** while stepping: slow wandering rather than stepping on the spot. A candidate for a stronger base-velocity penalty in a later run.

Hardware implication for milestone 2: an actuator module that holds about 2.5 Nm continuous and about 9–10 Nm for short bursts covers this policy, with the masses currently estimated in the model. If the bench shows less, retrain with the measured limit; the policy will trade gait aggressiveness and push recovery for it.

## Stage 2: walking

The robot follows a base-velocity command: forward/back speed, sideways speed and yaw rate in the body frame, resampled every 10 s. `--task walk` applies `dodo_env.WALK_OVERRIDES`:

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| `lin_vel_x` (m/s) | 0 | −0.1 to 0.3 |
| `lin_vel_y` (m/s) | 0 | −0.1 to 0.1 |
| `ang_vel_yaw` (rad/s) | 0 | −0.5 to 0.5 |
| `zero_command_probability` (stepping segments told to step in place) | — | 0.2 |
| `stand_probability` | 0.25 | 0.15 |

Stand episodes always get a zero command. The speed ranges are ESTIMATES: 0.3 m/s is a slow walk for a 0.33 m leg (Froude number about 0.03). Gait clock, randomization and pushes are unchanged from stage 1. Since `stage2_walk` was trained, `WALK_OVERRIDES` also includes the symmetry reward (`reward_config.scales.symmetry=-4.0`; see "Left/right symmetry" below).

Run on the GPU machine, starting from the stage-1 policy:

```bash
python simulation/training/train_ppo.py --run-name stage2_walk --task walk \
    --warm-start simulation/results/local/train/stage1_v5 2>&1 | tee stage2.log
```

**Why `--warm-start` and not `--restore`:** stage 1 only ever saw a zero command, so its observation normalizer stores a std of 1e-6 for the command inputs. Restoring it unchanged would feed a 0.3 m/s command to the network as about 300,000. Because the normalizer averages over all 150M past steps, it would take a long time to correct. `warm_start.py` sets the command normalization to the stage-2 command distribution and zeroes the first-layer weights on the command inputs. Those weights never got any gradient in stage 1. The warm-started policy therefore starts out acting exactly like the stage-1 policy, whatever the command, and learns to use the command from there. (Checked: identical actions for zero and non-zero commands; the first eval of a smoke run already has stepping contact match 0.82, against 0.52 from scratch.) Only inputs that never varied in the source run are adapted. Warm-starting from a run that already saw commands, such as `stage2_walk`, keeps the policy exactly as trained (the script prints `adapted command inputs [False, False, False]`). Use `--restore` only to resume a run of the same task with the same rewards.

What to look for in `eval.json` (now also reported by `evaluate.py` for any run):

- `velocity_tracking.moving`: `lin_vel_error_mps` and `yaw_rate_error_radps`, plus commanded vs. actual mean speed. A policy that ignores the command shows actual speed near 0 and an error close to the commanded speed.
- `velocity_tracking.zero_command`: whether it still steps in place without wandering.
- Survival, contact match and the torque tables as in stage 1. Faster walking may raise knee and hip torque.

If it learns to stand or shuffle instead of walking, the usual fixes are a larger `tracking_lin_vel` scale or a narrower speed range to start, e.g. `--env-override lin_vel_x=[0,0.15]`.

### Stage 2 result: `stage2_walk` (2026-11-02)

150M steps, warm-started from `stage1_v5`, MJX-Warp on the g5.xlarge (about 40 min), no symmetry reward. GPU evaluation at the end of training: 64 episodes of 20 s with noise, pushes and action delay.

| | Stepping (52 episodes) | Standing (12 episodes) |
| --- | ---: | ---: |
| Survival over 20 s | 100% | 100% |
| Commanded / actual mean speed while moving | 0.146 / 0.132 m/s | — |
| Speed error while moving (ignoring the command would be about 0.15) | 0.067 m/s | — |
| Yaw-rate error while moving (commands up to ±0.5 rad/s) | 0.058 rad/s | — |
| Mean speed with zero command (includes step-to-step sway) | 0.052 m/s | 0.010 m/s |
| Contact match | 0.84 | 1.00 |
| Mean peak foot height | 4.2 cm | 0.9 cm |
| Peak joint torque | 10.0 Nm | 5.3 Nm |

Walking reaches about 90% of the commanded speed and tracks yaw well. The legs never cross in the videos.

**Torque while walking** (steady stepping, N m, compared with stage 1): p95 and RMS barely change. Hip roll p99 rises to 6.3 Nm (stage 1: 5.7), about the static single-support moment. The limit is touched only in rare spikes: left ankle 0.01% of control steps, right hip roll once. RMS stays at or below 2.5 Nm on every joint. The actuator target is unchanged from stage 1: about 2.5 Nm continuous, about 6.5 Nm p99 and brief 10 Nm peaks.

| Joint | Left p95 | Left p99 | Left max | Right p95 | Right p99 | Right max | RMS (L / R) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hip yaw | 1.5 | 2.0 | 3.1 | 1.3 | 1.7 | 2.7 | 0.7 / 0.6 |
| hip roll | 4.9 | 6.3 | 8.9 | 4.8 | 5.5 | 10.0 | 2.4 / 2.5 |
| hip pitch | 1.7 | 2.3 | 4.6 | 1.7 | 2.2 | 3.4 | 0.8 / 0.9 |
| knee pitch | 4.1 | 5.0 | 6.9 | 3.5 | 4.6 | 7.5 | 2.3 / 1.9 |
| ankle pitch | 4.4 | 6.6 | 10.0 | 2.3 | 3.1 | 5.2 | 1.9 / 1.1 |

### Left/right symmetry

In the videos the left leg moves as if injured, without the right leg's range of motion. The model is exactly mirror-symmetric (checked: body positions, masses, inertias, joint axes, ranges and actuators; CoM on the midline), so this is a learned habit. Nothing in the stage-1 or stage-2 reward asked for symmetry, so the policy settled into a favoured-leg gait in stage 1, and stage 2 inherited it.

`evaluate.py` now measures this. Stepping episodes, re-evaluated with `--impl jax` (ROM = p95 − p5 joint angle in rad; mirror error = RMS of `right(t) − M·left(t − T/2)` and vice versa, where `M` flips hip yaw and roll):

| Joint | `stage1_v5` ROM L / R | `stage2_walk` ROM L / R | `stage2_walk` torque p95 L / R (N m) | `stage2_walk` mirror error (rad) |
| --- | ---: | ---: | ---: | ---: |
| hip yaw | 0.149 / 0.091 | 0.194 / 0.135 | 1.5 / 1.3 | 0.10 |
| hip roll | 0.086 / 0.146 | 0.119 / 0.144 | 4.9 / 4.8 | 0.05 |
| hip pitch | 0.202 / 0.177 | 0.252 / 0.262 | 1.7 / 1.7 | 0.10 |
| knee pitch | 0.402 / 0.354 | 0.419 / 0.401 | 4.1 / 3.5 | 0.11 |
| ankle pitch | 0.226 / 0.325 | 0.349 / 0.410 | 4.4 / 2.4 | 0.21 |

The left ankle uses 15–30% less range than the right while carrying nearly twice the torque, and its mirror error (about 0.2 rad, 11°) is twice that of any other joint. The left hip roll also moves less.

**Symmetry reward.** In a symmetric gait each leg repeats what the other did half a gait cycle earlier, mirrored. The env keeps the last 32 control steps of joint angles and, in stepping episodes, penalizes the squared error `|q_R(t) − M q_L(t − T/2)|² + |q_L(t) − M q_R(t − T/2)|²`, where T is taken from the gait clock. Pitch joints always count. Hip yaw and roll are weighted by `exp(−(v_y/0.03)² − (ω_z/0.1)²)`, because sideways and turning commands legitimately need asymmetric hip motion. Scale −4.0, about 5% of the per-step reward for the current gait. The penalty isn't in the critic's observation; the history is left out so the observation layout stays compatible with existing runs.

Fine-tune from `stage2_walk` on the g5 (about 40 min):

```bash
python simulation/training/train_ppo.py --run-name stage2_sym --task walk \
    --warm-start simulation/results/local/train/stage2_walk 2>&1 | tee stage2_sym.log
```

Check the symmetry table at the end of the log: ROM L/R ratios should move toward 1 and the ankle mirror error should drop well below 0.2 rad, without losing velocity tracking or survival. If the habit doesn't train out, retrain stages 1 and 2 from scratch with the symmetry reward on from the start (`--env-override reward_config.scales.symmetry=-4.0` for stage 1). A policy that is mirror-symmetric by construction is the stronger fallback.

## Validation (CPU, 2026-11-01)

Apple M2, JAX backend, 512 environments, about 3.9M steps, about 10 minutes. This checks for a learning signal; it is not a trained policy.

| Steps | Eval reward | Episode length (of 1000) |
| ---: | ---: | ---: |
| 0 | -1.05 | 45 |
| 1.3M | 2.38 | 122 |
| 2.6M | 30.0 | 609 |
| 3.9M | 63.0 | 965 |

Final deterministic eval, 64 episodes with pushes, noise and latency:

- Survival: 100% over 20 s.
- Base drift: 3 cm.
- Peak torque: 5.7 Nm.
- Stepping episodes: contact match 0.60, the score of a robot that **never lifts a foot**.

So, as expected, the policy learned to stand and absorb pushes first. Stepping has not emerged at this budget. On the GPU run, watch for stepping contact match to rise above 0.6 and peak foot height to approach 3–5 cm. If it stays stuck in "stand still" well past ~50M steps, try `--env-override stand_probability=0.0` and a larger `feet_phase` / `contact_match` scale.

## GPU machine setup

Tested on a **g5.xlarge** (1 × NVIDIA A10G 24 GB, 4 vCPU, 16 GiB RAM), Python 3.14. A g6e.xlarge (1 × L40S 48 GB, 32 GiB RAM) should be faster. 4 vCPUs are enough because the simulation runs on the GPU. If the GPU runs out of memory, use `--num-envs 4096`.

Measured on the g5.xlarge: about 35k env steps/s with 8192 envs (10M steps in 283 s plus 46 s of JIT), so the default 150M-step run takes about 70 min.

### Recommended: AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)

This AMI ships the NVIDIA driver, so `nvidia-smi` works on first boot. [uv](https://docs.astral.sh/uv/) makes the venv and fetches Python 3.14 if needed (3.11–3.14 all work; the pinned requirements have binary wheels for each).

```bash
nvidia-smi                                   # GPU visible
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone https://github.com/pbharrin/dodo-bot.git && cd dodo-bot
git checkout v5-policy-training
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -r simulation/training/requirements.txt "jax[cuda12]==0.9.2"
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

If JAX complains about cuDNN/cuSPARSE versions, `unset LD_LIBRARY_PATH` (the AMI's own CUDA can shadow the pip-installed one). Use `uv pip install`, not `pip`, inside a uv venv.

### Fallback: plain Ubuntu AMI (tested on 26.04)

Plain Ubuntu has no NVIDIA driver (`nvidia-smi: command not found`, and JAX fails with `cuInit(0) failed: Unknown CUDA error 303`). Install the driver and reboot, then follow the steps above from `curl`:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu nvidia:580-server
sudo reboot
```

### Video rendering (optional)

Videos need EGL/OpenGL system libraries. Without them, `import mujoco` fails with `'NoneType' object has no attribute 'eglQueryString'` or `... 'glGetError'`. To enable rendering:

```bash
sudo apt-get install -y libegl1 libegl-mesa0 libgl1 libopengl0 ffmpeg
sudo apt-get install -y libnvidia-gl-580-server   # optional, GPU rendering; must match the driver
```

You don't need them to train. `train_ppo.py` and `evaluate.py` check for EGL at startup; if it's missing they print a warning, set `MUJOCO_GL=disable` and skip videos. You can also set `MUJOCO_GL=disable` yourself. Render later on a Mac by copying the run directory back and running `python simulation/training/evaluate.py <run_dir> --impl jax`.

Pre-flight on the GPU, using both backends:

```bash
python -m unittest discover -s simulation/training -p test_dodo_env.py
DODO_TEST_IMPL=warp python -m unittest discover -s simulation/training -p test_dodo_env.py
```

Short GPU trial (about 10M steps, ~5 min on a g5) to check throughput and that reward climbs:

```bash
python simulation/training/train_ppo.py --run-name warp_trial --num-timesteps 10000000 --num-evals 5
```

Full stage-1 run (150M steps by default). Use `tmux` so the run survives an SSH disconnect:

```bash
tmux new -s train
python simulation/training/train_ppo.py --run-name stage1_v5
```

Results go to `simulation/results/local/train/<run-name>/` (git-ignored): `progress.csv`, `checkpoints/`, `params.pkl`, `eval.json` and `rollout*_step.mp4` / `rollout*_stand.mp4`. Copy back the run directory (not the checkpoints) with `scp -r` or `aws s3 cp --recursive`.

Useful flags:

- `--task walk` trains stage 2 (see above). `--warm-start <run_dir>` starts from that run's `params.pkl` with the command inputs adapted.
- `--restore <run>/checkpoints/<step>` resumes a run of the same task exactly.
- `--no-randomization` is a debugging baseline. If the policy can't learn without randomization, the rewards are the problem, not robustness.
- `--env-override KEY=JSON` changes env config without editing code, e.g. `stand_probability=0.0`, `push_config.enable=false`, `swing_height=[0.04,0.06]`, `lin_vel_x=[0,0.15]`.
- `--impl jax` runs on CPU or without Warp. It works but is much slower.
- `--env-override solver_iterations=4` (and `ls_iterations=…`) overrides the model's contact-solver iterations (default 2, from the XML). Warp prints `solver iterations limit reached - please increase iterations beyond 2` when a step doesn't converge. Occasional warnings are expected; if they appear constantly, raise it to 3–4 at some speed cost.

If Warp reports contact or constraint overflow, raise `naconmax` (total contacts across all worlds; each world uses up to 8 foot-floor contacts) or `njmax` (constraint rows per world) with `--env-override`.

## Reading a run

`progress.csv` / console:

- `len` (episode length, 1000 = 20 s without falling) should rise first.
- `contact_match` is the fraction of steps where each foot's contact state agrees with the gait clock. A policy that never lifts a foot still scores about 0.7: stand episodes match 100%, and standing through a stepping episode still matches the 60% stance portion of the clock. Only real stepping pushes it toward 1.0.

`eval.json` (from `evaluate.py`, nominal model with noise, pushes and latency) reports, separately for stepping and standing episodes:

- `survival_rate`
- `contact_match`
- `mean_peak_foot_height_m` (target swing clearance 3–5 cm)
- `base_drift_m` (should stay small for in-place stepping)
- `peak_joint_torque_Nm` (compare with the 10 Nm limit and, later, measured actuator limits)
- `velocity_tracking`: command-tracking error while commanded to move and while commanded zero (stage 2).
- `symmetry`: left vs. right range of motion, p95 torque and half-cycle mirror error while stepping (also printed as a table).
- `torque`: per joint, the fraction of control steps at the torque limit (`saturated_frac`, |τ| ≥ 98% of limit) plus p50/p95/p99/max/RMS torque. It is split into `steady` and `push_recovery` (the 1 s after each push). Saturation during push recovery is acceptable. Saturation in `steady` stepping means the gait needs more torque than the actuator has. The same table prints to the console.

Probe specific conditions with `--env-override` (e.g. `push_config.enable=false`). Those results go to `eval_override.json`, so the training-time `eval.json` is kept.

**Watch the videos.** Reward can go up for degenerate reasons: shuffling, toe-dragging, or one foot vibrating. Common fixes: raise `feet_phase` or `contact_match` if the feet barely lift, raise `feet_slip` for shuffling, raise `action_rate` for jitter.

## Design choices and what to replace with hardware data

- **Action:** `q* = q_home + 0.5 rad × a`, with `a` tanh-bounded to [-1, 1], clamped to the joint range, and tracked by a joint-level PD servo (kp 30, kd 1, ±10 Nm). This is the "bounded joint-position targets with validated local gains" interface from the project plan. The real MKS XDrive position loop runs on the motor side through the capstan. Once a module is on the bench, identify its joint-level response and match `KP`/`KD` (or model it explicitly).
- **Observations:** the policy gets only what the robot will have: IMU gyro and gravity direction, joint encoders, the previous action and the gait clock. Base velocity, contacts and foot heights go only to the critic.
- **Sim-to-real randomization:** floor friction, link masses, torso mass (−0.3 to +0.8 kg) and CoM (±2 cm), armature, damping, friction loss, servo gains, torque limit, a one-control-step action delay on half the episodes, ±0.02 rad encoder offsets, sensor noise and base-velocity pushes.
- **Model estimates** (tagged `ESTIMATE` in `build_model_v5.py`) still need hardware data: 0.45 kg per actuator module, 1.6 kg torso, link masses, rotor inertia (→ 0.0096 kg m² armature), joint friction/damping, and where the knee and ankle motors sit. When modules are weighed and characterized (milestone 2), update the generator, regenerate, rerun the tests and retrain. Policies are tied to a model version.
- **Collision:** only feet touch the floor; leg-on-leg contact isn't simulated. The `feet_distance` reward keeps the feet at least 120 mm apart. Check the videos for leg crossing that CAD would forbid.
- **No ankle roll:** sideways balance must come from hip roll and foot placement, as on the planned hardware.

## Next stages

1. Stage 1 (this): stand and step in place, robust to pushes.
2. Stage 2 (set up, see above): walk forward slowly, sidestep and turn, warm-started from stage 1.
3. Before hardware: replay the policy in C MuJoCo (not MJX) with added latency and measured actuator models, then export the actor network for UNO Q inference. Keep joint ordering, units, gains, normalization statistics and action scaling identical.
