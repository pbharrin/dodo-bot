# Learned stand / step-in-place policy (stage 1)

Trains a 50 Hz joint-position policy for the [v5 model](../models/build_model_v5.py) with MuJoCo Playground, Brax PPO and MJX-Warp. Each episode is either **stand** (25%) or **step in place** following a two-foot gait clock (75%). The base-velocity command is zero, so a good policy marches on the spot without drifting and recovers from pushes.

Stage 1 is meant to replace the quasi-static two-foot-to-one-foot work. Stepping in place includes weight transfer and single support, but done dynamically.

| File | Purpose |
| --- | --- |
| `dodo_env.py` | `DodoStepInPlace` environment, rewards, observations, domain randomization |
| `train_ppo.py` | PPO training, checkpoints, final evaluation and videos |
| `evaluate.py` | Reload a run, run deterministic rollouts, write `eval.json` and videos |
| `test_dodo_env.py` | Pre-flight checks: model mass/geometry, MJX vs MuJoCo, passive standing, gait clock, randomization |

## Status

- **CPU-validated (JAX backend, Apple M2):** all tests pass, the full train → save → evaluate → render pipeline runs, and the zero-action robot stands for 4 s. See "Validation" below for a short CPU learning-signal run.
- **Not yet run:** MJX-Warp on a GPU, or a full-length training run. No walking or stepping result is claimed yet. The first GPU session should start with the Warp tests below.

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

## g6e.xlarge setup

g6e.xlarge = 1 × NVIDIA L40S (48 GB), 4 vCPU, 32 GiB RAM. Use an Ubuntu 22.04/24.04 AMI with an NVIDIA driver that supports CUDA 12 (e.g. the AWS Deep Learning Base GPU AMI). 4 vCPUs are enough because the simulation runs on the GPU.

```bash
nvidia-smi                                  # driver present, L40S visible
sudo apt-get install -y python3.11-venv ffmpeg libegl1   # EGL for headless video
git clone <this repo> && cd dodo-bot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r simulation/training/requirements.txt "jax[cuda12]==0.9.2"
python -c "import jax; print(jax.devices())"   # expect a CudaDevice
```

Pre-flight on the GPU, using both backends:

```bash
python -m unittest discover -s simulation/training -p test_dodo_env.py
DODO_TEST_IMPL=warp python -m unittest discover -s simulation/training -p test_dodo_env.py
```

Short GPU trial (about 10M steps) to check throughput and that reward climbs:

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

- `--restore <run>/checkpoints/<step>` resumes a run.
- `--no-randomization` is a debugging baseline. If the policy can't learn without randomization, the rewards are the problem, not robustness.
- `--env-override KEY=JSON` changes env config without editing code, e.g. `stand_probability=0.0`, `push_config.enable=false`, `swing_height=[0.04,0.06]`, `lin_vel_x=[-0.2,0.3]` (stage 2: walking).
- `--impl jax` runs on CPU or without Warp. It works but is much slower.

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
2. Stage 2: widen `lin_vel_x` / `lin_vel_y` / `ang_vel_yaw` to walk forward slowly and turn, resuming from stage 1.
3. Before hardware: replay the policy in C MuJoCo (not MJX) with added latency and measured actuator models, then export the actor network for UNO Q inference. Keep joint ordering, units, gains, normalization statistics and action scaling identical.
