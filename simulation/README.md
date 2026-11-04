# Simulation

MuJoCo models, standing and walking controllers, diagnostic experiments, and selected results for the Dodo Bot biped.

## Layout

- `models/`: model generators and MuJoCo XML model definitions.
- `controllers/`: reusable control algorithms (standing, CoP, and future walking).
- `experiments/`: reproducible experiments, probes, and tests.
- `training/`: MuJoCo Playground / MJX-Warp reinforcement-learning environments and training scripts.
- `results/`: small selected CSV outputs and documented experiment findings. Avoid committing large video files.

## Current work

Initial prototype: approximately 50 cm tall dodo-inspired biped, ten actuated leg joints, flat-foot model in MuJoCo. Standing and push recovery experiments established a baseline. Single-foot balance from an initialized pose has passed ten continuous seconds; transition from two feet and walking remain in progress.

## Successful single-foot balance

See [the recorded experiment and reproduction instructions](results/single_support_left/README.md). The standalone runner is [experiments/dodo_single_support.py](experiments/dodo_single_support.py), and the exact tested model is [models/dodo_floating.xml](models/dodo_floating.xml).

The model retains its original 130 mm hip spacing. No geometry changes were made for this experiment.

## Active-unloading transition experiment

[Transition v2](experiments/TRANSITION_V2.md) adds measured-load feedback, bounded posture updates, collision clearance checks, and explicit failure diagnostics on the unchanged model. It is an experimental follow-up; the two-foot-to-one-foot transition is not yet solved. See [validation results](results/transition_v2/README.md).

## Learned stand / step-in-place policy (v5 model)

Development has moved from quasi-static single-support control to dynamic policy training. The [v5 model](models/README.md) uses the project-plan geometry and estimated actuator masses (6.8 kg). The [stage 1 training environment](training/README.md) trains standing and stepping in place with MuJoCo Playground, Brax PPO and MJX-Warp. It is CPU-validated; GPU training has not been run yet.
