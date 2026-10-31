# Simulation

MuJoCo models, standing and walking controllers, diagnostic experiments, and selected results for the Dodo Bot biped.

## Layout

- `models/`: model generators and MuJoCo XML model definitions.
- `controllers/`: reusable control algorithms (standing, CoP, and future walking).
- `experiments/`: reproducible experiments, probes, and tests.
- `results/`: small selected CSV outputs and documented experiment findings. Avoid committing large video files.

## Current work

Initial prototype: approximately 50 cm tall dodo-inspired biped, ten actuated leg joints, flat-foot model in MuJoCo. Standing and push recovery experiments established a baseline. Single-foot balance from an initialized pose has passed ten continuous seconds; transition from two feet and walking remain in progress.

## Successful single-foot balance

See [the recorded experiment and reproduction instructions](results/single_support_left/README.md). The standalone runner is [experiments/dodo_single_support.py](experiments/dodo_single_support.py), and the exact tested model is [models/dodo_floating.xml](models/dodo_floating.xml).

The model retains its original 130 mm hip spacing. No geometry changes were made for this experiment.
