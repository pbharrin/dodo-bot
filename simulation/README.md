# Simulation

MuJoCo models, standing and walking controllers, diagnostic experiments, and selected results for the Dodo Bot biped.

## Layout

- `models/`: model generators and MuJoCo XML model definitions.
- `controllers/`: reusable control algorithms (standing, CoP, and future walking).
- `experiments/`: reproducible experiments, probes, and tests.
- `results/`: small selected CSV outputs and documented experiment findings. Avoid committing large video files.

## Current work

Initial prototype: approximately 50 cm tall dodo-inspired biped, ten actuated leg joints, flat-foot model in MuJoCo. Standing and push recovery experiments established a baseline; weight transfer and single-leg support are in progress.

Scripts and models will be added after confirming the working versions from the simulation environment.
