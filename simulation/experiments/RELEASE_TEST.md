# Outer-toe release controller experiment

`dodo_release_test.py` executes the sampled release plan exported by the feasibility search, starting from two-foot `bird_stand`. It imports v2's whole-body torque QP, measurements, and bounded force PI. It commands no lift and does not establish a walking policy. The original model, 130 mm hip spacing, motor limits, and successful initialized single-foot baseline remain unchanged.

From the repository root:

```bash
python -m pip install -r simulation/requirements.txt
python simulation/experiments/dodo_release_test.py \
  --output simulation/results/local/release_left
```

The model and plan default paths are relative to the script, so the command also works from another directory. Only the validated **left-support/right-outer-toe** path is accepted. To specify inputs explicitly:

```bash
python simulation/experiments/dodo_release_test.py \
  --model simulation/models/dodo_floating.xml \
  --plan simulation/results/feasibility_search/release_path.json \
  --transfer-seconds 24 --seconds 80 \
  --output simulation/results/local/release_left_slow
```

JSON is atomically updated once per simulation second and finalized on ordinary failures. CSV contains 50 Hz measurements, targets' tracking errors, QP residuals and predicted forces, force-feedback commands, per-corner force caps, path progress, governor reasons, and release blockers. A failed QP retains v2's independent linear-feasibility diagnosis. Exit code 0 means release-hold success; 2 means failure or timeout. Numerical findings are in [the results report](../results/release_test/README.md).

## Controller and contact sequence

1. Settle on both feet for at least one second and verify measured two-foot support.
2. Follow all 65 plan poses with geodesic interpolation and a quintic path clock, nominally 12 seconds. Geometry and COM references come directly from the plan; no pose resets occur after initialization.
3. At the endpoint, ramp the measured-force PI target toward 1% of robot weight over four seconds. The PI gains and anti-windup are inherited from v2. Its additional load-to-COM IK correction is disabled so the reference stays on the validated path.
4. Remain at that reference and require **two continuous seconds** of all release gates. There is no lift phase and no switch to a single-support force model to manufacture zero swing-foot force.

The toe-pivot acceleration and orientation tasks come from v2. A new optional `force_contact_caps` QP argument separates the force estimate from the pivot task during the initial contact handoff. For the right foot's non-pivot corners, the upper normal-force bound is `3*weight*exp(-(max(measured_height,0)/0.001)**2)`; the pivot retains `3*weight`. Thus a nearly flat sole can initially carry forces at multiple corners, then lose those modeled force capacities as the corners rise. These are optimization bounds, not forces injected into MuJoCo. Contact itself remains the unchanged MuJoCo contact model. This heuristic does not prove complementarity or force-estimation accuracy; measured forces remain authoritative. Existing v2 callers omit the option and retain their previous behavior.

`--instant-point-contact` provides a diagnostic comparison with the original immediate restriction to one force-bearing corner. `--disable-active-unloading` disables the endpoint force PI for a comparison; it has no effect on transfer before the endpoint is reached.

## Progress and acceptance

The governor slows linearly and then pauses as tracking or geometry approaches its thresholds:

| Quantity | Slowdown begins | Pause |
|---|---:|---:|
| Maximum joint tracking error | 8 degrees | 15 degrees |
| COM tracking error | 2.5 mm | 5 mm |
| Checked body clearance | 5 mm | 4 mm |
| Remaining joint margin | 1.5 degrees | 0.75 degrees |
| Support-foot drift | 3 mm | 5 mm |
| Support-foot tilt | 2 degrees | 3 degrees |

The joint thresholds allow the approximately 7.7-degree posture error already observed while v2 settles on two feet. They do not relax the plan's geometric validation or the release gates. Eight continuous seconds below 1% path speed stops with `PATH_TRACKING_STALLED`; the total simulation timeout also limits prolonged slow progress.

Success requires the endpoint to have been reached, verified initial two-foot support, swing load below 5% of weight (about 1.015 N), support load above 85%, COM and measured CoP margins above 2 mm, COM speed below 0.03 m/s, support sole speed below 0.01 m/s, support drift below 5 mm, tilt below 3 degrees, at least two support contacts, torque utilization below 98%, body clearance at least 5 mm, joint margin at least 0.5 degrees, and no unwanted body contact. A foot need not be airborne for this **release-only** result. The CSV column `release_blockers_all` includes the additional endpoint/clearance gates beyond v2's original release diagnostics.

Hard stops include checked body clearance below 2 mm, unwanted body contact, support drift above 10 mm or tilt above 5 degrees, joint-limit overrun above 0.5 degrees, nonfinite state, fall/excessive base drift, and any rejected QP. No failed or relaxed QP solution is applied.

## Video and checks

For headless Ubuntu rendering, after installing the OSMesa system library if needed:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python simulation/experiments/dodo_release_test.py \
  --output simulation/results/local/release_video \
  --video simulation/results/local/release_video.mp4
```

The dynamic validation runs in the report do not require rendering. The prior EGL import error is unrelated to controller feasibility.

```bash
python -m unittest discover -s simulation/experiments -p 'test_*.py'
```

The added checks reject mismatched model hashes and a non-`bird_stand` starting reference, prevent predicted unloading or an incomplete path from opening the success gate, enforce body clearance, and reject invalid contact-force caps. Existing v2 and feasibility regression tests also run.
