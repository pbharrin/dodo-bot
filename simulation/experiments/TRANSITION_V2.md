# Active-unloading transition experiment

`dodo_transition_v2.py` is a standalone follow-up to the stalled heel-pivot transfer. It retains the original v4 model, ten motors, 4 Nm motor limits, and 130 mm hip spacing. It starts in `bird_stand` with both feet loaded and applies motor torques only.

**This is a diagnostic experiment, not an established successful transition controller.** The successful initialized one-foot balance experiment remains unchanged in `dodo_single_support.py`.

## Run

From the repository root, inside your Python virtual environment:

```bash
python -m pip install -r simulation/requirements.txt
python simulation/experiments/dodo_transition_v2.py \
  --model simulation/models/dodo_floating.xml \
  --output simulation/results/local/transition_v2_left
```

For a control comparison using the same initial geometric planner, with active unloading disabled:

```bash
python simulation/experiments/dodo_transition_v2.py \
  --model simulation/models/dodo_floating.xml \
  --disable-active-unloading \
  --output simulation/results/local/transition_v2_disabled
```

For video on a headless Ubuntu machine:

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  python simulation/experiments/dodo_transition_v2.py \
  --model simulation/models/dodo_floating.xml \
  --output simulation/results/local/transition_v2_left_video \
  --video simulation/results/local/transition_v2_left.mp4
```

The script also works as a single downloaded file next to `dodo_floating.xml`; use `python dodo_transition_v2.py --model dodo_floating.xml --output transition_v2_left`.

## What changed

1. The unload phase smoothly ramps desired swing-foot load from its measured entry value to 1% of body weight over four seconds. Filtered measured force drives a bounded PI correction. The QP receives that corrected load target as a **soft objective**, not a fabricated contact force or an external wrench.
2. A 3 mm feedforward COM advance follows the load ramp toward the supporting foot. Remaining load error adds a feedback correction; total requested correction is capped at 15 mm. Every 0.5 seconds a local IK update requests at most 1 mm of additional lateral motion. Accepted references are smoothly blended; simulated `qpos` is never reset.
3. IK updates must pass stance, pivot, collision, joint-margin and joint-step checks. They must also move the COM toward the request without worsening its residual. Rejected updates retain the previous reference and freeze COM integral accumulation. Passing these local checks does not prove a dynamically feasible trajectory.
4. A measured release gate checks swing/support force, COM margin and speed, support-foot motion and CoP, torque reserve, and unwanted contact for 0.25 seconds. Predicted QP force cannot open this gate. If it opens, the lift path is replanned from the updated unloading reference.
5. The signed collision-distance gradient is corrected for penetration. Body contact is reported from active penetrated contacts, including penetration smaller than the old 0.1 mm tolerance. The default experiment stops earlier if checked body clearance falls below 0.5 mm.

The initial rock-transfer planner and controller remain the v1 approach. This experiment isolates whether active unloading and small local posture updates can extend that path; it does not search every possible transfer strategy.

## Read the diagnostics

Every run writes a JSON summary and, if dynamics ran, a CSV. Their shared `run_id` detects stale CSVs left after a planning-only failure. The JSON also records script/model SHA-256, library versions, parameters, planned poses, final state, and model-hash verification.

| Field | Meaning |
| --- | --- |
| `unload_target_fz_N` | Smooth desired swing-foot load |
| `unload_command_fz_N` | PI-corrected, nonnegative QP target |
| `unload_filtered_fz_N` | Low-pass filtered measured force |
| `right_predicted_fz_N` / `right_fz_N` | Predicted versus actual force for left-support runs |
| `force_objective_enabled` | Whether the load objective is actually being applied |
| `release_blockers`, `gate_*` | Individual measured conditions preventing lift |
| `minimum_body_clearance_m`, `closest_body_pair` | Smallest checked collision-pair gap, capped at 50 mm |
| `*_limit_margin_deg` | Signed distance to the nearer joint limit; negative means limit penetration |
| `max_joint_tracking_error_deg`, `com_tracking_error_m` | Actual-versus-target tracking errors |
| `qp_status`, `qp_violation`, `qp_*_residual` | Solver status and numerical diagnostics |
| JSON `posture_updates` | Every IK request, acceptance decision, and rejection reason |
| JSON `diagnosis.last_qp.constraint_diagnosis` | Independent linear feasibility check if a QP fails |

QP values correspond to the command computed immediately before the logged MuJoCo step; measured CSV values are from the state after that step (2 ms later). Force prediction error therefore includes that timing difference as well as model mismatch. Outside active unloading, target-feedback fields are NaN in CSV and null in JSON.

If linear constraints are infeasible, a diagnostic elastic LP lists one possible set of constraint relaxations. It does not prove a unique cause, and its relaxed solution is **never applied** as a control command.

## Outcomes and limits

- `SUCCESS_TWO_FEET_TO_ONE`: verified two-foot start, measured release, airborne foot without subsequent touchdown, and the requested continuous balance hold.
- `BODY_CLEARANCE_LIMIT`: checked bodies approach within the configured margin; inspect `closest_body_pair` and joint margins.
- `PHASE_TIMEOUT_UNLOAD`: the measured release gate never passed; inspect force tracking and `release_blockers`.
- `SUPPORT_POSE_LIMIT` / `UNWANTED_BODY_CONTACT`: supporting foot moved too far or an unwanted active body contact occurred.
- `QP_FAILURE` / `JOINT_STOPPING_INFEASIBLE`: inspect the last QP and joint diagnostics.
- `PLAN_FOUND_NOT_SIMULATED`: `--plan-only` completed; this is not a dynamic success.

Exit code 0 means measured success or successful plan-only completion; exit code 2 means the experiment did not meet its goal. Always read the JSON result. Default unload timeout is 12 seconds and total simulation timeout is 40 seconds.

Contact acceleration constraints are hard by default. `--contact-soft-weight 1000000` is an optional diagnostic comparison that makes sole/pivot tracking soft **during unloading only**, while keeping reference anchors fixed. Measured support drift over 10 mm or tilt over 5 degrees stops the run; release still requires drift below 5 mm and tilt below 3 degrees. Soft contact tracking is not evidence that the physical foot stayed planted.

Other useful controls: `--force-weight`, `--force-kp`, `--force-ki`, `--max-com-correction-mm` (0 disables posture correction), `--target-load-fraction`, `--unload-ramp-seconds`, and `--support right`. Right-support full dynamics has not been established by the left-side validation.

Development validation and recorded outcomes are in [the result notes](../results/transition_v2/README.md). Run feedback/geometry regressions with:

```bash
python -m unittest discover -s simulation/experiments -p test_transition_v2.py
```
