# Dynamic outer-toe release test results

**The transition remains unsolved.** The validated static path is now connected to v2's torque controller and measured-force loop, but the dynamic runs stall during early path tracking. None reaches endpoint unloading, so these results do not test the endpoint PI's ability to release the foot.

All final comparisons use the same unchanged v4 model, source files, left support, 65-pose outer-toe path, torque controller, and release criteria. The runner and controller SHA256 values are recorded in each JSON and match the checked-in sources. The model SHA256 is `221a508b97a0a16b66a7dec9a6afc3cb43232b1873c3cc5a2b6a64424ab2ccd0`. MuJoCo 3.17.0 and OSQP 1.1.3 were used, with a 2 ms model timestep and `OPENBLAS_NUM_THREADS=1`.

| Run | Result | Simulated time | Path completed | Final right-foot load | Final COM support margin | Logged peak torque |
|---|---|---:|---:|---:|---:|---:|
| Contact handoff, nominal 12 s transfer | PATH_TRACKING_STALLED | 13.898 s | 12.31% | 13.118 N | -43.01 mm | 1.004 Nm |
| Contact handoff, nominal 24 s transfer | PATH_TRACKING_STALLED | 18.128 s | 11.61% | 13.122 N | -41.79 mm | 1.014 Nm |
| Immediate single-corner force model | QP_FAILURE | 1.752 s | 0.00245% | 13.916 N | -79.10 mm | 1.766 Nm |

The release limit is approximately **1.015 N**, and the COM must be more than **2 mm inside** the left support sole. Negative margin means outside. All success streaks are zero. Every run verifies initial two-foot support and makes no state resets afterward.

The handoff improves the initial behavior relative to the immediate single-corner force assumption. It does not solve path tracking: the 12-second run ends with 7.18 mm COM tracking error and 33.23 degrees maximum joint tracking error at the left knee. Doubling the nominal transfer duration ends with 7.61 mm and 38.18 degrees respectively. Right-foot load increases rather than transferring to the left. The governor pauses progress and then stops after its eight-second stall timeout.

These two runs remain below motor limits, have no unwanted body contacts, and do not approach the checked collision pairs' 50 mm distance-query cutoff. The reported 50 mm minimum is a **capped query result**, not an exact minimum separation. Their final left-foot drift is 4.27/4.98 mm and tilt 1.70/2.02 degrees. Those measurements do not establish that the remainder of the path can be tracked safely.

For the immediate-corner comparison, OSQP reports primal infeasibility; v2's separate HiGHS check confirms the linear constraints are infeasible at the final state. The elastic diagnosis is one possible relaxation, not proof of a unique physical cause. No relaxed solution is applied.

## Files and reproduction

- `contact_handoff.json`: default runner, finalized measurements and one-second diagnostic timeline.
- `slow_transfer.json`: `--transfer-seconds 24 --seconds 80`.
- `instant_point.json`: `--instant-point-contact`, with all other final defaults unchanged.
- `comparison.csv`: compact one-second diagnostic records plus each run's final record. Full 50 Hz CSV files are generated locally by rerunning.
- `nominal.json`: the initial checkpoint from commit `19cb75c`, before adding the contact-force handoff and revising the governor's joint-error thresholds. Its source hashes differ intentionally; use the three files above for comparisons with current code.

```bash
OPENBLAS_NUM_THREADS=1 python simulation/experiments/dodo_release_test.py \
  --output simulation/results/local/contact_handoff
OPENBLAS_NUM_THREADS=1 python simulation/experiments/dodo_release_test.py \
  --transfer-seconds 24 --seconds 80 \
  --output simulation/results/local/slow_transfer
OPENBLAS_NUM_THREADS=1 python simulation/experiments/dodo_release_test.py \
  --instant-point-contact \
  --output simulation/results/local/instant_point
python -m unittest discover -s simulation/experiments -p 'test_*.py'
```

All **13 regression checks passed**. The final dynamic comparisons were rerun after the source was frozen, and all three report unchanged model hashes. Rendering was not needed or exercised for these comparisons. See [controller details and video instructions](../../experiments/RELEASE_TEST.md).

## Recommended next experiment

Keep the model and plan fixed. Isolate the first 10–15% of the path and investigate why the whole-body controller departs from the requested knee/posture and sends more load toward the right foot. Compare predicted versus measured contact forces and accelerations while testing posture-task priority and the flat-to-pivot contact estimate. Require stable tracking of this initial segment before resuming endpoint unloading or adding lift. Slowing the present controller alone did not fix the failure; these results do not establish geometric infeasibility or justify widening the model.
