# Planning-only feasibility search

This experiment searches release postures and connecting sampled quasi-static paths on the unchanged v4 model. It does **not** execute motor commands, call `mj_step`, or demonstrate dynamic balance. It imports the model/kinematic helpers from `dodo_transition_v2.py`.

```bash
python -m pip install -r simulation/requirements.txt
python simulation/experiments/dodo_feasibility_search.py \
  --max-candidates 360 --max-paths 12 --budget-seconds 600 \
  --output simulation/results/feasibility_search/current.json
```

The left-side search considers all four swing-foot sole corners as fixed pivots, COM heights 230/260/290 mm, fore–aft offsets -30/0/+30 mm relative to the support-foot center, and lateral COM targets 55/65 mm. Seeds include neutral, both torso-yaw directions, the previous release pose, and the successful initialized one-foot pose. A fixed random seed interleaves configurations so partial results cover multiple cases. The saved report includes the actual starting poses and input hashes.

Default acceptance criteria include 5 mm checked body clearance, 2 degrees remaining before every joint limit, contact position errors at most 0.5 mm, support orientation error at most 0.25 degrees, and continuation COM target error at most 2 mm. Endpoint COM targets guide optimization; release acceptance uses the actual COM projection at least 2 mm inside the supporting sole and actual COM height between 210 and 320 mm. Missing an arbitrary endpoint COM target alone does not invalidate static feasibility. Small nominal contact penetration is permitted consistently with the original contact anchors. These are explicit search requirements, not properties already demonstrated by the robot.

For a geometrically valid endpoint, a separate linear program minimizes swing-foot normal force subject to gravity/passive-force balance, nonnegative normal forces, conservative friction pyramids (mu 0.8), support CoP margins, and motor torque bounds of +/-3.8 Nm. This reserves 0.2 Nm below the original +/-4 Nm limits; the XML is not changed. A release candidate needs swing force below 5% of body weight. LP residuals are checked independently.

The search tries both forward continuation from `bird_stand` and reverse continuation from passing release candidates, followed by 25 mm foot lift. The example command allows twelve path attempts (six candidates if both directions are used). Reverse traces are reversed into execution order only after the transfer connection succeeds; failed partial traces retain their search order. Each segment is sampled at least four times and at no more than one degree of joint interpolation per sample. Geometry and statics are rechecked at these samples. Lift samples require a force solution with **only the stance foot**, including the start of lift. This guards against treating a low-load two-contact solution as a one-contact solution.

The JSON is atomically checkpointed after each endpoint and path waypoint. `RUNNING` means an incomplete search. `FEASIBLE_SAMPLED_QUASISTATIC_PATH_FOUND` means a candidate is ready for subsequent dynamic testing, not that a transition has succeeded. Other terminal outcomes record that no path was found within the evaluated configurations/time/path budget. None proves global infeasibility. Closest-pair distances are capped at 100 mm.

Both the seed distribution and imposed contact sequence restrict this search: it does not consider foot sliding, intermediate steps, arbitrary contact switches, momentum-assisted transfers, or every possible posture. A failed interpolated segment is not proof no connecting trajectory exists. Full right-side search is not implied by left-side results.

Tests:

```bash
python -m unittest discover -s simulation/experiments -p test_feasibility_search.py
```

The tests independently check the known one-foot pose's static support, rejection with insufficient motor torque, and the distinction between static force feasibility and the new joint-margin requirement.

For finer interpolation checks around the first candidate, repeat the same seeded search with `--max-candidates 41 --max-paths 2 --path-knots 64 --budget-seconds 300 --output simulation/results/feasibility_search/refined.json`. This retains the original tolerances and changes waypoint density.

## Constrained lift refinement and release-only export

The additional `--refine-lift` option uses SLSQP for lift waypoints with explicit stance-pose, swing-height, body-clearance, joint-limit, COM-margin, and COM-height constraints. The height may adapt within 210..320 mm. Feasible waypoints are still rejected if the connecting interpolation fails geometry or static force checks.

```bash
python simulation/experiments/dodo_feasibility_search.py \
  --max-candidates 41 --max-paths 2 --path-knots 64 --refine-lift \
  --budget-seconds 300 \
  --output simulation/results/feasibility_search/constrained_lift.json
python simulation/experiments/validate_release_plan.py
```

The separate validator reads `refined.json`, extracts a completed transfer from `bird_stand`, and recomputes its sampled geometry and statics. It exports `release_path.json` only as a **release-only** plan, explicitly excluding lift, timing, velocity/acceleration guarantees, and dynamic validation. A partial reverse trace that does not start at `bird_stand` is not eligible.
