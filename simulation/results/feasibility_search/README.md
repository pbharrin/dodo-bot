# Planning-only feasibility search: release path found, lift still unsolved

The unchanged model admits a **sampled quasi-static path from two-foot standing to a zero-load release pose** under the stated geometric/contact tolerances. The separate validator checked 320 interpolated samples across 65 poses. This is a useful candidate for subsequent controller development, not a completed airborne transition or a dynamic success.

No forward integration, motor-command execution, model modification, or walking-policy training was performed.

## Search results

| Run | Endpoint trials | Passing release trials | Transfer-plus-lift attempts | Complete transfer-plus-lift paths |
| --- | ---: | ---: | ---: | ---: |
| [Broad sweep](current.json) | 360 | 5 | 10 | 0 |
| [64-waypoint refinement](refined.json) | 41 | 1 | 2 | 0 |
| [Constrained lift refinement](constrained_lift.json) | 41 | 1 | 2 | 0 |

The refined runs repeat part of the same seeded endpoint grid; their endpoint counts are not additional unique candidates. The five passing broad-sweep trials include near-duplicate solutions from different seeds.

All passing release trials use toe pivots and approximately 290 mm COM height. The selected candidate uses the **outer toe corner of the right foot**, with the left foot supporting the robot. This differs from the earlier heel-pivot, 250 mm COM-height transition experiment.

The initial sweep required an arbitrary exact endpoint COM target and rejected physically feasible poses. An audit found static zero-swing-load solutions among those rejected candidates. Endpoint COM coordinates now guide optimization, while acceptance requires the actual COM support margin and a 210..320 mm height range. Clearance, joint, contact, friction, and torque requirements were retained. The early checkpoint is preserved in Git history; it is not a global infeasibility result.

## Independently rechecked release-only path

[release_path.json](release_path.json) contains the 65 ordered joint/base poses and validation metadata. The separate [validator](../../experiments/validate_release_plan.py) recomputed geometry and force LPs from the stored poses rather than trusting pass flags.

| Quantity | Rechecked result |
| --- | ---: |
| Interpolated samples checked | 320 |
| Minimum checked body clearance | 5.943 mm |
| Minimum remaining joint range | 2.000 degrees |
| Maximum stance position error | 0.316 mm |
| Maximum stance orientation error | 0.234 degrees |
| Maximum pivot position error | 0.300 mm |
| Peak torque among returned feasible static solutions | 1.834 Nm |
| Planning torque bound | 3.8 Nm; model limit remains 4 Nm |
| Endpoint COM height | 290.168 mm |
| Endpoint COM support margin | 2.433 mm |
| Endpoint computed swing-foot force | 0 N |
| Endpoint computed stance-foot force | 20.307 N |

The static LP minimizes swing load, not peak torque. These torque values describe feasible static solutions, not minimum actuator sizes or dynamic torque requirements. The 2.433 mm endpoint support margin leaves limited tracking tolerance beyond the required 2 mm.

The swing foot remains at the ground pivot, with the same nominal 0.2 mm penetration convention used by the model-contact references. Ideal rigid-contact statics cannot predict the measured force that MuJoCo's compliant contact model would produce at that pose. **Zero computed load is not measured unloading or airborne clearance.**

## What still failed

At 16 waypoints, several transfer paths failed interpolation checks even though their endpoints were acceptable. Increasing to 64 waypoints allowed a reverse-searched transfer to connect to the exact `bird_stand` configuration within the original tolerances.

The first ordinary lift waypoint then reduced COM support margin to about 1.710 mm, below the 2 mm requirement. Constrained optimization found a first lift waypoint satisfying the local geometric requirements, but the connection to it reduced left-shank/right-foot clearance to about **4.171 mm**, below the 5 mm requirement. That segment was rejected even though its static force LP was feasible. Neither a valid endpoint nor a feasible force distribution is sufficient to certify the motion between poses.

The exported plan deliberately stops at release. No time schedule, speed/acceleration bounds, collision-free airborne lift, ten-second hold, disturbance recovery, or hardware validity is claimed. Sampled checks do not prove continuous-path feasibility or dynamic trackability. The finite search does not cover arbitrary contact sequences, foot sliding, intermediate steps, or momentum-assisted transfers.

## Reproduce and provenance

See [search commands and assumptions](../../experiments/FEASIBILITY_SEARCH.md). Five regression checks passed, covering known-pose statics, insufficient-torque rejection, joint-margin checks, endpoint COM acceptance, and checkpoint serialization. The release validator passed all 320 sampled checks with zero forward integration steps.

Each JSON records its model/script hashes and parameters. The broad and finer sweeps precede the optional constrained-lift code addition; their recorded code versions are preserved in the checkpoint history. All reports use model SHA-256:

```text
221a508b97a0a16b66a7dec9a6afc3cb43232b1873c3cc5a2b6a64424ab2ccd0
```

Checkpoint commits during the search: `29ac769` (initial sweep), `247876d` (physical endpoint acceptance and path refinement), and `6d54f54` (completed sweeps and constrained lift findings). The final commit adds the independently validated release-only plan.

The result supports keeping the original model for the next investigation. The next controller test can track this release-only path with measured force and clearance gates; lifting still needs a verified connection. No such dynamic experiment was started here.
