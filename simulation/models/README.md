# Models

Store MuJoCo XML files and Python model-generation scripts here. Record which model version a controller or experiment expects.

| Model | Generator | Used by |
| --- | --- | --- |
| `dodo_floating.xml`, `dodo_fixed.xml` (v4) | `build_model_v4.py` | Classical experiments in `experiments/` and `controllers/`. Unchanged; archived results reference its SHA-256. |
| `dodo_v5.xml` | `build_model_v5.py` | Learned-policy training, stages 1–2b. The archived `stage2_smooth` policy was trained on it. Unchanged. |
| `dodo_v6.xml` | `build_model_v6.py` | Learned-policy training, default from 2026-11-04 (`model='v6'` in the env config). |

## v6 changes from v5

Only the knee and ankle motor positions change; masses, geometry, joints and actuators are identical.

| Item | v5 | v6 | Reason |
| --- | --- | --- | --- |
| Knee motor | 39 mm down the thigh (91 mm above the knee) | 48 mm above the knee | Integrated capstan cartridge: output drum on the joint axis, motor one drum centre distance away on the parent link |
| Ankle motor | 44 mm down the shank (103 mm above the ankle) | 48 mm above the ankle | Same |
| Centre distance | — | 40 mm output drum + 5 mm input drum (8:1) + 3 mm clearance = 48 mm | All `ESTIMATE` |

v5's placement would need a long-span rope drive with about twice the free rope length (more compliance and creep) and a different design from the bench cartridge. v6 uses the same cartridge for all ten joints. The cost is swing-leg inertia, at the home pose excluding armature:

| Inertia, kg m² | v5 | v6 |
| --- | ---: | ---: |
| Leg about hip pitch | 0.035 | 0.046 (+31%) |
| Leg about hip roll | 0.034 | 0.045 (+32%) |
| Shank + foot about knee | 0.0095 | 0.0130 (+37%) |

Each joint also carries 0.0096 kg m² of reflected rotor inertia, so the effective increase at the hip is about 25%. In practice the effect is small: `stage2_smooth` (trained on v5) runs on v6 with torques within ±3% per joint (see `training/README.md`, "Model v6").

Hip axis offsets are parameters of the v6 generator: `HIP_ROLL_OFFSET` (roll axis relative to the yaw axis) and `HIP_PITCH_OFFSET` (pitch centre relative to the roll axis), in metres, with x forward, y outward and z up, mirrored for the right leg. The pitch centre stays at 330 mm, so vertical offsets move the yaw and roll axes up into the torso. Both default to zero (coincident axes, as in v5), which produces the committed `dodo_v6.xml`. `make_model_with(HIP_ROLL_OFFSET=..., ...)` builds variants without editing the file; `training/hip_offset_sweep.py` uses it. Hip pitch and yaw are independently driven, so no coupling is modelled.

CAD check: the 83 mm motor can is larger than the 48 mm centre distance, so in side view the motor overlaps the joint. The child link must clear it through the full travel, so the motor and the output drum probably need to sit in different planes.

## v5 changes from v4

| Item | v4 | v5 | Source |
| --- | --- | --- | --- |
| Total mass | ~2.1 kg (motors massless) | 6.84 kg | 10 × 0.45 kg actuator modules (estimate) placed at the joints, 1.6 kg torso |
| Hip spacing | 130 mm | 200 mm | Plan: 180–220 mm |
| Foot | 156 × 36 mm | 160 × 80 mm | Plan: 140–170 × 70–90 mm |
| Hip pitch height / torso top | 365 / 500 mm | 330 / 500 mm | Plan |
| Joint travel | up to 190° | 120° windows | Capstan reference travel |
| Armature | 0.0005 | 0.0096 kg m² | Rotor inertia × 8² (estimate) |
| Actuators | ideal torque motors, ±4 Nm | PD position servos, kp 30, kd 1, ±10 Nm | Plan section 6 policy interface; section 3 sizing |
| Collision | all geoms | feet–floor only | MJX/Warp speed |

Values tagged `ESTIMATE` in the generator are placeholders until actuator modules and links are weighed and bench-characterized (plan milestone 2). Regenerate with `python simulation/models/build_model_v6.py` (or `_v5.py`); it prints a mass/geometry report. Edit the generator, not the XML.
