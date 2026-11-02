# Models

Store MuJoCo XML files and Python model-generation scripts here. Record which model version a controller or experiment expects.

| Model | Generator | Used by |
| --- | --- | --- |
| `dodo_floating.xml`, `dodo_fixed.xml` (v4) | `build_model_v4.py` | Classical experiments in `experiments/` and `controllers/`. Unchanged; archived results reference its SHA-256. |
| `dodo_v5.xml` | `build_model_v5.py` | Learned-policy training in `training/`. |

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

Values tagged `ESTIMATE` in the generator are placeholders until actuator modules and links are weighed and bench-characterized (plan milestone 2). Regenerate with `python simulation/models/build_model_v5.py`; it prints a mass/geometry report. Edit the generator, not the XML.
