# CAD conventions

Status: v6 master skeleton and estimated hardware envelopes are available. Manufacturing parts are not released.

## Start here

- [Master skeleton report and preview](reports/master_skeleton.md)
- [Combined STEP assembly](assemblies/dodo_v6_master.step)
- [Skeleton-only STEP](assemblies/dodo_v6_skeleton.step)
- [Hardware-envelope STEP](assemblies/dodo_v6_hardware_envelopes.step)
- [Editable Python source](source/master_skeleton.py)
- [Envelope parameters and unresolved measurements](source/envelopes_v6.json)
- [Joint frames and mass manifest](assemblies/dodo_v6_manifest.json)

The source reads `simulation/models/build_model_v6.py` directly. V6's nominal hip offsets remain zero; this first study preserves them exactly. Hip pitch and yaw are independent drives. The 90 × 44 mm cylinders are aggregate simulation allowances, not measured Eaglepower motor or complete capstan housings. The distal drum planes are explicitly marked CAD assumptions.

### Rebuild and verify

Tested with Python 3.12, CadQuery 2.7.0 and MuJoCo 3.17.0:

```bash
python -m pip install -r cad/requirements.txt
python cad/source/master_skeleton.py
python -m unittest discover -s cad/tests -v
```

The full build performs sampled clearance checks and generates STEP assemblies, JSON reports, and a PNG preview. `--skip-clearance` is for fast drafting only and explicitly marks the report unchecked. Source parameters are in millimeters; conversion to simulation meters is explicit.

To study vertical hip offsets, copy `source/envelopes_v6.json` to a separate configuration, set `hip_roll_offset_mm` and `hip_pitch_offset_mm`, and use `--config PATH --output-dir PATH`. For example, `[0, 0, -20]` for each offset places yaw at Z=370, roll at Z=350, and pitch at Z=330 mm in neutral. These are **study parameters, not an accepted packaging solution**. Positive local Y means outward on each leg. Keep roll axes at/inboard of ±100 mm, pitch spacing near 200 mm, and feed the selected physical geometry back into simulation. The source model's hip motor mass locations do not automatically become detailed mounting locations when offsets change.

The report includes a fixed-torso geometric swing path and 5° distal travel samples. These are clearance diagnostics, not a balance controller or proof of clearance across every combined joint pose. Six tests independently compare CAD kinematics/COM with MuJoCo, check the swing path and drum planes, and reimport all three STEP exports to validate solids, units and volumes.

## Authoring and export

Keep editable parametric CAD sources and export STEP for each part and assembly. STEP is the exchange format, not a replacement for feature history, constraints, joint definitions, material assignments, or simulation metadata.

Use millimeters for CAD; simulation uses meters, kilograms, seconds, radians, and N m. Convert explicitly when creating URDF/MJCF and mesh assets.

Proposed layout as geometry is developed:

- `cad/source/`: editable native files or deterministic CAD-generation scripts.
- `cad/step/`: released part STEP exports.
- `cad/assemblies/`: assembly STEP exports and placement metadata.
- `cad/print/`: oriented 3MF/STL exports and print-process notes.
- `simulation/`: robot description, collision geometry, inertial data and actuator parameters.
- `hardware/`: BOM, wiring, encoder mapping and calibration records.

## Coordinate convention

Body X forward, Y left, Z up. Plan five actuators per leg (ten total): hip roll, hip pitch, hip yaw, knee pitch and ankle/hock pitch. In the neutral pose, roll is about X, pitch about Y and yaw about Z; downstream axes follow their parent transforms. Confirm hip axis order and signs consistently for mirrored legs. Hip pitch and yaw are independently driven, one motor per joint, with no differential (resolved in the v6 project plan). The v6 chain is yaw → roll → pitch; preserve its local frames and signs when changing offsets.

The floor plane is Z = 0 in the nominal standing layout. The top of the torso is Z = 500 mm. Head/neck are additional.

Use stable part identifiers, for example `DB-ACT-001_motor_mount.step` and `DB-HIP-001_roll_yoke.step`. Track revisions with Git and assembly manifests. Every exported part should record source file, revision, material, intended print orientation, mass estimate and critical fits.

## First models

1. Skeleton layout and estimated envelopes; replace with exact purchased-hardware geometry as dimensions are verified.
2. Reusable actuator cartridge and bench fixture.
3. Hip roll support and two-motor hip pitch/yaw assembly.
4. One leg and its foot.
5. Pelvis and electronics supports.
6. Lightweight cosmetic shell.

Import and inspect referenced capstan geometry before specifying interfaces. Verify solid validity, units, assembly fit, joint travel, collision clearance, bearing/shaft stack-up, tool access, cable bend paths and printed tolerances before releasing a part for manufacture.

Upstream reference: https://github.com/aaedmusa/Capstan-Drive
