# CAD conventions

Status: directory prepared; no validated part geometry exists yet.

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

Body X forward, Y left, Z up. Plan five actuators per leg (ten total): hip roll, hip pitch, hip yaw, knee pitch and ankle/hock pitch. In the neutral pose, roll is about X, pitch about Y and yaw about Z; downstream axes follow their parent transforms. Confirm hip axis order and signs consistently for mirrored legs. The user identifies two motors for hip pitch/yaw; verify whether their transmission is independent or coupled before defining assembly joints. For a differential, retain separate motor and joint coordinates and document their position/velocity/torque mappings.

The floor plane is Z = 0 in the nominal standing layout. The top of the torso is Z = 500 mm. Head/neck are additional.

Use stable part identifiers, for example `DB-ACT-001_motor_mount.step` and `DB-HIP-001_roll_yoke.step`. Track revisions with Git and assembly manifests. Every exported part should record source file, revision, material, intended print orientation, mass estimate and critical fits.

## First models

1. Skeleton layout and exact purchased-hardware envelopes.
2. Reusable actuator cartridge and bench fixture.
3. Hip roll support and two-motor hip pitch/yaw assembly.
4. One leg and its foot.
5. Pelvis and electronics supports.
6. Lightweight cosmetic shell.

Import and inspect referenced capstan geometry before specifying interfaces. Verify solid validity, units, assembly fit, joint travel, collision clearance, bearing/shaft stack-up, tool access, cable bend paths and printed tolerances before releasing a part for manufacture.

Upstream reference: https://github.com/aaedmusa/Capstan-Drive
