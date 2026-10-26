# Dodo Bot: initial engineering plan

Date: 2026-10-26. Status: proposed baseline; dimensions and ratings below are not production specifications.

## 1. Requirements and interpretation

| Item | Baseline | Status |
| --- | --- | --- |
| Torso-top height | 500 mm from floor in nominal standing pose | User requirement |
| Appearance | Dodo-like rounded torso and folded bird legs | User requirement |
| Joint count | 3 in-plane joints plus hip roll per leg; 8 motors total | Interpretation of final user description |
| Motors | Eaglepower BLDC, 90 KV | User requirement; exact SKU needed |
| Transmission | Approximately 8:1 rope capstan | User requirement |
| Motor control | ODrive | User requirement; S1 proposed |
| Main control | Prefer UNO Q, with server-trained walking policy | Preference; validation needed |
| Structure | Printed links and housings, metal load-bearing hardware | Proposed implementation |
| CAD exchange | STEP, with editable source retained | Proposed workflow |
| Initial gait | Slow walking on a flat indoor surface, initially tethered | Proposed scope |
| Mass, speed, runtime, payload | Not yet fixed | Must close before detailed design |

The 24.6-second reference video shows a compact biped with folded legs, broad feet, and an elevated head. Frames were inspected across the clip. The video does not establish exact joint centers, actuator specifications, hidden linkage topology, dimensions, or controller design. We should reproduce the visible character and movement goals without treating unobservable mechanics as known.

## 2. Leg architecture

Use one serial four-joint chain per side:

| Joint | Axis in neutral body frame | Function |
| --- | --- | --- |
| Hip roll | X, forward | Moves the entire leg plane sideways for lateral foot placement |
| Hip pitch | Y, left | Swings the leg forward/backward |
| Knee pitch | Y in the rolled leg frame | Changes reach and crouch |
| Ankle/hock pitch | Y in the rolled leg frame | Controls distal-foot pitch |

Body convention: X forward, Y left, Z up. Name joints left/right consistently and verify encoder signs on hardware. Hip roll is not yaw: this layout does not offer independent toe-heading control. Turning and sideways balance must be demonstrated in simulation; add no yaw or ankle-roll actuator unless testing makes the need clear.

Bird anatomy needs care: the conspicuous rearward bend is the hock/ankle; the knee is higher and bends forward. For version one, explore a short upper leg partly concealed in the body, a rearward-sloping shank, and an elongated rigid distal-foot assembly ending in a broad sole. That keeps three independent pitch joints without requiring a fourth powered toe joint. The full foot geometry must allow a useful flat contact pose.

Begin with direct joint-mounted actuators for clear kinematics. Compare a remotely driven distal joint only if ankle mass or packaging is unacceptable; remote cables/linkages add compliance and coupling that must be modeled. Keep the battery and as many heavy components as practical near the pelvis. Make the head, beak, and tail lightweight and passive initially.

### Starting CAD envelope, not frozen dimensions

| Dimension | Initial study value |
| --- | --- |
| Torso top | 500 mm |
| Hip pitch center above floor | Approximately 330 mm |
| Hip roll/leg center spacing | Approximately 180-220 mm |
| Torso bounding box | Approximately 280 mm long x 240 mm wide x 200 mm high |
| Upper leg center spacing | Explore 120-150 mm |
| Shank center spacing | Explore 140-170 mm |
| Distal foot/hock-to-sole height | Explore 60-100 mm |
| Foot contact patch | Explore 140-170 mm long x 70-90 mm wide |

These ranges are a packaging search space, not a closed kinematic solution. Segment lengths do not add directly to standing height because the leg is folded. Solve joint positions, sole orientation, ground clearance, self-collision, and torso height together. Motor envelope checks may force a wider pelvis or different proportions.

## 3. Actuator feasibility comes first

The [reference project](https://www.aaedmusa.com/projects/capstandrive) reports an 8.55:1 ratio, 120 degrees of output travel, an 852 g test stand, Eagle Power 90 KV motor, and ODrive S1. It also discusses rope creep and chooses Dyneema DM20. Its author reports a difference between the intended drum ratio and measured reduction.

Therefore:

- Use the existing 8.55:1 design as an approximately-8:1 prototype reference. Measure actual ratio and variation through travel rather than assuming pulley outside diameters determine it.
- Review the STEP geometry before choosing bearing sizes, hole patterns, or motor mounting dimensions.
- Treat 120 degrees as a reference travel limit, not a guaranteed usable range for every joint. Clock each joint around its nominal pose, reserve stopping margin, and check walking/crouching travel before freezing drums.
- The 852 g number includes a test-stand implementation; it is not a measured robot joint-module mass. Eight exact copies would be 6.82 kg before the rest of the robot. Weigh a stripped-down module and make an itemized whole-robot mass budget.
- Do not assume KV determines continuous torque. Confirm exact motor SKU, winding resistance, pole count, dimensions, mass, voltage/current guidance, and thermistor provision.
- Characterize low-speed holding and reversal heating without propeller airflow. A drone power rating is not a standing-joint torque rating.

### Preliminary sizing calculation

Using [ODrive's torque/current convention](https://docs.odriverobotics.com/v/latest/manual/control.html):

- Motor torque constant estimate: Kt = 8.27 / 90 = 0.0919 N m/A.
- Joint torque estimate: Tj = efficiency x ratio x Kt x Iq.
- At ratio 8 and assumed efficiency 0.85, Tj is approximately 0.625 N m per amp of motor q-axis current.
- A 10 N m joint output therefore implies approximately 16 A motor current under those assumptions. This is neither a continuous rating nor a recommended current setting.
- Phase/q-axis current is not battery current. Measure DC power and thermal losses separately.

A simple single-support sensitivity check is T = m g d:

| Whole-robot mass | Moment arm 0.10 m | Moment arm 0.15 m |
| --- | --- | --- |
| 6 kg | 5.9 N m | 8.8 N m |
| 8 kg | 7.8 N m | 11.8 N m |
| 10 kg | 9.8 N m | 14.7 N m |

These are illustrative static moments about a joint under full supported weight, not predictions for all joints simultaneously. For actual sizing, calculate joint torques using link gravity, contact forces, the leg Jacobian, and dynamic acceleration. Evaluate hip roll in single support too. Derive peak and RMS demands from candidate gaits and compare with measured actuator torque-speed-temperature limits.

Capstan load example: 10 N m on a 40 mm effective output radius requires a tension difference of 250 N. Actual cable tensions and bearing loads also include pretension and routing. Design anchors, tensioner, shafts, and bearing supports from these loads and measured rope behavior.

## 4. Components to design first

| Order | Assembly / parts | Purpose and required checks |
| --- | --- | --- |
| 1 | Parametric skeleton and hardware envelopes | Joint centers/axes, folded stance, 500 mm torso top, motor clearances, motion ranges, center of mass |
| 2 | Actuator cartridge | Motor mount, small/large drums, rope anchors, adjustable tensioner, paired output-bearing supports, shaft/hub, encoder mounts, stops and guard |
| 3 | Bench test fixture | Rigid mount, known-length torque arm/load-cell interface, travel measurement, temperature sensing, guarded rope path |
| 4 | Hip roll yoke and pitch carrier | Orthogonal axes, short offsets, supported load path, service access and cable flex |
| 5 | One complete leg | Upper/shank links, hock mount, broad foot, replaceable TPU sole, contact sensor mount, accessible tensioners |
| 6 | Pelvis and torso frame | Both hips, battery adjustment, electronics cooling, power distribution, IMU datum, lifting/tether points |
| 7 | Cosmetic shell | Removable dodo body/head/tail after balance and mass distribution work |

For the actuator, carry leg loads through output bearings and the housing, not solely through the motor bearings. Prefer double-supported joints where packaging permits. Use steel shafts, proper bearings, bolts, washers, and captured nuts/inserts. Printed structures remain the default; hardware and local metal heat-spreading/load-spreading pieces are not intended to be printed.

Print fit prototypes first. Qualify the structural material with bearing-seat, clamp, and tensioner specimens in the intended print orientation; compare PETG/ASA or suitable nylon options according to printer capability and measured temperatures. Avoid choosing final material from room-temperature strength alone. Check creep, layer separation, bolt relaxation, and repeated impact. Keep hot motor/controller surfaces away from thermally vulnerable printed load paths. TPU is proposed for sole pads and bumpers.

## 5. Initial component inventory

Quantities refer to the full robot; acquire one axis for the first bench milestone.

| Component | Quantity | Selection notes |
| --- | --- | --- |
| Eaglepower 90 KV motors | 8 | Exact SKU and mounting drawing needed |
| Capstan drive modules | 8 | Approximately 8:1; travel and load qualification required |
| ODrive S1 | 8 proposed | One motor per controller; confirm thermal packaging |
| Motor position sensing | 8 channels | S1 onboard encoder where aligned, otherwise compatible external encoders; appropriate magnets |
| Output absolute encoders | 8 recommended | Actual joint angle, power-up reference, rope-slip/settling detection |
| Motor temperature sensors | 8 recommended | Exact attachment/type to be chosen |
| UNO Q | 1 preferred | 4 GB option is a candidate; runtime benchmark required |
| CAN interface | 1 initial bus | MCU-compatible transceiver, or supported USB-CAN for bring-up |
| IMU | 1 | Rigid pelvis mounting, calibrated frame and timestamped readings |
| Foot contact sensing | 2 assemblies | At least touchdown detection; heel/toe sensing useful |
| Battery | 1, deferred | Voltage/capacity from motor testing and measured duty-cycle energy |
| Logic DC/DC supply | 1 | Regulated supply sized for UNO Q and sensors |
| Power distribution and protection | 1 assembly | Branch protection, main disconnect/E-stop, inrush management, adequate conductors |
| Regeneration handling | As required | Battery charge acceptance and/or suitable dump/clamp hardware |
| Bearings, shafts, fasteners, cable, connectors | Per final CAD/BOM | Locking connectors and strain relief on moving harnesses |
| Fall-arrest test support | 1 | Tether gantry or controlled test fixture |

[ODrive S1](https://docs.odriverobotics.com/v/latest/hardware/s1-datasheet.html) is the starting candidate because it matches the reference actuator and offers materially more current headroom than [ODrive Micro](https://docs.odriverobotics.com/v/latest/hardware/micro-datasheet.html). Neither controller rating establishes the motor's allowable current.

If using S1 brake resistors, do not connect multiple S1 brake outputs to one shared resistor; follow the controller documentation. A bench supply may not absorb regenerative energy. Size the regeneration path for deceleration and a fully charged battery. A tentative 24 V-class bench bus is a study assumption only, pending exact motor and controller checks.

## 6. Control architecture

The [UNO Q documentation](https://docs.arduino.cc/resources/datasheets/ABX00162-datasheet.pdf) describes a Linux application processor plus an STM32U585 MCU running Arduino Core on Zephyr, with Bridge communication between them.

Proposed responsibilities:

- Training server: physics simulation, policy training, evaluation, and export of versioned policies.
- UNO Q Linux: onboard policy inference, state-estimation coordination, operator commands, and logging.
- MCU: timestamped sensor acquisition, command validation/limits, communication watchdogs, and CAN scheduling.
- ODrives: encoder-based commutation and fast motor current/servo loops.

Training is remote; walking does not depend on a live Wi-Fi connection. Start with a small proprioceptive policy and test CPU inference before depending on GPU acceleration. Inputs include body orientation/angular rate, joint position/velocity, command velocity, and previous actions; foot contact can augment observations. Start with bounded joint-position targets and validated local gains. Torque policies are a later option after actuator identification.

Initial performance targets, not validated guarantees:

- Policy: 50-100 Hz onboard.
- Local MCU sensor/supervision loop: approximately 500 Hz.
- Eight-axis command and principal feedback traffic: initially approximately 200 Hz, subject to measured bus load and latency.

At 200 Hz, one 8-byte command plus one feedback frame per axis, using about 130 bits/frame for planning, consumes approximately 416 kbit/s for eight axes. Extra telemetry, stuffing variation, arbitration and retries require headroom. Do not assume a single 1 Mbit/s bus can deliver every message for eight axes at 1 kHz. Add a second bus or validated CAN-FD path if required.

The [CAN guide](https://docs.odriverobotics.com/v/latest/guides/can-guide.html) defines ODrive CANSimple and physical-bus requirements. Bench-verify UNO Q MCU driver support, transceiver interface, Linux/MCU timing, and end-to-end latency before committing. Hardware CAN capability is not proof that the installed Arduino software exposes a working API. Keep USB-CAN for initial Linux bench control or a separate real-time CAN bridge as fallback options.

Use an independent hardware stop path and controller watchdogs; demonstrate stale-command handling before supporting robot weight. Loss of power can make a backdrivable robot collapse, so the tether is part of initial integration. Distinguish a planned controlled stop from emergency power removal.

## 7. Simulation and build milestones

1. **Requirements and envelopes:** identify exact motor, inventory masses, create the kinematic skeleton, and check 120-degree travel feasibility. Exit: a collision-checked nominal stance and candidate step cycle at 500 mm torso height.
2. **One actuator:** characterize output torque/current, backdrive behavior, ratio, reversal lost motion, thermal settling, usable travel, and repeated-cycle rope tension. Exit: documented continuous/peak limits and module mass sufficient for a plausible whole-robot budget.
3. **One leg on a fixture:** verify encoder signs, hard/soft limits, foot trajectories, contact detection, and load-bearing motion. Exit: repeatable stance/swing cycles under representative load without excessive heat, slip, or structural deformation.
4. **Whole robot in simulation:** use measured link mass/inertia, actuator dynamics, torque/speed limits, contact geometry, latency, and rope compliance. Evaluate standing and stepping; randomize uncertain friction, mass, delays and actuation. Exit: robust policy across the intended operating range with realistic limits.
5. **Tethered two-leg robot:** calibrate frames and joint zeros, verify watchdogs, stand with assistance, transfer weight, then step in place and forward. Exit: repeated stable trials without relying on the tether for ordinary support.
6. **Free walking and shell:** add body shell incrementally, remeasure center of mass, update simulation, and retrain/validate. Introduce turning and other surfaces only after flat-floor gait works.

Simple analytical/IK motions come before learned walking. Preserve identical joint ordering, units, coordinate conventions, gains, normalization and action scaling between training and hardware. STEP alone does not encode these; the simulation needs a separate articulated model such as URDF or MJCF.

## 8. Decisions needed before detailed CAD

Highest priority is the exact Eaglepower motor product link or a drawing/photo with dimensions. Next: whole-robot target mass, desired initial walking speed, runtime, available ODrive hardware, intended print material, and CAD authoring tool. Until specified, assume slow indoor walking, passive cosmetic head/tail, no payload, and defer the battery purchase.

## 9. Source provenance

- Capstan repository reviewed at [9c032fe83c1d9f3edaa0a6492fcbc8ce92e62ecf](https://github.com/aaedmusa/Capstan-Drive/tree/9c032fe83c1d9f3edaa0a6492fcbc8ce92e62ecf).
- Its STEP directory contains drums, base plate, brace, slider, motor magnet holder, ODrive cover, assembly jig, and a zipped test-stand assembly.
- Repository root inspected: no standalone license file was listed. Keep upstream CAD linked as reference until reuse terms are established; do not assign it a new license.
- Upstream STEP filenames were verified, but their solid geometry and BOM spreadsheet have not yet been dimensionally audited.
- No third-party CAD or user video is redistributed in this initial commit.
