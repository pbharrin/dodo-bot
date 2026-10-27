# Dodo Bot: initial engineering plan

Date: 2026-10-26. Status: proposed baseline; dimensions and ratings below are not production specifications.

## 1. Requirements and interpretation

| Item | Baseline | Status |
| --- | --- | --- |
| Torso-top height | 500 mm from floor in nominal standing pose | User requirement |
| Appearance | Dodo-like rounded torso and folded bird legs | User requirement |
| Joint count | 5 actuators per leg; 10 motors total, including two hip pitch/yaw motors | User correction; detailed transmission arrangement pending |
| Motors | Eaglepower 8308 BLDC, 90 KV | Model identified by user from reference BOM; dimensions/ratings pending |
| Transmission | Approximately 8:1 rope capstan | User requirement |
| Motor control | Makerbase MKS XDrive Mini preferred candidate | User proposes lower-cost ODrive v3.6-based alternative; exact board/firmware to qualify |
| Main control | Prefer UNO Q, with server-trained walking policy | Preference; validation needed |
| Structure | Printed links and housings, metal load-bearing hardware | Proposed implementation |
| CAD exchange | STEP, with editable source retained | Proposed workflow |
| Initial gait | Slow walking on a flat indoor surface, initially tethered | Proposed scope |
| Mass, speed, runtime, payload | Not yet fixed | Must close before detailed design |

The 24.6-second reference video shows a compact biped with folded legs, broad feet, and an elevated head. Frames were inspected across the clip. The video does not establish exact joint centers, actuator specifications, hidden linkage topology, dimensions, or controller design. We should reproduce the visible character and movement goals without treating unobservable mechanics as known.

## 2. Leg architecture

Use five actuators per leg, ten total. The user confirms that the reference has two motors for hip pitch/yaw. The working joint allocation below adds hip yaw to the previous hip roll, hip pitch, knee pitch and ankle/hock pitch layout. This allocation preserves the three in-plane joints; it does not add a second hip pitch joint. Verify the hip axis order and whether the pitch/yaw motors act independently or through a coupled transmission before freezing CAD or simulation kinematics:

| Joint | Axis in neutral body frame | Function |
| --- | --- | --- |
| Hip roll | X, forward | Moves the entire leg plane sideways for lateral foot placement |
| Hip pitch | Local Y, left in neutral pose | Swings the leg forward/backward; one of the two pitch/yaw functions |
| Hip yaw | Local Z, up in neutral pose | Changes leg/foot heading; the other pitch/yaw function |
| Knee pitch | Local Y after the hip rotations | Changes reach and crouch |
| Ankle/hock pitch | Local Y after the hip rotations | Controls distal-foot pitch |

Body convention: X forward, Y left, Z up. Name joints left/right consistently and verify encoder signs on hardware. Hip roll and hip yaw are separate functions. Yaw is now included for heading control; no ankle-roll actuator is planned. Turning and sideways balance still require simulation validation. If hip pitch/yaw uses a differential, derive motor-to-joint position, velocity and torque mappings and enforce the coupled motor limits rather than assuming one motor command equals one joint command.

Bird anatomy needs care: the conspicuous rearward bend is the hock/ankle; the knee is higher and bends forward. For version one, explore a short upper leg partly concealed in the body, a rearward-sloping shank, and an elongated rigid distal-foot assembly ending in a broad sole. That keeps three independent pitch joints without requiring a fourth powered toe joint. The full foot geometry must allow a useful flat contact pose.

Use direct joint-mounted actuators where appropriate, while leaving the hip pitch/yaw transmission topology open until inspected. Compare a remotely driven distal joint only if ankle mass or packaging is unacceptable; remote cables/linkages add compliance and coupling that must be modeled. Keep the battery and as many heavy components as practical near the pelvis. Make the head, beak, and tail lightweight and passive initially.

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
- The 852 g number includes a test-stand implementation; it is not a measured robot joint-module mass. Ten exact copies would be 8.52 kg before the rest of the robot. Weigh a stripped-down module and make an itemized whole-robot mass budget.
- Do not assume KV determines continuous torque. For the selected Eaglepower 8308 90 KV, verify winding resistance, pole count, dimensions, mass, voltage/current guidance, and thermistor provision.
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
| 4 | Hip roll support and pitch/yaw assembly | Three hip rotation functions; verify axis order, pitch/yaw coupling, short offsets, supported load path, service access and cable flex |
| 5 | One complete leg | Upper/shank links, hock mount, broad foot, replaceable TPU sole, contact sensor mount, accessible tensioners |
| 6 | Pelvis and torso frame | Both hips, battery adjustment, electronics cooling, power distribution, IMU datum, lifting/tether points |
| 7 | Cosmetic shell | Removable dodo body/head/tail after balance and mass distribution work |

For the actuator, carry leg loads through output bearings and the housing, not solely through the motor bearings. Prefer double-supported joints where packaging permits. Use steel shafts, proper bearings, bolts, washers, and captured nuts/inserts. Printed structures remain the default; hardware and local metal heat-spreading/load-spreading pieces are not intended to be printed.

Print fit prototypes first. Qualify the structural material with bearing-seat, clamp, and tensioner specimens in the intended print orientation; compare PETG/ASA or suitable nylon options according to printer capability and measured temperatures. Avoid choosing final material from room-temperature strength alone. Check creep, layer separation, bolt relaxation, and repeated impact. Keep hot motor/controller surfaces away from thermally vulnerable printed load paths. TPU is proposed for sole pads and bumpers.

## 5. Initial component inventory

Quantities refer to the full robot; acquire one axis for the first bench milestone.

| Component | Quantity | Selection notes |
| --- | --- | --- |
| Eaglepower 8308 90 KV motors | 10 | Model identified from BOM by user; mounting drawing and ratings to verify |
| Capstan drive modules | 10 | Approximately 8:1; travel and load qualification required |
| Makerbase MKS XDrive Mini | 10 proposed | Preferred candidate; qualify one unit, board revision, firmware and thermal packaging first |
| Motor position sensing | 10 channels | Mini onboard AS5047P aligned with rotor magnet, or a supported external encoder |
| Output absolute encoders | 10 recommended | Joint angle, power-up reference and rope-slip detection; MCU acquisition initially, not assumed dual-encoder support in Mini firmware |
| Motor temperature sensors | 10 recommended | Exact attachment/type to be chosen |
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

### Controller change: Makerbase MKS XDrive Mini

The user reports that S1 availability and cost make it unsuitable for this build and proposes a driver like the MKS XDrive Mini. Treat the **Mini as the preferred prototype candidate**, pending confirmation of the purchased board revision and testing. S1 remains part of the upstream actuator reference, not our planned purchase.

[Makerbase's product page](https://makerbase3d.com/product/makerbase-xdrive-mini-high-precision-brushless-servo-motor-controller-based-on-odrive3-6-with-as5047p-on-board/) identifies the Mini as ODrive v3.6-based with an onboard AS5047P. [Makerbase's firmware directory](https://github.com/makerbase-motor/MKS-ODrive/tree/master/Firmware/MKS%20ODrive%20MINI) contains Mini-specific 0.5.1 firmware dated 2025-03-26. This is a different hardware/software target from S1.

Design implications:

- Use the exact Mini-compatible firmware and matching host tools/API. Record the shipped firmware before changing anything. Do not flash S1 firmware or blindly apply current S1 configuration scripts, calibration procedures, GUI assumptions, or CAN message definitions.
- Use classic CAN as the initial communication target; no CAN-FD capability is assumed. Validate command modes, units, telemetry, watchdog behavior, node IDs and startup behavior against the actual firmware. Check all logical axes/nodes exposed by each board for ID conflicts before combining ten units.
- Retain position/velocity/torque-control capability as a qualification requirement. Measure low-speed current regulation, reversal behavior and holding temperatures with the Eaglepower 8308 90 KV. Controller advertising does not establish continuous joint torque. No advertised high-current rating has been adopted as a validated operating limit.
- The onboard AS5047P needs a centered, appropriately spaced diametrically magnetized rotor magnet. Rework the magnet holder and stationary board carrier around the actual encoder location, cooling and connector clearances. If board placement is impractical, assess a supported external motor encoder before releasing CAD.
- The AS5047P provides single-turn motor angle. At 8:1, one motor revolution is approximately 45 degrees of joint motion, so motor absolute angle alone does not uniquely locate a joint across the planned travel after power loss. Retain an output absolute encoder or a validated homing/reference procedure. Motor sensing also cannot directly observe rope slip.
- Do not assume S1-style dual-encoder feedback on the Mini. Initially plan to read output encoders with the robot MCU for joint reference and discrepancy monitoring; validate latency and firmware support before closing a load-side servo loop.
- Start with one board. Demonstrate repeatable calibration/restarts, encoder quality, CAN command/feedback, stale-command shutdown, controlled loaded reversals, regeneration handling and thermal operation before buying the complete set.
- A [SimpleFOC implementation](https://github.com/simplefoc/mks_xdrive_mini_firmware) exists for this board. Keep it as an alternative development path, not an assumed drop-in firmware replacement; protocol, protection behavior and calibration would need separate validation.

Use a brake-resistor/regeneration arrangement documented for the exact Mini revision and firmware. Give each controller's brake output its own correctly sized resistor when using individual choppers; do not tie independent brake outputs together. Verify dump capacity and configuration under deceleration and with a fully charged battery. A bench supply may not absorb regenerative energy. A tentative 24 V-class bench bus remains a study assumption pending exact motor and controller checks.

## 6. Control architecture

The [UNO Q documentation](https://docs.arduino.cc/resources/datasheets/ABX00162-datasheet.pdf) describes a Linux application processor plus an STM32U585 MCU running Arduino Core on Zephyr, with Bridge communication between them.

Proposed responsibilities:

- Training server: physics simulation, policy training, evaluation, and export of versioned policies.
- UNO Q Linux: onboard policy inference, state-estimation coordination, operator commands, and logging.
- MCU: timestamped sensor acquisition, command validation/limits, communication watchdogs, and CAN scheduling.
- MKS XDrive Mini controllers: encoder-based commutation and fast motor current/servo loops, subject to selected firmware qualification.

Training is remote; walking does not depend on a live Wi-Fi connection. Start with a small proprioceptive policy and test CPU inference before depending on GPU acceleration. Inputs include body orientation/angular rate, joint position/velocity, command velocity, and previous actions; foot contact can augment observations. Start with bounded joint-position targets and validated local gains. Torque policies are a later option after actuator identification.

Initial performance targets, not validated guarantees:

- Policy: 50-100 Hz onboard.
- Local MCU sensor/supervision loop: approximately 500 Hz.
- Ten-axis command and principal feedback traffic: initially approximately 200 Hz, subject to measured bus load and latency.

At 200 Hz, one 8-byte command plus one feedback frame per axis, using about 130 bits/frame for planning, consumes approximately 520 kbit/s for ten axes (52% of a 1 Mbit/s bus before additional traffic). Extra telemetry, stuffing variation, arbitration and retries require headroom. Reassess two buses (one per leg) or lower message rates after measuring the full ten-controller traffic load. Do not assume a single 1 Mbit/s bus can deliver every message for ten axes at 1 kHz. Add a second classic CAN bus if required; the selected Mini path does not assume CAN-FD.

Use the CAN protocol and configuration documented by the selected Makerbase firmware; the modern [ODrive CAN guide](https://docs.odriverobotics.com/v/latest/guides/can-guide.html) is background on bus wiring, not a Mini API specification. Bench-verify UNO Q MCU driver support, transceiver interface, Linux/MCU timing, and end-to-end latency before committing. Hardware CAN capability is not proof that the installed Arduino software exposes a working API. Keep a compatible USB-CAN adapter for initial Linux bench control or a separate real-time CAN bridge as fallback options.

Use an independent hardware stop path and controller watchdogs; demonstrate stale-command handling before supporting robot weight. Loss of power can make a backdrivable robot collapse, so the tether is part of initial integration. Distinguish a planned controlled stop from emergency power removal.

## 7. Simulation and build milestones

1. **Requirements and envelopes:** verify selected motor geometry and ratings, inventory masses for ten actuators, resolve the hip pitch/yaw transmission and axis order, create the five-joint-per-leg kinematic skeleton, and check 120-degree travel feasibility. Exit: a collision-checked nominal stance and candidate step cycle at 500 mm torso height.
2. **One actuator:** characterize output torque/current, backdrive behavior, ratio, reversal lost motion, thermal settling, usable travel, and repeated-cycle rope tension. Exit: documented continuous/peak limits and module mass sufficient for a plausible whole-robot budget.
3. **One leg on a fixture:** verify encoder signs, hard/soft limits, foot trajectories, contact detection, and load-bearing motion. Exit: repeatable stance/swing cycles under representative load without excessive heat, slip, or structural deformation.
4. **Whole robot in simulation:** use measured link mass/inertia, actuator dynamics, torque/speed limits, contact geometry, latency, and rope compliance. Evaluate standing and stepping; randomize uncertain friction, mass, delays and actuation. Exit: robust policy across the intended operating range with realistic limits.
5. **Tethered two-leg robot:** calibrate frames and joint zeros, verify watchdogs, stand with assistance, transfer weight, then step in place and forward. Exit: repeated stable trials without relying on the tether for ordinary support.
6. **Free walking and shell:** add body shell incrementally, remeasure center of mass, update simulation, and retrain/validate. Introduce turning and other surfaces only after flat-floor gait works.

Simple analytical/IK motions come before learned walking. Preserve identical joint ordering, units, coordinate conventions, gains, normalization and action scaling between training and hardware. STEP alone does not encode these; the simulation needs a separate articulated model such as URDF or MJCF.

## 8. Decisions needed before detailed CAD

The user identified the reference BOM motor as Eaglepower 8308 on 2026-10-26; retain the specified 90 KV winding. Highest priority is now verifying its mounting drawing, dimensions, mass, and electrical/thermal ratings against the reference assembly or manufacturer documentation. Next: whole-robot target mass, desired initial walking speed, runtime, exact Mini hardware/firmware revision, intended print material, and CAD authoring tool. Until specified, assume slow indoor walking, passive cosmetic head/tail, no payload, and defer the battery purchase.

## 9. Source provenance

- User correction: the reference has five actuators per leg, including two motors for hip pitch/yaw. Ten total actuators supersedes the earlier eight-actuator interpretation. Detailed hip coupling has not been independently established from the video.


- Makerbase product page and Mini firmware inventory reviewed for the controller change. Firmware repository tree observed at `e15782976ae93d42b1f0648ceec96503141a343b`; the directory includes `MKS_ODrive_MINI_0_5_1_20250326.bin` and `.hex`. Firmware has not been flashed or bench-tested.
- Encoder reference: [ams AS5047P datasheet, manufacturer document hosted by Mouser](https://www.mouser.com/datasheet/2/588/AS5047P_DS000324_2-00-1843440.pdf). Its 14-bit angle is a single-turn measurement, not a multi-turn joint reference.

- Capstan repository reviewed at [9c032fe83c1d9f3edaa0a6492fcbc8ce92e62ecf](https://github.com/aaedmusa/Capstan-Drive/tree/9c032fe83c1d9f3edaa0a6492fcbc8ce92e62ecf).
- Its STEP directory contains drums, base plate, brace, slider, motor magnet holder, ODrive cover, assembly jig, and a zipped test-stand assembly.
- Repository root inspected: no standalone license file was listed. Keep upstream CAD linked as reference until reuse terms are established; do not assign it a new license.
- The user inspected the upstream BOM spreadsheet and reported the motor model as Eaglepower 8308 on 2026-10-26. This identifies the motor family; no unverified dimensional or current ratings have been adopted.
- Upstream STEP filenames were verified, but their solid geometry and BOM spreadsheet have not yet been dimensionally audited by the assistant.
- No third-party CAD or user video is redistributed in this initial commit.
