"""Generate the Dodo v5 MuJoCo model for learned-policy training.

Run:  python simulation/models/build_model_v5.py
Writes dodo_v5.xml (floating base, floor included) next to this file.

v5 differs from v4 (which is kept unchanged for the classical experiments):

* Geometry follows docs/project-plan.md section 2: 500 mm torso top, ~330 mm hip
  pitch centre, 200 mm hip spacing, 160 x 80 mm foot, 70 mm hock-to-sole.
* Every actuator module has mass (ACTUATOR_MODULE_MASS) at an assumed location,
  so total mass lands in the plan's 6-10 kg range instead of ~2 kg.
* Rotor inertia is reflected through the 8:1 capstan as joint armature.
* Actuators are joint-level PD position servos with a torque clamp, matching the
  "bounded joint-position targets with validated local gains" plan (section 6).
* Joint travel is a 120 degree window per joint (capstan reference travel).
* Collision is reduced to foot-floor only, for MJX / MJX-Warp speed. Leg
  crossing is discouraged by a reward term instead of collision geometry.

Every number tagged ESTIMATE below is a placeholder for a bench measurement
(milestone 2). Replace it when the actuator and leg hardware are weighed and
characterised, regenerate, and retrain.

qpos = 0 is the neutral bird stance: the neutral joint angles are baked into the
body frames (same convention as v4), so a zero action is the nominal pose.
"""
from math import cos, sin, radians, pi
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dodo_v5.xml'

# ---------------------------------------------------------------- geometry (m)
TORSO_TOP_Z = 0.500            # requirement
HIP_Z = 0.330                  # plan: hip pitch centre ~330 mm above floor
HIP_OFFSET_Y = 0.100           # plan: 180-220 mm hip spacing -> 200 mm
TORSO_HALF = (0.140, 0.120, 0.100)   # plan bounding box 280 x 240 x 200 mm
THIGH_LENGTH = 0.130           # plan: upper leg 120-150 mm
SHANK_LENGTH = 0.147           # plan: shank 140-170 mm (solved for HIP_Z)
HOCK_TO_SOLE = 0.070           # plan: distal foot / hock-to-sole 60-100 mm
FOOT_LENGTH = 0.160            # plan: contact patch 140-170 mm long
FOOT_WIDTH = 0.080             # plan: 70-90 mm wide
FOOT_THICKNESS = 0.012         # TPU sole + plate
FOOT_FORWARD = 0.030           # sole centre ahead of the hock (x, foot frame)

# Neutral (bird) stance. Cumulative pitch 25 - 40 + 15 = 0, so the sole is flat.
HIP_PITCH_DEG = 25.0
KNEE_PITCH_DEG = -40.0
ANKLE_PITCH_DEG = 15.0

# 120 degree travel windows about neutral (capstan reference travel; joint
# clocking is not yet designed, so the windows are centred). Hip yaw and roll
# use less than the full window.
JOINT_RANGE_DEG = {
    'hip_yaw': (-30, 30),
    'hip_roll': (-25, 25),
    'hip_pitch': (-60, 60),
    'knee_pitch': (-60, 60),
    'ankle_pitch': (-60, 60),
}
JOINTS = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')

# ---------------------------------------------------------------- mass (kg)
ACTUATOR_MODULE_MASS = 0.45    # ESTIMATE: 8308 motor + drums + bearings + driver
ACTUATOR_RADIUS = 0.045        # 83 mm motor can, rough envelope
ACTUATOR_HALF_LEN = 0.022
TORSO_MASS = 1.60              # ESTIMATE: frame, battery, UNO Q, IMU, wiring
THIGH_LINK_MASS = 0.12         # ESTIMATE: printed link + hardware
SHANK_LINK_MASS = 0.10         # ESTIMATE
FOOT_MASS = 0.15               # ESTIMATE: distal foot, sole, contact sensor
# Where the knee and ankle motors sit, as a fraction of the link from its
# proximal joint. The plan wants heavy parts near the pelvis but also prefers
# direct-mounted actuators, so these are deliberately exposed for study.
KNEE_MOTOR_ALONG_THIGH = 0.30  # ESTIMATE
ANKLE_MOTOR_ALONG_SHANK = 0.30 # ESTIMATE

# ---------------------------------------------------------------- actuation
GEAR_RATIO = 8.0
ROTOR_INERTIA = 1.5e-4         # ESTIMATE kg m^2, outrunner bell ~100 g at r~38 mm
ARMATURE = ROTOR_INERTIA * GEAR_RATIO ** 2   # reflected to the joint
JOINT_DAMPING = 0.05           # ESTIMATE N m s/rad, capstan + bearings
JOINT_FRICTIONLOSS = 0.05      # ESTIMATE N m, coulomb friction at the joint
TORQUE_LIMIT = 10.0            # N m, plan section 3 sizing target (~16 A)
KP = 30.0                      # N m/rad joint-level stiffness (starting point)
KD = 1.0                       # N m s/rad

TIMESTEP = 0.004               # physics step; the policy runs at 50 Hz

# ---------------------------------------------------------------- derived
TORSO_CENTER_Z = TORSO_TOP_Z - TORSO_HALF[2]          # torso body origin (world)
HIP_OFFSET_Z = HIP_Z - TORSO_CENTER_Z                 # hip relative to torso


def neutral_leg_drop() -> float:
    """Vertical hip-to-sole distance in the neutral pose."""
    a1 = radians(HIP_PITCH_DEG)
    a2 = radians(HIP_PITCH_DEG + KNEE_PITCH_DEG)
    return THIGH_LENGTH * cos(a1) + SHANK_LENGTH * cos(a2) + HOCK_TO_SOLE


def quat_y(degrees: float) -> str:
    a = radians(degrees) / 2.0
    return f'{cos(a):.9f} 0 {sin(a):.9f} 0'


def motor_geom(name: str, pos: str, axis: str = 'y') -> str:
    """Visual-only cylinder carrying an actuator module's mass."""
    zaxis = {'x': '1 0 0', 'y': '0 1 0', 'z': '0 0 1'}[axis]
    return (f'<geom name="{name}" class="visual" type="cylinder" pos="{pos}" '
            f'zaxis="{zaxis}" size="{ACTUATOR_RADIUS} {ACTUATOR_HALF_LEN}" '
            f'mass="{ACTUATOR_MODULE_MASS}" rgba="0.25 0.25 0.28 1"/>')


def leg(p: str, y: float) -> str:
    knee_z = -KNEE_MOTOR_ALONG_THIGH * THIGH_LENGTH
    ankle_z = -ANKLE_MOTOR_ALONG_SHANK * SHANK_LENGTH
    sole_z = -HOCK_TO_SOLE + FOOT_THICKNESS / 2
    r = JOINT_RANGE_DEG
    return f'''
      <body name="{p}_hip_yaw" pos="0 {y} {HIP_OFFSET_Z:.6f}">
        <joint name="{p}_hip_yaw" axis="0 0 1" range="{r['hip_yaw'][0]} {r['hip_yaw'][1]}"/>
        {motor_geom(f'{p}_hip_roll_module', '0 0 0.03', 'x')}
        <body name="{p}_hip_roll">
          <joint name="{p}_hip_roll" axis="1 0 0" range="{r['hip_roll'][0]} {r['hip_roll'][1]}"/>
          {motor_geom(f'{p}_hip_pitch_module', f'0 {0.035 if y > 0 else -0.035} 0', 'y')}
          <body name="{p}_thigh" quat="{quat_y(HIP_PITCH_DEG)}">
            <joint name="{p}_hip_pitch" axis="0 1 0" range="{r['hip_pitch'][0]} {r['hip_pitch'][1]}"/>
            <geom name="{p}_thigh_geom" class="visual" type="capsule" fromto="0 0 0 0 0 -{THIGH_LENGTH}" size="0.020" mass="{THIGH_LINK_MASS}" rgba="0.78 0.56 0.28 1"/>
            {motor_geom(f'{p}_knee_module', f'0 0 {knee_z:.6f}', 'y')}
            <body name="{p}_shank" pos="0 0 -{THIGH_LENGTH}" quat="{quat_y(KNEE_PITCH_DEG)}">
              <joint name="{p}_knee_pitch" axis="0 1 0" range="{r['knee_pitch'][0]} {r['knee_pitch'][1]}"/>
              <geom name="{p}_shank_geom" class="visual" type="capsule" fromto="0 0 0 0 0 -{SHANK_LENGTH}" size="0.016" mass="{SHANK_LINK_MASS}" rgba="0.88 0.69 0.35 1"/>
              {motor_geom(f'{p}_ankle_module', f'0 0 {ankle_z:.6f}', 'y')}
              <body name="{p}_foot" pos="0 0 -{SHANK_LENGTH}" quat="{quat_y(ANKLE_PITCH_DEG)}">
                <joint name="{p}_ankle_pitch" axis="0 1 0" range="{r['ankle_pitch'][0]} {r['ankle_pitch'][1]}"/>
                <geom name="{p}_hock_geom" class="visual" type="capsule" fromto="0 0 0 {FOOT_FORWARD} 0 {-HOCK_TO_SOLE + FOOT_THICKNESS}" size="0.014" mass="0.0" rgba="0.43 0.36 0.29 1"/>
                <geom name="{p}_foot" class="foot" type="box" pos="{FOOT_FORWARD} 0 {sole_z:.6f}" size="{FOOT_LENGTH/2} {FOOT_WIDTH/2} {FOOT_THICKNESS/2}" mass="{FOOT_MASS}" rgba="0.43 0.36 0.29 1"/>
                <site name="{p}_foot" pos="{FOOT_FORWARD} 0 {-HOCK_TO_SOLE:.6f}" size="0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>'''


def make_model() -> str:
    actuators = '\n'.join(
        f'    <position name="{s}_{j}" joint="{s}_{j}"/>'
        for s in ('left', 'right') for j in JOINTS)
    sensors = '\n'.join([
        '    <gyro name="gyro" site="imu"/>',
        '    <accelerometer name="accelerometer" site="imu"/>',
        '    <velocimeter name="local_linvel" site="imu"/>',
        '    <framezaxis name="upvector" objtype="site" objname="imu"/>',
        '    <framelinvel name="global_linvel" objtype="site" objname="imu"/>',
        '    <frameangvel name="global_angvel" objtype="site" objname="imu"/>',
    ] + [f'    <framelinvel name="{s}_foot_global_linvel" objtype="site" objname="{s}_foot"/>'
         for s in ('left', 'right')] + [
        f'    <contact name="{s}_foot_floor_found" geom1="{s}_foot" geom2="floor" '
        f'num="1" reduce="mindist" data="found"/>' for s in ('left', 'right')])
    home = f'0 0 {TORSO_CENTER_Z:.6f} 1 0 0 0 ' + ' '.join(['0'] * 10)
    ctrl = ' '.join(['0'] * 10)
    return f'''<mujoco model="dodo_v5">
  <!-- Generated by build_model_v5.py; edit the generator, not this file. -->
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{TIMESTEP}" iterations="2" ls_iterations="8">
    <flag eulerdamp="disable"/>
  </option>
  <default>
    <joint type="hinge" armature="{ARMATURE:.6f}" damping="{JOINT_DAMPING}" frictionloss="{JOINT_FRICTIONLOSS}"/>
    <position kp="{KP}" kv="{KD}" forcerange="-{TORQUE_LIMIT} {TORQUE_LIMIT}" inheritrange="1"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="foot">
      <geom contype="2" conaffinity="1" condim="3" friction="1.0 0.02 0.01"/>
    </default>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.88" rgb2="0.74 0.76 0.80" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" condim="3"/>
    <body name="torso" pos="0 0 {TORSO_CENTER_Z:.6f}">
      <freejoint name="floating_base"/>
      <camera name="track" pos="0.9 -0.9 0.2" xyaxes="0.707 0.707 0 -0.2 0.2 0.96" mode="trackcom"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <geom name="torso_geom" class="visual" type="ellipsoid" size="{TORSO_HALF[0]} {TORSO_HALF[1]} {TORSO_HALF[2]}" mass="{TORSO_MASS}" rgba="0.37 0.57 0.70 0.6"/>
      {motor_geom('left_hip_yaw_module', f'0 {HIP_OFFSET_Y} {HIP_OFFSET_Z + 0.055:.6f}', 'z')}
      {motor_geom('right_hip_yaw_module', f'0 {-HIP_OFFSET_Y} {HIP_OFFSET_Z + 0.055:.6f}', 'z')}
{leg('left', HIP_OFFSET_Y)}
{leg('right', -HIP_OFFSET_Y)}
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
  <sensor>
{sensors}
  </sensor>
  <keyframe>
    <key name="home" qpos="{home}" ctrl="{ctrl}"/>
  </keyframe>
</mujoco>
'''


def report() -> None:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    total = float(m.body_subtreemass[1])
    com = d.subtree_com[1]
    lf = d.site('left_foot').xpos
    print(f'Total mass             {total:.2f} kg')
    print(f'Torso top              {TORSO_TOP_Z:.3f} m, hip z {HIP_Z:.3f} m')
    print(f'Neutral hip-to-sole    {neutral_leg_drop():.4f} m (target {HIP_Z:.3f})')
    print(f'Sole z (left)          {lf[2]:+.4f} m')
    print(f'COM                    x={com[0]:+.4f} y={com[1]:+.4f} z={com[2]:.4f} m')
    print(f'Foot centre x          {lf[0]:+.4f} m, foot spans x '
          f'[{lf[0] - FOOT_LENGTH/2:+.3f}, {lf[0] + FOOT_LENGTH/2:+.3f}]')
    print(f'Armature               {ARMATURE:.4f} kg m^2 per joint')
    print(f'Static single-support hip-roll moment ~ m g (hip_y) = '
          f'{total * 9.81 * HIP_OFFSET_Y:.1f} N m (limit {TORQUE_LIMIT})')


if __name__ == '__main__':
    drop = neutral_leg_drop()
    assert abs(drop - HIP_Z) < 0.003, f'leg drop {drop:.4f} does not reach the floor from {HIP_Z}'
    OUT.write_text(make_model())
    print('Wrote', OUT.relative_to(HERE.parent.parent))
    try:
        report()
    except ImportError:
        print('(install mujoco to print the mass/geometry report)')
