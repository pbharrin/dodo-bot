"""Generate Dodo v4 flat-foot MuJoCo models. Run this file to write dodo_fixed.xml and dodo_floating.xml."""
from math import cos, sin, radians
from pathlib import Path
HERE = Path(__file__).resolve().parent
TORSO_ORIGIN_Z = 0.395
TORSO_CENTER_Z = 0.035
TORSO_RADIUS_Z = 0.070
HIP_OFFSET_Z = -0.030
HIP_OFFSET_Y = 0.065
THIGH_LENGTH = 0.165
SHANK_LENGTH = 0.1915
FOOT_FROM_X = -0.035
FOOT_TO_X = 0.085
FOOT_RADIUS = 0.018
FOOT_HALF_LENGTH = (FOOT_TO_X - FOOT_FROM_X) / 2 + FOOT_RADIUS
FOOT_CENTER_X = (FOOT_FROM_X + FOOT_TO_X) / 2
FOOT_HALF_WIDTH = FOOT_RADIUS
FOOT_HALF_HEIGHT = 0.010
SOLE_PENETRATION = 0.0005
HIP_PITCH_DEG = 25.0
KNEE_PITCH_DEG = -40.0
ANKLE_PITCH_DEG = 15.0
ANKLE_Z = (TORSO_ORIGIN_Z + HIP_OFFSET_Z
           - THIGH_LENGTH * cos(radians(HIP_PITCH_DEG))
           - SHANK_LENGTH * cos(radians(HIP_PITCH_DEG + KNEE_PITCH_DEG)))
FOOT_Z = FOOT_HALF_HEIGHT - SOLE_PENETRATION - ANKLE_Z

def quat_y(degrees: float) -> str:
    a = radians(degrees) / 2.0
    return f'{cos(a):.9f} 0 {sin(a):.9f} 0'

def leg(side: str, y: float) -> str:
    p = side
    return f'''
    <body name="{p}_hip_yaw" pos="0 {y} {HIP_OFFSET_Z}">
      <inertial pos="0 0 0" mass="0.01" diaginertia="0.00001 0.00001 0.00001"/>
      <joint name="{p}_hip_yaw" type="hinge" axis="0 0 1" range="-45 45" damping="0.02"/>
      <body name="{p}_hip_roll">
        <inertial pos="0 0 0" mass="0.01" diaginertia="0.00001 0.00001 0.00001"/>
        <joint name="{p}_hip_roll" type="hinge" axis="1 0 0" range="-30 30" damping="0.02"/>
        <body name="{p}_thigh" quat="{quat_y(HIP_PITCH_DEG)}">
          <joint name="{p}_hip_pitch" type="hinge" axis="0 1 0" range="-75 75" damping="0.03"/>
          <geom name="{p}_thigh_geom" type="capsule" fromto="0 0 0 0 0 -{THIGH_LENGTH}" size="0.018" mass="0.18" rgba="0.78 0.56 0.28 1"/>
          <body name="{p}_shank" pos="0 0 -{THIGH_LENGTH}" quat="{quat_y(KNEE_PITCH_DEG)}">
            <joint name="{p}_knee_pitch" type="hinge" axis="0 1 0" range="-100 90" damping="0.03"/>
            <geom name="{p}_shank_geom" type="capsule" fromto="0 0 0 0 0 -{SHANK_LENGTH}" size="0.014" mass="0.13" rgba="0.88 0.69 0.35 1"/>
            <body name="{p}_foot" pos="0 0 -{SHANK_LENGTH}" quat="{quat_y(ANKLE_PITCH_DEG)}">
              <joint name="{p}_ankle_pitch" type="hinge" axis="0 1 0" range="-60 60" damping="0.02"/>
              <geom name="{p}_foot_geom" type="box" pos="{FOOT_CENTER_X} 0 {FOOT_Z}" size="{FOOT_HALF_LENGTH} {FOOT_HALF_WIDTH} {FOOT_HALF_HEIGHT}" mass="0.08" friction="1.2 0.01 0.001" rgba="0.43 0.36 0.29 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>'''

def make_model(floating: bool) -> str:
    base_joint = '<freejoint name="floating_base"/>' if floating else ''
    actuators = '\n'.join(
        f'    <motor name="{side}_{joint}_motor" joint="{side}_{joint}" gear="1" ctrllimited="true" ctrlrange="-4 4"/>'
        for side in ('left', 'right')
        for joint in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')
    )
    base_qpos = f'0 0 {TORSO_ORIGIN_Z} 1 0 0 0 ' if floating else ''
    key_qpos = base_qpos + ' '.join(['0'] * 10)
    return f'''<mujoco model="dodo_10motor_{"floating" if floating else "fixed"}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint armature="0.0005" limited="true"/>
    <geom contype="1" conaffinity="1" condim="3"/>
  </default>
  <visual><global offwidth="960" offheight="720"/></visual>
  <worldbody>
    <light pos="0 -1 2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.85 0.87 0.9 1" friction="1.2 0.01 0.001"/>
    <body name="torso" pos="0 0 {TORSO_ORIGIN_Z}">
      {base_joint}
      <geom name="torso_geom" type="ellipsoid" pos="0 0 {TORSO_CENTER_Z}" size="0.12 0.085 {TORSO_RADIUS_Z}" mass="1.25" rgba="0.37 0.57 0.70 1"/>
      {leg('left', HIP_OFFSET_Y)}
      {leg('right', -HIP_OFFSET_Y)}
    </body>
  </worldbody>
  <contact>
    <exclude body1="torso" body2="left_thigh"/>
    <exclude body1="torso" body2="right_thigh"/>
  </contact>
  <actuator>
{actuators}
  </actuator>
  <keyframe>
    <key name="bird_stand" qpos="{key_qpos}"/>
  </keyframe>
</mujoco>
'''

if __name__ == '__main__':
    for floating in (False, True):
        path = HERE / ('dodo_floating.xml' if floating else 'dodo_fixed.xml')
        path.write_text(make_model(floating))
        print('Created', path.name)
    print(f'Bird-like neutral pose: hip {HIP_PITCH_DEG:+g}°, knee {KNEE_PITCH_DEG:+g}°, ankle {ANKLE_PITCH_DEG:+g}°')
    print(f'Torso top at rest: {TORSO_ORIGIN_Z + TORSO_CENTER_Z + TORSO_RADIUS_Z:.3f} m')
    hip_z = TORSO_ORIGIN_Z + HIP_OFFSET_Z
    ankle_z = hip_z - THIGH_LENGTH * cos(radians(HIP_PITCH_DEG)) - SHANK_LENGTH * cos(radians(HIP_PITCH_DEG + KNEE_PITCH_DEG))
    sole_z = ankle_z + FOOT_Z - FOOT_HALF_HEIGHT
    print(f'Hip z={hip_z:.3f} m, ankle z={ankle_z:.3f} m, sole z={sole_z:.4f} m')
    ankle_x = -THIGH_LENGTH * sin(radians(HIP_PITCH_DEG)) - SHANK_LENGTH * sin(radians(HIP_PITCH_DEG + KNEE_PITCH_DEG))
    foot_center_x = ankle_x + FOOT_CENTER_X * cos(radians(HIP_PITCH_DEG + KNEE_PITCH_DEG + ANKLE_PITCH_DEG))
    print(f'Estimated foot center X={foot_center_x:.4f} m')
    print(f'Box foot full dimensions: {2*FOOT_HALF_LENGTH:.3f} x {2*FOOT_HALF_WIDTH:.3f} x {2*FOOT_HALF_HEIGHT:.3f} m')
