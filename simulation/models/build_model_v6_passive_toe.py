"""Experimental passive toe extension to Dodo v6; no new actuator.

Run from repository root:
    python simulation/models/build_model_v6_passive_toe.py
Generates dodo_v6_passive_toe.xml beside this script.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import build_model_v6 as base

HERE = Path(__file__).resolve().parent
OUT = HERE / "dodo_v6_passive_toe.xml"
# Initial estimates, not measured hardware values.
TOE_LENGTH = 0.050             # m, front 50 mm of the 160 mm sole
TOE_MASS = 0.035               # kg; subtracted from existing foot mass
TOE_STIFFNESS = 0.8            # N m/rad, passive restoring torque
TOE_DAMPING = 0.04             # N m s/rad
TOE_RANGE = "-35 0"            # degrees; negative rotates front toe upward (MuJoCo y-axis)
TOE_LIMIT = 35                 # degrees

def make_model():
    root = ET.fromstring(base.make_model())
    root.set("model", "dodo_v6_passive_toe")
    for side in ("left", "right"):
        foot = root.find(f".//body[@name='{side}_foot']")
        sole = foot.find(f"./geom[@name='{side}_foot']")
        # Preserve rear sole from original x=[-0.05, 0.11] (foot local coordinates).
        # Split at x=0.06, yielding rear length 110 mm and toe length 50 mm.
        rear_start = base.FOOT_FORWARD - base.FOOT_LENGTH / 2
        split = base.FOOT_FORWARD + base.FOOT_LENGTH / 2 - TOE_LENGTH
        rear_length = split - rear_start
        sole.set("pos", f"{(rear_start + split)/2:.6f} 0 {-base.HOCK_TO_SOLE + base.FOOT_THICKNESS/2:.6f}")
        sole.set("size", f"{rear_length/2:.6f} {base.FOOT_WIDTH/2:.6f} {base.FOOT_THICKNESS/2:.6f}")
        sole.set("mass", str(base.FOOT_MASS - TOE_MASS))
        toe = ET.SubElement(foot, "body", name=f"{side}_toe", pos=f"{split:.6f} 0 {-base.HOCK_TO_SOLE:.6f}")
        ET.SubElement(toe, "joint", name=f"{side}_toe_pitch", type="hinge",
                      axis="0 1 0", range=TOE_RANGE, stiffness=str(TOE_STIFFNESS),
                      damping=str(TOE_DAMPING), armature="0", frictionloss="0",
                      springref="0")
        ET.SubElement(toe, "geom", name=f"{side}_toe", type="box", **{
            "class": "foot",
            "pos": f"{TOE_LENGTH/2:.6f} 0 {base.FOOT_THICKNESS/2:.6f}",
            "size": f"{TOE_LENGTH/2:.6f} {base.FOOT_WIDTH/2:.6f} {base.FOOT_THICKNESS/2:.6f}",
            "mass": str(TOE_MASS), "rgba": "0.58 0.42 0.28 1"})
        # Existing foot contact sensor covers rear sole only. Toe contacts
        # must also be included in any future full-body controller diagnostics.
    # The base keyframe qpos has 7 free-base + 10 actuated joints; add toe joints
    # in tree traversal order: left toe after left ankle, right toe after right ankle.
    key = root.find("./keyframe/key")
    q = key.get("qpos").split()
    q.insert(12, "0")  # left toe after 5 left actuated joints
    q.append("0")      # right toe after 5 right actuated joints
    key.set("qpos", " ".join(q))
    return ET.tostring(root, encoding="unicode")

if __name__ == "__main__":
    OUT.write_text(make_model())
    print(f"Wrote {OUT}")
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(OUT))
        print(f"Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
        assert model.nu == 10, "Toe must not add an actuator"
    except ImportError:
        print("Install mujoco to validate model loading")
