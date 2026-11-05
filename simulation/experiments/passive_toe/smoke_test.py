"""Fast CPU-only model and passive spring smoke test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
import mujoco
import build_model_v6_passive_toe as toe

m = mujoco.MjModel.from_xml_string(toe.make_model())
d = mujoco.MjData(m)
assert m.nu == 10, f"Expected ten actuators, got {m.nu}"
assert m.nq == 19 and m.nv == 18, (m.nq, m.nv)
for side in ("left", "right"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_toe_pitch")
    assert jid >= 0
    assert m.jnt_stiffness[jid] > 0
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(100):
    mujoco.mj_step(m, d)
print("PASS: model compiles, 10 actuators, 2 passive toe joints, 100 CPU steps")
