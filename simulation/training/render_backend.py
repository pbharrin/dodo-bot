"""Pick a MuJoCo OpenGL backend before `mujoco` is imported.

Headless GPU machines often lack the EGL/GL system libraries. With MUJOCO_GL=egl,
`import mujoco` then fails outright, which would stop a training run before it
starts. `setup()` probes EGL in a subprocess and falls back to MUJOCO_GL=disable
(no rendering; physics and training are unaffected).

Must be called before anything imports mujoco, brax or mujoco_playground.
"""
import os
import subprocess
import sys

_PROBE = ("import mujoco; "
          "mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8).close()")


def setup() -> bool:
    """Set MUJOCO_GL if the user hasn't. Returns True if rendering is available."""
    if 'MUJOCO_GL' in os.environ:
        return os.environ['MUJOCO_GL'].lower() not in ('disable', 'disabled', 'off', 'false', '0')
    if sys.platform == 'darwin':
        return True                                   # default (CGL/GLFW) works on macOS
    env = dict(os.environ, MUJOCO_GL='egl')
    ok = subprocess.run([sys.executable, '-c', _PROBE], env=env,
                        capture_output=True).returncode == 0
    if ok:
        os.environ['MUJOCO_GL'] = 'egl'
    else:
        os.environ['MUJOCO_GL'] = 'disable'
        print('WARNING: EGL rendering unavailable; continuing with MUJOCO_GL=disable (no videos).\n'
              '  To enable: sudo apt-get install -y libegl1 libgl1 libopengl0 ffmpeg', flush=True)
    return ok
