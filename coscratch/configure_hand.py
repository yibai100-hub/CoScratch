"""Run in Isaac Script Editor with timeline STOPPED, before Play + v102.py.
Only twelve finger joints are changed, in the session layer. No root USD save.
"""
import hashlib
import types
from pathlib import Path
import builtins
import omni.usd
import omni.timeline

project = Path(__file__).resolve().parent
# Compile the exact bytes read here, bypassing cached Script Editor loaders/pyc.
_hand_model_path = project / "setup_hand_model.py"
_hand_model_source = _hand_model_path.read_bytes()
model = types.ModuleType("v102_hand_setup_model")
model.__file__ = str(_hand_model_path)
exec(compile(_hand_model_source, str(_hand_model_path), "exec"), model.__dict__)
print("[V102 HAND SOURCE]", _hand_model_path, "sha256=",
      hashlib.sha256(_hand_model_source).hexdigest(), "loader=direct-source")
model.verify_arc_csv(project / model.ARC_FILENAME, project / "robotnew_with_aero_hand_right_isaac.urdf")
stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_stopped():
    raise RuntimeError("Press STOP, not Pause, before running the hand setup.")
model.scene_joints(stage)
old_runtime = getattr(builtins, "_DOFBOT_SCRATCH_V4_RUNTIME", None)
if old_runtime is not None:
    old_runtime.shutdown()
snapshot = model.configure_scene(stage, timeline)
# Preserve the first pre-setup session snapshot for this stage across reruns.
key = str(stage.GetSessionLayer().identifier)
backups = getattr(builtins, "_v102_hand_setup_backups", {})
backups.setdefault(key, snapshot)
builtins._v102_hand_setup_backups = backups
print("[V102 HAND SETUP OK]", model.MODEL_ID)
print("Staged targets: (0,30,40) -> (70,45,45) -> (70,85,45); hard limits=70/90/45.")
print("Old fixed mimic removed; 12 position drives follow progress 0..100%. Session layer only; no USD saved.")
print("Press PLAY, then reopen/run v102.py. Repeat setup after reopening the stage.")
