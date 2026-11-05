"""Measured relative joint excursions; no scene changes on import.

Four fingers and curl/release are independent. Simulation zero is straight.
Initial PIP/DIP are measured; MCP uses a provisional palm reference.
Limits are the fitted observation envelope, NOT certified mechanical stops.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from project_paths import ROOT
MODEL_ID = "real_hand_20260927_initial_pose_v2"
FINGERS = ("index", "middle", "ring", "pinky")
SUFFIXES = ("mcp_flex", "pip", "dip")
ROBOT_ROOT = "/World/robotnew_with_aero_hand_right"
CONTROL_MAX = 100.0
DATA_PATH = ROOT / "assets" / "calibration" / "motor_to_joint.json"
_bytes = DATA_PATH.read_bytes()
if hashlib.sha256(_bytes).hexdigest() != (DATA_PATH.parent / "sha256.txt").read_text().strip():
    raise RuntimeError("Measured calibration hash mismatch")
DATA = json.loads(_bytes)
# Explicit provisional mapping, same positive flexion convention as ideal model.
SIGNS = {finger: np.ones(3) for finger in FINGERS}
INITIAL_POSE = json.loads((DATA_PATH.parent / "initial_pose.json").read_text(encoding="utf-8"))


def initial_angles_deg(finger):
    """Initial flexion; MCP reference is provisional, PIP/DIP are measured."""
    raw = np.asarray(DATA['fingers'][finger]['zero_raw_deg'], dtype=float)
    config = INITIAL_POSE['fingers'][finger]
    if config['source_session'] != DATA['fingers'][finger]['source_session']:
        raise ValueError('Initial pose source mismatch: ' + finger)
    mcp = config.get('mcp_initial_override_deg')
    if mcp is None:
        mcp = float(config['palm_reference_deg']) - raw[0]
    angles = (np.array([mcp, -raw[1], -raw[2]]) + 180) % 360 - 180
    if not np.isfinite(angles).all():
        raise ValueError('Non-finite initial pose: ' + finger)
    return angles


HOME_DEG = {finger: SIGNS[finger]*initial_angles_deg(finger) for finger in FINGERS}
COMMAND_RANGE_DEG = (1.0, 179.0)  # common measured domain; never extrapolate
MOTOR_DEG_PER_PROGRESS = (COMMAND_RANGE_DEG[1]-COMMAND_RANGE_DEG[0])/CONTROL_MAX


def motor_command_deg(progress_pct):
    """Equivalent calibrated servo command, never shaft position feedback."""
    p=float(progress_pct)
    if not math.isfinite(p) or not 0<=p<=CONTROL_MAX:
        raise ValueError('progress must be finite and within 0..100')
    return COMMAND_RANGE_DEG[0]+MOTOR_DEG_PER_PROGRESS*p


def joint_targets(finger, phase, progress_pct):
    """Return MCP/PIP/DIP degrees; phase must be supplied by the controller.

Progress 0..100 maps to servo COMMAND 1..179 degrees, not shaft feedback.
No inversion of a fingertip trajectory is involved.
"""
    if finger not in FINGERS or phase not in ("curl", "release"):
        raise ValueError("Specify a measured finger and curl/release phase")
    p = float(progress_pct)
    if not math.isfinite(p) or not 0 <= p <= CONTROL_MAX:
        raise ValueError("progress must be finite and within 0..100")
    u = motor_command_deg(p)
    g = DATA["fingers"][finger]["groups"][phase]
    lo, hi = g["command_range_deg"]
    if not lo <= u <= hi:
        raise ValueError("Command outside measured domain")
    knots = np.asarray(g["command_knots_deg"])
    c = np.asarray(g["interval_coefficients"])
    i = int(np.clip(np.searchsorted(knots, u, side="right") - 1, 0, len(knots)-2))
    dx = u - knots[i]
    q = ((c[0, i]*dx + c[1, i])*dx + c[2, i])*dx + c[3, i]
    return tuple(HOME_DEG[finger] + SIGNS[finger]*q)


def joint_values_rad(commands, phases):
    """Require every finger's command and phase to avoid hidden phase resets."""
    return {"right_%s_%s" % (f, s): math.radians(q)
            for f in FINGERS
            for s, q in zip(SUFFIXES, joint_targets(f, phases[f], commands[f]))}


def measured_limits_deg(finger):
    """Exact cubic extrema over the supported domain, rounded outward to degrees.

Include simulation home, both independent branches, and any small negative
measured excursion. Do not impose the old shared 70/90/45 stops.
"""
    candidates = [HOME_DEG[finger]]
    for phase in ("curl", "release"):
        g = DATA["fingers"][finger]["groups"][phase]
        knots = np.asarray(g["command_knots_deg"])
        c = np.asarray(g["interval_coefficients"])
        for i in range(len(knots)-1):
            low, high = max(1., knots[i]), min(179., knots[i+1])
            if low > high:
                continue
            points = [low, high]
            for j in range(3):
                roots = np.roots([3*c[0,i,j], 2*c[1,i,j], c[2,i,j]])
                points.extend(float(r.real + knots[i]) for r in roots
                              if abs(r.imag) < 1e-9 and low <= r.real+knots[i] <= high)
            candidates.extend(joint_targets(finger, phase, float(np.clip((u-1)/1.78, 0, 100)))
                              for u in points)
    values = np.asarray(candidates)
    return tuple(zip(np.floor(values.min(axis=0)), np.ceil(values.max(axis=0))))


LIMITS_BY_FINGER_DEG = {f: measured_limits_deg(f) for f in FINGERS}


def scene_joints(stage):
    from pxr import UsdPhysics, PhysxSchema
    wanted = {"right_%s_%s" % (f, s) for f in FINGERS for s in SUFFIXES}
    found = {}
    for prim in stage.TraverseAll():
        if str(prim.GetPath()).startswith(ROBOT_ROOT + "/") and prim.GetName() in wanted:
            if prim.GetName() in found or not prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError("Ambiguous/non-revolute finger joint: " + str(prim.GetPath()))
            found[prim.GetName()] = prim
    if set(found) != wanted:
        raise RuntimeError("Missing finger joints: " + str(sorted(wanted - set(found))))
    for finger in FINGERS:
        root = found["right_%s_mcp_flex" % finger]
        if PhysxSchema.PhysxMimicJointAPI.GetAll(root):
            raise RuntimeError("MCP must be the independent drive: " + str(root.GetPath()))
        for suffix in SUFFIXES:
            prim = found["right_%s_%s" % (finger, suffix)]
            if len(PhysxSchema.PhysxMimicJointAPI.GetAll(prim)) > 1:
                raise RuntimeError("Multiple mimic constraints: " + str(prim.GetPath()))
    return found


def audit_scene(stage, check_initial=False):
    from pxr import UsdPhysics, PhysxSchema
    joints = scene_joints(stage)
    for finger in FINGERS:
        root = joints["right_%s_mcp_flex" % finger]
        for suffix, (lower, upper) in zip(SUFFIXES, LIMITS_BY_FINGER_DEG[finger]):
            prim = joints["right_%s_%s" % (finger, suffix)]
            joint = UsdPhysics.RevoluteJoint(prim)
            lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            if lo is None or hi is None or not np.allclose([lo, hi], [lower, upper], atol=1e-4, rtol=0):
                raise RuntimeError("REAL_HAND_SETUP_REQUIRED_LIMIT_" + prim.GetName())
            if PhysxSchema.PhysxMimicJointAPI.GetAll(prim):
                raise RuntimeError("REAL_REMOVE_OLD_MIMIC_RUN_SETUP_" + prim.GetName())
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive or float(drive.GetStiffnessAttr().Get() or 0) <= 0:
                raise RuntimeError("REAL_INDEPENDENT_DRIVE_REQUIRED_" + prim.GetName())
            if prim.GetCustomDataByKey("realHandModel") != MODEL_ID:
                raise RuntimeError("REAL_STAGED_SETUP_REQUIRED_" + prim.GetName())
            if check_initial:
                expected = HOME_DEG[finger][SUFFIXES.index(suffix)]
                state = PhysxSchema.JointStateAPI.Get(prim, "angular")
                if (not state or not np.allclose(
                        [drive.GetTargetPositionAttr().Get(), state.GetPositionAttr().Get(),
                         state.GetVelocityAttr().Get()], [expected, expected, 0], atol=1e-4, rtol=0)):
                    raise RuntimeError("REAL_INITIAL_POSE_MISMATCH_" + prim.GetName())
    return joints


def remove_mimic_constraints(prim, mimic_api_type):
    """Remove multiple-apply instances without schema-wrapper GetName().

    Isaac's Python PhysxMimicJointAPI binding does not expose GetName on
    every version. USD's composed applied-schema tokens contain the instance
    name; snapshot them because RemoveAPI changes that list.
    """
    prefix = "PhysxMimicJointAPI:"
    instances = [str(token)[len(prefix):] for token in prim.GetAppliedSchemas()
                 if str(token).startswith(prefix)]
    for instance in instances:
        if not instance or not prim.RemoveAPI(mimic_api_type, instance):
            raise RuntimeError("Unable to remove old mimic: %s (%s)" %
                               (prim.GetPath(), instance))
    remaining = [str(token) for token in prim.GetAppliedSchemas()
                 if str(token).startswith(prefix)]
    if remaining:
        raise RuntimeError("Old mimic remains after removal: %s (%s)" %
                           (prim.GetPath(), remaining))


def configure_scene(stage, timeline):
    """Explicit user-run setup only, stopped timeline, reversible session edits."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    if not timeline.is_stopped():
        raise RuntimeError("Press STOP (not Pause), run hand setup, then Play and rerun a runtime explicitly integrated with real_hand_model.py")
    joints = scene_joints(stage)  # Preflight all targets before the first write.
    from demo_policy import POLICY, SIMULATION_ONLY
    if not SIMULATION_ONLY:
        raise RuntimeError("Measured demo numerical profile requires simulation-only policy")
    articulation_roots = [prim for prim in stage.TraverseAll()
                          if str(prim.GetPath()).startswith(ROBOT_ROOT + "/")
                          and prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(articulation_roots) != 1:
        raise RuntimeError("Expected one actual robot articulation root: " +
                           str([str(prim.GetPath()) for prim in articulation_roots]))
    position_iterations = int(POLICY["solver_position_iterations"])
    velocity_iterations = int(POLICY["solver_velocity_iterations"])
    if not 1 <= position_iterations <= 255 or not 1 <= velocity_iterations <= 255:
        raise RuntimeError("Invalid simulation solver iteration profile")
    physics_hz = POLICY.get('physics_step_hz', 120)
    if isinstance(physics_hz, bool) or physics_hz not in (30, 60, 120):
        raise RuntimeError('Simulation physics step must be 30, 60 or 120 Hz')
    physics_scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(physics_scenes) != 1:
        raise RuntimeError('Expected one simulation physics scene')
    gains = {}
    for finger in FINGERS:
        drive = UsdPhysics.DriveAPI.Get(joints["right_%s_mcp_flex" % finger], "angular")
        if not drive:
            raise RuntimeError("Missing MCP drive: " + finger)
        stiffness = float(drive.GetStiffnessAttr().Get() or 0)
        damping = float(drive.GetDampingAttr().Get() or 0)
        force = float(drive.GetMaxForceAttr().Get())
        if not all(math.isfinite(v) for v in (stiffness, damping, force)) or stiffness <= 0 or damping < 0 or force <= 0:
            raise RuntimeError("Finite positive MCP drive stiffness/maxForce required: " + finger)
        gains[finger] = (stiffness, damping, force, drive.GetTypeAttr().Get())
    layer = stage.GetSessionLayer()
    original = layer.ExportToString()
    try:
        with Usd.EditContext(stage, layer):
            # Set the actual articulation root, rather than its enclosing
            # Xform. A fixed setup profile improves loaded joint following;
            # no depth/pressure feedback changes these settings at runtime.
            solver_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_roots[0])
            solver_api.CreateSolverPositionIterationCountAttr(position_iterations)
            solver_api.CreateSolverVelocityIterationCountAttr(velocity_iterations)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scenes[0])
            scene_api.CreateTimeStepsPerSecondAttr(int(physics_hz))
            for finger in FINGERS:
                root = joints["right_%s_mcp_flex" % finger]
                for suffix, (lower, upper) in zip(SUFFIXES, LIMITS_BY_FINGER_DEG[finger]):
                    prim = joints["right_%s_%s" % (finger, suffix)]
                    joint = UsdPhysics.RevoluteJoint(prim)
                    joint.CreateLowerLimitAttr(float(lower))
                    joint.CreateUpperLimitAttr(float(upper))
                    # Remove the fixed constraints BEFORE installing independent
                    # drive targets. Session layer only; original USD is retained.
                    remove_mimic_constraints(prim, PhysxSchema.PhysxMimicJointAPI)
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    initial = float(HOME_DEG[finger][SUFFIXES.index(suffix)])
                    drive.CreateTargetPositionAttr(initial)
                    drive.CreateTargetVelocityAttr(0.0)
                    prim.SetCustomDataByKey("realHandModel", MODEL_ID)
                    state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                    state.CreatePositionAttr(initial)
                    state.CreateVelocityAttr(0.0)
                    if suffix != "mcp_flex":
                        stiffness, damping, force, drive_type = gains[finger]
                        drive.CreateStiffnessAttr(stiffness)
                        drive.CreateDampingAttr(damping)
                        drive.CreateMaxForceAttr(force / 3.0)
                        drive.CreateTypeAttr(drive_type)
                print("[REAL HAND DRIVE]", finger, "MCP unchanged; PIP/DIP maxForce each=", gains[finger][2] / 3.0)
            audit_scene(stage, check_initial=True)
            if (solver_api.GetSolverPositionIterationCountAttr().Get() != position_iterations
                    or solver_api.GetSolverVelocityIterationCountAttr().Get() != velocity_iterations
                    or scene_api.GetTimeStepsPerSecondAttr().Get() != physics_hz):
                raise RuntimeError("Actual articulation numerical setup did not apply")
            print('[DEMO PHYSICS STEP] hz=', physics_hz,
                  'session_only=True; geometry_collision_and_tracking_checks_retained=True')
            print("[DEMO ARTICULATION SOLVER] root=", articulation_roots[0].GetPath(),
                  "position_iterations=", position_iterations,
                  "velocity_iterations=", velocity_iterations,
                  "session_only=True; gains_and_force_caps_unchanged=True")
    except Exception:
        layer.ImportFromString(original)
        raise
    return original


