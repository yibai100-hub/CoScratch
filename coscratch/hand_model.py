"""Single source of truth for the V102 observed three-stage simulation hand.

Stage angles follow the user's observations; timing is provisional, not measured.
No Isaac imports or scene writes occur on import. The CLI generates separate
model-derived fingertip arcs; it never replaces the legacy measured CSV/URDF.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

MODEL_ID = "v102_staged_tip_root_middle_70_85_45_v1"
FINGERS = ("index", "middle", "ring", "pinky")
LIMITS_DEG = (70.0, 90.0, 45.0)
ENDPOINT_DEG = (70.0, 85.0, 45.0)
CONTROL_MAX = 100.0
# Provisional phase allocation: first 30%, root flexion 30..75%, final PIP 75..100%.
# DIP reaches its stop early in phase II (40% overall). These are tuneable timing
# assumptions, not new measurements. PCHIP is monotone, C1, and does not overshoot.
CURVE_KNOTS = (
    ((0, 30, 75, 100), (0, 0, 70, 70)),
    ((0, 30, 75, 100), (0, 30, 45, 85)),
    ((0, 30, 40, 100), (0, 40, 45, 45)),
)
_CURVES = tuple(PchipInterpolator(x, y) for x, y in CURVE_KNOTS)
SUFFIXES = ("mcp_flex", "pip", "dip")
PALM = "right_base_link"
ROBOT_ROOT = "/World/robotnew_with_aero_hand_right"
ARC_FILENAME = "finger_scratch_arcs_v102_staged.csv"


def joint_targets(progress_pct):
    value = float(progress_pct)
    if not math.isfinite(value) or not 0.0 <= value <= CONTROL_MAX:
        raise ValueError("Finger progress outside 0..100 percent: " + str(value))
    return tuple(float(curve(value)) for curve in _CURVES)


def progress_from_triplet(actual_deg):
    """Fit all three joints, since MCP alone is ambiguous in phases I and III."""
    actual = np.asarray(actual_deg, dtype=float)
    if actual.shape != (3,) or not np.all(np.isfinite(actual)):
        raise ValueError("Invalid MCP/PIP/DIP feedback")
    grid = np.linspace(0, CONTROL_MAX, 201)
    poses = np.column_stack([curve(grid) for curve in _CURVES])
    idx = int(np.argmin(np.sum((poses - actual) ** 2, axis=1)))
    lo, hi = grid[max(0, idx - 1)], grid[min(len(grid) - 1, idx + 1)]
    objective = lambda p: float(np.sum((np.array(joint_targets(p)) - actual) ** 2))
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    progress = min((lo, float(result.x), hi), key=objective)
    return progress, float(np.max(np.abs(np.array(joint_targets(progress)) - actual)))


def joint_values_rad(commands):
    return {
        "right_%s_%s" % (finger, suffix): math.radians(value)
        for finger in FINGERS
        for suffix, value in zip(SUFFIXES, joint_targets(commands.get(finger, 0.0)))
    }


def triplet_errors(actual, commands):
    errors = {}
    for finger in FINGERS:
        values = np.asarray(actual[finger], dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise RuntimeError("Invalid triplet feedback: " + finger)
        errors[finger] = np.abs(values - joint_targets(commands.get(finger, 0.0)))
    return errors


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(xyz="0 0 0", rpy="0 0 0"):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in rpy.split()]).as_matrix()
    matrix[:3, 3] = [float(v) for v in xyz.split()]
    return matrix


def hand_chains(urdf_path):
    root = ET.parse(urdf_path).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    chains = {}
    for finger in FINGERS:
        link = "right_%s_tip_link" % finger
        chain = []
        seen = set()
        while link != PALM:
            if link in seen or link not in parents:
                raise ValueError("Invalid palm-to-tip chain: " + finger)
            seen.add(link)
            joint = parents[link]
            name, kind = joint.get("name"), joint.get("type")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError("Unsupported hand joint: " + name)
            origin = joint.find("origin")
            origin_tf = transform(**({k: origin.get(k, "0 0 0") for k in ("xyz", "rpy")}
                                     if origin is not None else {}))
            axis_el = joint.find("axis")
            axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()])
            if kind == "fixed":
                axis = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(axis) < 1e-8:
                raise ValueError("Invalid axis: " + name)
            chain.append((name, kind, origin_tf, axis / np.linalg.norm(axis)))
            link = joint.find("parent").get("link")
        chain.reverse()
        movable = {n for n, kind, _, _ in chain if kind != "fixed"}
        expected = {"right_%s_%s" % (finger, s) for s in SUFFIXES}
        if movable != expected:
            raise ValueError("Unexpected moving joints in chain: " + finger)
        chains[finger] = chain
    return chains


def tip_local(chain, values):
    matrix = np.eye(4)
    for name, kind, origin, axis in chain:
        matrix = matrix @ origin
        if kind != "fixed":
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            matrix = matrix @ motion
    return matrix[:3, 3].copy()


def generate_arc_csv(urdf_path, output_path):
    output = Path(output_path)
    if output.name != ARC_FILENAME:
        raise ValueError("Only the dedicated V102 arc filename may be generated")
    chains = hand_chains(urdf_path)
    rows = []
    for finger in FINGERS:
        for angle in np.linspace(0.0, CONTROL_MAX, 201):
            values = joint_values_rad({finger: angle})
            point = tip_local(chains[finger], values)
            rows.append([finger, angle, *point, *joint_targets(angle), MODEL_ID])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["finger", "progress_pct", "local_x_m", "local_y_m", "local_z_m",
                         "mcp_deg", "pip_deg", "dip_deg", "model_id"])
        writer.writerows(rows)
    metadata = {"model_id": MODEL_ID, "source": "URDF_FK_SIMULATION_REFERENCE_NOT_MEASURED",
                "limits_deg": LIMITS_DEG, "endpoint_deg": ENDPOINT_DEG,
                "curve_knots": CURVE_KNOTS, "control_unit": "progress_pct", "step_pct": 0.5,
                "urdf_sha256": file_hash(urdf_path), "csv_sha256": file_hash(output),
                "rows": len(rows)}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def verify_arc_csv(path, urdf_path):
    metadata = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata.get("model_id") != MODEL_ID
            or metadata.get("limits_deg") != list(LIMITS_DEG)
            or metadata.get("endpoint_deg") != list(ENDPOINT_DEG)
            or metadata.get("curve_knots") != json.loads(json.dumps(CURVE_KNOTS))
            or metadata.get("control_unit") != "progress_pct"
            or metadata.get("urdf_sha256") != file_hash(urdf_path)
            or metadata.get("csv_sha256") != file_hash(path)):
        raise RuntimeError("V102_ARC_MODEL_MISMATCH: regenerate new arcs; never reuse legacy CSV")
    return metadata


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


def audit_scene(stage):
    from pxr import UsdPhysics, PhysxSchema
    joints = scene_joints(stage)
    for finger in FINGERS:
        root = joints["right_%s_mcp_flex" % finger]
        for suffix, limit in zip(SUFFIXES, LIMITS_DEG):
            prim = joints["right_%s_%s" % (finger, suffix)]
            joint = UsdPhysics.RevoluteJoint(prim)
            lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            if lo is None or hi is None or not np.allclose([lo, hi], [0, limit], atol=1e-4, rtol=0):
                raise RuntimeError("V102_HAND_SETUP_REQUIRED_LIMIT_" + prim.GetName())
            if PhysxSchema.PhysxMimicJointAPI.GetAll(prim):
                raise RuntimeError("V102_REMOVE_OLD_MIMIC_RUN_SETUP_" + prim.GetName())
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive or float(drive.GetStiffnessAttr().Get() or 0) <= 0:
                raise RuntimeError("V102_INDEPENDENT_DRIVE_REQUIRED_" + prim.GetName())
            if prim.GetCustomDataByKey("v102HandModel") != MODEL_ID:
                raise RuntimeError("V102_STAGED_SETUP_REQUIRED_" + prim.GetName())
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
        raise RuntimeError("Press STOP (not Pause), run hand setup, then Play and rerun v102.py")
    joints = scene_joints(stage)  # Preflight all targets before the first write.
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
            for finger in FINGERS:
                root = joints["right_%s_mcp_flex" % finger]
                for suffix, limit in zip(SUFFIXES, LIMITS_DEG):
                    prim = joints["right_%s_%s" % (finger, suffix)]
                    joint = UsdPhysics.RevoluteJoint(prim)
                    joint.CreateLowerLimitAttr(0.0)
                    joint.CreateUpperLimitAttr(limit)
                    # Remove the fixed constraints BEFORE installing independent
                    # drive targets. Session layer only; original USD is retained.
                    remove_mimic_constraints(prim, PhysxSchema.PhysxMimicJointAPI)
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.CreateTargetPositionAttr(0.0)
                    drive.CreateTargetVelocityAttr(0.0)
                    prim.SetCustomDataByKey("v102HandModel", MODEL_ID)
                    for attr_name in ("state:angular:physics:position", "state:angular:physics:velocity"):
                        attr = prim.GetAttribute(attr_name)
                        if attr:
                            attr.Set(0.0)
                    if suffix != "mcp_flex":
                        stiffness, damping, force, drive_type = gains[finger]
                        drive.CreateStiffnessAttr(stiffness)
                        drive.CreateDampingAttr(damping)
                        drive.CreateMaxForceAttr(force / 3.0)
                        drive.CreateTypeAttr(drive_type)
                print("[V102 STAGED DRIVE]", finger, "MCP unchanged; PIP/DIP maxForce each=", gains[finger][2] / 3.0)
            audit_scene(stage)
    except Exception:
        layer.ImportFromString(original)
        raise
    return original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(generate_arc_csv(args.urdf, args.output), indent=2))
