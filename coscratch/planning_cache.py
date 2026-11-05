"""Fail-closed V104 acceleration data for the Isaac scratch planner.

Only physically successful runs become exact cache entries. Loaded trajectories
remain untrusted until the production validator accepts them again. Neighbour
entries expose endpoint joint hints only, never old acceptance decisions.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = 1
BUILD = "V104-EXACT-AND-NEIGHBOUR-CACHE-v1"
CACHE_DIRECTORY = "v104_planning_cache"
MODEL_FILES = (
    "v104.py", "controller.py", "contact_wave.py", "hand_model.py",
    "base_guard.py", "depth_control.py", "palm_policy.py", "release_policy.py",
    "joint_search.py", "deformation_monitor.py",
    "setup_deformation_monitor.py", "planning_cache.py",
    "dense_ik_robot_descriptor.yaml", "robotnew_with_aero_hand_right_lula.urdf",
    "robotnew_with_aero_hand_right_isaac.urdf",
    "finger_scratch_arcs_v102_staged.csv", "dense_back_candidates_5mm.csv",
)


def _encode(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, cKDTree):
        return {"__kdtree__": np.asarray(value.data, dtype=np.float64).tolist()}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("NONFINITE_CACHE_VALUE")
        return value
    raise TypeError("UNSUPPORTED_CACHE_VALUE:" + type(value).__name__)


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            return np.asarray(value["__ndarray__"], dtype=np.float64)
        if set(value) == {"__kdtree__"}:
            return cKDTree(np.asarray(value["__kdtree__"], dtype=np.float64))
        return {key: _decode(item) for key, item in value.items()}
    return value


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_fingerprint(project_dir):
    root = Path(project_dir)
    digest = hashlib.sha256()
    for name in MODEL_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def surface_fingerprint(points, normals):
    digest = hashlib.sha256()
    for array in (points, normals):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _cache_path(project_dir, candidate_id):
    return Path(project_dir) / CACHE_DIRECTORY / ("candidate_%06d.json" % int(candidate_id))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_encode(value), stream, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_physical_success(project_dir, *, candidate_id, target_xyz, target_normal,
                          back_points, back_normals, saved_execution):
    """Persist one run only after the batch bridge declares dataset_pass."""
    trajectory = saved_execution.get("trajectory") if isinstance(saved_execution, dict) else None
    if not trajectory:
        raise ValueError("EMPTY_SUCCESS_TRAJECTORY")
    endpoint_q = np.asarray(saved_execution.get("endpoint_q"), dtype=np.float64)
    if endpoint_q.shape != (5,) or not np.all(np.isfinite(endpoint_q)):
        raise ValueError("INVALID_SUCCESS_ENDPOINT")
    first = trajectory[0]
    payload = {
        "schema": SCHEMA,
        "build": BUILD,
        "candidate_id": int(candidate_id),
        "created_wall_s": time.time(),
        "model_fingerprint": model_fingerprint(project_dir),
        "surface_fingerprint": surface_fingerprint(back_points, back_normals),
        "target_xyz_m": np.asarray(target_xyz, dtype=np.float64),
        "target_normal": np.asarray(target_normal, dtype=np.float64),
        "start_q": np.asarray(first["q"], dtype=np.float64),
        "start_fingers": dict(first["fingers"]),
        "endpoint_q": endpoint_q,
        "execution": saved_execution,
        "physical_success": True,
    }
    destination = _cache_path(project_dir, candidate_id)
    _atomic_json(destination, payload)
    return destination


def load_exact(project_dir, *, candidate_id, target_xyz, target_normal,
               back_points, back_normals, actual_q, actual_fingers,
               arm_tolerance_deg=1.0, finger_tolerance_pct=1.0):
    """Load structurally matching data; caller must run full validation."""
    path = _cache_path(project_dir, candidate_id)
    if not path.is_file():
        return None, "MISS"
    try:
        payload = _decode(json.loads(path.read_text(encoding="utf-8")))
        if (payload.get("schema") != SCHEMA or payload.get("build") != BUILD
                or payload.get("physical_success") is not True):
            return None, "SCHEMA_OR_SUCCESS_MISMATCH"
        if payload.get("model_fingerprint") != model_fingerprint(project_dir):
            return None, "MODEL_CHANGED"
        if payload.get("surface_fingerprint") != surface_fingerprint(back_points, back_normals):
            return None, "SURFACE_CHANGED"
        if int(payload.get("candidate_id", -1)) != int(candidate_id):
            return None, "TARGET_ID_CHANGED"
        if not np.allclose(payload["target_xyz_m"], target_xyz, atol=1e-9, rtol=0):
            return None, "TARGET_XYZ_CHANGED"
        if not np.allclose(payload["target_normal"], target_normal, atol=1e-8, rtol=0):
            return None, "TARGET_NORMAL_CHANGED"
        actual_q = np.asarray(actual_q, dtype=np.float64)
        if actual_q.shape != (5,) or np.max(np.abs(np.degrees(actual_q-payload["start_q"]))) > arm_tolerance_deg:
            return None, "ARM_START_CHANGED"
        for finger, expected in payload["start_fingers"].items():
            if finger not in actual_fingers or abs(float(actual_fingers[finger])-float(expected)) > finger_tolerance_pct:
                return None, "FINGER_START_CHANGED"
        execution = payload.get("execution")
        if not isinstance(execution, dict) or not execution.get("trajectory"):
            return None, "EMPTY_EXECUTION"
        return payload, "HIT_REQUIRES_FULL_REVALIDATION"
    except Exception as exc:
        return None, "INVALID:" + type(exc).__name__ + ":" + str(exc)


def _normal_angle_deg(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("inf")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def _historic_success_seeds(data_root):
    ledger = Path(data_root) / "simulation" / "success.jsonl"
    if not ledger.is_file():
        return []
    rows = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            result = json.loads(line)
            if result.get("dataset_pass") is not True or result.get("reason") != "COMPLETED":
                continue
            plan = Path(result["attempt"]) / "scratch_v103_last_plan.csv"
            with plan.open("r", encoding="utf-8-sig", newline="") as stream:
                record = next(csv.DictReader(stream))
            q = np.radians([float(record["j%d_deg" % index]) for index in range(1, 6)])
            xyz = np.asarray(result.get("xyz_m"), dtype=np.float64)
            normal = np.asarray(result.get("normal"), dtype=np.float64)
            if xyz.shape == (3,) and normal.shape == (3,) and np.all(np.isfinite(q)):
                rows.append((xyz, normal, q, "historic_physical_success"))
        except Exception:
            continue
    return rows


def neighbour_endpoint_seeds(project_dir, data_root, target_xyz, target_normal,
                             max_distance_m=0.035, max_normal_angle_deg=25.0, limit=4):
    """Return untrusted q hints only, ordered by target/normal similarity."""
    rows = _historic_success_seeds(data_root)
    cache_root = Path(project_dir) / CACHE_DIRECTORY
    if cache_root.is_dir():
        for path in cache_root.glob("candidate_*.json"):
            try:
                payload = _decode(json.loads(path.read_text(encoding="utf-8")))
                if payload.get("physical_success") is True:
                    rows.append((np.asarray(payload["target_xyz_m"]),
                                 np.asarray(payload["target_normal"]),
                                 np.asarray(payload["endpoint_q"]), "v104_exact_cache"))
            except Exception:
                continue
    ranked = []
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    for xyz, normal, q, source in rows:
        distance = float(np.linalg.norm(np.asarray(xyz)-target_xyz))
        angle = _normal_angle_deg(normal, target_normal)
        if distance <= float(max_distance_m) and angle <= float(max_normal_angle_deg):
            ranked.append((distance, angle, np.asarray(q, dtype=np.float64), source))
    ranked.sort(key=lambda row: (row[0], row[1]))
    unique, answer = set(), []
    for distance, angle, q, source in ranked:
        key = tuple(np.round(q, 6))
        if key in unique:
            continue
        unique.add(key)
        answer.append({"distance_m": distance, "normal_angle_deg": angle,
                       "q": q, "source": source})
        if len(answer) >= int(limit):
            break
    return answer
