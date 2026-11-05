# controller.py - near-base progressive palm-tilt trials; frozen V102 remains untouched.
# Run inside Isaac Sim Script Editor while Physics is already PLAYING.
# Run scratch_target_gui.py in a separate Windows Python process.

import builtins
import copy
import csv
import gc
import json
import itertools
import math
import os
import re
import time
import struct
import msvcrt
import hashlib
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import omni.usd
import omni.physx
import omni.timeline
from omni.physx.bindings._physx import ContactEventType

from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysicsSchemaTools, PhysxSchema
from scipy.optimize import minimize
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = str(Path(__file__).resolve().parent)

# Compile the exact bytes read here, bypassing cached Script Editor loaders/pyc.
_hand_model_path = Path(PROJECT_DIR) / "hand_model.py"
_hand_model_source = _hand_model_path.read_bytes()
HAND_MODEL = types.ModuleType("v103_current_hand_model")
HAND_MODEL.__file__ = str(_hand_model_path)
exec(compile(_hand_model_source, str(_hand_model_path), "exec"), HAND_MODEL.__dict__)
print("[V103 HAND SOURCE]", _hand_model_path, "sha256=",
      hashlib.sha256(_hand_model_source).hexdigest(), "loader=direct-source")

_deformation_path = Path(PROJECT_DIR) / "deformation_monitor.py"
DEFORMATION_MONITOR = types.ModuleType("v103_current_deformation_monitor")
exec(compile(_deformation_path.read_bytes(), str(_deformation_path), "exec"), DEFORMATION_MONITOR.__dict__)

_palm_policy_path = Path(PROJECT_DIR) / "palm_policy.py"
PALM_POLICY = types.ModuleType("v103_palm_policy_current")
exec(compile(_palm_policy_path.read_bytes(), str(_palm_policy_path), "exec"), PALM_POLICY.__dict__)
_joint_search_path = Path(PROJECT_DIR) / "joint_search.py"
JOINT_SEARCH = types.ModuleType("v103_joint_search_current")
exec(compile(_joint_search_path.read_bytes(), str(_joint_search_path), "exec"), JOINT_SEARCH.__dict__)
_contact_wave_path = Path(PROJECT_DIR) / "contact_wave.py"
CONTACT_WAVE = types.ModuleType("v103_contact_wave_current")
_contact_wave_source = _contact_wave_path.read_bytes()
exec(compile(_contact_wave_source, str(_contact_wave_path), "exec"), CONTACT_WAVE.__dict__)
print("[V103 CONTACT SOURCE]", _contact_wave_path, "sha256=",
      hashlib.sha256(_contact_wave_source).hexdigest(), "loader=direct-source")
print("[V103 BUILD]", getattr(CONTACT_WAVE, "BUILD", "UNKNOWN-CONTACT-BUILD"))
_base_guard_path = Path(PROJECT_DIR) / "base_guard.py"
BASE_GUARD = types.ModuleType("v103_base_guard_current")
exec(compile(_base_guard_path.read_bytes(), str(_base_guard_path), "exec"), BASE_GUARD.__dict__)
_depth_control_path = Path(PROJECT_DIR) / "depth_control.py"
DEPTH_CONTROL = types.ModuleType("v103_depth_control_current")
_depth_control_source = _depth_control_path.read_bytes()
exec(compile(_depth_control_source, str(_depth_control_path), "exec"), DEPTH_CONTROL.__dict__)
_v104_cache_path = Path(PROJECT_DIR) / "planning_cache.py"
V104_CACHE = types.ModuleType("v104_planning_cache_current")
_v104_cache_source = _v104_cache_path.read_bytes()
exec(compile(_v104_cache_source, str(_v104_cache_path), "exec"), V104_CACHE.__dict__)
_release_policy_path = Path(PROJECT_DIR) / "release_policy.py"
RELEASE_POLICY = types.ModuleType("v103_release_policy_current")
exec(compile(_release_policy_path.read_bytes(), str(_release_policy_path), "exec"), RELEASE_POLICY.__dict__)
print('[V103 RELEASE AUDIT v1] FOUR_FINGERS; PHYSX_BASE_STOP; MEASURED_12DOF_EXIT; RAW_DEPTH_QUALITY')
print('[V103 DEPTH POLICY] FOUR-FINGER-v3; index/middle/ring each require measured 2..4mm with 3mm preferred; pinky stays in the 4F trajectory and alone may finish at bounded safe local optimum; >4mm normal quality cap; diagnostic depth override is enabled only below the >8mm hard stop')
print('[V103 PLANNER BUILD] FULL-GATE-FIX-2TO4-NATURAL-MIN-ACQ-DEFERRED-LOCAL-OPTIMUM-PINKY-BOOTSTRAP-V1')
print('[V103 BASE POLICY] planning/replay margin disabled; post-scratch curl stops on BackSurface clearance OR measured base contact, then unload/open/Home')
print("[V104 BUILD] PLANNING-CACHE-v2 + CONTACT-WORKSPACE-v22-MINIMA-CENTROID-MCP-ROW-DEPTH-IK-ROLLING-WAVE-CENTER15 + SAFE-PLANE-NORMAL-APPROACH-V1 + PREPOSITION-SWEEP-OVERRIDE-V1 + DIAGNOSTIC-DEPTH-CAP-OVERRIDE-V1; preposition and planned-depth overrides are diagnostic-only")

BACK_CSV = os.path.join(PROJECT_DIR, "dense_back_candidates_5mm.csv")
ARC_CSV = os.path.join(PROJECT_DIR, HAND_MODEL.ARC_FILENAME)
LULA_URDF = os.path.join(PROJECT_DIR, "robotnew_with_aero_hand_right_lula.urdf")
COLLISION_URDF = os.path.join(PROJECT_DIR, "robotnew_with_aero_hand_right_isaac.urdf")
DESCRIPTOR = os.path.join(PROJECT_DIR, "dense_ik_robot_descriptor.yaml")

REQUEST_JSON = os.path.join(PROJECT_DIR, "scratch_target_request.json")
STATUS_JSON = os.path.join(PROJECT_DIR, "scratch_status.json")
HOME_JSON = os.path.join(PROJECT_DIR, "scratch_home_q.json")

PLAN_CSV = os.path.join(PROJECT_DIR, "scratch_v103_last_plan.csv")
TRAJECTORY_CSV = os.path.join(PROJECT_DIR, "scratch_v103_staged_last_trajectory.csv")
PALM_TRIAL_JSON = os.path.join(PROJECT_DIR, "scratch_v103_last_palm_trial.json")
# World-space fingertip samples captured from the live USD tip-link markers.
# This is intentionally separate from the planner's arc CSV: the latter is a
# proposal, while this file is the measured execution trace used for replay
# and visual inspection.
ACTUAL_TIP_TRACE_JSON = os.path.join(PROJECT_DIR, "scratch_v103_last_actual_tip_trace.json")
V104_DATA_ROOT = os.environ.get(
    "DOFBOT_SCRATCH_DATA_ROOT",
    str(Path(PROJECT_DIR) / "runs"),
)

ROBOT_ROOT = "/World/robotnew_with_aero_hand_right"
BASE_PATH = ROBOT_ROOT + "/base_link"
PALM_FRAME = "right_base_link"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
LONG_FINGERS = ["index", "middle", "ring", "pinky"]
V103_MANDATORY_QUALITY_FINGERS = ("index", "middle", "ring")
V103_OPTIONAL_QUALITY_FINGERS = ("pinky",)

FINGER_JOINT_NAMES = {
    "index": "right_index_mcp_flex",
    "middle": "right_middle_mcp_flex",
    "ring": "right_ring_mcp_flex",
    "pinky": "right_pinky_mcp_flex",
}

# Finger command/phase scalars throughout the legacy planner are now 0..100%
# progression, NOT MCP degrees. Actual arm and triplet angles remain degrees.
FINGER_MCP_STOP_DEG = HAND_MODEL.LIMITS_DEG[0]
FINGER_PIP_STOP_DEG = HAND_MODEL.LIMITS_DEG[1]
FINGER_DIP_STOP_DEG = HAND_MODEL.LIMITS_DEG[2]
FINGER_SAFE_COMMAND_MAX_DEG = HAND_MODEL.CONTROL_MAX  # legacy name; percent

FINGER_TIP_PATHS = {
    "index": ROBOT_ROOT + "/right_index_tip_link",
    "middle": ROBOT_ROOT + "/right_middle_tip_link",
    "ring": ROBOT_ROOT + "/right_ring_tip_link",
    "pinky": ROBOT_ROOT + "/right_pinky_tip_link",
}

FINGER_DISTAL_PATHS = {
    "index": ROBOT_ROOT + "/right_index_distal_link",
    "middle": ROBOT_ROOT + "/right_middle_distal_link",
    "ring": ROBOT_ROOT + "/right_ring_distal_link",
    "pinky": ROBOT_ROOT + "/right_pinky_distal_link",
}

# First run: put robot in the desired upright Home, press Play, then run V4.
# V4 captures the ACTUAL Isaac q and stores scratch_home_q.json.
RECAPTURE_HOME = False

# Project policy: this runtime plans and executes four-finger HUMAN_WAVE
# strategies only. Keep this guard close to the entry configuration so helper
# callers cannot accidentally re-enable 3F/2F/1F search by omitting a filter.
FOUR_FINGER_ONLY = True

# Radius of the finite GUI-selected scratch region. It is used both to collect
# the local point-cloud patch and as the hard per-fingertip contact boundary.
# Increased from 75 mm to 80 mm so a valid full scratch arc is not rejected by
# a sub-millimetre tracking/path excursion beyond the old finite boundary.
PATCH_RADIUS_M = 0.080

# Scratch depth policy (negative signed value = into the back surface).
# Target stays near 3 mm. 1-4 mm is the normal working zone.
# 4-8 mm is a recoverable DEEP_ADJUST zone. Only >8 mm is a hard reject.
IDEAL_SIGNED_M = -0.0020
GOOD_SIGNED_MIN_M = -0.0040
GOOD_SIGNED_MAX_M = +0.0010
SOFT_SIGNED_MIN_M = -0.0080
SOFT_SIGNED_MAX_M = +0.0020
HARD_PENETRATION_M = -0.0080
RUNTIME_HARD_PENETRATION_M = -0.0080
INACTIVE_PENETRATION_M = -0.0005

DEEP_ADJUST_ENTER_SIGNED_M = -0.0040
DEEP_ADJUST_EXIT_SIGNED_M = -0.0035
DEEP_ADJUST_HARD_SIGNED_M = -0.0080
DEEP_ADJUST_MAX_PASSES = 3
DEEP_ADJUST_KEYFRAME_TIME_S = 0.20
DEEP_ADJUST_PENALTY_WEIGHT = 8.0
DEEP_ADJUST_PENALTY_SCALE_M = 0.0020


# V4.6 surface-distance semantics.
#
# The 5 mm point cloud is excellent for local planning, but a nearest-point
# normal is only a LOCAL tangent-plane approximation. A point that is 30-60 mm
# away can easily produce a large negative dot product even though it is not
# physically inside the back.
#
# Therefore point-cloud signed distance is trusted only near the surface.
# Final trajectory/runtime safety uses the actual USD BackSurface triangle mesh
# when available.
POINTCLOUD_SIGNED_RELIABLE_M = 0.008
MESH_SIGNED_RELIABLE_M = 0.008

# V4.10 acquisition-only warm-contact acceptance.
# If the current arm pose already places the flexed working fingertip in
# this mesh band, keep q unchanged and accept the contact immediately.
WARM_CONTACT_MAX_DISTANCE_M = 0.0080
WARM_CONTACT_RAW_MIN_M = -0.0080
WARM_CONTACT_RAW_MAX_M = +0.0020


# ============================================================
# V4.11 MULTI-FINGER CONTACT BALANCING
# ============================================================

# Four / three finger contact is the primary objective on flat back patches.
# Pair and single-finger strategies are fallbacks only after multi-finger
# candidates have been thoroughly optimized and their trajectories rejected.
MULTI_BALANCE_PHASE_WINDOW_DEG = 14.0
MULTI_BALANCE_MAXITER = 48
FAST_ENDPOINT_VALID_TARGET = 2
FAST_LULA_SEED_LIMIT = 12
FAST_DIRECT_REFINE_SEED_LIMIT = 5

# J4/J5 get the largest freedom because they are the local "contact leveling"
# joints. J2/J3 may help moderately; J1 should move the least.
MULTI_BALANCE_LOCAL_DELTA_DEG = np.array(
    [12.0, 24.0, 24.0, 50.0, 55.0],
    dtype=np.float64,
)

MULTI_BALANCE_MOVE_WEIGHTS = np.array(
    [4.0, 1.3, 1.3, 0.04, 0.12],
    dtype=np.float64,
)

# Explicit common-mode / differential-mode contact objectives.
# J5 and overall wrist pose naturally reduce mean depth error.
# J4/J5 together naturally reduce left-right / finger-to-finger depth spread.
MULTI_MEAN_DEPTH_WEIGHT = 12.0
MULTI_DEPTH_SPREAD_WEIGHT = 24.0
MULTI_DEPTH_RANGE_WEIGHT = 8.0
MULTI_PHASE_REG_WEIGHT = 3.0


# Preserve the original wave ordering. Example 4F-A starts near
# 90/84/78/72 deg and should remain a wave rather than collapse to
# 84/83/82/82 deg.
# The back-normal minimum is pose- and surface-normal-dependent.  It is measured
# again for every candidate; no fixed phase is assumed here.  Do not force an
# old template wave after a finger has already passed its measured minimum.
MULTI_PHASE_SHAPE_WEIGHT = 1.0
MULTI_MIN_WAVE_GAP_DEG = 0.0
MULTI_WAVE_GAP_VIOLATION_WEIGHT = 0.0

# Keep the four-finger contact centroid on the GUI target while balancing
# depths. Convex-hull coverage is diagnostic only and is not optimized.
MULTI_TARGET_CENTER_WEIGHT = 18.0
MULTI_CENTER_SOFT_LIMIT_M = 0.025
MULTI_CENTER_BARRIER_WEIGHT = 220.0

# Raw finger-arc tangent is not a hard gate because J4/J5 later compensate
# dynamically, but a near-zero tangent endpoint is a bad seed for scratching.
MULTI_TANGENT_SOFT_TARGET = 0.35
MULTI_TANGENT_WEIGHT = 14.0

# Adaptive approach is based on REAL fingertip-to-mesh distance, not palm
# offset alone.
APPROACH_FAR_SEARCH_OFFSETS_M = [
    0.060,
    0.080,
    0.100,
    0.120,
    0.140,
    0.160,
]

APPROACH_PRECONTACT_NEAR_BAND_M = 0.006
APPROACH_PRECONTACT_RAW_MIN_M = -0.0010
APPROACH_INWARD_STEP_M = 0.005

# Working fingers stay fully open during free-space approach.
# They are allowed to start flexing only after the arm reaches the
# approach/contact handoff and the scratching/contact-acquisition phase begins.
# Non-working fingers may still be folded before approach to keep them clear.
WORKING_APPROACH_OPEN_DEG = 0.0

# V6: during gross arm approach every working finger stays straight.
# There is deliberately NO semantic minimum open-tip standoff (no 18 mm / 6 mm
# rule). A straight fingertip may be very close to the back as long as the
# complete hand geometry is collision-free. Any actual open-hand contact remains
# an error and is stopped by the mesh/PhysX contact safety layers.
# Nominal acquisition increment retained for compatibility with older
# templates.  The live V103 acquisition now derives the actual increment from
# measured depth error and FK sensitivity; these bounds prevent either a
# brittle fixed 4-degree jump or an impractically tiny crawl.
ARC_CONTACT_PHASE_STEP_DEG = 4.0
ARC_CONTACT_PHASE_MIN_STEP_DEG = 0.75
ARC_CONTACT_PHASE_MAX_STEP_DEG = 6.0
ARC_CONTACT_PHASE_TARGET_FRACTION = 0.60

# V9 late arc handoff.
# Gross arm approach remains fully open for every WORKING finger. If the next
# open-hand inward segment would sweep a distal/proximal link through the back,
# the planner STOPS at the last truly safe open pose instead of forcing the
# open hand to the endpoint. Only then, in free space, it introduces a measured
# wave-shaped pre-curve and lets the contact-acquisition solver coordinate the
# remaining arm motion with the finger arc. This relocates the useful old ~42
# degree bend to the handoff, not to the early gross approach.
ARC4_LATE_HANDOFF_PRECURVE_MAX_DEG = [42.0, 36.0, 30.0, 24.0, 18.0, 12.0]
ARC4_LATE_HANDOFF_PHASE_MARGIN_DEG = 6.0
ARC4_LATE_HANDOFF_KEYFRAME_TIME_S = 0.42

# GUI TARGET_PATCH lock. The clicked point is the CENTER of a local scratch
# region, not a single point that every finger must hit. Active fingers receive
# parallel lanes through the patch. The band center must stay near the GUI
# target and each finger must stay in its own lane and inside the finite patch.
# A plan uses a smaller radius so tracking error cannot immediately cross the
# 80 mm runtime boundary.
# V102 comfort-first mode treats the GUI click as the centre of a useful
# scratch region, not a compulsory four-tip bullseye.  Planning may use the
# full runtime lane/patch envelope to collect more valid motion data; the
# swept mesh, joint limits, start-clearance and hard penetration checks remain
# authoritative during validation and execution.
TARGET_PATCH_CENTER_HARD_M = 0.015
V103_PATH_REVIEW_CENTER_MAX_M = 0.015
V103_PATH_REVIEW_SHALLOW_MAX_M = 0.006
V103_EFFECTIVE_CONTACT_CENTER_MAX_M = 0.015
TARGET_PATCH_CENTER_STRONG_M = 0.012
# Data-collection mode: allow a wider lateral channel so a valid four-finger
# wave is not discarded merely because one finger's lane is offset on a steep
# or curved patch.  Swept collision, joint limits, start clearance, patch
# bounds, and the 4 mm penetration cap remain authoritative.
TARGET_LANE_CROSS_HARD_M = 0.020
TARGET_LANE_CROSS_STRONG_M = 0.012
TARGET_PATCH_RADIUS_HARD_M = PATCH_RADIUS_M
PLANNED_ACTIVE_PATCH_RADIUS_M = TARGET_PATCH_RADIUS_HARD_M
# Do not reject otherwise usable trajectories merely because their four-finger
# channel is outside the old strong 8 mm/65 mm planning envelope.  The hard
    # 20 mm lane and 80 mm finite patch are still bounded, while swept collision
# and penetration safety checks remain unchanged.
PLANNED_ACTIVE_LANE_CROSS_M = TARGET_LANE_CROSS_HARD_M
TARGET_PATCH_LOCK_WEIGHT = 36.0
TARGET_LANE_LOCK_WEIGHT = 26.0

# Per-target upper bound, not a desired angle. Shared by endpoint and path solvers.
# Context is reset after planning and restored from the selected mode at runtime.
def palm_alignment_limit_deg():
    return PALM_POLICY.limit_deg()


def palm_alignment_scale_deg():
    # Keep the V102 cost identical at 30deg, retain a parallel preference in
    # wider tiers without the old cost scale overwhelming reachable poses.
    return PALM_ALIGNMENT_SCALE_DEG * palm_alignment_limit_deg() / 30.0

# Offline whole-hand collision checking. Exact runtime PhysX contact stop is
# still authoritative. The URDF collision model is used here to catch palm /
# proximal / middle / distal link collisions before execution.
HAND_COLLISION_MODEL = None
V103_DEPTH_TARGET_M = 0.003
# Each required finger is judged independently.  A finger may be anywhere in
# the 1--4 mm working band; one finger's depth never invalidates another's.
V103_QUALITY_DEPTH_MIN_M = 0.0010
# Acquisition is only the entry into the continuous wave. Do not force the
# first owner to 2 mm before the wave controller can coordinate the other fingers.
# 0.5 mm is a real signed BackSurface contact, while the final mandatory quality
# uses the relaxed live 1--4 mm band.
V103_ACQUISITION_WAVE_ENTRY_DEPTH_M = 0.0005
V103_QUALITY_DEPTH_MAX_M = 0.0040
V103_PLANNED_DEPTH_MAX_M = 0.0040
# The first dynamically selected finger is the depth anchor.  It keeps the
# established 3 mm preference and relaxed 1--4 mm safety band.  The remaining three
# channels are still present in every command frame, but their initial/final
# depth is allowed to settle wherever the measured geometry and swept mesh
# permit; a small 0.5 mm floor prevents a completely airborne command from
# being mistaken for useful contact while avoiding the old all-fingers 2 mm
# deadlock.
V103_PRIMARY_DEPTH_TARGET_M = 0.0030
V103_ASSIST_DEPTH_MIN_M = 0.0005
# Use the full 4 mm quality band for data collection.  The physical -8 mm hard
# Keep a separate planning reserve below the hard 4 mm quality cap. Runtime
# tracking/interpolation can overshoot an endpoint, so the planner must not
# spend the full hard cap. Actual accepted depth remains in the 1--4 mm band.
V103_RUNTIME_PLANNED_DEPTH_CAP_M = 0.0036
# Base distance is retained as telemetry, but it is no longer a pre-emptive
# planner rejection.  The old 3 mm red line rejected otherwise usable scratch
# poses because the URDF base mesh is an imperfect proxy for the real pedestal.
# A base contact is handled only during the post-scratch curl-release phase:
# curl until the fingertips clear the BackSurface, or stop at the first base
# contact.  Both are valid exits; the base contact is not a scratch failure.
V103_BASE_MIN_CLEARANCE_M = 0.003  # legacy/reporting value; not a planning gate
BASE_PLANNING_GUARD_DISABLED = True
V103_BASE_CONTACT_TOLERANCE_M = 0.00030
BASE_CONTACT_STOP_ENABLED = True
V103_BASE_CONTACT_STOP_STAGES = ("SCRATCH_CURL_", "FINGER_CURL_RELEASE_")
# Near the base, a still-airborne pinky may be omitted from scratch quality
# rather than consuming the remaining collision-free wrist/arm workspace.
# This is a policy gate, not a relaxed collision or penetration threshold.
V103_OPTIONAL_PINKY_BASE_ZONE_M = 0.018
V103_OPTIONAL_PINKY_MIN_GAP_M = 0.006
# Retained only for compatibility with old offline reports.  V103 production
# never uses a radial-distance qualification gate for the RMI fallback.
V103_NEAR_BASE_3F_RADIAL_MAX_M = None
V103_STRICT_POINT_TEST = False  # All active fingers use their parallel local lanes.
# Retained for compatibility with old reports; no finger is singled out as
# the GUI-target anchor in the active planner.
V103_MIDDLE_ANCHOR_MAX_M = 0.025
V103_BASE_EARLY_EXIT_M = 0.030
# Single safety margin only. It is used for collision/lane hysteresis and
# free-space safety; it must never qualify contact or finish scratch execution.
V103_SAFETY_CLEARANCE_M = 0.0015
# Collection-only tolerance for the post-scratch unload/open transition.
# Acquisition, contact, and full curl retain their normal runtime gates.
UNLOAD_RUNTIME_PENETRATION_TOLERANCE_M = -0.0020
UNLOAD_RUNTIME_TOLERANCE_STAGES = (
    "UNLOAD_STEP_",
    "UNLOAD_J5_LIFT_",
    "UNLOAD_ALL_",
    "OPEN_AFTER_UNLOAD",
)
# The release trajectory still targets +20 mm.  This lower bound is only the
# measured runtime acceptance threshold, allowing tracking error while keeping
# the planned handoff target unchanged.
V103_ACTUAL_EXIT_ACCEPTANCE_M = 0.017
V103_EXIT_PLANNED_RESERVE_M = 0.006
V103_EXIT_PLANNED_BASE_RESERVE_M = 0.015
V103_EXIT_EXTRA_J5_MAX_DEG = 15.0
V103_EXIT_EXTRA_CURL_MAX_PCT = 12.0
HAND_COLLISION_COARSE_GATE_M = 0.020
# Numerical contact epsilon only. This is NOT a semantic approach standoff.
# Non-tip STL surface samples closer than this to BackSurface are treated as
# actual surface contact so a static/tangent touch cannot slip between sweep
# segments. The hard path decision otherwise comes from true segment/triangle
# intersection, never from the sign of an open-sheet triangle normal.
HAND_STATIC_CONTACT_EPS_M = 0.00020
HAND_SWEEP_INTERSECTION_EPS_M = 1.0e-9
HAND_SWEEP_AABB_PAD_M = 0.00010
HAND_COLLISION_MAX_PROBES_PER_COLLISION = 48
# Diagnostic trajectory mode requested for inspecting difficult points.  Only
# the open/preposition arm segment is allowed to cross the BackSurface; actual
# scratch frames, lane checks, joint limits, and depth quality gates remain
# strict.  This is intentionally not a production safety setting.
V103_ALLOW_PREPOSITION_SWEEP_INTERSECTION = False
# Preferred preposition route: finish palm/J4 orientation while the hand is
# well outside the BackSurface, then lower along the fixed surface normal.
# This prevents a late J4 rotation from sweeping an open ring/index tip through
# the sheet near the final pose.
V103_PREPOSITION_SAFE_HEIGHT_M = 0.10
# Finish the J4/palm orientation before descending to the 10 cm safe plane.
# The extra height keeps the late wrist rotation away from the BackSurface.
V103_PREPOSITION_ALIGN_HEIGHT_M = 0.15
V103_PREPOSITION_NORMAL_ROUTE_SAMPLES = 3
# The open/prepared fingertip positions must remain outside the BackSurface
# before the first finger begins its curl.  This is a small numerical margin,
# not a scratch-depth allowance.
V103_PREPOSITION_START_MIN_CLEARANCE_M = HAND_STATIC_CONTACT_EPS_M
# Diagnostic trajectory mode: allow a planned fingertip to exceed the normal
# 3.6 mm runtime reserve / 4 mm quality band so the user can inspect the real
# motion on difficult poses.  This never relaxes the hard -8 mm penetration
# stop, lane checks, joint limits, or mesh swept checks, and the resulting run
# is not a quality pass.
V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW = False
V103_DIAGNOSTIC_DEPTH_CAP_M = 0.0075


def v103_effective_runtime_depth_cap_m():
    """Return the planning cap, bounded strictly above the hard -8 mm stop."""
    normal = float(V103_RUNTIME_PLANNED_DEPTH_CAP_M)
    if not bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW):
        return normal
    hard_limit = abs(float(HARD_PENETRATION_M))
    return min(float(V103_DIAGNOSTIC_DEPTH_CAP_M), max(0.0, hard_limit - 1.0e-6))


def v103_diagnostic_depth_overflow_allowed(failure):
    """Whether a >4 mm depth is allowed only to expose a diagnostic trajectory."""
    return bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW) and str(failure) == 'DEPTH_MAX'
ARC_PRECONTACT_KEYFRAME_TIME_S = 0.035
ARC_CONTACT_KEYFRAME_TIME_S = 0.040

# A duration is only the command interpolation time, never proof that the
# articulation physically arrived. Hold each target until measured joints are
# settled, otherwise stop instead of accumulating tracking error.
KEYFRAME_ARM_ARRIVAL_TOL_DEG = 1.5
KEYFRAME_FINGER_ARRIVAL_TOL_DEG = 3.0
KEYFRAME_SETTLE_TIME_S = 0.06
SCRATCH_KEYFRAME_SETTLE_TIME_S = 0.02
SCRATCH_WAVE_KEYFRAME_SETTLE_TIME_S = 0.0
KEYFRAME_ARRIVAL_TIMEOUT_S = 1.50
KEYFRAME_WAIT_LOG_PERIOD_S = 0.30

# V4.12-ARC4 contact acquisition policy.
# The 4-8 mm band is recoverable, not the preferred place to stop the curl.
# Acquisition therefore keeps following the measured finger arc until the
# fingertip reaches the normal 0-4 mm working zone whenever possible.
# If the planned template phase is slightly early, the search may continue
# through the remaining measured arc instead of forcing a bad fixed phase.
ARC4_PREFERRED_CONTACT_MIN_SIGNED_M = GOOD_SIGNED_MIN_M
ARC4_PREFERRED_CONTACT_MAX_SIGNED_M = GOOD_SIGNED_MAX_M
ARC4_FULL_ARC_SCAN = True
ARC4_CONTACT_SCORE_PHASE_WEIGHT = 0.000015

# After all working fingers have acquired safe contact, ARC4 requires one
# genuine synchronized human-wave cycle before unload.  Large excursions are
# tried first; the planner may reduce amplitude to preserve all four contacts.
ARC4_WAVE_AMPLITUDE_CANDIDATES_DEG = [4.0, 3.0, 2.0, 1.5]
ARC4_WAVE_KEYFRAME_MIN_TIME_S = 0.08
ARC4_WAVE_KEYFRAME_MAX_TIME_S = 0.22

# One GUI target executes two human-like scratch strokes.  Between strokes the
# hand returns to the measured, open and unloaded ``cycle_open_q`` safe pose
# near the target; it does not return Home until the second stroke completes.
# Batch ``--repeats`` remains the explicit way to request another independent
# attempt from Home.
SCRATCH_CYCLE_COUNT = 2
SCRATCH_FULL_CURL_DEG = FINGER_SAFE_COMMAND_MAX_DEG
SCRATCH_CONTACT_RELEASE_ADVANCE_DEG = 5.0
SCRATCH_CURL_FINGER_SPEED_DEG_S = 120.0
SCRATCH_CONTACT_FINGER_ARRIVAL_TOL_DEG = 5.0
SCRATCH_WAVE_FINGER_ARRIVAL_TOL_DEG = 3.0
SCRATCH_ACQUISITION_FINGER_ARRIVAL_TOL_DEG = 8.0
SCRATCH_CONTACT_KEYFRAME_TIMEOUT_S = 2.50
# Contact drives normally retain a small error while the skin pushes back.
# Do not turn that harmless 2--4 degree residual into a complete trajectory
# stop.  Slow the shared clock first and reserve a hard hold for a genuinely
# lost drive.  Depth, swept collision and lane checks remain unchanged.
SCRATCH_WAVE_TRACKING_SLOW_FINGER_DEG = 2.0
SCRATCH_WAVE_TRACKING_HOLD_FINGER_DEG = 4.0
SCRATCH_WAVE_TRACKING_SLOW_ARM_DEG = 0.20
SCRATCH_WAVE_TRACKING_HOLD_ARM_DEG = 0.50
SCRATCH_WAVE_TRACKING_MIN_CLOCK_SCALE = 0.35
SCRATCH_WAVE_TRACKING_RECOVERY_WALL_S = 2.0
# Live PhysX pressure guard. Command-FK can under-predict a fingertip that is
# still physically catching up while J4/J5 continue pressing. Start lifting
# the shared wrist before the 4 mm hard limit; the current focus may be held,
# but a completed/non-focus finger is still allowed to curl away.
SCRATCH_WAVE_PRESSURE_GUARD_ENTER_M = 0.0032
SCRATCH_WAVE_PRESSURE_GUARD_EXIT_M = 0.0030
SCRATCH_WAVE_PRESSURE_GUARD_HARD_M = 0.0036
SCRATCH_WAVE_PRESSURE_GUARD_LIFT_DEG = 0.35
SCRATCH_WAVE_PRESSURE_GUARD_HARD_LIFT_DEG = 0.75
# Acquisition is itself the first part of the wave. Later fingers advance from
# the first staging frame instead of waiting for the owner to finish.
ACQUISITION_BACKGROUND_RATE_BY_RANK = (0.65, 0.42, 0.25)
ACQUISITION_BACKGROUND_MAX_STEP_DEG = 3.0
# Followers move from the first staging frame, but must preserve normal-depth
# headroom for the fingers ahead of them.  These are soft pre-contact reserves,
# not completion targets.  If additional curl is already lifting a follower
# away from the surface, the reserve does not block that useful motion.
ACQUISITION_BACKGROUND_DEPTH_RESERVE_BY_RANK_M = (0.0016, 0.0008, 0.0003)
ACQUISITION_BACKGROUND_DEPTH_SLOW_BAND_M = 0.00045
# Fixed-palm first-owner invariant. The selected scratch preposition is
# accepted when the FIRST frozen owner can reach its natural trajectory
# minimum inside the relaxed 1--4 mm quality band with J1-J5 fixed. 3 mm is
# the preferred setpoint; it is not allowed to turn a safe 2.x mm natural pose
# into a destructive wrist/normal correction.
V103_FIRST_OWNER_MIN_TARGET_M = 0.0030
V103_FIRST_OWNER_MIN_TARGET_TOL_M = 0.00035
# Near-base poses can be steep enough that the fixed-palm natural minimum is
# slightly shallower than 3 mm even though the trajectory is safe.  This is a
# bounded planning tolerance only; it never changes the 4 mm hard cap, lane,
# collision, or final core-finger quality rules.  Keep it zero for ordinary
# poses so the known-good 2444 behavior is unchanged.
V103_FIRST_OWNER_TILT_TOLERANCE_START_DEG = 60.0
V103_FIRST_OWNER_TILT_TOLERANCE_PER_DEG_M = 0.000020
V103_FIRST_OWNER_TILT_TOLERANCE_MAX_M = 0.000800
# For steep/near-base poses, the natural first-owner minimum may be shallower
# than the ordinary 1 mm band.  Lower only this planning/handoff floor with
# tilt, down to 1 mm at very steep poses; the 4 mm ceiling and all geometry
# safety checks remain unchanged.
V103_FIRST_OWNER_DEPTH_FLOOR_TILT_START_DEG = 60.0
V103_FIRST_OWNER_DEPTH_FLOOR_FULL_RELAX_DEG = 75.0
V103_FIRST_OWNER_DEPTH_FLOOR_MIN_M = 0.0010

# The 15 mm Cmin placement limit is a nominal target, not a reason to destroy
# an otherwise safe natural first-owner pose on steep/near-base approaches.
# Allow a bounded tilt-dependent placement tolerance during planning only;
# collision, lane, patch and the 4 mm hard depth cap remain unchanged.
V103_MINIMA_CENTROID_TILT_TOLERANCE_START_DEG = 60.0
V103_MINIMA_CENTROID_TILT_TOLERANCE_PER_DEG_M = 0.00050
V103_MINIMA_CENTROID_TILT_TOLERANCE_MAX_EXTRA_M = 0.015

# The executed moving-contact centroid can drift farther than the planning
# Cmin while a steep/near-base hand is handing off and releasing fingers.  Keep
# the ordinary 15 mm center target unchanged, but apply a bounded tilt-only
# tolerance to this final diagnostic/quality gate.  This is not a collision,
# lane, or depth waiver; every accepted segment has already passed ``safe()``.
V103_EFFECTIVE_CONTACT_CENTER_TILT_START_DEG = 60.0
V103_EFFECTIVE_CONTACT_CENTER_TILT_PER_DEG_M = 0.00080
V103_EFFECTIVE_CONTACT_CENTER_TILT_MAX_EXTRA_M = 0.015


def v103_first_owner_depth_tolerance_m(palm_angle_deg):
    """Return the 3 mm first-owner planning tolerance for this palm tilt."""
    try:
        angle = float(palm_angle_deg)
    except (TypeError, ValueError):
        angle = float("nan")
    if not np.isfinite(angle):
        return float(V103_FIRST_OWNER_MIN_TARGET_TOL_M)
    excess = max(0.0, angle - float(V103_FIRST_OWNER_TILT_TOLERANCE_START_DEG))
    extra = min(
        excess * float(V103_FIRST_OWNER_TILT_TOLERANCE_PER_DEG_M),
        float(V103_FIRST_OWNER_TILT_TOLERANCE_MAX_M),
    )
    return float(V103_FIRST_OWNER_MIN_TARGET_TOL_M + extra)


def v103_first_owner_min_depth_m(palm_angle_deg, finger=None):
    """Return the tilt-dependent first-owner natural-minimum floor.

    Ordinary poses retain the relaxed 1 mm acquisition floor.  At steep
    poses the floor decreases linearly to 1 mm, allowing a real shallow
    natural minimum to seed the continuous wave instead of forcing a normal
    IK correction that can destroy it.

    ``pinky`` is the only quality-optional channel.  If terrain ordering makes
    it the first owner, it may bootstrap from a safe shallow/airborne natural
    optimum; the remaining index/middle/ring channels still have to earn their
    normal 1--4 mm quality later.  This is a planning floor only: the open
    preposition clearance, 4 mm cap, collision and lane checks remain active.
    """
    if str(finger).lower() == "pinky":
        return 0.0
    try:
        angle = float(palm_angle_deg)
    except (TypeError, ValueError):
        angle = float("nan")
    if not np.isfinite(angle):
        return float(V103_QUALITY_DEPTH_MIN_M)
    start = float(V103_FIRST_OWNER_DEPTH_FLOOR_TILT_START_DEG)
    full = float(V103_FIRST_OWNER_DEPTH_FLOOR_FULL_RELAX_DEG)
    if angle <= start:
        return float(V103_QUALITY_DEPTH_MIN_M)
    fraction = np.clip((angle - start) / max(full - start, 1e-9), 0.0, 1.0)
    return float(
        V103_QUALITY_DEPTH_MIN_M
        - fraction * (V103_QUALITY_DEPTH_MIN_M - V103_FIRST_OWNER_DEPTH_FLOOR_MIN_M)
    )


def v103_quality_min_depth_m(palm_angle_deg, finger):
    """Return the effective lower quality floor for one finger.

    Core fingers use the same tilt-dependent floor that acquisition uses;
    otherwise a steep pose could be admitted at 1 mm and then rejected by a
    later fixed 1 mm validator.  Pinky remains an assist/local-optimum
    channel and is handled by its separate relaxed policy.
    """
    finger = str(finger).lower()
    if finger in V103_MANDATORY_QUALITY_FINGERS:
        return float(v103_first_owner_min_depth_m(palm_angle_deg, finger))
    return float(V103_ASSIST_DEPTH_MIN_M)


def v103_minima_centroid_tolerance_m(palm_angle_deg):
    """Return the bounded Cmin placement tolerance for a planning pose."""
    try:
        angle = float(palm_angle_deg)
    except (TypeError, ValueError):
        angle = float("nan")
    if not np.isfinite(angle):
        return float(V103_MINIMA_CENTROID_TARGET_TOL_M)
    excess = max(
        0.0,
        angle - float(V103_MINIMA_CENTROID_TILT_TOLERANCE_START_DEG),
    )
    extra = min(
        excess * float(V103_MINIMA_CENTROID_TILT_TOLERANCE_PER_DEG_M),
        float(V103_MINIMA_CENTROID_TILT_TOLERANCE_MAX_EXTRA_M),
    )
    return float(V103_MINIMA_CENTROID_TARGET_TOL_M + extra)


def v103_effective_contact_center_tolerance_m(palm_angle_deg):
    """Return the executed-contact center tolerance for a palm tilt.

    Flat/ordinary poses retain the original 15 mm limit.  Near-base steep
    poses get a bounded extra allowance because the handoff/release changes
    the four moving contact centers after the fixed-palm Cmin was solved.
    """
    try:
        angle = float(palm_angle_deg)
    except (TypeError, ValueError):
        angle = float("nan")
    if not np.isfinite(angle):
        return float(V103_EFFECTIVE_CONTACT_CENTER_MAX_M)
    excess = max(
        0.0,
        angle - float(V103_EFFECTIVE_CONTACT_CENTER_TILT_START_DEG),
    )
    extra = min(
        excess * float(V103_EFFECTIVE_CONTACT_CENTER_TILT_PER_DEG_M),
        float(V103_EFFECTIVE_CONTACT_CENTER_TILT_MAX_EXTRA_M),
    )
    return float(V103_EFFECTIVE_CONTACT_CENTER_MAX_M + extra)


V103_FIRST_OWNER_MIN_SCAN_COARSE_DEG = 1.0
V103_FIRST_OWNER_MIN_SCAN_FINE_DEG = 0.20
V103_FIRST_OWNER_MIN_SCAN_RADIUS_DEG = 2.0
# A natural minimum is a turning point of signed depth, not simply the
# deepest sampled/intersecting point.  These tolerances make the turning-point
# test robust to point-cloud/mesh numerical noise while still rejecting a
# monotonic trajectory that merely keeps entering the BackSurface.
V103_NATURAL_MIN_DERIVATIVE_EPS_M_PER_DEG = 5.0e-6
V103_NATURAL_MIN_TANGENT_NORMAL_MAX = 0.45
V103_FIRST_OWNER_PREPOSITION_SAMPLES = 9
# V20: if the ordinary staging->endpoint joint corridor does not place the
# first owner at 3 mm, solve the PALM pose directly along the local BackSurface
# normal instead of rejecting a whole IK family.  The target is one scalar
# normal offset; IK only realizes that palm pose.  Every solved pose is then
# re-measured with the full finger arc and existing mesh/lane/joint guards.
V103_FIRST_OWNER_DEPTH_IK_SEARCH_M = 0.020
V103_FIRST_OWNER_DEPTH_IK_COARSE_STEP_M = 0.0025
# Keep the original precise preposition search as the default.  A looser
# staging solve is defined separately and is used only after the original
# route plus the high-plane J4 route cannot produce a safe pose.
V103_FIRST_OWNER_DEPTH_IK_POSITION_TOLS_M = (0.00075, 0.0015, 0.0030)
V103_FIRST_OWNER_APPROX_POSITION_TOLS_M = (0.0060, 0.0100, 0.0150)
V103_FIRST_OWNER_DEPTH_IK_MAX_REFINES = 5
# V22: tangential placement is solved directly from the four natural-minimum
# centroid Cmin.  The whole palm target is shifted in the local BackSurface
# tangent plane so Cmin is driven toward the GUI target before normal-depth IK.
V103_MINIMA_CENTROID_TANGENT_RECENTER_MAX_M = 0.030
V103_MINIMA_CENTROID_TARGET_TOL_M = TARGET_PATCH_CENTER_HARD_M
# All four fingers move from the beginning. Their per-frame increments are
# recomputed from remaining command-to-minimum so successive natural minima
# arrive only about one keyframe apart instead of using fixed stagger speeds.
V103_MINIMUM_ARRIVAL_GAP_STEPS = 1.0
V103_FIRST_OWNER_MAX_STEP_DEG = 4.0
V103_FOLLOWER_MAX_STEP_DEG = 6.0
V103_MIN_CONTINUOUS_FINGER_STEP_DEG = 0.10
# The 75 mm GUI patch is a contact-region constraint, not a free-space tube.
# A moving fingertip may approach from outside while still clearly airborne;
# dense mesh collision checks remain active.  Near contact the normal 75 mm
# patch rule is restored before contact can be committed.
ASYNC_PRECONTACT_PATCH_NEAR_DISTANCE_M = 0.0040
# Simple contact-stall assist requested for the real wrist convention: positive
# J5 lifts the wrist. Increase only while a contacting finger is stalled.
CONTACT_J5_LIFT_MIN_PROGRESS_DEG = 0.25
CONTACT_J5_LIFT_STEP_DEG = 0.5
CONTACT_J5_LIFT_MAX_DEG = 5.0
CONTACT_J5_LIFT_STEP_INTERVAL_S = 0.25

# Reach the target with a lightly unloaded wrist, then lower J5 to the selected
# endpoint while the four fingers start curling. Positive J5 is wrist lift in
# this robot. This removes the visible "wrist first, fingers second" pause.
DIRECT_APPROACH_WRIST_LIFT_DEG = 3.0

# V103 dynamic handoff. Completion ownership follows the frozen staging order,
# but every later finger is already moving at a lower rate. Completed contacts
# curl away while the next owner continues; there is no stop-and-wait queue.
# Base clearance, depth and swept-mesh gates remain independent.
ASYNC_WAVE_DYNAMIC_MAX_DELAY_DEG = 18.0
ASYNC_WAVE_DYNAMIC_PROBE_DEG = 1.0
ASYNC_WAVE_DYNAMIC_MIN_SENSITIVITY_M_PER_DEG = 0.00002
ASYNC_WAVE_DYNAMIC_MAX_STEP_DELAY_FRACTION = 0.22
ASYNC_WAVE_PHASE_STEP_DEG = 6.0
ASYNC_PRESSURE_START_ADVANCE_DEG = 0.0
ASYNC_J5_PRESS_STEP_DEG = 0.5
ASYNC_J5_PRESS_INTERVAL_DEG = 6.0
ASYNC_J5_PRESS_MAX_DEG = 5.0
ASYNC_INITIAL_RELIEF_GAIN_DEG_PER_MM = 0.5
ASYNC_INITIAL_RELIEF_MAX_DEG = 3.0
ASYNC_J4_TRIM_CANDIDATES_DEG = (0.0, -0.35, 0.35, -0.70, 0.70)
ASYNC_J4_TRIM_MAX_TOTAL_DEG = 2.0
ASYNC_J123_SUPPORT_MAX_TOTAL_DEG = 3.0
ASYNC_J5_DYNAMIC_TOLERANCE_DEG = 0.5
ASYNC_MIN_CONTACT_STROKE_ADVANCE_DEG = 24.0
ASYNC_PLANNED_DEPTH_TARGET_M = 0.0030
ASYNC_MIN_ACCEPTED_SCRATCH_DEPTH_M = 0.0010
# Four-finger policy: one terrain-derived finger is the primary 3 mm anchor;
# every other finger remains in the same command/trajectory and is driven
# toward the same preferred 3 mm depth.  Assist fingers use a relaxed
# completion gate (2--4 mm when reachable), and only fall back to a bounded
# local optimum after their closed-loop branches have actually been tried.
V103_PINKY_RELAXED_LOCAL_OPTIMUM = True
V103_PINKY_LOCAL_OPTIMUM_PATIENCE_STEPS = 18
V103_PINKY_LOCAL_OPTIMUM_PATIENCE_S = 18.0
IMITATION_MIN_ACTUAL_DEPTH_MM = 2.0
IMITATION_MAX_ACTUAL_DEPTH_SPREAD_MM = None
ASYNC_MIN_COMBINED_TANGENT_RATIO = 0.50
# Geometric demonstration-quality gates, not skin/comfort measurements.
ASYNC_CONTACT_EXIT_CLEARANCE_M = 0.0005
ASYNC_MIN_CONTACT_SLIDE_M = 0.0020
ASYNC_MIN_MOVING_CONTACT_S = 0.010
ASYNC_CONTACT_SAMPLE_PROGRESS_PCT = 0.5
# Open BackSurface is a sheet, not a closed solid. A triangle normal only
# has penetration meaning near the sheet. Far from the sheet, distance is
# clearance regardless of which side of the triangle normal the point lies.

# V4.7 palm/back alignment.
#
# IMPORTANT:
# We do NOT optimize J5 toward 0 deg.
# We optimize the PALM PLANE to be parallel to the local back tangent plane.
# The palm local normal is inferred automatically from the common initial
# flex direction of the four long fingers, so no X/Y/Z palm axis is guessed.
PALM_ALIGNMENT_SCALE_DEG = 12.0
PALM_ALIGNMENT_STRONG_DEG = 15.0
PALM_ALIGNMENT_WEIGHT = 18.0

# A 5-DOF arm cannot realize an arbitrary 6-D palm pose.  The direction from
# the palm root toward the four finger roots/tips must stay compatible with the
# radial motion plane selected by J1.  This is evaluated from real FK, never by
# assigning a task-space yaw to J4.
ROOT_BASE_AXIS_HARD_DEG = 15.0


def active_contact_lane_limits(lane_spec):
    """Return per-plan contact bounds, always capped by the hard GUI bounds."""
    lane_spec = lane_spec if isinstance(lane_spec, dict) else {}
    cross = float(lane_spec.get("active_lane_cross_m", PLANNED_ACTIVE_LANE_CROSS_M))
    radius = float(lane_spec.get("active_patch_radius_m", PLANNED_ACTIVE_PATCH_RADIUS_M))
    return (
        min(TARGET_LANE_CROSS_HARD_M, max(PLANNED_ACTIVE_LANE_CROSS_M, cross)),
        min(TARGET_PATCH_RADIUS_HARD_M, max(PLANNED_ACTIVE_PATCH_RADIUS_M, radius)),
    )
ROOT_BASE_AXIS_STRONG_DEG = 7.0
ROOT_BASE_AXIS_SCALE_DEG = 8.0
ROOT_BASE_AXIS_WEIGHT = 45.0

# At a fixed palm pose the measured finger arc must approach the back in the
# expected order.  A 2 mm tolerance absorbs the 2.5 mm BackSurface sampling,
# arc interpolation and soft-tissue discretization noise; it does
# not permit an earlier part of the arc to dip materially below the selected
# contact point.
ARC_HEIGHT_ORDER_TOL_M = 0.002
ARC_HEIGHT_ORDER_WEIGHT = 45.0
ARC_HEIGHT_SAMPLES = 31
ARC_HEIGHT_PHASE_REPAIR_STEP_DEG = 0.1
ARC_HEIGHT_DYNAMIC_COUPLED_ITERATIONS = 4
ARC_HEIGHT_COUPLED_Q_MAXITER = 70
MULTI_BALANCE_FAR_CONTACT_REJECT_M = 0.050

V100_DEBUG_ROOT = "/World/V100_PlanningDebug"
V100_DEBUG_CURVE_WIDTH_M = 0.003
V100_DEBUG_AXIS_LENGTH_M = 0.090
V101_DEBUG_PALM_AXIS_LENGTH_M = 0.030
# Do not put measured traces under V100_DEBUG_ROOT.  Keeping a separate root
# makes the distinction between planned arcs and actual executed tip motion
# unambiguous in the Isaac stage tree.
V103_ACTUAL_TRACE_ROOT = "/World/V103_ActualTipTrajectories"
V103_ACTUAL_TRACE_WIDTH_M = 0.0045
V103_ACTUAL_TRACE_REFERENCE_WIDTH_M = 0.0015
V103_ACTUAL_TRACE_MAX_POINTS = 4000

ROBOT_BASE_WORLD_POSITION = None
ROBOT_BASE_J1_AXIS_WORLD = None
FOUR_FINGER_ROOT_LOCAL_CENTER = None
FOUR_FINGER_ROOT_LOCAL_POINTS = None


# V4.8 J4 aesthetic preference.
#
# J4 is NOT fixed and NOT limited around zero.
# J4 and J5 are both free to cooperate to keep the palm parallel to the back.
# If two solutions are otherwise similar, the planner softly prefers J4 near
# 0 deg because that posture looks cleaner and more natural.
J4_ZERO_SOFT_SCALE_DEG = 45.0
J4_ZERO_SOFT_WEIGHT = 0.5

# Combined normal motion is a soft optimization term only.
# Hard safety comes from actual USD triangle-mesh penetration.
COMBINED_NORMAL_IS_SOFT_ONLY = True

# Adaptive unload: fingers remain flexed until the actual USD mesh says they
# are clear of the back. Only then may OPEN_AFTER_UNLOAD happen.
UNLOAD_REQUIRED_DISTANCE_M = 0.010
UNLOAD_REQUIRED_SIGNED_M = 0.003
UNLOAD_OFFSETS_M = [
    0.012,
    0.018,
    0.025,
    0.035,
    0.050,
]
# Normal release uses the wrist only: positive J5 lifts the curled fingertips
# away from the back. Whole-arm normal-offset IK remains a safety fallback.
UNLOAD_J5_MIN_LIFT_DEG = 8.0
UNLOAD_J5_MAX_LIFT_DEG = 15.0
# Add a small margin after the first geometrically safe unload angle.  The
# planner's mesh check is nominal; the extra margin absorbs drive/settling
# error seen during the actual unload transition.
UNLOAD_J5_SAFETY_MARGIN_DEG = 2.0
UNLOAD_J5_LIFT_CANDIDATES_DEG = tuple(
    0.5 * step
    for step in range(
        int(round(UNLOAD_J5_MIN_LIFT_DEG / 0.5)),
        int(round(UNLOAD_J5_MAX_LIFT_DEG / 0.5)) + 1,
    )
)
UNLOAD_J5_SPEED_DEG_S = 30.0

# Multi-stage Cartesian-style approach along the local back normal.
# The hand is oriented while still far from the back, then follows the same
# IK branch inward in small increments.
APPROACH_OFFSETS_M = [
    0.060,
    0.045,
    0.030,
    0.025,
    0.020,
    0.016,
    0.012,
]
BACK_SURFACE_ROOT = "/World/BackSurface"
BACK_CONTACT_ROOTS = (BACK_SURFACE_ROOT, "/World/DeformableBackV2")
CONTACT_DIAGNOSTIC_LOG_INTERVAL_S = 0.5
MESH_QUERY_NEAREST_TRIANGLES = 32
BACK_ALIGNMENT_SAMPLE_COUNT = 384
BACK_ALIGNMENT_WARN_P95_M = 0.010
TARGET_BACK_ALIGNMENT_HARD_M = 0.020
REQUEST_TARGET_XYZ_TOL_M = 0.0005


MAX_ACTIVE_SURFACE_DISTANCE_M = 0.009

MIN_MAIN_PHASE_DEG = 1.0  # percent: do not reuse old MCP minimum bounds
MAX_MAIN_PHASE_DEG = 95.0  # leave some stroke after acquisition
MIN_CONTACT_PHASE_DEG = 20.0
MIN_CONTACT_SPAN_DEG = 8.0
MIN_USEFUL_STROKE_M = 0.008
MIN_TANGENT_SCORE = 0.40

STRONG_NORMAL_RMSE_M = 0.0015
STRONG_TANGENT_SCORE = 0.70
STRONG_STROKE_M = 0.012

IK_POSITION_TOL_M = 0.006
IK_ORIENTATION_TOL_RAD = [0.35, 0.60, 0.90, 1.20]
ENABLE_POSITION_ONLY_IK = True
ENABLE_Q_REFINEMENT_FALLBACK = True
MIN_JOINT_MARGIN_DEG = 1.0

PALM_TRANSLATION_SEARCH_M = 0.040
PALM_OPT_MAXITER = 50
Q_REFINE_MAXITER = 32

FLAT_STRATEGY_YAWS_DEG = [0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0]
V_STRATEGY_YAWS_DEG = [-90.0, 90.0, -60.0, 60.0, -30.0, 30.0, 0.0]

PRE_APPROACH_OFFSET_M = 0.030
UNLOAD_ALL_OFFSET_M = 0.012
SAFE_RETREAT_OFFSET_M = 0.050

ARM_MAX_SPEED_DEG_S = 18.0
MIN_ARM_MOVE_S = 0.45
MAX_ARM_MOVE_S = 3.5

FINGER_PRE_FLEX_DEG = 40.0
FINGER_PRE_FLEX_TIME_S = 0.30
FINGER_JOIN_TIME_S = 0.65

FULL_SCRATCH_HOLD_S = 0.35
UNLOAD_ALL_TIME_S = 0.50
OPEN_AFTER_UNLOAD_TIME_S = 0.50
RETURN_HOME_MIN_TIME_S = 1.20

REQUEST_CHECK_PERIOD_S = 0.20
SAFETY_CHECK_PERIOD_S = 0.035

CONTACT_LOSS_DISTANCE_M = 0.018
CONTACT_LOSS_TIMEOUT_S = 1.0

# ============================================================
# V4.3 TRAJECTORY-AWARE / SYNCHRONIZED SCRATCH
# ============================================================

# A candidate is not executable until its COMPLETE trajectory is safe.  Keep a
# wider bounded pool so endpoint IK variability cannot hide a good posture
# (e.g. a ring-first 909 solution) behind the first few valid seeds.  The
# current 909 endpoint search returns 9 valid 4F poses, so test all 9 rather
# than leaving one unverified.  This changes search coverage only; every
# retained candidate still goes through the same full four-finger
# swept/lane/depth checks before execution.
MAX_ENDPOINT_VALID_POOL = 9

# During scratching, inactive/future fingers may be slightly flexed in the air
# to avoid sweeping the back. The planner tries these automatically.
INACTIVE_CLEARANCE_OPTIONS_DEG = [0.0, 35.0, 45.0, 55.0, 65.0]

# Contact begins at about 60 deg. From there to the selected main phase,
# each finger is advanced through several synchronized arm/wrist keyframes.
SYNC_CONTACT_ONSET_DEG = 52.0
SYNC_PHASE_STEP_DEG = 4.0
SYNC_KEYFRAME_MIN_TIME_S = 0.16
SYNC_KEYFRAME_MAX_TIME_S = 0.42

# Combined-motion scratch quality.
# These are evaluated on the REAL world-space fingertip displacement after
# finger flex + J1..J5 compensation are combined.
MIN_COMBINED_TANGENT_RATIO = 0.55
MIN_COMBINED_TANGENTIAL_KEYFRAME_M = 0.0008
# V102 uses intentionally small ~3 degree micro-keyframes for continuity. The
# old 0.8 mm threshold was calibrated for much larger synchronized steps and
# can reject a valid 0.32 mm micro-stroke. Aggregate wave travel remains large;
# each micro-step must still show real non-zero tangential motion.
ASYNC_MIN_COMBINED_TANGENTIAL_KEYFRAME_M = 0.00020
MAX_COMBINED_NORMAL_KEYFRAME_M = 0.0025

# Existing contact fingers that are not the currently moving finger should
# remain nearly depth-stationary while J4/J5 adapt the hand.
MAX_EXISTING_CONTACT_NORMAL_SHIFT_M = 0.0020


# V4.5 contact-acquisition layer.
#
# V4.4 incorrectly declared the new finger "must already be in contact" at
# exactly 52 deg while the palm was still at the 30 mm pre-approach pose.
# That is why almost every candidate died at:
#     SYNC_CONTACT_FAIL_PINKY_52.0
#
# V4.5 separates:
#     APPROACH -> ACQUIRE CONTACT -> SYNCHRONIZED SCRATCH.
#
# Contact acquisition is allowed to be mainly normal motion. Tangential
# scratch-quality constraints start only AFTER the finger has established
# contact.
CONTACT_NEAR_OFFSET_M = 0.012
ACQUIRE_PHASE_START_DEG = 48.0
ACQUIRE_PHASE_STEP_DEG = 4.0
ACQUIRE_MIN_SYNC_STEP_DEG = 2.0
ACQUIRE_MAXITER = 48

# V101 full-stroke handoff policy.  V100 optimized the synchronized handoff
# over 12-30 degrees and softly preferred 18 degrees.  In practice it often
# selected roughly 30 degrees, which visually consumed the first part of the
# scratch before contact acquisition even began.  Prefer an open hand and only
# permit a small pre-bend when the 5-DOF reachability solve genuinely needs it.
HANDOFF_PREBEND_SEARCH_BANDS_DEG = (
    (0.0, 12.0),
    (12.0, 18.0),
    (18.0, 24.0),
)
HANDOFF_PREBEND_PREFERENCE_SCALE_DEG = 6.0
HANDOFF_PREBEND_MIN_CLEARANCE_M = 0.012

# First contact may require more arm motion than a 4-degree scratch micro-step.
# J4/J5 still have the largest local freedom.
ACQUIRE_LOCAL_DELTA_DEG = np.array(
    [10.0, 28.0, 28.0, 42.0, 46.0],
    dtype=np.float64,
)

# J4/J5 are deliberately allowed to do more of the local contact adjustment.
# Per synchronized micro-step:
#   J1 changes very little,
#   J2/J3 may compensate moderately,
#   J4/J5 may move more freely.
SYNC_LOCAL_DELTA_DEG = np.array(
    [5.0, 10.0, 10.0, 18.0, 22.0],
    dtype=np.float64,
)

# Continuity weights: larger = more expensive to move.
# This encodes the intended behavior:
# gross arm placement by J1/J2/J3, local scratch fitting mainly by J4/J5.
SYNC_JOINT_MOVE_WEIGHTS = np.array(
    [3.0, 1.5, 1.5, 0.55, 0.40],
    dtype=np.float64,
)

# Joint5 semantics for this robot:
# negative -> press/load toward back
# positive -> lift/unload away from back
# V4.7: No artificial J5 angle preference or sign limit.
# J5 is constrained only by the robot joint limits and geometric safety.



# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def rotation_delta_deg(reference, actual):
    """Smallest orientation difference between two 3x3 rotation matrices."""
    try:
        delta = np.asarray(reference, dtype=np.float64).T @ np.asarray(
            actual,
            dtype=np.float64,
        )
        return float(math.degrees(Rotation.from_matrix(delta).magnitude()))
    except Exception:
        return 999.0


def rotation_matrix_to_wxyz(rotation):
    """Convert a 3x3 matrix to Isaac/Lula's scalar-first quaternion format."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            "LULA_IK_ROTATION_MATRIX_SHAPE=" + str(tuple(matrix.shape))
        )
    quaternion_xyzw = Rotation.from_matrix(matrix).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm < 1e-12:
        raise ValueError("LULA_IK_ZERO_QUATERNION")
    return quaternion_wxyz / norm


def depth_zone_from_signed(signed):
    """Classify scratch depth using the V3 0-4 / 4-8 / >8 mm policy."""
    signed = float(signed)

    if signed < DEEP_ADJUST_HARD_SIGNED_M:
        return "HARD_REJECT"

    if signed < DEEP_ADJUST_ENTER_SIGNED_M:
        return "DEEP_ADJUST"

    return "NORMAL"


def update_deep_adjust_latch(is_latched, signed):
    """4.0 mm enters DEEP_ADJUST; 3.5 mm exits it (hysteresis)."""
    signed = float(signed)

    if signed < DEEP_ADJUST_HARD_SIGNED_M:
        return True

    if is_latched:
        return signed < DEEP_ADJUST_EXIT_SIGNED_M

    return signed < DEEP_ADJUST_ENTER_SIGNED_M


def deep_adjust_soft_penalty(signed):
    """Softly prefer returning from 4-8 mm toward the normal <=4 mm zone."""
    signed = float(signed)

    excess = max(
        0.0,
        DEEP_ADJUST_ENTER_SIGNED_M - signed,
    )

    return (
        DEEP_ADJUST_PENALTY_WEIGHT
        *
        (
            excess
            /
            max(DEEP_ADJUST_PENALTY_SCALE_M, 1e-6)
        ) ** 2
    )


def smoothstep(v):
    x = clamp(float(v), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def empty_fingers():
    return {
        "thumb": 0.0,
        "index": 0.0,
        "middle": 0.0,
        "ring": 0.0,
        "pinky": 0.0,
    }


def safe_write_json(path, data, retries=10):
    tmp = path + "." + str(os.getpid()) + "." + str(time.time_ns()) + ".tmp"

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
    except Exception as exc:
        print("[V4 WARNING] temp JSON write failed:", repr(exc))
        return False

    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except (PermissionError, OSError):
            time.sleep(0.015)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.isfile(tmp):
            os.remove(tmp)
        return True
    except Exception as exc:
        print("[V4 WARNING] status JSON write failed:", repr(exc))
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


# ============================================================
# USD / ARTICULATION HELPERS
# ============================================================

def get_world_pose(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError("Prim not found: " + prim_path)

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    t = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()
    qi = q.GetImaginary()

    position = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    quat = np.array(
        [float(q.GetReal()), float(qi[0]), float(qi[1]), float(qi[2])],
        dtype=np.float64,
    )
    quat /= np.linalg.norm(quat)

    return position, quat


def get_world_position(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    t = matrix.ExtractTranslation()

    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def find_articulation_root(stage):
    preferred = ROBOT_ROOT + "/root_joint"
    prim = stage.GetPrimAtPath(preferred)

    if prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return preferred

    for candidate in stage.TraverseAll():
        path = str(candidate.GetPath())
        if path.startswith(ROBOT_ROOT) and candidate.HasAPI(UsdPhysics.ArticulationRootAPI):
            return path

    raise RuntimeError("ArticulationRootAPI not found under robot.")


def find_joint_prim(stage, joint_name):
    direct = stage.GetPrimAtPath(ROBOT_ROOT + "/joints/" + joint_name)
    if direct.IsValid():
        return direct

    for prim in stage.TraverseAll():
        path = str(prim.GetPath())

        if not path.startswith(ROBOT_ROOT):
            continue

        if prim.GetName() == joint_name and prim.IsA(UsdPhysics.RevoluteJoint):
            return prim

    raise RuntimeError("Joint prim not found: " + joint_name)


def get_drive(stage, joint_name):
    drive = UsdPhysics.DriveAPI.Get(find_joint_prim(stage, joint_name), "angular")

    if not drive:
        raise RuntimeError("Angular drive missing: " + joint_name)

    return drive


def set_drive_deg(drive, value):
    """Author in the same strongest layer as hand setup, then verify resolution.

    Attribute.Set succeeding in a weaker layer does NOT mean PhysX sees that
    value. In particular, setup's session-layer zero used to mask every PIP/DIP
    command. EditContext is scoped so unrelated UI edits keep their own target.
    """
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("NONFINITE_DRIVE_TARGET")
    attr = drive.GetTargetPositionAttr()
    if not attr:
        raise RuntimeError("MISSING_DRIVE_TARGET_ATTRIBUTE")
    stage = drive.GetPrim().GetStage()
    layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, layer):
        if not attr.Set(value):
            raise RuntimeError("DRIVE_TARGET_WRITE_FAILED|path=" + str(attr.GetPath()))
    resolved = attr.Get()
    if (resolved is None or not math.isfinite(float(resolved))
            or abs(float(resolved) - value) > 1e-5):
        raise RuntimeError(
            "DRIVE_TARGET_NOT_EFFECTIVE|path=" + str(attr.GetPath())
            + "|requested_deg=" + str(value) + "|resolved_deg=" + str(resolved)
            + "|write_layer=" + str(layer.identifier))


def checked_finger_commands(fingers):
    """Reject, never silently truncate, a plan that asks to push past a stop."""
    result = {}
    for finger in LONG_FINGERS:
        value = float(fingers.get(finger, 0.0))
        if (not math.isfinite(value) or value < -0.05
                or value > FINGER_SAFE_COMMAND_MAX_DEG + 0.05):
            raise RuntimeError("FINGER_STOP_COMMAND_REJECT_%s_%.3f" % (finger, value))
        result[finger] = clamp(value, 0.0, FINGER_SAFE_COMMAND_MAX_DEG)
    return result


def keyframe_ready_to_advance(arrived, settled_s, required_s):
    return bool(arrived and settled_s >= required_s)


def path_is_within(path, root):
    return bool(path == root or path.startswith(root + "/"))


def contact_sample_for_log(header, contact_data):
    """Copy a bounded sample; separation is NOT tissue indentation or force."""
    try:
        offset, count = int(header.contact_data_offset), int(header.num_contact_data)
        if offset < 0 or count < 0 or offset + count > len(contact_data):
            return {"status": "INVALID_CONTACT_DATA_RANGE"}
        if count == 0:
            return {"status": "NO_CONTACT_POINTS_REPORTED", "count": 0}
        # Inspect a bounded prefix, not a claimed global deepest contact.
        samples = []
        for i in range(offset, offset + min(count, 3)):
            item = contact_data[i]
            position = [float(item.position[j]) for j in range(3)]
            normal = [float(item.normal[j]) for j in range(3)]
            separation = float(item.separation)
            if not all(math.isfinite(v) for v in position + normal + [separation]):
                return {"status": "NONFINITE_CONTACT_DATA"}
            samples.append(dict(world_position_m=position, normal=normal,
                                separation_mm=separation * 1000.0))
        return dict(status="REPORTED", count=count, first_points=samples)
    except Exception as exc:
        return dict(status="CONTACT_POINTS_UNAVAILABLE", reason=repr(exc))


def contact_lift_stall_update(state, key, elapsed_s, error_deg, tolerance_deg, eligible):
    """Observe one joint against the SAME tolerance used to advance its frame.

    A fresh window is required after each lift, joint switch or loss of eligibility.
    elapsed_s is the total arrival wait; relief must never restart the timeout.
    """
    if (not eligible or not all(math.isfinite(float(v)) for v in
            (elapsed_s, error_deg, tolerance_deg)) or error_deg <= tolerance_deg):
        return None, False
    if (state is None or state["key"] != key or elapsed_s < state["start_s"]):
        return dict(key=key, start_s=elapsed_s, error_deg=error_deg), False
    if elapsed_s - state["start_s"] < CONTACT_J5_LIFT_STEP_INTERVAL_S - 1e-9:
        return state, False
    stalled = state["error_deg"] - error_deg < CONTACT_J5_LIFT_MIN_PROGRESS_DEG
    return dict(key=key, start_s=elapsed_s, error_deg=error_deg), bool(stalled)


def check_finger_stop_model(stage):
    """Read-only audit: stop rather than mixing new arcs with the old USD hand."""
    HAND_MODEL.audit_scene(stage)
    HAND_MODEL.verify_arc_csv(ARC_CSV, COLLISION_URDF)
    print("[V102 HAND MODEL VERIFIED]", HAND_MODEL.MODEL_ID,
          "MCP/PIP/DIP limits=", HAND_MODEL.LIMITS_DEG,
          "source=URDF_FK_SIMULATION_REFERENCE_NOT_MEASURED")


def articulation_index(art, name):
    if hasattr(art, "get_dof_index"):
        return int(art.get_dof_index(name))
    if hasattr(art, "get_joint_index"):
        return int(art.get_joint_index(name))
    raise RuntimeError("No articulation index lookup method.")


def articulation_positions(art, indices):
    indices = np.asarray(indices, dtype=np.int64)

    try:
        values = art.get_joint_positions(joint_indices=indices)
    except TypeError:
        values = art.get_joint_positions(indices)

    if values is None:
        raise RuntimeError("Unable to read actual articulation joint positions.")

    return np.asarray(values, dtype=np.float64).reshape(-1)


def clear_old_debug(stage):
    # Exact scratch-debug roots only. Never search generic words such as "center".
    for path in [
        "/World/MultiFingerScratchStrategyV2",
        "/World/MultiFingerScratchStrategyV3",
        "/World/MultiFingerScratchStrategyV3_1",
        "/World/MultiFingerScratchStrategyV4",
        "/World/ScratchDebug",
        "/World/FingerScratchArcs",
        "/World/CenterScratchTest",
        "/World/ScratchPoseDebug",
        "/World/ScratchMarkers",
        V100_DEBUG_ROOT,
        V103_ACTUAL_TRACE_ROOT,
    ]:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)
            print("[V4] Removed old debug:", path)


def _debug_vec3(value):
    value = np.asarray(value, dtype=np.float64)
    return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))


def _define_debug_curve(stage, path, points, color, width=None):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([int(len(points))])
    curve.CreatePointsAttr([_debug_vec3(point) for point in points])
    curve.CreateWidthsAttr([
        float(V100_DEBUG_CURVE_WIDTH_M if width is None else width)
    ])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([_debug_vec3(color)])
    return curve


def _define_debug_sphere(stage, path, position, color, radius=0.006):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    sphere.CreateDisplayColorAttr([_debug_vec3(color)])
    return sphere


def _actual_trace_is_scratch_stage(stage_name):
    """Return whether a measured tip sample belongs to contact/scratch work.

    Approach, unload and Home samples are retained in the JSON trace, but are
    drawn as a thin reference line.  The thick line is reserved for the
    contact acquisition and scratch portion that the user actually wants to
    inspect.  This prevents a long Home return from hiding the scratch path.
    """
    name = str(stage_name)
    return name.startswith((
        "ARC_CONTACT_",
        "ARC_SYNC_CONTACT_",
        "ARC_ASYNC_CONTACT_",
        "DEEP_ADJUST_",
        "SCRATCH_WAVE_ASYNC_",
        "SCRATCH_CURL_",
        "FINGER_CURL_RELEASE_",
        "FULL_SCRATCH",
    ))


def _actual_trace_is_unload_stage(stage_name):
    """Return whether a measured sample belongs to unload/Home motion.

    These samples remain in the JSON trace for auditing, but drawing them in
    the same colored reference curve as the scratch motion makes the J5 lift
    look like an extra scratch stroke.
    """
    name = str(stage_name)
    return name.startswith((
        "UNLOAD_",
        "OPEN_AFTER_UNLOAD",
        "RETURN_HOME",
        "RETREAT_",
        "HOME_WAIT",
    ))


def _downsample_actual_trace(entries, maximum=V103_ACTUAL_TRACE_MAX_POINTS):
    """Keep endpoints and uniformly thin a dense live marker trace."""
    entries = list(entries or [])
    if len(entries) <= int(maximum):
        return entries
    count = max(2, int(maximum))
    indices = np.linspace(0, len(entries) - 1, count, dtype=np.int64)
    return [entries[int(i)] for i in np.unique(indices)]


def draw_actual_tip_trajectories(stage, traces, target=None):
    """Draw measured USD tip-link paths separately from planned V100 arcs.

    ``traces`` is populated by :meth:`IntegratedScratchV4.safety_check` and
    contains world-space positions read from the live tip-link prims.  The
    planned curves remain under ``/World/V100_PlanningDebug``; measured paths
    live under their own root and are intentionally thicker/bright so a visual
    review cannot mistake a planner proposal for executed motion.
    """
    if stage.GetPrimAtPath(V103_ACTUAL_TRACE_ROOT).IsValid():
        stage.RemovePrim(V103_ACTUAL_TRACE_ROOT)
    root = UsdGeom.Xform.Define(stage, V103_ACTUAL_TRACE_ROOT)
    colors = {
        "index": np.array([1.0, 0.20, 0.10]),
        "middle": np.array([0.10, 1.0, 0.20]),
        "ring": np.array([0.10, 0.75, 1.0]),
        "pinky": np.array([1.0, 0.15, 0.85]),
    }
    serialized = {
        "schema": 1,
        "source": "ACTUAL_USD_TIP_LINK_WORLD_POSES",
        "planned_trace_is_not_used": True,
        "target_xyz_m": None if target is None else [float(v) for v in target],
        "finger_order": list(LONG_FINGERS),
        "traces": {},
    }
    total_samples = 0
    scratch_samples = 0
    for finger in LONG_FINGERS:
        entries = []
        for entry in (traces or {}).get(finger, ()):
            point = np.asarray(entry.get("world_xyz_m", ()), dtype=np.float64)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                continue
            row = {
                "sim_time_s": float(entry.get("sim_time_s", 0.0)),
                "stage": str(entry.get("stage", "UNKNOWN")),
                "world_xyz_m": [float(v) for v in point],
                "scratch": bool(entry.get("scratch", False)),
                "active": bool(entry.get("active", False)),
            }
            entries.append(row)
        if not entries:
            continue
        total_samples += len(entries)
        scratch_entries = [row for row in entries if row["scratch"]]
        scratch_samples += len(scratch_entries)
        serialized["traces"][finger] = entries
        finger_root = V103_ACTUAL_TRACE_ROOT + "/" + finger
        # Keep unload/J5-lift samples in the saved trace, but do not connect
        # them to the colored execution curve shown in the viewport.
        visible_entries = [
            row for row in entries
            if not _actual_trace_is_unload_stage(row["stage"])
        ]
        all_points = [row["world_xyz_m"] for row in
                      _downsample_actual_trace(visible_entries)]
        # Keep the complete measured execution as a thin reference curve.  The
        # bright curve below is the scratch-only portion, but the approach,
        # unload, second cycle and Home return must remain visible too.
        if len(all_points) >= 2:
            _define_debug_curve(
                stage, finger_root + "/ActualExecutionTrace", all_points,
                0.32 * colors[finger] + 0.12,
                width=V103_ACTUAL_TRACE_REFERENCE_WIDTH_M,
            )
        scratch_points = [row["world_xyz_m"] for row in
                          _downsample_actual_trace(scratch_entries)]
        if len(scratch_points) >= 2:
            _define_debug_curve(
                stage, finger_root + "/ActualScratchTrace", scratch_points,
                colors[finger], width=V103_ACTUAL_TRACE_WIDTH_M,
            )
            _define_debug_sphere(
                stage, finger_root + "/ActualScratchStart", scratch_points[0],
                colors[finger], radius=0.005,
            )
            _define_debug_sphere(
                stage, finger_root + "/ActualScratchEnd", scratch_points[-1],
                colors[finger], radius=0.007,
            )
        elif len(all_points) >= 2:
            _define_debug_sphere(
                stage, finger_root + "/ActualExecutionEnd", all_points[-1],
                colors[finger], radius=0.006,
            )
        UsdGeom.Xform.Define(stage, finger_root).GetPrim().SetCustomDataByKey(
            "legend", finger + ": thick=actual scratch; thin=all actual execution"
        )

    root.GetPrim().SetCustomDataByKey("source", "LIVE_USD_TIP_LINK_WORLD_POSES")
    root.GetPrim().SetCustomDataByKey(
        "legend", "thick curves=actual scratch; thin curves=all actual motion; planned arcs remain in V100_PlanningDebug"
    )
    root.GetPrim().SetCustomDataByKey("sample_count", int(total_samples))
    root.GetPrim().SetCustomDataByKey("scratch_sample_count", int(scratch_samples))
    try:
        safe_write_json(ACTUAL_TIP_TRACE_JSON, serialized)
    except Exception as exc:
        print("[V103 ACTUAL TIP TRACE SAVE FAILED]", repr(exc))
    print(
        "[V103 ACTUAL TIP TRACE] root=", V103_ACTUAL_TRACE_ROOT,
        "samples=", total_samples,
        "scratch_samples=", scratch_samples,
        "legend=index:red middle:green ring:cyan pinky:magenta;",
        "thick=actual_scratch thin=all_actual_motion",
    )
    return serialized


def draw_v100_candidate_debug(stage, solver, arcs, candidate, mode):
    """Draw the actual selected finger arcs and required endpoint palm pose."""
    if stage.GetPrimAtPath(V100_DEBUG_ROOT).IsValid():
        stage.RemovePrim(V100_DEBUG_ROOT)
    root = UsdGeom.Xform.Define(stage, V100_DEBUG_ROOT)

    q = np.asarray(candidate["q"], dtype=np.float64)
    template = candidate["template"]
    phases = template["phases"]
    palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
    palm_p = np.asarray(palm_p, dtype=np.float64)
    palm_r = np.asarray(palm_r, dtype=np.float64)
    normal = normalize(mode["normal"])
    anchor = np.asarray(mode["anchor"], dtype=np.float64)

    colors = {
        "index": np.array([1.0, 0.20, 0.10]),
        "middle": np.array([0.10, 1.0, 0.20]),
        "ring": np.array([0.10, 0.75, 1.0]),
        "pinky": np.array([1.0, 0.15, 0.85]),
    }
    height_diagnostics = {}
    for finger in template["active"]:
        final_phase = float(phases[finger])
        full_angles = np.linspace(
            float(arcs[finger]["angles"][0]),
            float(arcs[finger]["angles"][-1]),
            91,
        )
        full_world_points = []
        for angle in full_angles:
            local_point, _ = interpolate_arc(arcs, finger, float(angle))
            full_world_points.append(palm_p + palm_r @ local_point)
        full_world_points = np.asarray(full_world_points, dtype=np.float64)
        sample_angles = np.linspace(0.0, final_phase, 61)
        world_points = []
        for angle in sample_angles:
            local_point, _ = interpolate_arc(arcs, finger, float(angle))
            world_points.append(palm_p + palm_r @ local_point)
        world_points = np.asarray(world_points, dtype=np.float64)
        finger_root = V100_DEBUG_ROOT + "/FingerArcs/" + finger
        # Keep the full measured arc only for numeric diagnostics. Drawing it
        # produced the large loops that obscured the real scratch. Show only
        # the selected planning arc until measured execution replaces it.
        _define_debug_curve(
            stage, finger_root + "/SelectedArc", world_points, colors[finger],
            width=0.0020,
        )
        _define_debug_sphere(
            stage, finger_root + "/StraightPoint", world_points[0],
            np.array([1.0, 1.0, 1.0]), radius=0.005,
        )
        _define_debug_sphere(
            stage, finger_root + "/SelectedContact", world_points[-1],
            colors[finger], radius=0.007,
        )
        projections = (world_points - anchor[None, :]) @ normal
        min_index = int(np.argmin(projections))
        full_projections = (full_world_points - anchor[None, :]) @ normal
        full_min_index = int(np.argmin(full_projections))
        _define_debug_sphere(
            stage, finger_root + "/LowestMeasuredArcPoint", full_world_points[full_min_index],
            np.array([0.05, 0.05, 0.05]), radius=0.004,
        )
        height_diagnostics[finger] = {
            "selected_phase_deg": round(final_phase, 2),
            "lowest_used_phase_deg": round(float(sample_angles[min_index]), 2),
            "straight_height_mm": round(float(projections[0]) * 1000.0, 2),
            "selected_height_mm": round(float(projections[-1]) * 1000.0, 2),
            "lowest_height_mm": round(float(projections[min_index]) * 1000.0, 2),
            "full_lowest_phase_deg": round(float(full_angles[full_min_index]), 2),
            "full_lowest_height_mm": round(float(full_projections[full_min_index]) * 1000.0, 2),
        }

    # GUI target and outward normal.
    _define_debug_sphere(
        stage, V100_DEBUG_ROOT + "/Target/GUI_Target", anchor,
        np.array([1.0, 1.0, 0.0]), radius=0.009,
    )
    _define_debug_curve(
        stage,
        V100_DEBUG_ROOT + "/Target/BackOutwardNormal",
        [anchor, anchor + V100_DEBUG_AXIS_LENGTH_M * normal],
        np.array([1.0, 1.0, 0.0]),
        width=0.004,
    )
    patch_u, patch_v = tangent_basis(normal)
    circle_angles = np.linspace(0.0, 2.0 * math.pi, 65)
    patch_circle = [
        anchor
        + PLANNED_ACTIVE_PATCH_RADIUS_M
        * (math.cos(angle) * patch_u + math.sin(angle) * patch_v)
        for angle in circle_angles
    ]
    _define_debug_curve(
        stage,
        V100_DEBUG_ROOT + "/Target/PlannedPatchBoundary",
        patch_circle,
        np.array([1.0, 0.75, 0.0]),
        width=0.002,
    )

    # Required palm pose: local XYZ axes plus the measured four-finger forward
    # direction.  The white arrow is the important root-to-fingers direction.
    axis_colors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.35, 1.0]),
    ]
    for axis_index, axis_name in enumerate(["PalmX", "PalmY", "PalmZ"]):
        _define_debug_curve(
            stage,
            V100_DEBUG_ROOT + "/PalmPose/" + axis_name,
            [palm_p, palm_p + V101_DEBUG_PALM_AXIS_LENGTH_M * palm_r[:, axis_index]],
            axis_colors[axis_index],
            width=0.004,
        )
    finger_forward = normalize(palm_r @ get_palm_local_forward(arcs))
    finger_root_world = palm_p + palm_r @ get_four_finger_root_local_center()
    _define_debug_curve(
        stage,
        V100_DEBUG_ROOT + "/PalmPose/FourFingerForward3D",
        [finger_root_world,
         finger_root_world + V100_DEBUG_AXIS_LENGTH_M * finger_forward],
        np.array([0.55, 0.55, 0.55]),
        width=0.002,
    )
    _define_debug_sphere(
        stage, V100_DEBUG_ROOT + "/PalmPose/PalmOrigin", palm_p,
        np.array([1.0, 1.0, 1.0]), radius=0.007,
    )

    # The requested constraint is two dynamic top-view collinear axes, not a
    # fixed plane. Orange: J1/base radial axis at hand-root height. White:
    # projected hand-root -> four-finger axis.
    root_axis = root_base_axis_alignment(arcs, palm_p, palm_r)
    if ROBOT_BASE_WORLD_POSITION is not None and bool(root_axis.get("available", False)):
        base_p = np.asarray(ROBOT_BASE_WORLD_POSITION, dtype=np.float64)
        j1_axis = np.asarray(root_axis["j1_axis"], dtype=np.float64)
        root_world = np.asarray(root_axis["root_world"], dtype=np.float64)
        base_axis_start = root_world - float(np.dot(root_world - base_p, j1_axis)) * j1_axis
        _define_debug_curve(
            stage, V100_DEBUG_ROOT + "/RootBaseAxis/BaseToHandRoot",
            [base_axis_start, root_world], np.array([1.0, 0.25, 0.0]), width=0.005,
        )
        _define_debug_sphere(
            stage, V100_DEBUG_ROOT + "/RootBaseAxis/BaseAxisPoint", base_axis_start,
            np.array([1.0, 0.25, 0.0]), radius=0.008,
        )
        _define_debug_sphere(
            stage, V100_DEBUG_ROOT + "/RootBaseAxis/FourFingerRootCenter", root_world,
            np.array([1.0, 1.0, 1.0]), radius=0.008,
        )
        _define_debug_curve(
            stage, V100_DEBUG_ROOT + "/RootBaseAxis/FourFingerProjectedAxis",
            [root_world,
             root_world + V100_DEBUG_AXIS_LENGTH_M * np.asarray(root_axis["finger_axis"])],
            np.array([1.0, 1.0, 1.0]), width=0.005,
        )

    candidate_valid = bool(candidate.get("valid", False))
    candidate_path_safe = bool(candidate.get("trajectory_safe", False))
    if not candidate_valid and not candidate_path_safe:
        # Visible red X means this is only the best rejected diagnostic pose.
        mark_u, mark_v = tangent_basis(normal)
        size = 0.025
        _define_debug_curve(
            stage, V100_DEBUG_ROOT + "/INVALID_DIAGNOSTIC/X1",
            [palm_p - size * mark_u - size * mark_v,
             palm_p + size * mark_u + size * mark_v],
            np.array([1.0, 0.0, 0.0]), width=0.007,
        )
        _define_debug_curve(
            stage, V100_DEBUG_ROOT + "/INVALID_DIAGNOSTIC/X2",
            [palm_p - size * mark_u + size * mark_v,
             palm_p + size * mark_u - size * mark_v],
            np.array([1.0, 0.0, 0.0]), width=0.007,
        )

    root.GetPrim().SetCustomDataByKey("template", str(template["name"]))
    root.GetPrim().SetCustomDataByKey(
        "endpoint_q_deg", str([round(float(v), 3) for v in np.degrees(q)])
    )
    root.GetPrim().SetCustomDataByKey("height_diagnostics", str(height_diagnostics))
    root.GetPrim().SetCustomDataByKey("candidate_valid", candidate_valid)
    root.GetPrim().SetCustomDataByKey("candidate_path_safe", candidate_path_safe)
    root.GetPrim().SetCustomDataByKey(
        "candidate_reject", str(candidate.get("geometry", {}).get("reject", "UNKNOWN"))
    )
    print(
        "[V100 VISUAL DEBUG] root=", V100_DEBUG_ROOT,
        "template=", template["name"],
        "candidate_status=", ("STRICT_VALID" if candidate_valid else
                              "FULL_PATH_VALIDATED" if candidate_path_safe else
                              "INVALID_DIAGNOSTIC_ONLY"),
        "root_base_axis_deg=", round(float(root_axis["angle_deg"]), 2),
        "legend=index:red middle:green ring:cyan pinky:magenta "
        "planning-lines:selected-arcs-only "
        "white-spheres:straight colored-spheres:selected black-spheres:lowest "
        "yellow:target/normal white:projected-finger-axis "
        "orange:dynamic-base-root-axis gray:3D-finger-forward "
        "red-X:invalid-diagnostic",
    )
    print("[V100 VISUAL ARC HEIGHTS]", height_diagnostics)


# ============================================================
# BACK / FINGER DATA
# ============================================================

def load_back(path):
    points = []
    normals = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            points.append(
                [
                    float(row["surface_x_m"]),
                    float(row["surface_y_m"]),
                    float(row["surface_z_m"]),
                ]
            )

            n = normalize(
                [
                    float(row["normal_x"]),
                    float(row["normal_y"]),
                    float(row["normal_z"]),
                ]
            )

            if n is None:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            normals.append(n)

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    if len(points) == 0:
        raise RuntimeError("Back candidate CSV is empty.")

    return points, normals


def load_arcs(path):
    raw = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            finger = row["finger"].strip().lower()

            raw.setdefault(finger, {"angles": [], "points": []})

            raw[finger]["angles"].append(float(row["progress_pct"]))
            raw[finger]["points"].append(
                [
                    float(row["local_x_m"]),
                    float(row["local_y_m"]),
                    float(row["local_z_m"]),
                ]
            )

    arcs = {}

    for finger, data in raw.items():
        angles = np.asarray(data["angles"], dtype=np.float64)
        points = np.asarray(data["points"], dtype=np.float64)

        order = np.argsort(angles)
        angles = angles[order]
        points = points[order]

        # Use the new model-derived progress curve, never the old MCP-angle CSV.
        # Resampling the cap avoids np.interp silently using an out-of-range arc.
        cap = FINGER_SAFE_COMMAND_MAX_DEG
        if len(angles) < 2 or angles[0] > 0.0 or angles[-1] < cap:
            raise RuntimeError("ARC_DOES_NOT_COVER_SAFE_CURL_" + finger)
        cap_point = np.array([np.interp(cap, angles, points[:, i]) for i in range(3)])
        keep = angles < cap
        points = np.vstack((points[keep], cap_point))
        angles = np.append(angles[keep], cap)

        tangents = np.zeros_like(points)

        if len(points) >= 2:
            tangents[0] = points[1] - points[0]
            tangents[-1] = points[-1] - points[-2]

        for i in range(1, len(points) - 1):
            tangents[i] = points[i + 1] - points[i - 1]

        for i in range(len(tangents)):
            t = normalize(tangents[i])
            if t is None:
                t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tangents[i] = t

        arcs[finger] = {"angles": angles, "points": points, "tangents": tangents}

    for finger in LONG_FINGERS:
        if finger not in arcs:
            raise RuntimeError("Missing arc model: " + finger)

    return arcs


def interpolate_arc(arcs, finger, angle_deg):
    arc = arcs[finger]

    angle_deg = clamp(
        float(angle_deg),
        float(arc["angles"][0]),
        float(arc["angles"][-1]),
    )

    point = np.array(
        [
            np.interp(angle_deg, arc["angles"], arc["points"][:, 0]),
            np.interp(angle_deg, arc["angles"], arc["points"][:, 1]),
            np.interp(angle_deg, arc["angles"], arc["points"][:, 2]),
        ],
        dtype=np.float64,
    )

    tangent = np.array(
        [
            np.interp(angle_deg, arc["angles"], arc["tangents"][:, 0]),
            np.interp(angle_deg, arc["angles"], arc["tangents"][:, 1]),
            np.interp(angle_deg, arc["angles"], arc["tangents"][:, 2]),
        ],
        dtype=np.float64,
    )

    tangent = normalize(tangent)

    if tangent is None:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    return point, tangent


def get_palm_local_normal(
    arcs,
):
    """
    Infer the REAL palm-plane normal in PALM_FRAME coordinates from the
    OPEN-HAND geometry.

    No local X/Y/Z axis is guessed.

    Geometry:
        lateral axis      = open index tip - open pinky tip
        forward/length axis = mean(open long-finger tips) - palm origin
        palm normal       = lateral x forward

    This directly describes the plane in which the open hand/fingers lie.
    It is much more appropriate than V4.7's finger-curl direction estimate.

    Only the plane matters for alignment, therefore the final sign is
    irrelevant because palm_back_alignment() uses abs(dot).
    """
    cached = arcs.get(
        "_palm_local_normal_v48",
        None,
    )

    if cached is not None:
        return np.asarray(
            cached,
            dtype=np.float64,
        )

    open_points = {}

    for finger in LONG_FINGERS:
        arc = arcs[
            finger
        ]

        open_angle = float(
            arc[
                "angles"
            ][0]
        )

        point, _ = interpolate_arc(
            arcs,
            finger,
            open_angle,
        )

        open_points[
            finger
        ] = np.asarray(
            point,
            dtype=np.float64,
        )

    lateral = normalize(
        open_points[
            "index"
        ]
        -
        open_points[
            "pinky"
        ]
    )

    forward = normalize(
        np.mean(
            np.asarray(
                [
                    open_points[
                        finger
                    ]
                    for finger
                    in LONG_FINGERS
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
    )

    normal = None

    if (
        lateral is not None
        and
        forward is not None
    ):
        normal = normalize(
            np.cross(
                lateral,
                forward,
            )
        )

    # Fallback: best-fit plane through the PALM_FRAME origin and the four
    # open fingertip positions.
    if normal is None:
        samples = np.asarray(
            [
                open_points[
                    finger
                ]
                for finger
                in LONG_FINGERS
            ],
            dtype=np.float64,
        )

        try:
            _, _, vh = np.linalg.svd(
                samples,
                full_matrices=False,
            )

            normal = normalize(
                vh[
                    -1
                ]
            )

        except Exception:
            normal = None

    if normal is None:
        raise RuntimeError(
            "PALM_PLANE_NORMAL_INFERENCE_FAILED"
        )

    arcs[
        "_palm_local_normal_v48"
    ] = np.asarray(
        normal,
        dtype=np.float64,
    )

    arcs[
        "_palm_open_points_v48"
    ] = {
        finger:
            open_points[
                finger
            ].copy()
        for finger
        in LONG_FINGERS
    }

    return np.asarray(
        normal,
        dtype=np.float64,
    )


def get_four_finger_root_local_points():
    """Read the four MCP/root positions in PALM_FRAME from the real URDF.

    These roots define the hand's stable transverse row direction.  Unlike
    fingertip minima, they are not distorted by different finger lengths.
    """
    global FOUR_FINGER_ROOT_LOCAL_POINTS, FOUR_FINGER_ROOT_LOCAL_CENTER
    if FOUR_FINGER_ROOT_LOCAL_POINTS is not None:
        return {
            finger: np.asarray(point, dtype=np.float64).copy()
            for finger, point in FOUR_FINGER_ROOT_LOCAL_POINTS.items()
        }

    wanted = {FINGER_JOINT_NAMES[finger]: finger for finger in LONG_FINGERS}
    root = ET.parse(LULA_URDF).getroot()
    points = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        finger = wanted.get(name)
        if finger is None:
            continue
        parent = joint.find("parent")
        origin = joint.find("origin")
        if parent is None or parent.attrib.get("link") != PALM_FRAME or origin is None:
            continue
        xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
        points[finger] = np.asarray(xyz, dtype=np.float64)
    if len(points) != len(LONG_FINGERS):
        raise RuntimeError("FOUR_FINGER_ROOT_POINTS_INFERENCE_FAILED")

    FOUR_FINGER_ROOT_LOCAL_POINTS = {
        finger: np.asarray(points[finger], dtype=np.float64).copy()
        for finger in LONG_FINGERS
    }
    FOUR_FINGER_ROOT_LOCAL_CENTER = np.mean(
        np.asarray([FOUR_FINGER_ROOT_LOCAL_POINTS[f] for f in LONG_FINGERS], dtype=np.float64),
        axis=0,
    )
    return {
        finger: np.asarray(point, dtype=np.float64).copy()
        for finger, point in FOUR_FINGER_ROOT_LOCAL_POINTS.items()
    }


def get_four_finger_root_local_center():
    """Mean MCP/root position in PALM_FRAME from the real URDF."""
    global FOUR_FINGER_ROOT_LOCAL_CENTER
    if FOUR_FINGER_ROOT_LOCAL_CENTER is None:
        get_four_finger_root_local_points()
    return np.asarray(FOUR_FINGER_ROOT_LOCAL_CENTER, dtype=np.float64)

def get_palm_local_forward(arcs):
    """Measured palm-root to four-finger direction in PALM_FRAME."""
    cached = arcs.get("_palm_local_forward_v100")
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)

    # get_palm_local_normal() also caches the measured open fingertip points.
    get_palm_local_normal(arcs)
    open_points = arcs.get("_palm_open_points_v48", {})
    root_center = get_four_finger_root_local_center()
    forward = normalize(
        np.mean(
            np.asarray([open_points[f] for f in LONG_FINGERS], dtype=np.float64),
            axis=0,
        )
        - root_center
    )
    if forward is None:
        raise RuntimeError("PALM_FORWARD_INFERENCE_FAILED")
    arcs["_palm_local_forward_v100"] = np.asarray(forward, dtype=np.float64)
    return np.asarray(forward, dtype=np.float64)


def _four_finger_reference_world_points(arcs, palm_position, palm_rotation):
    """Open/staging four-fingertip points used to define the fixed row geometry."""
    get_palm_local_normal(arcs)  # also caches measured open fingertip points
    open_points = arcs.get("_palm_open_points_v48", {})
    palm_position = np.asarray(palm_position, dtype=np.float64)
    palm_rotation = np.asarray(palm_rotation, dtype=np.float64)
    points = {}
    for finger in LONG_FINGERS:
        local = open_points.get(finger)
        if local is None:
            local = interpolate_arc(arcs, finger, 0.0)[0]
        points[finger] = palm_position + palm_rotation @ np.asarray(local, dtype=np.float64)
    return points


def centroid_base_axis_alignment(
    arcs,
    palm_position,
    palm_rotation,
    world_points_by_finger=None,
):
    """Top-view placement rule: MCP row is perpendicular to BASE->C.

    ``C`` is the centroid of the supplied four fingertip reference points.  For
    V22 staging these are the four independently scanned natural-minimum points
    (Cmin).  If no points are supplied, open/staging fingertips are used only
    as a compatibility fallback for older callers.

    Crucially, the row direction is NOT fitted through the four minima: fingers
    have different lengths, so those minima need not be collinear.  The stable
    transverse row is instead fitted from the four MCP/root locations in the
    real URDF, projected into the true J1 top-view plane.

    ``angle_deg`` is perpendicularity error: 0 deg is ideal.
    """
    if ROBOT_BASE_WORLD_POSITION is None or ROBOT_BASE_J1_AXIS_WORLD is None:
        return {
            "angle_deg": 0.0,
            "base_to_centroid": np.zeros(3, dtype=np.float64),
            "finger_row_axis": np.zeros(3, dtype=np.float64),
            "finger_forward": np.zeros(3, dtype=np.float64),
            "centroid_world": np.zeros(3, dtype=np.float64),
            "available": False,
        }

    base_p = np.asarray(ROBOT_BASE_WORLD_POSITION, dtype=np.float64)
    j1_axis = normalize(ROBOT_BASE_J1_AXIS_WORLD)
    if j1_axis is None:
        return {
            "angle_deg": 90.0,
            "base_to_centroid": np.zeros(3, dtype=np.float64),
            "finger_row_axis": np.zeros(3, dtype=np.float64),
            "finger_forward": np.zeros(3, dtype=np.float64),
            "centroid_world": np.zeros(3, dtype=np.float64),
            "available": False,
        }

    if world_points_by_finger is None:
        points_by_finger = _four_finger_reference_world_points(
            arcs, palm_position, palm_rotation
        )
    else:
        points_by_finger = {
            finger: np.asarray(point, dtype=np.float64)
            for finger, point in world_points_by_finger.items()
            if finger in LONG_FINGERS
        }
        if len(points_by_finger) < len(LONG_FINGERS):
            points_by_finger = _four_finger_reference_world_points(
                arcs, palm_position, palm_rotation
            )

    points = np.asarray([points_by_finger[f] for f in LONG_FINGERS], dtype=np.float64)
    centroid_world = np.mean(points, axis=0)

    base_to_centroid_raw = centroid_world - base_p
    base_to_centroid = normalize(
        base_to_centroid_raw
        - float(np.dot(base_to_centroid_raw, j1_axis)) * j1_axis
    )

    # Stable row direction from MCP/root locations, not from unequal-length
    # fingertip minima.
    palm_position = np.asarray(palm_position, dtype=np.float64)
    palm_rotation = np.asarray(palm_rotation, dtype=np.float64)
    local_roots = get_four_finger_root_local_points()
    root_points = np.asarray([
        palm_position + palm_rotation @ np.asarray(local_roots[f], dtype=np.float64)
        for f in LONG_FINGERS
    ], dtype=np.float64)
    root_centroid = np.mean(root_points, axis=0)
    centered_roots = root_points - root_centroid[None, :]
    centered_roots = centered_roots - np.outer(centered_roots @ j1_axis, j1_axis)
    try:
        _, singular, vh = np.linalg.svd(centered_roots, full_matrices=False)
        row_axis = normalize(vh[0]) if len(singular) else None
    except np.linalg.LinAlgError:
        row_axis = None

    if base_to_centroid is None or row_axis is None:
        return {
            "angle_deg": 90.0,
            "base_to_centroid": np.zeros(3, dtype=np.float64),
            "finger_row_axis": np.zeros(3, dtype=np.float64),
            "finger_forward": np.zeros(3, dtype=np.float64),
            "centroid_world": centroid_world,
            "mcp_centroid_world": root_centroid,
            "available": False,
        }

    parallel_component = clamp(abs(float(np.dot(base_to_centroid, row_axis))), 0.0, 1.0)
    perpendicular_error_deg = math.degrees(math.asin(parallel_component))

    radial_axis = normalize(np.cross(j1_axis, row_axis))
    if radial_axis is None:
        radial_axis = np.zeros(3, dtype=np.float64)
    elif float(np.dot(radial_axis, base_to_centroid)) < 0.0:
        radial_axis = -radial_axis

    return {
        "angle_deg": float(perpendicular_error_deg),
        "base_to_centroid": base_to_centroid,
        "finger_row_axis": row_axis,
        "finger_forward": radial_axis,
        "centroid_world": centroid_world,
        "mcp_centroid_world": root_centroid,
        "j1_axis": j1_axis,
        "available": True,
        "base_to_root": base_to_centroid,
        "root_world": centroid_world,
    }


def root_base_axis_alignment(arcs, palm_position, palm_rotation):
    """Compatibility wrapper for callers without natural-minimum points.

    V22 staging passes the four natural-minimum points explicitly to
    ``centroid_base_axis_alignment``.  Older endpoint/safety callers keep this
    wrapper and therefore use their historical open-tip centroid only for C;
    the row axis itself is still the stable MCP row.
    """
    return centroid_base_axis_alignment(arcs, palm_position, palm_rotation)

def finger_arc_height_order(
    arcs,
    palm_rotation,
    back_normal,
    phases,
    active_fingers,
    intermediate_commands=None,
):
    """Measure whether each selected arc moves from outside toward the back.

    Heights are projections on the outward back normal at one fixed palm pose.
    The selected final phase should be the lowest point of the used arc (within
    tolerance), and the straight point must not lie below an intermediate
    precontact point.
    """
    rotation = np.asarray(palm_rotation, dtype=np.float64)
    normal = normalize(back_normal)
    if normal is None:
        return {"valid": False, "max_violation_m": 999.0, "finger": {}}

    details = {}
    max_violation = 0.0
    for finger in active_fingers:
        final_phase = float(phases[finger])
        sample_angles = np.linspace(0.0, final_phase, ARC_HEIGHT_SAMPLES)
        heights = []
        for angle in sample_angles:
            local_point, _ = interpolate_arc(arcs, finger, float(angle))
            heights.append(float(np.dot(rotation @ local_point, normal)))
        heights = np.asarray(heights, dtype=np.float64)
        open_h = float(heights[0])
        final_h = float(heights[-1])
        min_index = int(np.argmin(heights))
        min_h = float(heights[min_index])
        final_above_min = max(0.0, final_h - min_h - ARC_HEIGHT_ORDER_TOL_M)
        full_angles = np.linspace(
            0.0,
            min(float(arcs[finger]["angles"][-1]), MAX_MAIN_PHASE_DEG),
            max(ARC_HEIGHT_SAMPLES, 61),
        )
        full_heights = []
        for angle in full_angles:
            local_point, _ = interpolate_arc(arcs, finger, float(angle))
            full_heights.append(float(np.dot(rotation @ local_point, normal)))
        full_heights = np.asarray(full_heights, dtype=np.float64)
        full_min_index = int(np.argmin(full_heights))
        full_min_h = float(full_heights[full_min_index])
        selected_above_full_min = max(
            0.0, final_h - full_min_h - ARC_HEIGHT_ORDER_TOL_M
        )

        # Positive increments mean a later phase moved outward rather than
        # toward the back.  Small local deviations within tolerance are noise.
        increments = np.diff(heights)
        reversal = max(
            0.0,
            float(np.max(increments)) - ARC_HEIGHT_ORDER_TOL_M,
        ) if len(increments) else 0.0
        open_below_final = max(
            0.0,
            final_h - open_h - ARC_HEIGHT_ORDER_TOL_M,
        )

        lead_h = None
        open_below_lead = 0.0
        lead_below_final = 0.0
        if intermediate_commands is not None:
            lead_phase = float(intermediate_commands.get(finger, 0.0))
            lead_point, _ = interpolate_arc(arcs, finger, lead_phase)
            lead_h = float(np.dot(rotation @ lead_point, normal))
            open_below_lead = max(
                0.0, lead_h - open_h - ARC_HEIGHT_ORDER_TOL_M
            )
            lead_below_final = max(
                0.0, final_h - lead_h - ARC_HEIGHT_ORDER_TOL_M
            )

        violation = max(
            final_above_min,
            selected_above_full_min,
            reversal,
            open_below_final,
            open_below_lead,
            lead_below_final,
        )
        max_violation = max(max_violation, violation)
        details[finger] = {
            "open_height_mm": open_h * 1000.0,
            "lead_height_mm": None if lead_h is None else lead_h * 1000.0,
            "final_height_mm": final_h * 1000.0,
            "minimum_height_mm": min_h * 1000.0,
            "minimum_phase_deg": float(sample_angles[min_index]),
            "full_minimum_height_mm": full_min_h * 1000.0,
            "full_minimum_phase_deg": float(full_angles[full_min_index]),
            "violation_mm": violation * 1000.0,
        }

    return {
        "valid": bool(max_violation <= 1e-12),
        "max_violation_m": float(max_violation),
        "finger": details,
    }


def repair_arc_height_order_phases(
    arcs,
    palm_rotation,
    back_normal,
    phases,
    active_fingers,
):
    """Move endpoints to the dynamically measured arc minimum.

    The old implementation could retreat by at most eight degrees from a fixed
    template phase.  That made a valid pose depend on whether a 50--54 degree
    template happened to lie near the minimum for the current palm attitude.
    Here every finger is sampled over its complete executable phase range using
    the current palm rotation and current local back normal.  The measured
    minimum therefore drives planning; it is not merely checked afterwards.

    The caller must still recompute depth, lane, patch, posture and collision
    constraints.  The existing two-millimetre height tolerance is unchanged.
    """
    repaired = {finger: float(value) for finger, value in phases.items()}
    adjustments = {}
    for finger in active_fingers:
        original_phase = float(repaired[finger])
        maximum_phase = min(
            float(arcs[finger]["angles"][-1]),
            float(MAX_MAIN_PHASE_DEG),
        )
        sample_angles = np.arange(
            float(MIN_MAIN_PHASE_DEG),
            maximum_phase + 0.5 * float(ARC_HEIGHT_PHASE_REPAIR_STEP_DEG),
            float(ARC_HEIGHT_PHASE_REPAIR_STEP_DEG),
            dtype=np.float64,
        )
        if len(sample_angles) == 0:
            continue

        rotation = np.asarray(palm_rotation, dtype=np.float64)
        normal = normalize(back_normal)
        if normal is None:
            continue
        heights = []
        for candidate_phase in sample_angles:
            local_point, _ = interpolate_arc(
                arcs,
                finger,
                float(candidate_phase),
            )
            heights.append(float(np.dot(rotation @ local_point, normal)))
        minimum_index = int(np.argmin(np.asarray(heights, dtype=np.float64)))
        measured_phase = float(sample_angles[minimum_index])

        repaired[finger] = measured_phase
        adjustments[finger] = {
            "from_deg": original_phase,
            "to_deg": measured_phase,
            "retreat_deg": original_phase - measured_phase,
            "dynamic_minimum_deg": measured_phase,
        }

    final = finger_arc_height_order(
        arcs,
        palm_rotation,
        back_normal,
        repaired,
        active_fingers,
    )
    return repaired, adjustments, final


def palm_back_alignment(
    arcs,
    palm_rotation,
    back_normal,
):
    """
    Return palm/back plane alignment.

    angle_deg = 0 means:
        palm plane parallel to local back tangent plane.

    This is independent of the numerical J5 angle.
    """
    local_normal = get_palm_local_normal(
        arcs
    )

    palm_normal = normalize(
        np.asarray(
            palm_rotation,
            dtype=np.float64,
        )
        @
        local_normal
    )

    back_normal = normalize(
        back_normal
    )

    if (
        palm_normal is None
        or
        back_normal is None
    ):
        return {
            "angle_deg":
                90.0,

            "parallel_score":
                0.0,

            "palm_normal":
                np.zeros(
                    3,
                    dtype=np.float64,
                ),

            "back_normal":
                np.zeros(
                    3,
                    dtype=np.float64,
                ),
        }

    cosine = clamp(
        abs(
            float(
                np.dot(
                    palm_normal,
                    back_normal,
                )
            )
        ),
        0.0,
        1.0,
    )

    angle_deg = math.degrees(
        math.acos(
            cosine
        )
    )

    return {
        "angle_deg":
            float(
                angle_deg
            ),

        "parallel_score":
            float(
                cosine
            ),

        "palm_normal":
            palm_normal,

        "back_normal":
            back_normal,
    }


def mean_contact_normal(
    normals,
    fallback,
):
    if normals:
        normal = normalize(
            np.mean(
                np.asarray(
                    normals,
                    dtype=np.float64,
                ),
                axis=0,
            )
        )

        if normal is not None:
            return normal

    fallback = normalize(
        fallback
    )

    if fallback is None:
        fallback = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    return fallback


# ============================================================
# PATCH ANALYSIS
# ============================================================

def tangent_basis(normal):
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if abs(float(np.dot(helper, normal))) > 0.90:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = normalize(np.cross(normal, helper))
    v = normalize(np.cross(normal, u))

    return u, v


def analyse_patch(
    patch_points,
    patch_normals,
    target,
    global_tree,
    global_points,
    global_normals,
):
    mean_normal = normalize(np.mean(patch_normals, axis=0))

    deviations = [
        math.degrees(
            math.acos(
                clamp(float(np.dot(n, mean_normal)), -1.0, 1.0)
            )
        )
        for n in patch_normals
    ]

    spread90 = float(np.percentile(deviations, 90))

    center_a = mean_normal.copy()
    center_b = patch_normals[int(np.argmin(patch_normals @ center_a))].copy()
    labels = np.zeros(len(patch_normals), dtype=np.int32)

    for _ in range(18):
        new_labels = ((patch_normals @ center_b) > (patch_normals @ center_a)).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        a = patch_normals[labels == 0]
        b = patch_normals[labels == 1]

        if len(a) == 0 or len(b) == 0:
            break

        center_a = normalize(np.mean(a, axis=0))
        center_b = normalize(np.mean(b, axis=0))

    separation = math.degrees(
        math.acos(
            clamp(float(np.dot(center_a, center_b)), -1.0, 1.0)
        )
    )

    ratio_a = float(np.mean(labels == 0))
    ratio_b = float(np.mean(labels == 1))

    is_v = (
        spread90 >= 10.0
        and separation >= 18.0
        and ratio_a >= 0.18
        and ratio_b >= 0.18
    )

    modes = []

    def add_mode(name, kind, points, normals, normal, anchor):
        u, v = tangent_basis(normal)

        modes.append(
            {
                "name": name,
                "kind": kind,
                "anchor": np.asarray(anchor, dtype=np.float64),
                "normal": normalize(normal),
                "points": np.asarray(points, dtype=np.float64),
                "normals": np.asarray(normals, dtype=np.float64),
                "tree": cKDTree(points),
                "u": u,
                "v": v,
                "global_tree": global_tree,
                "global_points": global_points,
                "global_normals": global_normals,
            }
        )

    # This is the local patch around the GUI-clicked point, not an old center pose.
    add_mode("TARGET_PATCH", "flat", patch_points, patch_normals, mean_normal, target)

    if is_v:
        for label, name, normal in [
            (0, "V_SIDE_A", center_a),
            (1, "V_SIDE_B", center_b),
        ]:
            ids = np.where(labels == label)[0]

            if len(ids) == 0:
                continue

            side_points = patch_points[ids]
            side_normals = patch_normals[ids]

            d = np.linalg.norm(side_points - target[None, :], axis=1)
            anchor = side_points[int(np.argmin(d))]

            add_mode(name, "v_side", side_points, side_normals, normal, anchor)

    return modes, spread90, separation, is_v


def planning_signed_distance(
    euclidean_distance,
    raw_signed,
):
    """
    Signed distance for point-cloud PLANNING only.

    Far from the surface, a nearest-point normal is not a valid global
    inside/outside classifier. In that region we report positive clearance
    instead of a fake large negative penetration.

    Final safety does NOT rely on this approximation; it uses the USD triangle
    mesh through BackTriangleSurfaceQuery.
    """
    euclidean_distance = float(
        euclidean_distance
    )

    raw_signed = float(
        raw_signed
    )

    if (
        euclidean_distance
        >
        POINTCLOUD_SIGNED_RELIABLE_M
    ):
        return euclidean_distance

    return raw_signed


def nearest_local(mode, point):
    distance, index = mode["tree"].query(
        point,
        k=1,
    )

    index = int(
        index
    )

    surface = mode["points"][
        index
    ]

    normal = mode["normals"][
        index
    ]

    raw_signed = float(
        np.dot(
            np.asarray(
                point,
                dtype=np.float64,
            )
            -
            surface,
            normal,
        )
    )

    signed = planning_signed_distance(
        distance,
        raw_signed,
    )

    return (
        float(
            distance
        ),
        surface,
        normal,
        signed,
    )


def nearest_global(mode, point):
    distance, index = mode[
        "global_tree"
    ].query(
        point,
        k=1,
    )

    index = int(
        index
    )

    surface = mode[
        "global_points"
    ][
        index
    ]

    normal = mode[
        "global_normals"
    ][
        index
    ]

    raw_signed = float(
        np.dot(
            np.asarray(
                point,
                dtype=np.float64,
            )
            -
            surface,
            normal,
        )
    )

    signed = planning_signed_distance(
        distance,
        raw_signed,
    )

    return (
        float(
            distance
        ),
        surface,
        normal,
        signed,
    )


def closest_point_on_triangle(
    point,
    a,
    b,
    c,
):
    """
    Closest point on triangle using the region tests from
    Real-Time Collision Detection (Ericson).

    Pure NumPy/Python so no extra Isaac environment dependency is required.
    """
    p = np.asarray(
        point,
        dtype=np.float64,
    )

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    c = np.asarray(
        c,
        dtype=np.float64,
    )

    ab = (
        b
        -
        a
    )

    ac = (
        c
        -
        a
    )

    ap = (
        p
        -
        a
    )

    d1 = float(
        np.dot(
            ab,
            ap,
        )
    )

    d2 = float(
        np.dot(
            ac,
            ap,
        )
    )

    if (
        d1 <= 0.0
        and
        d2 <= 0.0
    ):
        return a.copy()

    bp = (
        p
        -
        b
    )

    d3 = float(
        np.dot(
            ab,
            bp,
        )
    )

    d4 = float(
        np.dot(
            ac,
            bp,
        )
    )

    if (
        d3 >= 0.0
        and
        d4 <= d3
    ):
        return b.copy()

    vc = (
        d1
        *
        d4
        -
        d3
        *
        d2
    )

    if (
        vc <= 0.0
        and
        d1 >= 0.0
        and
        d3 <= 0.0
    ):
        v = (
            d1
            /
            max(
                d1
                -
                d3,
                1e-12,
            )
        )

        return (
            a
            +
            v
            *
            ab
        )

    cp = (
        p
        -
        c
    )

    d5 = float(
        np.dot(
            ab,
            cp,
        )
    )

    d6 = float(
        np.dot(
            ac,
            cp,
        )
    )

    if (
        d6 >= 0.0
        and
        d5 <= d6
    ):
        return c.copy()

    vb = (
        d5
        *
        d2
        -
        d1
        *
        d6
    )

    if (
        vb <= 0.0
        and
        d2 >= 0.0
        and
        d6 <= 0.0
    ):
        w = (
            d2
            /
            max(
                d2
                -
                d6,
                1e-12,
            )
        )

        return (
            a
            +
            w
            *
            ac
        )

    va = (
        d3
        *
        d6
        -
        d5
        *
        d4
    )

    if (
        va <= 0.0
        and
        (
            d4
            -
            d3
        )
        >= 0.0
        and
        (
            d5
            -
            d6
        )
        >= 0.0
    ):
        bc = (
            c
            -
            b
        )

        w = (
            (
                d4
                -
                d3
            )
            /
            max(
                (
                    d4
                    -
                    d3
                )
                +
                (
                    d5
                    -
                    d6
                ),
                1e-12,
            )
        )

        return (
            b
            +
            w
            *
            bc
        )

    denom = max(
        va
        +
        vb
        +
        vc,
        1e-12,
    )

    v = (
        vb
        /
        denom
    )

    w = (
        vc
        /
        denom
    )

    return (
        a
        +
        ab
        *
        v
        +
        ac
        *
        w
    )



def segment_triangle_intersection(
    p0,
    p1,
    triangle,
    eps=1.0e-9,
):
    """
    Two-sided finite-segment / triangle intersection using Moller-Trumbore.

    This routine is deliberately sign-free: BackSurface is an OPEN sheet, so
    triangle winding does not define inside/outside. A collision is reported
    only when the moving surface sample actually intersects the triangle.

    Returns None for no hit, otherwise a dict containing segment parameter t,
    world intersection point and triangle barycentric coordinates.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)

    direction = p1 - p0
    if float(np.dot(direction, direction)) <= float(eps) ** 2:
        return None

    a, b, c = tri
    e1 = b - a
    e2 = c - a
    h = np.cross(direction, e2)
    det = float(np.dot(e1, h))

    if abs(det) <= float(eps):
        # Parallel/coplanar motion is handled by the static surface-contact
        # epsilon at sampled poses; do not invent a signed-side collision.
        return None

    inv_det = 1.0 / det
    s = p0 - a
    u = float(inv_det * np.dot(s, h))
    if u < -float(eps) or u > 1.0 + float(eps):
        return None

    q = np.cross(s, e1)
    v = float(inv_det * np.dot(direction, q))
    if v < -float(eps) or (u + v) > 1.0 + float(eps):
        return None

    t = float(inv_det * np.dot(e2, q))
    if t < -float(eps) or t > 1.0 + float(eps):
        return None

    t_clamped = clamp(t, 0.0, 1.0)
    point = p0 + t_clamped * direction
    barycentric = np.array([1.0 - u - v, u, v], dtype=np.float64)

    return {
        "t": float(t_clamped),
        "point": point,
        "barycentric": barycentric,
    }


class BackTriangleSurfaceQuery:
    """
    Exact-ish nearest triangle surface relation for BackSurface.

    A centroid KD-tree selects a small local triangle set, then exact
    point-to-triangle closest points are evaluated inside that set.

    Triangle normals are aligned to the existing dense-back normals so that
    signed depth follows the same orientation convention used by the planner.
    """

    def __init__(
        self,
        stage,
        reference_points,
        reference_normals,
        reference_tree,
        root_path=BACK_SURFACE_ROOT,
    ):
        self.available = False
        self.triangles = None
        self.normals = None
        self.centroids = None
        self.tree = None
        self.triangle_aabb_min = None
        self.triangle_aabb_max = None
        self.triangle_radii = None
        self.max_triangle_radius = 0.0

        xform_cache = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        )

        triangle_list = []

        for prim in stage.Traverse():
            path = str(
                prim.GetPath()
            )

            if not (
                path == root_path
                or
                path.startswith(
                    root_path
                    +
                    "/"
                )
            ):
                continue

            if not prim.IsA(
                UsdGeom.Mesh
            ):
                continue

            mesh = UsdGeom.Mesh(
                prim
            )

            local_points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()

            if (
                local_points is None
                or
                counts is None
                or
                indices is None
            ):
                continue

            world_matrix = xform_cache.GetLocalToWorldTransform(
                prim
            )

            world_points = np.asarray(
                [
                    [
                        float(
                            transformed[0]
                        ),
                        float(
                            transformed[1]
                        ),
                        float(
                            transformed[2]
                        ),
                    ]
                    for transformed
                    in [
                        world_matrix.Transform(
                            Gf.Vec3d(
                                float(p[0]),
                                float(p[1]),
                                float(p[2]),
                            )
                        )
                        for p
                        in local_points
                    ]
                ],
                dtype=np.float64,
            )

            cursor = 0

            for count in counts:
                count = int(
                    count
                )

                face = [
                    int(
                        indices[
                            cursor
                            +
                            j
                        ]
                    )
                    for j
                    in range(
                        count
                    )
                ]

                cursor += (
                    count
                )

                if count < 3:
                    continue

                # Fan triangulation for general polygon faces.
                i0 = face[
                    0
                ]

                for j in range(
                    1,
                    count
                    -
                    1,
                ):
                    triangle_list.append(
                        [
                            world_points[
                                i0
                            ],
                            world_points[
                                face[
                                    j
                                ]
                            ],
                            world_points[
                                face[
                                    j
                                    +
                                    1
                                ]
                            ],
                        ]
                    )

        if not triangle_list:
            print(
                "[V4.12-ARC] WARNING: No BackSurface USD mesh triangles found. "
                "Final geometric safety will use point-cloud fallback."
            )

            return

        triangles = np.asarray(
            triangle_list,
            dtype=np.float64,
        )

        e1 = (
            triangles[
                :,
                1,
            ]
            -
            triangles[
                :,
                0,
            ]
        )

        e2 = (
            triangles[
                :,
                2,
            ]
            -
            triangles[
                :,
                0,
            ]
        )

        normals = np.cross(
            e1,
            e2,
        )

        lengths = np.linalg.norm(
            normals,
            axis=1,
        )

        valid = (
            lengths
            >
            1e-12
        )

        triangles = triangles[
            valid
        ]

        normals = normals[
            valid
        ]

        lengths = lengths[
            valid
        ]

        normals = (
            normals
            /
            lengths[
                :,
                None
            ]
        )

        centroids = np.mean(
            triangles,
            axis=1,
        )

        triangle_aabb_min = np.min(triangles, axis=1)
        triangle_aabb_max = np.max(triangles, axis=1)
        triangle_radii = np.max(
            np.linalg.norm(triangles - centroids[:, None, :], axis=2),
            axis=1,
        )

        # Align mesh winding to the already validated dense-back normal field.
        _, reference_ids = reference_tree.query(
            centroids,
            k=1,
        )

        reference_ids = np.asarray(
            reference_ids,
            dtype=np.int64,
        )

        reference_n = reference_normals[
            reference_ids
        ]

        dots = np.sum(
            normals
            *
            reference_n,
            axis=1,
        )

        normals[
            dots
            <
            0.0
        ] *= -1.0

        self.triangles = (
            triangles
        )

        self.normals = (
            normals
        )

        self.centroids = (
            centroids
        )

        self.triangle_aabb_min = triangle_aabb_min
        self.triangle_aabb_max = triangle_aabb_max
        self.triangle_radii = triangle_radii
        self.max_triangle_radius = float(np.max(triangle_radii)) if len(triangle_radii) else 0.0

        self.tree = cKDTree(
            centroids
        )

        self.available = True

        print(
            "[V4.12-ARC] BackSurface triangle query ready:",
            len(
                triangles
            ),
            "triangles",
        )


    def segment_intersection(
        self,
        p0,
        p1,
        eps=HAND_SWEEP_INTERSECTION_EPS_M,
        aabb_pad=HAND_SWEEP_AABB_PAD_M,
    ):
        """
        Return the earliest TRUE finite-segment intersection with BackSurface.

        Broad phase:
          centroid KD-tree radius + exact segment/triangle AABB overlap.
        Narrow phase:
          two-sided Moller-Trumbore finite segment/triangle intersection.

        No triangle-normal sign is used anywhere in this collision decision.
        """
        if not self.available:
            return None

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length <= float(eps):
            return None

        midpoint = 0.5 * (p0 + p1)
        search_radius = (
            0.5 * length
            + float(self.max_triangle_radius)
            + float(aabb_pad)
        )

        ids = self.tree.query_ball_point(midpoint, r=search_radius)
        if not ids:
            return None

        ids = np.asarray(ids, dtype=np.int64)
        seg_min = np.minimum(p0, p1) - float(aabb_pad)
        seg_max = np.maximum(p0, p1) + float(aabb_pad)
        overlap = (
            np.all(self.triangle_aabb_max[ids] >= seg_min[None, :], axis=1)
            & np.all(self.triangle_aabb_min[ids] <= seg_max[None, :], axis=1)
        )
        ids = ids[overlap]
        if len(ids) == 0:
            return None

        best = None
        for triangle_id in ids:
            triangle_id = int(triangle_id)
            hit = segment_triangle_intersection(
                p0,
                p1,
                self.triangles[triangle_id],
                eps=eps,
            )
            if hit is None:
                continue
            if best is None or float(hit["t"]) < float(best["t"]):
                best = dict(hit)
                best["triangle_id"] = triangle_id

        return best


    def query(
        self,
        point,
    ):
        if not self.available:
            raise RuntimeError(
                "BackTriangleSurfaceQuery unavailable."
            )

        point = np.asarray(
            point,
            dtype=np.float64,
        )

        k = min(
            MESH_QUERY_NEAREST_TRIANGLES,
            len(
                self.triangles
            ),
        )

        _, ids = self.tree.query(
            point,
            k=k,
        )

        ids = np.atleast_1d(
            ids
        ).astype(
            np.int64
        )

        best_distance_sq = float(
            "inf"
        )

        best_point = None
        best_normal = None
        best_id = None

        for triangle_id in ids:
            triangle = self.triangles[
                int(
                    triangle_id
                )
            ]

            closest = closest_point_on_triangle(
                point,
                triangle[
                    0
                ],
                triangle[
                    1
                ],
                triangle[
                    2
                ],
            )

            delta = (
                point
                -
                closest
            )

            distance_sq = float(
                np.dot(
                    delta,
                    delta,
                )
            )

            if (
                distance_sq
                <
                best_distance_sq
            ):
                best_distance_sq = (
                    distance_sq
                )

                best_point = (
                    closest
                )

                best_normal = self.normals[
                    int(
                        triangle_id
                    )
                ]

                best_id = int(
                    triangle_id
                )

        distance = math.sqrt(
            max(
                best_distance_sq,
                0.0,
            )
        )

        signed = float(
            np.dot(
                point
                -
                best_point,
                best_normal,
            )
        )

        return (
            distance,
            best_point,
            best_normal,
            signed,
            best_id,
        )


def diagnose_back_surface_alignment(back_mesh_query, back_points):
    """Measure whether the CSV point cloud and current USD mesh share world coordinates."""
    if back_mesh_query is None or not bool(getattr(back_mesh_query, "available", False)):
        print("[V100 BACK ALIGNMENT] mesh unavailable; point-cloud/mesh frame check skipped.")
        return None

    points = np.asarray(back_points, dtype=np.float64)
    sample_count = min(int(BACK_ALIGNMENT_SAMPLE_COUNT), len(points))
    if sample_count <= 0:
        return None

    sample_ids = np.unique(
        np.linspace(0, len(points) - 1, sample_count, dtype=np.int64)
    )
    distances = np.asarray(
        [float(back_mesh_query.query(points[int(i)])[0]) for i in sample_ids],
        dtype=np.float64,
    )
    summary = {
        "samples": int(len(distances)),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
    }
    print(
        "[V100 BACK ALIGNMENT]",
        "samples=", summary["samples"],
        "median_mm=", round(summary["median_m"] * 1000.0, 3),
        "p95_mm=", round(summary["p95_m"] * 1000.0, 3),
        "max_mm=", round(summary["max_m"] * 1000.0, 3),
    )
    if summary["p95_m"] > BACK_ALIGNMENT_WARN_P95_M:
        print(
            "[V100 BACK ALIGNMENT WARNING] CSV point cloud and current BackSurface may not share the same geometry/frame.",
            "p95_limit_mm=", round(BACK_ALIGNMENT_WARN_P95_M * 1000.0, 1),
        )
    return summary


def open_surface_safety_signed(
    distance,
    raw_signed,
    reliable_distance=MESH_SIGNED_RELIABLE_M,
):
    """
    Safety signed value for an OPEN triangle surface.

    BackSurface is a sheet, not a watertight closed body. Therefore the sign of
    a triangle normal is only meaningful close to the actual sheet.

    Far from the sheet:
        distance > reliable_distance
        -> treat as positive clearance.

    Near the sheet:
        use the raw oriented triangle signed value.

    A true crossing is still detected because any continuous crossing must pass
    through the near-surface band first.
    """
    distance = float(
        distance
    )

    raw_signed = float(
        raw_signed
    )

    if distance > float(
        reliable_distance
    ):
        return distance

    return raw_signed


def final_surface_relation(
    point,
    back_mesh_query,
    back_points,
    back_normals,
    back_tree,
):
    """
    Final surface relation for the open BackSurface sheet.

    Returned fields:
        distance    : Euclidean distance to the actual surface
        raw_signed  : oriented local triangle/normal signed value
        signed      : safety signed value with open-sheet distance gating

    Hard penetration checks must use `signed`.
    Contact-quality checks may use `raw_signed`, but only when distance is
    already inside the near-surface contact band.
    """
    policy = globals().get('PALM_POLICY')
    if policy is not None:
        policy.check_candidate_budget()
    point = np.asarray(
        point,
        dtype=np.float64,
    )

    if (
        back_mesh_query is not None
        and
        getattr(
            back_mesh_query,
            "available",
            False,
        )
    ):
        (
            distance,
            surface,
            normal,
            raw_signed,
            triangle_id,
        ) = back_mesh_query.query(
            point
        )

        signed = open_surface_safety_signed(
            distance,
            raw_signed,
            MESH_SIGNED_RELIABLE_M,
        )

        return {
            "distance":
                float(
                    distance
                ),

            "surface":
                surface,

            "normal":
                normal,

            "raw_signed":
                float(
                    raw_signed
                ),

            "signed":
                float(
                    signed
                ),

            "source":
                "usd_triangle_mesh_open_surface",

            "id":
                triangle_id,
        }

    distance, index = back_tree.query(
        point,
        k=1,
    )

    index = int(
        index
    )

    surface = back_points[
        index
    ]

    normal = back_normals[
        index
    ]

    raw_signed = float(
        np.dot(
            point
            -
            surface,
            normal,
        )
    )

    signed = open_surface_safety_signed(
        distance,
        raw_signed,
        POINTCLOUD_SIGNED_RELIABLE_M,
    )

    return {
        "distance":
            float(
                distance
            ),

        "surface":
            surface,

        "normal":
            normal,

        "raw_signed":
            float(
                raw_signed
            ),

        "signed":
            float(
                signed
            ),

        "source":
            "pointcloud_open_surface_fallback",

        "id":
            index,
    }


# ============================================================
# COVERAGE / CONTACT
# ============================================================

def segment_origin_distance(a, b):
    d = b - a
    denom = float(np.dot(d, d))

    if denom < 1e-12:
        return float(np.linalg.norm(a))

    t = clamp(float(-np.dot(a, d) / denom), 0.0, 1.0)
    return float(np.linalg.norm(a + t * d))


def coverage_error(mode, world_points):
    uv = []

    for point in world_points:
        delta = point - mode["anchor"]
        uv.append([float(np.dot(delta, mode["u"])), float(np.dot(delta, mode["v"]))])

    uv = np.asarray(uv, dtype=np.float64)

    if len(uv) == 1:
        return float(np.linalg.norm(uv[0]))

    if len(uv) == 2:
        return segment_origin_distance(uv[0], uv[1])

    try:
        hull = ConvexHull(uv)

        # For origin x=[0,0], the hull equation test is just offset <= 0.
        if bool(np.all(hull.equations[:, -1] <= 1e-9)):
            return 0.0

        polygon = uv[hull.vertices]

        return min(
            segment_origin_distance(polygon[i], polygon[(i + 1) % len(polygon)])
            for i in range(len(polygon))
        )

    except QhullError:
        return min(
            segment_origin_distance(a, b)
            for a, b in itertools.combinations(uv, 2)
        )


def contact_segment(arcs, finger, main_phase, palm_position, palm_rotation, mode):
    arc = arcs[finger]
    mask = arc["angles"] <= float(main_phase) + 1e-9

    angles = arc["angles"][mask]
    local_points = arc["points"][mask]
    local_tangents = arc["tangents"][mask]

    world_points = palm_position[None, :] + (palm_rotation @ local_points.T).T
    world_tangents = (palm_rotation @ local_tangents.T).T

    _, indices = mode["tree"].query(world_points, k=1)
    indices = np.asarray(indices, dtype=np.int64)

    surfaces = mode["points"][indices]
    normals = mode["normals"][indices]

    signed = np.einsum("ij,ij->i", world_points - surfaces, normals)

    tangent_scores = 1.0 - np.abs(
        np.einsum("ij,ij->i", world_tangents, normals)
    )

    valid = (
        (angles >= MIN_CONTACT_PHASE_DEG)
        & (signed >= GOOD_SIGNED_MIN_M)
        & (signed <= +0.0025)
        & (tangent_scores >= MIN_TANGENT_SCORE)
    )

    minimum_signed = float(np.min(signed)) if len(signed) else 999.0

    if len(valid) == 0 or not bool(valid[-1]):
        return {
            "start": None,
            "span": 0.0,
            "stroke": 0.0,
            "minimum_signed": minimum_signed,
        }

    start = len(valid) - 1

    while start > 0 and bool(valid[start - 1]):
        start -= 1

    stroke = 0.0

    for i in range(start + 1, len(world_points)):
        delta = world_points[i] - world_points[i - 1]
        tangent_delta = delta - float(np.dot(delta, mode["normal"])) * mode["normal"]
        stroke += float(np.linalg.norm(tangent_delta))

    return {
        "start": float(angles[start]),
        "span": float(angles[-1] - angles[start]),
        "stroke": stroke,
        "minimum_signed": minimum_signed,
    }


def estimate_available_arc_stroke(
    arcs,
    finger,
    phase_deg,
    window_deg=24.0,
):
    """
    Local finger-arc stroke potential near the selected phase.

    The arm/wrist is allowed to move while the finger flexes. Therefore
    this is only a local geometric stroke-potential measurement. Actual
    full-path skin penetration is checked later by validate_trajectory().
    """
    arc = arcs[finger]

    phase_deg = clamp(
        float(phase_deg),
        float(arc["angles"][0]),
        float(arc["angles"][-1]),
    )

    start_deg = max(
        float(arc["angles"][0]),
        phase_deg - float(window_deg),
    )

    sample_angles = np.linspace(
        start_deg,
        phase_deg,
        13,
    )

    local_points = np.asarray(
        [
            interpolate_arc(
                arcs,
                finger,
                angle,
            )[0]
            for angle
            in sample_angles
        ],
        dtype=np.float64,
    )

    if len(local_points) < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    local_points,
                    axis=0,
                ),
                axis=1,
            )
        )
    )


def contact_depth_balance_from_results(
    finger_results,
    active_fingers,
):
    """
    Multi-finger contact leveling metrics.

    mean_signed:
        common-mode contact depth (all fingers too deep / too shallow).

    std_signed / range_signed:
        differential contact error across fingers. These are exactly the
        quantities J4/J5 should reduce when one side is deeper than the other.
    """
    values = []

    for finger in active_fingers:
        result = finger_results.get(
            finger,
            {},
        )

        if "signed" not in result:
            continue

        values.append(
            float(
                result[
                    "signed"
                ]
            )
        )

    if not values:
        return {
            "count":
                0,

            "mean_signed":
                999.0,

            "mean_error":
                999.0,

            "std_signed":
                999.0,

            "range_signed":
                999.0,
        }

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean_signed = float(
        np.mean(
            values
        )
    )

    return {
        "count":
            int(
                len(
                    values
                )
            ),

        "mean_signed":
            mean_signed,

        "mean_error":
            float(
                mean_signed
                -
                IDEAL_SIGNED_M
            ),

        "std_signed":
            float(
                np.std(
                    values
                )
            ),

        "range_signed":
            float(
                np.max(
                    values
                )
                -
                np.min(
                    values
                )
            ),
    }


def clone_template_with_phases(
    template,
    phases,
    suffix="_BAL",
):
    result = dict(
        template
    )

    result[
        "phases"
    ] = {
        finger:
            float(
                phases[
                    finger
                ]
            )
        for finger
        in template[
            "active"
        ]
    }

    result[
        "name"
    ] = (
        str(
            template[
                "name"
            ]
        )
        +
        str(
            suffix
        )
    )

    result[
        "balanced_from"
    ] = template[
        "name"
    ]

    return result



def _project_tangent(vector, normal):
    vector = np.asarray(vector, dtype=np.float64)
    normal = normalize(normal)
    if normal is None:
        return vector.copy()
    return vector - float(np.dot(vector, normal)) * normal


def build_target_lane_spec(
    arcs,
    mode,
    palm_rotation,
    phases,
    active_fingers,
):
    """
    Build parallel scratch lanes centered on the GUI-selected TARGET_PATCH.

    The GUI target is a region center. The natural lateral spacing of the
    selected fingers is preserved from the measured finger-arc geometry, while
    the whole band is translated so that its center lies on mode["anchor"].
    Along-lane scratch motion is allowed only inside the finite GUI patch. The
    four-finger centroid remains target-locked and every fingertip must remain
    inside the per-finger patch-radius boundary.
    """
    active = list(active_fingers)
    if not active:
        return None

    local_points = {}
    for finger in active:
        local_points[finger] = np.asarray(
            interpolate_arc(arcs, finger, float(phases[finger]))[0],
            dtype=np.float64,
        )

    centroid = np.mean(np.asarray(list(local_points.values())), axis=0)

    # Natural local band direction across the fingers.
    if len(active) >= 2:
        centered = np.asarray([local_points[f] - centroid for f in active])
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        local_band = normalize(vh[0])
    else:
        # With one finger, use the palm-open finger row geometry to define the
        # lateral direction instead of inventing a world axis.
        open_points = np.asarray(
            [interpolate_arc(arcs, f, 0.0)[0] for f in LONG_FINGERS],
            dtype=np.float64,
        )
        open_centered = open_points - np.mean(open_points, axis=0)[None, :]
        _, _, vh = np.linalg.svd(open_centered, full_matrices=False)
        local_band = normalize(vh[0])

    if local_band is None:
        local_band = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Scratch direction comes from the measured local arc tangents.
    local_scratch = normalize(
        np.mean(
            np.asarray(
                [interpolate_arc(arcs, f, float(phases[f]))[1] for f in active],
                dtype=np.float64,
            ),
            axis=0,
        )
    )
    if local_scratch is None:
        local_scratch = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    local_band = _project_tangent(local_band, local_scratch)
    local_band = normalize(local_band)
    if local_band is None:
        local_band = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    normal = normalize(mode["normal"])
    world_band = normalize(_project_tangent(palm_rotation @ local_band, normal))
    world_scratch_guess = normalize(_project_tangent(palm_rotation @ local_scratch, normal))

    if world_band is None:
        world_band = normalize(mode["u"])
    if world_scratch_guess is None:
        world_scratch_guess = normalize(mode["v"])

    # Keep the pre-orthogonalization band direction so that a later handedness
    # correction can also be applied to the numerical lane offsets. Previously
    # the axis could flip while the offsets kept their old signs, creating a
    # false ~60 mm cross-lane error for a correctly spaced four-finger row.
    world_band_guess = np.asarray(world_band, dtype=np.float64).copy()

    # Orthonormal tangent frame; preserve the measured scratch-axis sign.
    world_scratch = normalize(np.cross(world_band, normal))
    if world_scratch is None:
        world_scratch = normalize(mode["v"])
    if float(np.dot(world_scratch, world_scratch_guess)) < 0.0:
        world_scratch = -world_scratch
    world_band = normalize(np.cross(normal, world_scratch))

    band_offset_sign = (
        -1.0
        if float(np.dot(world_band, world_band_guess)) < 0.0
        else 1.0
    )

    band_offsets = {}
    for finger in active:
        band_offsets[finger] = band_offset_sign * float(
            np.dot(local_points[finger] - centroid, local_band)
        )

    # Recenter offsets numerically so the mean lane is exactly the GUI target.
    mean_offset = float(np.mean(list(band_offsets.values()))) if band_offsets else 0.0
    for finger in band_offsets:
        band_offsets[finger] -= mean_offset

    # An anchored search targets one fingertip, not the average finger row.
    # Translate the entire lane band; preserve every pairwise finger spacing.
    anchor_finger = mode.get('v103_anchor_finger')
    if anchor_finger is not None:
        if anchor_finger not in band_offsets:
            raise ValueError('LANE_ANCHOR_NOT_ACTIVE:' + str(anchor_finger))
        anchor_offset = band_offsets[anchor_finger]
        for finger in band_offsets:
            band_offsets[finger] -= anchor_offset

    return {
        "anchor": np.asarray(mode["anchor"], dtype=np.float64),
        "normal": normal,
        "scratch_axis": world_scratch,
        "band_axis": world_band,
        "band_offsets": band_offsets,
        "band_offset_sign": band_offset_sign,
        "active": tuple(active),
        "anchor_finger": anchor_finger,
    }


def target_lane_metrics(world_points_by_finger, lane_spec):
    if lane_spec is None or not world_points_by_finger:
        return {
            "center_error": 999.0,
            "max_cross_error": 999.0,
            "mean_cross_error": 999.0,
            "max_patch_radius": 999.0,
            "finger": {},
        }

    anchor = lane_spec["anchor"]
    normal = lane_spec["normal"]
    band = lane_spec["band_axis"]
    scratch = lane_spec["scratch_axis"]

    tangential_points = []
    details = {}
    cross_errors = []
    patch_radii = []

    for finger, point in world_points_by_finger.items():
        point = np.asarray(point, dtype=np.float64)
        delta = point - anchor
        tangent_delta = _project_tangent(delta, normal)
        tangential_points.append(tangent_delta)

        band_coordinate = float(np.dot(tangent_delta, band))
        along_coordinate = float(np.dot(tangent_delta, scratch))
        desired_band = float(lane_spec["band_offsets"].get(finger, 0.0))
        cross_error = abs(band_coordinate - desired_band)
        patch_radius = float(np.linalg.norm(tangent_delta))

        cross_errors.append(cross_error)
        patch_radii.append(patch_radius)
        details[finger] = {
            "cross_error": cross_error,
            "band_coordinate": band_coordinate,
            "desired_band": desired_band,
            "along_coordinate": along_coordinate,
            "patch_radius": patch_radius,
        }

    center_tangent = np.mean(np.asarray(tangential_points), axis=0)
    center_error = float(np.linalg.norm(center_tangent))

    return {
        "center_error": center_error,
        "max_cross_error": float(max(cross_errors)) if cross_errors else 999.0,
        "mean_cross_error": float(np.mean(cross_errors)) if cross_errors else 999.0,
        "max_patch_radius": float(max(patch_radii)) if patch_radii else 999.0,
        "finger": details,
    }


def lane_relation_for_world_point(world_point, lane_spec, finger):
    if lane_spec is None:
        return {
            "valid": False,
            "cross_error": 999.0,
            "patch_radius": 999.0,
            "along_coordinate": 999.0,
        }

    anchor = lane_spec["anchor"]
    normal = lane_spec["normal"]
    band = lane_spec["band_axis"]
    scratch = lane_spec["scratch_axis"]

    tangent_delta = _project_tangent(np.asarray(world_point) - anchor, normal)
    band_coordinate = float(np.dot(tangent_delta, band))
    along_coordinate = float(np.dot(tangent_delta, scratch))
    desired_band = float(lane_spec["band_offsets"].get(finger, 0.0))
    cross_error = abs(band_coordinate - desired_band)
    patch_radius = float(np.linalg.norm(tangent_delta))

    return {
        "valid": bool(
            cross_error <= TARGET_LANE_CROSS_HARD_M
            and patch_radius <= TARGET_PATCH_RADIUS_HARD_M
        ),
        "cross_error": cross_error,
        "patch_radius": patch_radius,
        "along_coordinate": along_coordinate,
        "band_coordinate": band_coordinate,
        "desired_band": desired_band,
    }


def evaluate_geometry(
    arcs,
    mode,
    palm_position,
    palm_rotation,
    phases,
    active_fingers,
    strict=True,
):
    """
    Endpoint/contact-pose evaluation for V4.7.

    Important:
    The raw finger-arc tangent is DIAGNOSTIC ONLY here. It is NOT a hard
    rejection criterion, because the real scratch direction is created by:
        finger flex + J4/J5 + small J2/J3 compensation.

    Actual combined-motion tangency is checked later on the synchronized
    trajectory.

    Contact depth:
      ideal / preferred band : GOOD_SIGNED_*
      soft buffer            : between good band and HARD_PENETRATION_M
      hard reject            : below HARD_PENETRATION_M
    """
    active_set = set(
        active_fingers
    )

    contact_points = []
    contact_points_by_finger = {}
    normal_errors = []
    arc_tangent_scores = []
    potential_strokes = []
    active_surface_normals = []

    finger_results = {}

    active_valid = True
    endpoint_overpenetration = 0.0
    inactive_zero_penetration = 0.0

    signed_band_violation = 0.0
    surface_distance_excess = 0.0
    stroke_deficit_sum = 0.0

    for finger in LONG_FINGERS:

        if finger in active_set:
            phase = float(
                phases[finger]
            )

            local_point, local_tangent = interpolate_arc(
                arcs,
                finger,
                phase,
            )

            world_point = (
                palm_position
                +
                palm_rotation
                @ local_point
            )

            world_tangent = normalize(
                palm_rotation
                @ local_tangent
            )

            (
                distance,
                _,
                surface_normal,
                signed,
            ) = nearest_local(
                mode,
                world_point,
            )

            arc_tangent_score = clamp(
                1.0
                -
                abs(
                    float(
                        np.dot(
                            world_tangent,
                            surface_normal,
                        )
                    )
                ),
                0.0,
                1.0,
            )

            potential_stroke = estimate_available_arc_stroke(
                arcs,
                finger,
                phase,
            )

            reject = []
            warning = []

            if phase < MIN_MAIN_PHASE_DEG:
                reject.append(
                    "PHASE_LOW"
                )

            if phase > MAX_MAIN_PHASE_DEG:
                reject.append(
                    "PHASE_HIGH"
                )

            if signed < HARD_PENETRATION_M:
                reject.append(
                    "OVER_PENETRATION"
                )

            elif signed < GOOD_SIGNED_MIN_M:
                warning.append(
                    "SOFT_DEEP"
                )

            if signed > SOFT_SIGNED_MAX_M:
                reject.append(
                    "SIGNED_TOO_SHALLOW"
                )

            elif signed > GOOD_SIGNED_MAX_M:
                warning.append(
                    "SOFT_SHALLOW"
                )

            if distance > MAX_ACTIVE_SURFACE_DISTANCE_M:
                reject.append(
                    "SURFACE_DISTANCE"
                )

            if (
                strict
                and
                potential_stroke
                <
                MIN_USEFUL_STROKE_M
            ):
                reject.append(
                    "STROKE"
                )

            finger_valid = (
                len(
                    reject
                )
                == 0
            )

            active_valid = (
                active_valid
                and
                finger_valid
            )

            endpoint_overpenetration += max(
                0.0,
                HARD_PENETRATION_M
                -
                signed,
            )

            if signed < GOOD_SIGNED_MIN_M:
                signed_band_violation += (
                    GOOD_SIGNED_MIN_M
                    -
                    signed
                )

            elif signed > GOOD_SIGNED_MAX_M:
                signed_band_violation += (
                    signed
                    -
                    GOOD_SIGNED_MAX_M
                )

            surface_distance_excess += max(
                0.0,
                distance
                -
                MAX_ACTIVE_SURFACE_DISTANCE_M,
            )

            if strict:
                stroke_deficit_sum += max(
                    0.0,
                    MIN_USEFUL_STROKE_M
                    -
                    potential_stroke,
                )

            contact_points.append(
                world_point
            )
            contact_points_by_finger[finger] = np.asarray(world_point, dtype=np.float64)

            normal_errors.append(
                signed
                -
                IDEAL_SIGNED_M
            )

            arc_tangent_scores.append(
                arc_tangent_score
            )

            potential_strokes.append(
                potential_stroke
            )

            active_surface_normals.append(
                surface_normal
            )

            finger_results[
                finger
            ] = {
                "active":
                    True,

                "valid":
                    finger_valid,

                "reject":
                    (
                        "PASS"
                        if finger_valid
                        else "|".join(
                            reject
                        )
                    ),

                "warning":
                    (
                        "NONE"
                        if not warning
                        else "|".join(
                            warning
                        )
                    ),

                "phase":
                    phase,

                "signed":
                    signed,

                "distance":
                    distance,

                # Kept for diagnostics only.
                "tangent":
                    arc_tangent_score,

                "arc_tangent":
                    arc_tangent_score,

                "start":
                    None,

                "span":
                    0.0,

                "stroke":
                    potential_stroke,
            }

        else:
            # Endpoint planning evaluates inactive fingers at 0 deg only as
            # a preference. The real trajectory is allowed to flex them for
            # clearance (20/30/40 deg) and then validates the actual geometry.
            local_point, _ = interpolate_arc(
                arcs,
                finger,
                0.0,
            )

            world_point = (
                palm_position
                +
                palm_rotation
                @ local_point
            )

            (
                distance,
                _,
                _,
                signed,
            ) = nearest_global(
                mode,
                world_point,
            )

            excess = max(
                0.0,
                INACTIVE_PENETRATION_M
                -
                signed,
            )

            inactive_zero_penetration += (
                excess
            )

            finger_results[
                finger
            ] = {
                "active":
                    False,

                # Not a hard endpoint rejection in V4.4.
                "valid":
                    True,

                "reject":
                    "PASS",

                "warning":
                    (
                        "NONE"
                        if excess
                        <= 1e-9
                        else "INACTIVE_ZERO_COLLISION"
                    ),

                "phase":
                    0.0,

                "signed":
                    signed,

                "distance":
                    distance,

                "tangent":
                    None,

                "arc_tangent":
                    None,

                "start":
                    None,

                "span":
                    0.0,

                "stroke":
                    0.0,
            }

    if len(
        contact_points
    ) == 0:
        return {
            "valid":
                False,

            "strong":
                False,

            "cost":
                1e9,

            "reject":
                "NO_ACTIVE_CONTACT",

            "coverage":
                999.0,

            "normal_rmse":
                999.0,

            "tangent":
                0.0,

            "arc_tangent":
                0.0,

            "stroke":
                0.0,

            "span":
                0.0,

            "palm_back_angle_deg":
                90.0,

            "palm_parallel_score":
                0.0,

            "finger_results":
                finger_results,
        }

    target_error = coverage_error(
        mode,
        contact_points,
    )

    lane_spec = build_target_lane_spec(
        arcs,
        mode,
        palm_rotation,
        phases,
        active_fingers,
    )
    lane_metrics = target_lane_metrics(
        contact_points_by_finger,
        lane_spec,
    )

    normal_rmse = math.sqrt(
        float(
            np.mean(
                np.square(
                    normal_errors
                )
            )
        )
    )

    arc_tangent_mean = float(
        np.mean(
            arc_tangent_scores
        )
    )

    mean_potential_stroke = float(
        np.mean(
            potential_strokes
        )
    )


    local_back_normal = mean_contact_normal(
        active_surface_normals,
        mode[
            "normal"
        ],
    )

    palm_alignment = palm_back_alignment(
        arcs,
        palm_rotation,
        local_back_normal,
    )

    palm_back_angle_deg = float(
        palm_alignment[
            "angle_deg"
        ]
    )

    root_axis = centroid_base_axis_alignment(
        arcs,
        palm_position,
        palm_rotation,
        world_points_by_finger=contact_points_by_finger,
    )
    root_base_axis_angle_deg = float(root_axis["angle_deg"])
    arc_height_order = finger_arc_height_order(
        arcs,
        palm_rotation,
        local_back_normal,
        phases,
        active_fingers,
    )
    arc_height_violation_m = float(arc_height_order["max_violation_m"])

    depth_balance = contact_depth_balance_from_results(
        finger_results,
        active_fingers,
    )

    multi_balance_cost = 0.0

    if depth_balance[
        "count"
    ] >= 2:
        multi_balance_cost = (
            MULTI_MEAN_DEPTH_WEIGHT
            *
            (
                depth_balance[
                    "mean_error"
                ]
                /
                0.0020
            ) ** 2

            +

            MULTI_DEPTH_SPREAD_WEIGHT
            *
            (
                depth_balance[
                    "std_signed"
                ]
                /
                0.0015
            ) ** 2

            +

            MULTI_DEPTH_RANGE_WEIGHT
            *
            (
                depth_balance[
                    "range_signed"
                ]
                /
                0.0030
            ) ** 2
        )

    valid = (
        active_valid
        and
        endpoint_overpenetration <= 1e-9
        and
        lane_metrics["center_error"] <= TARGET_PATCH_CENTER_HARD_M
        and
        lane_metrics["max_cross_error"] <= TARGET_LANE_CROSS_HARD_M
        and
        lane_metrics["max_patch_radius"] <= TARGET_PATCH_RADIUS_HARD_M
        and
        palm_back_angle_deg <= palm_alignment_limit_deg()
        and
        root_base_axis_angle_deg <= ROOT_BASE_AXIS_HARD_DEG
        and
        bool(arc_height_order["valid"])
    )

    strong = (
        valid
        and
        lane_metrics["center_error"] <= TARGET_PATCH_CENTER_STRONG_M
        and
        lane_metrics["max_cross_error"] <= TARGET_LANE_CROSS_STRONG_M
        and
        lane_metrics["max_patch_radius"] <= PLANNED_ACTIVE_PATCH_RADIUS_M
        and
        normal_rmse <= STRONG_NORMAL_RMSE_M
        and
        mean_potential_stroke >= STRONG_STROKE_M
        and
        palm_back_angle_deg <= PALM_ALIGNMENT_STRONG_DEG
        and
        root_base_axis_angle_deg <= ROOT_BASE_AXIS_STRONG_DEG
        and
        bool(arc_height_order["valid"])
    )

    reject_reasons = []

    active_rejects = []

    for finger in active_fingers:
        result = finger_results.get(
            finger,
            {},
        )

        if not result.get(
            "valid",
            False,
        ):
            active_rejects.append(
                finger
                +
                "("
                +
                result.get(
                    "reject",
                    "UNKNOWN",
                )
                +
                ")"
            )

    if active_rejects:
        reject_reasons.append(
            "ACTIVE_ENDPOINT:"
            +
            ",".join(
                active_rejects
            )
        )

    if endpoint_overpenetration > 1e-9:
        reject_reasons.append("ENDPOINT_NONWORKING_PENETRATION")

    if lane_metrics["center_error"] > TARGET_PATCH_CENTER_HARD_M:
        reject_reasons.append("TARGET_PATCH_CENTER")

    if lane_metrics["max_cross_error"] > TARGET_LANE_CROSS_HARD_M:
        reject_reasons.append("TARGET_LANE_CROSS")

    if lane_metrics["max_patch_radius"] > TARGET_PATCH_RADIUS_HARD_M:
        reject_reasons.append("TARGET_PATCH_RADIUS")

    if palm_back_angle_deg > palm_alignment_limit_deg():
        reject_reasons.append("PALM_ALIGNMENT")

    if root_base_axis_angle_deg > ROOT_BASE_AXIS_HARD_DEG:
        reject_reasons.append("ROOT_BASE_AXIS_ALIGNMENT")

    if not bool(arc_height_order["valid"]):
        reject_reasons.append("FINGER_ARC_HEIGHT_ORDER")

    # Optimization still strongly prefers the normal contact band.
    # Soft-band excursions are allowed but expensive.
    cost = (
        TARGET_PATCH_LOCK_WEIGHT
        *
        (lane_metrics["center_error"] / max(TARGET_PATCH_CENTER_HARD_M, 1e-6)) ** 2

        +

        TARGET_LANE_LOCK_WEIGHT
        *
        (lane_metrics["max_cross_error"] / max(TARGET_LANE_CROSS_HARD_M, 1e-6)) ** 2

        +

        0.25
        *
        TARGET_PATCH_LOCK_WEIGHT
        *
        (lane_metrics["max_patch_radius"] / max(TARGET_PATCH_RADIUS_HARD_M, 1e-6)) ** 2

        +

        10.0
        *
        (
            normal_rmse
            /
            0.003
        ) ** 2

        +

        28.0
        *
        (
            signed_band_violation
            /
            0.001
        ) ** 2

        +

        25.0
        *
        (
            surface_distance_excess
            /
            0.003
        ) ** 2

        +

        12.0
        *
        (
            stroke_deficit_sum
            /
            max(
                MIN_USEFUL_STROKE_M,
                1e-6,
            )
        ) ** 2

        +

        120.0
        *
        (
            endpoint_overpenetration
            /
            0.001
        ) ** 2

        +

        # Prefer endpoints where straight inactive fingers are already clear,
        # but do NOT kill a candidate; trajectory clearance flex may solve it.
        8.0
        *
        (
            inactive_zero_penetration
            /
            0.001
        ) ** 2

        +

        PALM_ALIGNMENT_WEIGHT
        *
        (
            palm_back_angle_deg
            /
            palm_alignment_scale_deg()
        ) ** 2

        +

        ROOT_BASE_AXIS_WEIGHT
        *
        (
            root_base_axis_angle_deg
            /
            ROOT_BASE_AXIS_SCALE_DEG
        ) ** 2

        +

        ARC_HEIGHT_ORDER_WEIGHT
        *
        (
            arc_height_violation_m
            /
            max(ARC_HEIGHT_ORDER_TOL_M, 1e-6)
        ) ** 2

        +

        multi_balance_cost
    )

    return {
        "valid":
            valid,

        "strong":
            strong,

        "cost":
            float(
                cost
            ),

        "reject":
            (
                "PASS"
                if valid
                else "|".join(
                    reject_reasons
                )
            ),

        "coverage":
            target_error,

        "target_center_error":
            lane_metrics["center_error"],

        "lane_cross_max":
            lane_metrics["max_cross_error"],

        "patch_radius_max":
            lane_metrics["max_patch_radius"],

        "lane_spec":
            lane_spec,

        "lane_details":
            lane_metrics["finger"],

        "normal_rmse":
            normal_rmse,

        # Compatibility field. This is raw-arc diagnostic only.
        "tangent":
            arc_tangent_mean,

        "arc_tangent":
            arc_tangent_mean,

        "stroke":
            mean_potential_stroke,

        "span":
            0.0,

        "inactive_zero_penetration":
            inactive_zero_penetration,

        "palm_back_angle_deg":
            palm_back_angle_deg,

        "root_base_axis_angle_deg":
            root_base_axis_angle_deg,

        "base_to_root_world":
            root_axis["base_to_root"],

        "base_to_centroid_world":
            root_axis.get("base_to_centroid", root_axis["base_to_root"]),

        "finger_row_axis_world":
            root_axis.get("finger_row_axis", np.zeros(3, dtype=np.float64)),

        "four_finger_centroid_world":
            root_axis.get("centroid_world", np.zeros(3, dtype=np.float64)),

        "finger_forward_world":
            root_axis["finger_forward"],

        "arc_height_order":
            arc_height_order,

        "arc_height_violation_m":
            arc_height_violation_m,

        "palm_parallel_score":
            float(
                palm_alignment[
                    "parallel_score"
                ]
            ),

        "palm_normal_world":
            palm_alignment[
                "palm_normal"
            ],

        "back_normal_world":
            local_back_normal,

        "contact_mean_signed":
            depth_balance[
                "mean_signed"
            ],

        "contact_mean_error":
            depth_balance[
                "mean_error"
            ],

        "contact_depth_std":
            depth_balance[
                "std_signed"
            ],

        "contact_depth_range":
            depth_balance[
                "range_signed"
            ],

        "multi_balance_cost":
            float(
                multi_balance_cost
            ),

        "finger_results":
            finger_results,
    }

# ============================================================
# STRATEGIES
# ============================================================

def generate_templates(mode):
    templates = []

    def add(name, sequence, phases, family):
        templates.append(
            {
                "name": name,
                "family": family,
                "sequence": list(sequence),
                "active": tuple(sequence),
                "phases": {
                    f:
                        float(
                            phases[f]
                        )
                    for f
                    in sequence
                },
            }
        )

    if mode["kind"] == "flat":
        # Human-like sequence: pinky -> ring -> middle -> index.
        add(
            "HUMAN_WAVE_4_A",
            ["pinky", "ring", "middle", "index"],
            {
                "pinky": 54.0,
                "ring": 54.0,
                "middle": 54.0,
                "index": 54.0,
            },
            "human_wave",
        )

        add(
            "HUMAN_WAVE_4_B",
            ["pinky", "ring", "middle", "index"],
            {
                "pinky": 52.0,
                "ring": 52.0,
                "middle": 52.0,
                "index": 52.0,
            },
            "human_wave",
        )

        add(
            "HUMAN_WAVE_4_C",
            ["pinky", "ring", "middle", "index"],
            {
                "pinky": 50.0,
                "ring": 50.0,
                "middle": 50.0,
                "index": 50.0,
            },
            "human_wave",
        )

        add(
            "WAVE_3_PRM_A",
            ["pinky", "ring", "middle"],
            {
                "pinky": 86.0,
                "ring": 80.0,
                "middle": 74.0,
            },
            "three",
        )

        add(
            "WAVE_3_PRM_B",
            ["pinky", "ring", "middle"],
            {
                "pinky": 78.0,
                "ring": 72.0,
                "middle": 66.0,
            },
            "three",
        )

        add(
            "WAVE_3_RMI_A",
            ["ring", "middle", "index"],
            {
                "ring": 86.0,
                "middle": 80.0,
                "index": 74.0,
            },
            "three",
        )

        add(
            "WAVE_3_RMI_B",
            ["ring", "middle", "index"],
            {
                "ring": 78.0,
                "middle": 72.0,
                "index": 66.0,
            },
            "three",
        )

        for a, b in [
            ("pinky", "ring"),
            ("ring", "middle"),
            ("middle", "index"),
        ]:
            add(
                "PAIR_"
                + a
                + "_"
                + b
                + "_A",
                [a, b],
                {
                    a: 84.0,
                    b: 76.0,
                },
                "pair",
            )

            add(
                "PAIR_"
                + a
                + "_"
                + b
                + "_B",
                [a, b],
                {
                    a: 74.0,
                    b: 66.0,
                },
                "pair",
            )

        for finger in LONG_FINGERS:
            add(
                "SINGLE_"
                + finger
                + "_A",
                [finger],
                {
                    finger: 82.0
                },
                "single",
            )

            add(
                "SINGLE_"
                + finger
                + "_B",
                [finger],
                {
                    finger: 68.0
                },
                "single",
            )

    else:
        for a, b in [
            ("pinky", "ring"),
            ("ring", "middle"),
            ("middle", "index"),
        ]:
            add(
                "V_PAIR_"
                + a
                + "_"
                + b
                + "_A",
                [a, b],
                {
                    a: 86.0,
                    b: 78.0,
                },
                "v_pair",
            )

            add(
                "V_PAIR_"
                + a
                + "_"
                + b
                + "_B",
                [a, b],
                {
                    a: 74.0,
                    b: 66.0,
                },
                "v_pair",
            )

        for finger in LONG_FINGERS:
            add(
                "V_SINGLE_"
                + finger
                + "_A",
                [finger],
                {
                    finger: 84.0
                },
                "v_single",
            )

            add(
                "V_SINGLE_"
                + finger
                + "_B",
                [finger],
                {
                    finger: 68.0
                },
                "v_single",
            )

    return templates


def template_local_frame(arcs, template):
    points = []
    tangents = []

    for finger in template["active"]:
        point, tangent = interpolate_arc(arcs, finger, template["phases"][finger])
        points.append(point)
        tangents.append(tangent)

    points = np.asarray(points, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)

    centroid = np.mean(points, axis=0)
    scratch_axis = normalize(np.mean(tangents, axis=0))

    if len(points) >= 2:
        centered = points - centroid[None, :]
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        band_axis = vh[0]
    else:
        band_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    band_axis = band_axis - float(np.dot(band_axis, scratch_axis)) * scratch_axis
    band_axis = normalize(band_axis)

    if band_axis is None:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        band_axis = helper - float(np.dot(helper, scratch_axis)) * scratch_axis
        band_axis = normalize(band_axis)

    local_normal = normalize(np.cross(scratch_axis, band_axis))
    band_axis = normalize(np.cross(local_normal, scratch_axis))

    return centroid, np.column_stack([scratch_axis, band_axis, local_normal])


def make_palm_seed(arcs, template, mode, yaw_deg):
    local_center, local_frame = template_local_frame(arcs, template)

    world_scratch = mode["v"].copy()

    yaw_rotation = Rotation.from_rotvec(
        mode["normal"] * math.radians(float(yaw_deg))
    ).as_matrix()

    world_scratch = normalize(yaw_rotation @ world_scratch)
    world_band = normalize(np.cross(mode["normal"], world_scratch))
    world_scratch = normalize(np.cross(world_band, mode["normal"]))

    world_frame = np.column_stack([world_scratch, world_band, mode["normal"]])
    palm_rotation = world_frame @ local_frame.T

    desired_center = mode["anchor"] + IDEAL_SIGNED_M * mode["normal"]
    palm_position = desired_center - palm_rotation @ local_center

    return palm_position, palm_rotation


def optimize_palm(arcs, template, mode, yaw_deg):
    seed_position, seed_rotation = make_palm_seed(arcs, template, mode, yaw_deg)

    lower = np.array(
        [
            -PALM_TRANSLATION_SEARCH_M,
            -PALM_TRANSLATION_SEARCH_M,
            -PALM_TRANSLATION_SEARCH_M,
            -math.radians(palm_alignment_limit_deg()),
            -math.radians(palm_alignment_limit_deg()),
            -math.radians(palm_alignment_limit_deg()),
        ],
        dtype=np.float64,
    )

    upper = -lower

    def decode(x):
        p = seed_position + x[:3]
        r = Rotation.from_rotvec(x[3:]).as_matrix() @ seed_rotation
        return p, r

    def objective(x):
        p, r = decode(x)

        g = evaluate_geometry(
            arcs,
            mode,
            p,
            r,
            template["phases"],
            template["active"],
            strict=True,
        )

        reg = (
            0.08 * float(np.sum(np.square(x[:3] / 0.025)))
            + 0.04 * float(np.sum(np.square(x[3:] / math.radians(20.0))))
        )

        return g["cost"] + reg

    result = minimize(
        objective,
        np.zeros(6, dtype=np.float64),
        method="Powell",
        bounds=list(zip(lower, upper)),
        options={
            "maxiter": PALM_OPT_MAXITER,
            "xtol": 1e-4,
            "ftol": 1e-4,
            "disp": False,
        },
    )

    p, r = decode(np.asarray(result.x, dtype=np.float64))

    g = evaluate_geometry(
        arcs,
        mode,
        p,
        r,
        template["phases"],
        template["active"],
        strict=True,
    )

    return p, r, g


# ============================================================
# IK / q REFINEMENT
# ============================================================

def joint_margin_deg(q, lower, upper):
    q_deg = np.degrees(q)
    return float(
        np.min(
            np.minimum(
                q_deg - np.degrees(lower),
                np.degrees(upper) - q_deg,
            )
        )
    )


def build_q_seeds(current_q, home_q, lower, upper, _strategy_heading_deg=0.0):
    seeds = [current_q.copy(), home_q.copy()]

    q123_families = [
        [0.0, -10.0, -110.0],
        [0.0, -20.0, -105.0],
        [0.0, -30.0, -95.0],
        [0.0, -42.0, -82.0],
        [20.0, -25.0, -100.0],
        [-20.0, -25.0, -100.0],
        [35.0, -30.0, -92.0],
        [-35.0, -30.0, -92.0],
    ]

    q4_values = [
        0.0,
        -30.0,
        30.0,
        -60.0,
        60.0,
        -88.0,
        88.0,
    ]

    q5_values = [-70.0, -55.0, -40.0, -25.0, -10.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0]

    for q123 in q123_families:
        for q4 in q4_values:
            for q5 in q5_values:
                q = np.radians(
                    np.array([q123[0], q123[1], q123[2], q4, q5], dtype=np.float64)
                )

                q = np.maximum(q, lower)
                q = np.minimum(q, upper)
                seeds.append(q)

    unique = []
    seen = set()

    for q in seeds:
        key = tuple(np.round(np.degrees(q), 1))

        if key in seen:
            continue

        seen.add(key)
        unique.append(q)

    return unique


def build_continuous_refine_seeds(current_q, home_q, lower, upper):
    """Small, joint-space-diverse seed set for the authoritative 5-DOF solve.

    These are only numerical starting points.  None of the five joints is
    fixed, and no task-space heading is mapped onto J4 or J5.
    """
    seed_degrees = [
        [0.0, -20.0, -105.0, 0.0, 0.0],
        [0.0, -20.0, -105.0, 30.0, 75.0],
        [0.0, -20.0, -105.0, -30.0, -75.0],
        [0.0, -42.0, -82.0, 30.0, 75.0],
        [0.0, -10.0, -110.0, -30.0, 60.0],
        [20.0, -25.0, -100.0, 30.0, 60.0],
        [-20.0, -25.0, -100.0, -30.0, 60.0],
        [0.0, -30.0, -95.0, 60.0, -40.0],
    ]
    seeds = [
        np.asarray(current_q, dtype=np.float64).copy(),
        np.asarray(home_q, dtype=np.float64).copy(),
    ]
    for values in seed_degrees:
        q = np.radians(np.asarray(values, dtype=np.float64))
        seeds.append(np.minimum(np.maximum(q, lower), upper))

    unique = []
    seen = set()
    for q in seeds:
        key = tuple(np.round(np.degrees(q), 1))
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def solve_lula_ik(
    solver,
    position,
    rotation,
    seeds,
    lower,
    upper,
    diagnostics=None,
):
    solutions = []
    target_orientation_wxyz = rotation_matrix_to_wxyz(rotation)
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(
        {
            "orientation_attempts": 0,
            "orientation_solver_fail": 0,
            "position_only_attempts": 0,
            "position_only_solver_fail": 0,
            "exceptions": 0,
            "joint_limit_rejects": 0,
            "accepted_raw": 0,
            "accepted_unique": 0,
            "last_exception": None,
            "orientation_input_shape": tuple(target_orientation_wxyz.shape),
        }
    )

    for tolerance in IK_ORIENTATION_TOL_RAD:
        for seed in seeds:
            diagnostics["orientation_attempts"] += 1
            try:
                q, success = solver.compute_inverse_kinematics(
                    frame_name=PALM_FRAME,
                    target_position=position,
                    target_orientation=target_orientation_wxyz,
                    warm_start=seed,
                    position_tolerance=IK_POSITION_TOL_M,
                    orientation_tolerance=tolerance,
                )
            except Exception as exc:
                diagnostics["exceptions"] += 1
                diagnostics["last_exception"] = repr(exc)
                continue

            if not success:
                diagnostics["orientation_solver_fail"] += 1
                continue

            q = np.asarray(q, dtype=np.float64)

            if np.any(q < lower - 1e-8) or np.any(q > upper + 1e-8):
                diagnostics["joint_limit_rejects"] += 1
                continue

            fk_p, fk_r = solver.compute_forward_kinematics(PALM_FRAME, q)

            solutions.append(
                {
                    "kind": "orientation_ik",
                    "q": q,
                    "position": np.asarray(fk_p, dtype=np.float64),
                    "rotation": np.asarray(fk_r, dtype=np.float64),
                }
            )
            diagnostics["accepted_raw"] += 1

        if solutions:
            break

    # Position-only IK is a fallback, not a parallel candidate source. Mixing
    # it into a successful orientation-constrained pool can reintroduce the
    # 155-180 degree palm flips that the target/lane checks are meant to avoid.
    if ENABLE_POSITION_ONLY_IK and not solutions:
        for seed in seeds:
            diagnostics["position_only_attempts"] += 1
            try:
                q, success = solver.compute_inverse_kinematics(
                    frame_name=PALM_FRAME,
                    target_position=position,
                    target_orientation=None,
                    warm_start=seed,
                    position_tolerance=IK_POSITION_TOL_M,
                )
            except Exception as exc:
                diagnostics["exceptions"] += 1
                diagnostics["last_exception"] = repr(exc)
                continue

            if not success:
                diagnostics["position_only_solver_fail"] += 1
                continue

            q = np.asarray(q, dtype=np.float64)

            if np.any(q < lower - 1e-8) or np.any(q > upper + 1e-8):
                diagnostics["joint_limit_rejects"] += 1
                continue

            fk_p, fk_r = solver.compute_forward_kinematics(PALM_FRAME, q)

            solutions.append(
                {
                    "kind": "position_only_ik",
                    "q": q,
                    "position": np.asarray(fk_p, dtype=np.float64),
                    "rotation": np.asarray(fk_r, dtype=np.float64),
                }
            )
            diagnostics["accepted_raw"] += 1

    unique = []
    seen = set()

    for item in solutions:
        key = tuple(np.round(np.degrees(item["q"]), 1))

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    diagnostics["accepted_unique"] = len(unique)
    return unique


def contact_posture_penalty(
    q,
    reference_q=None,
):
    """
    V4.8 posture regularizer.

    Priorities:
      1. Palm/back geometric alignment is handled by evaluate_geometry() and
         dynamic_keyframe_geometry().
      2. J4 is SOFTLY preferred near 0 deg for a cleaner-looking posture.
      3. J4 remains fully free to move when that improves palm alignment,
         contact, or scratch safety.
      4. J5 has NO preferred sign and NO preferred numerical angle.
      5. J1/J2/J3 motion remains more expensive than local wrist motion.
    """
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    penalty = 0.0

    j4_deg = float(
        np.degrees(
            q[
                3
            ]
        )
    )

    penalty += (
        J4_ZERO_SOFT_WEIGHT
        *
        (
            j4_deg
            /
            J4_ZERO_SOFT_SCALE_DEG
        ) ** 2
    )

    if reference_q is not None:
        delta_deg = np.degrees(
            q
            -
            np.asarray(
                reference_q,
                dtype=np.float64,
            )
        )

        penalty += (
            0.0025
            *
            float(
                np.sum(
                    SYNC_JOINT_MOVE_WEIGHTS
                    *
                    np.square(
                        delta_deg
                    )
                )
            )
        )

    return float(
        penalty
    )


def refine_multifinger_balance(
    solver,
    arcs,
    mode,
    template,
    seed_q,
    lower,
    upper,
    reference_q,
    strategy_yaw,
):
    """
    V4.11 dedicated multi-finger rescue/refinement.

    Unlike the old endpoint test, this optimizer is allowed to change BOTH:
      - J1..J5, with J4/J5 deliberately cheapest and most free;
      - each active finger phase within a small window around the template.

    Therefore:
      * all fingers too deep/shallow -> wrist/J5/common pose can correct;
      * one side deep and the other side shallow -> J4/J5 can level the palm;
      * local back curvature -> individual finger phases may differ slightly.

    This is attempted BEFORE degrading to fewer fingers.
    """
    active = list(
        template[
            "active"
        ]
    )

    if len(
        active
    ) < 2:
        return None

    seed_q = np.asarray(
        seed_q,
        dtype=np.float64,
    )

    reference_q = np.asarray(
        reference_q,
        dtype=np.float64,
    )

    q_lower = np.maximum(
        lower,
        seed_q
        -
        np.radians(
            MULTI_BALANCE_LOCAL_DELTA_DEG
        ),
    )

    q_upper = np.minimum(
        upper,
        seed_q
        +
        np.radians(
            MULTI_BALANCE_LOCAL_DELTA_DEG
        ),
    )

    # The phase optimizer must start from geometry measured at this palm pose,
    # not from the old fixed 50--54 degree template.  Re-evaluate the current
    # local back normal, then move any height-order failures directly to their
    # dynamically sampled minima before constructing the joint/phase seed.
    seed_phase_map = dict(template["phases"])
    try:
        seed_p, seed_r = solver.compute_forward_kinematics(PALM_FRAME, seed_q)
        seed_p = np.asarray(seed_p, dtype=np.float64)
        seed_r = np.asarray(seed_r, dtype=np.float64)
        seed_geometry = evaluate_geometry(
            arcs,
            mode,
            seed_p,
            seed_r,
            seed_phase_map,
            active,
            strict=False,
        )
        dynamic_seed_phases, dynamic_seed_adjustments, _ = (
            repair_arc_height_order_phases(
                arcs,
                seed_r,
                seed_geometry["back_normal_world"],
                seed_phase_map,
                active,
            )
        )
        if dynamic_seed_adjustments:
            seed_phase_map = dict(dynamic_seed_phases)
            print(
                "  [V102 DYNAMIC PHASE SEED]",
                "template=", template["name"],
                "phases=", {
                    finger: round(float(seed_phase_map[finger]), 2)
                    for finger in active
                },
            )
    except Exception:
        # A failed diagnostic FK must not erase the original candidate; the
        # strict evaluation below will still reject it if no valid FK exists.
        seed_phase_map = dict(template["phases"])

    phase0 = np.asarray(
        [float(seed_phase_map[finger]) for finger in active],
        dtype=np.float64,
    )

    # Use the complete executable phase interval.  The dynamic minimum is the
    # seed and regularization centre; it is no longer clipped by a fixed window
    # around the legacy template angle.
    phase_lower = np.full(len(active), MIN_MAIN_PHASE_DEG, dtype=np.float64)
    phase_upper = np.full(len(active), MAX_MAIN_PHASE_DEG, dtype=np.float64)

    lower_x = np.concatenate(
        [
            q_lower,
            phase_lower,
        ]
    )

    upper_x = np.concatenate(
        [
            q_upper,
            phase_upper,
        ]
    )

    seed_x = np.concatenate(
        [
            np.clip(
                seed_q,
                q_lower,
                q_upper,
            ),
            phase0,
        ]
    )
    seed_x = np.clip(seed_x, lower_x, upper_x)

    def decode(x):
        x = np.asarray(
            x,
            dtype=np.float64,
        )

        q = x[
            :5
        ]

        phases = dict(
            template[
                "phases"
            ]
        )

        for index, finger in enumerate(
            active
        ):
            phases[
                finger
            ] = float(
                x[
                    5
                    +
                    index
                ]
            )

        return (
            q,
            phases,
        )

    def objective(x):
        q, phases = decode(
            x
        )

        try:
            p, r = solver.compute_forward_kinematics(
                PALM_FRAME,
                q,
            )
        except Exception:
            return 1e12

        p = np.asarray(
            p,
            dtype=np.float64,
        )

        r = np.asarray(
            r,
            dtype=np.float64,
        )

        geometry = evaluate_geometry(
            arcs,
            mode,
            p,
            r,
            phases,
            active,
            strict=False,
        )

        q_delta_deg = np.degrees(
            q
            -
            seed_q
        )

        motion_cost = (
            0.0015
            *
            float(
                np.sum(
                    MULTI_BALANCE_MOVE_WEIGHTS
                    *
                    np.square(
                        q_delta_deg
                    )
                )
            )
        )

        phase_delta = np.asarray(
            [
                phases[finger]
                -
                float(seed_phase_map[finger])
                for finger
                in active
            ],
            dtype=np.float64,
        )

        phase_cost = (
            MULTI_PHASE_REG_WEIGHT
            *
            float(
                np.mean(
                    np.square(
                        phase_delta
                        /
                        max(
                            MULTI_BALANCE_PHASE_WINDOW_DEG,
                            1.0,
                        )
                    )
                )
            )
        )

        template_gaps = []
        current_gaps = []

        for gap_index in range(
            len(
                active
            )
            -
            1
        ):
            finger_a = active[
                gap_index
            ]

            finger_b = active[
                gap_index
                +
                1
            ]

            template_gap = (
                float(seed_phase_map[finger_a])
                -
                float(seed_phase_map[finger_b])
            )

            current_gap = (
                float(
                    phases[
                        finger_a
                    ]
                )
                -
                float(
                    phases[
                        finger_b
                    ]
                )
            )

            template_gaps.append(
                template_gap
            )

            current_gaps.append(
                current_gap
            )

        if current_gaps:
            template_gaps = np.asarray(
                template_gaps,
                dtype=np.float64,
            )

            current_gaps = np.asarray(
                current_gaps,
                dtype=np.float64,
            )

            phase_shape_cost = (
                MULTI_PHASE_SHAPE_WEIGHT
                *
                float(
                    np.mean(
                        np.square(
                            (
                                current_gaps
                                -
                                template_gaps
                            )
                            /
                            2.0
                        )
                    )
                )
            )

            wave_gap_violation = np.maximum(
                0.0,
                MULTI_MIN_WAVE_GAP_DEG
                -
                current_gaps,
            )

            wave_gap_cost = (
                MULTI_WAVE_GAP_VIOLATION_WEIGHT
                *
                float(
                    np.sum(
                        np.square(
                            wave_gap_violation
                        )
                    )
                )
            )

        else:
            phase_shape_cost = 0.0
            wave_gap_cost = 0.0

        center_error = float(
            geometry.get(
                "target_center_error",
                999.0,
            )
        )

        target_center_cost = (
            MULTI_TARGET_CENTER_WEIGHT
            *
            (
                center_error
                /
                0.015
            ) ** 2
        )

        center_excess = max(
            0.0,
            center_error
            -
            MULTI_CENTER_SOFT_LIMIT_M,
        )

        center_barrier_cost = (
            MULTI_CENTER_BARRIER_WEIGHT
            *
            (
                center_excess
                /
                0.002
            ) ** 2
        )

        arc_tangent = float(
            geometry.get(
                "arc_tangent",
                0.0,
            )
        )

        tangent_deficit = max(
            0.0,
            MULTI_TANGENT_SOFT_TARGET
            -
            arc_tangent,
        )

        tangent_cost = (
            MULTI_TANGENT_WEIGHT
            *
            (
                tangent_deficit
                /
                MULTI_TANGENT_SOFT_TARGET
            ) ** 2
        )

        # J4=0 remains aesthetic only. Do not let it overpower contact leveling.
        j4_deg = abs(
            float(
                np.degrees(
                    q[
                        3
                    ]
                )
            )
        )

        j4_tie_break = (
            0.05
            *
            (
                j4_deg
                /
                45.0
            ) ** 2
        )

        return (
            float(
                geometry[
                    "cost"
                ]
            )
            +
            motion_cost
            +
            phase_cost
            +
            phase_shape_cost
            +
            wave_gap_cost
            +
            target_center_cost
            +
            center_barrier_cost
            +
            tangent_cost
            +
            j4_tie_break
        )

    seed_variants = [
        seed_x,
    ]

    # Explicit J4/J5 perturbations help Powell discover the two intuitive
    # leveling directions instead of staying trapped in the original IK pose.
    for j4_delta, j5_delta in [
        (-18.0, 0.0),
        (18.0, 0.0),
        (-32.0, 0.0),
        (32.0, 0.0),
        (0.0, -18.0),
        (0.0, 18.0),
    ]:
        variant = seed_x.copy()

        variant[
            3
        ] = clamp(
            variant[
                3
            ]
            +
            math.radians(
                j4_delta
            ),
            q_lower[
                3
            ],
            q_upper[
                3
            ],
        )

        variant[
            4
        ] = clamp(
            variant[
                4
            ]
            +
            math.radians(
                j5_delta
            ),
            q_lower[
                4
            ],
            q_upper[
                4
            ],
        )

        seed_variants.append(
            variant
        )

    best = None

    for seed_variant_index, seed_variant in enumerate(seed_variants):
        result = minimize(
            objective,
            seed_variant,
            method="Powell",
            bounds=list(
                zip(
                    lower_x,
                    upper_x,
                )
            ),
            options={
                "maxiter":
                    MULTI_BALANCE_MAXITER,

                "xtol":
                    1e-4,

                "ftol":
                    1e-4,

                "disp":
                    False,
            },
        )

        q, phases = decode(
            result.x
        )

        try:
            p, r = solver.compute_forward_kinematics(
                PALM_FRAME,
                q,
            )
        except Exception:
            continue

        p = np.asarray(
            p,
            dtype=np.float64,
        )

        r = np.asarray(
            r,
            dtype=np.float64,
        )

        geometry = evaluate_geometry(
            arcs,
            mode,
            p,
            r,
            phases,
            active,
            strict=True,
        )

        # Changing J1--J5 changes the projection of every measured finger arc
        # onto the local back normal, so one post-hoc repair is insufficient.
        # Alternate dynamic minimum measurement and five-joint correction until
        # the strict geometry passes or a different hard constraint proves this
        # branch infeasible.  No height threshold is relaxed here.
        for dynamic_iteration in range(ARC_HEIGHT_DYNAMIC_COUPLED_ITERATIONS):
            if str(geometry.get("reject", "")) != "FINGER_ARC_HEIGHT_ORDER":
                break
            repaired_phases, phase_adjustments, repaired_order = (
                repair_arc_height_order_phases(
                    arcs,
                    r,
                    geometry["back_normal_world"],
                    phases,
                    active,
                )
            )
            if not phase_adjustments or not bool(repaired_order["valid"]):
                break

            repaired_phase_vector = np.asarray(
                [float(repaired_phases[finger]) for finger in active],
                dtype=np.float64,
            )

            def repaired_q_objective(q_values):
                coupled_x = np.concatenate(
                    [
                        np.asarray(q_values, dtype=np.float64),
                        repaired_phase_vector,
                    ]
                )
                return objective(coupled_x)

            coupled_result = minimize(
                repaired_q_objective,
                np.asarray(q, dtype=np.float64),
                method="Powell",
                bounds=list(zip(q_lower, q_upper)),
                options={
                    "maxiter": ARC_HEIGHT_COUPLED_Q_MAXITER,
                    "xtol": 1e-4,
                    "ftol": 1e-4,
                    "disp": False,
                },
            )
            coupled_q = np.asarray(coupled_result.x, dtype=np.float64)
            try:
                coupled_p, coupled_r = solver.compute_forward_kinematics(
                    PALM_FRAME,
                    coupled_q,
                )
            except Exception:
                break

            coupled_p = np.asarray(coupled_p, dtype=np.float64)
            coupled_r = np.asarray(coupled_r, dtype=np.float64)
            coupled_geometry = evaluate_geometry(
                arcs,
                mode,
                coupled_p,
                coupled_r,
                repaired_phases,
                active,
                strict=True,
            )
            print(
                "  [V102 DYNAMIC HEIGHT COUPLING]",
                "iteration=", dynamic_iteration + 1,
                "template=", template["name"],
                "phase_adjustments=", {
                    finger: {
                        key: round(float(value), 2)
                        for key, value in values.items()
                    }
                    for finger, values in phase_adjustments.items()
                },
                "q_delta_deg=",
                np.round(np.degrees(coupled_q - q), 2),
                "strict_recheck=",
                "PASS" if bool(coupled_geometry.get("valid", False))
                else str(coupled_geometry.get("reject", "UNKNOWN")),
            )
            q = coupled_q
            phases = dict(repaired_phases)
            p = coupled_p
            r = coupled_r
            geometry = coupled_geometry

        # Endpoint contact candidates tens of centimetres from the back are a
        # wrong local optimizer branch, never a useful four-finger solution.
        # Drop them before ranking so they cannot displace a nearby candidate.
        if abs(float(geometry.get("contact_mean_signed", 999.0))) > (
            MULTI_BALANCE_FAR_CONTACT_REJECT_M
        ):
            continue

        score = float(
            geometry[
                "cost"
            ]
        )

        selection_key = (
            0 if bool(geometry.get("valid", False)) else 1,
            score,
        )
        if best is None or selection_key < best["selection_key"]:
            balanced_template = clone_template_with_phases(
                template,
                phases,
                suffix="_BAL",
            )

            best = {
                "solution":
                    {
                        "kind":
                            "multifinger_balance",

                        "q":
                            np.asarray(
                                q,
                                dtype=np.float64,
                            ),

                        "position":
                            p,

                        "rotation":
                            r,
                    },

                "template":
                    balanced_template,

                "geometry":
                    geometry,

                "score":
                    score,

                "selection_key":
                    selection_key,

                "strategy_yaw":
                    float(
                        strategy_yaw
                    ),
            }

        if best is not None and bool(best["geometry"].get("valid", False)):
            print(
                "  [V102 FAST BALANCE ACCEPT]",
                "template=", template["name"],
                "remaining_seed_variants_skipped=",
                max(0, len(seed_variants) - seed_variant_index - 1),
            )
            break

    return best


def refine_q_direct(
    solver,
    arcs,
    mode,
    template,
    seed_q,
    lower,
    upper,
    reference_q,
):
    def objective(q):
        q = np.asarray(
            q,
            dtype=np.float64,
        )

        try:
            p, r = solver.compute_forward_kinematics(
                PALM_FRAME,
                q,
            )
        except Exception:
            return 1e9

        p = np.asarray(
            p,
            dtype=np.float64,
        )

        r = np.asarray(
            r,
            dtype=np.float64,
        )

        g = evaluate_geometry(
            arcs,
            mode,
            p,
            r,
            template["phases"],
            template["active"],
            strict=True,
        )

        margin_penalty = (
            max(
                0.0,
                5.0
                -
                joint_margin_deg(
                    q,
                    lower,
                    upper,
                ),
            )
            / 5.0
        ) ** 2

        return (
            g["cost"]
            +
            contact_posture_penalty(
                q,
                reference_q,
            )
            +
            2.0
            *
            margin_penalty
        )

    result = minimize(
        objective,
        seed_q,
        method="Powell",
        bounds=list(
            zip(
                lower,
                upper,
            )
        ),
        options={
            "maxiter":
                Q_REFINE_MAXITER,

            "xtol":
                1e-4,

            "ftol":
                1e-4,

            "disp":
                False,
        },
    )

    q = np.asarray(
        result.x,
        dtype=np.float64,
    )

    try:
        p, r = solver.compute_forward_kinematics(
            PALM_FRAME,
            q,
        )
    except Exception:
        return None

    return {
        "kind":
            "direct_q_refine",

        "q":
            q,

        "position":
            np.asarray(
                p,
                dtype=np.float64,
            ),

        "rotation":
            np.asarray(
                r,
                dtype=np.float64,
            ),
    }

def build_candidate(
    arcs,
    mode,
    template,
    solution,
    current_q,
    lower,
    upper,
    yaw,
):
    q = np.asarray(
        solution["q"],
        dtype=np.float64,
    )

    candidate_template = template
    g = evaluate_geometry(
        arcs,
        mode,
        solution["position"],
        solution["rotation"],
        template["phases"],
        template["active"],
        strict=True,
    )

    # A measured arc can reach its local minimum shortly before the optimizer's
    # selected endpoint.  When height order is the *only* failed constraint,
    # move only those phase endpoints slightly earlier and then recompute the
    # complete strict geometry.  Never convert a failed candidate by changing a
    # flag or by relaxing the 2 mm limit.
    if str(g.get("reject", "")) == "FINGER_ARC_HEIGHT_ORDER":
        repaired_phases, phase_adjustments, repaired_order = (
            repair_arc_height_order_phases(
                arcs,
                solution["rotation"],
                g["back_normal_world"],
                template["phases"],
                template["active"],
            )
        )
        if phase_adjustments and bool(repaired_order["valid"]):
            repaired_g = evaluate_geometry(
                arcs,
                mode,
                solution["position"],
                solution["rotation"],
                repaired_phases,
                template["active"],
                strict=True,
            )
            if bool(repaired_g.get("valid", False)):
                candidate_template = dict(template)
                candidate_template["phases"] = {
                    finger: float(repaired_phases[finger])
                    for finger in template["active"]
                }
                candidate_template["arc_height_phase_repair"] = {
                    finger: dict(values)
                    for finger, values in phase_adjustments.items()
                }
                g = repaired_g
                print(
                    "  [V102 ARC HEIGHT PHASE REPAIR]",
                    "template=", template["name"],
                    "adjustments=", {
                        finger: {
                            key: round(float(value), 2)
                            for key, value in values.items()
                        }
                        for finger, values in phase_adjustments.items()
                    },
                    "height_limit_mm=", ARC_HEIGHT_ORDER_TOL_M * 1000.0,
                    "strict_recheck=PASS",
                )

    # Copy because we may append posture-level rejection reasons.
    g = dict(
        g
    )

    margin = joint_margin_deg(
        q,
        lower,
        upper,
    )

    requested_position = solution.get("requested_position")
    requested_rotation = solution.get("requested_rotation")
    fk_position_error_m = None
    fk_orientation_error_deg = None

    if requested_position is not None:
        fk_position_error_m = float(
            np.linalg.norm(
                np.asarray(solution["position"], dtype=np.float64)
                - np.asarray(requested_position, dtype=np.float64)
            )
        )

    if requested_rotation is not None:
        fk_orientation_error_deg = rotation_delta_deg(
            requested_rotation,
            solution["rotation"],
        )

    # Geometry used to say PASS even when the candidate was rejected only by
    # joint margin. That made the 4F failure summary point at the wrong layer.
    if margin < MIN_JOINT_MARGIN_DEG:
        old_reject = str(g.get("reject", "PASS"))
        margin_reject = "JOINT_MARGIN"
        g["reject"] = (
            margin_reject
            if old_reject == "PASS"
            else old_reject + "|" + margin_reject
        )
        g["valid"] = False
        g["strong"] = False

    valid = (
        g["valid"]
        and
        margin
        >= MIN_JOINT_MARGIN_DEG
    )

    strong = (
        g["strong"]
        and
        margin
        >= MIN_JOINT_MARGIN_DEG
    )

    delta_deg = np.degrees(
        q
        -
        current_q
    )

    weighted_motion = float(
        np.sqrt(
            np.sum(
                SYNC_JOINT_MOVE_WEIGHTS
                *
                np.square(
                    delta_deg
                )
            )
        )
    )

    score = (
        float(
            g["cost"]
        )
        +
        contact_posture_penalty(
            q,
            current_q,
        )
        +
        0.002
        *
        weighted_motion
    )

    return {
        "mode":
            mode,

        "template":
            candidate_template,

        "strategy_yaw":
            float(
                yaw
            ),

        "q":
            q,

        "palm_position":
            solution["position"],

        "palm_rotation":
            solution["rotation"],

        "desired_palm_position":
            requested_position,

        "desired_palm_rotation":
            requested_rotation,

        "fk_position_error_m":
            fk_position_error_m,

        "fk_orientation_error_deg":
            fk_orientation_error_deg,

        "ik_kind":
            solution["kind"],

        "geometry":
            g,

        "joint_margin":
            margin,

        "valid":
            valid,

        "strong":
            strong,

        "endpoint_valid":
            valid,

        "trajectory_safe":
            False,

        "trajectory_reason":
            None,

        "trajectory":
            None,

        "clearance_deg":
            None,

        "score":
            score,
    }



def hard_target_lock_recheck(candidate, emit=True):
    """
    Explicit post-refinement TARGET_PATCH lock validation.

    Direct-q refinement is free to choose the reachable five-joint posture, but
    it is NEVER allowed to ignore the selected GUI patch, the dynamic
    base-to-root/finger-root axis alignment or the per-finger lanes.
    """
    g = candidate.get("geometry", {})
    active = list(candidate.get("template", {}).get("active", []))
    reasons = []

    # Coverage is retained for diagnostics only. It measures whether the click
    # lies inside the active-contact hull, which is not a valid per-finger gate
    # during sequential acquisition. The four-finger centroid is authoritative.
    coverage = float(g.get("coverage", 999.0))
    center_error = float(g.get("target_center_error", 999.0))
    max_cross = float(g.get("lane_cross_max", 999.0))
    max_patch_radius = float(g.get("patch_radius_max", 999.0))
    palm_angle = float(g.get("palm_back_angle_deg", 90.0))
    root_base_axis_angle = float(g.get("root_base_axis_angle_deg", 90.0))
    arc_height_violation = float(g.get("arc_height_violation_m", 999.0))

    if center_error > TARGET_PATCH_CENTER_HARD_M:
        reasons.append("TARGET_PATCH_CENTER")
    if max_cross > TARGET_LANE_CROSS_HARD_M:
        reasons.append("TARGET_LANE_CROSS")
    if max_patch_radius > TARGET_PATCH_RADIUS_HARD_M:
        reasons.append("TARGET_PATCH_RADIUS")
    if palm_angle > palm_alignment_limit_deg():
        reasons.append("PALM_ALIGNMENT")
    if root_base_axis_angle > ROOT_BASE_AXIS_HARD_DEG:
        reasons.append("ROOT_BASE_AXIS_ALIGNMENT")
    if arc_height_violation > 1e-12:
        reasons.append("FINGER_ARC_HEIGHT_ORDER")

    lane_details = g.get("lane_details", {})
    finger_results = g.get("finger_results", {})
    for finger in active:
        detail = lane_details.get(finger)
        result = finger_results.get(finger)
        if detail is None:
            reasons.append("LANE_DETAIL_MISSING:" + str(finger))
            continue
        if float(detail.get("cross_error", 999.0)) > TARGET_LANE_CROSS_HARD_M:
            reasons.append("LANE_CROSS:" + str(finger))
        if float(detail.get("patch_radius", 999.0)) > TARGET_PATCH_RADIUS_HARD_M:
            reasons.append("PATCH_RADIUS:" + str(finger))
        if result is None or not bool(result.get("valid", False)):
            reasons.append("ACTIVE_ENDPOINT:" + str(finger))

    anchor_finger = candidate.get("mode", {}).get("v103_anchor_finger")
    if anchor_finger is not None:
        anchor_detail = lane_details.get(anchor_finger)
        if (anchor_detail is None
                or float(anchor_detail.get("patch_radius", 999.0)) > V103_MIDDLE_ANCHOR_MAX_M):
            reasons.append("ANCHOR_FINGER_TARGET:" + str(anchor_finger))
    if candidate.get('mode', {}).get('v103_near_base_three_finger'):
        # This three-finger mode uses the ordinary contact corridor. Solve
        # inside that corridor instead of accepting 14 mm here and later
        # rejecting the same endpoint at the 8 mm acquisition gate.
        if float(g.get('lane_cross_max',999.0)) > PLANNED_ACTIVE_LANE_CROSS_M:
            reasons.append('THREE_FINGER_CONTACT_LANE')
        if float(g.get('patch_radius_max',999.0)) > PLANNED_ACTIVE_PATCH_RADIUS_M:
            reasons.append('THREE_FINGER_CONTACT_PATCH')

    ok = len(reasons) == 0

    if emit:
        print(
            "  [V8 TARGET LOCK RECHECK]",
            candidate.get("template", {}).get("name", "UNKNOWN"),
            "patch_center_mm=", round(center_error * 1000.0, 2),
            "coverage_diag_mm=", round(coverage * 1000.0, 2),
            "max_lane_cross_mm=", round(max_cross * 1000.0, 2),
            "max_patch_radius_mm=", round(max_patch_radius * 1000.0, 2),
            "palm_angle_deg=", round(palm_angle, 2),
            "root_base_axis_deg=", round(root_base_axis_angle, 2),
            "arc_height_violation_mm=", round(arc_height_violation * 1000.0, 2),
            "->", "PASS" if ok else "FAIL:" + "|".join(reasons),
        )
        for finger in active:
            detail = lane_details.get(finger, {})
            result = finger_results.get(finger, {})
            print(
                "    [V8 TARGET LANE]",
                finger,
                "cross_mm=", round(float(detail.get("cross_error", 999.0)) * 1000.0, 2),
                "click_distance_diag_mm=", round(float(detail.get("patch_radius", 999.0)) * 1000.0, 2),
                "along_mm=", round(float(detail.get("along_coordinate", 999.0)) * 1000.0, 2),
                "depth_mm=", round(float(result.get("signed", 999.0)) * 1000.0, 2),
            )

    candidate["target_lock_rechecked"] = True
    candidate["target_lock_recheck_ok"] = bool(ok)

    if not ok:
        candidate["valid"] = False
        candidate["strong"] = False
        candidate["endpoint_valid"] = False
        g = dict(g)
        old_reject = str(g.get("reject", "PASS"))
        extra = "TARGET_LOCK_RECHECK:" + "|".join(reasons)
        g["reject"] = extra if old_reject == "PASS" else old_reject + "|" + extra
        g["valid"] = False
        g["strong"] = False
        candidate["geometry"] = g

    return bool(ok), reasons


def mark_path_validation_eligibility(candidate, strict_ok, strict_reasons):
    """Admit a narrowly bounded near-base endpoint to full path validation.

    This is deliberately *not* endpoint acceptance.  It only prevents a single
    slightly shallow finger, or a static four-tip centroid that will move during
    the scratch wave, from deleting the posture before acquisition/contact/base
    and retreat validation can inspect it.  Every other geometric safety gate
    remains hard, and the moving-contact centroid is checked after rollout.
    """
    if strict_ok:
        candidate["path_validation_eligible"] = True
        candidate["path_validation_mode"] = "STRICT_ENDPOINT"
        return True

    g = candidate.get("geometry", {})
    active = list(candidate.get("template", {}).get("active", []))
    lane_details = g.get("lane_details", {})
    finger_results = g.get("finger_results", {})
    allowed_reasons = {"TARGET_PATCH_CENTER"}
    allowed_reasons.update("ACTIVE_ENDPOINT:" + str(f) for f in active)
    reason_set = set(str(r) for r in strict_reasons)
    eligible = (
        len(active) == 4
        and reason_set.issubset(allowed_reasons)
        and float(g.get("target_center_error", 999.0)) <= V103_PATH_REVIEW_CENTER_MAX_M
        and float(g.get("lane_cross_max", 999.0)) <= TARGET_LANE_CROSS_HARD_M
        and float(g.get("patch_radius_max", 999.0)) <= TARGET_PATCH_RADIUS_HARD_M
        and float(g.get("palm_back_angle_deg", 999.0)) <= palm_alignment_limit_deg()
        and float(g.get("root_base_axis_angle_deg", 999.0)) <= ROOT_BASE_AXIS_HARD_DEG
        and float(g.get("arc_height_violation_m", 999.0)) <= 1e-12
    )
    valid_count = 0
    shallow = []
    if eligible:
        for finger in active:
            detail = lane_details.get(finger)
            result = finger_results.get(finger)
            if detail is None or result is None:
                eligible = False
                break
            if (float(detail.get("cross_error", 999.0)) > TARGET_LANE_CROSS_HARD_M
                    or float(detail.get("patch_radius", 999.0)) > TARGET_PATCH_RADIUS_HARD_M
                    or float(result.get("distance", 999.0)) > MAX_ACTIVE_SURFACE_DISTANCE_M):
                eligible = False
                break
            if bool(result.get("valid", False)):
                valid_count += 1
                continue
            signed = float(result.get("signed", 999.0))
            if (str(result.get("reject", "")) != "SIGNED_TOO_SHALLOW"
                    or signed <= SOFT_SIGNED_MAX_M
                    or signed > V103_PATH_REVIEW_SHALLOW_MAX_M):
                eligible = False
                break
            shallow.append((str(finger), signed))
    # Never review a posture where more than one finger has missed the surface.
    eligible = bool(eligible and valid_count >= 3 and len(shallow) <= 1)
    candidate["path_validation_eligible"] = eligible
    candidate["path_validation_mode"] = "PROVISIONAL_MOVING_CONTACT" if eligible else "REJECTED"
    candidate["path_validation_shallow_mm"] = {
        finger: round(signed * 1000.0, 3) for finger, signed in shallow
    }
    return eligible


def candidate_rank(item):
    """
    V4.13 HARD strategy hierarchy.

    Flat patch priority is truly:
        4F -> 3F -> 2F -> 1F

    Finger count is evaluated BEFORE strong/valid quality tiers, so a STRONG
    single-finger endpoint can never jump ahead of a valid 3F/4F endpoint.
    """
    count = len(
        item[
            "template"
        ][
            "active"
        ]
    )

    if item[
        "mode"
    ][
        "kind"
    ] == "flat":
        finger_key = -count

    else:
        finger_key = (
            0
            if count == 2
            else
            1
            if count == 1
            else
            2
        )

    tier = (
        0
        if item[
            "strong"
        ]
        else
        1
        if item[
            "valid"
        ]
        else
        2
    )

    g = item[
        "geometry"
    ]

    j4_abs_deg = abs(
        float(
            np.degrees(
                np.asarray(
                    item[
                        "q"
                    ],
                    dtype=np.float64,
                )
            )[3]
        )
    )

    balance_key = float(
        g.get(
            "contact_depth_std",
            999.0,
        )
    )

    return (
        finger_key,
        tier,

        balance_key,

        g.get(
            "palm_back_angle_deg",
            90.0,
        ),

        g.get(
            "coverage",
            999.0,
        ),

        g.get(
            "normal_rmse",
            999.0,
        ),

        -g.get(
            "arc_tangent",
            0.0,
        ),

        j4_abs_deg,

        -g.get(
            "stroke",
            0.0,
        ),

        item[
            "score"
        ],
    )


def candidate_signature(
    candidate,
):
    """
    Deduplicate repeated direct-q results produced by different seeds.
    """
    q_deg = np.degrees(
        np.asarray(
            candidate["q"],
            dtype=np.float64,
        )
    )

    return (
        candidate["mode"]["name"],
        candidate["template"]["name"],
        tuple(
            np.round(
                q_deg,
                1,
            )
        ),
    )


def deduplicate_candidates(
    candidates,
):
    ordered = sorted(
        candidates,
        key=candidate_rank,
    )

    unique = []
    seen = set()

    for candidate in ordered:
        key = candidate_signature(
            candidate
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            candidate
        )

    return unique


def build_search_stages(
    mode,
    templates,
):
    """
    Search easy / human-like solutions first.

    A normal flat patch should never require a blind 70+ template/yaw sweep.
    Extreme J4 orientations are deliberately delayed until simpler strategies
    have failed.
    """
    if mode["kind"] == "v_side":
        return [
            (
                "V_PAIR",
                [
                    t
                    for t
                    in templates
                    if t["family"] == "v_pair"
                ],
                [-90.0, 90.0, -60.0, 60.0],
            ),
            (
                "V_SINGLE",
                [
                    t
                    for t
                    in templates
                    if t["family"] == "v_single"
                ],
                [-90.0, 90.0, -60.0, 60.0, -30.0, 30.0, 0.0],
            ),
        ]

    return [
        (
            "FLAT_4F",
            [
                t
                for t
                in templates
                if t["family"] == "human_wave"
            ],
            # A single neutral task frame.  In-plane heading is an outcome of
            # the five-joint geometric optimization, not a discrete strategy.
            [0.0],
        ),
        (
            "FLAT_3F",
            [
                t
                for t
                in templates
                if t["family"] == "three"
            ],
            [0.0, -30.0, 30.0, -60.0, 60.0],
        ),
        (
            "FLAT_PAIR",
            [
                t
                for t
                in templates
                if t["family"] == "pair"
            ],
            [0.0, -30.0, 30.0, -60.0, 60.0],
        ),
        (
            "FLAT_SINGLE",
            [
                t
                for t
                in templates
                if t["family"] == "single"
            ],
            [0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0],
        ),
    ]


def print_finger_diagnostic(
    geometry,
    active_fingers,
    prefix="    ",
):
    for finger in active_fingers:
        result = (
            geometry
            .get(
                "finger_results",
                {},
            )
            .get(
                finger,
                {},
            )
        )

        tangent = result.get(
            "tangent"
        )

        tangent_text = (
            "None"
            if tangent is None
            else str(
                round(
                    float(tangent),
                    3,
                )
            )
        )

        print(
            prefix
            + finger
            + ": phase="
            + str(
                round(
                    float(
                        result.get(
                            "phase",
                            0.0,
                        )
                    ),
                    1,
                )
            )
            + " signed="
            + str(
                round(
                    float(
                        result.get(
                            "signed",
                            999.0,
                        )
                    )
                    * 1000.0,
                    2,
                )
            )
            + "mm dist="
            + str(
                round(
                    float(
                        result.get(
                            "distance",
                            999.0,
                        )
                    )
                    * 1000.0,
                    2,
                )
            )
            + "mm arc_tangent="
            + tangent_text
            + " stroke="
            + str(
                round(
                    float(
                        result.get(
                            "stroke",
                            0.0,
                        )
                    )
                    * 1000.0,
                    2,
                )
            )
            + "mm reject="
            + str(
                result.get(
                    "reject",
                    "UNKNOWN",
                )
            )
            + " warning="
            + str(
                result.get(
                    "warning",
                    "NONE",
                )
            )
        )


def endpoint_failure_bucket(candidate):
    """Return the earliest useful diagnostic layer for an invalid endpoint."""
    geometry = candidate.get("geometry", {})
    reject = str(geometry.get("reject", "UNKNOWN"))

    if float(candidate.get("joint_margin", -999.0)) < MIN_JOINT_MARGIN_DEG:
        return "JOINT_MARGIN"
    if "ACTIVE_ENDPOINT" in reject:
        return "ACTIVE_FINGER_GEOMETRY"
    if any(
        token in reject
        for token in (
            "TARGET_PATCH_CENTER",
            "TARGET_PATCH_RADIUS",
            "TARGET_LANE_CROSS",
            "TARGET_LOCK_RECHECK",
        )
    ):
        return "TARGET_PATCH_LANE_FRAME"
    if "PALM_ALIGNMENT" in reject:
        return "PALM_ALIGNMENT"
    if reject == "PASS" and not candidate.get("valid", False):
        return "UNEXPLAINED_INVALID"
    return reject.split("|")[0] if reject else "UNKNOWN"


def endpoint_failure_summary(candidates):
    counts = {}
    for candidate in candidates:
        if candidate.get("valid", False) or candidate.get("path_validation_eligible", False):
            continue
        bucket = endpoint_failure_bucket(candidate)
        counts[bucket] = counts.get(bucket, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def trajectory_failure_bucket(reason):
    """Collapse a verbose trajectory reject into a readable root-cause label."""
    text = str(reason or "UNKNOWN")
    # The validator prefixes rejects with the numeric clearance value.  It is
    # useful in the raw example, but not as the grouping key.
    if text.startswith("clearance=") and ":" in text:
        text = text.split(":", 1)[1]
    if "BUILD_FAIL_clearance=0.0_" in text:
        text = text.split("BUILD_FAIL_clearance=0.0_", 1)[1]
    return text.split("|", 1)[0] or "UNKNOWN"


def trajectory_failure_summary(result):
    """Return counts plus one raw example for every rejected candidate family."""
    counts = {}
    examples = {}
    rejects = list(result.get("rejects", []) or [])
    for item in rejects:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            template, reason = str(item[0]), str(item[1])
        else:
            template, reason = "UNKNOWN", str(item)
        bucket = trajectory_failure_bucket(reason)
        counts[bucket] = int(counts.get(bucket, 0)) + 1
        examples.setdefault(bucket, {
            "template": template,
            "reason": reason,
        })
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total_rejected_candidates": len(rejects),
        "counts": ordered,
        "examples": examples,
    }


def print_candidate_frame_diagnostic(candidate, mode, prefix="  "):
    """Print the complete 4F target-frame -> IK/FK -> lane evidence chain."""
    geometry = candidate.get("geometry", {})
    lane_spec = geometry.get("lane_spec") or {}
    lane_details = geometry.get("lane_details", {})
    finger_results = geometry.get("finger_results", {})

    position_error = candidate.get("fk_position_error_m")
    orientation_error = candidate.get("fk_orientation_error_deg")
    print(
        prefix + "[V100 FIRST FAILURE]",
        endpoint_failure_bucket(candidate),
        "template=", candidate.get("template", {}).get("name", "UNKNOWN"),
        "ik=", candidate.get("ik_kind", "UNKNOWN"),
        "yaw=", round(float(candidate.get("strategy_yaw", 0.0)), 1),
        "joint_margin_deg=", round(float(candidate.get("joint_margin", -999.0)), 3),
        "desired_to_fk_mm=", (
            "N/A" if position_error is None else round(float(position_error) * 1000.0, 3)
        ),
        "desired_to_fk_deg=", (
            "N/A" if orientation_error is None else round(float(orientation_error), 3)
        ),
    )
    print(
        prefix + "[V100 TARGET FRAME]",
        "anchor=", np.round(np.asarray(mode.get("anchor"), dtype=np.float64), 6),
        "N=", np.round(np.asarray(mode.get("normal"), dtype=np.float64), 6),
        "U=", np.round(np.asarray(mode.get("u"), dtype=np.float64), 6),
        "V=", np.round(np.asarray(mode.get("v"), dtype=np.float64), 6),
    )
    print(
        prefix + "[V100 PALM FK]",
        "position=", np.round(np.asarray(candidate.get("palm_position"), dtype=np.float64), 6),
        "palm_normal=", np.round(
            np.asarray(geometry.get("palm_normal_world", [999.0] * 3), dtype=np.float64),
            6,
        ),
        "back_normal=", np.round(
            np.asarray(geometry.get("back_normal_world", [999.0] * 3), dtype=np.float64),
            6,
        ),
        "angle_deg=", round(float(geometry.get("palm_back_angle_deg", 999.0)), 3),
    )
    if lane_spec:
        print(
            prefix + "[V100 LANE FRAME]",
            "band=", np.round(np.asarray(lane_spec.get("band_axis"), dtype=np.float64), 6),
            "scratch=", np.round(np.asarray(lane_spec.get("scratch_axis"), dtype=np.float64), 6),
            "offset_sign=", float(lane_spec.get("band_offset_sign", 1.0)),
            "offsets_mm=", {
                finger: round(float(value) * 1000.0, 3)
                for finger, value in lane_spec.get("band_offsets", {}).items()
            },
        )

    for finger in candidate.get("template", {}).get("active", []):
        detail = lane_details.get(finger, {})
        result = finger_results.get(finger, {})
        print(
            prefix + "  [V100 FINGER]",
            finger,
            "band_mm=", round(float(detail.get("band_coordinate", 999.0)) * 1000.0, 3),
            "lane_mm=", round(float(detail.get("desired_band", 999.0)) * 1000.0, 3),
            "cross_mm=", round(float(detail.get("cross_error", 999.0)) * 1000.0, 3),
            "along_mm=", round(float(detail.get("along_coordinate", 999.0)) * 1000.0, 3),
            "click_distance_diag_mm=", round(float(detail.get("patch_radius", 999.0)) * 1000.0, 3),
            "signed_mm=", round(float(result.get("signed", 999.0)) * 1000.0, 3),
            "distance_mm=", round(float(result.get("distance", 999.0)) * 1000.0, 3),
            "reject=", result.get("reject", "UNKNOWN"),
        )


def plan_fk_first(solver, arcs, mode, current_q, home_q, lower, upper, cached_seeds=(),
                  active_fingers=None, anchor_finger=None, priority_only=False):
    """V103: reachable five-joint search followed by unchanged strict gates."""
    requested = tuple(active_fingers or LONG_FINGERS)
    if (FOUR_FINGER_ONLY and mode.get('v103_contact_coordination')
            and set(requested) != set(LONG_FINGERS)):
        return [], {"error": "FOUR_FINGER_ONLY_LIVE_POLICY", "active": list(requested)}
    templates = [t for t in generate_templates(mode)
                 if len(t['active']) == len(requested)
                 and set(t['active']) == set(requested)]
    if not templates:
        return [], {"error": "NO_REQUESTED_FINGER_TEMPLATE", "active": list(requested)}
    template = templates[0]
    sampler = JOINT_SEARCH.ArcMinima(arcs, template['active'], MIN_MAIN_PHASE_DEG, MAX_MAIN_PHASE_DEG)

    def evaluate(q, phases, p, r):
        # Coarse projection uses the clicked patch normal. Recompute the local
        # normal and each finger minimum at this actual FK pose before acceptance.
        g = evaluate_geometry(arcs, mode, p, r, phases, template['active'], strict=True)
        phases, _ = sampler.at(r, g.get('back_normal_world', mode['normal']))
        dynamic_template = clone_template_with_phases(template, phases, suffix='_FK_MIN')
        solution = dict(kind='fk_first_dynamic_minimum', q=q.copy(), position=p, rotation=r)
        candidate = build_candidate(arcs, mode, dynamic_template, solution,
                                    current_q, lower, upper, 0.0)
        strict_ok, strict_reasons = hard_target_lock_recheck(candidate, emit=False)
        mark_path_validation_eligibility(candidate, strict_ok, strict_reasons)
        # Ranking and hard gates alone do not steer an optimizer away from a
        # cheap-but-invalid point just outside the angle/patch boundary.
        geometry = candidate['geometry']
        violations = [
            max(0., geometry['palm_back_angle_deg']-palm_alignment_limit_deg()) / 5.,
            max(0., geometry['root_base_axis_angle_deg']-ROOT_BASE_AXIS_HARD_DEG) / 5.,
            max(0., geometry['target_center_error']-TARGET_PATCH_CENTER_HARD_M) / .005,
            max(0., geometry['lane_cross_max']-TARGET_LANE_CROSS_HARD_M) / .003,
            max(0., geometry['patch_radius_max']-TARGET_PATCH_RADIUS_HARD_M) / .005,
        ]
        if anchor_finger is not None:
            anchor_radius = float(geometry.get('lane_details', {}).get(
                anchor_finger, {}).get('patch_radius', 999.0))
            violations.append(max(0., anchor_radius-V103_MIDDLE_ANCHOR_MAX_M) / .004)
            if (not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))
                    and mode.get('v103_near_base_three_finger')
                    and set(requested) == {'ring', 'middle', 'index'}):
                violations.append(max(0., geometry['lane_cross_max']-PLANNED_ACTIVE_LANE_CROSS_M) / .003)
                violations.append(max(0., geometry['patch_radius_max']-PLANNED_ACTIVE_PATCH_RADIUS_M) / .005)
                # Finger-tip endpoint validity says nothing about the rest of
                # the hand fitting beside base. Use the same real collision geometry
                # here to steer search before expensive acquisition is attempted.
                guard = getattr(HAND_COLLISION_MODEL, 'base_guard', None)
                if guard is not None:
                    parking = []
                    for pinky_progress in (0.,30.,60.,90.):
                        commands = empty_fingers()
                        commands.update(phases)
                        commands['pinky'] = pinky_progress
                        gap, link = guard.clearance(q, commands)
                        if math.isfinite(float(gap)):
                            parking.append((float(gap),pinky_progress,link))
                        # Changing pinky cannot improve a clearance bottleneck on
                        # index/middle/ring/palm. The open option is already the
                        # best achievable bound, so avoid three redundant full-hand
                        # geometry queries in every optimizer evaluation.
                        if pinky_progress == 0. and parking and not str(link).startswith('right_pinky'):
                            break
                    if not parking:
                        raise RuntimeError('THREE_FINGER_BASE_CLEARANCE_UNAVAILABLE')
                    parking.sort(key=lambda row: (-row[0], row[1]))
                    gap, preferred_pinky, link = parking[0]
                    ranked_parking = [row[1] for row in parking]
                    candidate['inactive_parking_options'] = ranked_parking + [
                        value for value in (0.,30.,60.,90.) if value not in ranked_parking]
                    candidate['endpoint_base_clearance_mm'] = gap*1000.
                    candidate['endpoint_base_closest_link'] = link
                    penalty = 200. * (max(0., V103_BASE_MIN_CLEARANCE_M+.002-gap)/.002)**2
                    # The search retains and ranks by geometry.cost as well as
                    # minimizing search_cost; both must see the base penalty.
                    geometry['cost'] = float(geometry['cost'] + penalty)
                    if gap < V103_BASE_MIN_CLEARANCE_M:
                        candidate.update(valid=False,strong=False,endpoint_valid=False,
                                         path_validation_eligible=False,path_validation_mode='REJECTED')
                        geometry.update(valid=False,strong=False)
                        geometry['reject'] = str(geometry.get('reject','PASS')) + '|THREE_FINGER_ENDPOINT_BASE_MARGIN'
        candidate['search_cost'] = float(geometry['cost'] + 200. * np.dot(violations, violations))
        return candidate

    # Stay inside mechanical limits by the existing safety margin. No joint
    # limit is enlarged; no searched joint is fixed to a heading or wrist angle.
    margin = math.radians(MIN_JOINT_MARGIN_DEG + 0.1)
    seeds = list(cached_seeds) + build_continuous_refine_seeds(current_q, home_q, lower, upper)
    # Isaac read-only diagnostic verified this endpoint, not physical execution.
    # Narrow target neighborhood only; current geometry and base gates recheck it.
    priority_seeds = list(cached_seeds)[:4]
    if (set(requested) == {'ring', 'middle', 'index'} and anchor_finger == 'middle'
            and np.linalg.norm(np.asarray(mode['anchor']) -
                               np.array([.0220236387103796, -.056773556396365166,
                                         .15222825855016708])) <= .005):
        priority_seeds.append(np.radians([-15.826, 30.991, -143.225, -1.273, -56.922]))
    return JOINT_SEARCH.search(
        fk=lambda q: solver.compute_forward_kinematics(PALM_FRAME, q),
        arcs=arcs, fingers=template['active'], minimum=MIN_MAIN_PHASE_DEG,
        maximum=MAX_MAIN_PHASE_DEG, lower=np.asarray(lower)+margin,
        upper=np.asarray(upper)-margin, anchor=mode['anchor'], normal=mode['normal'],
        palm_normal=get_palm_local_normal(arcs),
        root_angle=lambda p, r: root_base_axis_alignment(arcs, p, r)['angle_deg'],
        evaluate=evaluate, seeds=seeds, seconds=120., palm_limit_deg=palm_alignment_limit_deg(),
        anchor_finger=anchor_finger, priority_seeds=priority_seeds, priority_only=priority_only,
    )


def plan_mode(
    solver,
    arcs,
    mode,
    current_q,
    home_q,
    lower,
    upper,
    stage_filter=None,
):
    templates = generate_templates(
        mode
    )

    stages = build_search_stages(
        mode,
        templates,
    )

    if FOUR_FINGER_ONLY:
        if mode.get("kind") != "flat":
            return []
        stages = [stage for stage in stages if stage[0] == "FLAT_4F"]

    if stage_filter is not None:
        allowed_stages = set(stage_filter)
        stages = [
            stage
            for stage in stages
            if stage[0] in allowed_stages
        ]

    all_candidates = []

    for (
        stage_name,
        stage_templates,
        stage_yaws,
    ) in stages:

        print("")
        print(
            "----------------------------------------------"
        )
        print(
            "[V4.12-ARC SEARCH STAGE]",
            stage_name,
        )
        print(
            "templates=",
            len(stage_templates),
            "neutral_task_headings=",
            stage_yaws,
        )

        stage_candidates = []

        for template_index, template in enumerate(stage_templates):
            template_valid = []
            template_balance_seeds = []
            fallback_valid_q_seen = set()

            for yaw in stage_yaws:
                (
                    desired_p,
                    desired_r,
                    template_geometry,
                ) = optimize_palm(
                    arcs,
                    template,
                    mode,
                    yaw,
                )

                print("")
                print(
                    "[V4.12-ARC PLAN]",
                    mode["name"],
                    template["name"],
                    "neutral_frame_seed",
                )

                print(
                    "  template geometry:",
                    template_geometry["reject"],
                    "coverage=",
                    round(
                        template_geometry.get(
                            "coverage",
                            999.0,
                        )
                        * 1000.0,
                        2,
                    ),
                    "mm",
                    "normal=",
                    round(
                        template_geometry.get(
                            "normal_rmse",
                            999.0,
                        )
                        * 1000.0,
                        2,
                    ),
                    "mm",
                )

                if not template_geometry.get(
                    "valid",
                    False,
                ):
                    print_finger_diagnostic(
                        template_geometry,
                        template["active"],
                        prefix="    template ",
                    )

                print(
                    "  desired palm:",
                    np.round(
                        desired_p,
                        4,
                    ),
                )

                seeds = build_q_seeds(
                    current_q,
                    home_q,
                    lower,
                    upper,
                    yaw,
                )

                # Staged search: do not throw hundreds of seeds at each pose.
                ik_diagnostics = {}
                solutions = solve_lula_ik(
                    solver,
                    desired_p,
                    desired_r,
                    seeds[:FAST_LULA_SEED_LIMIT],
                    lower,
                    upper,
                    diagnostics=ik_diagnostics,
                )

                print(
                    "  Lula IK solutions:",
                    len(solutions),
                )
                print(
                    "  [V100 IK GATE]",
                    {
                        key: value
                        for key, value in ik_diagnostics.items()
                        if key != "last_exception"
                    },
                )
                if not solutions and ik_diagnostics.get("last_exception"):
                    print(
                        "  [V100 IK LAST EXCEPTION]",
                        ik_diagnostics["last_exception"],
                    )

                local_candidates = []

                for solution in solutions:
                    solution = dict(solution)
                    solution["requested_position"] = np.asarray(
                        desired_p,
                        dtype=np.float64,
                    )
                    solution["requested_rotation"] = np.asarray(
                        desired_r,
                        dtype=np.float64,
                    )
                    candidate = build_candidate(
                        arcs,
                        mode,
                        template,
                        solution,
                        current_q,
                        lower,
                        upper,
                        yaw,
                    )

                    local_candidates.append(
                        candidate
                    )

                    stage_candidates.append(
                        candidate
                    )

                    all_candidates.append(
                        candidate
                    )

                    if candidate["strong"]:
                        print(
                            "  -> STRONG ENDPOINT VALID"
                        )

                local_candidates.sort(
                    key=candidate_rank
                )

                if local_candidates:
                    template_balance_seeds.extend(
                        local_candidates[
                            :2
                        ]
                    )

                valid_here = [
                    c
                    for c
                    in local_candidates
                    if c["valid"]
                ]

                if valid_here:
                    template_valid.extend(
                        valid_here
                    )

                    # Continue only the remaining yaws of THIS template,
                    # then stop the whole stage and choose the best one.
                    continue

                # ------------------------------------------------
                # Authoritative five-joint solve.  The pose IK above is only a
                # warm start: all J1-J5 remain free and the objective itself
                # decides palm position, plane alignment and four-finger fit.
                # ------------------------------------------------

                needs_fallback = (
                    ENABLE_Q_REFINEMENT_FALLBACK

                    and

                    template_geometry.get(
                        "coverage",
                        999.0,
                    )
                    <= 0.050
                )

                if needs_fallback:
                    refine_seeds = list(mode.get("palm_refine_seed_qs", []))

                    if local_candidates:
                        refine_seeds.append(
                            local_candidates[0]["q"]
                        )

                    refine_seeds.extend(
                        build_continuous_refine_seeds(
                            current_q,
                            home_q,
                            lower,
                            upper,
                        )
                    )
                    refine_seeds = refine_seeds[:FAST_DIRECT_REFINE_SEED_LIMIT]

                    seen = set()

                    for seed in refine_seeds:
                        key = tuple(
                            np.round(
                                np.degrees(seed),
                                1,
                            )
                        )

                        if key in seen:
                            continue

                        seen.add(
                            key
                        )

                        refined = refine_q_direct(
                            solver,
                            arcs,
                            mode,
                            template,
                            seed,
                            lower,
                            upper,
                            current_q,
                        )

                        if refined is None:
                            continue

                        refined = dict(refined)
                        refined["requested_position"] = np.asarray(
                            desired_p,
                            dtype=np.float64,
                        )
                        refined["requested_rotation"] = np.asarray(
                            desired_r,
                            dtype=np.float64,
                        )

                        candidate = build_candidate(
                            arcs,
                            mode,
                            template,
                            refined,
                            current_q,
                            lower,
                            upper,
                            yaw,
                        )

                        # V8: direct-q fallback/refine may not inherit VALID
                        # implicitly. Re-run and print the complete TARGET_PATCH
                        # + per-finger lane hard lock before the candidate can
                        # enter the endpoint-valid pool.
                        hard_target_lock_recheck(candidate, emit=True)

                        q_key = tuple(
                            np.round(np.degrees(candidate["q"]), 2)
                        )
                        duplicate_valid_q = bool(
                            candidate["valid"] and q_key in fallback_valid_q_seen
                        )

                        if duplicate_valid_q:
                            print(
                                "  [V8 DIRECT-Q DUPLICATE] target lock revalidated PASS; "
                                "same q already accepted for this template, skip duplicate yaw seed",
                                np.round(np.degrees(candidate["q"]), 2),
                            )
                            break

                        stage_candidates.append(
                            candidate
                        )

                        all_candidates.append(
                            candidate
                        )

                        template_balance_seeds.append(
                            candidate
                        )

                        if candidate["valid"]:
                            fallback_valid_q_seen.add(q_key)
                            template_valid.append(
                                candidate
                            )

                            print(
                                "  direct-q fallback -> VALID (TARGET_LOCK_RECHECK=PASS)",
                                np.round(
                                    np.degrees(
                                        candidate["q"]
                                    ),
                                    2,
                                ),
                            )

                            if candidate["strong"]:
                                print(
                                    "  -> STRONG ENDPOINT VALID"
                                )

                            break

            # ------------------------------------------------
            # V4.11: MULTI-FINGER RESCUE BEFORE DEGRADING.
            #
            # For 4F and 3F templates, actively level the contact depths using
            # J4/J5 + small J2/J3 motion + per-finger phase adjustment.
            # Do this even when the first static endpoint test was invalid.
            # ------------------------------------------------

            active_count = len(
                template[
                    "active"
                ]
            )

            if (
                mode[
                    "kind"
                ]
                ==
                "flat"
                and
                active_count
                >=
                2
            ):
                if template_balance_seeds:
                    template_balance_seeds = sorted(
                        template_balance_seeds,
                        key=candidate_rank,
                    )

                    seed_candidates = template_balance_seeds[
                        :2
                    ]

                else:
                    seed_candidates = []

                if not seed_candidates:
                    seed_candidates = [
                        {
                            "q":
                                current_q,

                            "strategy_yaw":
                                0.0,
                        }
                    ]

                seen_balance_seed = set()

                for seed_candidate in seed_candidates:
                    seed_q = np.asarray(
                        seed_candidate[
                            "q"
                        ],
                        dtype=np.float64,
                    )

                    seed_key = tuple(
                        np.round(
                            np.degrees(
                                seed_q
                            ),
                            1,
                        )
                    )

                    if seed_key in seen_balance_seed:
                        continue

                    seen_balance_seed.add(
                        seed_key
                    )

                    balance_result = refine_multifinger_balance(
                        solver,
                        arcs,
                        mode,
                        template,
                        seed_q,
                        lower,
                        upper,
                        current_q,
                        float(
                            seed_candidate.get(
                                "strategy_yaw",
                                0.0,
                            )
                        ),
                    )

                    if balance_result is None:
                        continue

                    balanced_candidate = build_candidate(
                        arcs,
                        mode,
                        balance_result[
                            "template"
                        ],
                        balance_result[
                            "solution"
                        ],
                        current_q,
                        lower,
                        upper,
                        balance_result[
                            "strategy_yaw"
                        ],
                    )

                    stage_candidates.append(
                        balanced_candidate
                    )

                    all_candidates.append(
                        balanced_candidate
                    )

                    bg = balanced_candidate[
                        "geometry"
                    ]

                    print(
                        "  [V4.12-ARC MULTI BALANCE]",
                        balance_result[
                            "template"
                        ][
                            "balanced_from"
                        ],
                        "->",
                        balance_result[
                            "template"
                        ][
                            "name"
                        ],
                        "valid=",
                        balanced_candidate[
                            "valid"
                        ],
                        "q=",
                        np.round(
                            np.degrees(
                                balanced_candidate[
                                    "q"
                                ]
                            ),
                            2,
                        ),
                    )

                    print(
                        "    phases=",
                        {
                            finger:
                                round(
                                    float(
                                        balance_result[
                                            "template"
                                        ][
                                            "phases"
                                        ][
                                            finger
                                        ]
                                    ),
                                    1,
                                )
                            for finger
                            in balance_result[
                                "template"
                            ][
                                "active"
                            ]
                        },
                        "mean_depth=",
                        round(
                            float(
                                bg.get(
                                    "contact_mean_signed",
                                    999.0,
                                )
                            )
                            *
                            1000.0,
                            2,
                        ),
                        "mm",
                        "std=",
                        round(
                            float(
                                bg.get(
                                    "contact_depth_std",
                                    999.0,
                                )
                            )
                            *
                            1000.0,
                            2,
                        ),
                        "mm",
                        "range=",
                        round(
                            float(
                                bg.get(
                                    "contact_depth_range",
                                    999.0,
                                )
                            )
                            *
                            1000.0,
                            2,
                        ),
                        "mm",
                        "palm_angle=",
                        round(
                            float(
                                bg.get(
                                    "palm_back_angle_deg",
                                    90.0,
                                )
                            ),
                            2,
                        ),
                        "deg",
                        "coverage=",
                        round(
                            float(
                                bg.get(
                                    "coverage",
                                    999.0,
                                )
                            )
                            *
                            1000.0,
                            2,
                        ),
                        "mm",
                        "arc_tangent=",
                        round(
                            float(
                                bg.get(
                                    "arc_tangent",
                                    0.0,
                                )
                            ),
                            3,
                        ),
                        "reject=",
                        bg.get(
                            "reject",
                            "UNKNOWN",
                        ),
                    )

                    print_finger_diagnostic(
                        bg,
                        balance_result[
                            "template"
                        ][
                            "active"
                        ],
                        prefix="      ",
                    )

            # Endpoint candidates are collected across ALL strategy tiers.
            # Trajectory selection later will exhaust 4F before 3F, 3F before
            # pair, and pair before single.
            valid_stage_count = sum(
                1 for candidate in stage_candidates if candidate["valid"]
            )
            if valid_stage_count >= FAST_ENDPOINT_VALID_TARGET:
                print(
                    "  [V102 FAST ENDPOINT STOP]",
                    "valid_endpoints=", valid_stage_count,
                    "remaining_templates_skipped=",
                    max(0, len(stage_templates) - template_index - 1),
                )
                break
        stage_candidates.sort(
            key=candidate_rank
        )

        print(
            "[V4.12-ARC] Stage",
            stage_name,
            "complete. candidates=",
            len(stage_candidates),
            "valid=",
            len(
                [
                    c
                    for c
                    in stage_candidates
                    if c["valid"]
                ]
            ),
        )

    return deduplicate_candidates(
        all_candidates
    )


# ============================================================
# TRAJECTORY
# ============================================================

def arm_move_duration(q0, q1, minimum=MIN_ARM_MOVE_S):
    max_delta = float(np.max(np.abs(np.degrees(np.asarray(q1) - np.asarray(q0)))))
    return clamp(max_delta / ARM_MAX_SPEED_DEG_S, minimum, MAX_ARM_MOVE_S)


def add_keyframe(
    trajectory,
    stage,
    duration,
    q,
    fingers,
    active,
    authorized_contact_fingers=None,
):
    if authorized_contact_fingers is None:
        authorized_contact_fingers = active

    trajectory.append(
        {
            "stage": str(stage),
            "duration": float(duration),
            "q": np.asarray(q, dtype=np.float64).copy(),
            "fingers": dict(fingers),
            "active": list(active),
            "authorized_contact_fingers": list(authorized_contact_fingers),
        }
    )


def solve_offset_pose(
    solver,
    position,
    rotation,
    normal,
    offset,
    warm_q,
    home_q,
    lower,
    upper,
):
    target = np.asarray(position) + float(offset) * np.asarray(normal)

    seeds = [warm_q, home_q] + build_q_seeds(
        warm_q,
        home_q,
        lower,
        upper,
        0.0,
    )[:55]

    solutions = solve_lula_ik(
        solver,
        target,
        rotation,
        seeds,
        lower,
        upper,
    )

    if not solutions:
        return None

    solutions.sort(key=lambda x: float(np.linalg.norm(x["q"] - warm_q)))
    return solutions[0]["q"]


def dynamic_keyframe_geometry(
    solver,
    arcs,
    mode,
    q,
    finger_commands,
    required_active,
):
    """
    Geometry at one REAL synchronized keyframe.

    V4.7 does NOT reject a keyframe because of the raw finger-only tangent.
    The actual scratch-direction test is performed on combined world-space
    fingertip motion between consecutive synchronized keyframes.
    """
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    palm_p, palm_r = solver.compute_forward_kinematics(
        PALM_FRAME,
        q,
    )

    palm_p = np.asarray(
        palm_p,
        dtype=np.float64,
    )

    palm_r = np.asarray(
        palm_r,
        dtype=np.float64,
    )

    required_active = set(
        required_active
    )

    contacts = []
    normal_errors = []
    arc_tangent_scores = []
    active_surface_normals = []

    active_valid = True
    active_penalty = 0.0
    inactive_penetration = 0.0
    inactive_near_penalty = 0.0

    finger_results = {}

    for finger in LONG_FINGERS:
        angle = float(
            finger_commands.get(
                finger,
                0.0,
            )
        )

        local_point, local_tangent = interpolate_arc(
            arcs,
            finger,
            angle,
        )

        world_point = (
            palm_p
            +
            palm_r
            @ local_point
        )

        world_tangent = normalize(
            palm_r
            @ local_tangent
        )

        if finger in required_active:
            (
                distance,
                _,
                surface_normal,
                signed,
            ) = nearest_local(
                mode,
                world_point,
            )

            arc_tangent_score = clamp(
                1.0
                -
                abs(
                    float(
                        np.dot(
                            world_tangent,
                            surface_normal,
                        )
                    )
                ),
                0.0,
                1.0,
            )

            signed_error = (
                signed
                -
                IDEAL_SIGNED_M
            )

            contacts.append(
                world_point
            )

            normal_errors.append(
                signed_error
            )

            arc_tangent_scores.append(
                arc_tangent_score
            )

            active_surface_normals.append(
                surface_normal
            )

            reject = []
            warning = []

            if signed < HARD_PENETRATION_M:
                reject.append(
                    "OVER_PENETRATION"
                )

            elif signed < GOOD_SIGNED_MIN_M:
                warning.append(
                    "SOFT_DEEP"
                )

            if signed > SOFT_SIGNED_MAX_M:
                reject.append(
                    "SIGNED_TOO_SHALLOW"
                )

            elif signed > GOOD_SIGNED_MAX_M:
                warning.append(
                    "SOFT_SHALLOW"
                )

            if distance > MAX_ACTIVE_SURFACE_DISTANCE_M:
                reject.append(
                    "SURFACE_DISTANCE"
                )

            finger_valid = (
                len(
                    reject
                )
                == 0
            )

            active_valid = (
                active_valid
                and
                finger_valid
            )

            active_penalty += (
                10.0
                *
                (
                    signed_error
                    /
                    0.003
                ) ** 2
            )

            # 4-8 mm is a recoverable DEEP_ADJUST zone. Keep it expensive
            # enough to drive J4/J5 back toward ~2 mm, but do not make a
            # recoverable multi-finger state numerically equivalent to failure.
            if signed < GOOD_SIGNED_MIN_M:
                active_penalty += deep_adjust_soft_penalty(signed)

            elif signed > GOOD_SIGNED_MAX_M:
                active_penalty += (
                    22.0
                    *
                    (
                        (
                            signed
                            -
                            GOOD_SIGNED_MAX_M
                        )
                        /
                        0.001
                    ) ** 2
                )

            active_penalty += (
                120.0
                *
                (
                    max(
                        0.0,
                        HARD_PENETRATION_M
                        -
                        signed,
                    )
                    /
                    0.001
                ) ** 2
            )

            active_penalty += (
                16.0
                *
                (
                    max(
                        0.0,
                        distance
                        -
                        MAX_ACTIVE_SURFACE_DISTANCE_M,
                    )
                    /
                    0.004
                ) ** 2
            )

            finger_results[
                finger
            ] = {
                "required":
                    True,

                "angle":
                    angle,

                "signed":
                    signed,

                "distance":
                    distance,

                # Diagnostic only.
                "tangent":
                    arc_tangent_score,

                "arc_tangent":
                    arc_tangent_score,

                "valid":
                    finger_valid,

                "reject":
                    (
                        "PASS"
                        if finger_valid
                        else "|".join(
                            reject
                        )
                    ),

                "warning":
                    (
                        "NONE"
                        if not warning
                        else "|".join(
                            warning
                        )
                    ),
            }

        else:
            (
                distance,
                _,
                _,
                signed,
            ) = nearest_global(
                mode,
                world_point,
            )

            penetration = max(
                0.0,
                INACTIVE_PENETRATION_M
                -
                signed,
            )

            inactive_penetration += (
                penetration
            )

            inactive_near_penalty += (
                max(
                    0.0,
                    0.003
                    -
                    signed,
                )
                /
                0.003
            ) ** 2

            finger_results[
                finger
            ] = {
                "required":
                    False,

                "angle":
                    angle,

                "signed":
                    signed,

                "distance":
                    distance,

                "tangent":
                    None,

                "arc_tangent":
                    None,

                "valid":
                    penetration
                    <= 1e-9,

                "reject":
                    (
                        "PASS"
                        if penetration
                        <= 1e-9
                        else "INACTIVE_COLLISION"
                    ),

                "warning":
                    "NONE",
            }

    coverage = (
        coverage_error(
            mode,
            contacts,
        )
        if contacts
        else 0.0
    )

    normal_rmse = (
        math.sqrt(
            float(
                np.mean(
                    np.square(
                        normal_errors
                    )
                )
            )
        )
        if normal_errors
        else 0.0
    )

    arc_tangent_mean = (
        float(
            np.mean(
                arc_tangent_scores
            )
        )
        if arc_tangent_scores
        else 1.0
    )

    local_back_normal = mean_contact_normal(
        active_surface_normals,
        mode[
            "normal"
        ],
    )

    palm_alignment = palm_back_alignment(
        arcs,
        palm_r,
        local_back_normal,
    )

    palm_back_angle_deg = float(
        palm_alignment[
            "angle_deg"
        ]
    )

    depth_balance = contact_depth_balance_from_results(
        finger_results,
        required_active,
    )

    multi_balance_cost = 0.0

    if depth_balance[
        "count"
    ] >= 2:
        multi_balance_cost = (
            MULTI_MEAN_DEPTH_WEIGHT
            *
            (
                depth_balance[
                    "mean_error"
                ]
                /
                0.0020
            ) ** 2

            +

            MULTI_DEPTH_SPREAD_WEIGHT
            *
            (
                depth_balance[
                    "std_signed"
                ]
                /
                0.0015
            ) ** 2

            +

            MULTI_DEPTH_RANGE_WEIGHT
            *
            (
                depth_balance[
                    "range_signed"
                ]
                /
                0.0030
            ) ** 2
        )

    # A sequential keyframe can contain only pinky (then pinky+ring, etc.). Its
    # contact hull is naturally offset from the GUI click and must not be used
    # as a stop condition. Lane membership plus surface/depth/collision safety
    # are the correct per-keyframe constraints; final four-finger centering is
    # enforced by evaluate_geometry()/hard_target_lock_recheck().
    valid = active_valid and inactive_penetration <= 1e-9

    cost = (
        active_penalty

        +

        50.0
        *
        (
            inactive_penetration
            /
            0.001
        ) ** 2

        +

        1.5
        *
        inactive_near_penalty

        +

        PALM_ALIGNMENT_WEIGHT
        *
        (
            palm_back_angle_deg
            /
            palm_alignment_scale_deg()
        ) ** 2

        +

        multi_balance_cost
    )

    return {
        "valid":
            valid,

        "cost":
            float(
                cost
            ),

        "coverage":
            coverage,

        "normal_rmse":
            normal_rmse,

        "tangent":
            arc_tangent_mean,

        "arc_tangent":
            arc_tangent_mean,

        "inactive_penetration":
            inactive_penetration,

        "palm_back_angle_deg":
            palm_back_angle_deg,

        "palm_parallel_score":
            float(
                palm_alignment[
                    "parallel_score"
                ]
            ),

        "palm_normal_world":
            palm_alignment[
                "palm_normal"
            ],

        "back_normal_world":
            local_back_normal,

        "contact_mean_signed":
            depth_balance[
                "mean_signed"
            ],

        "contact_mean_error":
            depth_balance[
                "mean_error"
            ],

        "contact_depth_std":
            depth_balance[
                "std_signed"
            ],

        "contact_depth_range":
            depth_balance[
                "range_signed"
            ],

        "multi_balance_cost":
            float(
                multi_balance_cost
            ),

        "finger_results":
            finger_results,

        "palm_position":
            palm_p,

        "palm_rotation":
            palm_r,
    }


def world_tip_position(
    solver,
    arcs,
    q,
    finger_commands,
    finger,
):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    palm_p, palm_r = solver.compute_forward_kinematics(
        PALM_FRAME,
        q,
    )

    palm_p = np.asarray(
        palm_p,
        dtype=np.float64,
    )

    palm_r = np.asarray(
        palm_r,
        dtype=np.float64,
    )

    angle = float(
        finger_commands.get(
            finger,
            0.0,
        )
    )

    local_point, _ = interpolate_arc(
        arcs,
        finger,
        angle,
    )

    return (
        palm_p
        +
        palm_r
        @ local_point
    )


def combined_motion_metrics(
    solver,
    arcs,
    mode,
    q_prev,
    commands_prev,
    q_next,
    commands_next,
    fingers,
):
    """
    Measure actual world-space fingertip motion after finger flex and arm/wrist
    motion are combined.

    tangent_ratio = tangential displacement / total displacement.
    """
    metrics = {}

    for finger in fingers:
        p0 = world_tip_position(
            solver,
            arcs,
            q_prev,
            commands_prev,
            finger,
        )

        p1 = world_tip_position(
            solver,
            arcs,
            q_next,
            commands_next,
            finger,
        )

        delta = (
            p1
            -
            p0
        )

        total = float(
            np.linalg.norm(
                delta
            )
        )

        (
            _,
            _,
            normal,
            _,
        ) = nearest_local(
            mode,
            p1,
        )

        normal_step = float(
            np.dot(
                delta,
                normal,
            )
        )

        tangent_delta = (
            delta
            -
            normal_step
            *
            normal
        )

        tangential = float(
            np.linalg.norm(
                tangent_delta
            )
        )

        tangent_ratio = (
            tangential
            /
            max(
                total,
                1e-9,
            )
        )

        metrics[
            finger
        ] = {
            "total":
                total,

            "tangential":
                tangential,

            "normal":
                normal_step,

            "tangent_ratio":
                tangent_ratio,

            "p0":
                p0,

            "p1":
                p1,
        }

    return metrics

def solve_dynamic_contact_q(
    solver,
    arcs,
    mode,
    finger_commands,
    required_active,
    warm_q,
    lower,
    upper,
    endpoint_q=None,
    previous_finger_commands=None,
    previous_active=None,
    moving_finger=None,
    lane_spec=None,
    failure_out=None,
):
    """
    Solve one synchronized arm/wrist keyframe.

    V4.4:
    The motion-quality term is based on the REAL combined world-space
    fingertip displacement:
        finger flex + J1..J5.

    J4/J5 remain the cheapest local adjustment joints.
    """
    warm_q = np.asarray(
        warm_q,
        dtype=np.float64,
    )

    if previous_finger_commands is None:
        previous_finger_commands = dict(
            finger_commands
        )

    if previous_active is None:
        previous_active = []

    previous_active = set(
        previous_active
    )

    def active_lane_state(q):
        """Keep every already-contacting fingertip inside its assigned lane."""
        if lane_spec is None or not required_active:
            return 0.0, True, {}

        cost = 0.0
        valid = True
        details = {}
        for active_finger in required_active:
            tip = world_tip_position(
                solver,
                arcs,
                q,
                finger_commands,
                active_finger,
            )
            state = lane_relation_for_world_point(
                tip,
                lane_spec,
                active_finger,
            )
            details[active_finger] = state
            valid = valid and bool(
                state["cross_error"] <= PLANNED_ACTIVE_LANE_CROSS_M
                and state["patch_radius"] <= PLANNED_ACTIVE_PATCH_RADIUS_M
            )
            cost += (
                TARGET_LANE_LOCK_WEIGHT
                * (state["cross_error"] / max(PLANNED_ACTIVE_LANE_CROSS_M, 1e-6)) ** 2
            )
            cost += (
                0.25
                * TARGET_PATCH_LOCK_WEIGHT
                * (state["patch_radius"] / max(PLANNED_ACTIVE_PATCH_RADIUS_M, 1e-6)) ** 2
            )
        return float(cost), bool(valid), details

    lower_local = np.maximum(
        lower,
        warm_q
        -
        np.radians(
            SYNC_LOCAL_DELTA_DEG
        ),
    )

    upper_local = np.minimum(
        upper,
        warm_q
        +
        np.radians(
            SYNC_LOCAL_DELTA_DEG
        ),
    )

    def objective(q):
        q = np.asarray(
            q,
            dtype=np.float64,
        )

        geometry = dynamic_keyframe_geometry(
            solver,
            arcs,
            mode,
            q,
            finger_commands,
            required_active,
        )

        delta_deg = np.degrees(
            q
            -
            warm_q
        )

        motion_cost = (
            0.010
            *
            float(
                np.sum(
                    SYNC_JOINT_MOVE_WEIGHTS
                    *
                    np.square(
                        delta_deg
                    )
                )
            )
        )

        endpoint_cost = 0.0

        if endpoint_q is not None:
            endpoint_delta_deg = np.degrees(
                q
                -
                np.asarray(
                    endpoint_q,
                    dtype=np.float64,
                )
            )

            endpoint_cost = (
                0.0012
                *
                float(
                    np.sum(
                        np.square(
                            endpoint_delta_deg
                        )
                    )
                )
            )

        combined_cost = 0.0
        lane_cost, _, _ = active_lane_state(q)

        check_fingers = list(
            previous_active
        )

        if (
            moving_finger is not None
            and
            moving_finger in previous_active
            and
            moving_finger not in check_fingers
        ):
            check_fingers.append(
                moving_finger
            )

        if check_fingers:
            metrics = combined_motion_metrics(
                solver,
                arcs,
                mode,
                warm_q,
                previous_finger_commands,
                q,
                finger_commands,
                check_fingers,
            )

            for finger, metric in metrics.items():
                normal_abs = abs(
                    metric[
                        "normal"
                    ]
                )

                if finger == moving_finger:
                    # This is the real scratch-motion criterion.
                    ratio_deficit = max(
                        0.0,
                        MIN_COMBINED_TANGENT_RATIO
                        -
                        metric[
                            "tangent_ratio"
                        ],
                    )

                    tangent_deficit = max(
                        0.0,
                        MIN_COMBINED_TANGENTIAL_KEYFRAME_M
                        -
                        metric[
                            "tangential"
                        ],
                    )

                    normal_excess = max(
                        0.0,
                        normal_abs
                        -
                        MAX_COMBINED_NORMAL_KEYFRAME_M,
                    )

                    combined_cost += (
                        35.0
                        *
                        (
                            ratio_deficit
                            /
                            max(
                                MIN_COMBINED_TANGENT_RATIO,
                                1e-6,
                            )
                        ) ** 2
                    )

                    combined_cost += (
                        22.0
                        *
                        (
                            tangent_deficit
                            /
                            max(
                                MIN_COMBINED_TANGENTIAL_KEYFRAME_M,
                                1e-6,
                            )
                        ) ** 2
                    )

                    combined_cost += (
                        30.0
                        *
                        (
                            normal_excess
                            /
                            max(
                                MAX_COMBINED_NORMAL_KEYFRAME_M,
                                1e-6,
                            )
                        ) ** 2
                    )

                else:
                    # Existing contact fingers may stay almost stationary.
                    # We mainly prevent J4/J5 compensation from pumping them
                    # in/out of the back.
                    normal_excess = max(
                        0.0,
                        normal_abs
                        -
                        MAX_EXISTING_CONTACT_NORMAL_SHIFT_M,
                    )

                    combined_cost += (
                        20.0
                        *
                        (
                            normal_excess
                            /
                            max(
                                MAX_EXISTING_CONTACT_NORMAL_SHIFT_M,
                                1e-6,
                            )
                        ) ** 2
                    )

        return (
            geometry[
                "cost"
            ]
            +
            motion_cost
            +
            contact_posture_penalty(
                q,
                warm_q,
            )
            +
            endpoint_cost
            +
            combined_cost
            +
            lane_cost
        )

    result = minimize(
        objective,
        warm_q,
        method="Powell",
        bounds=list(
            zip(
                lower_local,
                upper_local,
            )
        ),
        options={
            "maxiter":
                32,

            "xtol":
                1e-4,

            "ftol":
                1e-4,

            "disp":
                False,
        },
    )

    q = np.asarray(
        result.x,
        dtype=np.float64,
    )

    geometry = dynamic_keyframe_geometry(
        solver,
        arcs,
        mode,
        q,
        finger_commands,
        required_active,
    )

    if not geometry[
        "valid"
    ]:
        if failure_out is not None:
            finger_rejects = {
                finger: result.get("reject", "UNKNOWN")
                for finger, result in geometry.get("finger_results", {}).items()
                if not bool(result.get("valid", False))
            }
            failure_out.clear()
            failure_out.update(
                {
                    "reason": "CONTACT_GEOMETRY",
                    "finger_rejects": finger_rejects,
                    "inactive_penetration_mm": round(
                        float(geometry.get("inactive_penetration", 0.0)) * 1000.0,
                        3,
                    ),
                    "palm_angle_deg": round(
                        float(geometry.get("palm_back_angle_deg", 999.0)),
                        3,
                    ),
                }
            )
        return None

    lane_cost, lane_valid, lane_details = active_lane_state(q)
    if not lane_valid:
        if failure_out is not None:
            offenders = {
                finger: {
                    "cross_mm": round(float(detail["cross_error"]) * 1000.0, 3),
                    "radius_mm": round(float(detail["patch_radius"]) * 1000.0, 3),
                }
                for finger, detail in lane_details.items()
                if (
                    float(detail["cross_error"]) > PLANNED_ACTIVE_LANE_CROSS_M
                    or float(detail["patch_radius"]) > PLANNED_ACTIVE_PATCH_RADIUS_M
                )
            }
            failure_out.clear()
            failure_out.update(
                {
                    "reason": "TARGET_REGION",
                    "offenders": offenders,
                    "limits_mm": {
                        "cross": round(PLANNED_ACTIVE_LANE_CROSS_M * 1000.0, 1),
                        "radius": round(PLANNED_ACTIVE_PATCH_RADIUS_M * 1000.0, 1),
                    },
                }
            )
        return None

    metrics = {}

    if previous_active:
        metrics = combined_motion_metrics(
            solver,
            arcs,
            mode,
            warm_q,
            previous_finger_commands,
            q,
            finger_commands,
            previous_active,
        )

    return {
        "q":
            q,

        "geometry":
            geometry,

        "combined_motion":
            metrics,

        "lane_cost":
            lane_cost,

        "lane_details":
            lane_details,
    }

def make_sync_phase_samples(
    start_deg,
    final_deg,
):
    start_deg = float(
        start_deg
    )

    final_deg = float(
        final_deg
    )

    onset = min(
        final_deg,
        max(
            SYNC_CONTACT_ONSET_DEG,
            start_deg,
        ),
    )

    values = []

    if onset > start_deg + 1e-6:
        values.append(
            onset
        )

    current = onset

    while (
        current
        +
        SYNC_PHASE_STEP_DEG
        <
        final_deg
        -
        1e-6
    ):
        current += (
            SYNC_PHASE_STEP_DEG
        )

        values.append(
            current
        )

    if (
        not values
        or
        abs(
            values[-1]
            -
            final_deg
        )
        > 1e-6
    ):
        values.append(
            final_deg
        )

    return values


def arm_move_duration(
    q0,
    q1,
    minimum=MIN_ARM_MOVE_S,
):
    max_delta = float(
        np.max(
            np.abs(
                np.degrees(
                    np.asarray(
                        q1
                    )
                    -
                    np.asarray(
                        q0
                    )
                )
            )
        )
    )

    return clamp(
        max_delta
        / ARM_MAX_SPEED_DEG_S,
        minimum,
        MAX_ARM_MOVE_S,
    )


def add_keyframe(
    trajectory,
    stage,
    duration,
    q,
    fingers,
    active,
    authorized_contact_fingers=None,
):
    if authorized_contact_fingers is None:
        authorized_contact_fingers = active

    trajectory.append(
        {
            "stage":
                str(
                    stage
                ),

            "duration":
                float(
                    duration
                ),

            "q":
                np.asarray(
                    q,
                    dtype=np.float64,
                ).copy(),

            "fingers":
                dict(
                    fingers
                ),

            "active":
                list(
                    active
                ),

            "authorized_contact_fingers":
                list(
                    authorized_contact_fingers
                ),
        }
    )


def solve_offset_pose(
    solver,
    position,
    rotation,
    normal,
    offset,
    warm_q,
    home_q,
    lower,
    upper,
):
    target = (
        np.asarray(
            position
        )
        +
        float(
            offset
        )
        *
        np.asarray(
            normal
        )
    )

    seeds = (
        [
            warm_q,
            home_q,
        ]
        +
        build_q_seeds(
            warm_q,
            home_q,
            lower,
            upper,
            0.0,
        )[:55]
    )

    solutions = solve_lula_ik(
        solver,
        target,
        rotation,
        seeds,
        lower,
        upper,
    )

    if not solutions:
        return None

    solutions.sort(
        key=lambda x:
            float(
                np.linalg.norm(
                    x["q"]
                    -
                    warm_q
                )
            )
    )

    return solutions[0]["q"]


def make_acquisition_phase_samples(
    start_deg,
    final_deg,
):
    """
    Candidate phases at which a NEW finger is allowed to establish contact.

    We do not insist that contact happens at one arbitrary fixed angle.
    """
    start_deg = float(
        start_deg
    )

    final_deg = float(
        final_deg
    )

    first = max(
        start_deg,
        ACQUIRE_PHASE_START_DEG,
    )

    if first > final_deg:
        first = final_deg

    values = []
    phase = first

    while phase < final_deg - 1e-6:
        values.append(
            float(
                phase
            )
        )

        phase += (
            ACQUIRE_PHASE_STEP_DEG
        )

    if (
        not values
        or
        abs(
            values[-1]
            -
            final_deg
        )
        > 1e-6
    ):
        values.append(
            final_deg
        )

    return values


def warm_tip_surface_state(
    solver,
    arcs,
    mode,
    q,
    finger_commands,
    finger,
):
    """
    Diagnostic state before the arm/wrist acquisition solve.
    """
    point = world_tip_position(
        solver,
        arcs,
        q,
        finger_commands,
        finger,
    )

    (
        distance,
        _,
        _,
        signed,
    ) = nearest_local(
        mode,
        point,
    )

    return (
        float(
            signed
        ),
        float(
            distance
        ),
    )


def solve_contact_acquisition_q(
    solver,
    arcs,
    mode,
    finger_commands,
    required_active,
    previous_finger_commands,
    previous_active,
    warm_q,
    lower,
    upper,
    endpoint_q,
    lane_spec=None,
):
    """
    Establish contact for a NEW finger.

    This stage is intentionally different from scratch motion:
      - normal approach is allowed;
      - no tangential-stroke requirement is imposed on the new finger yet;
      - already-contacting fingers must remain safe;
      - J4/J5 are preferred for local fitting;
      - several seeds are tried instead of trusting one Powell basin.
    """
    warm_q = np.asarray(
        warm_q,
        dtype=np.float64,
    )

    endpoint_q = np.asarray(
        endpoint_q,
        dtype=np.float64,
    )

    # V10: a late open-hand handoff can still be tens of millimetres away
    # from the target-locked endpoint.  The old V9 bounds were centred only
    # on warm_q, so the first working finger could flex through its whole arc
    # while the arm was mathematically forbidden from reaching endpoint_q.
    # Keep the local warm-q envelope, but expand it just enough to include the
    # already validated target-locked endpoint.  Swept collision checking still
    # decides whether that larger motion is executable.
    local_delta = np.radians(ACQUIRE_LOCAL_DELTA_DEG)
    lower_local = np.maximum(
        lower,
        np.minimum(warm_q - local_delta, endpoint_q),
    )

    upper_local = np.minimum(
        upper,
        np.maximum(warm_q + local_delta, endpoint_q),
    )

    seeds = [
        warm_q,
        np.clip(
            0.50
            *
            (
                warm_q
                +
                endpoint_q
            ),
            lower_local,
            upper_local,
        ),
        np.clip(
            0.75
            *
            warm_q
            +
            0.25
            *
            endpoint_q,
            lower_local,
            upper_local,
        ),
        np.clip(
            endpoint_q,
            lower_local,
            upper_local,
        ),
    ]

    previous_active = list(
        previous_active
    )

    required_active = list(
        required_active
    )

    def lane_lock_state(q):
        """Return continuous lane cost plus the hard per-finger lane gate."""
        if lane_spec is None:
            return 0.0, True, {}

        lane_cost = 0.0
        lane_valid = True
        details = {}

        for active_finger in required_active:
            tip = world_tip_position(
                solver,
                arcs,
                q,
                finger_commands,
                active_finger,
            )
            state = lane_relation_for_world_point(
                tip,
                lane_spec,
                active_finger,
            )
            details[active_finger] = state
            lane_valid = lane_valid and bool(
                state["cross_error"] <= PLANNED_ACTIVE_LANE_CROSS_M
                and state["patch_radius"] <= PLANNED_ACTIVE_PATCH_RADIUS_M
            )

            lane_cost += (
                TARGET_LANE_LOCK_WEIGHT
                * (state["cross_error"] / max(PLANNED_ACTIVE_LANE_CROSS_M, 1e-6)) ** 2
            )
            lane_cost += (
                0.25
                * TARGET_PATCH_LOCK_WEIGHT
                * (state["patch_radius"] / max(PLANNED_ACTIVE_PATCH_RADIUS_M, 1e-6)) ** 2
            )

        return float(lane_cost), bool(lane_valid), details

    best = None

    for seed in seeds:

        def objective(q):
            q = np.asarray(
                q,
                dtype=np.float64,
            )

            geometry = dynamic_keyframe_geometry(
                solver,
                arcs,
                mode,
                q,
                finger_commands,
                required_active,
            )

            delta_deg = np.degrees(
                q
                -
                warm_q
            )

            motion_cost = (
                0.006
                *
                float(
                    np.sum(
                        SYNC_JOINT_MOVE_WEIGHTS
                        *
                        np.square(
                            delta_deg
                        )
                    )
                )
            )

            endpoint_delta_deg = np.degrees(
                q
                -
                endpoint_q
            )

            endpoint_bias = (
                0.0008
                *
                float(
                    np.sum(
                        np.square(
                            endpoint_delta_deg
                        )
                    )
                )
            )

            existing_contact_cost = 0.0

            lane_cost, _, _ = lane_lock_state(q)

            if previous_active:
                metrics = combined_motion_metrics(
                    solver,
                    arcs,
                    mode,
                    warm_q,
                    previous_finger_commands,
                    q,
                    finger_commands,
                    previous_active,
                )

                for metric in metrics.values():
                    normal_excess = max(
                        0.0,
                        abs(
                            metric[
                                "normal"
                            ]
                        )
                        -
                        MAX_EXISTING_CONTACT_NORMAL_SHIFT_M,
                    )

                    existing_contact_cost += (
                        30.0
                        *
                        (
                            normal_excess
                            /
                            max(
                                MAX_EXISTING_CONTACT_NORMAL_SHIFT_M,
                                1e-6,
                            )
                        ) ** 2
                    )

            return (
                geometry[
                    "cost"
                ]
                +
                motion_cost
                +
                endpoint_bias
                +
                existing_contact_cost
                +
                lane_cost
                +
                contact_posture_penalty(
                    q,
                    warm_q,
                )
            )

        result = minimize(
            objective,
            np.asarray(
                seed,
                dtype=np.float64,
            ),
            method="Powell",
            bounds=list(
                zip(
                    lower_local,
                    upper_local,
                )
            ),
            options={
                "maxiter":
                    ACQUIRE_MAXITER,

                "xtol":
                    1e-4,

                "ftol":
                    1e-4,

                "disp":
                    False,
            },
        )

        q = np.asarray(
            result.x,
            dtype=np.float64,
        )

        geometry = dynamic_keyframe_geometry(
            solver,
            arcs,
            mode,
            q,
            finger_commands,
            required_active,
        )

        if not geometry[
            "valid"
        ]:
            continue

        lane_cost, lane_valid, lane_details = lane_lock_state(q)
        if not lane_valid:
            continue

        score = (
            float(
                geometry[
                    "cost"
                ]
            )
            +
            contact_posture_penalty(
                q,
                warm_q,
            )
            +
            0.003
            *
            float(
                np.sum(
                    SYNC_JOINT_MOVE_WEIGHTS
                    *
                    np.square(
                        np.degrees(
                            q
                            -
                            warm_q
                        )
                    )
                )
            )
            +
            lane_cost
        )

        if (
            best is None
            or
            score
            <
            best[
                "score"
            ]
        ):
            best = {
                "q":
                    q,

                "geometry":
                    geometry,

                "lane_details":
                    lane_details,

                "score":
                    score,
            }

    return best



# ============================================================
# V6 OFFLINE FULL-ROBOT COLLISION PROBES
# ============================================================

def _xyz(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = [float(v) for v in str(text).replace(",", " ").split()]
    if len(values) != 3:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _transform_matrix(xyz=None, rpy=None):
    t = np.eye(4, dtype=np.float64)
    if rpy is not None:
        t[:3, :3] = Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()
    if xyz is not None:
        t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def _axis_rotation(axis, angle):
    axis = normalize(axis)
    t = np.eye(4, dtype=np.float64)
    if axis is not None and abs(float(angle)) > 1e-12:
        t[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
    return t


def _axis_translation(axis, distance):
    axis = normalize(axis)
    t = np.eye(4, dtype=np.float64)
    if axis is not None:
        t[:3, 3] = axis * float(distance)
    return t


def _aabb_probe_points(minimum, maximum, max_points=15):
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    center = 0.5 * (minimum + maximum)
    points = []
    for x in [minimum[0], maximum[0]]:
        for y in [minimum[1], maximum[1]]:
            for z in [minimum[2], maximum[2]]:
                points.append([x, y, z])
    # Face centers + volume center catch broad flat palm/link collisions that
    # corner-only probing can miss.
    points.extend([
        [minimum[0], center[1], center[2]],
        [maximum[0], center[1], center[2]],
        [center[0], minimum[1], center[2]],
        [center[0], maximum[1], center[2]],
        [center[0], center[1], minimum[2]],
        [center[0], center[1], maximum[2]],
        center.tolist(),
    ])
    points = np.asarray(points[: int(max_points)], dtype=np.float64)
    return points


class URDFRobotCollisionProbeModel:
    """
    Lightweight offline collision model built from the real URDF collision
    geometry. It is intentionally conservative and supplements, rather than
    replaces, the exact PhysX contact callback.

    For mesh collisions V8 samples REAL STL surface vertices/triangle centroids;
    it never substitutes or inflates an AABB. Between adjacent trajectory samples,
    each real surface probe is swept as a finite segment and intersected against the
    true BackSurface triangle mesh. For primitive collisions we create equivalent
    local probes. Candidate link transforms are computed from the URDF using
    J1-J5 plus the commanded MCP/mimic joints. The complete model is calibrated
    to Lula at right_base_link on every query, so the probe geometry stays in
    the same world frame as the existing planner.
    """

    def __init__(self, urdf_path):
        self.urdf_path = str(urdf_path)
        self.project_dir = os.path.dirname(self.urdf_path)
        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()
        self.joints = {}
        self.parent_joint = {}
        self.link_probes = {}
        self._mesh_lookup_cache = {}

        for joint in self.root.findall("joint"):
            name = joint.attrib.get("name", "")
            jtype = joint.attrib.get("type", "fixed")
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is None or child_el is None:
                continue
            parent = parent_el.attrib.get("link", "")
            child = child_el.attrib.get("link", "")
            origin_el = joint.find("origin")
            xyz = _xyz(origin_el.attrib.get("xyz") if origin_el is not None else None)
            rpy = _xyz(origin_el.attrib.get("rpy") if origin_el is not None else None)
            axis_el = joint.find("axis")
            axis = _xyz(axis_el.attrib.get("xyz") if axis_el is not None else "1 0 0", (1.0, 0.0, 0.0))
            mimic_el = joint.find("mimic")
            mimic = None
            if mimic_el is not None:
                mimic = {
                    "joint": mimic_el.attrib.get("joint", ""),
                    "multiplier": float(mimic_el.attrib.get("multiplier", "1")),
                    "offset": float(mimic_el.attrib.get("offset", "0")),
                }
            data = {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin": _transform_matrix(xyz, rpy),
                "axis": axis,
                "mimic": mimic,
            }
            self.joints[name] = data
            self.parent_joint[child] = data

        for link in self.root.findall("link"):
            name = link.attrib.get("name", "")
            if not self._should_probe_link(name):
                continue
            probes = []
            for collision in link.findall("collision"):
                origin_el = collision.find("origin")
                xyz = _xyz(origin_el.attrib.get("xyz") if origin_el is not None else None)
                rpy = _xyz(origin_el.attrib.get("rpy") if origin_el is not None else None)
                collision_tf = _transform_matrix(xyz, rpy)
                geometry = collision.find("geometry")
                if geometry is None:
                    continue
                local = self._geometry_probe_points(geometry)
                if local is None or len(local) == 0:
                    continue
                homogeneous = np.column_stack([local, np.ones(len(local), dtype=np.float64)])
                transformed = (collision_tf @ homogeneous.T).T[:, :3]
                probes.append(transformed)
            if probes:
                merged = np.vstack(probes)
                # Deduplicate close probes so repeated collision elements do not
                # multiply the trajectory-check cost.
                rounded = np.round(merged, 5)
                _, ids = np.unique(rounded, axis=0, return_index=True)
                self.link_probes[name] = merged[np.sort(ids)]

        if PALM_FRAME not in self.parent_joint and PALM_FRAME not in [l.attrib.get("name", "") for l in self.root.findall("link")]:
            raise RuntimeError("Collision URDF does not contain PALM_FRAME " + PALM_FRAME)

        required_links = [PALM_FRAME]
        for finger in LONG_FINGERS:
            required_links.extend([
                "right_" + finger + "_proximal_link",
                "right_" + finger + "_middle_link",
                "right_" + finger + "_distal_link",
            ])
        missing = [name for name in required_links if name not in self.link_probes]
        if missing:
            raise RuntimeError(
                "Offline collision probes missing required palm/finger links: "
                + ",".join(missing)
            )

        self.base_guard = BASE_GUARD.BaseGuard(self, HAND_MODEL)

        print(
            "[V8 COLLISION MODEL] real STL surface probes loaded",
            "links=", len(self.link_probes),
            "points=", int(sum(len(v) for v in self.link_probes.values())),
            "mesh_aabb_inflation=OFF",
        )

    def _should_probe_link(self, name):
        # Planned fingertip contact is handled by the measured arc model, so
        # tip links are excluded. Everything else with collision geometry may
        # never touch BackSurface during approach/scratch/unload.
        if not name:
            return False
        if name in ["base_link"]:
            return False
        if name.endswith("_tip_link"):
            return False
        return True

    def _resolve_mesh(self, filename):
        filename = str(filename or "")
        if not filename:
            return None
        filename = filename.replace("\\", "/")
        candidates = []
        if filename.startswith("file://"):
            candidates.append(filename[7:])
        elif filename.startswith("package://"):
            rel = filename[len("package://"):]
            parts = rel.split("/", 1)
            tail = parts[1] if len(parts) == 2 else parts[0]
            candidates.extend([
                os.path.join(PROJECT_DIR, tail),
                os.path.join(self.project_dir, tail),
                os.path.join(PROJECT_DIR, os.path.basename(tail)),
            ])
        elif os.path.isabs(filename):
            candidates.append(filename)
        else:
            candidates.extend([
                os.path.join(self.project_dir, filename),
                os.path.join(PROJECT_DIR, filename),
            ])

        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                return candidate

        basename = os.path.basename(filename)
        if basename in self._mesh_lookup_cache:
            return self._mesh_lookup_cache[basename]

        # One-time basename search in the project directory. This handles URDFs
        # copied out of a ROS package while keeping their old package:// URI.
        found = None
        try:
            for root_dir, _, files in os.walk(PROJECT_DIR):
                if basename in files:
                    found = os.path.join(root_dir, basename)
                    break
        except Exception:
            found = None
        self._mesh_lookup_cache[basename] = found
        return found

    def _stl_triangles(self, path):
        """Return real STL triangles in mesh-local coordinates."""
        try:
            data = Path(path).read_bytes()
        except Exception:
            return None
        if len(data) < 84:
            return None

        # Binary STL.
        try:
            count = struct.unpack_from("<I", data, 80)[0]
            expected = 84 + int(count) * 50
            if expected == len(data):
                dtype = np.dtype([
                    ("normal", "<f4", (3,)),
                    ("verts", "<f4", (3, 3)),
                    ("attr", "<u2"),
                ])
                arr = np.frombuffer(data, dtype=dtype, count=int(count), offset=84)
                triangles = np.asarray(arr["verts"], dtype=np.float64)
                if len(triangles):
                    return triangles
        except Exception:
            pass

        # ASCII STL fallback.
        verts = []
        try:
            text = data.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("vertex "):
                    values = stripped.split()[1:4]
                    if len(values) == 3:
                        verts.append([float(v) for v in values])
        except Exception:
            verts = []
        if len(verts) < 3:
            return None
        verts = np.asarray(verts, dtype=np.float64)
        usable = (len(verts) // 3) * 3
        return verts[:usable].reshape(-1, 3, 3)

    def _surface_probe_points(self, triangles, max_points):
        """
        Select representative points that lie ON the real collision mesh.

        V6 used the STL AABB corners/face-centres and expanded the box by 1 mm.
        For tapered finger links those synthetic points can lie well outside the
        actual STL and create false BackSurface penetrations. V8 samples actual
        triangle vertices and centroids only, then uses deterministic directional
        extrema + farthest-point filling to cover the link surface.
        """
        triangles = np.asarray(triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
            return None

        vertices = triangles.reshape(-1, 3)
        centroids = np.mean(triangles, axis=1)
        candidates = np.vstack([vertices, centroids])
        rounded = np.round(candidates, 6)
        _, ids = np.unique(rounded, axis=0, return_index=True)
        candidates = candidates[np.sort(ids)]
        if len(candidates) <= int(max_points):
            return candidates

        center = np.mean(vertices, axis=0)
        selected = []
        selected_set = set()

        # 26 deterministic directions capture all axis/edge/corner extremes.
        directions = []
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                for dz in (-1.0, 0.0, 1.0):
                    if dx == 0.0 and dy == 0.0 and dz == 0.0:
                        continue
                    d = np.asarray([dx, dy, dz], dtype=np.float64)
                    d /= max(float(np.linalg.norm(d)), 1e-12)
                    directions.append(d)

        relative = candidates - center
        for d in directions:
            idx = int(np.argmax(relative @ d))
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= int(max_points):
                break

        # Fill remaining slots by deterministic farthest-point sampling over
        # actual surface candidates. This improves coverage without inventing
        # any AABB geometry.
        if not selected:
            selected = [0]
            selected_set = {0}
        chosen = candidates[np.asarray(selected, dtype=int)]
        min_d2 = np.min(
            np.sum((candidates[:, None, :] - chosen[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        while len(selected) < int(max_points):
            idx = int(np.argmax(min_d2))
            if idx in selected_set or float(min_d2[idx]) <= 1e-14:
                break
            selected.append(idx)
            selected_set.add(idx)
            d2 = np.sum((candidates - candidates[idx]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        return candidates[np.asarray(selected, dtype=int)]

    def _geometry_probe_points(self, geometry):
        mesh = geometry.find("mesh")
        if mesh is not None:
            path = self._resolve_mesh(mesh.attrib.get("filename", ""))
            if path is None or not str(path).lower().endswith(".stl"):
                return None
            triangles = self._stl_triangles(path)
            if triangles is None:
                return None
            local = self._surface_probe_points(
                triangles,
                HAND_COLLISION_MAX_PROBES_PER_COLLISION,
            )
            if local is None or len(local) == 0:
                return None
            scale = _xyz(
                mesh.attrib.get("scale") if mesh.attrib.get("scale") is not None else "1 1 1",
                (1.0, 1.0, 1.0),
            )
            # IMPORTANT: no synthetic +/-1 mm expansion. These are actual STL
            # surface points, preserving the true collision shape.
            return np.asarray(local, dtype=np.float64) * scale

        box = geometry.find("box")
        if box is not None:
            size = _xyz(box.attrib.get("size"), (0.01, 0.01, 0.01))
            return _aabb_probe_points(-0.5 * size, 0.5 * size, HAND_COLLISION_MAX_PROBES_PER_COLLISION)

        sphere = geometry.find("sphere")
        if sphere is not None:
            radius = float(sphere.attrib.get("radius", "0.005"))
            points = [[0.0, 0.0, 0.0]]
            for axis in np.eye(3):
                points.append((radius * axis).tolist())
                points.append((-radius * axis).tolist())
            return np.asarray(points, dtype=np.float64)

        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = float(cylinder.attrib.get("radius", "0.005"))
            length = float(cylinder.attrib.get("length", "0.01"))
            points = [[0.0, 0.0, 0.0]]
            for z in [-0.5 * length, 0.5 * length]:
                points.append([0.0, 0.0, z])
                for angle in np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False):
                    points.append([radius * math.cos(angle), radius * math.sin(angle), z])
            return np.asarray(points[:HAND_COLLISION_MAX_PROBES_PER_COLLISION], dtype=np.float64)

        return None

    def _joint_value(self, joint_name, arm_values, finger_values, cache):
        if joint_name in cache:
            return cache[joint_name]
        if joint_name in arm_values:
            value = float(arm_values[joint_name])
        elif joint_name in finger_values:
            value = float(finger_values[joint_name])
        else:
            # MCP alone cannot determine the other joints in a staged curve.
            # Require the explicit twelve-joint mapping; never use old mimic.
            for finger in LONG_FINGERS:
                if joint_name in {"right_%s_%s" % (finger, s) for s in HAND_MODEL.SUFFIXES}:
                    raise RuntimeError("STAGED_HAND_REQUIRES_ALL_JOINT_VALUES_" + joint_name)
            joint = self.joints.get(joint_name, {})
            mimic = joint.get("mimic")
            if mimic and mimic.get("joint"):
                value = (
                    float(mimic.get("multiplier", 1.0))
                    * self._joint_value(mimic["joint"], arm_values, finger_values, cache)
                    + float(mimic.get("offset", 0.0))
                )
            else:
                value = 0.0
        cache[joint_name] = value
        return value

    def _link_transform(self, link_name, arm_values, finger_values, link_cache, joint_cache):
        if link_name in link_cache:
            return link_cache[link_name]
        joint = self.parent_joint.get(link_name)
        if joint is None:
            t = np.eye(4, dtype=np.float64)
            link_cache[link_name] = t
            return t
        parent_t = self._link_transform(joint["parent"], arm_values, finger_values, link_cache, joint_cache)
        q = self._joint_value(joint["name"], arm_values, finger_values, joint_cache)
        motion = np.eye(4, dtype=np.float64)
        if joint["type"] in ("revolute", "continuous"):
            motion = _axis_rotation(joint["axis"], q)
        elif joint["type"] == "prismatic":
            motion = _axis_translation(joint["axis"], q)
        t = parent_t @ joint["origin"] @ motion
        link_cache[link_name] = t
        return t

    def world_probe_points(self, solver, q, finger_commands, actual_joint_values=None):
        q = np.asarray(q, dtype=np.float64)
        arm_values = {name: float(q[i]) for i, name in enumerate(ARM_JOINTS)}
        # Explicitly supply all 12 values. Legacy URDF mimic coefficients must
        # not silently drive collision geometry differently from the new arcs.
        finger_values = (HAND_MODEL.joint_values_rad(checked_finger_commands(finger_commands))
                         if actual_joint_values is None else actual_joint_values)
        link_cache = {}
        joint_cache = {}

        # Calibrate the URDF model to the Lula palm transform at this q.
        urdf_palm = self._link_transform(PALM_FRAME, arm_values, finger_values, link_cache, joint_cache)
        palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        lula_palm = np.eye(4, dtype=np.float64)
        lula_palm[:3, :3] = np.asarray(palm_r, dtype=np.float64)
        lula_palm[:3, 3] = np.asarray(palm_p, dtype=np.float64)
        align = lula_palm @ np.linalg.inv(urdf_palm)

        result = {}
        for link_name, local_points in self.link_probes.items():
            link_t = self._link_transform(link_name, arm_values, finger_values, link_cache, joint_cache)
            world_t = align @ link_t
            homogeneous = np.column_stack([local_points, np.ones(len(local_points), dtype=np.float64)])
            result[link_name] = (world_t @ homogeneous.T).T[:, :3]
        return result

    def check_static_points(
        self,
        points_by_link,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
        allowed_contact_fingers=None,
    ):
        """
        Detect ACTUAL near-surface contact at a sampled pose.

        Gross open-hand approach remains strict.  After the measured finger arc
        handoff begins, the terminal/distal phalanx of an explicitly allowed
        working finger may enter the compliant BackSurface within the same 8 mm
        scratch-depth envelope used by that finger's measured contact point.
        Proximal/middle links, palm and every inactive finger remain strict.

        Negative signed distance is used only inside the existing reliable
        near-surface gate; BackSurface is still treated as an open sheet.
        """
        if back_mesh_query is None or not getattr(back_mesh_query, "available", False):
            return False, "TRUE_SWEEP_BACK_MESH_UNAVAILABLE"

        allowed_contact_fingers = set(allowed_contact_fingers or [])

        for link_name, points in points_by_link.items():
            allowed_terminal_finger = None
            for finger in allowed_contact_fingers:
                if link_name == "right_" + finger + "_distal_link":
                    allowed_terminal_finger = finger
                    break
            if len(points) == 0:
                continue

            distances, _ = back_tree.query(points, k=1)
            near_ids = np.where(
                np.asarray(distances) <= HAND_COLLISION_COARSE_GATE_M
            )[0]

            for probe_index in near_ids:
                point = points[int(probe_index)]
                relation = final_surface_relation(
                    point,
                    back_mesh_query,
                    back_points,
                    back_normals,
                    back_tree,
                )
                if allowed_terminal_finger is not None:
                    # Planned working terminal contact is allowed on the soft
                    # surface, but it is not unlimited.  The existing distance-
                    # gated signed relation is reliable through the 8 mm hard
                    # depth limit and prevents the distal phalanx from simply
                    # disappearing through the sheet.
                    if float(relation["signed"]) < HARD_PENETRATION_M:
                        return (
                            False,
                            "WORKING_DISTAL_TOO_DEEP"
                            + "|finger=" + allowed_terminal_finger
                            + "|robot_link=" + link_name
                            + "|probe=" + str(int(probe_index))
                            + "|triangle=" + str(relation.get("id"))
                            + "|signed_mm=" + str(round(float(relation["signed"]) * 1000.0, 3))
                            + "|distance_mm=" + str(round(float(relation["distance"]) * 1000.0, 3)),
                        )
                    continue

                # A steep palm can legitimately place an open-hand link very
                # close to the sheet before the first fingertip contact.  The
                # old distance-only epsilon rejected that valid positive-side
                # clearance (for example ring distal at +0.063 mm) even when
                # there was no mesh intersection.  Keep true penetration
                # protection, but do not reject mere near-surface tangency.
                if (float(relation["distance"]) <= HAND_STATIC_CONTACT_EPS_M
                        and float(relation["signed"]) < INACTIVE_PENETRATION_M):
                    return (
                        False,
                        "STATIC_SURFACE_PENETRATION"
                        + "|robot_link=" + link_name
                        + "|probe=" + str(int(probe_index))
                        + "|triangle=" + str(relation.get("id"))
                        + "|distance_mm=" + str(round(float(relation["distance"]) * 1000.0, 3))
                        + "|raw_signed_mm=" + str(round(float(relation["raw_signed"]) * 1000.0, 3))
                        + "|signed_mm=" + str(round(float(relation["signed"]) * 1000.0, 3)),
                    )

        return True, "PASS"

    def check_swept_points(
        self,
        previous_points_by_link,
        current_points_by_link,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
        step_index,
        path_fraction,
        allowed_contact_fingers=None,
    ):
        """
        True swept-surface test for every tracked STL surface probe.

        For each probe, the previous and current world positions define a finite
        segment. Only a real segment/BackSurface-triangle intersection rejects
        the motion. Open-sheet signed side never rejects the segment.
        """
        if back_mesh_query is None or not getattr(back_mesh_query, "available", False):
            return False, "TRUE_SWEEP_BACK_MESH_UNAVAILABLE"

        allowed_contact_fingers = set(allowed_contact_fingers or [])

        for link_name, current_points in current_points_by_link.items():
            allowed_terminal = any(
                link_name == "right_" + finger + "_distal_link"
                for finger in allowed_contact_fingers
            )
            previous_points = previous_points_by_link.get(link_name)
            if previous_points is None:
                return False, "TRUE_SWEEP_LINK_HISTORY_MISSING=" + str(link_name)
            if len(previous_points) != len(current_points):
                return False, "TRUE_SWEEP_PROBE_COUNT_CHANGED=" + str(link_name)

            for probe_index, (p0, p1) in enumerate(zip(previous_points, current_points)):
                hit = back_mesh_query.segment_intersection(
                    p0,
                    p1,
                    eps=HAND_SWEEP_INTERSECTION_EPS_M,
                    aabb_pad=HAND_SWEEP_AABB_PAD_M,
                )
                if hit is None:
                    continue

                # Once the measured finger-arc handoff has begun, crossing the
                # soft BackSurface is an intended terminal-contact event for
                # the working distal phalanx.  Every other link remains a hard
                # swept-intersection reject.  Static depth is checked below.
                if allowed_terminal:
                    continue

                point = np.asarray(hit["point"], dtype=np.float64)
                bary = np.asarray(hit["barycentric"], dtype=np.float64)
                return (
                    False,
                    "TRUE_SWEEP_INTERSECTION"
                    + "|robot_link=" + link_name
                    + "|probe=" + str(int(probe_index))
                    + "|step=" + str(int(step_index))
                    + "|path_fraction=" + str(round(float(path_fraction), 4))
                    + "|segment_t=" + str(round(float(hit["t"]), 6))
                    + "|triangle=" + str(int(hit["triangle_id"]))
                    + "|p0=" + np.array2string(np.asarray(p0), precision=5, separator=",")
                    + "|p1=" + np.array2string(np.asarray(p1), precision=5, separator=",")
                    + "|hit=" + np.array2string(point, precision=5, separator=",")
                    + "|bary=" + np.array2string(bary, precision=5, separator=","),
                )

        # Catch a sampled pose that is tangent/on-surface without crossing the
        # sheet between frames. This uses Euclidean surface distance only.
        return self.check_static_points(
            current_points_by_link,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            allowed_contact_fingers=allowed_contact_fingers,
        )



def segment_mesh_safe(
    solver,
    arcs,
    q0,
    fingers0,
    q1,
    fingers1,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    samples=18,
    allowed_contact_fingers=None,
):
    """
    Hard safety check for one interpolated arm+finger segment.

    V8 non-tip safety uses true swept STL-surface probe segments against the
    BackSurface triangles. Open-sheet signed side is never treated as body
    penetration. Planned fingertips keep their separate measured-arc contact
    depth policy.

    Only fingers explicitly listed in allowed_contact_fingers may use the
    scratch hard limit (8 mm). Every other finger keeps the stricter inactive
    threshold, so widening the scratch-depth window never weakens non-working
    finger safety.
    """
    allowed_contact_fingers = set(allowed_contact_fingers or [])
    previous_hand_points = None

    for sample_index, fraction in enumerate(np.linspace(
        0.0,
        1.0,
        int(
            samples
        ),
    )):
        alpha = smoothstep(
            fraction
        )

        q = (
            np.asarray(
                q0,
                dtype=np.float64,
            )
            +
            alpha
            *
            (
                np.asarray(
                    q1,
                    dtype=np.float64,
                )
                -
                np.asarray(
                    q0,
                    dtype=np.float64,
                )
            )
        )

        palm_p, palm_r = solver.compute_forward_kinematics(
            PALM_FRAME,
            q,
        )

        palm_p = np.asarray(
            palm_p,
            dtype=np.float64,
        )

        palm_r = np.asarray(
            palm_r,
            dtype=np.float64,
        )

        finger_commands_now = {
            finger: (
                float(fingers0.get(finger, 0.0))
                + alpha * (float(fingers1.get(finger, 0.0)) - float(fingers0.get(finger, 0.0)))
            )
            for finger in LONG_FINGERS
        }

        if (HAND_COLLISION_MODEL is not None
                and hasattr(HAND_COLLISION_MODEL, "base_guard")
                and not bool(globals().get("BASE_PLANNING_GUARD_DISABLED", False))):
            gap, link = HAND_COLLISION_MODEL.base_guard.clearance(q, finger_commands_now)
            if gap < V103_BASE_MIN_CLEARANCE_M:
                return False, "HAND_BASE_MARGIN|link=" + str(link) + "|gap_mm=" + str(round(gap*1000.,2))

        if HAND_COLLISION_MODEL is not None:
            try:
                hand_points_now = HAND_COLLISION_MODEL.world_probe_points(
                    solver,
                    q,
                    finger_commands_now,
                )
            except Exception as exc:
                return False, "COLLISION_MODEL_FK_ERROR=" + str(exc)

            if previous_hand_points is None:
                hand_safe, hand_reason = HAND_COLLISION_MODEL.check_static_points(
                    hand_points_now,
                    back_points,
                    back_normals,
                    back_tree,
                    back_mesh_query,
                    allowed_contact_fingers=allowed_contact_fingers,
                )
            else:
                hand_safe, hand_reason = HAND_COLLISION_MODEL.check_swept_points(
                    previous_hand_points,
                    hand_points_now,
                    back_points,
                    back_normals,
                    back_tree,
                    back_mesh_query,
                    step_index=sample_index,
                    path_fraction=float(fraction),
                    allowed_contact_fingers=allowed_contact_fingers,
                )

            if not hand_safe:
                return (
                    False,
                    hand_reason
                    + "|alpha=" + str(round(float(fraction), 3))
                    + "|smooth_alpha=" + str(round(float(alpha), 3)),
                )

            previous_hand_points = hand_points_now

        for finger in LONG_FINGERS:
            angle = (
                float(
                    fingers0.get(
                        finger,
                        0.0,
                    )
                )
                +
                alpha
                *
                (
                    float(
                        fingers1.get(
                            finger,
                            0.0,
                        )
                    )
                    -
                    float(
                        fingers0.get(
                            finger,
                            0.0,
                        )
                    )
                )
            )

            local_point, _ = interpolate_arc(
                arcs,
                finger,
                angle,
            )

            world_point = (
                palm_p
                +
                palm_r
                @
                local_point
            )

            relation = final_surface_relation(
                world_point,
                back_mesh_query,
                back_points,
                back_normals,
                back_tree,
            )

            hard_limit = (
                HARD_PENETRATION_M
                if finger in allowed_contact_fingers
                else INACTIVE_PENETRATION_M
            )

            if (
                float(
                    relation[
                        "signed"
                    ]
                )
                <
                hard_limit
            ):
                return (
                    False,
                    "finger="
                    +
                    finger
                    +
                    "|signed_mm="
                    +
                    str(
                        round(
                            float(
                                relation[
                                    "signed"
                                ]
                            )
                            *
                            1000.0,
                            2,
                        )
                    )
                    +
                    "|raw_signed_mm="
                    +
                    str(
                        round(
                            float(
                                relation[
                                    "raw_signed"
                                ]
                            )
                            *
                            1000.0,
                            2,
                        )
                    )
                    +
                    "|distance_mm="
                    +
                    str(
                        round(
                            float(
                                relation[
                                    "distance"
                                ]
                            )
                            *
                            1000.0,
                            2,
                        )
                    )
                    +
                    "|alpha="
                    +
                    str(
                        round(
                            float(
                                fraction
                            ),
                            3,
                        )
                    ),
                )

    return (
        True,
        "PASS",
    )


def active_finger_mesh_clearance(
    solver,
    arcs,
    q,
    finger_commands,
    active_fingers,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    """
    Measure actual USD-mesh clearance for the currently active fingers.
    """
    if not active_fingers:
        return {
            "clear":
                True,

            "minimum_distance":
                999.0,

            "minimum_signed":
                999.0,

            "details":
                {},
        }

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    palm_p, palm_r = solver.compute_forward_kinematics(
        PALM_FRAME,
        q,
    )

    palm_p = np.asarray(
        palm_p,
        dtype=np.float64,
    )

    palm_r = np.asarray(
        palm_r,
        dtype=np.float64,
    )

    minimum_distance = float(
        "inf"
    )

    minimum_signed = float(
        "inf"
    )

    details = {}

    for finger in active_fingers:
        angle = float(
            finger_commands.get(
                finger,
                0.0,
            )
        )

        local_point, _ = interpolate_arc(
            arcs,
            finger,
            angle,
        )

        world_point = (
            palm_p
            +
            palm_r
            @
            local_point
        )

        relation = final_surface_relation(
            world_point,
            back_mesh_query,
            back_points,
            back_normals,
            back_tree,
        )

        distance = float(
            relation[
                "distance"
            ]
        )

        signed = float(
            relation[
                "signed"
            ]
        )

        minimum_distance = min(
            minimum_distance,
            distance,
        )

        minimum_signed = min(
            minimum_signed,
            signed,
        )

        details[
            finger
        ] = {
            "distance":
                distance,

            "signed":
                signed,

            "source":
                relation[
                    "source"
                ],
        }

    clear = (
        minimum_distance
        >=
        UNLOAD_REQUIRED_DISTANCE_M
        and
        minimum_signed
        >=
        UNLOAD_REQUIRED_SIGNED_M
    )

    return {
        "clear":
            bool(
                clear
            ),

        "minimum_distance":
            float(
                minimum_distance
            ),

        "minimum_signed":
            float(
                minimum_signed
            ),

        "details":
            details,
    }


def finger_surface_relation_at_state(
    solver,
    arcs,
    q,
    finger_commands,
    finger,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    palm_p, palm_r = solver.compute_forward_kinematics(
        PALM_FRAME,
        q,
    )

    palm_p = np.asarray(
        palm_p,
        dtype=np.float64,
    )

    palm_r = np.asarray(
        palm_r,
        dtype=np.float64,
    )

    angle = float(
        finger_commands.get(
            finger,
            0.0,
        )
    )

    local_point, _ = interpolate_arc(
        arcs,
        finger,
        angle,
    )

    world_point = (
        palm_p
        +
        palm_r
        @
        local_point
    )

    relation = final_surface_relation(
        world_point,
        back_mesh_query,
        back_points,
        back_normals,
        back_tree,
    )
    relation = dict(relation)
    relation["world_point"] = np.asarray(world_point, dtype=np.float64)
    return relation


def warm_contact_is_usable(
    solver,
    arcs,
    mode,
    q_now,
    current_commands,
    next_commands,
    next_active,
    finger,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    """
    V4.10 acquisition-only warm-contact test.

    IMPORTANT:
    Contact acquisition is NOT the same thing as an already-established
    synchronized scratch state. Therefore we intentionally do NOT require
    dynamic_keyframe_geometry()["valid"] here.

    Accept immediately when:
      1. the target fingertip is genuinely close to the USD triangle mesh;
      2. its raw local signed depth is inside the safe contact band;
      3. flexing from the current finger command to this phase, with q held
         unchanged, is mesh-safe for every long finger.

    If accepted, the arm/wrist q stays unchanged. J4/J5 compensation begins in
    the later synchronized scratch phase instead of unnecessarily destroying a
    good first-contact pose.
    """
    relation = finger_surface_relation_at_state(
        solver,
        arcs,
        q_now,
        next_commands,
        finger,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
    )

    distance = float(
        relation[
            "distance"
        ]
    )

    raw_signed = float(
        relation[
            "raw_signed"
        ]
    )

    in_contact_band = (
        distance
        <=
        WARM_CONTACT_MAX_DISTANCE_M
        and
        raw_signed
        >=
        WARM_CONTACT_RAW_MIN_M
        and
        raw_signed
        <=
        WARM_CONTACT_RAW_MAX_M
    )

    if not in_contact_band:
        return (
            False,
            None,
            relation,
            "NOT_IN_WARM_CONTACT_BAND",
        )

    path_safe, reason = segment_mesh_safe(
        solver,
        arcs,
        q_now,
        current_commands,
        q_now,
        next_commands,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
        samples=24,
        allowed_contact_fingers=next_active,
    )

    if not path_safe:
        return (
            False,
            None,
            relation,
            reason,
        )

    # Geometry is retained only as a diagnostic snapshot. It is deliberately
    # NOT allowed to veto acquisition.
    try:
        geometry = dynamic_keyframe_geometry(
            solver,
            arcs,
            mode,
            q_now,
            next_commands,
            next_active,
        )
    except Exception:
        geometry = None

    return (
        True,
        geometry,
        relation,
        "PASS",
    )


def acquire_new_finger_contact(
    solver,
    arcs,
    mode,
    finger,
    final_phase,
    q_now,
    fingers,
    active,
    lower,
    upper,
    endpoint_q,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec=None,
):
    """
    V6 target-locked adaptive measured-arc contact acquisition.

    Important semantics:
      * 0-4 mm penetration is the preferred working zone.
      * 4-8 mm penetration is recoverable and may be used as a fallback.
      * >8 mm penetration is a hard reject.

    ARC3 returned at the FIRST point entering the broad 4-8 mm warm band.
    That was the direct cause of many SYNC_SCRATCH_FAIL events: a finger was
    declared acquired at ~6-7 mm, DEEP_ADJUST moved the wrist, and a second
    routine then tried to force the same finger toward the old template phase.

    ARC4 instead keeps traversing the REAL measured finger arc and prefers a
    contact state already in the normal 0-4 mm zone.  The template phase is a
    seed, not a hard endpoint.  If needed, the scan continues to the end of the
    measured arc.  A 4-8 mm state is retained only as a safe fallback.

    Existing active fingers are preserved by J4/J5-dominant dynamic
    compensation while the new finger is searched.
    """
    q_trial = np.asarray(q_now, dtype=np.float64).copy()
    q_handoff_start = q_trial.copy()
    commands_trial = dict(fingers)
    active_trial = list(active)

    start_phase = float(commands_trial.get(finger, 0.0))
    planned_phase = float(final_phase)
    arc_max_phase = float(arcs[finger]["angles"][-1])

    scan_end_phase = (
        arc_max_phase
        if ARC4_FULL_ARC_SCAN
        else min(arc_max_phase, planned_phase)
    )

    if scan_end_phase <= start_phase + 1e-6:
        return None

    phase_values = []
    phase = start_phase + ARC_CONTACT_PHASE_STEP_DEG
    while phase < scan_end_phase - 1e-6:
        phase_values.append(float(phase))
        phase += ARC_CONTACT_PHASE_STEP_DEG
    phase_values.append(float(scan_end_phase))

    # The precurve can be reduced below 42 deg when a swept collision is
    # detected.  Therefore the first-contact arm bridge must be referenced to
    # the actual first scanned phase, not to one fixed 48 deg assumption.
    assist_phase_start = float(phase_values[0])

    arc_steps = []
    fallback = None
    fallback_score = float("inf")
    preferred_seen = False

    for phase in phase_values:
        next_commands = dict(commands_trial)
        next_commands[finger] = float(phase)

        # Arm motion joins the measured finger arc either at the normal phase
        # gate or earlier when the new tip enters the real 6 mm near-surface
        # band.  The early trigger prevents an off-lane contact from occurring
        # before the fixed phase gate can be reached.
        q_candidate = q_trial.copy()
        acquisition_state = None

        preview_relation = finger_surface_relation_at_state(
            solver,
            arcs,
            q_trial,
            next_commands,
            finger,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )
        preview_distance = float(preview_relation["distance"])
        preview_raw_signed = float(preview_relation["raw_signed"])
        near_surface_trigger = (
            preview_distance <= APPROACH_PRECONTACT_NEAR_BAND_M + 1e-9
            # ``distance`` is the Euclidean tip-to-mesh separation.  On the
            # curved BackSurface it can still read 15-25 mm after the measured
            # arc has crossed the local tangent plane.  Waiting for distance
            # alone therefore let a non-tip link hit the back before the arm
            # contact solve ever ran.  The signed normal relation is the early
            # warning that the acquisition solver actually needs.
            or preview_raw_signed <= CONTACT_NEAR_OFFSET_M + 1e-9
        )
        phase_trigger = (
            float(phase) >= float(ACQUIRE_PHASE_START_DEG) - 1e-9
        )
        contact_solve_triggered = bool(near_surface_trigger or phase_trigger)

        if contact_solve_triggered:
            required_for_acquisition = list(active_trial)
            if finger not in required_for_acquisition:
                required_for_acquisition.append(finger)

            acquisition_state = solve_contact_acquisition_q(
                solver,
                arcs,
                mode,
                next_commands,
                required_for_acquisition,
                commands_trial,
                active_trial,
                q_trial,
                lower,
                upper,
                endpoint_q,
                lane_spec=lane_spec,
            )

            if acquisition_state is not None:
                q_candidate = np.asarray(acquisition_state["q"], dtype=np.float64)
                print(
                    "    [V9 ARC ARM-ASSIST CONTACT SOLVE]",
                    finger,
                    "phase=",
                    round(float(phase), 1),
                    "trigger=",
                    "NEAR_SURFACE" if near_surface_trigger and not phase_trigger else "PHASE",
                    "q_deg=",
                    np.round(np.degrees(q_candidate), 2),
                )

        # If the new finger is not yet contact-solvable, existing contacts may
        # still receive J4/J5-dominant compensation while this finger advances
        # along its measured arc.
        if acquisition_state is None and active_trial:
            state = solve_dynamic_contact_q(
                solver,
                arcs,
                mode,
                next_commands,
                active_trial,
                q_trial,
                lower,
                upper,
                endpoint_q=endpoint_q,
                previous_finger_commands=commands_trial,
                previous_active=active_trial,
                moving_finger=None,
                lane_spec=lane_spec,
            )
            if state is not None:
                q_candidate = np.asarray(state["q"], dtype=np.float64)

        # V10 first-contact bridge.  With no already-active finger, V9 could
        # leave q fixed at a late handoff 50-150 mm from the patch whenever the
        # strict contact optimizer had no valid state yet.  The finger then
        # completed its whole flex arc far from the target. Advance arm and
        # finger together from the actual first post-precurve sample toward the
        # already validated target-locked endpoint. This is only a fallback;
        # every segment still passes the complete swept collision/depth test.
        arm_bridge_delta = float(np.linalg.norm(endpoint_q - q_handoff_start))
        if (
            acquisition_state is None
            and not active_trial
            and arm_bridge_delta > np.radians(0.05)
            and float(phase) >= assist_phase_start - 1e-9
        ):
            denom = max(planned_phase - assist_phase_start, ARC_CONTACT_PHASE_STEP_DEG)
            assist_fraction = clamp(
                (float(phase) - assist_phase_start) / denom,
                0.0,
                1.0,
            )
            assist_alpha = smoothstep(assist_fraction)
            q_candidate = (
                q_handoff_start
                + assist_alpha * (endpoint_q - q_handoff_start)
            )
            q_candidate = np.clip(q_candidate, lower, upper)
            print(
                "    [V10 STAGED ARM-ASSIST]",
                finger,
                "phase=", round(float(phase), 1),
                "assist=", round(float(assist_fraction), 3),
                "q_deg=", np.round(np.degrees(q_candidate), 2),
            )

        transition_safe, transition_reason = segment_mesh_safe(
            solver,
            arcs,
            q_trial,
            commands_trial,
            q_candidate,
            next_commands,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=28,
            allowed_contact_fingers=(set(active_trial) | {finger}),
        )

        if not transition_safe and acquisition_state is not None:
            # The arm-assisted contact solve may be geometrically valid at its
            # endpoint yet sweep a non-tip finger link through the back on the
            # way there. Do NOT accept that. First try only the measured finger
            # flex at the current safe arm pose. If that is safe, defer the arm
            # motion and try the contact solve again at the next (more curled)
            # phase. This is the key V9 escape from the V8 ring-distal sweep.
            flex_only_safe, flex_only_reason = segment_mesh_safe(
                solver,
                arcs,
                q_trial,
                commands_trial,
                q_trial,
                next_commands,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=28,
                allowed_contact_fingers=(set(active_trial) | {finger}),
            )
            if flex_only_safe:
                print(
                    "    [V9 ARM-ASSIST DEFERRED; FLEX-ONLY SAFE]",
                    finger,
                    "phase=",
                    round(float(phase), 1),
                    "blocked_arm_reason=",
                    transition_reason,
                )
                q_candidate = q_trial.copy()
                transition_safe = True
                transition_reason = "PASS_FLEX_ONLY"
                acquisition_state = None
            else:
                transition_reason = (
                    transition_reason
                    + "|FLEX_ONLY_ALSO_UNSAFE="
                    + str(flex_only_reason)
                )

        if not transition_safe:
            # If a usable contact state was already found, do not throw it
            # away merely because a later exploratory full-arc sample is bad.
            if fallback is not None:
                print(
                    "    [V4.12-ARC4 FULL ARC STOP]",
                    finger,
                    "phase=",
                    round(float(phase), 1),
                    transition_reason,
                    "-> keep earlier safe contact phase",
                    round(float(fallback["phase"]), 1),
                )
                break

            print(
                "    [V4.12-ARC4 ARC PATH REJECT]",
                finger,
                "phase=",
                round(float(phase), 1),
                transition_reason,
            )
            return None

        relation = finger_surface_relation_at_state(
            solver,
            arcs,
            q_candidate,
            next_commands,
            finger,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )

        distance = float(relation["distance"])
        raw_signed = float(relation["raw_signed"])
        safety_signed = float(relation["signed"])
        lane_state = lane_relation_for_world_point(
            relation.get("world_point"),
            lane_spec,
            finger,
        )

        if safety_signed < HARD_PENETRATION_M:
            if fallback is not None:
                print(
                    "    [V4.12-ARC4 HARD LIMIT AHEAD]",
                    finger,
                    "phase=",
                    round(float(phase), 1),
                    "safety_signed_mm=",
                    round(safety_signed * 1000.0, 2),
                    "-> keep earlier safe contact phase",
                    round(float(fallback["phase"]), 1),
                )
                break

            print(
                "    [V4.12-ARC4 ARC OVER-PENETRATION]",
                finger,
                "phase=",
                round(float(phase), 1),
                "distance_mm=",
                round(distance * 1000.0, 2),
                "safety_signed_mm=",
                round(safety_signed * 1000.0, 2),
                "raw_signed_mm=",
                round(raw_signed * 1000.0, 2),
            )
            return None

        geometric_contact_band = (
            distance <= WARM_CONTACT_MAX_DISTANCE_M
            and raw_signed >= WARM_CONTACT_RAW_MIN_M
            and raw_signed <= WARM_CONTACT_RAW_MAX_M
        )

        # A new contact is valid only inside both its lateral lane and the
        # finite GUI-selected patch. Contact outside the patch is never latched.
        planned_lane_valid = bool(
            lane_state["cross_error"] <= PLANNED_ACTIVE_LANE_CROSS_M
            and lane_state["patch_radius"] <= PLANNED_ACTIVE_PATCH_RADIUS_M
        )
        if geometric_contact_band and not planned_lane_valid:
            if raw_signed <= 0.0:
                print(
                    "    [V100 TARGET-REGION CONTACT REJECT]",
                    finger,
                    "phase=", round(float(phase), 1),
                    "cross_mm=", round(lane_state["cross_error"] * 1000.0, 2),
                    "click_distance_diag_mm=", round(lane_state["patch_radius"] * 1000.0, 2),
                    "limits_mm=",
                    {
                        "cross": round(PLANNED_ACTIVE_LANE_CROSS_M * 1000.0, 1),
                        "radius": round(PLANNED_ACTIVE_PATCH_RADIUS_M * 1000.0, 1),
                    },
                )
                return None

        in_broad_contact_band = geometric_contact_band and planned_lane_valid

        in_preferred_zone = (
            in_broad_contact_band
            and raw_signed >= ARC4_PREFERRED_CONTACT_MIN_SIGNED_M
            and raw_signed <= ARC4_PREFERRED_CONTACT_MAX_SIGNED_M
        )

        next_active = list(active_trial)
        if in_broad_contact_band and finger not in next_active:
            next_active.append(finger)

        step_record = {
            "phase": float(phase),
            "q": q_candidate.copy(),
            "commands": dict(next_commands),
            "active": list(next_active),
            # This mirrors segment_mesh_safe(... allowed_contact_fingers=...).
            # The working finger is already executing a planned measured-arc
            # terminal-contact motion even before it is latched into `active`.
            "authorized_contact_fingers": list(set(active_trial) | {finger}),
            "contact": bool(in_broad_contact_band),
            "distance": distance,
            "raw_signed": raw_signed,
            "safety_signed": safety_signed,
        }
        arc_steps.append(step_record)

        print(
            "    [V4.12-ARC4 ARC STEP]",
            finger,
            "phase=",
            round(float(phase), 1),
            "distance_mm=",
            round(distance * 1000.0, 2),
            "raw_signed_mm=",
            round(raw_signed * 1000.0, 2),
            "lane_cross_mm=",
            round(float(lane_state["cross_error"]) * 1000.0, 2),
            "click_distance_diag_mm=",
            round(float(lane_state["patch_radius"]) * 1000.0, 2),
            "zone=",
            (
                "PREFERRED_0_4MM"
                if in_preferred_zone
                else (
                    "ADJUST_4_8MM"
                    if in_broad_contact_band
                    else "CLEAR"
                )
            ),
        )

        if in_broad_contact_band:
            # Primary score is contact depth error from ~2 mm.  A very small
            # phase regularizer only breaks ties in favor of the planned wave
            # shape; it cannot force an unsafe old template phase.
            score = (
                abs(raw_signed - IDEAL_SIGNED_M)
                + ARC4_CONTACT_SCORE_PHASE_WEIGHT
                * abs(float(phase) - planned_phase)
            )

            if score < fallback_score:
                fallback_score = score
                fallback = {
                    "phase": float(phase),
                    "commands": dict(next_commands),
                    "active": list(next_active),
                    "q": q_candidate.copy(),
                    "geometry": None,
                    "warm_contact": True,
                    "arc_steps": list(arc_steps),
                    "raw_signed": raw_signed,
                    "distance": distance,
                    "preferred": bool(in_preferred_zone),
                }

            # 0-4 mm is the normal operating zone.  Once reached, there is no
            # benefit in forcing the finger farther simply to satisfy an old
            # template phase.  Lock this finger and let the next finger join.
            if in_preferred_zone:
                preferred_seen = True
                print(
                    "    [V4.12-ARC4 CONTACT ACQUIRED IN NORMAL ZONE]",
                    finger,
                    "phase=",
                    round(float(phase), 1),
                    "depth_mm=",
                    round(max(0.0, -raw_signed) * 1000.0, 2),
                    "planned_phase=",
                    round(planned_phase, 1),
                    "q=",
                    np.round(np.degrees(q_candidate), 2),
                )
                return fallback

        q_trial = q_candidate
        commands_trial = next_commands

    if fallback is not None:
        print(
            "    [V4.12-ARC4 CONTACT ACQUIRED IN ADJUST ZONE]",
            finger,
            "phase=",
            round(float(fallback["phase"]), 1),
            "depth_mm=",
            round(max(0.0, -float(fallback["raw_signed"])) * 1000.0, 2),
            "planned_phase=",
            round(planned_phase, 1),
            "preferred_found=",
            bool(preferred_seen),
        )
        return fallback

    print(
        "    [V4.12-ARC4 ARC CONTACT FAIL]",
        finger,
        "planned_phase=",
        round(planned_phase, 1),
        "scan_end_phase=",
        round(scan_end_phase, 1),
    )
    return None

def recover_deep_adjust_state(
    trajectory,
    solver,
    arcs,
    mode,
    q_now,
    fingers,
    active,
    lower,
    upper,
    endpoint_q,
    label,
    lane_spec=None,
):
    """
    Recover a planned active fingertip from the 4-8 mm DEEP_ADJUST zone.

    The correction keeps finger commands fixed and re-solves the arm pose.
    Because solve_dynamic_contact_q() already makes J4/J5 cheapest, the wrist
    levels first; J2/J3 may assist and J1 remains expensive.  Hysteresis is
    4.0 mm enter / 3.5 mm exit.  Failure to fully exit the soft zone is not a
    trajectory failure as long as the state remains above the 8 mm hard limit.
    """
    q_now = np.asarray(q_now, dtype=np.float64).copy()
    fingers = dict(fingers)
    active = list(active)

    if not active:
        return {
            "q": q_now,
            "fingers": fingers,
            "active": active,
            "latched": False,
        }

    latched = False

    for pass_index in range(DEEP_ADJUST_MAX_PASSES + 1):
        geometry = dynamic_keyframe_geometry(
            solver,
            arcs,
            mode,
            q_now,
            fingers,
            active,
        )

        signed_values = [
            float(geometry["finger_results"][finger]["signed"])
            for finger in active
            if finger in geometry.get("finger_results", {})
        ]

        if not signed_values:
            break

        deepest_signed = min(signed_values)
        was_latched = latched
        latched = update_deep_adjust_latch(latched, deepest_signed)

        if latched and not was_latched:
            print(
                "    [V4.12-ARC4 DEEP_ADJUST ENTER]",
                label,
                "deepest_mm=",
                round(-deepest_signed * 1000.0, 2),
            )

        if not latched:
            if was_latched:
                print(
                    "    [V4.12-ARC4 DEEP_ADJUST EXIT]",
                    label,
                    "deepest_mm=",
                    round(-deepest_signed * 1000.0, 2),
                )

            return {
                "q": q_now,
                "fingers": fingers,
                "active": active,
                "latched": False,
            }

        if deepest_signed < DEEP_ADJUST_HARD_SIGNED_M:
            return None

        if pass_index >= DEEP_ADJUST_MAX_PASSES:
            break

        state = solve_dynamic_contact_q(
            solver,
            arcs,
            mode,
            fingers,
            active,
            q_now,
            lower,
            upper,
            endpoint_q=endpoint_q,
            previous_finger_commands=fingers,
            previous_active=active,
            moving_finger=None,
            lane_spec=lane_spec,
        )

        if state is None:
            break

        q_target = np.asarray(state["q"], dtype=np.float64)
        delta_deg = float(
            np.max(
                np.abs(
                    np.degrees(q_target - q_now)
                )
            )
        )

        if delta_deg < 0.05:
            break

        add_keyframe(
            trajectory,
            "DEEP_ADJUST_" + str(label).upper() + "_" + str(pass_index + 1),
            max(
                DEEP_ADJUST_KEYFRAME_TIME_S,
                min(0.40, delta_deg / 30.0),
            ),
            q_target,
            fingers,
            active,
        )

        q_now = q_target

    # Remaining 4-8 mm depth is still recoverable and is allowed to continue.
    final_geometry = dynamic_keyframe_geometry(
        solver,
        arcs,
        mode,
        q_now,
        fingers,
        active,
    )

    final_signed = [
        float(final_geometry["finger_results"][finger]["signed"])
        for finger in active
        if finger in final_geometry.get("finger_results", {})
    ]

    if final_signed:
        deepest_signed = min(final_signed)

        if deepest_signed < DEEP_ADJUST_HARD_SIGNED_M:
            return None

        print(
            "    [V4.12-ARC4 DEEP_ADJUST CONTINUE]",
            label,
            "deepest_mm=",
            round(-deepest_signed * 1000.0, 2),
            "hard_limit_mm=",
            round(-DEEP_ADJUST_HARD_SIGNED_M * 1000.0, 1),
        )

    return {
        "q": q_now,
        "fingers": fingers,
        "active": active,
        "latched": True,
    }


def advance_contact_finger_synchronously(
    trajectory,
    solver,
    arcs,
    mode,
    finger,
    final_phase,
    q_now,
    fingers,
    active,
    lower,
    upper,
    endpoint_q,
):
    """
    After contact has been acquired, advance the finger to its final phase.

    J4/J5 are re-solved at every small flex step. If a 4-degree step is too
    large, V4.5 automatically retries at 2 degrees before rejecting the
    candidate.
    """
    current_phase = float(
        fingers[
            finger
        ]
    )

    step_deg = float(
        SYNC_PHASE_STEP_DEG
    )

    sample_index = 0

    while current_phase < final_phase - 1e-6:
        target_phase = min(
            final_phase,
            current_phase
            +
            step_deg,
        )

        next_commands = dict(
            fingers
        )

        next_commands[
            finger
        ] = float(
            target_phase
        )

        state = solve_dynamic_contact_q(
            solver,
            arcs,
            mode,
            next_commands,
            active,
            q_now,
            lower,
            upper,
            endpoint_q=endpoint_q,
            previous_finger_commands=fingers,
            previous_active=active,
            moving_finger=finger,
        )

        if state is None:
            if (
                step_deg
                >
                ACQUIRE_MIN_SYNC_STEP_DEG
                +
                1e-6
            ):
                step_deg = max(
                    ACQUIRE_MIN_SYNC_STEP_DEG,
                    step_deg
                    /
                    2.0,
                )

                print(
                    "    [V4.12-ARC SYNC RETRY]",
                    finger,
                    "reduce step to",
                    step_deg,
                    "deg",
                )

                continue

            return None

        q_target = state[
            "q"
        ]

        q_delta_max = float(
            np.max(
                np.abs(
                    np.degrees(
                        q_target
                        -
                        q_now
                    )
                )
            )
        )

        finger_delta = abs(
            target_phase
            -
            current_phase
        )

        duration = clamp(
            max(
                q_delta_max
                /
                28.0,
                finger_delta
                /
                45.0,
            ),
            SYNC_KEYFRAME_MIN_TIME_S,
            SYNC_KEYFRAME_MAX_TIME_S,
        )

        add_keyframe(
            trajectory,
            (
                "SYNC_"
                +
                finger.upper()
                +
                "_"
                +
                str(
                    sample_index
                )
            ),
            duration,
            q_target,
            next_commands,
            active,
        )

        q_now = (
            q_target
        )

        fingers = (
            next_commands
        )

        current_phase = (
            target_phase
        )

        sample_index += 1

        # Once a reduced step works, cautiously return toward 4 deg.
        step_deg = min(
            float(
                SYNC_PHASE_STEP_DEG
            ),
            step_deg
            *
            1.5,
        )

    return {
        "q":
            q_now,

        "fingers":
            fingers,

        "active":
            active,
    }


def move_locked_active_finger(
    solver,
    arcs,
    mode,
    finger,
    target_phase,
    q_now,
    fingers,
    active,
    lower,
    upper,
    endpoint_q,
    stage_name,
    lane_spec=None,
):
    """Move one already-active finger while preserving every active contact."""
    q_now = np.asarray(q_now, dtype=np.float64).copy()
    fingers = dict(fingers)
    active = list(active)

    start_phase = float(fingers.get(finger, 0.0))
    target_phase = float(target_phase)

    if abs(target_phase - start_phase) < 1e-6:
        return {
            "q": q_now,
            "fingers": fingers,
            "active": active,
            "frame": None,
        }

    next_commands = dict(fingers)
    next_commands[finger] = target_phase

    failure_detail = {}
    state = solve_dynamic_contact_q(
        solver,
        arcs,
        mode,
        next_commands,
        active,
        q_now,
        lower,
        upper,
        endpoint_q=endpoint_q,
        previous_finger_commands=fingers,
        previous_active=active,
        moving_finger=finger,
        lane_spec=lane_spec,
        failure_out=failure_detail,
    )

    if state is None:
        print(
            "    [V100 SCRATCH KEYFRAME REJECT]",
            "stage=", stage_name,
            "moving_finger=", finger,
            "target_phase=", round(target_phase, 2),
            "detail=", failure_detail or {"reason": "NO_DYNAMIC_CONTACT_SOLUTION"},
        )
        return None

    q_target = np.asarray(state["q"], dtype=np.float64)
    q_delta_max = float(np.max(np.abs(np.degrees(q_target - q_now))))
    finger_delta = abs(target_phase - start_phase)

    duration = clamp(
        max(q_delta_max / 30.0, finger_delta / 35.0),
        ARC4_WAVE_KEYFRAME_MIN_TIME_S,
        ARC4_WAVE_KEYFRAME_MAX_TIME_S,
    )

    frame = {
        "stage": stage_name,
        "duration": float(duration),
        "q": q_target.copy(),
        "fingers": dict(next_commands),
        "active": list(active),
        "authorized_contact_fingers": list(active),
    }

    return {
        "q": q_target,
        "fingers": next_commands,
        "active": active,
        "frame": frame,
    }


def build_locked_multifinger_wave(
    solver,
    arcs,
    mode,
    sequence,
    q_now,
    fingers,
    active,
    lower,
    upper,
    endpoint_q,
    lane_spec=None,
):
    """
    Build one complete synchronized scratch wave while keeping all acquired
    fingers active.  This is the actual post-acquisition scratching motion.

    The planner tries 4 -> 3 -> 2 -> 1.5 degree excursions.  A smaller wave is
    preferable to losing four-finger contact.  Every accepted keyframe still
    passes solve_dynamic_contact_q(), which makes J4/J5 the cheapest contact
    compensation and allows J2/J3 to assist.
    """
    active = list(active)
    sequence = [f for f in sequence if f in active]

    if not sequence:
        return None

    base_q = np.asarray(q_now, dtype=np.float64).copy()
    base_fingers = dict(fingers)

    for amplitude in ARC4_WAVE_AMPLITUDE_CANDIDATES_DEG:
        q_trial = base_q.copy()
        commands_trial = dict(base_fingers)
        frames = []
        excursion_targets = {}
        failed = False

        # Forward human wave: pinky -> ring -> middle -> index.
        for finger in sequence:
            current = float(commands_trial.get(finger, 0.0))
            arc_min = float(arcs[finger]["angles"][0])
            arc_max = float(arcs[finger]["angles"][-1])

            if current + amplitude <= arc_max + 1e-6:
                target = current + amplitude
            elif current - amplitude >= arc_min - 1e-6:
                target = current - amplitude
            else:
                failed = True
                break

            excursion_targets[finger] = float(target)
            result = move_locked_active_finger(
                solver,
                arcs,
                mode,
                finger,
                target,
                q_trial,
                commands_trial,
                active,
                lower,
                upper,
                endpoint_q,
                "SCRATCH_WAVE_OUT_" + finger.upper() + "_A" + str(amplitude),
                lane_spec=lane_spec,
            )

            if result is None:
                failed = True
                break

            if result["frame"] is not None:
                frames.append(result["frame"])

            q_trial = result["q"]
            commands_trial = result["fingers"]

        if failed:
            print(
                "    [V4.12-ARC4 WAVE RETRY] amplitude_deg=",
                amplitude,
                "failed on outward wave",
            )
            continue

        # Return wave in reverse order to the locked contact phases.
        for finger in reversed(sequence):
            target = float(base_fingers.get(finger, 0.0))
            result = move_locked_active_finger(
                solver,
                arcs,
                mode,
                finger,
                target,
                q_trial,
                commands_trial,
                active,
                lower,
                upper,
                endpoint_q,
                "SCRATCH_WAVE_RETURN_" + finger.upper() + "_A" + str(amplitude),
                lane_spec=lane_spec,
            )

            if result is None:
                failed = True
                break

            if result["frame"] is not None:
                frames.append(result["frame"])

            q_trial = result["q"]
            commands_trial = result["fingers"]

        if failed:
            print(
                "    [V4.12-ARC4 WAVE RETRY] amplitude_deg=",
                amplitude,
                "failed on return wave",
            )
            continue

        print(
            "    [V4.12-ARC4 FULL MULTI-FINGER WAVE READY]",
            "active=",
            list(active),
            "amplitude_deg=",
            amplitude,
            "frames=",
            len(frames),
        )

        return {
            "q": q_trial,
            "fingers": commands_trial,
            "active": active,
            "frames": frames,
            "amplitude_deg": float(amplitude),
        }

    return None

def approach_surface_metrics(
    solver,
    arcs,
    q,
    finger_commands,
    working_fingers,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    """
    Measure the REAL fingertip-to-BackSurface state at one approach posture.

    Palm offset is only a pose-generation hint. Safety and handoff are decided
    from the actual four fingertip mesh relations.
    """
    working_set = set(
        working_fingers
    )

    relations = {}
    all_distances = []
    working_distances = []
    premature_contacts = []

    for finger in LONG_FINGERS:
        relation = finger_surface_relation_at_state(
            solver,
            arcs,
            q,
            finger_commands,
            finger,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )

        relations[
            finger
        ] = relation

        distance = float(
            relation[
                "distance"
            ]
        )

        raw_signed = float(
            relation[
                "raw_signed"
            ]
        )

        all_distances.append(
            distance
        )

        if finger in working_set:
            working_distances.append(
                distance
            )

        if (
            distance
            <=
            APPROACH_PRECONTACT_NEAR_BAND_M
            and
            raw_signed
            <
            APPROACH_PRECONTACT_RAW_MIN_M
        ):
            premature_contacts.append(
                (
                    finger,
                    distance,
                    raw_signed,
                )
            )

    return {
        "relations":
            relations,

        "min_all_distance":
            (
                float(
                    min(
                        all_distances
                    )
                )
                if all_distances
                else 999.0
            ),

        "min_working_distance":
            (
                float(
                    min(
                        working_distances
                    )
                )
                if working_distances
                else 999.0
            ),

        "premature_contacts":
            premature_contacts,
    }


def generate_adaptive_inward_offsets(
    far_offset,
):
    far_offset = float(
        far_offset
    )

    offsets = []

    current = (
        far_offset
        -
        APPROACH_INWARD_STEP_M
    )

    while current >= -1e-9:
        offsets.append(
            max(
                0.0,
                float(
                    current
                ),
            )
        )

        current -= (
            APPROACH_INWARD_STEP_M
        )

    if (
        not offsets
        or
        offsets[
            -1
        ]
        >
        1e-9
    ):
        offsets.append(
            0.0
        )

    return offsets


def select_late_handoff_precurve(
    solver,
    arcs,
    q_now,
    fingers,
    working_fingers,
    phases,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    """
    V9 free-space late handoff curl.

    The gross arm approach has already stopped at the last open-hand pose whose
    complete swept hand geometry is safe. We now curl the working fingers ONLY
    at that fixed arm pose. No contact is authorized here: the selected posture
    must remain strictly clear of BackSurface. The target phase ratios are kept
    so the human-wave shape is not collapsed to one equal angle.

    This is deliberately not an "open-tip standoff" rule. The decision uses
    actual whole-hand swept geometry plus a numerical surface-contact epsilon.
    """
    working = [f for f in LONG_FINGERS if f in set(working_fingers)]
    if not working:
        return {
            "commands": dict(fingers),
            "metrics": approach_surface_metrics(
                solver,
                arcs,
                q_now,
                fingers,
                working,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
            ),
            "lead_deg": 0.0,
        }

    target_phase = {
        finger: max(0.0, float(phases.get(finger, 0.0)))
        for finger in working
    }
    max_target = max(target_phase.values()) if target_phase else 0.0
    if max_target <= 1e-9:
        return None

    for lead_deg in ARC4_LATE_HANDOFF_PRECURVE_MAX_DEG:
        # Keep the original phase ordering/gaps approximately intact.  Also
        # leave a few degrees for the measured-arc acquisition stage itself.
        usable_lead = min(
            float(lead_deg),
            max(0.0, max_target - ARC4_LATE_HANDOFF_PHASE_MARGIN_DEG),
        )
        if usable_lead <= 1e-6:
            continue

        scale = min(1.0, usable_lead / max_target)
        commands = dict(fingers)
        for finger in working:
            target = target_phase[finger]
            cap = max(0.0, target - ARC4_LATE_HANDOFF_PHASE_MARGIN_DEG)
            commands[finger] = min(cap, target * scale)

        safe, reason = segment_mesh_safe(
            solver,
            arcs,
            q_now,
            fingers,
            q_now,
            commands,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=34,
            allowed_contact_fingers=set(),
        )
        if not safe:
            print(
                "    [V9 LATE HANDOFF PRECURVE REJECT]",
                "lead_deg=",
                round(usable_lead, 1),
                reason,
            )
            continue

        metrics = approach_surface_metrics(
            solver,
            arcs,
            q_now,
            commands,
            working,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )

        # The handoff precurve itself remains contact-free. Planned fingertip
        # contact becomes authorized only inside acquire_new_finger_contact,
        # one finger at a time in the declared human-wave sequence.

        print(
            "    [V9 LATE HANDOFF PRECURVE SAFE]",
            "lead_deg=",
            round(usable_lead, 1),
            "commands=",
            {f: round(float(commands[f]), 1) for f in working},
            "min_working_mm=",
            round(float(metrics["min_working_distance"]) * 1000.0, 2),
        )

        return {
            "commands": commands,
            "metrics": metrics,
            "lead_deg": usable_lead,
        }

    return None


def _scaled_four_finger_commands(base_commands, phases, working_fingers, lead_deg):
    """Scale the four measured arcs together without changing their wave ratios."""
    working = [finger for finger in LONG_FINGERS if finger in set(working_fingers)]
    max_phase = max(float(phases.get(finger, 0.0)) for finger in working)
    if max_phase <= 1e-9:
        return None
    scale = min(1.0, float(lead_deg) / max_phase)
    commands = dict(base_commands)
    for finger in working:
        commands[finger] = max(
            float(base_commands.get(finger, 0.0)),
            float(phases[finger]) * scale,
        )
    return commands


def _legacy_pose_ik_phase_compensated_handoff(
    solver,
    arcs,
    mode,
    endpoint_q,
    endpoint_palm_p,
    endpoint_palm_r,
    approach_normal,
    base_commands,
    phases,
    working_fingers,
    home_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
):
    """
    Find the precontact arm pose that belongs to the LOW finger phase.

    The old trajectory used ``endpoint_q`` for every finger phase.  That pose is
    valid only after the fingers have reached their final 65-90 degree curl; at
    12-18 degrees the four tips can be 120-140 mm away along the back.  Here the
    palm is translated by the measured change of the four-tip centroid and kept
    a small distance outside the surface.  The subsequent synchronized solver
    removes this compensation continuously as the fingers curl.
    """
    working = [finger for finger in LONG_FINGERS if finger in set(working_fingers)]
    if len(working) != 4:
        return None

    final_center = np.mean(
        [
            world_tip_position(solver, arcs, endpoint_q, phases, finger)
            for finger in working
        ],
        axis=0,
    )
    best = None
    for lead_deg in [18.0, 12.0, 24.0, 30.0]:
        commands = _scaled_four_finger_commands(
            base_commands,
            phases,
            working,
            lead_deg,
        )
        if commands is None:
            continue
        phase_center_at_endpoint = np.mean(
            [
                world_tip_position(solver, arcs, endpoint_q, commands, finger)
                for finger in working
            ],
            axis=0,
        )
        centroid_compensation = final_center - phase_center_at_endpoint

        for clearance_m in [0.020, 0.015, 0.025, 0.030]:
            target_palm = (
                np.asarray(endpoint_palm_p, dtype=np.float64)
                + centroid_compensation
                + float(clearance_m) * np.asarray(approach_normal, dtype=np.float64)
            )
            solutions = solve_lula_ik(
                solver,
                target_palm,
                endpoint_palm_r,
                [endpoint_q, home_q]
                + build_continuous_refine_seeds(
                    endpoint_q,
                    home_q,
                    lower,
                    upper,
                )
                + build_q_seeds(
                    endpoint_q,
                    home_q,
                    lower,
                    upper,
                    0.0,
                )[:55],
                lower,
                upper,
            )
            if not solutions:
                continue
            aligned_solutions = []
            for solution in solutions:
                q_solution = np.asarray(solution["q"], dtype=np.float64)
                try:
                    _, rotation_solution = solver.compute_forward_kinematics(
                        PALM_FRAME, q_solution
                    )
                except Exception:
                    continue
                alignment_deg = float(
                    palm_back_alignment(
                        arcs, rotation_solution, lane_spec["normal"]
                    )["angle_deg"]
                )
                if alignment_deg > palm_alignment_limit_deg():
                    continue
                delta_deg = np.degrees(q_solution - endpoint_q)
                continuity_cost = float(
                    np.sum(SYNC_JOINT_MOVE_WEIGHTS * np.square(delta_deg))
                )
                aligned_solutions.append((continuity_cost, q_solution))
            if not aligned_solutions:
                continue
            aligned_solutions.sort(key=lambda item: item[0])
            q_handoff = aligned_solutions[0][1]

            open_safe, _ = segment_mesh_safe(
                solver,
                arcs,
                q_handoff,
                base_commands,
                q_handoff,
                base_commands,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=2,
                allowed_contact_fingers=set(),
            )
            if not open_safe:
                continue
            precurve_safe, _ = segment_mesh_safe(
                solver,
                arcs,
                q_handoff,
                base_commands,
                q_handoff,
                commands,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=34,
                allowed_contact_fingers=set(),
            )
            if not precurve_safe:
                continue

            lane_details = {}
            lane_valid = True
            min_surface_distance = float("inf")
            for finger in working:
                relation = finger_surface_relation_at_state(
                    solver,
                    arcs,
                    q_handoff,
                    commands,
                    finger,
                    back_points,
                    back_normals,
                    back_tree,
                    back_mesh_query,
                )
                lane_state = lane_relation_for_world_point(
                    relation.get("world_point"), lane_spec, finger
                )
                min_surface_distance = min(
                    min_surface_distance, float(relation["distance"])
                )
                lane_valid = lane_valid and bool(
                    # No contact is authorized at the handoff.  Before contact,
                    # the four-finger group only has to lie over the selected
                    # finite patch; individual 8 mm lanes become mandatory at
                    # the first actual contact sample.
                    lane_state["patch_radius"] <= PLANNED_ACTIVE_PATCH_RADIUS_M
                )
                lane_details[finger] = {
                    "patch_mm": round(float(lane_state["patch_radius"]) * 1000.0, 2),
                    "cross_mm": round(float(lane_state["cross_error"]) * 1000.0, 2),
                    "surface_mm": round(float(relation["distance"]) * 1000.0, 2),
                }
            if not lane_valid:
                continue
            if min_surface_distance <= WARM_CONTACT_MAX_DISTANCE_M:
                continue

            score = (
                float(np.linalg.norm(q_handoff - endpoint_q))
                + 0.02 * abs(float(lead_deg) - 18.0)
                + 2.0 * float(clearance_m)
            )
            candidate = {
                "q": q_handoff,
                "commands": commands,
                "lead_deg": float(lead_deg),
                "clearance_m": float(clearance_m),
                "final_center": np.asarray(final_center, dtype=np.float64),
                "lane_details": lane_details,
                "score": float(score),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    return best


def _solve_phase_compensated_handoff_band(
    solver,
    arcs,
    mode,
    endpoint_q,
    endpoint_palm_p,
    endpoint_palm_r,
    approach_normal,
    base_commands,
    phases,
    working_fingers,
    home_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
    prebend_lower_deg,
    prebend_upper_deg,
):
    """Joint-space handoff solve on the real 5-DOF reachable manifold.

    Unlike the retired pose-IK version, this never freezes the endpoint palm
    rotation and translates it to a new position.  J1-J5, the low finger phase
    and clearance are optimized together, including base/root collinearity.
    """
    working = [finger for finger in LONG_FINGERS if finger in set(working_fingers)]
    if len(working) != 4:
        return None

    endpoint_q = np.asarray(endpoint_q, dtype=np.float64)
    endpoint_palm_p = np.asarray(endpoint_palm_p, dtype=np.float64)
    endpoint_palm_r = np.asarray(endpoint_palm_r, dtype=np.float64)
    approach_normal = np.asarray(approach_normal, dtype=np.float64)
    final_local_points = {
        finger: interpolate_arc(arcs, finger, float(phases[finger]))[0]
        for finger in working
    }
    final_center = np.mean(
        [endpoint_palm_p + endpoint_palm_r @ final_local_points[f] for f in working],
        axis=0,
    )

    def decode(x):
        x = np.asarray(x, dtype=np.float64)
        q = x[:5]
        lead_deg = float(x[5])
        clearance_m = float(x[6])
        commands = _scaled_four_finger_commands(
            base_commands, phases, working, lead_deg
        )
        return q, lead_deg, clearance_m, commands

    def objective(x):
        q, lead_deg, clearance_m, commands = decode(x)
        if commands is None:
            return 1e12
        try:
            palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        except Exception:
            return 1e12
        palm_p = np.asarray(palm_p, dtype=np.float64)
        palm_r = np.asarray(palm_r, dtype=np.float64)

        low_local_points = {
            finger: interpolate_arc(arcs, finger, float(commands[finger]))[0]
            for finger in working
        }
        phase_center_at_endpoint = np.mean(
            [endpoint_palm_p + endpoint_palm_r @ low_local_points[f] for f in working],
            axis=0,
        )
        target_palm = (
            endpoint_palm_p
            + (final_center - phase_center_at_endpoint)
            + clearance_m * approach_normal
        )
        position_error = float(np.linalg.norm(palm_p - target_palm))
        palm_angle = float(
            palm_back_alignment(arcs, palm_r, lane_spec["normal"])["angle_deg"]
        )
        root_axis_angle = float(
            root_base_axis_alignment(arcs, palm_p, palm_r)["angle_deg"]
        )
        height_order = finger_arc_height_order(
            arcs,
            palm_r,
            lane_spec["normal"],
            phases,
            working,
            intermediate_commands=commands,
        )

        lane_cost = 0.0
        clearance_cost = 0.0
        for finger in working:
            world_point = palm_p + palm_r @ low_local_points[finger]
            lane = lane_relation_for_world_point(world_point, lane_spec, finger)
            lane_cost += (
                float(lane["patch_radius"])
                / max(PLANNED_ACTIVE_PATCH_RADIUS_M, 1e-6)
            ) ** 2
            distance, _, _, _ = nearest_local(mode, world_point)
            clearance_cost += (
                max(0.0, WARM_CONTACT_MAX_DISTANCE_M + 0.002 - float(distance))
                / 0.003
            ) ** 2

        delta_deg = np.degrees(q - endpoint_q)
        return float(
            12.0 * (position_error / 0.006) ** 2
            + PALM_ALIGNMENT_WEIGHT * (palm_angle / palm_alignment_scale_deg()) ** 2
            + ROOT_BASE_AXIS_WEIGHT
            * (root_axis_angle / ROOT_BASE_AXIS_SCALE_DEG) ** 2
            + ARC_HEIGHT_ORDER_WEIGHT
            * (
                float(height_order["max_violation_m"])
                / max(ARC_HEIGHT_ORDER_TOL_M, 1e-6)
            ) ** 2
            + 3.0 * lane_cost
            + 20.0 * clearance_cost
            + 0.0008 * np.sum(SYNC_JOINT_MOVE_WEIGHTS * np.square(delta_deg))
            + 2.0
            * (lead_deg / HANDOFF_PREBEND_PREFERENCE_SCALE_DEG) ** 2
            + 0.5 * ((clearance_m - 0.020) / 0.010) ** 2
        )

    lower_x = np.concatenate([lower, [float(prebend_lower_deg), 0.015]])
    upper_x = np.concatenate([upper, [float(prebend_upper_deg), 0.030]])
    q_seeds = build_continuous_refine_seeds(
        endpoint_q, home_q, lower, upper
    )
    # Endpoint, home and several independent full-joint configurations are
    # numerical starts only; every one is optimized over all five joints.
    lead_seed_deg = min(
        float(prebend_upper_deg),
        float(prebend_lower_deg) + 2.0,
    )
    seed_x_values = [
        np.concatenate([q_seed, [lead_seed_deg, 0.020]])
        for q_seed in q_seeds[:6]
    ]

    best = None
    reject_buckets = {}
    reject_examples = {}
    for seed_x in seed_x_values:
        result = minimize(
            objective,
            seed_x,
            method="Powell",
            bounds=list(zip(lower_x, upper_x)),
            options={
                "maxiter": 64,
                "xtol": 1e-4,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        q_handoff, lead_deg, clearance_m, commands = decode(result.x)
        if commands is None:
            continue
        try:
            palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q_handoff)
        except Exception:
            continue
        palm_angle = float(
            palm_back_alignment(arcs, palm_r, lane_spec["normal"])["angle_deg"]
        )
        root_axis_angle = float(
            root_base_axis_alignment(arcs, palm_p, palm_r)["angle_deg"]
        )
        if palm_angle > palm_alignment_limit_deg():
            reject_buckets["PALM_ALIGNMENT"] = reject_buckets.get("PALM_ALIGNMENT", 0) + 1
            continue
        if root_axis_angle > ROOT_BASE_AXIS_HARD_DEG:
            reject_buckets["ROOT_BASE_AXIS_ALIGNMENT"] = reject_buckets.get("ROOT_BASE_AXIS_ALIGNMENT", 0) + 1
            continue
        height_order = finger_arc_height_order(
            arcs,
            palm_r,
            lane_spec["normal"],
            phases,
            working,
            intermediate_commands=commands,
        )
        if not bool(height_order["valid"]):
            reject_buckets["ARC_HEIGHT_ORDER"] = reject_buckets.get("ARC_HEIGHT_ORDER", 0) + 1
            reject_examples.setdefault("ARC_HEIGHT_ORDER", str(height_order["finger"]))
            continue

        open_safe, open_reason = segment_mesh_safe(
            solver, arcs, q_handoff, base_commands, q_handoff, base_commands,
            back_points, back_normals, back_tree, back_mesh_query,
            samples=2, allowed_contact_fingers=set(),
        )
        if not open_safe:
            reject_buckets["OPEN_COLLISION"] = reject_buckets.get("OPEN_COLLISION", 0) + 1
            reject_examples.setdefault("OPEN_COLLISION", str(open_reason))
            continue
        precurve_safe, precurve_reason = segment_mesh_safe(
            solver, arcs, q_handoff, base_commands, q_handoff, commands,
            back_points, back_normals, back_tree, back_mesh_query,
            samples=34, allowed_contact_fingers=set(),
        )
        if not precurve_safe:
            reject_buckets["PRECURVE_COLLISION"] = reject_buckets.get("PRECURVE_COLLISION", 0) + 1
            reject_examples.setdefault("PRECURVE_COLLISION", str(precurve_reason))
            continue

        lane_details = {}
        min_surface_distance = float("inf")
        lane_valid = True
        for finger in working:
            relation = finger_surface_relation_at_state(
                solver, arcs, q_handoff, commands, finger,
                back_points, back_normals, back_tree, back_mesh_query,
            )
            lane = lane_relation_for_world_point(
                relation.get("world_point"), lane_spec, finger
            )
            min_surface_distance = min(
                min_surface_distance, float(relation["distance"])
            )
            lane_valid = lane_valid and bool(
                lane["patch_radius"] <= PLANNED_ACTIVE_PATCH_RADIUS_M
            )
            lane_details[finger] = {
                "patch_mm": round(float(lane["patch_radius"]) * 1000.0, 2),
                "cross_mm": round(float(lane["cross_error"]) * 1000.0, 2),
                "surface_mm": round(float(relation["distance"]) * 1000.0, 2),
            }
        if not lane_valid:
            reject_buckets["TARGET_PATCH"] = reject_buckets.get("TARGET_PATCH", 0) + 1
            continue
        if min_surface_distance <= HANDOFF_PREBEND_MIN_CLEARANCE_M:
            reject_buckets["PREBEND_CLEARANCE"] = reject_buckets.get(
                "PREBEND_CLEARANCE", 0
            ) + 1
            continue

        candidate = {
            "q": np.asarray(q_handoff, dtype=np.float64),
            "commands": commands,
            "lead_deg": lead_deg,
            "clearance_m": clearance_m,
            "final_center": np.asarray(final_center, dtype=np.float64),
            "lane_details": lane_details,
            "palm_angle_deg": palm_angle,
            "root_base_axis_angle_deg": root_axis_angle,
            "score": float(objective(result.x)),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate

    print(
        "    [V101 JOINT-SPACE HANDOFF BAND]",
        "prebend_deg=",
        (round(float(prebend_lower_deg), 1), round(float(prebend_upper_deg), 1)),
        "status=", "PASS" if best is not None else "FAIL",
        "rejects=", reject_buckets,
        "examples=", reject_examples,
        "q=", None if best is None else np.round(np.degrees(best["q"]), 2),
        "palm_deg=", None if best is None else round(best["palm_angle_deg"], 2),
        "root_base_axis_deg=", None if best is None else round(best["root_base_axis_angle_deg"], 2),
    )
    return best


def solve_phase_compensated_handoff(
    solver,
    arcs,
    mode,
    endpoint_q,
    endpoint_palm_p,
    endpoint_palm_r,
    approach_normal,
    base_commands,
    phases,
    working_fingers,
    home_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
):
    """Return the first safe handoff from progressively larger pre-bend bands.

    This makes pre-bend a minimum-feasible quantity rather than a free global
    optimizer choice.  A solution in a lower band prevents every larger band
    from being searched or selected.
    """
    for prebend_lower_deg, prebend_upper_deg in HANDOFF_PREBEND_SEARCH_BANDS_DEG:
        candidate = _solve_phase_compensated_handoff_band(
            solver,
            arcs,
            mode,
            endpoint_q,
            endpoint_palm_p,
            endpoint_palm_r,
            approach_normal,
            base_commands,
            phases,
            working_fingers,
            home_q,
            lower,
            upper,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            lane_spec,
            prebend_lower_deg,
            prebend_upper_deg,
        )
        if candidate is not None:
            candidate["prebend_search_band_deg"] = (
                float(prebend_lower_deg),
                float(prebend_upper_deg),
            )
            print(
                "    [V101 MINIMUM-FEASIBLE PREBEND]",
                "selected_band_deg=",
                candidate["prebend_search_band_deg"],
                "selected_lead_deg=",
                round(float(candidate["lead_deg"]), 2),
                "minimum_clearance_mm=",
                round(HANDOFF_PREBEND_MIN_CLEARANCE_M * 1000.0, 1),
            )
            return candidate

    print(
        "    [V101 MINIMUM-FEASIBLE PREBEND]",
        "status=FAIL",
        "searched_bands_deg=",
        HANDOFF_PREBEND_SEARCH_BANDS_DEG,
    )
    return None


def continuous_phase_joint_candidates(
    solver,
    arcs,
    q_previous,
    linear_seed,
    target_palm,
    finger_commands,
    endpoint_phases,
    working_fingers,
    lower,
    upper,
    lane_spec,
):
    """Solve one acquisition step without imposing a fixed wrist yaw.

    Position, palm/back parallelism, four-finger region fit and joint
    continuity are optimized together.  J1-J3 are available but expensive;
    J4/J5 are the preferred local adjustment.  Several nearby minima are
    returned so the caller can select a collision-free transition.
    """
    q_previous = np.asarray(q_previous, dtype=np.float64)
    linear_seed = np.asarray(linear_seed, dtype=np.float64)
    target_palm = np.asarray(target_palm, dtype=np.float64)
    local_delta = np.radians(SYNC_LOCAL_DELTA_DEG)
    local_lower = np.maximum(lower, q_previous - local_delta)
    local_upper = np.minimum(upper, q_previous + local_delta)
    linear_seed = np.minimum(np.maximum(linear_seed, local_lower), local_upper)
    back_normal = np.asarray(lane_spec["normal"], dtype=np.float64)
    anchored_acquisition = lane_spec.get('anchor_finger') in working_fingers
    # Finger commands and arcs are fixed for this solve. Only the arm pose
    # changes in Powell's objective, so interpolate each tip once per step.
    local_tips = {finger: interpolate_arc(
        arcs, finger, float(finger_commands.get(finger, 0.0)))[0]
        for finger in working_fingers}
    objective_cache = {}
    objective_calls = 0
    objective_hits = 0
    solve_started = time.perf_counter()
    def reference_error(palm_p, palm_r):
        reference = lane_spec.get('acquisition_reference_world')
        if anchored_acquisition and reference is not None:
            finger = lane_spec['anchor_finger']
            tip = np.asarray(palm_p)+np.asarray(palm_r) @ local_tips[finger]
            return float(np.linalg.norm(tip-np.asarray(reference)))
        return float(np.linalg.norm(np.asarray(palm_p)-target_palm))

    def compute_objective(q):
        q = np.asarray(q, dtype=np.float64)
        try:
            palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        except Exception:
            return 1e12
        palm_p = np.asarray(palm_p, dtype=np.float64)
        palm_r = np.asarray(palm_r, dtype=np.float64)
        position_error = reference_error(palm_p,palm_r)
        alignment_deg = float(
            palm_back_alignment(arcs, palm_r, back_normal)["angle_deg"]
        )
        root_axis_deg = float(
            root_base_axis_alignment(arcs, palm_p, palm_r)["angle_deg"]
        )
        height_order = {"max_violation_m": 0.0} if anchored_acquisition else finger_arc_height_order(
            arcs,
            palm_r,
            back_normal,
            {
                finger: float(endpoint_phases.get(finger, 0.0))
                for finger in working_fingers
            },
            working_fingers,
            intermediate_commands={
                finger: float(finger_commands.get(finger, 0.0))
                for finger in working_fingers
            },
        )

        lane_cost = 0.0
        for finger in working_fingers:
            local_point = local_tips[finger]
            world_point = palm_p + palm_r @ local_point
            lane = lane_relation_for_world_point(world_point, lane_spec, finger)
            lane_cost += 1.6 * (
                float(lane["patch_radius"])
                / max(PLANNED_ACTIVE_PATCH_RADIUS_M, 1e-6)
            ) ** 2
            lane_cost += 1.0 * (
                float(lane["cross_error"])
                / max(PLANNED_ACTIVE_LANE_CROSS_M, 1e-6)
            ) ** 2

        continuity_deg = np.degrees(q - q_previous)
        path_deg = np.degrees(q - linear_seed)
        return float(
            12.0 * (position_error / 0.004) ** 2
            + (5.0 * (max(0., alignment_deg-(palm_alignment_limit_deg()-.2)) / 2.0) ** 2
               if anchored_acquisition else 5.0 * (alignment_deg / 12.0) ** 2)
            + ROOT_BASE_AXIS_WEIGHT
            * (root_axis_deg / ROOT_BASE_AXIS_SCALE_DEG) ** 2
            + (0.0 if anchored_acquisition else ARC_HEIGHT_ORDER_WEIGHT)
            * (
                float(height_order["max_violation_m"])
                / max(ARC_HEIGHT_ORDER_TOL_M, 1e-6)
            ) ** 2
            + lane_cost
            + 0.004 * np.sum(SYNC_JOINT_MOVE_WEIGHTS * np.square(continuity_deg))
            + 0.001 * np.sum(np.square(path_deg))
        )

    def objective(q):
        nonlocal objective_calls, objective_hits
        objective_calls += 1
        # Exact float64 bytes preserve optimizer finite differences. The cache
        # expires with this step and never caches physical/swept acceptance.
        key = np.asarray(q, dtype=np.float64).tobytes()
        if key in objective_cache:
            objective_hits += 1
            return objective_cache[key]
        value = compute_objective(q)
        if len(objective_cache) >= 4096:
            objective_cache.clear()
        if np.isfinite(value) and value < 1e12:
            objective_cache[key] = value
        return value

    seeds = [q_previous.copy(), linear_seed.copy()]
    for joint_index, delta_deg in ((3, -6.0), (3, 6.0), (4, -8.0), (4, 8.0)):
        seed = linear_seed.copy()
        seed[joint_index] += math.radians(delta_deg)
        seeds.append(np.minimum(np.maximum(seed, local_lower), local_upper))

    ranked = []
    hard_rejects = {
        "FK": 0,
        "POSITION": 0,
        "PALM_ALIGNMENT": 0,
        "ROOT_BASE_AXIS_ALIGNMENT": 0,
        "ARC_HEIGHT_ORDER": 0,
    }
    best_observed = {
        "position_mm": float("inf"),
        "palm_deg": float("inf"),
        "root_base_axis_deg": float("inf"),
        "arc_height_violation_mm": float("inf"),
    }
    seen = set()
    for seed in seeds:
        result = minimize(
            objective,
            seed,
            method="Powell",
            bounds=list(zip(local_lower, local_upper)),
            options={
                "maxiter": ACQUIRE_MAXITER,
                "xtol": 1e-4,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        q = np.asarray(result.x, dtype=np.float64)
        key = tuple(np.round(np.degrees(q), 2))
        if key in seen:
            continue
        seen.add(key)
        try:
            palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        except Exception:
            hard_rejects["FK"] += 1
            continue
        position_error = reference_error(palm_p,palm_r)
        alignment_deg = float(
            palm_back_alignment(arcs, palm_r, back_normal)["angle_deg"]
        )
        root_axis_deg = float(
            root_base_axis_alignment(arcs, palm_p, palm_r)["angle_deg"]
        )
        height_order = finger_arc_height_order(
            arcs,
            palm_r,
            back_normal,
            {
                finger: float(endpoint_phases.get(finger, 0.0))
                for finger in working_fingers
            },
            working_fingers,
            intermediate_commands={
                finger: float(finger_commands.get(finger, 0.0))
                for finger in working_fingers
            },
        )
        best_observed["position_mm"] = min(
            best_observed["position_mm"], position_error * 1000.0
        )
        best_observed["palm_deg"] = min(best_observed["palm_deg"], alignment_deg)
        best_observed["root_base_axis_deg"] = min(
            best_observed["root_base_axis_deg"], root_axis_deg
        )
        best_observed["arc_height_violation_mm"] = min(
            best_observed["arc_height_violation_mm"],
            float(height_order["max_violation_m"]) * 1000.0,
        )
        rejected = False
        if position_error > 0.008:
            hard_rejects["POSITION"] += 1
            rejected = True
        if alignment_deg > palm_alignment_limit_deg():
            hard_rejects["PALM_ALIGNMENT"] += 1
            rejected = True
        if root_axis_deg > ROOT_BASE_AXIS_HARD_DEG:
            hard_rejects["ROOT_BASE_AXIS_ALIGNMENT"] += 1
            rejected = True
        if not anchored_acquisition and not bool(height_order["valid"]):
            hard_rejects["ARC_HEIGHT_ORDER"] += 1
            rejected = True
        if rejected:
            continue
        ranked.append((float(objective(q)), q))

    ranked.sort(key=lambda item: item[0])
    print('[V103 ACQUISITION SOLVE PROFILE]',
          'seconds=', round(time.perf_counter()-solve_started, 3),
          'objective_calls=', objective_calls, 'exact_cache_hits=', objective_hits,
          'valid_branches=', len(ranked), flush=True)
    return [q for _, q in ranked], hard_rejects, best_observed


def evaluate_sync_acquisition_candidate(
    solver,
    arcs,
    q_previous,
    commands_previous,
    q_candidate,
    commands_candidate,
    working,
    active_previous,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
    acquisition_focus=None,
    forbid_unfocused_entering=False,
):
    # Use the same patch normal and angular bounds as the contact rollout.
    # Collision clearance alone does not make a valid contact handoff.
    palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q_candidate)
    angle = float(palm_back_alignment(arcs, palm_r, lane_spec['normal'])['angle_deg'])
    root_angle = float(root_base_axis_alignment(arcs, palm_p, palm_r)['angle_deg'])
    if not math.isfinite(angle) or angle > palm_alignment_limit_deg():
        return None, 'ACQUISITION_PALM_ALIGNMENT:%.3f>%.3f' % (angle, palm_alignment_limit_deg())
    if not math.isfinite(root_angle) or root_angle > ROOT_BASE_AXIS_HARD_DEG:
        return None, 'ACQUISITION_ROOT_ALIGNMENT:%.3f>%.3f' % (root_angle, ROOT_BASE_AXIS_HARD_DEG)
    active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)
    # Physical skin-contact authorization is deliberately separate from
    # completion ownership. A later follower may touch the BackSurface before
    # its frozen turn, and it remains a legitimate planned contact even if its
    # reserve controller holds its command for one frame. Completion/handoff
    # is still owned only by ``acquisition_focus``.
    moving_fingers = {
        finger for finger in working
        if float(commands_candidate.get(finger, 0.0))
           > float(commands_previous.get(finger, 0.0)) + 1e-9
    }
    touching_fingers = set()
    relations = {}
    entering = []
    for finger in working:
        # Authorize contact at BOTH ends of the swept segment. A follower may
        # have been touching at q_previous and be curling/lifting away at the
        # candidate; without this union the dense sweep can reject alpha=0 as
        # an "inactive" collision even though the previous accepted state was
        # already a planned skin contact.
        previous_relation = finger_surface_relation_at_state(
            solver, arcs, q_previous, commands_previous, finger,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        previous_lane = lane_relation_for_world_point(
            previous_relation.get("world_point"), lane_spec, finger
        )
        previous_geometric_contact = bool(
            float(previous_relation["distance"]) <= WARM_CONTACT_MAX_DISTANCE_M
            and WARM_CONTACT_RAW_MIN_M <= float(previous_relation["raw_signed"]) <= WARM_CONTACT_RAW_MAX_M
            and float(previous_lane["cross_error"]) <= active_cross_limit
            and float(previous_lane["patch_radius"]) <= active_radius_limit
        )
        if previous_geometric_contact:
            touching_fingers.add(finger)

        relation = finger_surface_relation_at_state(
            solver, arcs, q_candidate, commands_candidate, finger,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        lane_state = lane_relation_for_world_point(
            relation.get("world_point"), lane_spec, finger
        )
        raw_signed = float(relation["raw_signed"])
        safety_signed = float(relation["signed"])
        distance = float(relation["distance"])
        region_valid = bool(
            lane_state["cross_error"] <= active_cross_limit
            and lane_state["patch_radius"] <= active_radius_limit
        )
        if safety_signed < HARD_PENETRATION_M:
            return None, (
                "finger=%s reason=HARD_PENETRATION signed_mm=%.2f"
                % (finger, safety_signed * 1000.0)
            )
        if safety_signed < -v103_effective_runtime_depth_cap_m() - 1e-12:
            return None, (
                "finger=%s reason=ACQUISITION_PLANNED_DEPTH_CAP depth_mm=%.2f"
                % (finger, -safety_signed * 1000.0)
            )
        geometric_contact = bool(
            distance <= WARM_CONTACT_MAX_DISTANCE_M
            and WARM_CONTACT_RAW_MIN_M <= raw_signed <= WARM_CONTACT_RAW_MAX_M
        )
        # A future follower that is still on the air side of the BackSurface
        # may approach from outside the active contact lane.  Treating a
        # +0.5 mm airborne follower as already committed contact caused 909 to
        # fail TARGET_REGION before its frozen turn.  The current owner, an
        # already-active finger, or any actual penetration still uses the
        # unchanged active lane/patch limits.
        region_committed = bool(
            geometric_contact
            and (raw_signed <= 0.0
                 or finger == acquisition_focus
                 or finger in active_previous)
        )
        if region_committed and not region_valid:
            # A non-focus follower can cross the surface before its locked
            # handoff turn because all four curl channels run continuously.
            # Do not reject that harmless early overlap against the *active*
            # lane width; require only the finite hard patch/lane envelope.
            # The focus and already-active fingers retain the tighter limits.
            follower_hard_region = bool(
                finger not in active_previous and finger != acquisition_focus
                and float(lane_state["cross_error"]) <= TARGET_LANE_CROSS_HARD_M
                and float(lane_state["patch_radius"]) <= TARGET_PATCH_RADIUS_HARD_M
            )
            if not follower_hard_region:
                return None, (
                    "finger=%s reason=TARGET_REGION patch_mm=%.2f cross_mm=%.2f"
                    % (
                        finger,
                        float(lane_state["patch_radius"]) * 1000.0,
                        float(lane_state["cross_error"]) * 1000.0,
                    )
                )
        if geometric_contact and region_valid:
            touching_fingers.add(finger)
        if finger in active_previous and not (geometric_contact and region_valid):
            return None, "finger=%s reason=CONTACT_LOST_DURING_JOIN" % finger
        # During the measured handoff only the current dynamic focus may
        # commit a new contact.  A later finger can be geometrically close at
        # the same wrist pose, but latching it early recreates the old
        # ring->middle overlap.  ``None`` preserves the legacy behavior for
        # isolated callers/tests that do not request serial acquisition.
        if (geometric_contact and region_valid and finger not in active_previous
                and (not bool(forbid_unfocused_entering)
                     or acquisition_focus is None
                     or finger == acquisition_focus)
                and (acquisition_focus is None or finger == acquisition_focus)):
            entering.append(finger)
        relations[finger] = {
            "patch_mm": round(float(lane_state["patch_radius"]) * 1000.0, 2),
            "cross_mm": round(float(lane_state["cross_error"]) * 1000.0, 2),
            "distance_mm": round(distance * 1000.0, 2),
            "raw_mm": round(raw_signed * 1000.0, 2),
        }

    next_active = list(active_previous)
    for finger in entering:
        if finger not in next_active:
            next_active.append(finger)
    # A follower that is already physically touching must stay authorized for
    # the WHOLE swept segment even if its command is held by the depth-reserve
    # controller. This fixes the alpha=0 self-rejection where a follower was
    # legal while moving, then became an "inactive collision" one frame later.
    authorized = (
        set(active_previous)
        | set(entering)
        | set(moving_fingers)
        | set(touching_fingers)
    )
    transition_safe, transition_reason = segment_mesh_safe(
        solver, arcs, q_previous, commands_previous,
        q_candidate, commands_candidate,
        back_points, back_normals, back_tree, back_mesh_query,
        samples=24,
        allowed_contact_fingers=authorized,
    )
    if not transition_safe:
        return None, transition_reason
    return {
        "relations": relations,
        "entering": entering,
        "next_active": next_active,
        "authorized": authorized,
    }, None


def local_acquisition_joint_candidates(
    solver, arcs, nominal_q, previous_q, commands, working, active_previous,
    lower, upper, back_points, back_normals, back_tree, back_mesh_query, lane_spec,
):
    """Lazy fallback: wrist grid first, bounded five-axis repair only if needed.

    These are proposals, NOT accepted motions. The caller must pass each through
    evaluate_sync_acquisition_candidate (including the swept collision check).
    Never relax the 8 mm planned lane / penetration / contact-retention gates.
    """
    active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)
    nominal_q = np.asarray(nominal_q, dtype=np.float64)
    previous_q = np.asarray(previous_q, dtype=np.float64)
    seen = set()
    offsets = sorted(
        itertools.product((0.0, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0), repeat=2),
        key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[1])),
    )
    for j4, j5 in offsets:
        q = nominal_q.copy()
        q[3:] += np.radians([j4, j5])
        if np.any(q < lower) or np.any(q > upper):
            continue
        if np.max(np.abs(np.degrees(q - previous_q))) > 3.0:
            continue
        key = tuple(np.round(q, 8))
        if key in seen:
            continue
        seen.add(key)
        yield "J4_J5_LOCAL", q

    span = np.radians([1.5, 1.5, 1.5, 3.0, 3.0])
    lo = np.maximum(np.maximum(lower, nominal_q - span), previous_q - span)
    hi = np.minimum(np.minimum(upper, nominal_q + span), previous_q + span)
    if np.any(lo >= hi):
        return

    base_guard = getattr(HAND_COLLISION_MODEL, 'base_guard', None)
    near_base = False
    if (base_guard is not None
            and not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))):
        nominal_gap, _ = base_guard.clearance(nominal_q, commands)
        near_base = math.isfinite(float(nominal_gap)) and nominal_gap < V103_OPTIONAL_PINKY_BASE_ZONE_M

    def objective(q):
        delta = np.degrees(q - nominal_q)
        cost = float(np.dot(delta * delta, [8.0, 8.0, 8.0, 0.3, 0.3]))
        palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        angle = palm_back_alignment(arcs, palm_r, lane_spec['normal'])['angle_deg']
        root_angle = root_base_axis_alignment(arcs, palm_p, palm_r)['angle_deg']
        cost += 1000. * (max(0., angle-(palm_alignment_limit_deg()-.2))**2
                        + max(0., root_angle-(ROOT_BASE_AXIS_HARD_DEG-.2))**2)
        # The old bounded five-axis repair optimized only lane/depth and often
        # returned the nominal branch that was already 0.01-0.5 mm inside the
        # base guard.  Give the optimizer a small buffer above the 3 mm
        # hard gate so J1-J3 can actually steer the acquisition around base.
        if near_base and not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False)):
            try:
                base_gap, _ = base_guard.clearance(q, commands)
                if not math.isfinite(float(base_gap)):
                    return 1e12
                base_shortfall = max(0.0, V103_BASE_MIN_CLEARANCE_M + 0.002 - float(base_gap))
                cost += 600.0 * (base_shortfall * 1000.0) ** 2
            except Exception:
                return 1e12
        for finger in working:
            relation = finger_surface_relation_at_state(
                solver, arcs, q, commands, finger,
                back_points, back_normals, back_tree, back_mesh_query,
            )
            lane = lane_relation_for_world_point(relation["world_point"], lane_spec, finger)
            signed = float(relation["raw_signed"])
            distance = float(relation["distance"])
            # Guide near-contact points slightly inside the existing planning
            # boundary. Far-away open fingers need not occupy their lanes yet.
            if distance <= 0.012 or finger in active_previous:
                cross = max(0.0, float(lane["cross_error"]) - (active_cross_limit - 0.001))
                radius = max(0.0, float(lane["patch_radius"]) - (active_radius_limit - 0.001))
                cost += 200.0 * ((cross * 1000.0) ** 2 + (radius * 1000.0) ** 2)
            depth_excess = max(0.0, HARD_PENETRATION_M - float(relation["signed"]))
            cost += 1000.0 * (depth_excess * 1000.0) ** 2
            if finger in active_previous:
                gap = max(0.0, signed - WARM_CONTACT_RAW_MAX_M)
                lost = max(0.0, distance - WARM_CONTACT_MAX_DISTANCE_M)
                cost += 500.0 * ((gap * 1000.0) ** 2 + (lost * 1000.0) ** 2)
        return cost

    seeds = [previous_q, nominal_q]
    # Explicit arm-link seeds are important near base: wrist-only offsets can
    # move the fingertip yet leave the colliding proximal/distal link in place.
    for joint_index in (range(3) if near_base else ()):
        for delta_deg in (-1.5, 1.5):
            seed = nominal_q.copy()
            seed[joint_index] += math.radians(delta_deg)
            seeds.append(np.clip(seed, lo, hi))
    for seed in seeds:
        result = minimize(
            objective, np.clip(seed, lo, hi), method="Powell",
            bounds=list(zip(lo, hi)),
            options={"maxiter": 20, "maxfev": 160, "xtol": 1e-4, "ftol": 1e-3},
        )
        if np.all(np.isfinite(result.x)):
            yield "BOUNDED_J1_J5_LOCAL", np.clip(result.x, lo, hi)


def v103_acquisition_handoff(solver, arcs, q0, f0, q1, f1, working,
                              back_points, back_normals, back_tree, back_mesh_query,
                              owner_finger=None):
    """Clip at the FIRST FROZEN OWNER's first real contact-depth crossing.

    Acquisition is only a staging-to-wave bridge. Later followers may already
    be touching, but they never trigger this handoff and they never earn
    completion out of order. The normal contact-wave controller owns the full
    2--4 mm mandatory quality band.
    """
    owner = owner_finger if owner_finger in working else (working[0] if working else None)
    if owner is None:
        return np.asarray(q1), dict(f1), False

    def state(t):
        q = np.asarray(q0) + t*(np.asarray(q1)-np.asarray(q0))
        f = {name: float(f0[name])+t*(float(f1[name])-float(f0[name])) for name in LONG_FINGERS}
        p,r = solver.compute_forward_kinematics(PALM_FRAME,q)
        tip = np.asarray(p) + np.asarray(r) @ interpolate_arc(arcs, owner, f[owner])[0]
        relation = final_surface_relation(
            tip, back_mesh_query, back_points, back_normals, back_tree)
        depth = max(0.0, -float(relation['signed']))
        return q, f, depth

    previous = 0.
    for t in np.linspace(0.,1.,25):
        q,f,depth = state(t)
        if depth >= V103_ACQUISITION_WAVE_ENTRY_DEPTH_M:
            lo,hi = previous,float(t)
            for _ in range(14):
                mid = (lo+hi)/2
                if state(mid)[2] >= V103_ACQUISITION_WAVE_ENTRY_DEPTH_M:
                    hi = mid
                else:
                    lo = mid
            q,f,_ = state(hi)
            return q,f,True
        previous = float(t)
    return np.asarray(q1),dict(f1),False

def _wave_surface_error_m(relation):
    """Return the measured fingertip error used for dynamic ordering.

    Outside the BackSurface we use the unsigned clearance; once a tip is on
    the surface we use positive penetration depth.  This keeps the scheduler
    continuous across first contact and avoids treating a signed normal from a
    distant tangent approximation as a real depth measurement.
    """
    signed = float(relation.get("signed", float("nan")))
    distance = float(relation.get("distance", float("nan")))
    if not np.isfinite(signed) or not np.isfinite(distance):
        return float("inf")
    return max(0.0, -signed) if signed <= 0.0 else max(0.0, distance)


def _dynamic_wave_schedule(
    solver,
    arcs,
    q,
    commands,
    working,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    tie_order=None,
    max_delay_deg=ASYNC_WAVE_DYNAMIC_MAX_DELAY_DEG,
):
    """Measure a fresh order/delay schedule from the real BackSurface.

    The one-command finite difference is only guidance.  The caller still
    validates every proposed segment against the complete mesh and base guard.
    """
    tie_order = list(tie_order or working)
    tie_rank = {finger: index for index, finger in enumerate(tie_order)}
    measured = {}
    for finger in working:
        base = finger_surface_relation_at_state(
            solver, arcs, q, commands, finger,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        metric = _wave_surface_error_m(base)
        probe_commands = dict(commands)
        current = float(commands.get(finger, 0.0))
        probe = min(float(SCRATCH_FULL_CURL_DEG), current + ASYNC_WAVE_DYNAMIC_PROBE_DEG)
        probe_delta = probe - current
        sensitivity = 0.0
        signed_sensitivity = 0.0
        if probe_delta > 1e-9:
            probe_commands[finger] = probe
            probe_relation = finger_surface_relation_at_state(
                solver, arcs, q, probe_commands, finger,
                back_points, back_normals, back_tree, back_mesh_query,
            )
            sensitivity = abs(_wave_surface_error_m(probe_relation) - metric) / probe_delta
            base_signed = float(base.get("signed", float("nan")))
            probe_signed = float(probe_relation.get("signed", float("nan")))
            if np.isfinite(base_signed) and np.isfinite(probe_signed):
                signed_sensitivity = (probe_signed - base_signed) / probe_delta
        sensitivity = max(
            float(sensitivity),
            ASYNC_WAVE_DYNAMIC_MIN_SENSITIVITY_M_PER_DEG,
        )
        measured[finger] = dict(
            metric_m=metric,
            signed_m=float(base.get("signed", float("nan"))),
            distance_m=float(base.get("distance", float("nan"))),
            sensitivity_m_per_deg=sensitivity,
            # Negative means more curl deepens the fingertip; positive means
            # the follower is already curling away from the BackSurface.
            signed_sensitivity_m_per_deg=float(signed_sensitivity),
        )
    finite_metrics = [row["metric_m"] for row in measured.values() if np.isfinite(row["metric_m"])]
    reference = min(finite_metrics) if finite_metrics else 0.0
    for finger, row in measured.items():
        error_m = max(0.0, float(row["metric_m"]) - reference) if np.isfinite(row["metric_m"]) else float(max_delay_deg) * float(ASYNC_WAVE_DYNAMIC_MIN_SENSITIVITY_M_PER_DEG)
        row["error_m"] = error_m
        row["delay_deg"] = min(float(max_delay_deg), error_m / row["sensitivity_m_per_deg"])
    order = sorted(
        working,
        key=lambda finger: (
            measured[finger]["metric_m"] if np.isfinite(measured[finger]["metric_m"]) else float("inf"),
            tie_rank.get(finger, len(tie_rank)),
        ),
    )
    return order, {finger: measured[finger]["delay_deg"] for finger in working}, measured


def _adaptive_acquisition_step_deg(schedule_row, target_depth_m,
                                   phase_remaining_deg,
                                   nominal_step_deg=ARC_CONTACT_PHASE_STEP_DEG):
    """Choose one bounded contact-acquisition increment from live geometry.

    ``schedule_row`` is measured at the current pose.  Outside the surface,
    the remaining distance is the clearance plus the preferred target depth;
    once inside, it is the target depth minus the measured penetration.  The
    one-step FK sensitivity is only a local predictor, so the result is
    deliberately multiplied by a fraction and clipped.  The full swept
    evaluator remains authoritative after this proposal.
    """
    remaining_phase = max(0.0, float(phase_remaining_deg))
    if remaining_phase <= 1e-9:
        return 0.0
    row = schedule_row or {}
    signed = float(row.get("signed_m", float("nan")))
    sensitivity = float(row.get("sensitivity_m_per_deg", float("nan")))
    target = max(0.0, float(target_depth_m))
    if not (np.isfinite(signed) and np.isfinite(sensitivity)
            and sensitivity > ASYNC_WAVE_DYNAMIC_MIN_SENSITIVITY_M_PER_DEG):
        return min(float(nominal_step_deg), remaining_phase)
    remaining_depth = signed + target if signed >= 0.0 else target + signed
    remaining_depth = max(0.00025, float(remaining_depth))
    predicted = remaining_depth / sensitivity
    adaptive = float(ARC_CONTACT_PHASE_TARGET_FRACTION) * predicted
    adaptive = float(np.clip(
        adaptive,
        ARC_CONTACT_PHASE_MIN_STEP_DEG,
        ARC_CONTACT_PHASE_MAX_STEP_DEG,
    ))
    return min(adaptive, remaining_phase)



def _scan_fixed_palm_natural_minimum(
    solver,
    arcs,
    q,
    commands,
    finger,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    """Return the remaining fixed-palm natural minimum for one finger.

    This is a geometry prediction only. The arm is not moved and no safety
    threshold is relaxed. Command units are the existing Aero-hand arc degrees.
    """
    start = float(commands.get(finger, 0.0))
    maximum = min(float(SCRATCH_FULL_CURL_DEG), float(arcs[finger]["angles"][-1]))
    start = float(np.clip(start, 0.0, maximum))
    coarse = np.arange(
        start,
        maximum + 0.5 * float(V103_FIRST_OWNER_MIN_SCAN_COARSE_DEG),
        float(V103_FIRST_OWNER_MIN_SCAN_COARSE_DEG),
        dtype=np.float64,
    )
    if len(coarse) == 0 or coarse[-1] < maximum - 1e-9:
        coarse = np.r_[coarse, maximum]
    def sample(command):
        trial = dict(commands)
        trial[finger] = float(command)
        relation = finger_surface_relation_at_state(
            solver, arcs, q, trial, finger,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        signed = float(relation.get("signed", float("nan")))
        if not np.isfinite(signed):
            return None
        return (float(command), signed, relation)

    def turning_candidates(rows):
        candidates = []
        for index in range(1, len(rows) - 1):
            previous = rows[index - 1]
            current = rows[index]
            following = rows[index + 1]
            left_step = float(current[0] - previous[0])
            right_step = float(following[0] - current[0])
            if left_step <= 1.0e-9 or right_step <= 1.0e-9:
                continue
            left_derivative = float((current[1] - previous[1]) / left_step)
            right_derivative = float((following[1] - current[1]) / right_step)
            # Signed depth must first decrease (deeper), then increase
            # (shallower).  A monotonic crossing has no natural minimum.
            if (left_derivative > -float(V103_NATURAL_MIN_DERIVATIVE_EPS_M_PER_DEG)
                    or right_derivative < float(V103_NATURAL_MIN_DERIVATIVE_EPS_M_PER_DEG)):
                continue
            normal = np.asarray(current[2].get('normal'), dtype=np.float64)
            previous_point = np.asarray(previous[2].get('world_point'), dtype=np.float64)
            following_point = np.asarray(following[2].get('world_point'), dtype=np.float64)
            tangent = following_point - previous_point
            tangent_norm = float(np.linalg.norm(tangent))
            normal_norm = float(np.linalg.norm(normal))
            if tangent_norm <= 1.0e-9 or normal_norm <= 1.0e-9:
                continue
            tangent_normal_cos = abs(float(np.dot(
                tangent / tangent_norm,
                normal / normal_norm,
            )))
            if tangent_normal_cos > float(V103_NATURAL_MIN_TANGENT_NORMAL_MAX):
                continue
            candidates.append({
                'row': current,
                'left_derivative': left_derivative,
                'right_derivative': right_derivative,
                'tangent_normal_cos': tangent_normal_cos,
            })
        return candidates

    rows = [row for command in coarse if (row := sample(command)) is not None]
    coarse_turns = turning_candidates(rows)
    if not coarse_turns:
        return None

    refined_turns = []
    for coarse_turn in coarse_turns:
        coarse_command = float(coarse_turn['row'][0])
        lo = max(start, coarse_command - float(V103_FIRST_OWNER_MIN_SCAN_RADIUS_DEG))
        hi = min(maximum, coarse_command + float(V103_FIRST_OWNER_MIN_SCAN_RADIUS_DEG))
        fine = np.arange(
            lo,
            hi + 0.5 * float(V103_FIRST_OWNER_MIN_SCAN_FINE_DEG),
            float(V103_FIRST_OWNER_MIN_SCAN_FINE_DEG),
            dtype=np.float64,
        )
        fine_rows = [row for command in fine if (row := sample(command)) is not None]
        refined_turns.extend(turning_candidates(fine_rows) or [coarse_turn])

    selected = min(refined_turns, key=lambda item: float(item['row'][1]))
    command, signed, relation = selected['row']
    return {
        "command_deg": float(command),
        "signed_m": float(signed),
        "depth_m": max(0.0, -float(signed)),
        "distance_m": float(relation.get("distance", float("nan"))),
        "world_point": np.asarray(relation.get("world_point"), dtype=np.float64),
        "remaining_deg": max(0.0, float(command) - start),
        "minimum_kind": "LOCAL_TANGENT_TURN",
        "left_derivative_m_per_deg": float(selected['left_derivative']),
        "right_derivative_m_per_deg": float(selected['right_derivative']),
        "tangent_normal_cos": float(selected['tangent_normal_cos']),
    }


def _scan_fixed_palm_bounded_local_optimum(
    solver,
    arcs,
    q,
    commands,
    finger,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
):
    """Return a safe bounded endpoint when no derivative turning point exists.

    Point-cloud arcs are often monotonic at a steep/near-base pose, so the
    strict turning-point scan can legitimately return ``None`` even though a
    finite, executable curl exists.  That absence must not discard a whole
    four-finger candidate for a *non-current* finger.  Select the deepest
    sampled state that stays inside the existing 4 mm planning cap and mark it
    as a local optimum.  It never grants completion credit by itself; the live
    contact wave still requires the finger's normal quality gate.
    """
    start = float(np.clip(commands.get(finger, 0.0), 0.0, 100.0))
    maximum = min(float(SCRATCH_FULL_CURL_DEG), float(arcs[finger]["angles"][-1]))
    if maximum < start - 1e-9:
        return None
    step = max(float(V103_FIRST_OWNER_MIN_SCAN_COARSE_DEG), 1.0)
    rows = []
    for command in np.unique(np.r_[np.arange(start, maximum + 0.5 * step, step), maximum]):
        trial = dict(commands)
        trial[finger] = float(command)
        relation = finger_surface_relation_at_state(
            solver, arcs, q, trial, finger,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        signed = float(relation.get("signed", float("nan")))
        point = np.asarray(relation.get("world_point"), dtype=np.float64)
        if not np.isfinite(signed) or point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        # Do not use an over-deep endpoint as a planning seed.  The hard
        # runtime/swept guards remain authoritative for accepted motions.
        if signed < -float(V103_QUALITY_DEPTH_MAX_M) - 1e-12:
            continue
        rows.append((float(command), signed, relation))
    if not rows:
        return None
    command, signed, relation = min(rows, key=lambda row: row[1])
    return {
        "command_deg": float(command),
        "signed_m": float(signed),
        "depth_m": max(0.0, -float(signed)),
        "distance_m": float(relation.get("distance", float("nan"))),
        "world_point": np.asarray(relation.get("world_point"), dtype=np.float64),
        "remaining_deg": max(0.0, float(command) - start),
        "minimum_kind": "BOUNDED_LOCAL_OPTIMUM_NO_TURN",
        "optional_local_optimum": True,
    }



def _scan_fixed_palm_all_natural_minima(
    solver,
    arcs,
    q,
    commands,
    fingers,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    allow_local_optimum=False,
):
    """Independently scan each finger's own fixed-palm natural minimum.

    The four minima are deliberately NOT forced onto a line.  Each finger owns
    its own curl channel; the returned world points are used only for the true
    scratch-area centroid Cmin.
    """
    minima = {}
    for finger in fingers:
        info = _scan_fixed_palm_natural_minimum(
            solver, arcs, q, commands, finger,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        if info is None and bool(allow_local_optimum):
            info = _scan_fixed_palm_bounded_local_optimum(
                solver, arcs, q, commands, finger,
                back_points, back_normals, back_tree, back_mesh_query,
            )
        if info is None:
            return None
        minima[finger] = info
    return minima


def _minimum_centroid_geometry(minima, lane_spec):
    """Return Cmin and its tangential error to the GUI target."""
    points = np.asarray([
        np.asarray(minima[f]["world_point"], dtype=np.float64)
        for f in LONG_FINGERS
    ], dtype=np.float64)
    centroid = np.mean(points, axis=0)
    anchor = np.asarray(lane_spec["anchor"], dtype=np.float64)
    normal = normalize(lane_spec["normal"])
    if normal is None:
        return None
    delta = centroid - anchor
    tangent = delta - float(np.dot(delta, normal)) * normal
    return {
        "centroid_world": centroid,
        "tangent_error_vector": tangent,
        "tangent_error_m": float(np.linalg.norm(tangent)),
        "points_by_finger": {
            f: np.asarray(minima[f]["world_point"], dtype=np.float64)
            for f in LONG_FINGERS
        },
    }

def _select_fixed_palm_first_owner_preposition(
    solver,
    arcs,
    q_start,
    endpoint_q,
    commands_start,
    first_owner,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
    historical_replay=False,
):
    """Solve a fixed-palm first-owner preposition whose natural minimum is ~3 mm.

    V22 uses the four independently scanned fixed-palm natural minima as the scratch footprint.
    Their centroid Cmin determines tangential placement; the stable MCP/root row
    determines top-view yaw/perpendicularity.  The four minima are never forced
    onto one line.
    The existing staging->endpoint joint corridor is retained as a cheap fast
    path.  If it does not already yield the requested natural minimum, the
    first-owner arc supplies the scalar depth error and the palm target is
    translated directly along the local BackSurface normal.  Lula IK then
    realizes that palm pose.  Every IK solution is re-measured from the full
    finger arc and must still pass the unchanged joint, palm/root alignment,
    active lane/patch, Cmin-to-GUI-target <=15 mm, MCP-row perpendicular to BASE->Cmin, and dense open-hand swept-mesh checks.

    This keeps the FIRST owner mechanically free: J1-J5 are solved before the
    scratch starts and remain fixed until that owner reaches its natural
    minimum.  Wrist compensation is therefore never used to rescue owner 1.
    """
    q0 = np.asarray(q_start, dtype=np.float64)
    q1 = np.asarray(endpoint_q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)
    target = float(V103_FIRST_OWNER_MIN_TARGET_M)
    evaluated = []
    validation_rejects = []

    def _record_validation_reject(source, reason):
        validation_rejects.append((str(source), str(reason)))
        return None

    def _safe_plane_normal_route(q_target, palm_target_p, palm_target_r):
        """Build lift -> high-plane J4 alignment -> normal-line approach waypoints."""
        normal = normalize(np.asarray(lane_spec.get('normal'), dtype=np.float64))
        if normal is None:
            return None, 'NORMAL_INVALID'
        p0, r0 = solver.compute_forward_kinematics(PALM_FRAME, q0)
        p0, r0 = np.asarray(p0, dtype=np.float64), np.asarray(r0, dtype=np.float64)

        # First try the user's direct joint-space version of the workaround:
        # lift J5 while the hand is still open, rotate J4 at that high pose,
        # move the remaining arm joints to the original target while staying
        # lifted, and only then lower J5 to the original target.  This keeps
        # the original endpoint/trajectory; it only changes the order of the
        # approach joints when the direct swept path is unsafe.
        for lift_deg in (6.0, 10.0, 14.0, 18.0, 24.0, 30.0):
            lifted_j5 = min(
                float(upper[4]),
                max(float(q0[4]), float(q_target[4]))
                + math.radians(float(lift_deg)),
            )
            if lifted_j5 <= max(float(q0[4]), float(q_target[4])) + 1e-8:
                continue
            q_lift = q0.copy()
            q_lift[4] = lifted_j5
            q_j4_high = q_lift.copy()
            q_j4_high[3] = float(np.clip(q_target[3], lower[3], upper[3]))
            q_target_high = np.asarray(q_target, dtype=np.float64).copy()
            q_target_high[4] = lifted_j5
            route_segments = (
                (q0, q_lift, 18),
                (q_lift, q_j4_high, 24),
                (q_j4_high, q_target_high, 28),
                (q_target_high, q_target, 18),
            )
            segment_reasons = []
            route_ok = True
            for segment_start, segment_end, samples in route_segments:
                segment_ok, segment_reason = segment_mesh_safe(
                    solver, arcs,
                    segment_start, commands_start,
                    segment_end, commands_start,
                    back_points, back_normals, back_tree, back_mesh_query,
                    samples=int(samples), allowed_contact_fingers=set(),
                )
                segment_reasons.append(segment_reason)
                if not segment_ok:
                    route_ok = False
                    break
            if route_ok:
                print(
                    '    [V103 PREPOSITION HIGH-J4 JOINT ROUTE]',
                    'j5_lift_deg=', round(float(lift_deg), 2),
                    'j4_before_safe_plane=True',
                    'segments=', 4,
                )
                return [q_lift, q_j4_high, q_target_high], 'HIGH_J4_JOINT_ROUTE'

        height = float(V103_PREPOSITION_SAFE_HEIGHT_M)
        align_height = max(
            height + 0.02,
            float(V103_PREPOSITION_ALIGN_HEIGHT_M),
        )
        # Keep the first lift in the current orientation.  Complete the final
        # J4/palm orientation at the higher alignment plane, then descend with
        # that orientation fixed.  This prevents a late J4 rotation from
        # sweeping an open finger through the BackSurface near the target.
        targets = [
            (p0 + normal * align_height, r0, 'ALIGN_LIFT'),
            (np.asarray(palm_target_p) + normal * align_height,
             np.asarray(palm_target_r), 'ALIGN_J4_HIGH'),
            (np.asarray(palm_target_p) + normal * height,
             np.asarray(palm_target_r), 'SAFE_DESCEND'),
        ]
        route = []
        seed = q0.copy()
        for position, rotation, tag in targets:
            solutions = solve_lula_ik(
                solver, position, rotation, [seed, q0, np.asarray(q_target)],
                lower, upper, diagnostics={},
            )
            if not solutions:
                return None, tag + '_IK_FAIL'
            oriented = [row for row in solutions if row.get('kind') == 'orientation_ik']
            if oriented:
                solutions = oriented
            item = min(
                solutions,
                key=lambda row: float(np.linalg.norm(np.asarray(row['q']) - seed)),
            )
            seed = np.asarray(item['q'], dtype=np.float64).copy()
            route.append(seed.copy())

        # Keep the final orientation fixed while approaching the endpoint along
        # the local normal.  Intermediate IK solves make this a Cartesian-normal
        # approach rather than a late joint-space J4 rotation.
        p_target = np.asarray(palm_target_p, dtype=np.float64)
        r_target = np.asarray(palm_target_r, dtype=np.float64)
        for fraction in np.linspace(
                1.0 - 1.0 / (V103_PREPOSITION_NORMAL_ROUTE_SAMPLES + 1),
                1.0 / (V103_PREPOSITION_NORMAL_ROUTE_SAMPLES + 1),
                int(V103_PREPOSITION_NORMAL_ROUTE_SAMPLES)):
            position = p_target + normal * height * float(fraction)
            solutions = solve_lula_ik(
                solver, position, r_target, [seed, np.asarray(q_target)],
                lower, upper, diagnostics={},
            )
            if not solutions:
                return None, 'NORMAL_APPROACH_IK_FAIL'
            oriented = [row for row in solutions if row.get('kind') == 'orientation_ik']
            if oriented:
                solutions = oriented
            item = min(
                solutions,
                key=lambda row: float(np.linalg.norm(np.asarray(row['q']) - seed)),
            )
            seed = np.asarray(item['q'], dtype=np.float64).copy()
            route.append(seed.copy())

        # The actual fixed-palm endpoint remains the last waypoint.  Do not
        # duplicate it when a numerical IK solution landed on the same q.
        route.append(np.asarray(q_target, dtype=np.float64).copy())
        return route, 'SAFE_PLANE_NORMAL_APPROACH'

    def validate_q(
        q,
        source,
        alpha=None,
        normal_offset_m=None,
        target_position=None,
        natural_relaxed=False,
    ):
        q = np.asarray(q, dtype=np.float64)
        if np.any(q < lower - 1e-10) or np.any(q > upper + 1e-10):
            return _record_validation_reject(source, 'Q_BOUNDS')
        palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        palm_p = np.asarray(palm_p, dtype=np.float64)
        palm_r = np.asarray(palm_r, dtype=np.float64)
        palm_angle = float(palm_back_alignment(arcs, palm_r, lane_spec['normal'])['angle_deg'])
        if not np.isfinite(palm_angle) or palm_angle > palm_alignment_limit_deg():
            return _record_validation_reject(source, 'PALM_ALIGNMENT')
        depth_tolerance = v103_first_owner_depth_tolerance_m(palm_angle)
        depth_floor = v103_first_owner_min_depth_m(palm_angle, first_owner)

        # Hard pre-contact invariant: all four open fingertip start points must
        # be outside the BackSurface.  The first-owner natural minimum is a
        # later curled state; it must never be used to excuse another finger
        # already penetrating at the prepared pose.
        preposition_signed = {}
        preposition_bad = {}
        for finger in LONG_FINGERS:
            relation = finger_surface_relation_at_state(
                solver, arcs, q, commands_start, finger,
                back_points, back_normals, back_tree, back_mesh_query,
            )
            signed = float(relation.get('signed', float('nan')))
            preposition_signed[finger] = signed
            if (not np.isfinite(signed)
                    or signed < -float(V103_PREPOSITION_START_MIN_CLEARANCE_M)):
                preposition_bad[finger] = (
                    None if not np.isfinite(signed)
                    else round(signed * 1000.0, 3)
                )
        if preposition_bad:
            return _record_validation_reject(
                source,
                'PREPOSITION_START_CONTACT:' + str(preposition_bad),
            )
        print(
            '    [V103 PREPOSITION START CLEAR]',
            'signed_mm=', {
                finger: round(preposition_signed[finger] * 1000.0, 3)
                for finger in LONG_FINGERS
            },
            'minimum_clearance_mm=', round(
                float(V103_PREPOSITION_START_MIN_CLEARANCE_M) * 1000.0, 3
            ),
        )

        # V22 scratch-footprint geometry.  Every finger is scanned independently
        # in its own curl channel; their four minima may be non-collinear.
        minima = _scan_fixed_palm_all_natural_minima(
            solver, arcs, q, commands_start, LONG_FINGERS,
            back_points, back_normals, back_tree, back_mesh_query,
            allow_local_optimum=True,
        )
        if minima is None:
            return _record_validation_reject(source, 'NATURAL_MIN_SCAN')
        min_geometry = _minimum_centroid_geometry(minima, lane_spec)
        if min_geometry is None:
            return _record_validation_reject(source, 'CENTROID_GEOMETRY')
        center_error = float(min_geometry['tangent_error_m'])
        center_limit = (
            v103_minima_centroid_tolerance_m(palm_angle)
            if natural_relaxed
            else float(V103_MINIMA_CENTROID_TARGET_TOL_M)
        )
        if (not np.isfinite(center_error)
                or center_error > float(center_limit)):
            return _record_validation_reject(
                source,
                f'CENTROID_TOLERANCE:{center_error * 1000.0:.3f}>{center_limit * 1000.0:.3f}mm',
            )

        # Top-view yaw/perpendicularity uses the stable MCP row, while BASE->C
        # terminates at the true natural-minimum centroid Cmin.
        root_state = centroid_base_axis_alignment(
            arcs, palm_p, palm_r,
            world_points_by_finger=min_geometry['points_by_finger'],
        )
        root_angle = float(root_state['angle_deg'])
        if (not np.isfinite(root_angle) or root_angle > ROOT_BASE_AXIS_HARD_DEG):
            return _record_validation_reject(
                source,
                f'MCP_ROW_ANGLE:{root_angle:.3f}>{ROOT_BASE_AXIS_HARD_DEG:.3f}deg',
            )

        info = minima.get(first_owner)
        if info is None:
            return _record_validation_reject(source, 'FIRST_OWNER_MIN_MISSING')
        lane = lane_relation_for_world_point(info['world_point'], lane_spec, first_owner)
        if (float(lane['cross_error']) > active_cross_limit
                or float(lane['patch_radius']) > active_radius_limit):
            return _record_validation_reject(
                source,
                f"LANE_PATCH:cross={float(lane['cross_error']) * 1000.0:.3f}>{active_cross_limit * 1000.0:.3f}mm,"
                f"patch={float(lane['patch_radius']) * 1000.0:.3f}>{active_radius_limit * 1000.0:.3f}mm",
            )
        # The palm is placed BEFORE the first free scratch.  With the current
        # pre-shaped/open hand, this whole move must therefore remain clear.
        # A direct joint interpolation can cut through the sheet even when the
        # final natural-minimum pose is on the safe side.  In that case try a
        # J5-only lifted waypoint first; J5 is the preferred wrist depth axis
        # and this keeps the first-owner palm pose itself unchanged.
        ok, reason = segment_mesh_safe(
            solver, arcs, q0, commands_start, q, commands_start,
            back_points, back_normals, back_tree, back_mesh_query,
            samples=28, allowed_contact_fingers=set(),
        )
        preposition_waypoints = []
        if not ok:
            routed_waypoints, route_reason = _safe_plane_normal_route(q, palm_p, palm_r)
            if routed_waypoints is not None:
                preposition_waypoints = [
                    np.asarray(waypoint, dtype=np.float64).copy()
                    for waypoint in routed_waypoints[:-1]
                ]
                print(
                    '    [V103 PREPOSITION SAFE-PLANE ROUTE]',
                    'source=', source,
                    'align_height_mm=', round(V103_PREPOSITION_ALIGN_HEIGHT_M * 1000.0, 1),
                    'safe_height_mm=', round(V103_PREPOSITION_SAFE_HEIGHT_M * 1000.0, 1),
                    'waypoints=', len(preposition_waypoints),
                    'j4_aligned_before_safe_plane=True',
                    'normal_approach=True',
                )
                reason = route_reason
            elif V103_ALLOW_PREPOSITION_SWEEP_INTERSECTION:
                print(
                    '    [V103 PREPOSITION SWEEP OVERRIDE]',
                    'source=', source,
                    'reason=', reason,
                    'route_reason=', route_reason,
                    'diagnostic_only=True',
                    'scratch_safety_gates_retained=True',
                )
                reason = 'PREPOSITION_SWEEP_OVERRIDE'
        if not ok and not preposition_waypoints and reason != 'PREPOSITION_SWEEP_OVERRIDE':
            routed = False
            for lift_deg in (4.0, 8.0, 12.0, 16.0, 20.0):
                q_start_lift = q0.copy()
                q_start_lift[4] = min(
                    float(upper[4]),
                    float(q0[4]) + math.radians(float(lift_deg)),
                )
                q_lift = q.copy()
                q_lift[4] = min(
                    float(upper[4]),
                    float(q[4]) + math.radians(float(lift_deg)),
                )
                if (abs(float(q_start_lift[4] - q0[4])) <= 1e-8
                        or abs(float(q_lift[4] - q[4])) <= 1e-8):
                    continue
                first_ok, first_reason = segment_mesh_safe(
                    solver, arcs, q0, commands_start,
                    q_start_lift, commands_start,
                    back_points, back_normals, back_tree, back_mesh_query,
                    samples=16, allowed_contact_fingers=set(),
                )
                if not first_ok:
                    continue
                second_ok, second_reason = segment_mesh_safe(
                    solver, arcs, q_start_lift, commands_start,
                    q_lift, commands_start,
                    back_points, back_normals, back_tree, back_mesh_query,
                    samples=28, allowed_contact_fingers=set(),
                )
                if not second_ok:
                    continue
                third_ok, third_reason = segment_mesh_safe(
                    solver, arcs, q_lift, commands_start, q, commands_start,
                    back_points, back_normals, back_tree, back_mesh_query,
                    samples=22, allowed_contact_fingers=set(),
                )
                if third_ok:
                    preposition_waypoints = [q_start_lift.copy(), q_lift.copy()]
                    reason = (
                        'ROUTED_J5_LIFT_' + str(round(float(lift_deg), 1))
                        + 'DEG'
                    )
                    routed = True
                    print(
                        '    [V103 PREPOSITION ROUTED]',
                        'source=', source,
                        'j5_lift_deg=', round(float(lift_deg), 2),
                        'legs=3',
                        'direct_reason=', reason,
                    )
                    break
            if not routed:
                return _record_validation_reject(source, f'SWEPT_PREPOSITION:{reason}')
        depth = float(info['depth_m'])
        position_error = 0.0
        if target_position is not None:
            position_error = float(np.linalg.norm(
                palm_p - np.asarray(target_position, dtype=np.float64)
            ))
        # Depth is authoritative.  IK position residual and joint travel are
        # only tie-breakers after all hard geometry checks have passed.
        score = (
            abs(depth - target)
            + 0.08 * position_error
            + 0.04 * center_error
            + 1e-7 * float(np.linalg.norm(np.degrees(q - q0)))
        )
        row = {
            'score': float(score),
            'q': q.copy(),
            'info': info,
            'palm_p': palm_p,
            'palm_r': palm_r,
            'palm_angle': float(palm_angle),
            'depth_tolerance_m': float(depth_tolerance),
            'depth_floor_m': float(depth_floor),
            'centroid_tolerance_m': float(center_limit),
            'root_angle': float(root_angle),
            'centroid_target_error_m': float(center_error),
            'centroid_tangent_error_vector': np.asarray(min_geometry['tangent_error_vector'], dtype=np.float64),
            'minimum_centroid_world': np.asarray(min_geometry['centroid_world'], dtype=np.float64),
            'minima_by_finger': minima,
            'preposition_waypoints': preposition_waypoints,
            'source': str(source),
            'alpha': None if alpha is None else float(alpha),
            'normal_offset_m': None if normal_offset_m is None else float(normal_offset_m),
            'position_error_m': float(position_error),
        }
        evaluated.append(row)
        return row

    # Cheap fast path: keep a known-good corridor solution if it already puts
    # the natural minimum at 3 mm.  Unlike V19, failure here does NOT reject
    # the candidate; it triggers the normal-depth IK solve below.
    corridor_rows = []
    coarse_alphas = np.linspace(0.0, 1.0, int(V103_FIRST_OWNER_PREPOSITION_SAMPLES))
    for alpha in coarse_alphas:
        q = q0 + float(alpha) * (q1 - q0)
        row = validate_q(q, 'CORRIDOR', alpha=alpha)
        if row is not None:
            corridor_rows.append(row)
    if corridor_rows:
        best_corridor = min(corridor_rows, key=lambda row: row['score'])
        spacing = 1.0 / max(int(V103_FIRST_OWNER_PREPOSITION_SAMPLES) - 1, 1)
        lo = max(0.0, float(best_corridor['alpha']) - spacing)
        hi = min(1.0, float(best_corridor['alpha']) + spacing)
        for alpha in np.linspace(lo, hi, 7):
            q = q0 + float(alpha) * (q1 - q0)
            validate_q(q, 'CORRIDOR_REFINE', alpha=alpha)
        best_corridor = min(
            [row for row in evaluated if row['source'].startswith('CORRIDOR')],
            key=lambda row: row['score'],
        )
        if abs(float(best_corridor['info']['depth_m']) - target) <= float(
                best_corridor.get('depth_tolerance_m', V103_FIRST_OWNER_MIN_TARGET_TOL_M)):
            info = best_corridor['info']
            print(
                '    [V103 FIRST OWNER PREPOSITION]',
                'owner=', first_owner,
                'source=CORRIDOR_EXACT',
                'corridor_alpha=', round(float(best_corridor['alpha']), 4),
                'minimum_command_deg=', round(float(info['command_deg']), 3),
                'minimum_depth_mm=', round(float(info['depth_m']) * 1000.0, 3),
                'target_mm=3.0',
                'tolerance_mm=', round(float(best_corridor.get(
                    'depth_tolerance_m', V103_FIRST_OWNER_MIN_TARGET_TOL_M
                )) * 1000.0, 3),
                'palm_fixed_until_minimum=True',
                'palm_angle_deg=', round(float(best_corridor['palm_angle']), 3),
                'centroid_perp_error_deg=', round(float(best_corridor['root_angle']), 3),
                'centroid_target_error_mm=', round(float(best_corridor.get('centroid_target_error_m', 0.0)) * 1000.0, 3),
            )
            return {
                'q': np.asarray(best_corridor['q'], dtype=np.float64),
                'minimum': info,
                'alpha': float(best_corridor['alpha']),
                'preposition_waypoints': [
                    np.asarray(waypoint, dtype=np.float64).copy()
                    for waypoint in best_corridor.get('preposition_waypoints', [])
                ],
                'source': 'CORRIDOR_EXACT',
            }

    # Choose a geometrically meaningful seed even when the corridor itself was
    # not fully valid.  Its only job is to supply a palm orientation and a
    # measured natural-minimum depth from which the normal correction is read.
    seed_rows = list(corridor_rows)
    if not seed_rows and bool(historical_replay):
        # A physically successful historical endpoint is a replay seed, not an
        # ordinary optimizer result.  The current geometry filters can reject
        # every sampled corridor point even when the exact endpoint previously
        # produced a valid four-finger trajectory.  Reuse that endpoint only
        # after re-measuring its open start, natural minima, and first-owner
        # quality; the complete swept-path validator still runs afterwards.
        try:
            replay_q = np.asarray(q1, dtype=np.float64).copy()
            if (np.all(np.isfinite(replay_q))
                    and np.all(replay_q >= lower - 1e-10)
                    and np.all(replay_q <= upper + 1e-10)):
                replay_bad = {}
                for finger in LONG_FINGERS:
                    relation = finger_surface_relation_at_state(
                        solver, arcs, replay_q, commands_start, finger,
                        back_points, back_normals, back_tree, back_mesh_query,
                    )
                    signed = float(relation.get('signed', float('nan')))
                    if not np.isfinite(signed) or signed < -float(V103_PREPOSITION_START_MIN_CLEARANCE_M):
                        replay_bad[finger] = None if not np.isfinite(signed) else round(signed * 1000.0, 3)
                replay_minima = _scan_fixed_palm_all_natural_minima(
                    solver, arcs, replay_q, commands_start, LONG_FINGERS,
                    back_points, back_normals, back_tree, back_mesh_query,
                    allow_local_optimum=True,
                )
                replay_info = None if replay_minima is None else replay_minima.get(first_owner)
                if not replay_bad and replay_info is not None:
                    replay_depth = float(replay_info.get('depth_m', float('nan')))
                    replay_palm_p, replay_palm_r = solver.compute_forward_kinematics(PALM_FRAME, replay_q)
                    replay_angle = float(palm_back_alignment(
                        arcs, np.asarray(replay_palm_r, dtype=np.float64), lane_spec['normal']
                    )['angle_deg'])
                    replay_floor = v103_first_owner_min_depth_m(replay_angle, first_owner)
                    if (np.isfinite(replay_depth) and np.isfinite(replay_angle)
                            and replay_depth >= replay_floor - 1e-12
                            and replay_depth <= V103_QUALITY_DEPTH_MAX_M + 1e-12):
                        print(
                            '[V103 HISTORICAL REPLAY PREPOSITION]',
                            'owner=', first_owner,
                            'minimum_depth_mm=', round(replay_depth * 1000.0, 3),
                            'palm_angle_deg=', round(replay_angle, 3),
                            'strict_corridor_reject_recovered=True',
                            'full_path_validation_still_required=True',
                        )
                        return {
                            'q': replay_q,
                            'minimum': replay_info,
                            'alpha': 1.0,
                            'preposition_waypoints': [],
                            'depth_tolerance_m': v103_first_owner_depth_tolerance_m(replay_angle),
                            'depth_floor_m': replay_floor,
                            'source': 'HISTORICAL_REPLAY_ENDPOINT',
                        }
        except Exception as replay_exc:
            print('[V103 HISTORICAL REPLAY PREPOSITION REJECT]', repr(replay_exc))
    if not seed_rows:
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            q = q0 + float(alpha) * (q1 - q0)
            if np.any(q < lower - 1e-10) or np.any(q > upper + 1e-10):
                continue
            try:
                palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
                palm_p = np.asarray(palm_p, dtype=np.float64)
                palm_r = np.asarray(palm_r, dtype=np.float64)
                palm_angle = float(palm_back_alignment(
                    arcs, palm_r, lane_spec['normal'])['angle_deg'])
                if (not np.isfinite(palm_angle)
                        or palm_angle > palm_alignment_limit_deg()):
                    continue
                minima = _scan_fixed_palm_all_natural_minima(
                    solver, arcs, q, commands_start, LONG_FINGERS,
                    back_points, back_normals, back_tree, back_mesh_query,
                    allow_local_optimum=True,
                )
                if minima is None:
                    continue
                min_geometry = _minimum_centroid_geometry(minima, lane_spec)
                if min_geometry is None:
                    continue
                center_error = float(min_geometry['tangent_error_m'])
                if (not np.isfinite(center_error)
                        or center_error > float(V103_MINIMA_CENTROID_TANGENT_RECENTER_MAX_M)):
                    continue
                root_angle = float(centroid_base_axis_alignment(
                    arcs, palm_p, palm_r,
                    world_points_by_finger=min_geometry['points_by_finger'],
                )['angle_deg'])
                if not np.isfinite(root_angle) or root_angle > ROOT_BASE_AXIS_HARD_DEG:
                    continue
                info = minima.get(first_owner)
                if info is None:
                    continue
                seed_rows.append({
                    'q': q.copy(), 'info': info,
                    'palm_p': palm_p, 'palm_r': palm_r,
                    'palm_angle': palm_angle, 'root_angle': root_angle,
                    'depth_tolerance_m': v103_first_owner_depth_tolerance_m(palm_angle),
                    'depth_floor_m': v103_first_owner_min_depth_m(palm_angle, first_owner),
                    'centroid_target_error_m': center_error,
                    'centroid_tangent_error_vector': np.asarray(min_geometry['tangent_error_vector'], dtype=np.float64),
                    'minimum_centroid_world': np.asarray(min_geometry['centroid_world'], dtype=np.float64),
                    'minima_by_finger': minima,
                    'score': (abs(float(info['depth_m']) - target) + 0.04 * center_error),
                    'source': 'RAW_SEED', 'alpha': float(alpha),
                    'normal_offset_m': None, 'position_error_m': 0.0,
                })
            except Exception:
                continue
    if not seed_rows:
        print(
            '    [V103 FIRST OWNER DEPTH-CONDITIONED IK REJECT]',
            'owner=', first_owner,
            'reason=NO_FINITE_BASE_GEOMETRY',
        )
        return None

    base = min(seed_rows, key=lambda row: float(row.get('score', abs(float(row['info']['depth_m']) - target))))
    base_q = np.asarray(base['q'], dtype=np.float64)
    base_p = np.asarray(base['palm_p'], dtype=np.float64)
    base_r = np.asarray(base['palm_r'], dtype=np.float64)
    base_depth = float(base['info']['depth_m'])
    base_depth_tolerance = float(base.get(
        'depth_tolerance_m',
        v103_first_owner_depth_tolerance_m(base.get('palm_angle', float('nan'))),
    ))
    base_depth_floor = float(base.get(
        'depth_floor_m',
        v103_first_owner_min_depth_m(base.get('palm_angle', float('nan')), first_owner),
    ))

    # A natural first-owner minimum in the tilt-dependent working band is
    # already a valid result.  3 mm is a preference, not a reason to invoke a
    # normal-offset IK solve that can turn a safe 2.607 mm pose into 0 mm.
    # The relaxed validator still enforces joint limits, palm alignment, Cmin
    # placement (with bounded tilt tolerance), MCP/base orientation, owner
    # lane/patch and swept-mesh safety.  If that natural pose is unsafe,
    # retain its measured depth as evidence, but continue through the ordered
    # strict-IK and last-resort approximate-IK paths below.
    natural_in_quality_band = bool(
        base_depth_floor - 1e-12
        <= base_depth
        <= V103_QUALITY_DEPTH_MAX_M + 1e-12
    )
    if natural_in_quality_band:
        natural_row = validate_q(
            base_q,
            'NATURAL_MIN_TILT_TOLERATED',
            natural_relaxed=True,
        )
        if natural_row is not None:
            print(
                '    [V103 FIRST OWNER NATURAL MINIMUM ACCEPTED]',
                'owner=', first_owner,
                'minimum_depth_mm=', round(base_depth * 1000.0, 3),
                'target_mm=', round(target * 1000.0, 3),
                'preferred_tolerance_mm=', round(base_depth_tolerance * 1000.0, 3),
                'quality_band_mm=', (
                    round(base_depth_floor * 1000.0, 3), 'to',
                    round(V103_QUALITY_DEPTH_MAX_M * 1000.0, 3),
                ),
                'tilt_floor_relaxed=', bool(base_depth_floor < V103_QUALITY_DEPTH_MIN_M),
                'palm_fixed_until_minimum=True',
                'normal_ik_correction_skipped=True',
            )
            return {
                'q': np.asarray(natural_row['q'], dtype=np.float64),
                'minimum': natural_row['info'],
                'alpha': natural_row.get('alpha'),
                'preposition_waypoints': [
                    np.asarray(waypoint, dtype=np.float64).copy()
                    for waypoint in natural_row.get('preposition_waypoints', [])
                ],
                'depth_tolerance_m': float(natural_row.get(
                    'depth_tolerance_m', base_depth_tolerance
                )),
                'source': 'NATURAL_MIN_TILT_TOLERATED',
            }
        print(
            '    [V103 FIRST OWNER NATURAL MINIMUM REJECTED]',
            'owner=', first_owner,
            'minimum_depth_mm=', round(base_depth * 1000.0, 3),
            'quality_band_mm=', (
                round(base_depth_floor * 1000.0, 3), 'to',
                round(V103_QUALITY_DEPTH_MAX_M * 1000.0, 3),
            ),
            'tilt_floor_relaxed=', bool(base_depth_floor < V103_QUALITY_DEPTH_MIN_M),
            'reason=GEOMETRY_OR_SWEEP_SAFETY',
            'normal_ik_fallback=True',
            'approximate_ik_allowed_after_strict=True',
            'details=', validation_rejects[-1] if validation_rejects else None,
        )
        # Do not discard a mechanically good 2--4 mm natural minimum merely
        # because its direct approach (and the conditional high-plane J4
        # route) was unsafe.  Continue into the original strict normal-IK
        # search, then the explicit approximate fallback if strict IK still
        # cannot realize a safe preposition.  This is the required
        # original -> high-plane J4 -> last-resort ordering.
    normal = np.asarray(lane_spec['normal'], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-9:
        return None
    normal = normal / normal_norm

    # V22 direct tangential placement: Cmin decides WHERE the scratch footprint
    # sits.  Translate the whole palm target by the opposite Cmin->GUI tangential
    # error before solving depth.  This is a geometric recenter, not a joint-space
    # search.  The realized pose is re-scanned and must still be within 15 mm.
    base_tangent_error = np.asarray(
        base.get('centroid_tangent_error_vector', np.zeros(3, dtype=np.float64)),
        dtype=np.float64,
    )
    tangent_norm = float(np.linalg.norm(base_tangent_error))
    if not np.isfinite(tangent_norm):
        base_tangent_error = np.zeros(3, dtype=np.float64)
        tangent_norm = 0.0
    if tangent_norm > float(V103_MINIMA_CENTROID_TANGENT_RECENTER_MAX_M) and tangent_norm > 1e-12:
        base_tangent_error = (
            base_tangent_error / tangent_norm
            * float(V103_MINIMA_CENTROID_TANGENT_RECENTER_MAX_M)
        )
    tangent_recenter = -base_tangent_error

    # Positive offset means move INTO the back (opposite its outward normal).
    # First prediction is exactly the measured depth error; subsequent solves
    # re-measure the real arc and refine rather than assuming 1 mm == 1 mm.
    predicted_offset = float(np.clip(
        target - base_depth,
        -float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
        float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
    ))
    orientation_wxyz = rotation_matrix_to_wxyz(base_r)

    seed_qs = [base_q, q0, q1]
    for row in sorted(seed_rows, key=lambda r: abs(float(r['info']['depth_m']) - target))[:4]:
        seed_qs.append(np.asarray(row['q'], dtype=np.float64))
    unique_seeds = []
    seen_seed = set()
    for seed in seed_qs:
        key = tuple(np.round(np.degrees(seed), 2))
        if key not in seen_seed:
            seen_seed.add(key)
            unique_seeds.append(seed)

    solved_offsets = {}

    def solve_offset(offset_m, tag='NORMAL_IK', position_tolerances=None):
        """Solve one normal offset, keeping strict IK as the default.

        The approximate position tolerances are deliberately an explicit
        fallback argument.  They must never silently replace the original
        preposition trajectory or the high-plane J4 route.
        """
        if position_tolerances is None:
            position_tolerances = V103_FIRST_OWNER_DEPTH_IK_POSITION_TOLS_M
        position_tolerances = tuple(float(value) for value in position_tolerances)
        offset_m = float(np.clip(
            offset_m,
            -float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
            float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
        ))
        key = (
            round(offset_m, 7),
            tuple(round(value, 7) for value in position_tolerances),
        )
        if key in solved_offsets:
            return solved_offsets[key]
        target_p = base_p + tangent_recenter - normal * offset_m
        rows = []
        seen_q = set()
        for position_tol in position_tolerances:
            for orientation_tol in (0.25, 0.35, 0.60, 0.90):
                for seed in unique_seeds:
                    try:
                        q, success = solver.compute_inverse_kinematics(
                            frame_name=PALM_FRAME,
                            target_position=target_p,
                            target_orientation=orientation_wxyz,
                            warm_start=seed,
                            position_tolerance=float(position_tol),
                            orientation_tolerance=float(orientation_tol),
                        )
                    except Exception:
                        continue
                    if not success:
                        continue
                    q = np.asarray(q, dtype=np.float64)
                    qkey = tuple(np.round(np.degrees(q), 3))
                    if qkey in seen_q:
                        continue
                    seen_q.add(qkey)
                    row = validate_q(
                        q, tag,
                        normal_offset_m=offset_m,
                        target_position=target_p,
                    )
                    if row is not None:
                        row['ik_position_tolerances_m'] = position_tolerances
                        row['ik_mode'] = (
                            'APPROXIMATE'
                            if position_tolerances == tuple(
                                float(value)
                                for value in V103_FIRST_OWNER_APPROX_POSITION_TOLS_M
                            )
                            else 'STRICT'
                        )
                        rows.append(row)
                        if abs(float(row['info']['depth_m']) - target) <= float(
                                row.get('depth_tolerance_m', V103_FIRST_OWNER_MIN_TARGET_TOL_M)):
                            solved_offsets[key] = rows
                            return rows
                if rows:
                    # Do not widen orientation tolerance once strict IK already
                    # produced valid candidates at this normal position.
                    break
            if rows:
                break
        solved_offsets[key] = rows
        return rows

    # Try the analytically obvious depth correction first, then a bounded set
    # around it.  This is a 1-D normal solve, not a 5-D blind joint search.
    offsets_to_try = [predicted_offset]
    step = float(V103_FIRST_OWNER_DEPTH_IK_COARSE_STEP_M)
    for k in range(1, 5):
        offsets_to_try.extend([predicted_offset - k * step, predicted_offset + k * step])
    offsets_to_try.extend([0.0, -float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
                           float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M)])
    for offset in offsets_to_try:
        solve_offset(offset)
        valid = [row for rows in solved_offsets.values() for row in rows]
        if valid:
            best_now = min(valid, key=lambda row: row['score'])
            if abs(float(best_now['info']['depth_m']) - target) <= float(
                    best_now.get('depth_tolerance_m', V103_FIRST_OWNER_MIN_TARGET_TOL_M)):
                break

    # Secant/bracket refinement in NORMAL OFFSET space.  The full natural arc
    # is rescanned after each IK solve, so this remains valid even when the IK
    # residual or local surface curvature makes depth-vs-offset non-unitary.
    for _ in range(int(V103_FIRST_OWNER_DEPTH_IK_MAX_REFINES)):
        valid = [row for rows in solved_offsets.values() for row in rows]
        if not valid:
            break
        best_now = min(valid, key=lambda row: row['score'])
        best_error = float(best_now['info']['depth_m']) - target
        if abs(best_error) <= float(
                best_now.get('depth_tolerance_m', V103_FIRST_OWNER_MIN_TARGET_TOL_M)):
            break
        # One representative row per solved offset, then find the nearest
        # depth bracket.  If no bracket exists, estimate a local slope from the
        # two depth-closest samples.
        reps = []
        for rows in solved_offsets.values():
            if rows:
                reps.append(min(rows, key=lambda row: row['score']))
        reps.sort(key=lambda row: float(row['normal_offset_m']))
        proposal = None
        for left, right in zip(reps[:-1], reps[1:]):
            e0 = float(left['info']['depth_m']) - target
            e1 = float(right['info']['depth_m']) - target
            if e0 == 0.0 or e1 == 0.0 or e0 * e1 < 0.0:
                x0 = float(left['normal_offset_m'])
                x1 = float(right['normal_offset_m'])
                if abs(e1 - e0) > 1e-9:
                    proposal = x0 - e0 * (x1 - x0) / (e1 - e0)
                else:
                    proposal = 0.5 * (x0 + x1)
                break
        if proposal is None and len(reps) >= 2:
            near = sorted(reps, key=lambda row: abs(float(row['info']['depth_m']) - target))[:2]
            x0, x1 = (float(near[0]['normal_offset_m']), float(near[1]['normal_offset_m']))
            e0 = float(near[0]['info']['depth_m']) - target
            e1 = float(near[1]['info']['depth_m']) - target
            if abs(e1 - e0) > 1e-9 and abs(x1 - x0) > 1e-9:
                proposal = x0 - e0 * (x1 - x0) / (e1 - e0)
        if proposal is None:
            # Last-resort depth-error correction around the best measured pose.
            proposal = float(best_now['normal_offset_m']) - best_error
        proposal = float(np.clip(
            proposal,
            -float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
            float(V103_FIRST_OWNER_DEPTH_IK_SEARCH_M),
        ))
        before = len(solved_offsets)
        solve_offset(proposal, tag='NORMAL_IK_REFINE')
        if len(solved_offsets) == before:
            break

    strict_valid = [row for rows in solved_offsets.values() for row in rows]
    strict_in_quality_band = [
        row for row in strict_valid
        if float(row.get('depth_floor_m', V103_QUALITY_DEPTH_MIN_M)) - 1e-12
        <= float(row['info']['depth_m'])
        <= V103_QUALITY_DEPTH_MAX_M + 1e-12
    ]

    # The original trajectory and the conditional high-plane J4 route are
    # always first.  Only when strict IK produced no usable 1--4 mm pose do we
    # try the explicitly approximate preposition solve.  This is the user's
    # last-resort path, not a replacement for the normal planner.
    approximate_used = False
    if not strict_in_quality_band:
        approximate_used = True
        print(
            '    [V103 FIRST OWNER APPROX PREPOSITION IK]',
            'strict_rows=', len(strict_valid),
            'strict_quality_rows=', len(strict_in_quality_band),
            'reason=ORIGINAL_AND_HIGH_PLANE_J4_ROUTE_NO_USABLE_POSE',
            'position_tolerances_mm=', tuple(
                round(float(value) * 1000.0, 3)
                for value in V103_FIRST_OWNER_APPROX_POSITION_TOLS_M
            ),
        )
        for offset in offsets_to_try:
            solve_offset(
                offset,
                tag='NORMAL_IK_APPROX',
                position_tolerances=V103_FIRST_OWNER_APPROX_POSITION_TOLS_M,
            )
        approximate_valid = [
            row for rows in solved_offsets.values() for row in rows
            if row.get('ik_mode') == 'APPROXIMATE'
        ]
        approximate_in_quality_band = [
            row for row in approximate_valid
            if float(row.get('depth_floor_m', V103_QUALITY_DEPTH_MIN_M)) - 1e-12
            <= float(row['info']['depth_m'])
            <= V103_QUALITY_DEPTH_MAX_M + 1e-12
        ]
        # Prefer an approximate pose only when it is still inside the same
        # final quality band.  If it is not, retain the rows for diagnostics
        # and let the existing quality rejection explain the failure.
        valid = (
            approximate_in_quality_band
            if approximate_in_quality_band
            else approximate_valid
        )
    else:
        valid = strict_valid

    if not valid:
        # Preserve the best corridor diagnostic when IK itself cannot realize
        # a safe normal-offset pose.
        best_depth = None
        if corridor_rows:
            best_depth = min(
                abs(float(row['info']['depth_m']) - target) for row in corridor_rows
            )
        print(
            '    [V103 FIRST OWNER DEPTH-CONDITIONED IK REJECT]',
            'owner=', first_owner,
            'reason=NO_SAFE_NORMAL_IK_SOLUTION',
            'base_min_depth_mm=', round(base_depth * 1000.0, 3),
            'predicted_normal_offset_mm=', round(predicted_offset * 1000.0, 3),
            'tangent_recenter_mm=', round(float(np.linalg.norm(tangent_recenter)) * 1000.0, 3),
            'best_corridor_abs_error_mm=', None if best_depth is None else round(best_depth * 1000.0, 3),
            'approximate_preposition_ik=', approximate_used,
        )
        return None

    best = min(valid, key=lambda row: row['score'])
    info = best['info']
    best_depth = float(info['depth_m'])
    best_depth_tolerance = float(
        best.get('depth_tolerance_m', V103_FIRST_OWNER_MIN_TARGET_TOL_M)
    )
    best_depth_floor = float(best.get('depth_floor_m', base_depth_floor))
    if not (
        best_depth_floor - 1e-12
        <= best_depth
        <= V103_QUALITY_DEPTH_MAX_M + 1e-12
    ):
        print(
            '    [V103 FIRST OWNER DEPTH-CONDITIONED IK REJECT]',
            'owner=', first_owner,
            'reason=OUTSIDE_TILT_ADJUSTED_DEPTH_BAND',
            'base_min_depth_mm=', round(base_depth * 1000.0, 3),
            'best_min_depth_mm=', round(best_depth * 1000.0, 3),
            'target_mm=', round(target * 1000.0, 3),
            'depth_floor_mm=', round(best_depth_floor * 1000.0, 3),
            'tolerance_mm=', round(best_depth_tolerance * 1000.0, 3),
            'normal_offset_mm=', round(float(best['normal_offset_m']) * 1000.0, 3),
            'ik_position_error_mm=', round(float(best['position_error_m']) * 1000.0, 3),
        )
        return None

    if abs(best_depth - target) > best_depth_tolerance:
        print(
        '    [V103 FIRST OWNER PREFERRED TARGET MISS ACCEPTED]',
            'owner=', first_owner,
            'minimum_depth_mm=', round(best_depth * 1000.0, 3),
            'preferred_target_mm=', round(target * 1000.0, 3),
            'quality_band_mm=1_to_4',
            'tolerance_mm=', round(best_depth_tolerance * 1000.0, 3),
        )

    print(
        '    [V103 FIRST OWNER DEPTH-CONDITIONED IK]',
        'owner=', first_owner,
        'base_min_depth_mm=', round(base_depth * 1000.0, 3),
        'depth_floor_mm=', round(best_depth_floor * 1000.0, 3),
        'normal_offset_mm=', round(float(best['normal_offset_m']) * 1000.0, 3),
        'tangent_recenter_mm=', round(float(np.linalg.norm(tangent_recenter)) * 1000.0, 3),
        'minimum_command_deg=', round(float(info['command_deg']), 3),
        'minimum_depth_mm=', round(float(info['depth_m']) * 1000.0, 3),
        'target_mm=3.0',
        'tolerance_mm=', round(best_depth_tolerance * 1000.0, 3),
        'ik_position_error_mm=', round(float(best['position_error_m']) * 1000.0, 3),
        'palm_fixed_until_minimum=True',
        'palm_angle_deg=', round(float(best['palm_angle']), 3),
        'centroid_perp_error_deg=', round(float(best['root_angle']), 3),
        'centroid_target_error_mm=', round(float(best.get('centroid_target_error_m', 0.0)) * 1000.0, 3),
        'ik_mode=', best.get('ik_mode', 'STRICT'),
        'approximate_preposition_ik=', approximate_used,
    )
    return {
        'q': np.asarray(best['q'], dtype=np.float64),
        'minimum': info,
        'alpha': None,
        'preposition_waypoints': [
            np.asarray(waypoint, dtype=np.float64).copy()
            for waypoint in best.get('preposition_waypoints', [])
        ],
        'normal_offset_m': float(best['normal_offset_m']),
        'source': (
            'DEPTH_CONDITIONED_NORMAL_IK_APPROX'
            if best.get('ik_mode') == 'APPROXIMATE'
            else 'DEPTH_CONDITIONED_NORMAL_IK'
        ),
    }

def acquire_asynchronous_four_finger_contact(
    solver,
    arcs,
    mode,
    q_start,
    commands_start,
    endpoint_q,
    phases,
    working_fingers,
    endpoint_palm_p,
    endpoint_palm_r,
    approach_normal,
    handoff_clearance_m,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
    fixed_arm=False,
    linear_arm_sync=False,
    historical_replay=False,
):
    """Acquire the first natural minimum as a continuous four-finger wave.

    Completion ownership follows one frozen shallow-to-deep order, but every
    later finger advances from the first staging frame at a lower rate. This
    removes the old stop-and-wait acquisition while keeping the same swept
    mesh, patch/lane and joint-limit checks.
    """
    working = [finger for finger in LONG_FINGERS if finger in set(working_fingers)]
    anchor_finger = lane_spec.get('anchor_finger')
    reference_fingers = [anchor_finger] if anchor_finger in working else working
    if anchor_finger in working:
        # Linear wrist lowering ignores the target-palm compensation computed
        # below. Strict point acquisition must actually solve that moving pose.
        linear_arm_sync = False
        fixed_arm = False
        print('[V103 ANCHORED ACQUISITION] reference=',anchor_finger,
              'continuous_joint_solve=True; swept checks retained')
    q_trial = np.asarray(q_start, dtype=np.float64).copy()
    commands_trial = dict(commands_start)
    active_trial = []
    frames = []
    balanced_handoff = False
    previous_nominal_q = q_trial.copy()
    start_palm_p, _ = solver.compute_forward_kinematics(PALM_FRAME, q_trial)
    start_palm_p = np.asarray(start_palm_p, dtype=np.float64)
    final_center = np.mean(
        [
            world_tip_position(solver, arcs, endpoint_q, phases, finger)
            for finger in reference_fingers
        ],
        axis=0,
    )
    # Overlapping acquisition uses one shared wave clock. Budget from the
    # largest remaining phase, not the sum of four serial arcs. A few extra
    # frames keep the wrist solve smooth while followers are already moving.
    max_phase_delta = max(
        max(0.0, float(phases[finger]) - float(commands_trial.get(finger, 0.0)))
        for finger in working
    )
    step_count = max(
        16,
        int(math.ceil(max_phase_delta / ARC_CONTACT_PHASE_STEP_DEG)) + 8,
    )
    acquisition_focus = None
    first_owner = None

    start_phase_center_at_endpoint = np.mean(
        [
            world_tip_position(
                solver, arcs, endpoint_q, commands_start, finger
            )
            for finger in reference_fingers
        ],
        axis=0,
    )
    ideal_start_palm = (
        np.asarray(endpoint_palm_p, dtype=np.float64)
        + (final_center - start_phase_center_at_endpoint)
        + float(handoff_clearance_m)
        * np.asarray(approach_normal, dtype=np.float64)
    )
    # The handoff optimizer is allowed a small pose residual.  Carry that
    # residual into the first acquisition frame and fade it to zero, instead
    # of introducing a discontinuous palm target at step 1.
    handoff_target_residual = start_palm_p - ideal_start_palm

    # Freeze the first staging/endpoint BackSurface order for this scratch
    # cycle. It is measured from the real BackSurface, not click-centre
    # Euclidean distance, and wrist motion later in the cycle may update
    # sensitivities/delays but must not reshuffle ownership.
    contact_order, initial_delays, initial_schedule = _dynamic_wave_schedule(
        solver, arcs, endpoint_q, commands_start, working,
        back_points, back_normals, back_tree, back_mesh_query,
        tie_order=working,
    )
    acquisition_priority_order = list(contact_order)
    first_owner = acquisition_priority_order[0] if acquisition_priority_order else None
    acquisition_focus = first_owner
    print(
        "    [V103 FROZEN INITIAL ARC ORDER]",
        "shallow_to_deep=", acquisition_priority_order,
        "measured_error_mm=", {
            finger: round(initial_schedule[finger]["metric_m"] * 1000.0, 2)
            for finger in acquisition_priority_order
        },
        "dynamic_stagger_deg=", {
            finger: round(initial_delays[finger], 2)
            for finger in acquisition_priority_order
        },
        "rerank_during_cycle=False",
    )

    # V19: choose the actual scratch preposition BEFORE finger motion. The
    # first frozen owner must reach its natural minimum at ~3 mm with the palm
    # completely fixed. This makes the first stroke a free mechanical scratch,
    # not a wrist-press recovery problem.
    preposition = _select_fixed_palm_first_owner_preposition(
        solver, arcs, q_start, endpoint_q, commands_start, first_owner,
        lower, upper, back_points, back_normals, back_tree, back_mesh_query,
        lane_spec,
        historical_replay=historical_replay,
    )
    if preposition is None:
        print(
            '    [V100 SYNC ACQUISITION REJECT]',
            'reason=FIRST_OWNER_DEPTH_CONDITIONED_IK_FAIL',
            'owner=', first_owner,
        )
        return None

    q_preposition = np.asarray(preposition['q'], dtype=np.float64)
    _, q_preposition_r = solver.compute_forward_kinematics(
        PALM_FRAME, q_preposition
    )
    first_owner_palm_angle = float(palm_back_alignment(
        arcs, np.asarray(q_preposition_r, dtype=np.float64), lane_spec['normal']
    )['angle_deg'])
    first_owner_depth_floor = v103_first_owner_min_depth_m(
        first_owner_palm_angle, first_owner
    )
    print(
        '    [V103 FIRST OWNER TILT FLOOR]',
        'owner=', first_owner,
        'palm_angle_deg=', round(first_owner_palm_angle, 3),
        'depth_floor_mm=', round(first_owner_depth_floor * 1000.0, 3),
        'preferred_target_mm=3.0',
    )
    # Use the same tilt-aware lower floor for every mandatory core finger in
    # this wave.  Pinky remains on the separate assist/local-optimum policy.
    quality_minimums_m = {
        finger: v103_quality_min_depth_m(first_owner_palm_angle, finger)
        for finger in V103_MANDATORY_QUALITY_FINGERS
    }
    print(
        '    [V103 QUALITY TILT FLOORS]',
        'palm_angle_deg=', round(first_owner_palm_angle, 3),
        'minimums_mm=', {
            finger: round(value * 1000.0, 3)
            for finger, value in quality_minimums_m.items()
        },
    )
    preposition_waypoints = [
        np.asarray(waypoint, dtype=np.float64).copy()
        for waypoint in preposition.get('preposition_waypoints', [])
    ]
    q_trial = q_preposition.copy()
    commands_trial = dict(commands_start)
    active_trial = []
    previous_nominal_q = q_preposition.copy()
    route_q = np.asarray(q_start, dtype=np.float64).copy()
    for waypoint_index, waypoint_q in enumerate(preposition_waypoints, start=1):
        if np.max(np.abs(np.degrees(waypoint_q - route_q))) <= 1e-5:
            continue
        frames.append({
            'stage': f'ARC_FIRST_OWNER_PREPOSITION_ROUTE_{waypoint_index}',
            'duration': max(ARC_PRECONTACT_KEYFRAME_TIME_S, 0.08),
            'q': waypoint_q.copy(),
            'fingers': dict(commands_trial),
            'active': [],
            'authorized_contact_fingers': [],
            'dynamic_focus': first_owner,
            'dynamic_handoff_policy': 'J5_LIFT_PREPOSITION_ROUTE',
        })
        route_q = waypoint_q.copy()
    if np.max(np.abs(np.degrees(q_preposition - route_q))) > 1e-5:
        frames.append({
            'stage': 'ARC_FIRST_OWNER_FIXED_PALM_PREPOSITION',
            'duration': max(ARC_PRECONTACT_KEYFRAME_TIME_S, 0.08),
            'q': q_preposition.copy(),
            'fingers': dict(commands_trial),
            'active': [],
            'authorized_contact_fingers': [],
            'dynamic_focus': first_owner,
            'dynamic_handoff_policy': 'FIXED_PALM_FIRST_MINIMUM_THEN_ROLLING_WAVE',
        })

    # Natural minima are recomputed every accepted frame, but J1-J5 are held
    # exactly at q_preposition until the first owner reaches its minimum. All
    # four fingers move from frame 1. Their command increments are derived from
    # remaining command-to-minimum and a one-keyframe target gap, so later
    # fingers make up longer arcs with speed rather than waiting for ownership.
    first_initial_minimum = _scan_fixed_palm_natural_minimum(
        solver, arcs, q_trial, commands_trial, first_owner,
        back_points, back_normals, back_tree, back_mesh_query,
    )
    if first_initial_minimum is None:
        # The planning preposition intentionally permits a bounded local
        # optimum when a steep/near-base arc has no derivative turning point.
        # Use the same measured fallback here instead of accepting it during
        # preposition and then killing the candidate at acquisition start.
        fallback_minimum = _scan_fixed_palm_bounded_local_optimum(
            solver, arcs, q_trial, commands_trial, first_owner,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        fallback_depth = (
            float(fallback_minimum.get('depth_m', float('nan')))
            if fallback_minimum is not None else float('nan')
        )
        if (fallback_minimum is None
                or not np.isfinite(fallback_depth)
                or fallback_depth < first_owner_depth_floor - 1e-12):
            print(
                '    [V100 SYNC ACQUISITION REJECT]',
                'reason=NATURAL_MINIMUM_SCAN_FAIL',
                'owner=', first_owner,
                'fallback_depth_mm=', (
                    None if not np.isfinite(fallback_depth)
                    else round(fallback_depth * 1000.0, 3)
                ),
                'required_floor_mm=', round(first_owner_depth_floor * 1000.0, 3),
            )
            return None
        first_initial_minimum = fallback_minimum
        print(
            '    [V103 FIRST OWNER BOUNDED LOCAL OPTIMUM]',
            'owner=', first_owner,
            'depth_mm=', round(fallback_depth * 1000.0, 3),
            'required_floor_mm=', round(first_owner_depth_floor * 1000.0, 3),
            'completion_credit=True',
            'minimum_kind=', fallback_minimum.get('minimum_kind'),
        )
    first_remaining = max(
        0.0,
        float(first_initial_minimum['command_deg'])
        - float(commands_trial.get(first_owner, 0.0)),
    )
    step_count = max(
        6,
        int(math.ceil(first_remaining / max(V103_FIRST_OWNER_MAX_STEP_DEG, 1e-6)))
        + len(working) + 4,
    )

    for step_index in range(1, step_count + 1):
        measured_order, dynamic_delays, dynamic_schedule = _dynamic_wave_schedule(
            solver, arcs, q_trial, commands_trial, working,
            back_points, back_normals, back_tree, back_mesh_query,
            tie_order=acquisition_priority_order,
            max_delay_deg=ASYNC_WAVE_DYNAMIC_MAX_DELAY_DEG,
        )
        contact_order = list(acquisition_priority_order)
        acquisition_focus = first_owner

        minima = {}
        for finger in contact_order:
            # The first owner was solved at the fixed preposition above.  Its
            # natural minimum is the handoff trigger for this phase and must
            # stay locked while the other fingers catch up.  Re-scanning it
            # from a moving command can select a shallower local turning point
            # (e.g. ring ~=28 deg) and falsely declare the owner complete at
            # 0 mm before it reaches the measured ~=45 deg minimum.
            if finger == first_owner:
                minima[finger] = dict(first_initial_minimum)
                continue
            minima[finger] = _scan_fixed_palm_natural_minimum(
                solver, arcs, q_trial, commands_trial, finger,
                back_points, back_normals, back_tree, back_mesh_query,
            )
            if minima[finger] is None:
                # A follower may have a monotonic/airborne arc at this palm
                # pose, so there is no derivative turning point to report.
                # Keep the four-channel acquisition alive with a bounded local
                # optimum; it is only a motion seed and never earns contact
                # completion.  The finger is re-scanned when it becomes the
                # locked focus in the real contact wave.
                minima[finger] = _scan_fixed_palm_bounded_local_optimum(
                    solver, arcs, q_trial, commands_trial, finger,
                    back_points, back_normals, back_tree, back_mesh_query,
                )
                if minima[finger] is not None:
                    print(
                        '    [V103 DEFERRED NATURAL MINIMUM]',
                        'step=', step_index,
                        'finger=', finger,
                        'kind=', minima[finger].get('minimum_kind'),
                        'depth_mm=', round(float(minima[finger].get('depth_m', 0.0)) * 1000.0, 3),
                        'completion_credit=False',
                    )
            if minima[finger] is None:
                # Pinky is intentionally optional.  Its short/offset arc may
                # have no signed-depth turning point at the current fixed
                # palm pose (it can remain airborne or monotonic all the way
                # through the available curl range).  That is a valid local
                # optimum for the optional channel, not a failure of the
                # mandatory index/middle/ring acquisition wave.  Hold its
                # current command and keep all normal safety checks in the
                # candidate evaluator; do not invent a depth/contact credit.
                if finger in V103_OPTIONAL_QUALITY_FINGERS:
                    relation = finger_surface_relation_at_state(
                        solver, arcs, q_trial, commands_trial, finger,
                        back_points, back_normals, back_tree, back_mesh_query,
                    )
                    signed = float(relation.get('signed', float('nan')))
                    world_point = np.asarray(
                        relation.get('world_point'), dtype=np.float64
                    )
                    if (np.isfinite(signed)
                            and world_point.shape == (3,)
                            and np.all(np.isfinite(world_point))):
                        current_command = float(commands_trial.get(finger, 0.0))
                        minima[finger] = {
                            'command_deg': current_command,
                            'signed_m': signed,
                            'depth_m': max(0.0, -signed),
                            'distance_m': float(
                                relation.get('distance', float('nan'))
                            ),
                            'world_point': world_point,
                            'remaining_deg': 0.0,
                            'minimum_kind': 'OPTIONAL_LOCAL_OPTIMUM_NO_TURN',
                            'optional_local_optimum': True,
                        }
                        print(
                            '    [V103 OPTIONAL FINGER LOCAL OPTIMUM]',
                            'finger=', finger,
                            'reason=NO_NATURAL_TURNING_POINT',
                            'command_deg=', round(current_command, 3),
                            'signed_mm=', round(signed * 1000.0, 3),
                            'depth_mm=', round(max(0.0, -signed) * 1000.0, 3),
                        )
                        continue
                print(
                    '    [V100 SYNC ACQUISITION REJECT]',
                    'step=', step_index,
                    'reason=NATURAL_MINIMUM_SCAN_FAIL',
                    'finger=', finger,
                )
                return None

        owner_remaining = max(
            0.0,
            float(minima[first_owner]['command_deg'])
            - float(commands_trial.get(first_owner, 0.0)),
        )
        owner_eta_steps = max(
            1.0,
            math.ceil(owner_remaining / max(V103_FIRST_OWNER_MAX_STEP_DEG, 1e-6)),
        )
        next_commands = dict(commands_trial)
        background_step_deg = {}
        eta_steps = {}
        adaptive_step_deg = 0.0

        for rank, finger in enumerate(contact_order):
            current_command = float(commands_trial.get(finger, 0.0))
            minimum_command = float(minima[finger]['command_deg'])
            remaining = max(0.0, minimum_command - current_command)
            target_steps = max(
                1.0,
                float(owner_eta_steps)
                + float(rank) * float(V103_MINIMUM_ARRIVAL_GAP_STEPS),
            )
            eta_steps[finger] = float(target_steps)
            if remaining <= 1e-9:
                increment = 0.0
            else:
                increment = remaining / target_steps
                max_step = (
                    float(V103_FIRST_OWNER_MAX_STEP_DEG)
                    if finger == first_owner
                    else float(V103_FOLLOWER_MAX_STEP_DEG)
                )
                increment = min(max_step, remaining, max(
                    float(V103_MIN_CONTINUOUS_FINGER_STEP_DEG),
                    float(increment),
                ))

                # Preserve only the genuine planned depth cap. There is no
                # artificial rank reserve: if a later finger is naturally
                # close, let speed prediction place it near its minimum, but
                # never spend the 3.6 mm hard planning reserve early.
                row = dynamic_schedule.get(finger, {})
                signed_now = float(row.get('signed_m', float('nan')))
                signed_rate = float(row.get('signed_sensitivity_m_per_deg', 0.0))
                if np.isfinite(signed_now) and signed_rate < -1e-12:
                    depth_now = max(0.0, -signed_now)
                    room = max(
                        0.0,
                        float(v103_effective_runtime_depth_cap_m() - 0.00015)
                        - depth_now,
                    )
                    if room <= 1e-9:
                        increment = 0.0
                    else:
                        increment = min(increment, room / (-signed_rate))
            next_commands[finger] = min(minimum_command, current_command + increment)
            if finger == first_owner:
                adaptive_step_deg = float(increment)
            else:
                background_step_deg[finger] = float(increment)

        # Hard invariant: the palm/arm does not move during the first owner's
        # scratch-to-minimum interval. Any candidate requiring J4/J5 repair is
        # rejected and the planner must choose another IK/preposition instead.
        q_candidate = q_preposition.copy()
        accepted, reject_reason = evaluate_sync_acquisition_candidate(
            solver, arcs, q_trial, commands_trial, q_candidate, next_commands,
            working, active_trial, back_points, back_normals, back_tree,
            back_mesh_query, lane_spec, acquisition_focus=acquisition_focus,
            forbid_unfocused_entering=True,
        )
        if accepted is None:
            print(
                '    [V100 SYNC ACQUISITION REJECT]',
                'step=', step_index,
                'reason=FIXED_PALM_FINGER_BRANCH_FAIL',
                'detail=', reject_reason,
                'q_delta_deg=', np.round(np.degrees(q_candidate-q_trial), 6).tolist(),
            )
            return None

        relations = accepted['relations']
        entering = accepted['entering']
        next_active = accepted['next_active']
        authorized = accepted['authorized']
        frames.append({
            'stage': (
                'ARC_ASYNC_CONTACT_' if entering else 'ARC_ASYNC_PRECONTACT_'
            ) + str(step_index),
            'duration': (
                ARC_CONTACT_KEYFRAME_TIME_S if entering
                else ARC_PRECONTACT_KEYFRAME_TIME_S
            ),
            'q': q_candidate.copy(),
            'fingers': dict(next_commands),
            'active': list(next_active),
            'authorized_contact_fingers': list(authorized),
            'dynamic_focus': acquisition_focus,
            'dynamic_handoff_policy': 'FIXED_PALM_FIRST_MINIMUM_THEN_ROLLING_WAVE',
        })
        print(
            '    [V103 ROLLING MINIMUM ACQUISITION STEP]',
            'step=', step_index,
            'focus=', first_owner,
            'q_fixed=True',
            'order=', contact_order,
            'minimum_command_deg=', {
                f: round(float(minima[f]['command_deg']), 3) for f in contact_order
            },
            'eta_steps=', {f: round(float(eta_steps[f]), 2) for f in contact_order},
            'finger_step_deg=', {
                first_owner: round(float(adaptive_step_deg), 3),
                **{f: round(float(v), 3) for f, v in background_step_deg.items()},
            },
            'depth_mm=', {
                f: round(max(0.0, -float(
                    finger_surface_relation_at_state(
                        solver, arcs, q_candidate, next_commands, f,
                        back_points, back_normals, back_tree, back_mesh_query,
                    )['signed']
                )) * 1000.0, 3)
                for f in contact_order
            },
        )

        q_trial = q_candidate
        commands_trial = next_commands
        active_trial = next_active

        first_now = finger_surface_relation_at_state(
            solver, arcs, q_trial, commands_trial, first_owner,
            back_points, back_normals, back_tree, back_mesh_query,
        )
        first_depth = max(0.0, -float(first_now['signed']))
        first_minimum_command = float(minima[first_owner]['command_deg'])
        first_at_minimum = bool(
            float(commands_trial[first_owner])
            >= first_minimum_command - max(0.05, V103_FIRST_OWNER_MIN_SCAN_FINE_DEG)
        )
        if first_at_minimum:
            if not (
                first_owner_depth_floor - 1e-12
                <= first_depth
                <= V103_QUALITY_DEPTH_MAX_M + 1e-12
            ):
                print(
                    '    [V103 FIRST OWNER MINIMUM REJECT]',
                    'owner=', first_owner,
                    'depth_mm=', round(first_depth * 1000.0, 3),
                    'required_mm=', (
                        round(first_owner_depth_floor * 1000.0, 3), 'to',
                        round(V103_QUALITY_DEPTH_MAX_M * 1000.0, 3),
                    ),
                )
                return None
            # 3 mm is a preferred target only. Once the fixed-palm first
            # owner reaches the tilt-adjusted natural-minimum band, complete the handoff
            # even when the local pose cannot land exactly on 3 mm. Do not
            # invoke a wrist/normal correction here: the palm is intentionally
            # fixed until this natural minimum is reached.
            if abs(first_depth - V103_FIRST_OWNER_MIN_TARGET_M) > V103_FIRST_OWNER_MIN_TARGET_TOL_M + 1e-12:
                print(
                    '    [V103 FIRST OWNER PREFERRED TARGET MISS ACCEPTED]',
                    'owner=', first_owner,
                    'depth_mm=', round(first_depth * 1000.0, 3),
                    'preferred_target_mm=3.0',
                    'quality_band_mm=', (
                        round(first_owner_depth_floor * 1000.0, 3), 'to',
                        round(V103_QUALITY_DEPTH_MAX_M * 1000.0, 3),
                    ),
                    'q_fixed=True',
                )
            balanced_handoff = True
            print(
                '[V103 DEPTH HANDOFF] first_owner_natural_minimum;',
                'owner=', first_owner,
                'depth_mm=', round(first_depth * 1000.0, 3),
                'q_fixed=True',
                'physical_contacts=', sorted(set(authorized) & set(working)),
                'completion_owner_only=True',
                'progress_pct=', next_commands,
            )
            break

    if not balanced_handoff:
        print(
            "    [V100 SYNC ACQUISITION REJECT]",
            "reason=FIRST_FROZEN_OWNER_NATURAL_MINIMUM_NOT_ESTABLISHED",
            "owner=", first_owner,
            "physical_contacts=", list(active_trial),
        )
        return None
    # Keep the exact initial measured order. Pinky quality optionality is
    # separate from order; it is not globally forced to the end.
    priority_order = [f for f in acquisition_priority_order if f in working]
    for f in working:
        if f not in priority_order:
            priority_order.append(f)
    return {
        "q": q_trial,
        "commands": commands_trial,
        "active": active_trial,
        "frames": frames,
        "contact_order": priority_order,
        "acquisition_priority_order": acquisition_priority_order,
        "balanced_handoff": bool(balanced_handoff),
    }


def new_scratch_contact_stats(fingers):
    return {f: dict(state="UNTOUCHED", min_signed=float("inf"),
                    slide_m=0.0, path_m=0.0, moving_contact_s=0.0,
                    deepest_stage=None) for f in fingers}


def accumulate_scratch_contact_interval(stats, p0, r0, p1, r1, dt,
                                        progress_delta, stage):
    """Clip each interval to actual contact; hysteresis is state only.

    Positive-clearance motion never earns depth/slide credit, even while the
    state remains CONTACT inside the exit deadband. No static/extension credit.
    """
    s0, s1 = float(r0["signed"]), float(r1["signed"])
    for relation in (r0, r1):
        signed, distance = float(relation["signed"]), float(relation["distance"])
        if not np.all(np.isfinite([signed, distance])):
            raise RuntimeError("NONFINITE_CONTACT_GEOMETRY")
        near = distance <= abs(HARD_PENETRATION_M) + ASYNC_CONTACT_EXIT_CLEARANCE_M
        if signed <= 0.0 and near:
            stats["state"] = "CONTACT"
        elif stats["state"] == "CONTACT" and (
            signed > ASYNC_CONTACT_EXIT_CLEARANCE_M or not near
        ):
            stats["state"] = "RELEASED"
    delta = np.asarray(p1) - np.asarray(p0)
    if progress_delta <= 1e-9 or np.linalg.norm(delta) <= 1e-10:
        return
    if max(float(r0["distance"]), float(r1["distance"])) > (
        abs(HARD_PENETRATION_M) + ASYNC_CONTACT_EXIT_CLEARANCE_M
    ):
        return
    if min(s0, s1) >= 0.0:
        return
    lo, hi = 0.0, 1.0
    if s0 > 0.0:
        lo = s0 / (s0 - s1)
    elif s1 > 0.0:
        hi = -s0 / (s1 - s0)
    contact_fraction = max(0.0, hi - lo)
    n = np.asarray(r0["normal"], dtype=float) + np.asarray(r1["normal"], dtype=float)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm < 1e-9:
        raise RuntimeError("INVALID_CONTACT_NORMAL")
    n /= norm
    contact_delta = delta * contact_fraction
    stats["path_m"] += float(np.linalg.norm(contact_delta))
    slide = float(np.linalg.norm(contact_delta - np.dot(contact_delta, n) * n))
    stats["slide_m"] += slide
    # Weight only the moving, surface-contact part of a curl. This measures
    # where scratching occurred, rather than the static template endpoint.
    midpoint = np.asarray(p0)+delta*((lo+hi)*.5)
    stats['contact_position_sum'] = np.asarray(stats.get('contact_position_sum',np.zeros(3)))+slide*midpoint
    stats['contact_position_weight'] = stats.get('contact_position_weight',0.)+slide
    stats["moving_contact_s"] += max(0.0, float(dt)) * contact_fraction
    if min(s0, s1) < stats["min_signed"]:
        stats["min_signed"] = min(s0, s1)
        stats["deepest_stage"] = str(stage)


def scratch_contact_quality_failure(stats, minimum_depth_m=None):
    """Return only a hard planning failure for a required finger.

    The planning contract is an independently measured 1--4 mm depth band
    with a 3 mm preferred target. Slide, moving-contact time and tangent ratio
    describe the demonstration, but must not keep a depth-qualified finger
    pending until the rollout budget expires.
    """
    signed = float(stats["min_signed"])
    minimum_depth_m = (
        ASYNC_MIN_ACCEPTED_SCRATCH_DEPTH_M
        if minimum_depth_m is None
        else max(0.0, float(minimum_depth_m))
    )
    if not math.isfinite(signed):
        return "DEPTH_UNAVAILABLE"
    if max(0.0, -signed) > V103_QUALITY_DEPTH_MAX_M + 1e-12:
        return "DEPTH_MAX"
    if max(0.0, -signed) < minimum_depth_m - 1e-12:
        return "DEPTH"
    return None


def scratch_contact_assist_quality_failure(stats):
    """Return only hard safety failures for a non-anchor finger.

    The anchor finger is the only channel that must be driven toward the
    3 mm setpoint.  Other fingers still participate in the four-channel
    trajectory and remain subject to the 4 mm penetration cap, but their
    measured depth is allowed to settle below the old 2 mm gate.  This keeps a
    harmless FK/mesh mismatch on one assist finger from discarding an otherwise
    executable scratch motion.
    """
    signed = float(stats["min_signed"])
    if not math.isfinite(signed):
        # No contact evidence is a planning diagnostic, not permission to
        # drive deeper.  Let the caller decide whether a no-contact assist is
        # useful; never turn it into a depth pass here.
        return "DEPTH_UNAVAILABLE"
    depth = max(0.0, -signed)
    if depth > V103_QUALITY_DEPTH_MAX_M + 1e-12:
        return "DEPTH_MAX"
    if depth < V103_ASSIST_DEPTH_MIN_M - 1e-12:
        return "DEPTH_ASSIST_SHALLOW"
    return None


def v103_contact_quality_failure(stats, minimum_depth_m=None):
    """Planning gate with an explicit diagnostic-only >4 mm escape hatch."""
    failure = scratch_contact_quality_failure(stats, minimum_depth_m)
    if failure == 'DEPTH_MAX':
        depth = max(0.0, -float(stats.get('min_signed', float('nan'))))
        if (bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW)
                and math.isfinite(depth)
                and depth <= v103_effective_runtime_depth_cap_m() + 1e-12):
            return None
    return failure


def v103_assist_quality_failure(stats):
    """Assist-finger equivalent of the diagnostic depth-only override."""
    failure = scratch_contact_assist_quality_failure(stats)
    if failure == 'DEPTH_MAX':
        depth = max(0.0, -float(stats.get('min_signed', float('nan'))))
        if (bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW)
                and math.isfinite(depth)
                and depth <= v103_effective_runtime_depth_cap_m() + 1e-12):
            return None
    return failure


def scratch_contact_assist_target_reached(stats):
    """Return whether an assist finger reached the normal 1--4 mm band.

    Assist channels are *not* allowed to finish merely because they touched
    0.5 mm.  The 3 mm value remains their preferred optimizer target; this
    predicate only supplies the relaxed hard completion band.  If a safe
    branch cannot reach that band, ``roll`` may still emit a bounded local
    optimum after its patience budget, without converting the assist into a
    3F re-plan or blocking the primary scratch.
    """
    signed = float(stats["min_signed"])
    if not math.isfinite(signed):
        return False
    depth = max(0.0, -signed)
    return bool(
        depth >= ASYNC_MIN_ACCEPTED_SCRATCH_DEPTH_M - 1e-12
        and depth <= (
            v103_effective_runtime_depth_cap_m()
            if bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW)
            else V103_QUALITY_DEPTH_MAX_M
        ) + 1e-12
    )


def scratch_contact_motion_diagnostics(stats):
    """Report legacy motion observations without rejecting a safe plan."""
    observations = []
    if stats["slide_m"] < ASYNC_MIN_CONTACT_SLIDE_M:
        observations.append("SLIDE_BELOW_2MM")
    if stats["moving_contact_s"] < ASYNC_MIN_MOVING_CONTACT_S:
        observations.append("MOVING_CONTACT_BELOW_10MS")
    if stats["slide_m"] / max(stats["path_m"], 1e-12) < ASYNC_MIN_COMBINED_TANGENT_RATIO:
        observations.append("TANGENT_RATIO_BELOW_0.50")
    return observations


def v103_early_exit_evidence(solver, arcs, q, fingers, stats,
                            back_points, back_normals, back_tree, back_mesh_query,
                            required_skin_clearance=None, required_fingers=None,
                            base_contact_stop=False, minimums_m=None):
    if HAND_COLLISION_MODEL is None:
        return None  # Offline evidence still needs the collision model.
    # A true base_contact_stop is supplied only after the runtime PhysX/base
    # event has been latched.  Keep validating the achieved scratch quality
    # and skin clearance, but do not require an offline base-gap reserve for
    # that already-authorized stop.
    required_fingers = list(LONG_FINGERS if required_fingers is None else required_fingers)
    minimums_m = {} if minimums_m is None else dict(minimums_m)
    # Core fingers must have earned their tilt-aware quality floor and reached
    # RELEASED.  Pinky is optional/local-optimum: airborne is valid, while an
    # actual pinky contact may not remain latched at the exit.
    for finger in required_fingers:
        state = str(stats[finger].get('state', 'UNTOUCHED'))
        if finger in V103_MANDATORY_QUALITY_FINGERS:
            if (v103_contact_quality_failure(stats[finger], minimums_m.get(finger)) is not None
                    or state != 'RELEASED'):
                return None
        elif state == 'CONTACT':
            return None
    p,r = solver.compute_forward_kinematics(PALM_FRAME,q)
    clearances = [float(final_surface_relation(np.asarray(p)+np.asarray(r)@interpolate_arc(arcs,f,fingers[f])[0],
                     back_mesh_query,back_points,back_normals,back_tree)['signed']) for f in LONG_FINGERS]
    required = V103_SAFETY_CLEARANCE_M if required_skin_clearance is None else required_skin_clearance
    if (not base_contact_stop
            and not all(math.isfinite(s) and s >= required for s in clearances)):
        return None
    current_gap, link = (float('nan'), None)
    if hasattr(HAND_COLLISION_MODEL, 'base_guard'):
        current_gap, link = HAND_COLLISION_MODEL.base_guard.clearance(q, fingers)
    # Preserve the legacy reserve for isolated/offline callers that do not
    # define the production policy flag.  In Isaac V103 the flag is explicitly
    # true, so a measured base contact may terminate the curl instead of being
    # rejected as an endpoint margin violation.
    if (not base_contact_stop
            and not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))
            and (not math.isfinite(current_gap) or current_gap < V103_BASE_MIN_CLEARANCE_M)):
        return None
    return dict(reason=('BASE_CONTACT_STOP_AFTER_EFFECTIVE_SCRATCH'
                        if base_contact_stop else 'SURFACE_CLEARANCE_AFTER_EFFECTIVE_SCRATCH'),
                skin_clearance_mm=min(clearances)*1000.,
                base_clearance_mm=(None if not math.isfinite(current_gap)
                                   else current_gap*1000.),
                closest_link=link,
                base_contact_stop=bool(base_contact_stop))


def sample_scratch_contact_segment(stats, solver, arcs, a, b, back_points,
                                   back_normals, back_tree, back_mesh_query):
    """Sample all four tips with one palm FK per sample; match runtime timing."""
    progress_delta = max(abs(float(b["fingers"].get(f, 0.0)) -
                             float(a["fingers"].get(f, 0.0))) for f in stats)
    arm_delta = float(np.max(np.abs(np.degrees(np.asarray(b["q"]) - np.asarray(a["q"])))))
    intervals = max(18, int(math.ceil(progress_delta / ASYNC_CONTACT_SAMPLE_PROGRESS_PCT)),
                    int(math.ceil(arm_delta / 0.5)))
    previous = None
    for fraction in np.linspace(0.0, 1.0, intervals + 1):
        alpha = (float(fraction) if str(b["stage"]).startswith("SCRATCH_WAVE_ASYNC_")
                 else smoothstep(float(fraction)))
        q = np.asarray(a["q"]) + alpha * (np.asarray(b["q"]) - np.asarray(a["q"]))
        palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
        current = {}
        for finger in stats:
            p0 = float(a["fingers"].get(finger, 0.0))
            progress = p0 + alpha * (float(b["fingers"].get(finger, 0.0)) - p0)
            local, _ = interpolate_arc(arcs, finger, progress)
            tip = np.asarray(palm_p) + np.asarray(palm_r) @ local
            relation = final_surface_relation(tip, back_mesh_query, back_points, back_normals, back_tree)
            current[finger] = (tip, relation, progress)
            if previous is not None:
                old_tip, old_relation, old_progress = previous[finger]
                accumulate_scratch_contact_interval(
                    stats[finger], old_tip, old_relation, tip, relation,
                    float(b["duration"]) / intervals, progress - old_progress, b["stage"])
        previous = current


def acquisition_scratch_contact_stats(trajectory, working, solver, arcs,
                                       back_points, back_normals, back_tree, back_mesh_query):
    # Only this cycle's contiguous acquisition tail, never an earlier cycle,
    # approach, unload, or opening stroke.
    stats = new_scratch_contact_stats(working)
    first = len(trajectory)
    while first > 0 and str(trajectory[first - 1]["stage"]).startswith("ARC_ASYNC_"):
        first -= 1
    for index in range(max(1, first), len(trajectory)):
        sample_scratch_contact_segment(stats, solver, arcs, trajectory[index - 1], trajectory[index],
                                       back_points, back_normals, back_tree, back_mesh_query)
    return stats


def validate_continuous_scratch_quality(trajectory, solver, arcs, back_points,
                                         back_normals, back_tree, back_mesh_query,
                                         lane_spec=None):
    """Recompute quality, never trust keyframe labels/cached summary alone."""
    stats = None
    # A later cycle may replay a wave that already passed the complete
    # acquisition/quality checks.  Keep that validated snapshot so the replay
    # can inherit contact credit after the first cycle is closed; a fresh wave
    # without either acquisition frames or a validated snapshot must still be
    # rejected below.
    validated_wave_stats = None
    stats_complete = False
    stats_cycle = None
    replay_seen = False
    previous_acquiring = False
    # A later cycle can be assembled by copying the already validated wave
    # segment.  Keep replay detection scoped to the current cycle; a marker
    # in an unrelated cycle must never authorize a fresh wave.
    replay_cycles = set()
    for frame in trajectory:
        stage_upper = str(frame.get('stage', '')).upper()
        match = re.search(r'_CYCLE_(\d+)', stage_upper)
        if (bool(frame.get('replayed_validated_wave')) or '_REPLAY' in stage_upper) and match:
            replay_cycles.add(int(match.group(1)))
    replay_bypass_logged = False

    def quality_minimums_for_q(q):
        angle = float('nan')
        try:
            if lane_spec is not None and lane_spec.get('normal') is not None:
                _, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q)
                angle = float(palm_back_alignment(
                    arcs, np.asarray(palm_r, dtype=np.float64),
                    np.asarray(lane_spec['normal'], dtype=np.float64),
                )['angle_deg'])
        except Exception:
            angle = float('nan')
        return {
            finger: v103_quality_min_depth_m(angle, finger)
            for finger in V103_MANDATORY_QUALITY_FINGERS
        }, angle

    for index in range(1, len(trajectory)):
        a, b = trajectory[index - 1], trajectory[index]
        acquiring = str(b["stage"]).startswith("ARC_ASYNC_")
        waving = str(b["stage"]).startswith("SCRATCH_WAVE_ASYNC_")
        stage_upper = str(b.get('stage', '')).upper()
        if bool(b.get('replayed_validated_wave')) or '_REPLAY' in stage_upper:
            replay_seen = True
        wave_minimums_m, wave_angle_deg = (
            quality_minimums_for_q(b['q']) if waving else ({}, float('nan'))
        )
        if acquiring and not previous_acquiring:
            cycle_match = re.search(r'_CYCLE_(\d+)', str(b.get('stage', '')).upper())
            cycle_number = int(cycle_match.group(1)) if cycle_match else None
            if stats is not None:
                # A completed wave may be followed by the next cycle's
                # acquisition.  Only a second acquisition in the same cycle
                # is an incomplete lifecycle.
                if not stats_complete:
                    return False, "ASYNC_CONTACT_QUALITY_INCOMPLETE_CYCLE"
                if (cycle_number is not None and stats_cycle is not None
                        and cycle_number <= stats_cycle):
                    return False, "ASYNC_CONTACT_QUALITY_INCOMPLETE_CYCLE"
                if cycle_number is None and stats_cycle is not None:
                    return False, "ASYNC_CONTACT_QUALITY_INCOMPLETE_CYCLE"
            stats = new_scratch_contact_stats(LONG_FINGERS)
            stats_complete = False
            stats_cycle = cycle_number
        if waving and stats is None:
            # Cycle 2 may replay the exact wave segment that was validated in
            # cycle 1.  It intentionally has no new ARC_ASYNC acquisition
            # segment; rejecting it as "missing acquisition" invalidates an
            # otherwise identical, already-validated motion.  Do not borrow
            # contact credit for a fresh wave: the marker is scoped to this
            # cycle, while all original cycle-1 checks remain in force and
            # trajectory geometry is still validated separately.
            replay_stage = str(b.get('stage', '')).upper()
            cycle_match = re.search(r'_CYCLE_(\d+)', replay_stage)
            cycle_replay_present = bool(
                cycle_match and int(cycle_match.group(1)) in replay_cycles
            )
            if (bool(b.get('replayed_validated_wave'))
                    or '_REPLAY' in replay_stage
                    or cycle_replay_present):
                if validated_wave_stats is None:
                    return False, "ASYNC_CONTACT_QUALITY_MISSING_ACQUISITION"
                stats = copy.deepcopy(validated_wave_stats)
                stats_complete = True
                stats_cycle = int(cycle_match.group(1)) if cycle_match else stats_cycle
                if not replay_bypass_logged:
                    print('[V103 REPLAY VALIDATION BYPASS]',
                          'stage=' + replay_stage,
                          'cycle_replay_marker=' + str(cycle_replay_present),
                          'restored_validated_stats=' + str(validated_wave_stats is not None),
                          'index=' + str(index))
                    replay_bypass_logged = True
                continue
            return False, "ASYNC_CONTACT_QUALITY_MISSING_ACQUISITION"
        if acquiring or waving:
            sample_scratch_contact_segment(stats, solver, arcs, a, b,
                                           back_points, back_normals, back_tree, back_mesh_query)
            if waving and b.get("contact_state_source") == "sampled_segment_v1":
                # Independently recompute, rather than trusting the builder's
                # state labels or its cumulative quality summary.
                required = list(b.get('scratch_required_fingers', LONG_FINGERS))
                sampled = {f: stats[f]["state"] for f in required}
                if b.get("predicted_contact_states") != sampled:
                    return False, "CONTACT_STATE_REPLAY_MISMATCH|stage=" + str(b["stage"])
        early_exit = bool(b.get('early_exit'))
        required = list(b.get('scratch_required_fingers', LONG_FINGERS)) if waving else list(LONG_FINGERS)
        relaxed = set(b.get('scratch_relaxed_fingers', ())) if waving else set()
        if waving and (set(required) != set(LONG_FINGERS)
                       or not relaxed.issubset(set(LONG_FINGERS))):
            return False, 'FOUR_FINGER_QUALITY_METADATA_INVALID|stage=' + str(b['stage'])
        if waving and early_exit and not v103_early_exit_evidence(
                solver,arcs,b['q'],b['fingers'],stats,back_points,back_normals,back_tree,back_mesh_query,
                required_skin_clearance=V103_EXIT_PLANNED_RESERVE_M if b['early_exit'].get('all_finger_reserve') else None,
                required_fingers=required,
                base_contact_stop=bool(b.get('base_contact_stop')
                                       or b.get('early_exit', {}).get('base_contact_stop')),
                minimums_m=wave_minimums_m):
            return False, 'EARLY_EXIT_EVIDENCE_REJECT|stage=' + str(b['stage'])
        if waving and (early_exit or bool(b.get('scratch_complete')) or
                       all(float(b["fingers"].get(f, 0)) >= SCRATCH_FULL_CURL_DEG - 1e-9
                           for f in required)):
            # Motion statistics are diagnostics. Only the required 1 mm
            # minimum depth remains a planning/final-validation gate.
            # A relaxed pinky is still present in the trajectory and remains
            # subject to all geometry/depth-cap checks, but its preferred
            # 1--4 mm contact depth is not allowed to reject the three core
            # fingers' completed scratch.
            quality_required = [f for f in required
                                if f in V103_MANDATORY_QUALITY_FINGERS]
            failures = {}
            for f in quality_required:
                failure = v103_contact_quality_failure(stats[f], wave_minimums_m.get(f))
                if failure is not None:
                    failures[f] = failure
            # Pinky remains physically present and keeps the 4 mm hard cap, but
            # its lower depth may finish at the bounded local optimum.
            for f in required:
                if f in quality_required:
                    continue
                failure = v103_assist_quality_failure(stats[f])
                if failure is not None and failure not in {'DEPTH_ASSIST_SHALLOW', 'DEPTH_UNAVAILABLE'}:
                    failures[f] = failure
            if failures:
                return False, "ASYNC_CONTACT_QUALITY_REJECT|stage=" + str(b["stage"]) + "|" + str(failures)
            replay_depths = {
                f: (float('nan') if not math.isfinite(float(stats[f]['min_signed']))
                    else max(0.0, -float(stats[f]['min_signed'])) * 1000.0)
                for f in LONG_FINGERS
            }
            # A relaxed pinky remains in the four-channel trajectory and is
            # still subject to the hard penetration cap when it actually
            # contacts the surface.  However, no-contact is a valid local
            # optimum for that auxiliary finger: its ``min_signed`` value is
            # +inf and therefore becomes NaN in this replay summary.  Do not
            # turn that allowed no-contact state back into a whole-wave
            # rejection.  Core fingers (and any finite pinky contact) keep the
            # normal depth-cap gate.
            replay_offenders = {}
            for f, depth in replay_depths.items():
                if f in relaxed and not math.isfinite(depth):
                    continue
                if (not math.isfinite(depth)
                        or depth > V103_PLANNED_DEPTH_MAX_M * 1000.0 + 1e-9):
                    replay_offenders[f] = round(depth, 3)
            if replay_offenders:
                if bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW):
                    allowed = {
                        f: depth for f, depth in replay_offenders.items()
                        if math.isfinite(float(depth))
                        and float(depth) <= v103_effective_runtime_depth_cap_m() * 1000.0 + 1e-9
                    }
                    hard = {f: depth for f, depth in replay_offenders.items() if f not in allowed}
                    if hard:
                        return (False, 'BALANCED_DEPTH_REPLAY_REJECT|stage=' + str(b['stage'])
                                + '|depth_mm=' + str(hard))
                    print('[V103 DIAGNOSTIC DEPTH OVERFLOW]',
                          'stage=', str(b['stage']),
                          'depth_mm=', allowed,
                          'normal_quality_cap_mm=', V103_PLANNED_DEPTH_MAX_M * 1000.0,
                          'diagnostic_cap_mm=', v103_effective_runtime_depth_cap_m() * 1000.0)
                else:
                    return (False, 'BALANCED_DEPTH_REPLAY_REJECT|stage=' + str(b['stage'])
                            + '|depth_mm=' + str(replay_offenders))
            # Preserve the just-validated acquisition/contact state for a
            # subsequent replay cycle.  The next fresh acquisition cycle will
            # replace this snapshot when its own ARC_ASYNC segment begins.
            validated_wave_stats = copy.deepcopy(stats)
            stats_complete = True
        previous_acquiring = acquiring
    if stats is not None and not stats_complete:
        # A replay may be followed by a transition/release frame that does
        # not carry the original early-exit marker.  The first wave was
        # already validated and the replay itself is geometry-validated by
        # validate_trajectory(), so do not turn that bookkeeping detail into
        # a false lifecycle rejection.
        if replay_seen and validated_wave_stats is not None:
            print('[V103 REPLAY VALIDATION COMPLETE]',
                  'restored_validated_stats=True')
            return True, "PASS"
        return False, "ASYNC_CONTACT_QUALITY_INCOMPLETE_CYCLE"
    return True, "PASS"


def append_asynchronous_four_finger_wave(*args, **kwargs):
    """Build one complete four-finger wave with bounded safety retries.

    V103 no longer retries a failed four-finger wave as a three-finger plan.
    The contact controller handles a pinky-only miss as a local optimum while
    retaining all four command channels in the trajectory.
    """
    trajectory = args[0] if args else kwargs["trajectory"]
    mode = args[3] if len(args) > 3 else kwargs.get('mode', {})
    if mode.get('v103_contact_coordination'):
        original_length = len(args[0] if args else kwargs["trajectory"])
        try:
            return append_v103_contact_wave(*args, **kwargs)
        except Exception:
            trajectory = args[0] if args else kwargs["trajectory"]
            del trajectory[original_length:]
            raise
    original_length = len(trajectory)
    for attempt, stagger_scale in enumerate((1.0, 0.5, 1.5)):
        try:
            return _append_asynchronous_four_finger_wave(
                *args, **kwargs, stagger_scale=stagger_scale)
        except RuntimeError as exc:
            del trajectory[original_length:]
            if "ASYNC_CONTACT_QUALITY_REJECT" not in str(exc) or attempt == 2:
                raise
            print("[V102 CONTACT TIMING RETRY]", "reason=", str(exc),
                  "next_stagger_scale=", (0.5, 1.5)[attempt], "safety_limits_unchanged=True")


def v103_contact_rollout(solver, arcs, mode, q, fingers, working, lower, upper,
                        lane_spec, back_points, back_normals, back_tree, back_mesh_query,
                        stats, accept_frame=None, seconds=25., depth_cap_m=None,
                        priority_order=None, primary_finger=None,
                        assist_target_depth=None, assist_quality=None):
    """Same contact controller for the coarse screen and actual path builder.

    Preview uses point-cloud relations; actual builder supplies triangle mesh
    and whole-hand swept probes. Neither preview success nor depth targets
    bypass the complete production trajectory validator.
    """
    rollout_started = time.perf_counter()
    planned_depth_cap = float(
        V103_PLANNED_DEPTH_MAX_M if depth_cap_m is None else depth_cap_m
    )
    if not np.isfinite(planned_depth_cap) or planned_depth_cap <= 0.0:
        raise ValueError('V103_PLANNED_DEPTH_CAP_INVALID')
    allowed_planned_cap = (
        abs(float(HARD_PENETRATION_M)) - 1.0e-6
        if bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW)
        else float(V103_PLANNED_DEPTH_MAX_M)
    )
    if planned_depth_cap > allowed_planned_cap + 1e-12:
        raise ValueError('V103_PLANNED_DEPTH_CAP_EXCEEDS_HARD_STOP')
    active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)
    # The contact rollout owns the quality callbacks below, so its effective
    # depth floors must be computed in this scope.  The acquisition planner
    # has a similarly named local mapping, but Python does not share locals
    # across functions.  Recompute the tilt-aware floors from the actual
    # rollout entry pose; this keeps the preferred 3 mm target separate from
    # the pose-dependent minimum and avoids a late NameError when CONTACT_WAVE
    # invokes ``qualified``/``finish``.
    try:
        _, palm_r_for_quality = solver.compute_forward_kinematics(
            PALM_FRAME, np.asarray(q, dtype=np.float64)
        )
        quality_angle_deg = float(palm_back_alignment(
            arcs,
            np.asarray(palm_r_for_quality, dtype=np.float64),
            lane_spec['normal'],
        )['angle_deg'])
    except Exception as exc:
        # A diagnostic floor must never make rollout construction fail before
        # the normal safety/IK checks run.  The helper falls back to the
        # conservative non-tilt floor for non-finite angles.
        quality_angle_deg = float('nan')
        print('[V103 QUALITY TILT FLOOR FALLBACK]',
              'reason=', type(exc).__name__, str(exc))
    quality_minimums_m = {
        finger: v103_quality_min_depth_m(quality_angle_deg, finger)
        for finger in V103_MANDATORY_QUALITY_FINGERS
    }
    print('[V103 CONTACT QUALITY TILT FLOORS]',
          'palm_angle_deg=', (
              round(quality_angle_deg, 3)
              if np.isfinite(quality_angle_deg) else 'nan'
          ),
          'minimums_mm=', {
              finger: round(value * 1000.0, 3)
              for finger, value in quality_minimums_m.items()
          })
    # Pinky stays in the same four-finger rollout.  It is not removed from
    # ``working`` and is never converted into a 3F plan; only its final depth
    # quality gate is relaxed after the other three fingers are complete.
    cooperative_pinky = False
    checked_fingers = list(working)
    def measure(cq, commands):
        p, r = solver.compute_forward_kinematics(PALM_FRAME, cq)
        p, r = np.asarray(p), np.asarray(r)
        state = dict(
            signed={}, distance={}, cross={}, radius={},
            # The contact-wave controller uses these optional fields to
            # estimate how much of a curl step is actually normal to the
            # local BackSurface.  They are diagnostic/control geometry only;
            # signed depth remains the authoritative quality measurement.
            tip={}, normal={},
            valid=True, reason='PASS')
        reasons = []
        for f in checked_fingers:
            tip = p + r @ interpolate_arc(arcs, f, commands[f])[0]
            relation = final_surface_relation(tip, back_mesh_query, back_points, back_normals, back_tree)
            lane = lane_relation_for_world_point(tip, lane_spec, f)
            signed, distance = float(relation['signed']), float(relation['distance'])
            state['signed'][f], state['distance'][f] = signed, distance
            state['tip'][f] = np.asarray(tip, dtype=np.float64).copy()
            state['normal'][f] = np.asarray(
                relation.get('normal', np.zeros(3, dtype=np.float64)),
                dtype=np.float64,
            ).copy()
            state['cross'][f], state['radius'][f] = float(lane['cross_error']), float(lane['patch_radius'])
            if signed < -planned_depth_cap:
                reasons.append('DEPTH:' + f)
            # A straight/pending tip may pass through the broad 8 mm
            # neighborhood while it is still outside the selected patch.  Do
            # not reject that approach merely because its nearest point is
            # laterally outside the active lane; lane/region constraints start
            # when the tip actually establishes contact (or is already being
            # held in contact).  This is important for the V103 partial 2 mm
            # handoff: the two untouched fingers can be near the surface while
            # the remaining fingers finish their own acquisition.
            lane_required = bool(
                signed <= 0.0 or (
                    stats.get(f, {}).get('state') == 'CONTACT' and signed <= V103_SAFETY_CLEARANCE_M
                )
            )
            if distance <= WARM_CONTACT_MAX_DISTANCE_M and lane_required:
                if state['cross'][f] > active_cross_limit:
                    reasons.append('LANE:' + f)
                if state['radius'][f] > active_radius_limit:
                    reasons.append('REGION:' + f)
        state['angle'] = palm_back_alignment(arcs, r, lane_spec['normal'])['angle_deg']
        state['root_angle'] = root_base_axis_alignment(arcs, p, r)['angle_deg']
        state['angle_limit'] = palm_alignment_limit_deg()
        state['root_angle_limit'] = ROOT_BASE_AXIS_HARD_DEG
        if state['angle'] > palm_alignment_limit_deg():
            reasons.append('PALM_ALIGNMENT')
        if state['root_angle'] > ROOT_BASE_AXIS_HARD_DEG:
            reasons.append('ROOT_BASE_AXIS_ALIGNMENT')
        state['valid'], state['reason'] = not reasons, '|'.join(reasons)
        return state

    def safe(q0, f0, q1, f1):
        # Balance must hold throughout the interpolation, not just at a target.
        # Use the same density as the final contact-quality replay.  A fixed
        # 19-point sweep could miss a deep interior excursion on a larger
        # handoff step, causing an expensive candidate to pass here and then
        # be rejected minutes later by validate_continuous_scratch_quality.
        progress_delta = max(
            abs(float(f1.get(f, 0.0)) - float(f0.get(f, 0.0)))
            for f in working
        )
        arm_delta = float(np.max(np.abs(np.degrees(np.asarray(q1) - np.asarray(q0)))))
        sweep_samples = max(
            19,
            int(math.ceil(progress_delta / ASYNC_CONTACT_SAMPLE_PROGRESS_PCT)) + 1,
            int(math.ceil(arm_delta / 0.5)) + 1,
        )
        for t in np.linspace(0.,1.,sweep_samples):
            cq = np.asarray(q0)+t*(np.asarray(q1)-np.asarray(q0))
            cf = {f:f0[f]+t*(f1[f]-f0[f]) for f in f0}
            state = measure(cq,cf)
            if not state['valid']:
                return False, ('CONTACT_SWEEP:' + state['reason']
                               + '|alpha=' + str(round(float(t),3))
                               + '|predicted_depth_mm=' + str({
                                   f:round(max(0.,-s)*1000.,3) for f,s in state['signed'].items()}))
            if any(s < -planned_depth_cap for s in state['signed'].values()):
                return False, 'BALANCED_SWEPT_DEPTH_CAP'
            if (HAND_COLLISION_MODEL is not None
                    and hasattr(HAND_COLLISION_MODEL,'base_guard')
                    and not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))):
                gap, link = HAND_COLLISION_MODEL.base_guard.clearance(cq,cf)
                if gap < V103_BASE_MIN_CLEARANCE_M:
                    return False, ('HAND_BASE_MARGIN_WAVE|link=' + str(link)
                                   + '|gap_mm=' + str(round(gap*1000.,3))
                                   + '|alpha=' + str(round(float(t),3)))
        if back_mesh_query is not None:
            return segment_mesh_safe(solver, arcs, q0, f0, q1, f1,
                back_points, back_normals, back_tree, back_mesh_query,
                samples=18, allowed_contact_fingers=set(checked_fingers))
        # Preview only: finite interpolated tip checks, never whole-hand proof.
        for t in np.linspace(0., 1., 9):
            state = measure(q0+t*(q1-q0), {f: f0[f]+t*(f1[f]-f0[f]) for f in working})
            if not state['valid']:
                return False, state['reason']
        return True, 'PREVIEW_TIPS_ONLY'

    def accept(q0, f0, q1, f1, duration, focus, level, state):
        stage = 'SCRATCH_WAVE_ASYNC_CONTACT_PREVIEW'
        a = dict(q=q0, fingers=f0, stage=stage, duration=0.)
        b = dict(q=q1, fingers=f1, stage=stage, duration=duration)
        sample_scratch_contact_segment(stats, solver, arcs, a, b,
                                      back_points, back_normals, back_tree, back_mesh_query)
        if accept_frame is not None:
            accept_frame(q1, f1, duration, focus, level, state)

    def contact_objective(q0, f0, q1, f1):
        anchor_finger = mode.get('v103_anchor_finger')
        if anchor_finger is None:
            return []
        # Reuse the SAME moving-contact integration as final acceptance.
        # Work on a copy: optimizer probes must never earn contact credit.
        trial = copy.deepcopy(stats)
        sample_scratch_contact_segment(trial,solver,arcs,
            dict(q=q0,fingers=f0,stage='SCRATCH_WAVE_ASYNC_OBJECTIVE',duration=0.),
            dict(q=q1,fingers=f1,stage='SCRATCH_WAVE_ASYNC_OBJECTIVE',duration=.1),
            back_points,back_normals,back_tree,back_mesh_query)
        row = trial[anchor_finger]
        weight = float(row.get('contact_position_weight',0.))
        if weight <= 1e-12:
            # Approach the target before earning contact; fixed residual size.
            p,r = solver.compute_forward_kinematics(PALM_FRAME,q1)
            center = np.asarray(p)+np.asarray(r) @ interpolate_arc(arcs,anchor_finger,f1[anchor_finger])[0]
        else:
            center = np.asarray(row['contact_position_sum'])/weight
        normal = np.asarray(lane_spec['normal']); normal=normal/np.linalg.norm(normal)
        delta = center-np.asarray(mode['anchor'])
        return ((delta-np.dot(delta,normal)*normal)/.003).tolist()

    def cooperate(q0, f0, q1, f1):
        return CONTACT_WAVE.cooperative_pinky(q0, f0, q1, f1, measure, safe,
                                               V103_DEPTH_TARGET_M)

    result = CONTACT_WAVE.roll(q=q, commands=fingers, fingers=working,
        cooperate=cooperate if set(working) == set(LONG_FINGERS) else None,
        optional_fingers=tuple(f for f in working if f in V103_OPTIONAL_QUALITY_FINGERS),
        mandatory_fingers=tuple(f for f in working if f in V103_MANDATORY_QUALITY_FINGERS),
        freeze_priority_order=True,
        optional_patience_steps=V103_PINKY_LOCAL_OPTIMUM_PATIENCE_STEPS,
        optional_patience_s=V103_PINKY_LOCAL_OPTIMUM_PATIENCE_S,
         stop_when_qualified=True,
         # Do not let CONTACT_WAVE run its legacy release-tail/J5 handoff
         # before the scratch curl.  The caller appends the four-finger curl
         # from this qualified pose, then performs the measured release.
         stop_after_qualification=True,
        lower=np.asarray(lower)+math.radians(MIN_JOINT_MARGIN_DEG),
        upper=np.asarray(upper)-math.radians(MIN_JOINT_MARGIN_DEG),
        measure=measure, qualified=lambda f: v103_contact_quality_failure(
            stats[f], quality_minimums_m.get(f)
        ) is None,
        accept=accept, safe=safe, speed=SCRATCH_CURL_FINGER_SPEED_DEG_S,
        contact_objective=contact_objective if mode.get('v103_anchor_finger') else None,
        arm_speed=ARM_MAX_SPEED_DEG_S, seconds=seconds,
        depth_max=v103_effective_runtime_depth_cap_m(),
        target_depth=ASYNC_PLANNED_DEPTH_TARGET_M,
        finish=lambda cq,cf: v103_early_exit_evidence(solver,arcs,cq,cf,stats,
                        back_points,back_normals,back_tree,back_mesh_query,
                        required_fingers=working,
                        minimums_m=quality_minimums_m),
        # A finished finger curls away during the next focus handoff; the
        # rollout's measured release rate and swept guard authorize that move.
        # The tilt-aware core minimum is the contact-quality floor; 3 mm is
        # still preferred and the 4 mm upper cap remains hard.
        released=lambda f,state: stats[f]['state']=='RELEASED' and state['signed'][f]>ASYNC_CONTACT_EXIT_CLEARANCE_M,
        # The acquisition state is updated only by accepted, sampled contact
        # motion.  An airborne curl never earns contact credit.
        acquiring=lambda f: stats[f]['state']=='UNTOUCHED',
        base_clearance=(HAND_COLLISION_MODEL.base_guard.clearance
                        if HAND_COLLISION_MODEL is not None and hasattr(HAND_COLLISION_MODEL,'base_guard') else None),
        base_guard_enabled=not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False)),
        # Dynamic order is measured from the live BackSurface, but the
        # handoff is serial: keep the current finger's contact/depth stable
        # before allowing the next finger to advance.  This prevents the old
        # ring->middle overlap that later forced a large runtime rejoin.
        sequential_handoff=True,
        priority_order=priority_order,
        primary_finger=primary_finger,
        assist_target_depth=assist_target_depth,
        assist_quality=assist_quality)
    if result.get('early_exit'):
        # All fingers have already earned moving contact and released. The
        # original 1.5mm trigger is only a handoff, not enough execution reserve.
        # Build a checked release tail before exposing an early-exit keyframe.
        release = CONTACT_WAVE.prepare_exit(q=result['q'],commands=result['commands'],
            # Only J4/J5 move here. J1-J3 retain their already-validated pose;
            # check their physical bounds without imposing a new endpoint
            # preference on a held J3 close to its limit.
            fingers=working,lower=np.asarray(lower)+np.radians([0.,0.,0.,MIN_JOINT_MARGIN_DEG,MIN_JOINT_MARGIN_DEG]),
            upper=np.asarray(upper)-np.radians([0.,0.,0.,MIN_JOINT_MARGIN_DEG,MIN_JOINT_MARGIN_DEG]),measure=measure,safe=safe,
            base_clearance=HAND_COLLISION_MODEL.base_guard.clearance,
            base_guard_enabled=not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False)),
            stop_on_base=False,
            base_contact_tolerance=V103_BASE_CONTACT_TOLERANCE_M,
            skin_clearance=V103_EXIT_PLANNED_RESERVE_M,base_margin=V103_EXIT_PLANNED_BASE_RESERVE_M,
            max_lift_deg=V103_EXIT_EXTRA_J5_MAX_DEG,max_curl_extra_pct=V103_EXIT_EXTRA_CURL_MAX_PCT)
        cq,cf = release['q'],release['commands']
        duration = max(.08, max(release['extra_curl_pct'].values())/SCRATCH_CURL_FINGER_SPEED_DEG_S,
                       float(np.max(np.abs(np.degrees(cq-result['q']))))/ARM_MAX_SPEED_DEG_S)
        accept(result['q'],result['commands'],cq,cf,duration,None,'ALL_FINGER_EXIT_RESERVE',release['state'])
        result['q'],result['commands'] = cq,cf
        result['frames'].append(dict(q=cq.copy(),fingers=dict(cf),duration=duration,
                                     focus=None,level='ALL_FINGER_EXIT_RESERVE'))
        evidence = v103_early_exit_evidence(solver,arcs,cq,cf,stats,
                            back_points,back_normals,back_tree,back_mesh_query,
                            required_skin_clearance=V103_EXIT_PLANNED_RESERVE_M,
                            required_fingers=working,
                            base_contact_stop=bool(release.get('base_contact_stop', False)),
                            minimums_m=quality_minimums_m)
        if evidence is None:
            raise RuntimeError('ALL_FINGER_EXIT_EVIDENCE_LOST')
        result['early_exit'] = dict(evidence, all_finger_reserve=True,
            base_contact_stop=bool(release.get('base_contact_stop', False)),
            base_contact_link=release.get('base_contact_link'),
            signed_mm={f:round(release['state']['signed'][f]*1000.,3) for f in working},
            extra_curl_pct=release['extra_curl_pct'],extra_j5_deg=release['j5_lift_deg'],
            extra_j4_deg=release['j4_trim_deg'])
        print('[V103 ALL FINGER EXIT RESERVE]',result['early_exit'])
    relaxed_fingers = set(result.get('relaxed_fingers', ()))
    result['quality'] = {f: dict(depth_mm=round(max(0., -stats[f]['min_signed'])*1000., 3),
                                slide_mm=round(stats[f]['slide_m']*1000., 3),
                                moving_ms=round(stats[f]['moving_contact_s']*1000., 3),
                                failure=(None if f in relaxed_fingers
                                         else (v103_contact_quality_failure(
                                                   stats[f], quality_minimums_m.get(f)
                                               )
                                               if f in V103_MANDATORY_QUALITY_FINGERS
                                               else v103_assist_quality_failure(stats[f]))),
                                motion_diagnostics=scratch_contact_motion_diagnostics(stats[f]))
                         for f in working}
    if relaxed_fingers:
        print('[V103 ASSIST LOCAL OPTIMUM]',
              'relaxed_fingers=', sorted(relaxed_fingers),
              'depth_mm=', {f: result['quality'][f]['depth_mm'] for f in relaxed_fingers},
              'hard_cap_mm=', round(V103_PLANNED_DEPTH_MAX_M*1000., 3),
              'trajectory_fingers=', list(working),
              'primary_depth_anchor=', primary_finger,
              'assist_policy=PREFERRED_3MM_RELAXED_2_TO_4MM_LOCAL_OPTIMUM')
    if mode.get('anchor') is not None:
        centers = {f:np.asarray(stats[f]['contact_position_sum'])/stats[f]['contact_position_weight']
                   for f in working if stats[f].get('contact_position_weight',0.)>1e-12}
        if len(centers)==len(working):
            normal = np.asarray(lane_spec['normal'],dtype=float)
            normal = normal/np.linalg.norm(normal)
            def tangent_distance(p):
                d = np.asarray(p)-np.asarray(mode['anchor'])
                return float(np.linalg.norm(d-np.dot(d,normal)*normal))*1000.
            result['contact_location'] = dict(source='planned_moving_contact',
                center_error_mm=round(tangent_distance(np.mean(list(centers.values()),axis=0)),3),
                finger_center_distance_mm={f:round(tangent_distance(p),3) for f,p in centers.items()})
            print('[V103 EFFECTIVE CONTACT LOCATION]',result['contact_location'])
            contact_palm_angle_deg = float('nan')
            try:
                _contact_palm_p, _contact_palm_r = solver.compute_forward_kinematics(
                    PALM_FRAME, result['q']
                )
                contact_palm_angle_deg = float(
                    palm_back_alignment(
                        arcs, _contact_palm_r, lane_spec['normal']
                    )['angle_deg']
                )
            except Exception:
                pass
            effective_center_limit_m = v103_effective_contact_center_tolerance_m(
                contact_palm_angle_deg
            )
            result['contact_location']['palm_angle_deg'] = (
                round(contact_palm_angle_deg, 3)
                if np.isfinite(contact_palm_angle_deg) else None
            )
            result['contact_location']['center_limit_mm'] = round(
                effective_center_limit_m * 1000.0, 3
            )
            print(
                '[V103 EFFECTIVE CONTACT CENTER LIMIT]',
                'palm_angle_deg=', (
                    round(contact_palm_angle_deg, 3)
                    if np.isfinite(contact_palm_angle_deg) else None
                ),
                'limit_mm=', round(effective_center_limit_m * 1000.0, 3),
                'base_limit_mm=', round(V103_EFFECTIVE_CONTACT_CENTER_MAX_M * 1000.0, 3),
            )
            if result['contact_location']['center_error_mm'] > effective_center_limit_m * 1000.0:
                raise RuntimeError(
                    'EFFECTIVE_CONTACT_CENTER_TOO_FAR_mm=' +
                    str(result['contact_location']['center_error_mm'])
                )
            anchor_finger = mode.get('v103_anchor_finger')
            if (anchor_finger is not None
                    and result['contact_location']['finger_center_distance_mm'].get(
                        anchor_finger, 999.0) > V103_MIDDLE_ANCHOR_MAX_M * 1000.0):
                raise RuntimeError(
                    'EFFECTIVE_ANCHOR_FINGER_TOO_FAR_' + str(anchor_finger) + '_mm=' +
                    str(result['contact_location']['finger_center_distance_mm'].get(anchor_finger))
                )
    result['seconds'] = time.perf_counter()-rollout_started
    return result


def v103_optional_pinky_decision(stats, distances, base_gap):
    """Return True only for the narrow, auditable near-base 3F fallback."""
    required = [f for f in LONG_FINGERS if f != 'pinky']
    return bool(
        stats['pinky']['state'] == 'UNTOUCHED'
        and math.isfinite(float(base_gap))
        and float(base_gap) <= V103_OPTIONAL_PINKY_BASE_ZONE_M
        and math.isfinite(float(distances['pinky']))
        and float(distances['pinky']) >= V103_OPTIONAL_PINKY_MIN_GAP_M
        and all(math.isfinite(float(distances[f])) for f in required)
        and float(distances['pinky']) >= max(float(distances[f]) for f in required)
    )


def append_v103_contact_wave(trajectory, solver, arcs, mode, q_now, fingers,
                            working_fingers, shallow_to_deep, endpoint_q, lower, upper,
                            back_points, back_normals, back_tree, back_mesh_query,
                            lane_spec, cycle_index):
    if back_mesh_query is None:
        raise RuntimeError('V103_CONTACT_WAVE_REQUIRES_TRIANGLE_MESH')
    working = [f for f in LONG_FINGERS if f in working_fingers]
    if len(working) != len(LONG_FINGERS) or set(working) != set(LONG_FINGERS):
        raise RuntimeError('V103_CONTACT_WAVE_REQUIRES_FOUR_FINGERS')
    all_stats = acquisition_scratch_contact_stats(trajectory, working, solver, arcs,
                                             back_points, back_normals, back_tree, back_mesh_query)
    print('[V103 PLANNED FOUR-FINGER SCRATCH]',
          'required_fingers=', working,
          'core_required=', list(V103_MANDATORY_QUALITY_FINGERS),
          'optional_quality=', list(V103_OPTIONAL_QUALITY_FINGERS),
          'core_depth_gate=LIVE_2_TO_4MM_3MM_PREFERRED',
          'runtime_planned_depth_cap_mm=', round(v103_effective_runtime_depth_cap_m()*1000., 3),
          'normal_quality_cap_mm=', round(V103_PLANNED_DEPTH_MAX_M*1000., 3),
          'diagnostic_depth_overflow=', bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW),
          'three_finger_fallback=DISABLED',
          'handoff=FROZEN_INITIAL_ORDER_COUPLED_BACKGROUND',
          'natural_minimum=FIRST_OWNER_FIXED_PALM_3MM_LATER_OWNERS_ROLLING_PREDICTION',
          'dynamic_release=WRIST_COUPLED_CURL_AWAY')
    stats = {f:all_stats[f] for f in working}
    # Acquisition already froze the initial real-BackSurface order. Preserve
    # it exactly; quality optionality does not reorder fingers.
    priority_order = [f for f in shallow_to_deep if f in working]
    for f in working:
        if f not in priority_order:
            priority_order.append(f)
    primary_finger = next((f for f in priority_order
                           if f in V103_MANDATORY_QUALITY_FINGERS),
                          priority_order[0] if priority_order else None)
    optional = [f for f in working if f in V103_OPTIONAL_QUALITY_FINGERS]
    assist_relaxed_fingers = list(optional)
    print('[V103 CONTACT DYNAMIC ORDER SEED]',
          'order=', priority_order,
          'primary_depth_anchor=', primary_finger,
          'completion_order_frozen=True',
          'pinky_optional_quality_only=True',
          'dynamic_metric_used_for_delays_sensitivity_only=True')
    cycle_trajectory_start = len(trajectory)
    count = 0
    def append(q, commands, duration, focus, level, state):
        nonlocal count
        count += 1
        active = [f for f in working if stats[f]['state'] == 'CONTACT']
        add_keyframe(trajectory, 'SCRATCH_WAVE_ASYNC_%d_CYCLE_%d' % (count, cycle_index),
                     duration, q, commands, active,
                     authorized_contact_fingers=set(active))
        # accept() samples the accepted segment BEFORE invoking this callback.
        # Snapshot all states now: stats is mutable across steps and cycles.
        # Authorization permits planned contact, it does not assert contact.
        trajectory[-1]['predicted_contact_states'] = {f: stats[f]['state'] for f in working}
        trajectory[-1]['contact_state_source'] = 'sampled_segment_v1'
        trajectory[-1]['scratch_required_fingers'] = list(working)
        trajectory[-1]['scratch_optional_fingers'] = list(optional)
        trajectory[-1]['scratch_relaxed_fingers'] = list(assist_relaxed_fingers)
        trajectory[-1]['scratch_primary_finger'] = primary_finger
        trajectory[-1]['scratch_depth_policy'] = 'CORE3_LIVE_1_TO_4MM_3MM_PREFERRED_PINKY_LOCAL_OPTIMUM'
        trajectory[-1]['trajectory_fingers'] = list(LONG_FINGERS)
        # Runtime tracking uses the same dynamic focus that produced this
        # frame.  It must not pick a stale, already-completed finger as the
        # "worst" lagging joint while the next finger is being handed off.
        trajectory[-1]['dynamic_focus'] = focus
        trajectory[-1]['dynamic_handoff_policy'] = 'DYNAMIC_MEASURED_ORDER_SOFT_FOCUS_LOCK'
        trajectory[-1]['dynamic_release_policy'] = 'ROLLING_WRIST_COUPLED_CURL_AWAY'
        print('[V103 CONTACT STEP]', 'cycle=', cycle_index, 'step=', count,
              'focus=', focus, 'solver=', level,
              'q_deg=', np.round(np.degrees(q), 2),
              'progress_pct=', {f: round(commands[f], 2) for f in working},
              'depth_mm=', {f: round(max(0., -state['signed'][f])*1000., 2) for f in working},
              'base_gap_mm=', round(state['base_gap']*1000.,3) if 'base_gap' in state else None,
              'closest_link=', state.get('base_link'),
              'contact_states=', trajectory[-1]['predicted_contact_states'])
    result = v103_contact_rollout(solver, arcs, mode, q_now, fingers, working, lower, upper,
        lane_spec, back_points, back_normals, back_tree, back_mesh_query, stats, append,
        # The explicit +20 mm handoff-clearance tail can require roughly
        # forty measured release frames after the last successor qualifies.
        # Keep the same safety/step limits, but give this release-only tail
        # enough wall-clock budget instead of cutting it at the old 60 s
        # contact budget.
        seconds=120., depth_cap_m=v103_effective_runtime_depth_cap_m(),
        priority_order=priority_order, primary_finger=primary_finger,
        # Assist fingers are still optimized toward 3 mm.  Their relaxed
        # policy applies only to the completion decision after the bounded
        # search; 0.5 mm is a diagnostic/local-optimum floor, never a normal
        # assist success gate.
        assist_target_depth=ASYNC_PLANNED_DEPTH_TARGET_M,
        assist_quality=lambda f: scratch_contact_assist_target_reached(stats[f]))
    relaxed = list(result.get('relaxed_fingers', ()))
    if result.get('depth_complete') and len(trajectory) == cycle_trajectory_start:
        # The three core fingers may already be qualified at acquisition
        # handoff.  Still emit one explicit four-finger wave frame so the
        # pinky command channel is represented in the executable trajectory;
        # never mark an acquisition frame as scratch-complete by accident.
        active = [f for f in working if stats[f]['state'] == 'CONTACT']
        add_keyframe(trajectory, 'SCRATCH_WAVE_ASYNC_LOCAL_OPTIMUM_CYCLE_%d' % int(cycle_index),
                     0.05, result['q'], result['commands'], active,
                     authorized_contact_fingers=set(active))
        trajectory[-1]['predicted_contact_states'] = {f: stats[f]['state'] for f in working}
        trajectory[-1]['contact_state_source'] = 'sampled_segment_v1'
        trajectory[-1]['scratch_required_fingers'] = list(working)
        trajectory[-1]['scratch_optional_fingers'] = list(optional)
        trajectory[-1]['scratch_relaxed_fingers'] = sorted(
            set(assist_relaxed_fingers) | set(relaxed))
        trajectory[-1]['scratch_primary_finger'] = primary_finger
        trajectory[-1]['scratch_depth_policy'] = 'CORE3_LIVE_1_TO_4MM_3MM_PREFERRED_PINKY_LOCAL_OPTIMUM'
        trajectory[-1]['trajectory_fingers'] = list(LONG_FINGERS)
    for frame in trajectory[cycle_trajectory_start:]:
        if str(frame.get('stage', '')).startswith('SCRATCH_WAVE_ASYNC_'):
            frame['scratch_relaxed_fingers'] = sorted(
                set(assist_relaxed_fingers) | set(relaxed))
    if result.get('depth_complete'):
        # Contact qualification is only the start of the release phase.  It is
        # deliberately *not* scratch completion: the old marker here let the
        # runtime treat a 1.5 mm safety clearance (or any 1--4 mm contact) as
        # the end of the stroke and lift the wrist before the +20 mm curl-away.
        print('[V103 CONTACT QUALIFIED]', 'cycle=', cycle_index,
              'progress_pct=', {f:round(float(result['commands'][f]),2) for f in working},
              'contact_qualified=True',
               'next=FULL_FOUR_FINGER_CURL_THEN_MEASURED_RELEASE_TO_20MM')

        # Release fingers independently. A pinky local optimum is still part
        # of this four-finger trajectory; it is never converted to a 3F plan.
        release_start = dict(result['commands'])
        release_commands = dict(release_start)
        release_q = np.asarray(result['q'], dtype=np.float64).copy()
        release_reason = 'NO_CANDIDATE'
        active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)
        def release_segment_safe(q0, f0, q1, f1):
            # Use the SAME active lane/65 mm patch reserve during release that
            # the final trajectory validator uses. Pure finger curl that would
            # leave the patch is rejected here, early enough to try wrist
            # unload instead of failing minutes later in final validation.
            progress_delta = max(abs(float(f1[f])-float(f0[f])) for f in working)
            arm_delta = float(np.max(np.abs(np.degrees(np.asarray(q1)-np.asarray(q0)))))
            samples = max(24, int(math.ceil(progress_delta / ASYNC_CONTACT_SAMPLE_PROGRESS_PCT)) + 1,
                          int(math.ceil(arm_delta / 0.5)) + 1)
            for alpha in np.linspace(0.0, 1.0, samples):
                cq = np.asarray(q0) + float(alpha) * (np.asarray(q1) - np.asarray(q0))
                cf = {f: float(f0[f]) + float(alpha) * (float(f1[f]) - float(f0[f]))
                      for f in f0}
                palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, cq)
                if not PALM_POLICY.angle_ok(float(palm_back_alignment(
                        arcs, palm_r, lane_spec['normal'])['angle_deg'])):
                    return False, 'RELEASE_PALM_ALIGNMENT'
                root_angle = float(root_base_axis_alignment(arcs, palm_p, palm_r)['angle_deg'])
                if root_angle > ROOT_BASE_AXIS_HARD_DEG:
                    return False, 'RELEASE_ROOT_BASE_AXIS_ALIGNMENT'
                for f in working:
                    tip = np.asarray(palm_p) + np.asarray(palm_r) @ interpolate_arc(arcs, f, cf[f])[0]
                    relation = final_surface_relation(tip, back_mesh_query, back_points, back_normals, back_tree)
                    lane = lane_relation_for_world_point(tip, lane_spec, f)
                    signed = float(relation['signed'])
                    if signed < -v103_effective_runtime_depth_cap_m() - 1e-12:
                        return False, 'RELEASE_DEPTH:' + f
                    if float(relation['distance']) <= WARM_CONTACT_MAX_DISTANCE_M:
                        if float(lane['cross_error']) > active_cross_limit:
                            return False, 'RELEASE_LANE:' + f
                        if float(lane['patch_radius']) > active_radius_limit:
                            return False, 'RELEASE_REGION:' + f
            return segment_mesh_safe(
                solver, arcs, q0, f0, q1, f1,
                back_points, back_normals, back_tree, back_mesh_query,
                samples=max(40, samples), allowed_contact_fingers=set(LONG_FINGERS))
        initial_clearance = active_finger_mesh_clearance(
            solver, arcs, release_q, release_commands, LONG_FINGERS,
            back_points, back_normals, back_tree, back_mesh_query)
        initial_signed = {f:float(initial_clearance['details'][f]['signed'])
                          for f in LONG_FINGERS}
        release_required = [
            f for f in LONG_FINGERS
            if f in working or initial_signed[f] <= 0.0
        ]
        # Deepest/contacting tips release first. Optional airborne fingers are
        # deliberately absent from this list and keep their current command.
        release_order = sorted(release_required, key=lambda f: initial_signed[f])
        # This is the experiment's release-completion target.  The 1.5 mm
        # skin-clearance constant remains a safety/lane hysteresis margin, but
        # must never decide that the scratch stroke is finished.
        release_target_signed = float(getattr(
            CONTACT_WAVE, 'HANDOFF_RELEASE_TARGET_SIGNED_M', .020))
        per_finger_extra = dict.fromkeys(LONG_FINGERS, 0.0)

        def release_future_j5_guard(q0, commands, finger_label, context):
            """Reject a release that is safe now but unsafe after the planned J5 lift.

            The runtime unload stage intentionally lifts the wrist after all
            finger-release keyframes.  Because J5 is coupled to the fingertip
            normal, validating only the release endpoint can accept a state
            that immediately re-enters deep-adjust on the following lift.
            Preview the first unload lift (the one that will actually be tried
            at runtime) and keep the existing 3.5 mm hysteresis exit as a
            conservative guard.  This does not relax any collision, lane, or
            hard-depth checks; it only rejects a coupled overshoot earlier.
            """
            q0 = np.asarray(q0, dtype=np.float64).copy()
            q_probe = q0.copy()
            available_deg = math.degrees(max(0.0, float(upper[4] - q0[4])))
            preview_lift_deg = min(float(UNLOAD_J5_MIN_LIFT_DEG), available_deg)
            if preview_lift_deg <= 1e-6:
                print('[V103 RELEASE COUPLED J5 CHECK]',
                      'finger=', finger_label, 'context=', context,
                      'preview_lift_deg=', round(float(preview_lift_deg), 3),
                      'accepted=False', 'reason=J5_LIMIT')
                return False
            q_probe[4] += math.radians(preview_lift_deg)

            # Keep the same swept-path/lane/patch checks used by release
            # itself.  This also catches a J5-induced lateral drift before the
            # later unload segment reaches PhysX.
            sweep_ok, sweep_reason = release_segment_safe(
                q0, commands, q_probe, commands)
            projected = active_finger_mesh_clearance(
                solver, arcs, q_probe, commands, LONG_FINGERS,
                back_points, back_normals, back_tree, back_mesh_query)
            projected_min_signed = float(projected.get('minimum_signed', float('nan')))
            depth_ok = (math.isfinite(projected_min_signed)
                        and projected_min_signed >= DEEP_ADJUST_EXIT_SIGNED_M - 1e-12)
            accepted = bool(sweep_ok and depth_ok)
            print('[V103 RELEASE COUPLED J5 CHECK]',
                  'finger=', finger_label, 'context=', context,
                  'preview_lift_deg=', round(float(preview_lift_deg), 3),
                  'projected_min_signed_mm=',
                  round(projected_min_signed * 1000.0, 3),
                  'sweep=', sweep_reason,
                  'accepted=', accepted,
                  'reason=', 'PASS' if accepted else
                   ('J5_PROJECTED_DEEP' if not depth_ok else sweep_reason))
            return accepted

        # Contact qualification is a phase boundary, not the end of the
        # scratch stroke.  Continue the four finger curl from the actual
        # qualified pose BEFORE any optional-local-optimum exit reserve is
        # allowed to lift J5.  This ordering is important: otherwise the
        # hand visibly returns to a lifted/preposition pose and only then
        # curls, which is not the intended scratch motion.
        full_curl_q, full_curl_commands, full_curl_active = append_full_four_finger_curl(
            trajectory,
            solver,
            arcs,
            release_q,
            release_commands,
            working,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            cycle_index,
        )
        release_q = np.asarray(full_curl_q, dtype=np.float64).copy()
        release_commands = dict(full_curl_commands)
        result['q'] = release_q.copy()
        result['commands'] = dict(release_commands)
        result['frames'].append(dict(
            q=release_q.copy(),
            fingers=dict(release_commands),
            duration=0.0,
            focus=None,
            level='SCRATCH_FULL_CURL_CONTINUATION',
        ))
        print('[V103 FULL CURL CONTINUATION]',
              'cycle=', cycle_index,
              'commands=', {f: round(float(release_commands[f]), 2)
                            for f in working},
              'next=OPTIONAL_RESERVE_THEN_MEASURED_RELEASE_TO_20MM')

        # A pinky local optimum relaxes only the contact-quality credit.  It
        # must still get the same measured curl-away opportunity as a normal
        # four-finger wave.  In particular, do not jump straight to the old
        # 1.5 mm per-finger release just because pinky did not touch.  First
        # try the wave's explicit +20 mm reserve; if this local geometry
        # cannot provide it, retain the established 6 mm planning reserve and
        # let the runtime measured J5/unload tail finish the clearance.  Both
        # attempts use the existing swept mesh, lane, patch, depth-cap, and
        # joint-limit checks.
        if result.get('local_optimum'):
            def reserve_measure(cq, cf):
                clearance = active_finger_mesh_clearance(
                    solver, arcs, cq, cf, LONG_FINGERS,
                    back_points, back_normals, back_tree, back_mesh_query)
                signed = {
                    f: float(clearance['details'][f]['signed'])
                    for f in LONG_FINGERS
                }
                return dict(
                    signed=signed,
                    valid=all(math.isfinite(value) for value in signed.values()),
                )

            reserve_base_clearance = (
                HAND_COLLISION_MODEL.base_guard.clearance
                if HAND_COLLISION_MODEL is not None
                and hasattr(HAND_COLLISION_MODEL, 'base_guard')
                else (lambda _q, _cf: (float('inf'), None))
            )
            reserve = None
            reserve_error = None
            reserve_targets = (
                float(getattr(CONTACT_WAVE,
                              'HANDOFF_RELEASE_TARGET_SIGNED_M', .020)),
                float(V103_EXIT_PLANNED_RESERVE_M),
            )
            for reserve_target in reserve_targets:
                try:
                    reserve = CONTACT_WAVE.prepare_exit(
                        q=result['q'], commands=result['commands'],
                        fingers=working,
                        lower=np.asarray(lower) + np.radians(
                            [0., 0., 0., MIN_JOINT_MARGIN_DEG, MIN_JOINT_MARGIN_DEG]),
                        upper=np.asarray(upper) - np.radians(
                            [0., 0., 0., MIN_JOINT_MARGIN_DEG, MIN_JOINT_MARGIN_DEG]),
                        measure=reserve_measure,
                        safe=release_segment_safe,
                        base_clearance=reserve_base_clearance,
                        base_guard_enabled=not bool(globals().get(
                            'BASE_PLANNING_GUARD_DISABLED', False)),
                        stop_on_base=False,
                        base_contact_tolerance=V103_BASE_CONTACT_TOLERANCE_M,
                        skin_clearance=reserve_target,
                        base_margin=V103_EXIT_PLANNED_BASE_RESERVE_M,
                        max_lift_deg=V103_EXIT_EXTRA_J5_MAX_DEG,
                        max_curl_extra_pct=V103_EXIT_EXTRA_CURL_MAX_PCT,
                    )
                except Exception as exc:
                    reserve_error = exc
                    print('[V103 OPTIONAL EXIT RESERVE TRY]',
                          'target_mm=', round(reserve_target * 1000., 3),
                          'accepted=False',
                          'reason=', type(exc).__name__ + ':' + str(exc))
                    continue
                print('[V103 OPTIONAL EXIT RESERVE TRY]',
                      'target_mm=', round(reserve_target * 1000., 3),
                      'accepted=True')
                break
            if reserve is not None:
                cq, cf = reserve['q'], reserve['commands']
                duration = max(
                    .08,
                    max(reserve['extra_curl_pct'].values()) /
                    SCRATCH_CURL_FINGER_SPEED_DEG_S,
                    float(np.max(np.abs(np.degrees(
                        np.asarray(cq) - np.asarray(result['q']))))) /
                    ARM_MAX_SPEED_DEG_S,
                )
                reserve_state = dict(reserve['state'])
                # This branch runs outside v103_contact_rollout(), so its
                # nested ``accept`` callback is not in scope.  Sample the
                # accepted reserve segment explicitly, then use the same
                # append callback as ordinary wave frames.  Without this,
                # every pinky local-optimum candidate raised
                # ``NameError: name 'accept' is not defined`` after a valid
                # release plan had already been found.
                reserve_a = dict(
                    q=np.asarray(result['q'], dtype=np.float64).copy(),
                    fingers=dict(result['commands']),
                    stage='SCRATCH_WAVE_ASYNC_CONTACT_PREVIEW',
                    duration=0.0,
                )
                reserve_b = dict(
                    q=np.asarray(cq, dtype=np.float64).copy(),
                    fingers=dict(cf),
                    stage='SCRATCH_WAVE_ASYNC_CONTACT_PREVIEW',
                    duration=duration,
                )
                sample_scratch_contact_segment(
                    stats, solver, arcs, reserve_a, reserve_b,
                    back_points, back_normals, back_tree, back_mesh_query,
                )
                append(cq, cf, duration, None,
                       'ALL_FINGER_EXIT_RESERVE_OPTIONAL_LOCAL', reserve_state)
                result['q'], result['commands'] = np.asarray(cq).copy(), dict(cf)
                # The per-finger compatibility tail below must start from the
                # accepted reserve pose; otherwise it would overwrite this
                # frame with the pre-reserve local-optimum state.
                release_q = np.asarray(cq).copy()
                release_commands = dict(cf)
                result['frames'].append(dict(
                    q=np.asarray(cq).copy(), fingers=dict(cf),
                    duration=duration, focus=None,
                    level='ALL_FINGER_EXIT_RESERVE_OPTIONAL_LOCAL',
                ))
                # The callback emits a SCRATCH_WAVE_ASYNC keyframe for shared
                # accounting. Carry the explicit completion/reserve metadata
                # on that executable frame so runtime tracking does not treat
                # this local-optimum reserve as an ordinary contact hold.
                trajectory[-1]['planner_level'] = 'ALL_FINGER_EXIT_RESERVE_OPTIONAL_LOCAL'
                trajectory[-1]['optional_local_optimum_exit_reserve'] = True
                trajectory[-1]['exit_reserve_target_mm'] = round(
                    min(float(reserve_target),
                        float(getattr(CONTACT_WAVE,
                                      'HANDOFF_RELEASE_TARGET_SIGNED_M', .020))) * 1000., 3)
                print('[V103 OPTIONAL EXIT RESERVE]',
                      'signed_mm=', {
                          f: round(float(reserve['state']['signed'][f]) * 1000., 3)
                          for f in working
                      },
                      'pinky_quality_relaxed=True',
                      'safety_limits_unchanged=True')
            else:
                print('[V103 OPTIONAL EXIT RESERVE FALLBACK]',
                      'reason=', type(reserve_error).__name__ + ':' + str(reserve_error)
                      if reserve_error is not None else 'NO_RESERVE_RESULT',
                      'fallback=PER_FINGER_RELEASE',
                      'safety_limits_unchanged=True')

        for finger in release_order:
            if (not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))
                    and HAND_COLLISION_MODEL is not None
                    and hasattr(HAND_COLLISION_MODEL, 'base_guard')):
                # Compatibility path for old callers. Production has this
                # guard disabled and uses the explicit contact-stop rule below.
                gap_now, link_now = HAND_COLLISION_MODEL.base_guard.clearance(
                    release_q, release_commands)
                if gap_now < V103_BASE_MIN_CLEARANCE_M:
                    raise RuntimeError('FINGER_CURL_RELEASE_BASE_MARGIN|link=' + str(link_now))
            current_relation = finger_surface_relation_at_state(
                solver, arcs, release_q, release_commands, finger,
                back_points, back_normals, back_tree, back_mesh_query)
            if float(current_relation['signed']) >= release_target_signed:
                continue
            finger_start = float(release_commands[finger])
            max_extra = FINGER_SAFE_COMMAND_MAX_DEG - finger_start
            extras = list(np.arange(2.0, min(40.0, max_extra) + 0.001, 2.0))
            extras += list(np.arange(45.0, max_extra + 0.001, 5.0))
            if max_extra > 0.0 and (not extras or extras[-1] < max_extra - 1e-6):
                extras.append(max_extra)
            accepted_commands = None
            accepted_relation = None
            for extra in extras:
                candidate = dict(release_commands)
                candidate[finger] = min(FINGER_SAFE_COMMAND_MAX_DEG,
                                        finger_start + float(extra))
                safe_release, release_reason = release_segment_safe(
                    release_q, release_commands, release_q, candidate)
                if not safe_release:
                    continue
                relation = finger_surface_relation_at_state(
                    solver, arcs, release_q, candidate, finger,
                    back_points, back_normals, back_tree, back_mesh_query)
                if float(relation['signed']) >= release_target_signed:
                    if not release_future_j5_guard(
                            release_q, candidate, finger, 'pure_curl'):
                        continue
                    accepted_commands = candidate
                    accepted_relation = relation
                    break
            accepted_q = release_q.copy()
            if accepted_commands is None:
                # Patch-aware coupled release: if pure curl would drift outside
                # the active patch, lift J5 while the finger continues curling.
                # The old code waited until final validation to discover the
                # 65.18 mm failure on 909.
                for j4_trim_deg in (0.0, 0.5, -0.5, 1.0, -1.0):
                    if accepted_commands is not None:
                        break
                    for j5_lift_deg in np.arange(0.5, 14.0 + 1e-9, 0.5):
                        if accepted_commands is not None:
                            break
                        cq = release_q.copy()
                        cq[3] += math.radians(float(j4_trim_deg))
                        cq[4] += math.radians(float(j5_lift_deg))
                        if np.any(cq < np.asarray(lower)-1e-10) or np.any(cq > np.asarray(upper)+1e-10):
                            continue
                        for extra in extras:
                            candidate = dict(release_commands)
                            candidate[finger] = min(FINGER_SAFE_COMMAND_MAX_DEG,
                                                    finger_start + float(extra))
                            safe_release, release_reason = release_segment_safe(
                                release_q, release_commands, cq, candidate)
                            if not safe_release:
                                continue
                            relation = finger_surface_relation_at_state(
                                solver, arcs, cq, candidate, finger,
                                back_points, back_normals, back_tree, back_mesh_query)
                            if float(relation['signed']) >= release_target_signed:
                                if not release_future_j5_guard(
                                        cq, candidate, finger, 'patch_assist'):
                                    continue
                                accepted_q = cq
                                accepted_commands = candidate
                                accepted_relation = relation
                                print('[V103 PATCH-AWARE RELEASE ASSIST]',
                                      'cycle=', cycle_index, 'finger=', finger,
                                      'j4_trim_deg=', round(float(j4_trim_deg),2),
                                      'j5_lift_deg=', round(float(j5_lift_deg),2),
                                      'reason=pure_curl_patch_or_mesh_blocked')
                                break
            if accepted_commands is None:
                raise RuntimeError('FINGER_CURL_RELEASE_NO_SAFE_CLEARANCE|finger=' +
                                   str(finger) + '|last=' + str(release_reason))
            delta = float(accepted_commands[finger]) - finger_start
            endpoint_clearance = active_finger_mesh_clearance(
                solver, arcs, accepted_q, accepted_commands, LONG_FINGERS,
                back_points, back_normals, back_tree, back_mesh_query)
            endpoint_active = [
                f for f,row in endpoint_clearance['details'].items()
                if float(row['signed']) <= 0.0
            ]
            release_arm_delta_deg = float(np.max(np.abs(np.degrees(
                np.asarray(accepted_q, dtype=np.float64) - release_q))))
            add_keyframe(
                trajectory,
                'FINGER_CURL_RELEASE_' + finger.upper() + '_CYCLE_' + str(int(cycle_index)),
                max(.08, delta / SCRATCH_CURL_FINGER_SPEED_DEG_S,
                    release_arm_delta_deg / ARM_MAX_SPEED_DEG_S),
                accepted_q, accepted_commands, endpoint_active,
                authorized_contact_fingers=set(LONG_FINGERS))
            release_q = np.asarray(accepted_q, dtype=np.float64).copy()
            release_commands = dict(accepted_commands)
            per_finger_extra[finger] = delta
            print('[V103 FINGER RELEASE STEP]', 'cycle=', cycle_index,
                  'finger=', finger, 'extra_pct=', round(delta,2),
                  'signed_mm=', round(float(accepted_relation['signed'])*1000.,3),
                  'held_commands=', {f:round(float(release_commands[f]),2)
                                    for f in LONG_FINGERS})
        release_clearance = active_finger_mesh_clearance(
            solver, arcs, release_q, release_commands, LONG_FINGERS,
            back_points, back_normals, back_tree, back_mesh_query)
        blocked = [f for f in LONG_FINGERS
                   if not math.isfinite(float(release_clearance['details'][f]['signed']))
                   or float(release_clearance['details'][f]['signed']) < release_target_signed]
        if blocked:
            raise RuntimeError('FINGER_CURL_RELEASE_FINAL_CLEARANCE_LOST|' + str(blocked))
        if trajectory[-1]['stage'].startswith('FINGER_CURL_RELEASE_'):
            trajectory[-1]['release_after_scratch'] = True
        # Mark completion only after every required fingertip has reached the
        # explicit +20 mm measured release target.  Keep the marker on the
        # last wave frame so the existing replay validator sees the wave's
        # quality evidence; it no longer appears at the 1--4 mm contact gate.
        completion_frame = next(
            (frame for frame in reversed(trajectory[cycle_trajectory_start:])
             if str(frame.get('stage', '')).startswith('SCRATCH_WAVE_ASYNC_')),
            None,
        )
        if completion_frame is not None:
            completion_frame['scratch_complete'] = True
            completion_frame['scratch_completion_target_mm'] = round(
                release_target_signed * 1000.0, 3)
        print('[V103 FINGER CURL RELEASE]', 'cycle=', cycle_index,
              'per_finger_extra_pct=', {f:round(v,2) for f,v in per_finger_extra.items()},
              'signed_mm=', {f:round(float(row['signed'])*1000.,3)
                            for f,row in release_clearance['details'].items()},
              'optional_airborne_frozen=', [f for f in LONG_FINGERS
                                            if f not in release_required],
              'base_contact_stop=runtime_PhysX_event_only',
              'next=ARM_LIFT')
        result['q'] = release_q.copy()
        result['commands'] = dict(release_commands)
    if result.get('early_exit'):
        trajectory[-1]['early_exit'] = dict(result['early_exit'])
        print('[V103 EARLY EXIT PLANNED]', result['early_exit'], 'progress_pct=',result['commands'])
    print('[V103 CONTACT QUALITY]', 'cycle=', cycle_index, 'deep_first=', result['order'],
          'required_fingers=',working,'optional_fingers=',optional,
          'planning_seconds=', round(result.get('seconds',0.),3), result['quality'])
    # Release frames carry the recomputed endpoint state, not the earlier
    # wave's latched contacts. Keep any remaining contact until verified lift.
    remaining_active = list(trajectory[-1].get('active', []))
    return result['q'], result['commands'], remaining_active


def _append_asynchronous_four_finger_wave(
    trajectory,
    solver,
    arcs,
    mode,
    q_now,
    fingers,
    working_fingers,
    shallow_to_deep,
    endpoint_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    lane_spec,
    cycle_index,
    stagger_scale=1.0,
):
    """Overlap four curls while progressively transferring wrist pressure.

    Finger phase creates the primary shallow-to-deep wave. Pressure assistance
    is hierarchical: J5 first, small J4 trims second, then the existing dynamic
    five-axis solver. Every accepted transition still passes the complete swept
    mesh test and the later full trajectory validator.
    """
    q_trial = np.asarray(q_now, dtype=np.float64).copy()
    pressure_anchor_q = np.asarray(q_now, dtype=np.float64).copy()
    commands_trial = dict(fingers)
    wave_start_commands = dict(fingers)
    working = [finger for finger in LONG_FINGERS if finger in set(working_fingers)]
    order = [finger for finger in shallow_to_deep if finger in set(working)]
    for finger in working:
        if finger not in order:
            order.append(finger)
    if len(order) != 4:
        raise RuntimeError("ASYNC_WAVE_REQUIRES_FOUR_FINGERS")
    active_trial = list(working)
    active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)

    # Acquisition may end several millimetres deeper than the 2 mm comfort
    # target.  Previously runtime tracking stopped at the bottom and lifted J5
    # in three separate 0.5-degree waits.  Convert the same correction into the
    # moving trajectory: establish a lightly relieved pressure anchor and blend
    # toward it while the first curl step is already running.
    start_signed = {}
    for finger in working:
        relation = finger_surface_relation_at_state(
            solver,
            arcs,
            q_trial,
            commands_trial,
            finger,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )
        start_signed[finger] = float(relation["signed"])
    start_mean_depth_mm = max(
        0.0,
        -float(np.mean(list(start_signed.values()))) * 1000.0,
    )
    initial_relief_deg = min(
        ASYNC_INITIAL_RELIEF_MAX_DEG,
        max(0.0, start_mean_depth_mm - 2.0)
        * ASYNC_INITIAL_RELIEF_GAIN_DEG_PER_MM,
    )
    pressure_anchor_q[4] = np.clip(
        pressure_anchor_q[4] + math.radians(initial_relief_deg),
        lower[4],
        upper[4],
    )

    order, delay_deg, initial_schedule = _dynamic_wave_schedule(
        solver, arcs, q_trial, commands_trial, working,
        back_points, back_normals, back_tree, back_mesh_query,
        tie_order=order,
        max_delay_deg=ASYNC_WAVE_DYNAMIC_MAX_DELAY_DEG * float(stagger_scale),
    )
    max_dynamic_delay_deg = ASYNC_WAVE_DYNAMIC_MAX_DELAY_DEG * float(stagger_scale)
    total_progress_deg = max(
        SCRATCH_FULL_CURL_DEG
        - float(commands_trial.get(finger, 0.0))
        + max_dynamic_delay_deg
        for finger in working
    )
    step_count = max(
        8,
        int(math.ceil(total_progress_deg / ASYNC_WAVE_PHASE_STEP_DEG)),
    )

    print(
        "    [V102 ASYNC COMFORT WAVE READY]",
        "cycle=", int(cycle_index),
        "shallow_to_deep=", order,
        "measured_error_mm=", {
            f: round(float(initial_schedule[f]["metric_m"]) * 1000.0, 2)
            for f in order
        },
        "dynamic_delay_deg=", {f: round(delay_deg[f], 1) for f in order},
        "fixed_phase_offset=DISABLED",
        "start_depth_mm=", {
            finger: round(max(0.0, -start_signed[finger]) * 1000.0, 2)
            for finger in order
        },
        "moving_initial_j5_relief_deg=", round(initial_relief_deg, 2),
        "pressure_priority=J5_THEN_J4_THEN_J1_J2_J3",
    )
    contact_stats = acquisition_scratch_contact_stats(
        trajectory, working, solver, arcs, back_points, back_normals, back_tree, back_mesh_query)

    for step_index in range(1, step_count + 1):
        progress_deg = total_progress_deg * float(step_index) / float(step_count)
        order, delay_deg, dynamic_schedule = _dynamic_wave_schedule(
            solver, arcs, q_trial, commands_trial, working,
            back_points, back_normals, back_tree, back_mesh_query,
            tie_order=order,
            max_delay_deg=max_dynamic_delay_deg,
        )
        target_commands = dict(commands_trial)
        for finger in working:
            start_deg = float(wave_start_commands.get(finger, 0.0))
            finger_progress = max(0.0, progress_deg - float(delay_deg[finger]))
            scheduled_value = min(
                SCRATCH_FULL_CURL_DEG,
                start_deg + finger_progress,
            )
            target_commands[finger] = max(
                float(commands_trial.get(finger, start_deg)),
                scheduled_value,
            )

        leading_advance_deg = max(
            0.0,
            float(target_commands[order[0]])
            - float(wave_start_commands.get(order[0], 0.0)),
        )
        # Solver goals are not proof of contact. A qualified, geometrically
        # released tip need not be chased back down during free-space closure.
        next_active = [f for f in order if contact_stats[f]["state"] != "RELEASED"
                       or v103_contact_quality_failure(contact_stats[f]) is not None]
        pressure_steps = max(
            0,
            int(
                math.floor(
                    (leading_advance_deg - ASYNC_PRESSURE_START_ADVANCE_DEG)
                    / ASYNC_J5_PRESS_INTERVAL_DEG
                )
            ) + 1,
        )
        requested_pressure_deg = min(
            ASYNC_J5_PRESS_MAX_DEG,
            ASYNC_J5_PRESS_STEP_DEG * float(pressure_steps),
        )

        # J5 is the primary pressure/depth actuator.  Keep the zero-trim
        # candidate separate so a numerically better J4 pose cannot win while
        # a J5-only transition is already safe and geometrically valid.
        j5_primary_candidate = None
        j4_assist_candidates = []
        # Once every finger has passed its useful contact stroke, hold the last
        # proven-safe wrist pose and finish the commanded curl. Running the
        # contact solver with an empty active set would incorrectly treat the
        # intentionally released brushing fingers as contacts that the wrist
        # must continue chasing.
        released_curl_only = not next_active
        for j4_trim_deg in (
            () if released_curl_only else ASYNC_J4_TRIM_CANDIDATES_DEG
        ):
            candidate_q = q_trial.copy()
            candidate_q[3] = np.clip(
                q_trial[3] + math.radians(float(j4_trim_deg)),
                max(
                    lower[3],
                    pressure_anchor_q[3]
                    - math.radians(ASYNC_J4_TRIM_MAX_TOTAL_DEG),
                ),
                min(
                    upper[3],
                    pressure_anchor_q[3]
                    + math.radians(ASYNC_J4_TRIM_MAX_TOTAL_DEG),
                ),
            )
            candidate_q[4] = np.clip(
                pressure_anchor_q[4] - math.radians(requested_pressure_deg),
                lower[4],
                upper[4],
            )
            safe, reason = segment_mesh_safe(
                solver,
                arcs,
                q_trial,
                commands_trial,
                candidate_q,
                target_commands,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=30,
                allowed_contact_fingers=set(working),
            )
            if not safe:
                continue
            geometry = dynamic_keyframe_geometry(
                solver,
                arcs,
                mode,
                candidate_q,
                target_commands,
                next_active,
            )
            if not bool(geometry.get("valid", False)):
                continue
            region_score = 0.0
            for finger in working:
                tip = world_tip_position(
                    solver, arcs, candidate_q, target_commands, finger
                )
                lane_state = lane_relation_for_world_point(tip, lane_spec, finger)
                region_score += (
                    float(lane_state["cross_error"])
                    / max(active_cross_limit, 1e-6)
                ) ** 2
                region_score += 0.25 * (
                    float(lane_state["patch_radius"])
                    / max(active_radius_limit, 1e-6)
                ) ** 2
            candidate = (
                float(region_score),
                abs(float(j4_trim_deg)),
                candidate_q,
                j4_trim_deg,
            )
            if abs(float(j4_trim_deg)) <= 1e-9:
                j5_primary_candidate = candidate
            else:
                j4_assist_candidates.append(candidate)

        solver_level = "J5"
        if released_curl_only:
            flex_only_safe, flex_only_reason = segment_mesh_safe(
                solver,
                arcs,
                q_trial,
                commands_trial,
                q_trial,
                target_commands,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=30,
                allowed_contact_fingers=set(working),
            )
            if not flex_only_safe:
                raise RuntimeError(
                    "ASYNC_RELEASED_CURL_PATH_UNSAFE_CYCLE_"
                    + str(int(cycle_index))
                    + "_STEP_" + str(int(step_index))
                    + "_" + str(flex_only_reason)
                )
            q_candidate = q_trial.copy()
            solver_level = "RELEASED_CURL_WRIST_HELD"
        elif j5_primary_candidate is not None:
            # Do not trade a valid J5-only pressure step for a lower region
            # score obtained by rolling the whole hand with J4.
            _, _, q_candidate, _ = j5_primary_candidate
            solver_level = "J5_PRIMARY"
        elif j4_assist_candidates:
            # J4 is an assist only: use it after the J5-only candidate failed
            # a swept-mesh or geometry gate for this same pressure step.
            _, _, q_candidate, used_j4_trim_deg = min(
                j4_assist_candidates, key=lambda item: (item[0], item[1])
            )
            solver_level = "J5_J4_ASSIST"
        else:
            failure = {}
            dynamic_state = solve_dynamic_contact_q(
                solver,
                arcs,
                mode,
                target_commands,
                next_active,
                q_trial,
                lower,
                upper,
                endpoint_q=endpoint_q,
                previous_finger_commands=commands_trial,
                previous_active=[
                    finger for finger in active_trial if finger in set(next_active)
                ],
                moving_finger=(next_active[0] if next_active else None),
                lane_spec=lane_spec,
                failure_out=failure,
            )
            q_candidate = None
            if dynamic_state is not None:
                dynamic_q = np.asarray(dynamic_state["q"], dtype=np.float64)
                dynamic_delta_deg = np.degrees(dynamic_q - pressure_anchor_q)
                dynamic_pressure_deg = math.degrees(
                    float(pressure_anchor_q[4] - dynamic_q[4])
                )
                dynamic_bounded = bool(
                    float(np.max(np.abs(dynamic_delta_deg[:3])))
                    <= ASYNC_J123_SUPPORT_MAX_TOTAL_DEG + 1e-9
                    and abs(float(dynamic_delta_deg[3]))
                    <= ASYNC_J4_TRIM_MAX_TOTAL_DEG + 1e-9
                    and dynamic_pressure_deg
                    >= -ASYNC_J5_DYNAMIC_TOLERANCE_DEG - 1e-9
                    and dynamic_pressure_deg
                    <= requested_pressure_deg
                    + ASYNC_J5_DYNAMIC_TOLERANCE_DEG + 1e-9
                )
                if dynamic_bounded:
                    safe, reason = segment_mesh_safe(
                        solver,
                        arcs,
                        q_trial,
                        commands_trial,
                        dynamic_q,
                        target_commands,
                        back_points,
                        back_normals,
                        back_tree,
                        back_mesh_query,
                        samples=30,
                        allowed_contact_fingers=set(working),
                    )
                    if safe:
                        q_candidate = dynamic_q
                        solver_level = "J1_TO_J5_DYNAMIC_BOUNDED"
                else:
                    failure["absolute_support_reject_deg"] = {
                        "j123_max": round(
                            float(np.max(np.abs(dynamic_delta_deg[:3]))), 3
                        ),
                        "j4": round(float(dynamic_delta_deg[3]), 3),
                        "j5_pressure": round(float(dynamic_pressure_deg), 3),
                    }
            if q_candidate is None:
                flex_only_safe, flex_only_reason = segment_mesh_safe(
                    solver,
                    arcs,
                    q_trial,
                    commands_trial,
                    q_trial,
                    target_commands,
                    back_points,
                    back_normals,
                    back_tree,
                    back_mesh_query,
                    samples=30,
                    allowed_contact_fingers=set(working),
                )
                if not flex_only_safe:
                    raise RuntimeError(
                        "ASYNC_WAVE_PATH_UNSAFE_CYCLE_"
                        + str(int(cycle_index))
                        + "_STEP_" + str(int(step_index))
                        + "_WRIST=" + str(reason)
                        + "_DYNAMIC=" + str(failure)
                        + "_FLEX=" + str(flex_only_reason)
                    )
                q_candidate = q_trial.copy()
                solver_level = "FLEX_ONLY_PRESSURE_HELD"

        max_delta = max(
            abs(float(target_commands[f]) - float(commands_trial.get(f, 0.0)))
            for f in working
        )
        predicted_depth_mm = {}
        predicted_stage = (
            "SCRATCH_WAVE_ASYNC_"
            + str(int(step_index))
            + "_CYCLE_" + str(int(cycle_index))
        )
        for finger in working:
            predicted_tip = world_tip_position(
                solver, arcs, q_candidate, target_commands, finger
            )
            predicted_relation = final_surface_relation(
                predicted_tip,
                back_mesh_query,
                back_points,
                back_normals,
                back_tree,
            )
            predicted_signed = float(predicted_relation["signed"])
            predicted_depth_mm[finger] = round(
                max(0.0, -predicted_signed) * 1000.0, 2
            )
        add_keyframe(
            trajectory,
            predicted_stage,
            max(0.028, max_delta / SCRATCH_CURL_FINGER_SPEED_DEG_S),
            q_candidate,
            target_commands,
            next_active,
            authorized_contact_fingers=set(working),
        )
        sample_scratch_contact_segment(
            contact_stats, solver, arcs, trajectory[-2], trajectory[-1],
            back_points, back_normals, back_tree, back_mesh_query)
        next_active = [f for f in order if contact_stats[f]["state"] == "CONTACT"]
        trajectory[-1]["active"] = list(next_active)
        trajectory[-1]["predicted_contact_states"] = {f: contact_stats[f]["state"] for f in order}
        print(
            "    [V102 ASYNC WAVE STEP]",
            "cycle=", int(cycle_index),
            "step=", int(step_index),
            "dynamic_order=", order,
            "dynamic_delay_deg=", {
                finger: round(float(delay_deg[finger]), 2) for finger in order
            },
            "local_sensitivity_mm_per_deg=", {
                finger: round(float(dynamic_schedule[finger]["sensitivity_m_per_deg"]) * 1000.0, 4)
                for finger in order
            },
            "requested_pressure_deg=", round(requested_pressure_deg, 2),
            "actual_j5_pressure_deg=", round(
                math.degrees(float(pressure_anchor_q[4] - q_candidate[4])), 2
            ),
            "actual_j4_trim_deg=", round(
                math.degrees(float(q_candidate[3] - pressure_anchor_q[3])), 2
            ),
            "j123_support_max_deg=", round(
                float(
                    np.max(
                        np.abs(
                            np.degrees(q_candidate[:3] - pressure_anchor_q[:3])
                        )
                    )
                ),
                2,
            ),
            "solver=", solver_level,
            "active=", list(next_active),
            "commands=", {f: round(float(target_commands[f]), 1) for f in order},
            "predicted_depth_mm=", predicted_depth_mm,
        )
        q_trial = np.asarray(q_candidate, dtype=np.float64).copy()
        commands_trial = dict(target_commands)
        active_trial = list(next_active)

    if any(
        abs(float(commands_trial[f]) - SCRATCH_FULL_CURL_DEG) > 1e-6
        for f in working
    ):
        raise RuntimeError("ASYNC_WAVE_DID_NOT_REACH_FULL_CURL")
    planned_max_depth_mm = {
        finger: round(max(0.0, -contact_stats[finger]["min_signed"]) * 1000.0, 2)
        for finger in order
    }
    planned_depth_values = list(planned_max_depth_mm.values())
    quality_failures = {f: v103_contact_quality_failure(contact_stats[f]) for f in order
                        if v103_contact_quality_failure(contact_stats[f]) is not None}
    motion_diagnostics = {f: scratch_contact_motion_diagnostics(contact_stats[f]) for f in order
                          if scratch_contact_motion_diagnostics(contact_stats[f])}
    print(
        "    [V102 PLANNED DYNAMIC DEPTH SUMMARY]",
        "cycle=", int(cycle_index),
        "shallow_to_deep=", order,
        "max_depth_mm=", planned_max_depth_mm,
        "spread_mm=", round(
            max(planned_depth_values) - min(planned_depth_values), 2
        ),
        "deepest_stage=", {
            finger: contact_stats[finger]["deepest_stage"] for finger in order
        },
        "contact_slide_mm=", {f: round(contact_stats[f]["slide_m"] * 1000.0, 2) for f in order},
        "moving_contact_ms=", {f: round(contact_stats[f]["moving_contact_s"] * 1000.0, 2) for f in order},
        "contact_tangent_ratio=", {f: round(contact_stats[f]["slide_m"] /
                                           max(contact_stats[f]["path_m"], 1e-12), 3) for f in order},
        "final_contact_state=", {f: contact_stats[f]["state"] for f in order},
        "scope=acquisition_plus_wave_continuous_samples",
        "quality_failures=", quality_failures,
        "motion_diagnostics_only=", motion_diagnostics,
    )
    if quality_failures:
        raise RuntimeError(
            "ASYNC_CONTACT_QUALITY_REJECT_CYCLE_"
            + str(int(cycle_index))
            + "_failures=" + str(quality_failures)
        )
    return q_trial, commands_trial, active_trial


def append_full_four_finger_curl(
    trajectory,
    solver,
    arcs,
    q_now,
    fingers,
    working_fingers,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    cycle_index,
):
    """Finish one scratch stroke up to the configured safe coupled curl limit.

    The first short continuation completes the surface stroke. Contact remains
    active/authorized throughout the curl because a distal link can continue
    brushing the compliant surface after the fingertip starts moving away.
    Contact is released only after the later wrist unload is mesh-confirmed.
    """
    q_now = np.asarray(q_now, dtype=np.float64).copy()
    commands_now = dict(fingers)
    working = [finger for finger in LONG_FINGERS if finger in set(working_fingers)]
    if len(working) != 4:
        raise RuntimeError("FULL_CURL_REQUIRES_FOUR_FINGERS")

    release_commands = dict(commands_now)
    for finger in working:
        release_commands[finger] = min(
            SCRATCH_FULL_CURL_DEG,
            float(commands_now.get(finger, 0.0))
            + SCRATCH_CONTACT_RELEASE_ADVANCE_DEG,
        )

    release_target_signed = float(getattr(
        CONTACT_WAVE, 'HANDOFF_RELEASE_TARGET_SIGNED_M', .020))

    def measured_signed(commands):
        clearance = active_finger_mesh_clearance(
            solver, arcs, q_now, commands, LONG_FINGERS,
            back_points, back_normals, back_tree, back_mesh_query)
        return clearance, {
            f: float(clearance['details'][f]['signed'])
            for f in LONG_FINGERS
        }

    def target_reached(signed):
        return all(
            value == float('inf') or (
                math.isfinite(value)
                and value >= release_target_signed - 1e-12)
            for value in signed.values()
        )

    # Do not jump straight from contact to 70 degrees.  The old helper only
    # measured clearance after that fixed 70-degree stage, so a fingertip that
    # crossed +20 mm at 25--40 degrees was still curled far past the requested
    # handoff height.  Probe in small monotone increments and stop at the first
    # measured state that reaches the target; the 70-degree stage and the hard
    # curl limit are only later fallbacks for geometries that genuinely need
    # them.
    targets = [(
        "SCRATCH_CURL_CONTACT_CYCLE_" + str(int(cycle_index)),
        release_commands,
    )]
    last_level = max(float(release_commands[f]) for f in working)
    for curl_deg in np.arange(
            max(5.0, math.ceil(last_level / 5.0) * 5.0),
            float(SCRATCH_FULL_CURL_DEG) + 0.001,
            5.0):
        target_commands = dict(release_commands)
        for finger in working:
            target_commands[finger] = max(
                float(target_commands.get(finger, 0.0)),
                min(float(curl_deg), float(SCRATCH_FULL_CURL_DEG)),
            )
        if any(
            abs(float(target_commands[f]) - float(targets[-1][1].get(f, 0.0))) > 1e-6
            for f in working
        ):
            targets.append((
                "SCRATCH_CURL_"
                + str(int(round(float(curl_deg))))
                + "_CYCLE_"
                + str(int(cycle_index)),
                target_commands,
            ))
    # If the configured full-curl limit is not an exact 5-degree step, retain
    # it as the final fallback without skipping the measured intermediate
    # states above.
    full_commands = dict(targets[-1][1])
    for finger in working:
        full_commands[finger] = float(SCRATCH_FULL_CURL_DEG)
    if any(
            abs(float(full_commands[f]) - float(targets[-1][1].get(f, 0.0))) > 1e-6
            for f in working):
        targets.append((
            "SCRATCH_CURL_" + str(int(round(float(SCRATCH_FULL_CURL_DEG))))
            + "_CYCLE_" + str(int(cycle_index)),
            full_commands,
        ))
    target_reached_at = None
    for stage_name, target_commands in targets:
        safe, reason = segment_mesh_safe(
            solver,
            arcs,
            q_now,
            commands_now,
            q_now,
            target_commands,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=36,
            allowed_contact_fingers=set(working),
        )
        if not safe:
            raise RuntimeError(stage_name + "_PATH_UNSAFE_" + str(reason))
        _, signed = measured_signed(target_commands)
        max_delta = max(
            abs(float(target_commands[f]) - float(commands_now.get(f, 0.0)))
            for f in working
        )
        add_keyframe(
            trajectory,
            stage_name,
            max(0.28, max_delta / SCRATCH_CURL_FINGER_SPEED_DEG_S),
            q_now,
            target_commands,
            working,
            authorized_contact_fingers=set(working),
        )
        # This stage is a continuation of the already-planned contact wave.
        # Do not make the lifecycle validator infer a new synchronized contact
        # commit from the command jump to the full curl.  In some candidate
        # paths the preceding sampled frame carries no ``active`` list even
        # though the wave has already established contact, so the validator
        # otherwise reports all four fingers as an illegal fresh commit.
        trajectory[-1]["contact_continuation"] = True
        trajectory[-1]["predicted_contact_states"] = {
            finger: "CONTACT" for finger in working
        }
        commands_now = dict(target_commands)
        if target_reached(signed):
            target_reached_at = stage_name
            trajectory[-1]['curl_target_reached'] = True
            trajectory[-1]['curl_target_signed_mm'] = {
                f: round(value * 1000.0, 3) for f, value in signed.items()
            }
            print('[V103 CURL TARGET REACHED]',
                  'cycle=', int(cycle_index),
                  'stage=', stage_name,
                  'signed_mm=', {f: round(value * 1000.0, 3)
                                 for f, value in signed.items()})
            break

    print(
        "    [V100 FULL CURL READY]",
        "cycle=", int(cycle_index),
        "commands=", {f: round(float(commands_now[f]), 1) for f in working},
        "target_stage=", target_reached_at or "FULL_CURL_FALLBACK",
    )
    return q_now, commands_now, working


def append_v103_local_cycle_reset(trajectory, solver, arcs, q_now, fingers,
                                 open_q, open_fingers, lower, upper,
                                 back_points, back_normals, back_tree, back_mesh_query, cycle_index):
    """Reset between strokes with an overlapped lift-and-open when safe.

    The first scratch must finish its release before the next one starts, but
    there is no reason to serialize the wrist lift and finger extension.  The
    preferred transition sweeps both together to a lifted open pose, then the
    next cycle rapidly re-approaches ``open_q``.  The old three-segment route
    remains a checked fallback for awkward local geometry.
    """
    opened = empty_fingers()
    # The inter-cycle reset opens the fingers immediately after the first
    # stroke.  A 0--2 degree J5 lift can look clear in the static mesh test but
    # still let the index distal link sweep back into the surface during the
    # real opening motion.  Keep J4 fixed and start the preferred J5 unload at
    # 6 degrees; larger candidates remain available for steep local geometry.
    for j4, j5 in [(0., x) for x in (6., 8., 12., 16.)] + [
            (a,b) for b in (6.,8.,12.) for a in (-1.,1.,-2.,2.)]:
        q = np.asarray(q_now).copy()
        q[3:] += np.radians([j4,j5])
        if np.any(q < lower) or np.any(q > upper):
            continue
        # Preferred path: lift the wrist and open the fingers in one
        # synchronized, no-contact sweep.  Duration is set by the slower of
        # the arm and finger motion, so the transition is as short as the
        # configured physical rates permit rather than an arbitrary dwell.
        overlap_safe, _ = segment_mesh_safe(
            solver, arcs, q_now, fingers, q, opened,
            back_points, back_normals, back_tree, back_mesh_query,
            samples=96, allowed_contact_fingers=set(),
        )
        return_safe, _ = segment_mesh_safe(
            solver, arcs, q, opened, open_q, open_fingers,
            back_points, back_normals, back_tree, back_mesh_query,
            samples=96, allowed_contact_fingers=set(),
        )
        if overlap_safe and return_safe:
            arm_time = arm_move_duration(q_now, q, .08)
            finger_delta = max(
                (abs(float(opened[f]) - float(fingers.get(f, 0.0)))
                 for f in LONG_FINGERS),
                default=0.0,
            )
            finger_time = finger_delta / max(SCRATCH_CURL_FINGER_SPEED_DEG_S, 1e-6)
            add_keyframe(
                trajectory,
                'UNLOAD_OPEN_OVERLAPPED_CYCLE_' + str(cycle_index),
                max(.12, arm_time, finger_time),
                q,
                opened,
                [],
                authorized_contact_fingers=[],
            )
            trajectory[-1]['overlapped_wrist_lift_and_open'] = True
            trajectory[-1]['intercycle_safe_pose'] = 'LOCAL_LIFTED_OPEN'
            print('[V103 INTER-CYCLE TRANSITION]',
                  'overlap_wrist_lift_and_finger_open=True',
                  'cycle=', cycle_index,
                  'duration_s=', round(max(.12, arm_time, finger_time), 3),
                  'next=FAST_OPEN_REAPPROACH')
            trajectory[-1]['local_cycle_reset'] = True
            return q, opened, []

        # Compatibility fallback: keep the former serialized route, but
        # validate every segment independently before exposing any frame.
        segments = [(q_now,fingers,q,fingers),(q,fingers,q,opened),
                    (q,opened,open_q,open_fingers)]
        if not all(segment_mesh_safe(solver,arcs,a,fa,b,fb,
                back_points,back_normals,back_tree,back_mesh_query,
                samples=96,allowed_contact_fingers=set())[0]
                for a,fa,b,fb in segments):
            continue
        add_keyframe(trajectory,'UNLOAD_STEP_LOCAL_RESET_CYCLE_'+str(cycle_index),
                     arm_move_duration(q_now,q,.08),q,fingers,[],authorized_contact_fingers=[])
        add_keyframe(trajectory,'OPEN_AFTER_UNLOAD_CYCLE_'+str(cycle_index),
                     OPEN_AFTER_UNLOAD_TIME_S,q,opened,[],authorized_contact_fingers=[])
        trajectory[-1]['local_cycle_reset'] = True
        print('[V103 LOCAL CYCLE RESET]', 'j4_delta_deg=',j4,'j5_delta_deg=',j5,
              'opening_and_reapproach=PASS; no-contact whole-hand swept checks')
        return q,opened,[]
    print('[V103 LOCAL CYCLE RESET] no safe local branch; using full unload')
    return None


def append_v103_early_elbow_exit(trajectory, solver, arcs, q_now, fingers, home_q,
                               back_points, back_normals, back_tree, back_mesh_query, cycle_index):
    """Hold the released partial curl; move J1-J3 away before opening."""
    opened = empty_fingers()
    for fraction in (.08,.15,.25,.40,.60,1.):
        q = np.asarray(q_now) + fraction*(np.asarray(home_q)-np.asarray(q_now))
        if np.max(np.abs(q[:3]-np.asarray(q_now)[:3])) < math.radians(.5):
            continue
        clear = active_finger_mesh_clearance(solver,arcs,q,fingers,LONG_FINGERS,
                         back_points,back_normals,back_tree,back_mesh_query)
        if not clear['clear'] or float(clear['minimum_signed']) < V103_SAFETY_CLEARANCE_M:
            continue
        # The base is not a planning rejection in the production V103 policy.
        # It is sampled only by the post-scratch curl stop below; keeping this
        # branch free of a base margin lets a valid surface trajectory reach
        # the actual release decision instead of failing during route search.
        if (not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))
                and HAND_COLLISION_MODEL.base_guard.clearance(q,fingers)[0] < V103_BASE_MIN_CLEARANCE_M):
            continue
        safe, _ = segment_mesh_safe(solver,arcs,q_now,fingers,q,fingers,
                    back_points,back_normals,back_tree,back_mesh_query,samples=96,allowed_contact_fingers=set())
        if not safe:
            continue
        safe, _ = segment_mesh_safe(solver,arcs,q,fingers,q,opened,
                    back_points,back_normals,back_tree,back_mesh_query,samples=64,allowed_contact_fingers=set())
        if not safe:
            continue
        add_keyframe(trajectory,'UNLOAD_STEP_EARLY_ELBOW_CYCLE_'+str(cycle_index),
                     arm_move_duration(q_now,q,.25),q,fingers,[],authorized_contact_fingers=[])
        add_keyframe(trajectory,'OPEN_AFTER_UNLOAD_CYCLE_'+str(cycle_index),
                     OPEN_AFTER_UNLOAD_TIME_S,q,opened,[],authorized_contact_fingers=[])
        print('[V103 EARLY ELBOW EXIT] partial_curl_held=True; J1-J3 active;',
              'q_deg=',np.round(np.degrees(q),2),'home_fraction=',fraction)
        return q,opened,[]
    raise RuntimeError('EARLY_ELBOW_EXIT_NO_SAFE_ROUTE')


def append_adaptive_unload_and_open(
    trajectory,
    solver,
    arcs,
    mode,
    q_now,
    fingers,
    clearance_fingers,
    home_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    cycle_index,
):
    """Release with positive J5 first; use whole-arm lift only as fallback."""
    q_now = np.asarray(q_now, dtype=np.float64).copy()
    fingers = dict(fingers)
    clearance_fingers = list(LONG_FINGERS)  # Empty active set is not clearance.
    palm_p, palm_r = solver.compute_forward_kinematics(PALM_FRAME, q_now)
    palm_p = np.asarray(palm_p, dtype=np.float64)
    palm_r = np.asarray(palm_r, dtype=np.float64)
    unload_clear = False
    open_fingers = empty_fingers()
    last_open_reason = "NOT_TESTED"

    # Preferred release requested for the real robot convention: keep J1-J4
    # fixed and increase J5 in 0.5-degree increments. Select the first wrist
    # angle whose complete curled-to-open swept geometry is collision-free.
    unload_contact_fingers = set(LONG_FINGERS)
    for j5_lift_deg in UNLOAD_J5_LIFT_CANDIDATES_DEG:
        requested_lift_deg = float(j5_lift_deg) + float(UNLOAD_J5_SAFETY_MARGIN_DEG)
        j5_unload_q = q_now.copy()
        j5_unload_q[4] = np.clip(
            q_now[4] + math.radians(requested_lift_deg),
            lower[4],
            upper[4],
        )
        actual_lift_deg = math.degrees(float(j5_unload_q[4] - q_now[4]))
        if actual_lift_deg < requested_lift_deg - 0.25:
            break

        # Cheap fingertip-clearance gate avoids running the expensive complete
        # hand sweep for every 0.5-degree trial during planning.
        j5_clearance = active_finger_mesh_clearance(
            solver,
            arcs,
            j5_unload_q,
            fingers,
            LONG_FINGERS,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )
        projected_min_signed = float(
            j5_clearance.get('minimum_signed', float('nan')))
        if (not math.isfinite(projected_min_signed)
                or projected_min_signed < DEEP_ADJUST_EXIT_SIGNED_M - 1e-12):
            # A planned J5 unload must not move any core fingertip back into
            # the deep-adjust band.  The old endpoint-only check could pass
            # here, then let the runtime tracking overshoot ring/index on the
            # very next arm-lift frame.
            last_open_reason = 'J5_LIFT_PROJECTED_DEEP'
            print(
                '    [V103 J5-ONLY UNLOAD DEPTH GUARD]',
                'cycle=', int(cycle_index),
                'base_safe_candidate_deg=', round(float(j5_lift_deg), 1),
                'safety_margin_deg=', round(float(UNLOAD_J5_SAFETY_MARGIN_DEG), 1),
                'j5_lift_deg=', round(float(actual_lift_deg), 1),
                'projected_min_signed_mm=',
                round(projected_min_signed * 1000.0, 3),
                'threshold_mm=', round(DEEP_ADJUST_EXIT_SIGNED_M * 1000.0, 3),
                'accepted=False',
            )
            continue
        if not bool(j5_clearance["clear"]):
            last_open_reason = "J5_TIP_CLEARANCE_PENDING"
            continue

        lift_safe, lift_reason = segment_mesh_safe(
            solver,
            arcs,
            q_now,
            fingers,
            j5_unload_q,
            fingers,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=32,
            allowed_contact_fingers=unload_contact_fingers,
        )
        if not lift_safe:
            last_open_reason = "J5_LIFT_" + str(lift_reason)
            continue

        open_safe, open_reason = segment_mesh_safe(
            solver,
            arcs,
            j5_unload_q,
            fingers,
            j5_unload_q,
            open_fingers,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=40,
            allowed_contact_fingers=set(),
        )
        last_open_reason = str(open_reason)
        if not open_safe:
            continue

        add_keyframe(
            trajectory,
            "UNLOAD_J5_LIFT_"
            + str(round(float(actual_lift_deg), 1)).replace(".", "P")
            + "DEG_CYCLE_" + str(int(cycle_index)),
            max(0.12, float(actual_lift_deg) / UNLOAD_J5_SPEED_DEG_S),
            j5_unload_q,
            fingers,
            [],
            authorized_contact_fingers=unload_contact_fingers,
        )
        q_now = np.asarray(j5_unload_q, dtype=np.float64)
        unload_clear = True
        print(
            "    [V102 J5-ONLY UNLOAD CLEAR]",
            "cycle=", int(cycle_index),
            "base_safe_candidate_deg=", round(float(j5_lift_deg), 1),
            "safety_margin_deg=", round(float(UNLOAD_J5_SAFETY_MARGIN_DEG), 1),
            "j5_lift_deg=", round(float(actual_lift_deg), 1),
            "j1_to_j4_fixed=", True,
        )
        break

    if not unload_clear:
        print(
            "    [V102 J5-ONLY UNLOAD FALLBACK]",
            "cycle=", int(cycle_index),
            "reason=", last_open_reason,
            "fallback=WHOLE_ARM_NORMAL_OFFSET",
        )

    for unload_offset in (UNLOAD_OFFSETS_M if not unload_clear else ()):
        unload_q = solve_offset_pose(
            solver,
            palm_p,
            palm_r,
            mode["normal"],
            float(unload_offset),
            q_now,
            home_q,
            lower,
            upper,
        )
        if unload_q is None:
            continue
        stage_name = (
            "UNLOAD_STEP_"
            + str(int(round(float(unload_offset) * 1000.0)))
            + "MM_CYCLE_"
            + str(int(cycle_index))
        )
        add_keyframe(
            trajectory,
            stage_name,
            max(0.24, arm_move_duration(q_now, unload_q, 0.24)),
            unload_q,
            fingers,
            clearance_fingers,
            authorized_contact_fingers=set(clearance_fingers),
        )
        q_now = np.asarray(unload_q, dtype=np.float64)
        clearance = active_finger_mesh_clearance(
            solver,
            arcs,
            q_now,
            fingers,
            clearance_fingers,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )
        print(
            "    [V100 CYCLE UNLOAD CHECK]",
            "cycle=", int(cycle_index),
            "offset_mm=", round(float(unload_offset) * 1000.0, 1),
            "min_dist_mm=", round(float(clearance["minimum_distance"]) * 1000.0, 2),
            "min_signed_mm=", round(float(clearance["minimum_signed"]) * 1000.0, 2),
        )
        if clearance["clear"]:
            # Fingertip clearance alone is insufficient: when a 90-degree curl
            # opens, the distal link can swing toward the back before moving
            # away. Continue unloading until the complete curled-to-open swept
            # geometry is also collision-free.
            open_safe, open_reason = segment_mesh_safe(
                solver,
                arcs,
                q_now,
                fingers,
                q_now,
                open_fingers,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=40,
                allowed_contact_fingers=set(),
            )
            last_open_reason = str(open_reason)
            if open_safe:
                unload_clear = True
                print(
                    "    [V102 WHOLE-HAND OPEN SWEEP CLEAR]",
                    "cycle=", int(cycle_index),
                    "offset_mm=", round(float(unload_offset) * 1000.0, 1),
                )
                break
            print(
                "    [V102 UNLOAD EXTEND FOR OPEN SWEEP]",
                "cycle=", int(cycle_index),
                "offset_mm=", round(float(unload_offset) * 1000.0, 1),
                "reason=", last_open_reason,
            )

    if not unload_clear:
        raise RuntimeError(
            "UNLOAD_OR_OPEN_SWEEP_NOT_CONFIRMED_CYCLE_"
            + str(int(cycle_index))
            + "_" + last_open_reason
        )

    add_keyframe(
        trajectory,
        "UNLOAD_ALL_CYCLE_" + str(int(cycle_index)),
        0.12,
        q_now,
        fingers,
        [],
        authorized_contact_fingers=[],
    )
    # Recheck the exact final sweep defensively. The loop above already proved
    # this same transition safe at the selected unload offset.
    open_safe, open_reason = segment_mesh_safe(
        solver,
        arcs,
        q_now,
        fingers,
        q_now,
        open_fingers,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
        samples=40,
        allowed_contact_fingers=set(),
    )
    if not open_safe:
        raise RuntimeError(
            "OPEN_AFTER_UNLOAD_PATH_UNSAFE_CYCLE_"
            + str(int(cycle_index))
            + "_"
            + str(open_reason)
        )
    add_keyframe(
        trajectory,
        "OPEN_AFTER_UNLOAD_CYCLE_" + str(int(cycle_index)),
        OPEN_AFTER_UNLOAD_TIME_S,
        q_now,
        open_fingers,
        [],
    )
    return q_now, open_fingers, []


def v103_plan_retreat_route(q_start, approach_qs, home_q, lower, upper,
                            check_segment, lift_step, *, prefer_direct_home=False):
    """Return a completely checked free-space route, without mutating a plan.

    Reuse ingress geometry, NOT its contact authorizations or finger commands.
    check_segment must use the released/open hand and the inactive depth limit.
    Every bridge, reversed edge and final Home edge is checked again.
    """
    start = np.asarray(q_start, dtype=np.float64).copy()
    home = np.asarray(home_q, dtype=np.float64).copy()
    ingress = [np.asarray(q, dtype=np.float64).copy() for q in approach_qs]
    failures = []

    def checked(route, label, origin=None):
        previous = start if origin is None else origin
        result = []
        for q in route:
            q = np.asarray(q, dtype=np.float64)
            if (q.shape != start.shape or not np.all(np.isfinite(q))
                    or np.any(q < lower) or np.any(q > upper)):
                failures.append(label + ":JOINT_LIMIT_OR_NONFINITE")
                return None
            ok, reason = check_segment(previous, q)
            if not ok:
                failures.append(label + ":" + str(reason))
                return None
            if np.max(np.abs(q - previous)) > 1e-9:
                result.append(q.copy())
            previous = q
        return result

    # A single scratch cycle should leave the open hand directly at Home when
    # the complete open-hand sweep is safe.  The old implementation always
    # reversed the ingress path first; that made J5 visibly lower again after
    # the fingers opened, even though no second scratch cycle was requested.
    # Keep the reverse-ingress route as a checked fallback for targets where a
    # direct open-hand return would cross the table/back geometry.
    routes = []
    if prefer_direct_home:
        routes.append(("DIRECT_HOME", [home]))

    # Start with the original open handoff. If its bridge is blocked, try
    # joining earlier on the last ingress edge, then earlier saved waypoints.
    if ingress:
        routes.append(("REVERSE_APPROACH", list(reversed(ingress)) + [home]))
        if len(ingress) >= 2:
            a, b = ingress[-2:]
            for fraction in (0.75, 0.50, 0.25):
                join = a + fraction * (b - a)
                routes.append(("REVERSE_APPROACH_JOIN_" + str(fraction),
                               [join] + list(reversed(ingress[:-1])) + [home]))
        routes.append(("REVERSE_APPROACH_START", [ingress[0], home]))
    for label, route in routes:
        result = checked(route, label)
        if result is not None:
            return result, label

    # Fallback: ten local 5 mm normal lifts, each swept-checked immediately.
    # A safe elevated endpoint alone must never authorize a crossing segment.
    route, previous = [], start
    for _ in range(10):
        q = lift_step(previous.copy())
        if q is None:
            failures.append("LOCAL_NORMAL_LIFT:IK_FAIL")
            break
        q = np.asarray(q, dtype=np.float64)
        result = checked([q], "LOCAL_NORMAL_LIFT", origin=previous)
        if result is None:
            break
        route.append(q.copy())
        previous = q
    else:
        result = checked([home], "LOCAL_NORMAL_LIFT_HOME", origin=previous)
        if result is not None:
            return route + result, "LOCAL_NORMAL_LIFT"
    raise RuntimeError("NO_SAFE_RETREAT_ROUTE|" + "|".join(failures))


def build_trajectory(
    solver,
    arcs,
    best,
    current_q,
    home_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    inactive_clearance_deg=0.0,
    current_fingers=None,
    historical_replay=False,
):
    """
    V103 continuous four-finger trajectory with frozen completion ownership.

    Flow:
        HOME/current
        -> palm-oriented approach 60/45/30/25/20/16/12 mm
        -> acquire each finger at an ADAPTIVE measured-arc phase
        -> establish shallow-to-deep serial four-finger contact
        -> transfer pressure through J5, then J4, then small J1-J3 support
        -> continue the measured focus handoff to the safe coupled curl limit
        -> lift the wrist until mesh clearance is confirmed
        -> open the fingers in free space
        -> repeat the approach/scratch cycle
        -> direct open-hand RETURN_HOME (preferred; fully swept-checked)
        -> SAFE_RETREAT -> RETURN_HOME (only if direct return is blocked)

    A new finger is never forced to be in contact at one arbitrary 52-degree
    angle. Contact is established first; tangential scratch requirements start
    only after contact exists.
    """
    mode = best[
        "mode"
    ]

    template = best[
        "template"
    ]

    sequence = list(
        template[
            "sequence"
        ]
    )

    phases = dict(
        template[
            "phases"
        ]
    )

    trajectory = []

    q_now = np.asarray(
        current_q,
        dtype=np.float64,
    ).copy()

    # Execution must start from the ACTUAL Play-time finger state. Previously
    # this was unconditionally replaced with synthetic 0 deg commands, which
    # could make the pinky jump as soon as execution began.
    fingers = empty_fingers()
    if current_fingers is not None:
        for finger in LONG_FINGERS:
            fingers[finger] = float(current_fingers.get(finger, fingers[finger]))
    active = []

    add_keyframe(
        trajectory,
        "CURRENT_HOLD",
        0.0,
        q_now,
        fingers,
        active,
    )

    # --------------------------------------------------------
    # V4.6: fold NON-WORKING fingers at Home/current pose BEFORE
    # the arm moves toward the back.
    # --------------------------------------------------------

    working_fingers = set(
        template[
            "active"
        ]
    )

    clearance_commands = dict(fingers)

    for finger in LONG_FINGERS:
        if finger in working_fingers:
            # Preserve the measured straight/open Play-time angle exactly.
            # Do not reinterpret the first execution frame as synthetic 0 deg.
            clearance_commands[finger] = float(fingers.get(finger, 0.0))

        else:
            clearance_commands[
                finger
            ] = min(
                max(
                    float(
                        inactive_clearance_deg
                    ),
                    0.0,
                ),
                90.0,
            )

    if any(
        clearance_commands[
            finger
        ]
        >
        1e-6
        for finger
        in LONG_FINGERS
    ):
        pre_shape_safe, pre_shape_reason = segment_mesh_safe(
            solver,
            arcs,
            q_now,
            fingers,
            q_now,
            clearance_commands,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=24,
        )

        if not pre_shape_safe:
            raise RuntimeError(
                "PRE_SHAPE_PATH_UNSAFE_"
                +
                pre_shape_reason
            )

        add_keyframe(
            trajectory,
            "PRE_SHAPE_HAND",
            0.45,
            q_now,
            clearance_commands,
            [],
        )

        print(
            "    [V4.12-ARC PRE-SHAPE]",
            {
                finger:
                    round(
                        float(
                            clearance_commands[
                                finger
                            ]
                        ),
                        1,
                    )
                for finger
                in LONG_FINGERS
            },
        )

    fingers = dict(
        clearance_commands
    )

    # --------------------------------------------------------
    # V101 DIRECT ZERO-PREBEND OPEN-HAND APPROACH.
    #
    # The endpoint IK palm pose is reached with straight fingers.  The measured
    # 0-to-contact arc is then executed at that same J1-J5 pose, followed by the
    # full safe coupled curl. There is no separate pre-bent handoff.
    # --------------------------------------------------------

    endpoint_q = np.asarray(best["q"], dtype=np.float64)
    endpoint_palm_p, endpoint_palm_r = solver.compute_forward_kinematics(
        PALM_FRAME,
        endpoint_q,
    )

    approach_normal = normalize(mode["normal"])
    if approach_normal is None:
        raise RuntimeError("APPROACH_NORMAL_INVALID")

    lane_spec = build_target_lane_spec(
        arcs,
        mode,
        np.asarray(endpoint_palm_r, dtype=np.float64),
        phases,
        template["active"],
    )
    if lane_spec is None:
        raise RuntimeError("TARGET_LANE_SPEC_INVALID")
    if best.get("path_validation_mode") == "PROVISIONAL_MOVING_CONTACT":
        lane_spec = dict(lane_spec)
        lane_spec["active_lane_cross_m"] = TARGET_LANE_CROSS_HARD_M
        lane_spec["active_patch_radius_m"] = TARGET_PATCH_RADIUS_HARD_M
        print(
            "[V103 PROVISIONAL CONTACT CORRIDOR]",
            "cross_mm=", round(TARGET_LANE_CROSS_HARD_M * 1000.0, 1),
            "radius_mm=", round(TARGET_PATCH_RADIUS_HARD_M * 1000.0, 1),
            "moving_center_limit_mm=", round(V103_EFFECTIVE_CONTACT_CENTER_MAX_M * 1000.0, 1),
            "base_guard_unchanged=True",
        )
    best["runtime_lane_spec"] = lane_spec

    # V101 direct full-stroke policy.  The selected endpoint already contains
    # the IK palm pose whose measured four-finger arcs intersect the requested
    # patch.  Reach that pose with all four fingers straight, then execute the
    # measured arcs from 0 degrees.  Do not solve or execute a separate bent
    # handoff pose.
    open_endpoint_q = np.asarray(endpoint_q, dtype=np.float64).copy()
    open_endpoint_q[4] = np.clip(
        open_endpoint_q[4] + math.radians(DIRECT_APPROACH_WRIST_LIFT_DEG),
        lower[4],
        upper[4],
    )
    direct_open_wrist_lift_deg = math.degrees(
        float(open_endpoint_q[4] - endpoint_q[4])
    )
    planned_sync_precontact = dict(fingers)
    phase_handoff = {
        "q": open_endpoint_q,
        "commands": planned_sync_precontact,
        "lead_deg": 0.0,
        "clearance_m": 0.0,
        "lane_details": {},
    }
    approach_position, approach_rotation = solver.compute_forward_kinematics(
        PALM_FRAME,
        open_endpoint_q,
    )
    approach_position = np.asarray(approach_position, dtype=np.float64)
    approach_rotation = np.asarray(approach_rotation, dtype=np.float64)
    print(
        "    [V101 DIRECT ZERO-PREBEND SCRATCH POSE]",
        "lead_deg=", round(float(phase_handoff["lead_deg"]), 1),
        "clearance_mm=", round(float(phase_handoff["clearance_m"]) * 1000.0, 1),
        "j5_open_lift_deg=", round(float(direct_open_wrist_lift_deg), 2),
        "q_deg=", np.round(np.degrees(open_endpoint_q), 2),
        "geometry=", phase_handoff["lane_details"],
    )

    far_solution = None
    far_metrics = None
    far_offset = None

    # V101 direct-open approach. First test the exact joint-space sweep that
    # execution will use from the measured current pose to the verified open
    # handoff. If the whole-hand mesh sweep is safe, one continuous keyframe
    # replaces the former 60/55/50/.../5/0 mm IK staircase. The staged route
    # remains an automatic fallback for targets whose direct sweep is blocked.
    direct_safe, direct_reason = segment_mesh_safe(
        solver,
        arcs,
        q_now,
        fingers,
        open_endpoint_q,
        fingers,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
        samples=96,
        allowed_contact_fingers=set(),
    )
    direct_metrics = approach_surface_metrics(
        solver,
        arcs,
        open_endpoint_q,
        fingers,
        working_fingers,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
    )
    if not direct_safe or direct_metrics['premature_contacts']:
        # Near base, the fixed 3-degree open pose may already violate the
        # hand/base margin. Search an open wrist pose before rejecting this
        # endpoint; the acquisition below still starts with straight fingers.
        for lift_deg in (5., 7., 9., 12., 15.):
            trial_q = np.asarray(endpoint_q, dtype=np.float64).copy()
            trial_q[4] += math.radians(lift_deg)
            if np.any(trial_q < lower) or np.any(trial_q > upper):
                continue
            trial_p, trial_r = solver.compute_forward_kinematics(PALM_FRAME,trial_q)
            if palm_back_alignment(arcs,trial_r,lane_spec['normal'])['angle_deg'] > palm_alignment_limit_deg():
                continue
            metrics = approach_surface_metrics(solver,arcs,trial_q,fingers,working_fingers,
                back_points,back_normals,back_tree,back_mesh_query)
            if metrics['premature_contacts']:
                continue
            safe, reason = segment_mesh_safe(solver,arcs,q_now,fingers,trial_q,fingers,
                back_points,back_normals,back_tree,back_mesh_query,samples=96,
                allowed_contact_fingers=set())
            if not safe:
                continue
            open_endpoint_q = trial_q
            direct_open_wrist_lift_deg = lift_deg
            phase_handoff['q'] = trial_q.copy()
            approach_position, approach_rotation = np.asarray(trial_p),np.asarray(trial_r)
            direct_safe, direct_reason, direct_metrics = True,reason,metrics
            print('[V103 OPEN WRIST FALLBACK] lift_deg=',lift_deg,'full_sweep=PASS')
            break
    direct_route_used = bool(
        direct_safe and not direct_metrics["premature_contacts"]
    )
    if direct_route_used:
        far_solution = np.asarray(open_endpoint_q, dtype=np.float64).copy()
        far_metrics = direct_metrics
        far_offset = 0.0
        print(
            "    [V101 DIRECT OPEN APPROACH PASS]",
            "samples=96",
            "min_all_mm=", round(float(direct_metrics["min_all_distance"]) * 1000.0, 2),
            "min_working_mm=", round(float(direct_metrics["min_working_distance"]) * 1000.0, 2),
        )
    else:
        print(
            "    [V101 DIRECT OPEN SCRATCH POSE REJECT]",
            "reason=", (
                str(direct_reason)
                if not direct_safe
                else "STRAIGHT_FINGER_PRECONTACT"
            ),
        )
        # Do not abort here.  The staged offset search below is the intended
        # fallback for precisely this case; the old unconditional raise made
        # that fallback unreachable and converted recoverable approach blocks
        # into a hard "no safe four-finger trajectory" rejection.
        for candidate_offset in APPROACH_FAR_SEARCH_OFFSETS_M:
            candidate_q = solve_offset_pose(
                solver,
                approach_position,
                approach_rotation,
                approach_normal,
                float(candidate_offset),
                q_now,
                home_q,
                lower,
                upper,
            )
            if candidate_q is None:
                continue

            candidate_safe, _ = segment_mesh_safe(
                solver,
                arcs,
                q_now,
                fingers,
                candidate_q,
                fingers,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                samples=28,
            )
            if not candidate_safe:
                continue

            candidate_metrics = approach_surface_metrics(
                solver,
                arcs,
                candidate_q,
                fingers,
                working_fingers,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
            )
            if candidate_metrics["premature_contacts"]:
                continue

            far_solution = np.asarray(candidate_q, dtype=np.float64)
            far_metrics = candidate_metrics
            far_offset = float(candidate_offset)
            break

    if far_solution is None:
        raise RuntimeError("ARC_APPROACH_NO_CLEAR_START")

    far_stage_name = (
        "ARC_APPROACH_DIRECT_OPEN_SCRATCH_POSE"
        if direct_route_used
        else "ARC_APPROACH_CLEAR_" + str(int(round(far_offset * 1000.0))) + "MM"
    )
    add_keyframe(
        trajectory,
        far_stage_name,
        arm_move_duration(q_now, far_solution, 0.60),
        far_solution,
        fingers,
        [],
    )

    print(
        "    [V4.12-ARC OPEN APPROACH CLEAR]",
        far_stage_name,
        "min_all=",
        round(far_metrics["min_all_distance"] * 1000.0, 2),
        "mm",
    )

    q_now = far_solution
    current_open_offset = float(far_offset)
    open_path_block_reason = None
    reached_endpoint_open = bool(direct_route_used)

    inward_offsets = (
        []
        if direct_route_used
        else generate_adaptive_inward_offsets(far_offset)
    )
    for inward_offset in inward_offsets:
        inward_offset = float(inward_offset)

        if inward_offset <= 1e-9:
            candidate_q = open_endpoint_q.copy()
            stage_name = "ARC_APPROACH_ENDPOINT_OPEN"
        else:
            candidate_q = solve_offset_pose(
                solver,
                approach_position,
                approach_rotation,
                approach_normal,
                inward_offset,
                q_now,
                endpoint_q,
                lower,
                upper,
            )
            if candidate_q is None:
                open_path_block_reason = (
                    "IK_FAIL_" + str(int(round(inward_offset * 1000.0))) + "MM"
                )
                print(
                    "    [V9 OPEN APPROACH HANDOFF]",
                    "last_safe_offset_mm=",
                    round(current_open_offset * 1000.0, 1),
                    "blocked=",
                    open_path_block_reason,
                )
                break
            stage_name = "ARC_APPROACH_OPEN_" + str(int(round(inward_offset * 1000.0))) + "MM"

        candidate_safe, candidate_reason = segment_mesh_safe(
            solver,
            arcs,
            q_now,
            fingers,
            candidate_q,
            fingers,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            samples=30,
        )
        if not candidate_safe:
            # V9: do not force an OPEN distal link through BackSurface merely
            # to reach endpoint_q.  Stop at the last safe open pose and hand
            # the rest of the motion to the measured finger-arc transition.
            open_path_block_reason = candidate_reason
            print(
                "    [V9 OPEN APPROACH HANDOFF]",
                "last_safe_offset_mm=",
                round(current_open_offset * 1000.0, 1),
                "next_offset_mm=",
                round(inward_offset * 1000.0, 1),
                "blocked=",
                candidate_reason,
            )
            break

        candidate_metrics = approach_surface_metrics(
            solver,
            arcs,
            candidate_q,
            fingers,
            working_fingers,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
        )

        if candidate_metrics["premature_contacts"]:
            first_contact = candidate_metrics["premature_contacts"][0]
            open_path_block_reason = (
                "STRAIGHT_FINGER_PRECONTACT_"
                + str(first_contact[0])
                + "_raw_mm="
                + str(round(float(first_contact[2]) * 1000.0, 2))
            )
            print(
                "    [V9 OPEN APPROACH HANDOFF]",
                "last_safe_offset_mm=",
                round(current_open_offset * 1000.0, 1),
                "blocked=",
                open_path_block_reason,
            )
            break

        # Distance alone is never a rejection reason. A straight working tip
        # may pass arbitrarily close to the back as long as there is no actual
        # contact/penetration and the whole-hand swept geometry is safe.
        add_keyframe(
            trajectory,
            stage_name,
            arm_move_duration(q_now, candidate_q, 0.16),
            candidate_q,
            fingers,
            [],
        )

        print(
            "    [V4.12-ARC OPEN APPROACH SAFE]",
            stage_name,
            "min_all=",
            round(candidate_metrics["min_all_distance"] * 1000.0, 2),
            "mm",
            "min_working=",
            round(candidate_metrics["min_working_distance"] * 1000.0, 2),
            "mm",
        )

        q_now = np.asarray(candidate_q, dtype=np.float64)
        current_open_offset = inward_offset
        if inward_offset <= 1e-9:
            reached_endpoint_open = True

    if open_path_block_reason is not None:
        print(
            "    [V100 SEQUENTIAL OPEN ARC HANDOFF READY]",
            "open_stage_offset_mm=",
            round(current_open_offset * 1000.0, 1),
            "reason=",
            open_path_block_reason,
            "working_open=",
            {f: round(float(fingers.get(f, 0.0)), 1) for f in working_fingers},
            "first_finger=",
            sequence[0] if sequence else "NONE",
            "arm_assist_trigger=",
            "near_surface_6mm_or_phase_" + str(round(ACQUIRE_PHASE_START_DEG, 1)),
            "other_fingers_hold_open=",
            True,
        )
    else:
        print(
            "    [V4.12-ARC4 SCRATCH ARC STAGING READY]",
            "endpoint_q_deg=",
            np.round(np.degrees(q_now), 2),
            "working_open=",
            {f: round(float(fingers.get(f, 0.0)), 1) for f in working_fingers},
            "endpoint_open_reached=",
            bool(reached_endpoint_open),
            "open_contact_allowed=",
            False,
        )

    # Save the verified open handoff so later scratch cycles can return to the
    # same reachable, collision-free starting state.
    cycle_open_q = np.asarray(q_now, dtype=np.float64).copy()
    cycle_open_fingers = dict(fingers)
    retreat_approach_qs = [frame["q"].copy() for frame in trajectory]

    # The arm has reached the phase-compensated open pose.  Perform the exact
    # four-finger precontact posture for which that pose was solved, then remove
    # the palm compensation continuously while all four measured arcs curl.
    precurve_safe, precurve_reason = segment_mesh_safe(
        solver,
        arcs,
        q_now,
        fingers,
        q_now,
        planned_sync_precontact,
        back_points,
        back_normals,
        back_tree,
        back_mesh_query,
        samples=34,
        allowed_contact_fingers=set(),
    )
    if not precurve_safe:
        raise RuntimeError("SYNC_PRECONTACT_PATH_UNSAFE_" + str(precurve_reason))
    max_precurve_change = max(
        abs(float(planned_sync_precontact[finger]) - float(fingers.get(finger, 0.0)))
        for finger in working_fingers
    )
    if max_precurve_change > 0.25:
        add_keyframe(
            trajectory,
            "ARC_SYNCHRONIZED_PRECONTACT",
            max(0.45, 0.012 * max_precurve_change),
            q_now,
            planned_sync_precontact,
            [],
            authorized_contact_fingers=set(),
        )
    else:
        print(
            "    [V101 ZERO-PREBEND PRECONTACT SKIPPED]",
            "max_change_deg=", round(float(max_precurve_change), 3),
        )
    fingers = dict(planned_sync_precontact)
    print(
        "    [V102 ASYNCHRONOUS FOUR-FINGER PRECONTACT]",
        "commands=",
        {f: round(float(fingers.get(f, 0.0)), 1) for f in LONG_FINGERS},
        "contact_authorized=",
        False,
    )

    # Preserve the original endpoint corridor first: 2444's known-good path
    # uses it.  If the current policy rejects that corridor at first-owner
    # acquisition, retry once with the measured open pose actually executed by
    # the wave.  Both variants go through the same swept-mesh and depth gates;
    # this is a deterministic geometry fallback, not a relaxed safety path.
    raw_acquisition_q = np.asarray(endpoint_q, dtype=np.float64).copy()
    raw_acquisition_p = np.asarray(endpoint_palm_p, dtype=np.float64).copy()
    raw_acquisition_r = np.asarray(endpoint_palm_r, dtype=np.float64).copy()
    open_acquisition_q = np.asarray(q_now, dtype=np.float64).copy()
    open_acquisition_p, open_acquisition_r = solver.compute_forward_kinematics(
        PALM_FRAME, open_acquisition_q
    )
    open_acquisition_p = np.asarray(open_acquisition_p, dtype=np.float64)
    open_acquisition_r = np.asarray(open_acquisition_r, dtype=np.float64)
    print(
        "    [V103 ACQUISITION OPEN PRECONTACT POSE]",
        "raw_endpoint_q_deg=", np.round(np.degrees(endpoint_q), 2),
        "actual_wave_q_deg=", np.round(np.degrees(q_now), 2),
        "planning_endpoint_q_deg=", np.round(np.degrees(raw_acquisition_q), 2),
        "raw_endpoint_reused=True",
    )

    synchronized_acquisition = None
    selected_acquisition_anchor = None
    for anchor_name, anchor_q, anchor_p, anchor_r in (
        ("RAW_ENDPOINT", raw_acquisition_q, raw_acquisition_p, raw_acquisition_r),
        ("ACTUAL_OPEN_Q", open_acquisition_q, open_acquisition_p, open_acquisition_r),
    ):
        print("    [V103 ACQUISITION ANCHOR TRY]", anchor_name,
              "q_deg=", np.round(np.degrees(anchor_q), 2))
        synchronized_acquisition = acquire_asynchronous_four_finger_contact(
            solver,
            arcs,
            mode,
            q_now,
            dict(fingers),
            anchor_q,
            phases,
            working_fingers,
            anchor_p,
            anchor_r,
            approach_normal,
            phase_handoff["clearance_m"],
            lower,
            upper,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            lane_spec,
            fixed_arm=False,
            linear_arm_sync=True,
            historical_replay=historical_replay,
        )
        if synchronized_acquisition is not None:
            selected_acquisition_anchor = anchor_name
            print("    [V103 ACQUISITION ANCHOR SELECTED]", anchor_name)
            break
        if anchor_name == "RAW_ENDPOINT":
            print("    [V103 ACQUISITION ANCHOR FALLBACK]",
                  "raw endpoint rejected; retrying actual open pose")
    if synchronized_acquisition is None:
        raise RuntimeError("ASYNCHRONOUS_FOUR_FINGER_ACQUISITION_FAIL")
    for frame in synchronized_acquisition["frames"]:
        trajectory.append(frame)
    acquisition_frame_template = [
        {
            **frame,
            "q": np.asarray(frame["q"], dtype=np.float64).copy(),
            "fingers": dict(frame["fingers"]),
            "active": list(frame.get("active", [])),
            "authorized_contact_fingers": list(
                frame.get("authorized_contact_fingers", frame.get("active", []))
            ),
        }
        for frame in synchronized_acquisition["frames"]
    ]
    q_now = synchronized_acquisition["q"]
    fingers = synchronized_acquisition["commands"]
    active = synchronized_acquisition["active"]
    async_contact_order = list(synchronized_acquisition["contact_order"])

    # Complete scratch cycles. Contact acquisition/controlled closure is the
    # start of the scratch stroke. Once measured depth qualifies, the planner
    # appends the shared four-finger curl continuation, then performs the
    # measured release/unload; contact qualification alone never ends a cycle.
    # Once cycle 1 has passed all contact/depth/release checks, retain its
    # complete wave segment.  Cycle 2 should replay that validated motion from
    # the same measured precontact state instead of launching a fresh expensive
    # contact search that can diverge or consume the full candidate budget.
    validated_wave_segment = None
    for cycle_index in range(1, SCRATCH_CYCLE_COUNT + 1):
        if cycle_index > 1:
            # After the local reset, combine the wrist's fast downward
            # re-approach with the harmless pre-contact finger curl whenever
            # the complete open-to-precontact sweep is safe.  This removes a
            # needless open-pose dwell while keeping contact unauthorized
            # until the first measured acquisition frame.
            overlap_reset = bool(trajectory[-1].get('local_cycle_reset'))
            overlap_reapproach_safe = False
            if overlap_reset:
                overlap_reapproach_safe, _ = segment_mesh_safe(
                    solver, arcs, q_now, fingers,
                    cycle_open_q, planned_sync_precontact,
                    back_points, back_normals, back_tree, back_mesh_query,
                    samples=96, allowed_contact_fingers=set(),
                )
            if overlap_reapproach_safe:
                finger_delta = max(
                    (abs(float(planned_sync_precontact[f])
                         - float(fingers.get(f, 0.0)))
                     for f in LONG_FINGERS),
                    default=0.0,
                )
                arm_time = arm_move_duration(q_now, cycle_open_q, .12)
                finger_time = finger_delta / max(SCRATCH_CURL_FINGER_SPEED_DEG_S, 1e-6)
                add_keyframe(
                    trajectory,
                    "CYCLE_REAPPROACH_PRECONTACT_OVERLAPPED_" + str(int(cycle_index)),
                    max(.12, arm_time, finger_time),
                    cycle_open_q,
                    planned_sync_precontact,
                    [],
                    authorized_contact_fingers=set(),
                )
                trajectory[-1]['overlapped_wrist_down_and_precontact'] = True
                q_now = cycle_open_q.copy()
                fingers = dict(planned_sync_precontact)
                print('[V103 INTER-CYCLE REAPPROACH]',
                      'overlap_wrist_down_and_precontact=True',
                      'cycle=', cycle_index,
                      'duration_s=', round(max(.12, arm_time, finger_time), 3),
                      'next=FIRST_MEASURED_CONTACT')
            else:
                reapproach_safe, reapproach_reason = segment_mesh_safe(
                    solver,
                    arcs,
                    q_now,
                    fingers,
                    cycle_open_q,
                    cycle_open_fingers,
                    back_points,
                    back_normals,
                    back_tree,
                    back_mesh_query,
                    samples=42,
                    allowed_contact_fingers=set(),
                )
                if not reapproach_safe:
                    raise RuntimeError(
                        "CYCLE_REAPPROACH_OPEN_PATH_UNSAFE_"
                        + str(int(cycle_index))
                        + "_"
                        + str(reapproach_reason)
                    )
                add_keyframe(
                    trajectory,
                    "CYCLE_REAPPROACH_OPEN_" + str(int(cycle_index)),
                    arm_move_duration(q_now, cycle_open_q,
                                      .12 if overlap_reset else .45),
                    cycle_open_q,
                    cycle_open_fingers,
                    [],
                )
                q_now = cycle_open_q.copy()
                fingers = dict(cycle_open_fingers)

                precurve_safe, precurve_reason = segment_mesh_safe(
                    solver,
                    arcs,
                    q_now,
                    fingers,
                    q_now,
                    planned_sync_precontact,
                    back_points,
                    back_normals,
                    back_tree,
                    back_mesh_query,
                    samples=34,
                    allowed_contact_fingers=set(),
                )
                if not precurve_safe:
                    raise RuntimeError(
                        "CYCLE_SYNC_PRECONTACT_PATH_UNSAFE_"
                        + str(int(cycle_index))
                        + "_"
                        + str(precurve_reason)
                    )
                if max_precurve_change > 0.25:
                    add_keyframe(
                        trajectory,
                        "ARC_SYNCHRONIZED_PRECONTACT_CYCLE_" + str(int(cycle_index)),
                        max(0.45, 0.012 * max_precurve_change),
                        q_now,
                        planned_sync_precontact,
                        [],
                        authorized_contact_fingers=set(),
                    )
                fingers = dict(planned_sync_precontact)

            for template_frame in acquisition_frame_template:
                repeated_frame = {
                    **template_frame,
                    "stage": str(template_frame["stage"])
                    + "_CYCLE_"
                    + str(int(cycle_index)),
                    "q": np.asarray(template_frame["q"], dtype=np.float64).copy(),
                    "fingers": dict(template_frame["fingers"]),
                    "active": list(template_frame.get("active", [])),
                    "authorized_contact_fingers": list(
                        template_frame.get(
                            "authorized_contact_fingers",
                            template_frame.get("active", []),
                        )
                    ),
                }
                trajectory.append(repeated_frame)
            q_now = np.asarray(synchronized_acquisition["q"], dtype=np.float64).copy()
            fingers = dict(synchronized_acquisition["commands"])
            active = list(synchronized_acquisition["active"])

        # V103 deliberately hands off at the first validated 1 mm crossing;
        # that crossing may involve only the shallow/deep fingers that have
        # actually made contact.  The contact wave owns the remaining fingers
        # and must be allowed to establish them instead of rejecting the handoff
        # as an artificial four-contact mismatch.
        if len(active) != len(sequence) and not bool(
            synchronized_acquisition.get("balanced_handoff", False)
        ):
            raise RuntimeError(
                "ACTIVE_FINGER_COUNT_MISMATCH_CYCLE_"
                + str(int(cycle_index))
                + "_expected="
                + str(len(sequence))
                + "_actual="
                + str(len(active))
            )

        # Do not insert a stationary "full contact" keyframe here.  The last
        # acquisition state flows directly into the first moving wave state so
        # demonstrations contain no artificial bottom dwell.
        wave_start_index = len(trajectory)
        if cycle_index > 1 and validated_wave_segment:
            replay_last = None
            for source_frame in validated_wave_segment:
                replay_frame = copy.deepcopy(source_frame)
                stage = str(replay_frame.get('stage', 'SCRATCH_WAVE'))
                replay_frame['stage'] = stage.replace(
                    '_CYCLE_1', '_CYCLE_' + str(int(cycle_index)) + '_REPLAY'
                )
                replay_frame['replayed_validated_wave'] = True
                trajectory.append(replay_frame)
                replay_last = replay_frame
            if replay_last is None:
                raise RuntimeError('VALIDATED_WAVE_REPLAY_EMPTY')
            q_now = np.asarray(replay_last['q'], dtype=np.float64).copy()
            fingers = dict(replay_last.get('fingers', {}))
            active = list(replay_last.get('active', []))
            print(
                '[V103 VALIDATED WAVE REPLAY]',
                'cycle=', int(cycle_index),
                'frames=', len(validated_wave_segment),
                'fresh_contact_search=False',
            )
        else:
            q_now, fingers, active = append_asynchronous_four_finger_wave(
                trajectory,
                solver,
                arcs,
                mode,
                q_now,
                fingers,
                working_fingers,
                async_contact_order,
                endpoint_q,
                lower,
                upper,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                lane_spec,
                cycle_index,
            )
            if cycle_index == 1:
                validated_wave_segment = [
                    copy.deepcopy(frame)
                    for frame in trajectory[wave_start_index:]
                ]
                print(
                    '[V103 VALIDATED WAVE CACHED]',
                    'cycle=1',
                    'frames=', len(validated_wave_segment),
                    'reuse_for_cycle_2=True',
                )
        if cycle_index < SCRATCH_CYCLE_COUNT:
            local_reset = append_v103_local_cycle_reset(
                trajectory,solver,arcs,q_now,fingers,cycle_open_q,cycle_open_fingers,
                lower,upper,back_points,back_normals,back_tree,back_mesh_query,cycle_index)
            if local_reset is not None:
                q_now,fingers,active = local_reset
                continue
        if trajectory[-1].get('early_exit'):
            q_now,fingers,active = append_v103_early_elbow_exit(
                trajectory,solver,arcs,q_now,fingers,home_q,
                back_points,back_normals,back_tree,back_mesh_query,cycle_index)
            continue
        q_now, fingers, active = append_adaptive_unload_and_open(
            trajectory,
            solver,
            arcs,
            mode,
            q_now,
            fingers,
            active,
            home_q,
            lower,
            upper,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            cycle_index,
        )

    # --------------------------------------------------------
    # Retreat and return Home.
    # --------------------------------------------------------

    def check_retreat_segment(a, b):
        return segment_mesh_safe(
            solver, arcs, a, fingers, b, fingers,
            back_points, back_normals, back_tree, back_mesh_query,
            samples=96, allowed_contact_fingers=set(),
        )

    def local_lift(a):
        p, r = solver.compute_forward_kinematics(PALM_FRAME, a)
        return solve_offset_pose(solver, p, r, approach_normal,
                                 SAFE_RETREAT_OFFSET_M / 10.0,
                                 a, home_q, lower, upper)

    return_route, retreat_method = v103_plan_retreat_route(
        q_now, retreat_approach_qs, home_q, lower, upper,
        check_retreat_segment, local_lift, prefer_direct_home=True,
    )
    print("[V103 RETREAT ROUTE PASS] method=", retreat_method,
          "segments=", len(return_route), "samples_per_segment=96",
          "open_fingers_fixed=True contact_authorized=False",
          "direct_home_preferred=True")
    for index, retreat_q in enumerate(return_route):
        is_home = index == len(return_route) - 1
        add_keyframe(trajectory, "RETURN_HOME" if is_home else "SAFE_RETREAT",
                     arm_move_duration(q_now, retreat_q,
                                       RETURN_HOME_MIN_TIME_S if is_home else 0.30),
                     retreat_q, fingers, [], authorized_contact_fingers=[])
        q_now = retreat_q

    add_keyframe(
        trajectory,
        "HOME_WAIT",
        0.20,
        home_q,
        fingers,
        [],
    )

    return trajectory


def validate_contact_lifecycle(trajectory):
    """Validate semantic contact-state transitions before geometric checks."""
    previous_active = set()
    contact_seen = False
    contact_release_seen = False

    for index, frame in enumerate(trajectory):
        stage = str(frame.get("stage", "UNKNOWN"))
        try:
            checked_finger_commands(frame.get("fingers", {}))
        except RuntimeError as exc:
            return False, str(exc) + "|stage=" + stage
        active = set(frame.get("active", []))
        authorized = set(frame.get("authorized_contact_fingers", active))

        if frame.get("contact_state_source") == "sampled_segment_v1":
            states = frame.get("predicted_contact_states")
            required = set(frame.get('scratch_required_fingers', LONG_FINGERS))
            if (not isinstance(states, dict) or set(states) != required
                    or any(s not in {"UNTOUCHED", "CONTACT", "RELEASED"} for s in states.values())):
                return False, "CONTACT_STATE_METADATA_INVALID|stage=" + stage
            sampled_active = {f for f, s in states.items() if s == "CONTACT"}
            if active != sampled_active:
                return False, "CONTACT_STATE_ACTIVE_MISMATCH|stage=" + stage

        if not active.issubset(authorized):
            return False, "ACTIVE_NOT_AUTHORIZED|stage=" + stage

        if stage.startswith(("APPROACH_", "ARC_APPROACH_")) and active:
            return False, "ACTIVE_DURING_OPEN_APPROACH|stage=" + stage

        is_strict_free_space = bool(
            stage.startswith(("UNLOAD_ALL", "OPEN_AFTER_UNLOAD"))
            or stage in {"SAFE_RETREAT", "RETURN_HOME", "HOME_WAIT"}
        )
        if is_strict_free_space and (active or authorized):
            return False, "CONTACT_STATE_LEAK_AFTER_UNLOAD|stage=" + stage

        added = active - previous_active
        removed = previous_active - active

        if added:
            contact_seen = True
            synchronized_commit = stage.startswith(
                ("ARC_SYNC_CONTACT_", "ARC_ASYNC_CONTACT_")
            )
            legacy_single_commit = (
                len(added) == 1
                and stage.startswith(("ARC_CONTACT_", "ACQUIRE_", "JOIN_CONTACT_"))
            )
            geometric_wave_recontact = (
                stage.startswith("SCRATCH_WAVE_ASYNC_")
                and all(frame.get("predicted_contact_states", {}).get(f) == "CONTACT" for f in added)
            )
            continuation_commit = (
                bool(frame.get("contact_continuation", False))
                and stage.startswith("SCRATCH_CURL_CONTACT_")
                and added.issubset(authorized)
                and all(
                    frame.get("predicted_contact_states", {}).get(f) == "CONTACT"
                    for f in added
                )
            )
            if not (
                synchronized_commit
                or legacy_single_commit
                or geometric_wave_recontact
                or continuation_commit
            ):
                return False, "INVALID_CONTACT_COMMIT|stage=" + stage + "|added=" + str(sorted(added))

        if removed:
            asynchronous_release = stage.startswith(
                ("SCRATCH_WAVE_ASYNC_", "FINGER_CURL_RELEASE_")
            )
            final_unload_release = bool(
                stage.startswith((
                    "UNLOAD_ALL",
                    "UNLOAD_J5_LIFT_",
                    "UNLOAD_STEP_",
                    # The overlapped inter-cycle reset opens the fingers and
                    # lifts the wrist in one checked no-contact segment.  It
                    # is a legitimate release, not a contact-state leak.
                    "UNLOAD_OPEN_OVERLAPPED_",
                )) and not active
            )
            if not (asynchronous_release or final_unload_release):
                return False, "INVALID_CONTACT_RELEASE|stage=" + stage
            contact_release_seen = True

        previous_active = active

    if contact_seen and (previous_active or not contact_release_seen):
        return False, "CONTACT_NEVER_RELEASED"

    return True, "PASS"



def validate_trajectory(
    solver,
    arcs,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    trajectory,
    lower,
    upper,
    mode,
    lane_spec=None,
):
    """
    Complete V4.6 trajectory validation.

    Final penetration checks use the actual USD BackSurface triangle mesh when
    available. This removes the fake "negative 30-60 mm penetration" caused by
    extending a point-cloud normal plane infinitely far from its sample point.
    """
    validation_started = __import__('time').perf_counter()
    quality_started = __import__('time').perf_counter()
    quality_ok, quality_reason = validate_continuous_scratch_quality(
        trajectory, solver, arcs, back_points, back_normals, back_tree,
        back_mesh_query, lane_spec=lane_spec)
    print('[V103 FINAL VALIDATION PHASE]', 'phase=contact_quality',
          'seconds=', round(__import__('time').perf_counter()-quality_started, 3),
          'pass=', bool(quality_ok))
    if not quality_ok:
        return False, quality_reason
    for i in range(
        1,
        len(
            trajectory
        ),
    ):
        a = trajectory[
            i - 1
        ]

        b = trajectory[
            i
        ]

        if (HAND_COLLISION_MODEL is not None
                and hasattr(HAND_COLLISION_MODEL, 'base_guard')
                and not bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False))):
            base_started = __import__('time').perf_counter()
            async_base_segment = str(b.get('stage', '')).startswith('SCRATCH_WAVE_ASYNC_')
            for sample_index, fraction in enumerate(np.linspace(0.,1.,96)):
                # Match the trajectory builder/executor interpolation.  This
                # also reuses exact BaseGuard cache entries from the earlier
                # swept-mesh check for non-async segments; the same 96-sample
                # replay and 10 mm threshold remain in force.
                t = float(fraction) if async_base_segment else smoothstep(float(fraction))
                cq = np.asarray(a['q']) + t*(np.asarray(b['q'])-np.asarray(a['q']))
                cf = {f:a['fingers'][f]+t*(b['fingers'][f]-a['fingers'][f]) for f in LONG_FINGERS}
                gap,link = HAND_COLLISION_MODEL.base_guard.clearance(cq,cf)
                if gap < V103_BASE_MIN_CLEARANCE_M:
                    return False, 'HAND_BASE_MARGIN_REPLAY|stage='+str(b['stage'])+'|link='+str(link)
                if sample_index % 8 == 0:
                    PALM_POLICY.check_candidate_budget()
            print('[V103 FINAL VALIDATION SEGMENT]', 'index=', i,
                  'of=', len(trajectory)-1, 'stage=', str(b.get('stage')),
                  'base_replay_seconds=', round(__import__('time').perf_counter()-base_started, 3))

        moving_contact_fingers = []
        async_wave_segment = str(b.get("stage", "")).startswith(
            "SCRATCH_WAVE_ASYNC_"
        )

        for finger in LONG_FINGERS:
            if (
                finger in a[
                    "active"
                ]
                and
                (
                    finger in b["active"]
                    or (
                        async_wave_segment
                        and finger in set(
                            b.get("authorized_contact_fingers", [])
                        )
                    )
                )
                and
                abs(
                    float(
                        b[
                            "fingers"
                        ].get(
                            finger,
                            0.0,
                        )
                    )
                    -
                    float(
                        a[
                            "fingers"
                        ].get(
                            finger,
                            0.0,
                        )
                    )
                )
                >
                1.0
            ):
                moving_contact_fingers.append(
                    finger
                )

        strict_scratch_motion = (
            str(b.get("stage", "")).startswith("SCRATCH_WAVE_")
            or str(b.get("stage", "")).startswith("SYNC_")
        )

        # Async quality is measured only over sampled in-contact motion across
        # acquisition + wave above. Airborne closure must not pass the legacy
        # whole-keyframe tangent test. All swept collision/limit gates below
        # still apply to every segment, including release and full closure.
        if moving_contact_fingers and strict_scratch_motion and not async_wave_segment:
            tangential_threshold_m = (
                ASYNC_MIN_COMBINED_TANGENTIAL_KEYFRAME_M
                if str(b.get("stage", "")).startswith("SCRATCH_WAVE_ASYNC_")
                else MIN_COMBINED_TANGENTIAL_KEYFRAME_M
            )
            tangent_ratio_threshold = (
                ASYNC_MIN_COMBINED_TANGENT_RATIO
                if str(b.get("stage", "")).startswith("SCRATCH_WAVE_ASYNC_")
                else MIN_COMBINED_TANGENT_RATIO
            )
            metrics = combined_motion_metrics(
                solver,
                arcs,
                mode,
                a[
                    "q"
                ],
                a[
                    "fingers"
                ],
                b[
                    "q"
                ],
                b[
                    "fingers"
                ],
                moving_contact_fingers,
            )

            for finger in moving_contact_fingers:
                metric = metrics[
                    finger
                ]

                if (
                    metric[
                        "tangential"
                    ]
                    <
                    tangential_threshold_m
                ):
                    return (
                        False,
                        "COMBINED_STROKE_TOO_SMALL"
                        + "|stage="
                        + str(
                            b[
                                "stage"
                            ]
                        )
                        + "|finger="
                        + finger
                        + "|tangent_mm="
                        + str(
                            round(
                                metric[
                                    "tangential"
                                ]
                                *
                                1000.0,
                                2,
                            )
                        ),
                    )

                if (
                    metric[
                        "tangent_ratio"
                    ]
                    <
                    tangent_ratio_threshold
                ):
                    return (
                        False,
                        "COMBINED_TANGENT_LOW"
                        + "|stage="
                        + str(
                            b[
                                "stage"
                            ]
                        )
                        + "|finger="
                        + finger
                        + "|ratio="
                        + str(
                            round(
                                metric[
                                    "tangent_ratio"
                                ],
                                3,
                            )
                        ),
                    )

                # V4.8: combined normal displacement is a SOFT term only.
                # Actual USD triangle-mesh penetration remains the hard
                # safety criterion below.

        for fraction in np.linspace(
            0.0,
            1.0,
            18,
        ):
            alpha = smoothstep(
                fraction
            )

            q = (
                a[
                    "q"
                ]
                +
                alpha
                *
                (
                    b[
                        "q"
                    ]
                    -
                    a[
                        "q"
                    ]
                )
            )

            if (
                np.any(
                    q
                    <
                    lower
                    -
                    1e-8
                )
                or
                np.any(
                    q
                    >
                    upper
                    +
                    1e-8
                )
            ):
                return (
                    False,
                    "JOINT_LIMIT"
                    + "|stage="
                    + str(
                        b[
                            "stage"
                        ]
                    )
                    + "|q_deg="
                    + np.array2string(
                        np.round(
                            np.degrees(
                                q
                            ),
                            2,
                        ),
                        separator=",",
                    ),
                )


            palm_p, palm_r = solver.compute_forward_kinematics(
                PALM_FRAME,
                q,
            )

            palm_p = np.asarray(
                palm_p,
                dtype=np.float64,
            )

            palm_r = np.asarray(
                palm_r,
                dtype=np.float64,
            )

            for finger in LONG_FINGERS:
                angle = (
                    a[
                        "fingers"
                    ].get(
                        finger,
                        0.0,
                    )
                    +
                    alpha
                    *
                    (
                        b[
                            "fingers"
                        ].get(
                            finger,
                            0.0,
                        )
                        -
                        a[
                            "fingers"
                        ].get(
                            finger,
                            0.0,
                        )
                    )
                )

                local_point, _ = interpolate_arc(
                    arcs,
                    finger,
                    angle,
                )

                world_point = (
                    palm_p
                    +
                    palm_r
                    @ local_point
                )

                relation = final_surface_relation(
                    world_point,
                    back_mesh_query,
                    back_points,
                    back_normals,
                    back_tree,
                )

                signed = relation[
                    "signed"
                ]

                distance = relation[
                    "distance"
                ]

                # Planned contact authorization is not permission to touch
                # anywhere. Both an already-active finger and an acquiring
                # finger must enter the surface only inside its finite target
                # region. Once unload creates real clearance, free-space motion
                # is no longer lane/patch constrained.
                source_active_fingers = set(a.get("active", []))
                target_active_fingers = set(b.get("active", []))
                source_authorized_fingers = set(
                    a.get(
                        "authorized_contact_fingers",
                        a.get("active", []),
                    )
                )
                target_authorized_fingers = set(
                    b.get(
                        "authorized_contact_fingers",
                        b.get("active", []),
                    )
                )

                # Contact state belongs to a sampled pose, not to the whole
                # segment.  In particular alpha=0 is exactly frame `a`; using
                # the union here used to apply frame `b`'s newly-active state
                # one frame early.  That rejected a valid acquisition when the
                # old clear pose was inside the 75 mm GUI patch but just outside
                # the 65 mm executable-contact reserve.
                at_source = bool(float(fraction) <= 1e-9)
                at_target = bool(float(fraction) >= 1.0 - 1e-9)
                authorized_contact_fingers = (
                    source_authorized_fingers
                    if at_source
                    else source_authorized_fingers | target_authorized_fingers
                )

                # Fingers already active at the start retain the tighter
                # executable reserve throughout the segment.  A newly
                # acquiring finger uses the immutable GUI hard region during
                # interpolation and receives the tighter reserve only at the
                # target keyframe where contact is actually committed.
                planned_region_required = set(source_active_fingers)
                if at_target:
                    planned_region_required.update(target_active_fingers)
                hard_region_required = (
                    authorized_contact_fingers | planned_region_required
                )
                stage_name = str(b.get('stage', ''))
                patch_distance_limit = WARM_CONTACT_MAX_DISTANCE_M
                if (stage_name.startswith('ARC_ASYNC_PRECONTACT_')
                        and finger not in source_active_fingers
                        and finger not in target_active_fingers):
                    # Free-space precontact may enter the GUI patch from the
                    # outside.  Do not apply the 75 mm contact-region gate at
                    # 5--8 mm airborne clearance; mesh/swept collision remains
                    # authoritative and the patch gate returns at <=4 mm.
                    patch_distance_limit = min(
                        WARM_CONTACT_MAX_DISTANCE_M,
                        ASYNC_PRECONTACT_PATCH_NEAR_DISTANCE_M,
                    )
                if (
                    lane_spec is not None
                    and finger in hard_region_required
                    and float(distance) <= float(patch_distance_limit)
                    and not (stage_name.startswith('SCRATCH_WAVE_ASYNC_')
                             and float(signed) > V103_SAFETY_CLEARANCE_M)
                ):
                    # Apply the selected tier while near contact, including
                    # acquisition and unload. Free-space Home need not align.
                    contact_palm_angle = float(palm_back_alignment(
                        arcs, palm_r, lane_spec["normal"],
                    )["angle_deg"])
                    if not PALM_POLICY.angle_ok(contact_palm_angle):
                        return False, (
                            "PALM_ALIGNMENT_SWEEP|stage=" + str(b.get("stage"))
                            + "|angle_deg=" + str(round(contact_palm_angle, 3))
                            + "|limit_deg=" + str(palm_alignment_limit_deg())
                        )
                    lane_state = lane_relation_for_world_point(
                        world_point,
                        lane_spec,
                        finger,
                    )
                    planned_region = finger in planned_region_required
                    active_cross_limit, active_radius_limit = active_contact_lane_limits(lane_spec)
                    cross_limit = (
                        active_cross_limit
                        if planned_region
                        else TARGET_LANE_CROSS_HARD_M
                    )
                    patch_limit = (
                        active_radius_limit
                        if planned_region
                        else TARGET_PATCH_RADIUS_HARD_M
                    )
                    if float(lane_state["cross_error"]) > cross_limit:
                        return (
                            False,
                            "ACTIVE_LANE_DEPARTURE"
                            + "|stage=" + str(b.get("stage", "UNKNOWN"))
                            + "|finger=" + str(finger)
                            + "|cross_mm="
                            + str(round(float(lane_state["cross_error"]) * 1000.0, 2))
                            + "|limit_mm="
                            + str(round(float(cross_limit) * 1000.0, 2))
                            + "|distance_mm="
                            + str(round(float(distance) * 1000.0, 2))
                            + "|alpha=" + str(round(float(fraction), 3)),
                        )
                    if float(lane_state["patch_radius"]) > patch_limit:
                        return (
                            False,
                            "ACTIVE_PATCH_DEPARTURE"
                            + "|stage=" + str(b.get("stage", "UNKNOWN"))
                            + "|finger=" + str(finger)
                            + "|radius_mm="
                            + str(round(float(lane_state["patch_radius"]) * 1000.0, 2))
                            + "|limit_mm="
                            + str(round(float(patch_limit) * 1000.0, 2))
                            + "|distance_mm="
                            + str(round(float(distance) * 1000.0, 2))
                            + "|alpha=" + str(round(float(fraction), 3)),
                        )

                # V12: validation must use the same planned-contact
                # authorization that was used while BUILDING this segment.
                # Falling back to `active` preserves the old behavior for
                # legacy frames, while precurve/acquisition frames can now
                # explicitly authorize the intended working finger before it
                # is formally latched active. This does NOT authorize palm,
                # proximal or middle links; segment_mesh_safe remains strict
                # for those links and only permits the working distal phalanx
                # within the existing 8 mm hard depth envelope.
                hard_limit = (
                    HARD_PENETRATION_M
                    if finger in authorized_contact_fingers
                    else INACTIVE_PENETRATION_M
                )

                if signed < hard_limit:
                    return (
                        False,
                        (
                            "PREDICTED_OVER_PENETRATION"
                            if finger in authorized_contact_fingers
                            else "PREDICTED_UNPLANNED_FINGER_PENETRATION"
                        )
                        + "|stage="
                        + str(
                            b[
                                "stage"
                            ]
                        )
                        + "|finger="
                        + finger
                        + "|signed_mm="
                        + str(
                            round(
                                signed
                                *
                                1000.0,
                                2,
                            )
                        )
                        + "|raw_signed_mm="
                        + str(
                            round(
                                float(
                                    relation[
                                        "raw_signed"
                                    ]
                                )
                                *
                                1000.0,
                                2,
                            )
                        )
                        + "|distance_mm="
                        + str(
                            round(
                                distance
                                *
                                1000.0,
                                2,
                            )
                        )
                        + "|source="
                        + str(
                            relation[
                                "source"
                            ]
                        )
                        + "|alpha="
                        + str(
                            round(
                                float(
                                    fraction
                                ),
                                3,
                            )
                        )
                        + "|q_deg="
                        + np.array2string(
                            np.round(
                                np.degrees(
                                    q
                                ),
                                2,
                            ),
                            separator=",",
                        ),
                    )

    print('[V103 FINAL VALIDATION COMPLETE]', 'segments=', len(trajectory)-1,
          'seconds=', round(__import__('time').perf_counter()-validation_started, 3),
          'base_min_mm=', round(V103_BASE_MIN_CLEARANCE_M*1000., 3),
          'planned_depth_max_mm=', round(V103_PLANNED_DEPTH_MAX_M*1000., 3),
          'diagnostic_depth_cap_mm=', round(v103_effective_runtime_depth_cap_m()*1000., 3),
          'diagnostic_depth_overflow=', bool(V103_ALLOW_PLANNED_DEPTH_CAP_OVERFLOW))
    return (
        True,
        "PASS",
    )


def inactive_clearance_options_for_candidate(
    candidate,
):
    """
    Deterministic inactive-finger posture policy.

    The legacy non-working-finger policy always folded to 90 degrees. Near base
    that can move the omitted pinky *toward* the pedestal. For the explicit RMI
    fallback, try a bounded set from open to folded; every option still passes
    the same whole-hand skin/base swept checks. Other modes remain unchanged.
    """
    working = len(
        candidate[
            "template"
        ][
            "active"
        ]
    )

    inactive = max(
        0,
        len(
            LONG_FINGERS
        )
        -
        working,
    )

    if inactive == 0:
        return [
            0.0
        ]

    mode = candidate.get('mode', {})
    active = set(candidate.get('template', {}).get('active', []))
    if (mode.get('v103_near_base_three_finger')
            and active == {'index', 'middle', 'ring'}
            and inactive == 1):
        ranked = candidate.get('inactive_parking_options', [0.0,30.0,60.0,90.0])
        if len(ranked) == 4 and set(ranked) == {0.,30.,60.,90.}:
            return list(ranked)
        return [0.0,30.0,60.0,90.0]

    return [
        90.0
    ]


def try_candidate_trajectory(
    solver,
    arcs,
    candidate,
    current_q,
    home_q,
    lower,
    upper,
    back_points,
    back_normals,
    back_tree,
    back_mesh_query,
    current_fingers=None,
    historical_replay=False,
):
    """
    Use the deterministic inactive-finger pre-fold posture for one endpoint-valid
    candidate. Return immediately when a complete safe trajectory is found.
    """
    reasons = []

    clearance_options = inactive_clearance_options_for_candidate(
        candidate
    )

    print(
        "  [V4.12-ARC INACTIVE PRE-FOLD]",
        candidate["template"]["name"],
        "options=",
        clearance_options,
    )

    for clearance_deg in clearance_options:
        try:
            trajectory = build_trajectory(
                solver,
                arcs,
                candidate,
                current_q,
                home_q,
                lower,
                upper,
                back_points,
                back_normals,
                back_tree,
                back_mesh_query,
                inactive_clearance_deg=clearance_deg,
                current_fingers=current_fingers,
                historical_replay=historical_replay,
            )

        except Exception as exc:
            reason = (
                "BUILD_FAIL_clearance="
                +
                str(
                    clearance_deg
                )
                +
                "_"
                +
                str(
                    exc
                )
            )

            reasons.append(
                reason
            )

            print(
                "  [V4.12-ARC TRAJ REJECT]",
                candidate["template"]["name"],
                reason,
            )

            continue

        lifecycle_ok, lifecycle_reason = validate_contact_lifecycle(trajectory)
        if not lifecycle_ok:
            reason = "CONTACT_LIFECYCLE_INVALID_" + str(lifecycle_reason)
            reasons.append(reason)
            print(
                "  [V100 CONTACT LIFECYCLE REJECT]",
                candidate["template"]["name"],
                reason,
            )
            continue

        safe, reason = validate_trajectory(
            solver,
            arcs,
            back_points,
            back_normals,
            back_tree,
            back_mesh_query,
            trajectory,
            lower,
            upper,
            candidate["mode"],
            lane_spec=candidate.get("runtime_lane_spec"),
        )

        if safe:
            candidate[
                "trajectory_safe"
            ] = True

            candidate[
                "trajectory_reason"
            ] = "PASS"

            candidate[
                "trajectory"
            ] = trajectory

            candidate[
                "clearance_deg"
            ] = float(
                clearance_deg
            )

            print(
                "  [V4.12-ARC TRAJECTORY SAFE]",
                candidate["template"]["name"],
                "clearance=",
                clearance_deg,
                "deg",
            )

            return (
                True,
                trajectory,
                "PASS",
            )

        reject_text = (
            "clearance="
            +
            str(
                clearance_deg
            )
            +
            ":"
            +
            reason
        )

        reasons.append(
            reject_text
        )

        print(
            "  [V4.12-ARC TRAJ REJECT]",
            candidate["template"]["name"],
            reject_text,
        )

    candidate[
        "trajectory_safe"
    ] = False

    candidate[
        "trajectory_reason"
    ] = (
        " || ".join(
            reasons
        )
    )

    return (
        False,
        None,
        candidate[
            "trajectory_reason"
        ],
    )


# ============================================================
# OUTPUT
# ============================================================

def save_plan(target_id, best):
    g = best["geometry"]
    q = np.degrees(best["q"])

    with open(PLAN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "target_id",
                "mode",
                "template",
                "sequence",
                "strategy_yaw_deg",
                "ik_kind",
                "valid",
                "strong",
                "coverage_mm",
                "normal_rmse_mm",
                "tangent",
                "stroke_mm",
                "j1_deg",
                "j2_deg",
                "j3_deg",
                "j4_deg",
                "j5_deg",
            ]
        )

        writer.writerow(
            [
                target_id,
                best["mode"]["name"],
                best["template"]["name"],
                "|".join(best["template"]["sequence"]),
                best["strategy_yaw"],
                best["ik_kind"],
                best["valid"],
                best["strong"],
                g.get("coverage", 999.0) * 1000.0,
                g.get("normal_rmse", 999.0) * 1000.0,
                g.get("tangent", 0.0),
                g.get("stroke", 0.0) * 1000.0,
                q[0],
                q[1],
                q[2],
                q[3],
                q[4],
            ]
        )


def save_trajectory(trajectory):
    with open(TRAJECTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "stage",
                "duration_s",
                "j1_deg",
                "j2_deg",
                "j3_deg",
                "j4_deg",
                "j5_deg",
                "index_progress_pct",
                "middle_progress_pct",
                "ring_progress_pct",
                "pinky_progress_pct",
                "active",
                "hand_model_id",
            ] + [f + "_" + suffix + "_target_deg" for f in LONG_FINGERS for suffix in HAND_MODEL.SUFFIXES]
        )

        for i, item in enumerate(trajectory):
            q = np.degrees(item["q"])

            writer.writerow(
                [
                    i,
                    item["stage"],
                    item["duration"],
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                    q[4],
                    item["fingers"]["index"],
                    item["fingers"]["middle"],
                    item["fingers"]["ring"],
                    item["fingers"]["pinky"],
                    "|".join(item["active"]),
                    HAND_MODEL.MODEL_ID,
                ] + [angle for finger in LONG_FINGERS
                     for angle in HAND_MODEL.joint_targets(checked_finger_commands(item["fingers"])[finger])]
            )


def persist_v104_physical_success(runtime):
    """Called by the batch bridge only after its measured dataset-pass gate."""
    saved = getattr(runtime, "saved_execution", None)
    if not isinstance(saved, dict):
        raise ValueError("V104_SUCCESS_WITHOUT_SAVED_EXECUTION")
    candidate_id = saved.get("candidate_id")
    target_xyz = saved.get("target_xyz")
    target_normal = saved.get("target_normal")
    if candidate_id is None or target_xyz is None or target_normal is None:
        raise ValueError("V104_SUCCESS_METADATA_MISSING")
    destination = V104_CACHE.save_physical_success(
        PROJECT_DIR, candidate_id=candidate_id, target_xyz=target_xyz,
        target_normal=target_normal, back_points=runtime.back_points,
        back_normals=runtime.back_normals, saved_execution=saved,
    )
    print("[V104 PHYSICAL SUCCESS CACHED]", destination,
          "future_exact_revalidation_required=True",
          "future_neighbour_use=endpoint_seed_only")
    return str(destination)


# ============================================================
# RUNTIME
# ============================================================

class IntegratedScratchV4:

    def __init__(self):
        self.alive = True

        self.stage = omni.usd.get_context().get_stage()

        if self.stage is None:
            raise RuntimeError("No USD stage is open.")

        clear_old_debug(self.stage)

        self.timeline = omni.timeline.get_timeline_interface()

        # Deliberately DO NOT auto-play.
        if not self.timeline.is_playing():
            raise RuntimeError(
                "Physics is paused. Press PLAY first, put the robot in the desired "
                "upright Home, then run this V4. V4 does not auto-play old targets."
            )

        self.back_points, self.back_normals = load_back(BACK_CSV)
        self.back_tree = cKDTree(self.back_points)

        self.back_mesh_query = BackTriangleSurfaceQuery(
            self.stage,
            self.back_points,
            self.back_normals,
            self.back_tree,
        )
        self.back_alignment_summary = diagnose_back_surface_alignment(
            self.back_mesh_query,
            self.back_points,
        )

        check_finger_stop_model(self.stage)
        self.arcs = load_arcs(ARC_CSV)

        palm_local_normal = get_palm_local_normal(
            self.arcs
        )

        print(
            "[V4.12-ARC] Open-hand geometric PALM_FRAME plane normal:",
            np.round(
                palm_local_normal,
                5,
            ),
        )
        print(
            "[V4.12-ARC] J4+J5 cooperate for palm/back parallelism; J4 has a soft 0-deg aesthetic preference; J5 has no sign/angle preference."
        )

        articulation_root = find_articulation_root(self.stage)
        print("[V4] Articulation root:", articulation_root)

        self.articulation = SingleArticulation(
            prim_path=articulation_root,
            name="integrated_scratch_v4_robot",
        )

        self.articulation.initialize()

        self.arm_indices = np.array(
            [articulation_index(self.articulation, name) for name in ARM_JOINTS],
            dtype=np.int64,
        )

        self.finger_indices = {
            finger: articulation_index(self.articulation, FINGER_JOINT_NAMES[finger])
            for finger in LONG_FINGERS
        }
        self.finger_triplet_indices = {
            finger: [articulation_index(self.articulation, "right_%s_%s" % (finger, suffix))
                     for suffix in HAND_MODEL.SUFFIXES]
            for finger in LONG_FINGERS
        }

        self.arm_drives = {
            name: get_drive(self.stage, name)
            for name in ARM_JOINTS
        }

        self.finger_drives = {
            finger: get_drive(self.stage, FINGER_JOINT_NAMES[finger])
            for finger in LONG_FINGERS
        }
        self.finger_triplet_drives = {
            finger: [get_drive(self.stage, "right_%s_%s" % (finger, suffix))
                     for suffix in HAND_MODEL.SUFFIXES] for finger in LONG_FINGERS
        }

        # CRITICAL FIX:
        # Synchronize old USD drive targets to ACTUAL physics state immediately.
        actual_q = self.read_actual_arm_q()
        actual_fingers = self.read_actual_fingers_deg()

        self.hold_actual_state()

        print("[V4] Actual startup q deg:", np.round(np.degrees(actual_q), 3))

        self.home_q = self.load_or_capture_home(actual_q)

        self.solver = LulaKinematicsSolver(
            robot_description_path=DESCRIPTOR,
            urdf_path=LULA_URDF,
        )

        base_p, base_q = get_world_pose(self.stage, BASE_PATH)
        self.solver.set_robot_base_pose(base_p, base_q)
        global ROBOT_BASE_WORLD_POSITION, ROBOT_BASE_J1_AXIS_WORLD
        ROBOT_BASE_WORLD_POSITION = np.asarray(base_p, dtype=np.float64).copy()
        base_q = np.asarray(base_q, dtype=np.float64)
        base_rotation = Rotation.from_quat(base_q[[1, 2, 3, 0]]).as_matrix()
        # Transform joint1's declared local Y axis through its URDF origin
        # rotation (rpy="1.5708 0 0.073547") and then into world coordinates.
        joint1_axis_base = (
            Rotation.from_euler(
                "xyz", [1.5708, 0.0, 0.073547]
            ).as_matrix()
            @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        ROBOT_BASE_J1_AXIS_WORLD = normalize(
            base_rotation @ joint1_axis_base
        )
        print(
            "[V100 COORDINATE FRAME] robot_base_world_xyz=",
            np.round(np.asarray(base_p, dtype=np.float64), 6),
            "robot_base_world_wxyz=",
            np.round(np.asarray(base_q, dtype=np.float64), 6),
        )
        print(
            "[V100 ROOT/BASE AXIS] base_j1_axis_world=",
            np.round(ROBOT_BASE_J1_AXIS_WORLD, 6),
            "centroid_row_perpendicular_hard_deg=", ROOT_BASE_AXIS_HARD_DEG,
        )

        global HAND_COLLISION_MODEL
        collision_urdf = COLLISION_URDF if os.path.isfile(COLLISION_URDF) else LULA_URDF
        try:
            HAND_COLLISION_MODEL = URDFRobotCollisionProbeModel(collision_urdf)
            print(
                "[V6] Offline robot collision probes ready:",
                len(HAND_COLLISION_MODEL.link_probes),
                "links from",
                collision_urdf,
            )
        except Exception as exc:
            HAND_COLLISION_MODEL = None
            raise RuntimeError(
                "V6 requires offline palm/proximal/middle/distal collision probes: "
                + repr(exc)
            )

        self.lower, self.upper = self.solver.get_cspace_position_limits()
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)

        self.state = "idle"
        self.message = "Home ready - waiting for GUI target"
        self.target_id = None

        self.last_request_id = None

        if os.path.isfile(REQUEST_JSON):
            try:
                with open(REQUEST_JSON, "r", encoding="utf-8") as f:
                    old_request = json.load(f)

                self.last_request_id = old_request.get("request_id")
            except Exception:
                pass

        self.executing = False
        self.trajectory = None
        self.saved_execution = None
        self.console_prompt_mode = None
        self.console_input_buffer = ""
        self.segment_index = 0
        self.segment_elapsed = 0.0
        self.segment_arrival_wait_elapsed = 0.0
        self.segment_settle_elapsed = 0.0
        self.segment_wait_log_elapsed = 0.0
        # Persistent feedback state for the asynchronous contact wave.  These
        # must survive individual physics callbacks: a one-frame recovery
        # must not clear a lag hold and allow the shared trajectory clock to
        # run ahead of the real joints.
        self._wave_tracking_hold_key = None
        self._wave_tracking_wait_s = 0.0
        self._wave_tracking_recover_s = 0.0
        self._wave_tracking_log_s = -1.0

        self.current_mode = None
        self.current_lane_spec = None
        self.active_fingers = []
        self.current_authorized_contact_fingers = []
        self.planned_working_fingers = []
        self.current_stage = "IDLE"
        self.current_segment_alpha = 0.0
        self.current_lane_spec = None
        self.contact_j5_lift_deg = 0.0
        self.deep_adjust_latched = {
            finger: False
            for finger in LONG_FINGERS
        }
        # A repeated recoverable depth adjustment during a scratch wave is
        # recorded as data instead of aborting the simulated stroke.  The
        # key is stage/cycle/finger-scoped so a single over-depth episode does
        # not disable recovery for unrelated stages.
        self._deep_adjust_record_only_keys = set()
        self.current_finger_commands = dict(actual_fingers)

        self.request_timer = 0.0
        self.safety_timer = 0.0

        self.contact_loss_time = {
            finger: 0.0
            for finger in LONG_FINGERS
        }
        self.dynamic_depth_stats = {}
        self._actual_contact_locations = {}
        # Measured world-space marker trace.  This is deliberately kept apart
        # from the planned arc data so the post-run viewport can show what the
        # tip links really did rather than what the planner intended.
        self._actual_tip_trace = {finger: [] for finger in LONG_FINGERS}
        self._actual_tip_trace_last_sim = -1.0
        self._actual_tip_trace_error_logged = False
        self._speed_sample = None
        self.dynamic_depth_reported_cycles = set()
        self.imitation_cycle_quality = {}
        self.full_curl_joint_records = {}
        self.early_exit_joint_records = {}
        self.base_contact_stop_records = {}
        self.scratch_completed_records = {}
        self._base_contact_pairs = {}
        self._release_tail_active = False
        self.optional_scratch_fingers = {}
        self.relaxed_scratch_fingers = {}

        self.write_status("idle", "Home ready - waiting for GUI target")
        self.deformation_monitor = DEFORMATION_MONITOR.Monitor(self.stage)

    def read_actual_arm_q(self):
        q = articulation_positions(self.articulation, self.arm_indices)

        if len(q) != 5:
            raise RuntimeError("Actual arm state length is not 5.")

        return q

    def read_actual_fingers_deg(self):
        """Legacy method name; returns fitted PROGRESS PERCENT, never MCP angle."""
        result = {}
        for finger, actual in self.read_actual_finger_triplets_deg().items():
            progress, error = HAND_MODEL.progress_from_triplet(actual)
            if error > 3.0:
                raise RuntimeError("FINGER_OFF_STAGED_CURVE_OPEN_HAND_FIRST_%s_error_deg=%.2f" % (finger, error))
            result[finger] = progress
        return result

    def hold_actual_state(self):
        """Hold the measured 12-DOF pose, without projecting onto a curl curve."""
        actual_q = self.read_actual_arm_q()
        triplets = self.read_actual_finger_triplets_deg()
        for i, name in enumerate(ARM_JOINTS):
            set_drive_deg(self.arm_drives[name], float(np.degrees(actual_q[i])))
        for finger, values in triplets.items():
            for drive, value, limit in zip(self.finger_triplet_drives[finger], values, HAND_MODEL.LIMITS_DEG):
                set_drive_deg(drive, clamp(float(value), 0.0, limit))

    def read_actual_finger_triplets_deg(self):
        values = {
            finger: np.degrees(articulation_positions(self.articulation, indices))
            for finger, indices in self.finger_triplet_indices.items()
        }
        if any(len(v) != 3 or not np.all(np.isfinite(v)) for v in values.values()):
            raise RuntimeError("INVALID_TWELVE_FINGER_JOINT_FEEDBACK")
        if any(np.any(v < -1.0) or np.any(v > np.asarray(HAND_MODEL.LIMITS_DEG) + 1.0)
               for v in values.values()):
            raise RuntimeError("ACTUAL_FINGER_JOINT_LIMIT_EXCEEDED_" + str(values))
        return values

    def load_or_capture_home(self, actual_q):
        """
        V4.10 session-home policy.

        The ACTUAL articulation posture at runtime startup is always the Home
        for this session.

        This prevents an old scratch_home_q.json -- especially the historical
        center scratch pose -- from ever being used by RETURN_HOME.

        The JSON file is still refreshed for diagnostics/inspection, but it is
        never trusted over the current physical Isaac state.
        """
        actual_q = np.asarray(
            actual_q,
            dtype=np.float64,
        )

        if actual_q.shape != (5,):
            raise RuntimeError(
                "Actual startup Home q is invalid."
            )

        previous_q = None

        if os.path.isfile(
            HOME_JSON
        ):
            try:
                with open(
                    HOME_JSON,
                    "r",
                    encoding="utf-8",
                ) as f:
                    previous_data = json.load(
                        f
                    )

                previous_q = np.asarray(
                    previous_data.get(
                        "q_rad",
                        [],
                    ),
                    dtype=np.float64,
                )

                if previous_q.shape != (5,):
                    previous_q = None

            except Exception:
                previous_q = None

        if previous_q is not None:
            delta_deg = np.degrees(
                actual_q
                -
                previous_q
            )

            max_delta_deg = float(
                np.max(
                    np.abs(
                        delta_deg
                    )
                )
            )

            if max_delta_deg > 5.0:
                print(
                    "[V4.12-ARC] STALE HOME IGNORED. Old JSON q deg:",
                    np.round(
                        np.degrees(
                            previous_q
                        ),
                        3,
                    ),
                )

                print(
                    "[V4.12-ARC] Using ACTUAL startup q as session Home:",
                    np.round(
                        np.degrees(
                            actual_q
                        ),
                        3,
                    ),
                )

        data = {
            "joint_names":
                ARM_JOINTS,

            "q_rad":
                [
                    float(
                        value
                    )
                    for value
                    in actual_q
                ],

            "q_deg":
                [
                    float(
                        value
                    )
                    for value
                    in np.degrees(
                        actual_q
                    )
                ],

            "timestamp":
                time.time(),

            "note":
                (
                    "V4.12-ARC4 session Home. "
                    "Captured from ACTUAL Isaac articulation state at startup. "
                    "Historical center Home is never reused."
                ),
        }

        safe_write_json(
            HOME_JSON,
            data,
        )

        print(
            "[V4.12-ARC] Session Home q deg:",
            np.round(
                np.degrees(
                    actual_q
                ),
                3,
            ),
        )

        return actual_q.copy()


    def write_status(self, state, message):
        state = str(state)
        message = str(message)

        changed = state != self.state or message != self.message

        self.state = state
        self.message = message

        if changed:
            print("[V4]", state.upper(), "-", message)

        safe_write_json(
            STATUS_JSON,
            {
                "state": self.state,
                "message": self.message,
                "target_id": self.target_id,
                "timestamp": time.time(),
            },
        )

    def command(self, q, fingers):
        # Check all fingers before writing ANY drive, including the arm.
        fingers = checked_finger_commands(fingers)
        q = np.asarray(q, dtype=np.float64)
        q_deg = np.degrees(q)

        for i, name in enumerate(ARM_JOINTS):
            set_drive_deg(self.arm_drives[name], q_deg[i])

        for finger in LONG_FINGERS:
            targets = HAND_MODEL.joint_targets(fingers[finger])
            for drive, angle in zip(self.finger_triplet_drives[finger], targets):
                set_drive_deg(drive, angle)

        self.current_finger_commands = dict(fingers)
        self.current_arm_command = q.copy()

    def check_request(self):
        if not os.path.isfile(REQUEST_JSON):
            return

        try:
            with open(REQUEST_JSON, "r", encoding="utf-8") as f:
                request = json.load(f)
        except Exception:
            return

        request_id = request.get("request_id")

        if request_id is None or request_id == self.last_request_id:
            return

        if self.executing or self.console_prompt_mode is not None:
            self.write_status(
                "busy",
                "Current scratch cycle is running or waiting for terminal confirmation",
            )
            return

        self.last_request_id = request_id
        self.handle_target(request)

    def handle_target(self, request):
        planning_started_at = time.perf_counter()
        try:
            self.write_status("planning", "Planning selected GUI target")

            candidate_id = int(request["candidate_id"])

            if not 0 <= candidate_id < len(self.back_points):
                raise RuntimeError("INVALID_TARGET_ID")

            self.target_id = candidate_id
            target = self.back_points[candidate_id]

            request_xyz = request.get("target_xyz_m")
            if isinstance(request_xyz, dict):
                request_point = np.asarray(
                    [
                        float(request_xyz["x"]),
                        float(request_xyz["y"]),
                        float(request_xyz["z"]),
                    ],
                    dtype=np.float64,
                )
                request_delta = float(np.linalg.norm(request_point - target))
                print(
                    "[V100 GUI TARGET ID/XYZ CHECK]",
                    "candidate=", candidate_id,
                    "delta_mm=", round(request_delta * 1000.0, 6),
                )
                if request_delta > REQUEST_TARGET_XYZ_TOL_M:
                    raise RuntimeError(
                        "GUI_CSV_TARGET_MISMATCH_delta_mm="
                        + str(round(request_delta * 1000.0, 3))
                    )

            if bool(getattr(self.back_mesh_query, "available", False)):
                target_mesh_distance = float(self.back_mesh_query.query(target)[0])
                print(
                    "[V100 TARGET BACKSURFACE ALIGNMENT]",
                    "candidate=", candidate_id,
                    "csv_to_mesh_mm=", round(target_mesh_distance * 1000.0, 3),
                )
                if target_mesh_distance > TARGET_BACK_ALIGNMENT_HARD_M:
                    raise RuntimeError(
                        "TARGET_POINTCLOUD_BACKSURFACE_MISMATCH_mm="
                        + str(round(target_mesh_distance * 1000.0, 3))
                    )

            patch_ids = np.asarray(
                self.back_tree.query_ball_point(target, PATCH_RADIUS_M),
                dtype=np.int64,
            )

            if len(patch_ids) < 20:
                raise RuntimeError("TARGET_PATCH_TOO_SMALL")

            patch_points = self.back_points[patch_ids]
            patch_normals = self.back_normals[patch_ids]

            modes, spread, separation, is_v = analyse_patch(
                patch_points,
                patch_normals,
                target,
                self.back_tree,
                self.back_points,
                self.back_normals,
            )

            print("")
            print("==============================================")
            print("NEW GUI SCRATCH TARGET")
            print("==============================================")
            print("Candidate:", candidate_id)
            print("XYZ:", np.round(target, 5))
            print("Patch points:", len(patch_ids))
            print("Normal spread P90:", round(spread, 2))
            print("Normal separation:", round(separation, 2))
            print("V candidate:", is_v)

            current_q = self.read_actual_arm_q()
            current_fingers = self.read_actual_fingers_deg()
            print(
                "[V102 EXECUTION START STATE] fitted_finger_progress_pct=",
                {finger: round(float(current_fingers[finger]), 3) for finger in LONG_FINGERS},
            )

            print("Actual current q deg:", np.round(np.degrees(current_q), 2))

            # Exact cache reuse is fail-closed. Provenance, complete surface,
            # target and measured start state must match, after which the
            # current complete validator runs again. Cached PASS is never
            # trusted and this block never starts motion by itself.
            exact, exact_reason = V104_CACHE.load_exact(
                PROJECT_DIR, candidate_id=candidate_id, target_xyz=target,
                target_normal=self.back_normals[candidate_id],
                back_points=self.back_points, back_normals=self.back_normals,
                actual_q=current_q, actual_fingers=current_fingers,
            )
            print("[V104 EXACT CACHE]", exact_reason)
            if exact is not None:
                cached_execution = exact["execution"]
                cached_trajectory = cached_execution["trajectory"]
                cached_mode = cached_execution["mode"]
                lifecycle_ok, lifecycle_reason = validate_contact_lifecycle(cached_trajectory)
                cache_safe, cache_reason = False, lifecycle_reason
                if lifecycle_ok:
                    cache_cap = float(cached_mode.get("palm_alignment_limit_deg", 90.0))
                    with PALM_POLICY.scope(cache_cap):
                        cache_safe, cache_reason = validate_trajectory(
                            self.solver, self.arcs, self.back_points, self.back_normals,
                            self.back_tree, self.back_mesh_query, cached_trajectory,
                            self.lower, self.upper, cached_mode,
                            lane_spec=cached_execution.get("lane_spec"),
                        )
                print("[V104 EXACT CACHE REVALIDATION]", "PASS" if cache_safe else cache_reason,
                      "cached_acceptance_trusted=False", "motion_started=False")
                if cache_safe:
                    save_trajectory(cached_trajectory)
                    safe_write_json(PALM_TRIAL_JSON, {
                        "build": V104_CACHE.BUILD, "candidate_id": candidate_id,
                        "target_xyz_m": np.asarray(target).tolist(),
                        "planning_seconds": round(time.perf_counter()-planning_started_at, 3),
                        "exact_cache": True, "full_revalidation": True,
                        "motion_executed": False,
                    })
                    self.queue_execution_confirmation(
                        cached_trajectory, cached_mode,
                        planned_working_fingers=cached_execution.get("planned_working_fingers"),
                        lane_spec=cached_execution.get("lane_spec"),
                        endpoint_q=cached_execution.get("endpoint_q"),
                        candidate_id=candidate_id, target_xyz=target,
                        target_normal=self.back_normals[candidate_id],
                    )
                    return

            neighbour_rows = V104_CACHE.neighbour_endpoint_seeds(
                PROJECT_DIR, V104_DATA_ROOT, target, self.back_normals[candidate_id],
            )
            print("[V104 NEIGHBOUR SEEDS]", [
                {"distance_mm": round(row["distance_m"]*1000.0, 2),
                 "normal_deg": round(row["normal_angle_deg"], 2),
                 "source": row["source"]} for row in neighbour_rows
            ], "hints_only=True", "full_search_fallback=True")

            # ============================================================
            # V103 POLICY: FOUR FINGERS ONLY; PINKY LOCAL-OPTIMUM
            # ============================================================
            # The GUI target is always evaluated as the clicked TARGET_PATCH.
            # Four fingers are searched once.  A pinky depth miss is handled in
            # the same four-finger contact rollout as a bounded local optimum;
            # no fresh index/middle/ring (3F) search is launched.
            target_mode = next(
                (
                    mode
                    for mode in modes
                    if mode.get("kind") == "flat"
                    and mode.get("name") == "TARGET_PATCH"
                ),
                None,
            )

            if target_mode is None:
                raise RuntimeError("FOUR_FINGER_ONLY_TARGET_PATCH_MISSING")

            print("")
            print("==============================================")
            print("MODE: TARGET_PATCH / FOUR-FINGER ONLY")
            print("==============================================")
            print(
                "[V103 FINGER POLICY] HUMAN_WAVE 4F only. "
                "index/middle/ring each require live 1-4 mm with 3 mm preferred; "
                "pinky stays in the four-finger wave and alone may finish at a bounded local optimum; "
                "3F/2F/1F disabled."
            )

            search_active_fingers = tuple(LONG_FINGERS)
            # Legacy `search_active_fingers = ('ring', 'middle', 'index')`
            # fallback is intentionally not assigned in live V103.
            # Legacy `search_anchor_finger = 'middle'` fallback is likewise disabled.
            # Legacy active_count=3 replan is likewise disabled.
            search_anchor_finger = None
            search_stats_by_mode = {}
            priority_review_only = False

            def search_tier(cap, previous_candidates):
                tier_mode = dict(target_mode, palm_alignment_limit_deg=cap,
                                 v103_contact_coordination=True)
                tier_mode['v103_near_base_three_finger'] = False
                if search_anchor_finger is not None:
                    tier_mode['v103_anchor_finger'] = search_anchor_finger
                # Cache only successful q warm starts, never acceptance flags or
                # old trajectories. Recompute minima, geometry and swept path.
                persistent = [(np.asarray(target).copy(), row['q']) for row in neighbour_rows]
                cached = sorted(list(getattr(self, '_v103_success_seeds', [])) + persistent,
                                key=lambda row: np.linalg.norm(row[0]-target))
                warm_rows = cached[:4]
                historical_qs = [
                    np.asarray(row[1], dtype=np.float64)
                    for row in getattr(self, '_v103_success_seeds', [])
                ]
                historical_qs.extend(
                    np.asarray(row['q'], dtype=np.float64)
                    for row in neighbour_rows
                    if 'historic' in str(row.get('source', '')).lower()
                    and np.asarray(row.get('q')).shape == (5,)
                )
                print('[V103 WARM SEEDS]', [
                    {'distance_mm': round(float(np.linalg.norm(row[0]-target))*1000.0, 3),
                     'q_deg': np.round(np.degrees(row[1]), 3).tolist()}
                    for row in warm_rows
                ], 'priority_replay=True')
                candidates, stats = plan_fk_first(
                    self.solver, self.arcs, tier_mode, current_q, self.home_q,
                    self.lower, self.upper, [row[1] for row in warm_rows],
                    active_fingers=search_active_fingers,
                    anchor_finger=search_anchor_finger,
                    priority_only=priority_review_only,
                )
                # Keep the exact historical branch identifiable after endpoint
                # search.  It still receives full trajectory/swept validation;
                # this flag only enables the narrowly scoped replay preposition
                # fallback when the current exploratory corridor is empty.
                for candidate in candidates:
                    q_candidate = np.asarray(candidate.get('q'), dtype=np.float64)
                    if q_candidate.shape == (5,) and any(
                        np.linalg.norm(np.degrees(q_candidate - q_old)) <= 0.02
                        for q_old in historical_qs
                    ):
                        candidate['_historical_replay'] = True
                        print('[V103 HISTORICAL REPLAY CANDIDATE]',
                              'q_deg=', np.round(np.degrees(q_candidate), 3),
                              'full_path_validation=True')
                self._v103_search_stats = stats
                search_stats_by_mode['+'.join(search_active_fingers)] = stats
                print('[V103 ENDPOINT REJECT SUMMARY]', endpoint_failure_summary(candidates))
                # The approximate preview skipped acquisition repairs and
                # discarded all but ONE of its failures. It could therefore
                # erase usable alternatives while adding duplicate solve time.
                # Each retained strict endpoint now gets the same real-mesh
                # acquisition/wave/retreat validation, stopping at first success.
                stats['contact_preview_seconds'] = 0.
                print('[V103 PATH POLICY] exact acquisition + full swept path per candidate;',
                      'strict_retained=', sum(bool(c.get('valid')) for c in candidates),
                      'provisional_retained=', sum(bool(c.get('path_validation_eligible')) and not bool(c.get('valid')) for c in candidates),
                      'approximate_preview_skipped=True')
                return candidates

            def validate_tier_candidate(candidate):
                print("[V103 TRAJECTORY TEST]", candidate["template"]["name"],
                      "limit_deg=", palm_alignment_limit_deg(),
                      "endpoint_q_deg=", np.round(np.degrees(candidate["q"]), 2),
                      "endpoint_base_gap_mm=", candidate.get('endpoint_base_clearance_mm'),
                      "endpoint_base_link=", candidate.get('endpoint_base_closest_link'))
                ok, path, reason = try_candidate_trajectory(
                    self.solver, self.arcs, candidate, current_q, self.home_q,
                    self.lower, self.upper, self.back_points, self.back_normals,
                    self.back_tree, self.back_mesh_query, current_fingers=current_fingers,
                    historical_replay=bool(candidate.get('_historical_replay', False)),
                )
                return ok, path, reason

            base_delta = np.asarray(target) - ROBOT_BASE_WORLD_POSITION
            axis = normalize(np.asarray(ROBOT_BASE_J1_AXIS_WORLD, dtype=np.float64))
            radial_distance = float(np.linalg.norm(base_delta - np.dot(base_delta, axis) * axis))
            print("[V103 BASE TARGET] radial_distance_to_J1_axis_mm=",
                  round(radial_distance * 1000.0, 2),
                  "palm tilt cap=90 diagnostic only; no 60-deg hard tier; not a reachability certificate")
            # 4F is always searched first.  The old target-765 shortcut started
            # directly in 3F mode and therefore bypassed the policy below; it
            # is intentionally removed.  Warm seeds may still be used by the
            # ordinary 4F search, but never change the required finger count.
            result = None
            # The old 60/75/90 escalation made 60 degrees a de-facto hard
            # gate: a pose just over 60 degrees could be selected at the
            # first tier and then stopped at runtime, even when it was
            # collision-free and depth-valid.  Production V103 now searches
            # the full 90-degree workspace in one pass.  Palm angle remains
            # a ranking/diagnostic value; collision, joint limits, target
            # lane and release safety are the authoritative hard gates.
            result = PALM_POLICY.select_plan(
                search_tier, validate_tier_candidate, emit=print,
                tiers=(90.0,), active_count=4,
                max_valid_candidates=MAX_ENDPOINT_VALID_POOL)
            print('[V103 4F ONLY DECISION]',
                  'result=', '4F_SUCCEEDED' if result.get('best') is not None else 'NO_SAFE_4F',
                  'pinky=local_optimum_in_contact_rollout',
                  'three_finger_replan=DISABLED',
                  'radial_mm=', round(radial_distance*1000., 2),
                  'radial_gate=DISABLED')
            best, trajectory = result["best"], result["trajectory"]
            all_candidates = result["candidates"]
            trajectory_diagnostics = trajectory_failure_summary(result)
            trial_report = {
                "build": CONTACT_WAVE.BUILD,
                "endpoint_search": search_stats_by_mode,
                "endpoint_failure_counts": endpoint_failure_summary(all_candidates),
                "trajectory_failure_summary": trajectory_diagnostics,
                "candidate_id": candidate_id,
                "target_xyz_m": np.asarray(target).tolist(),
                "radial_distance_to_J1_axis_mm": round(radial_distance * 1000.0, 3),
                "attempts": result["attempts"],
                "selected_limit_deg": None if best is None else best["palm_alignment_limit_deg"],
                "selected_palm_deg": None if best is None else best["geometry"]["palm_back_angle_deg"],
                "planning_seconds": round(time.perf_counter() - planning_started_at, 3),
                "motion_executed": False,
                "scope": "planning-only; swept collision and contact-quality gates unchanged",
            }
            safe_write_json(PALM_TRIAL_JSON, trial_report)
            if best is None:
                endpoint_counts = endpoint_failure_summary(all_candidates)
                print('[V103 FAILURE SUMMARY]',
                      'endpoint_candidates=', len(all_candidates),
                      'endpoint_failure_counts=', endpoint_counts,
                      'trajectory_rejected=', trajectory_diagnostics['total_rejected_candidates'],
                      'trajectory_failure_counts=', trajectory_diagnostics['counts'])
                for bucket, count in trajectory_diagnostics['counts']:
                    example = trajectory_diagnostics['examples'].get(bucket, {})
                    print('[V103 FAILURE CAUSE]',
                          'count=', count,
                          'cause=', bucket,
                          'template=', example.get('template'),
                          'example=', example.get('reason'))
                for count in (4,):
                    group = [c for c in all_candidates if len(c['template']['active']) == count]
                    if not group:
                        continue
                    tested_group = [c for c in group if c.get('trajectory_reason')]
                    representative = min(tested_group or group,key=candidate_rank)
                    detail = representative['geometry'].get('lane_details',{}).get('middle',{})
                    print('[V103 FAILURE BY FINGER MODE]', 'active_count=',count,
                          'middle_error_mm=',float(detail.get('patch_radius',float('nan')))*1000.,
                          'endpoint_reasons=',representative['geometry'].get('reject'),
                          'path_reasons=',representative.get('trajectory_reason'),
                          'representative_only=True')
                    print_candidate_frame_diagnostic(representative,representative['mode'])
                if all_candidates:
                    tested = [c for c in all_candidates if c.get('trajectory_reason')]
                    # Explain the actual path failure when a pose was found,
                    # rather than printing an unrelated rejected 4F endpoint.
                    diagnostic = min(tested or all_candidates, key=candidate_rank)
                    print('[V103 BEST SEARCHED FAILURE] representative only; not a reachability proof')
                    print_candidate_frame_diagnostic(diagnostic, diagnostic['mode'])
                    print('[V103 FAILURE Q DEG]', np.round(np.degrees(diagnostic['q']), 3),
                          'endpoint_reasons=', diagnostic['geometry'].get('reject'),
                          'path_reasons=', diagnostic.get('trajectory_reason'),
                          'contact_preview=', diagnostic.get('contact_preview'))
                raise RuntimeError("V103_NO_SAFE_FOUR_FINGER_TRAJECTORY_WITHIN_90_DEG_SEARCH_BUDGET")
            cache = list(getattr(self, '_v103_success_seeds', []))
            cache.append((np.asarray(target).copy(), np.asarray(best['q']).copy()))
            self._v103_success_seeds = cache[-16:]
            try:
                draw_v100_candidate_debug(
                    self.stage, self.solver, self.arcs, best, best["mode"],
                )
            except Exception as debug_exc:
                print("[V103 VISUAL DEBUG ERROR]", repr(debug_exc))

            g = (
                best[
                    "geometry"
                ]
            )

            print("")
            print("==============================================")
            print("SELECTED EXECUTABLE STRATEGY")
            print("==============================================")
            print(
                "Mode:",
                best[
                    "mode"
                ][
                    "name"
                ],
            )

            print(
                "Template:",
                best[
                    "template"
                ][
                    "name"
                ],
            )

            print(
                "Sequence:",
                best[
                    "template"
                ][
                    "sequence"
                ],
            )

            print(
                "Phases:",
                best[
                    "template"
                ][
                    "phases"
                ],
            )

            print(
                "Solver:",
                best[
                    "ik_kind"
                ],
            )

            print(
                "Endpoint q deg:",
                np.round(
                    np.degrees(
                        best[
                            "q"
                        ]
                    ),
                    2,
                ),
            )

            print(
                "Endpoint J4:",
                round(
                    float(
                        np.degrees(
                            best[
                                "q"
                            ]
                        )[3]
                    ),
                    2,
                ),
                "deg (soft target 0 deg)",
            )

            print(
                "Endpoint J5:",
                round(
                    float(
                        np.degrees(
                            best[
                                "q"
                            ]
                        )[4]
                    ),
                    2,
                ),
                "deg (no angle target)",
            )

            print(
                "Inactive clearance:",
                best[
                    "clearance_deg"
                ],
                "deg",
            )

            print(
                "Coverage:",
                round(
                    g[
                        "coverage"
                    ]
                    *
                    1000.0,
                    2,
                ),
                "mm",
            )

            print(
                "Target center error:",
                round(float(g.get("target_center_error", 999.0)) * 1000.0, 2),
                "mm",
                "Lane cross max:",
                round(float(g.get("lane_cross_max", 999.0)) * 1000.0, 2),
                "mm",
                "Patch radius max:",
                round(float(g.get("patch_radius_max", 999.0)) * 1000.0, 2),
                "mm",
            )

            print(
                "Normal RMSE:",
                round(
                    g[
                        "normal_rmse"
                    ]
                    *
                    1000.0,
                    2,
                ),
                "mm",
            )


            print(
                "Palm/back plane angle:",
                round(
                    float(
                        g.get(
                            "palm_back_angle_deg",
                            90.0,
                        )
                    ),
                    2,
                ),
                "deg",
            )

            print(
                "Scratch stroke:",
                round(
                    g[
                        "stroke"
                    ]
                    *
                    1000.0,
                    2,
                ),
                "mm",
            )

            print(
                "Trajectory safety: TRUE"
            )

            save_plan(
                candidate_id,
                best,
            )

            save_trajectory(
                trajectory
            )

            print(
                "[V103 PLANNING TIME] seconds=",
                round(time.perf_counter() - planning_started_at, 3),
                "direct_endpoint_ik=True",
                "handoff_optimization=False",
                "per_step_ik=False",
            )

            self.queue_execution_confirmation(
                trajectory,
                best[
                    "mode"
                ],
                planned_working_fingers=best["template"]["active"],
                lane_spec=best.get("runtime_lane_spec"),
                endpoint_q=best["q"], candidate_id=candidate_id,
                target_xyz=target, target_normal=self.back_normals[candidate_id],
            )

        except Exception as exc:
            self.executing = False
            self.write_status("error", str(exc))
            print("[V4 ERROR]", repr(exc))

    def queue_execution_confirmation(
        self,
        trajectory,
        mode,
        planned_working_fingers=None,
        lane_spec=None,
        endpoint_q=None,
        candidate_id=None,
        target_xyz=None,
        target_normal=None,
    ):
        self.saved_execution = {
            "trajectory": trajectory,
            "mode": mode,
            "planned_working_fingers": list(planned_working_fingers or []),
            "lane_spec": lane_spec,
            "endpoint_q": None if endpoint_q is None else np.asarray(endpoint_q).copy(),
            "candidate_id": candidate_id,
            "target_xyz": None if target_xyz is None else np.asarray(target_xyz).copy(),
            "target_normal": None if target_normal is None else np.asarray(target_normal).copy(),
        }
        self.begin_console_prompt("initial")

    def begin_console_prompt(self, mode):
        self.console_prompt_mode = str(mode)
        self.console_input_buffer = ""
        print("")
        if self.console_prompt_mode == "initial":
            self.write_status(
                "awaiting_confirmation",
                "Plan ready - enter y in the terminal to start robot motion",
            )
            print(
                "[V103 MOTION CONFIRM] Enter y to start robot motion "
                "(n cancels): ",
                end="",
                flush=True,
            )
        else:
            self.write_status(
                "awaiting_repeat",
                "Cycle complete - enter n to stop repeating; any other input repeats",
            )
            print(
                "[V103 REPEAT] Enter n to stop repeating; "
                "any other input runs again: ",
                end="",
                flush=True,
            )

    def poll_console_prompt(self):
        """Poll the Windows terminal without blocking Isaac's physics thread."""
        if self.console_prompt_mode is None:
            return

        try:
            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                if char == "\b":
                    if self.console_input_buffer:
                        self.console_input_buffer = self.console_input_buffer[:-1]
                        print("\b \b", end="", flush=True)
                    continue
                if char not in ("\r", "\n"):
                    self.console_input_buffer += char
                    print(char, end="", flush=True)
                    continue

                print("")
                answer = self.console_input_buffer.strip().lower()
                prompt_mode = self.console_prompt_mode
                self.console_prompt_mode = None
                self.console_input_buffer = ""

                if prompt_mode == "initial":
                    if answer == "y":
                        self.start_saved_execution()
                    elif answer == "n":
                        self.saved_execution = None
                        self.write_status(
                            "idle",
                            "Motion cancelled - waiting for next GUI target",
                        )
                        print("[V102] Motion cancelled; waiting for next GUI target.")
                    else:
                        print("[V102] Initial motion requires an explicit y.")
                        self.begin_console_prompt("initial")
                else:
                    if answer == "n":
                        self.saved_execution = None
                        self.write_status(
                            "idle",
                            "Repeat stopped - waiting for next GUI target",
                        )
                        print("[V102] Repeat stopped; waiting for next GUI target.")
                    else:
                        self.start_saved_execution()
                break
        except Exception as exc:
            self.console_prompt_mode = None
            self.saved_execution = None
            self.write_status("error", "TERMINAL_INPUT_ERROR: " + str(exc))
            print("[V102 TERMINAL INPUT ERROR]", repr(exc))

    def _fitted_actual_finger_progress(self, allow_off_curve=False):
        """Return measured finger progress, optionally projecting a stopped hand.

        A safety stop can leave PIP/DIP a few degrees away from the staged curl
        curve.  Normal execution must reject that state.  Failure recovery is
        the sole exception: it projects the measured triplets onto the nearest
        staged progress value so a mesh-checked unload can be generated.
        """
        result = {}
        errors = {}
        for finger, actual in self.read_actual_finger_triplets_deg().items():
            progress, error = HAND_MODEL.progress_from_triplet(actual)
            errors[finger] = float(error)
            if error > 3.0 and not allow_off_curve:
                raise RuntimeError(
                    "FINGER_OFF_STAGED_CURVE_OPEN_HAND_FIRST_%s_error_deg=%.2f"
                    % (finger, error)
                )
            # A larger mismatch means feedback/model integrity is unknown;
            # do not turn an arbitrary stopped pose into a recovery command.
            if error > 8.0:
                raise RuntimeError(
                    "FAILURE_RECOVERY_FINGER_PROJECTION_TOO_LARGE_%s_error_deg=%.2f"
                    % (finger, error)
                )
            result[finger] = clamp(float(progress), 0.0, 100.0)
        return result, errors

    def start_failure_recovery(self):
        """Mesh-check an unload/open/retreat/Home route from measured state.

        This is intentionally separate from emergency_stop(): the stop first
        freezes the measured pose.  The batch bridge then calls this method
        while its authorization lease is still live and waits for HOME_WAIT
        before releasing the next queue item.
        """
        if self.executing:
            raise RuntimeError("FAILURE_RECOVERY_REQUIRES_STOPPED_RUNTIME")

        actual_q = np.asarray(self.read_actual_arm_q(), dtype=np.float64)
        actual_triplets = self.read_actual_finger_triplets_deg()
        current_fingers, projection_errors = self._fitted_actual_finger_progress(
            allow_off_curve=True
        )
        arm_home_error_deg = float(
            np.max(np.abs(np.degrees(actual_q - self.home_q)))
        )
        finger_open_error_deg = float(
            max(np.max(np.abs(values)) for values in actual_triplets.values())
        )
        print(
            "[V103 FAILURE RECOVERY START]",
            "arm_home_error_deg=", round(arm_home_error_deg, 3),
            "finger_open_error_deg=", round(finger_open_error_deg, 3),
            "projection_error_deg=",
            {finger: round(error, 3) for finger, error in projection_errors.items()},
        )

        # Planning failures normally happen before any command was issued.
        # Confirm that fact from measured feedback and complete immediately.
        if arm_home_error_deg <= 1.0 and finger_open_error_deg <= 2.0:
            self.command(self.home_q, empty_fingers())
            self.current_stage = "IDLE"
            self.write_status("idle", "Failure recovery confirmed at Home")
            print("[V103 FAILURE RECOVERY HOME CONFIRMED] no motion required")
            self.begin_console_prompt("repeat")
            return False

        mode = self.current_mode
        if not isinstance(mode, dict) or "normal" not in mode:
            raise RuntimeError("FAILURE_RECOVERY_HAS_NO_EXECUTED_TARGET_GEOMETRY")

        trajectory = []
        add_keyframe(
            trajectory,
            "FAILURE_RECOVERY_MEASURED_HOLD",
            0.05,
            actual_q,
            current_fingers,
            list(LONG_FINGERS),
            authorized_contact_fingers=set(LONG_FINGERS),
        )

        q_now, fingers, _ = append_adaptive_unload_and_open(
            trajectory,
            self.solver,
            self.arcs,
            mode,
            actual_q,
            current_fingers,
            list(LONG_FINGERS),
            self.home_q,
            self.lower,
            self.upper,
            self.back_points,
            self.back_normals,
            self.back_tree,
            self.back_mesh_query,
            0,
        )

        def check_retreat_segment(a, b):
            return segment_mesh_safe(
                self.solver, self.arcs, a, fingers, b, fingers,
                self.back_points, self.back_normals, self.back_tree,
                self.back_mesh_query, samples=96,
                allowed_contact_fingers=set(),
            )

        def local_lift(a):
            p, r = self.solver.compute_forward_kinematics(PALM_FRAME, a)
            return solve_offset_pose(
                self.solver, p, r, mode["normal"],
                SAFE_RETREAT_OFFSET_M / 10.0,
                a, self.home_q, self.lower, self.upper,
            )

        return_route, retreat_method = v103_plan_retreat_route(
            q_now, [], self.home_q, self.lower, self.upper,
            check_retreat_segment, local_lift, prefer_direct_home=True,
        )
        print(
            "[V103 FAILURE RECOVERY ROUTE PASS]",
            "method=", retreat_method,
            "segments=", len(return_route),
            "open_fingers_fixed=True",
            "direct_home_preferred=True",
        )
        for index, retreat_q in enumerate(return_route):
            is_home = index == len(return_route) - 1
            add_keyframe(
                trajectory,
                "RETURN_HOME" if is_home else "SAFE_RETREAT",
                arm_move_duration(
                    q_now, retreat_q,
                    RETURN_HOME_MIN_TIME_S if is_home else 0.30,
                ),
                retreat_q,
                fingers,
                [],
                authorized_contact_fingers=[],
            )
            q_now = np.asarray(retreat_q, dtype=np.float64)
        add_keyframe(
            trajectory, "HOME_WAIT", 0.20, self.home_q,
            fingers, [], authorized_contact_fingers=[],
        )

        self.start_execution(
            trajectory,
            mode,
            planned_working_fingers=[],
            lane_spec=None,
            allow_off_curve_start=True,
        )
        self.write_status(
            "unloading",
            "Failure recovery: mesh-checked unload, open, retreat and Home",
        )
        return True

    def start_saved_execution(self):
        saved = self.saved_execution
        if saved is None:
            self.write_status("error", "No saved trajectory is available")
            return
        try:
            self.start_execution(
                saved["trajectory"],
                saved["mode"],
                planned_working_fingers=saved["planned_working_fingers"],
                lane_spec=saved["lane_spec"],
            )
        except Exception as exc:
            self.saved_execution = None
            self.executing = False
            self.write_status("error", str(exc))
            print("[V102 EXECUTION START ERROR]", repr(exc))

    def start_execution(self, trajectory, mode, planned_working_fingers=None,
                        lane_spec=None, allow_off_curve_start=False):
        for frame in trajectory:
            if str(frame.get('stage', '')).startswith('SCRATCH_WAVE_ASYNC_'):
                # Four working fingers are mandatory.  The hand command
                # dictionary may also contain the non-working thumb hold
                # channel; that does not turn a valid 4F scratch into a 5F
                # trajectory.  Do not use exact key equality here.
                command_channels = set(frame.get('fingers', {}))
                expected = set(LONG_FINGERS)
                optional = set(frame.get('scratch_optional_fingers', ()))
                relaxed = set(frame.get('scratch_relaxed_fingers', ()))
                if (not expected.issubset(command_channels)
                        or not command_channels.issubset(expected | {'thumb'})
                        or set(frame.get('scratch_required_fingers', LONG_FINGERS)) != expected
                        or not optional.issubset(expected)
                        or not relaxed.issubset(expected)):
                    raise RuntimeError('EXECUTION_REQUIRES_FOUR_FINGER_TRAJECTORY')
        self._depth_recovery = None
        self._depth_recovery_count = 0
        self._deep_adjust_record_only_keys = set()
        self._wave_recovery_attempt_key = None
        check_finger_stop_model(self.stage)
        # Re-read the articulation immediately before the first command. Never
        # let a synthetic/planning-time finger state create a Play-time jump.
        actual_q = self.read_actual_arm_q()
        if allow_off_curve_start:
            actual_fingers, projection_errors = self._fitted_actual_finger_progress(
                allow_off_curve=True
            )
            print(
                "[V103 FAILURE RECOVERY PROJECTION]",
                {finger: round(error, 3)
                 for finger, error in projection_errors.items()},
            )
        else:
            actual_fingers = self.read_actual_fingers_deg()
        planned_q0 = np.asarray(trajectory[0]["q"], dtype=np.float64)
        planned_f0 = dict(trajectory[0]["fingers"])
        arm_start_delta_deg = float(
            np.max(np.abs(np.degrees(actual_q - planned_q0)))
        )
        finger_start_delta_deg = float(
            max(
                abs(float(actual_fingers[f]) - float(planned_f0.get(f, 0.0)))
                for f in LONG_FINGERS
            )
        )
        if arm_start_delta_deg > 5.0 or finger_start_delta_deg > 5.0:
            raise RuntimeError(
                "EXECUTION_START_STATE_CHANGED_REPLAN_REQUIRED"
                + "_arm_deg=" + str(round(arm_start_delta_deg, 2))
                + "_finger_deg=" + str(round(finger_start_delta_deg, 2))
            )
        trajectory[0]["q"] = np.asarray(actual_q, dtype=np.float64).copy()
        trajectory[0]["fingers"] = dict(actual_fingers)
        print(
            "[V102 NO-JUMP START] fitted_finger_progress_pct=",
            {finger: round(float(actual_fingers[finger]), 3) for finger in LONG_FINGERS},
            "max_planned_delta_deg=",
            round(finger_start_delta_deg, 3),
        )

        self.trajectory = trajectory
        self.contact_diag_elapsed_s = 0.0
        self.contact_diag_counts = {root: 0 for root in BACK_CONTACT_ROOTS}
        self.contact_diag_latest = {}
        self.contact_diag_last_log = {}
        self.stall_diag_logged = set()
        self.contact_diag_error_logged = False
        print("[V102 CONTACT DIAGNOSTICS V1] roots=", BACK_CONTACT_ROOTS,
              "subscription=", "ACTIVE" if self.contact_subscription is not None else "UNAVAILABLE",
              "delivery=UNVERIFIED; no events does NOT mean no collision")
        self.report_contact_sources()
        self.contact_lift_stall_state = None
        print("[V102 J5 STALL ASSIST V2] trigger=arrival_tolerance + measured_stall",
              "step_deg=", CONTACT_J5_LIFT_STEP_DEG,
              "max_lift_deg=", CONTACT_J5_LIFT_MAX_DEG,
              "window_s=", CONTACT_J5_LIFT_STEP_INTERVAL_S,
              "total_contact_wait_limit_s=", SCRATCH_CONTACT_KEYFRAME_TIMEOUT_S)
        self.current_mode = mode
        self.current_lane_spec = lane_spec
        self.segment_index = 1
        self.segment_elapsed = 0.0
        self.segment_arrival_wait_elapsed = 0.0
        self.segment_settle_elapsed = 0.0
        self.segment_wait_log_elapsed = 0.0
        self.contact_j5_lift_deg = 0.0
        self.dynamic_depth_stats = {}
        self._actual_contact_locations = {}
        self._actual_tip_trace = {finger: [] for finger in LONG_FINGERS}
        self._actual_tip_trace_last_sim = -1.0
        self._actual_tip_trace_error_logged = False
        self._speed_sample = None
        self.dynamic_depth_reported_cycles = set()
        self.imitation_cycle_quality = {}
        self.full_curl_joint_records = {}
        self.early_exit_joint_records = {}
        self.base_contact_stop_records = {}
        self.scratch_completed_records = {}
        self._base_contact_pairs = {}
        self._release_tail_active = False
        self.optional_scratch_fingers = {}
        self.relaxed_scratch_fingers = {}
        self.primary_scratch_fingers = {}
        self.executing = True
        self.active_fingers = []
        self.current_authorized_contact_fingers = list(
            trajectory[0].get(
                "authorized_contact_fingers",
                trajectory[0].get("active", []),
            )
        )
        self.planned_working_fingers = list(planned_working_fingers or [])
        self.current_stage = str(trajectory[0].get("stage", "CURRENT_HOLD"))
        self.current_segment_alpha = 0.0
        self.deep_adjust_latched = {
            finger: False
            for finger in LONG_FINGERS
        }

        self.command(trajectory[0]["q"], trajectory[0]["fingers"])
        self.write_status("moving", "Moving to selected target")

    def finish_cycle(self):
        self.report_contact_diagnostic_coverage("FINISHED")
        for cycle_index in sorted(self.dynamic_depth_stats):
            self.report_dynamic_depth_cycle(cycle_index)
        # Freeze and display the measured tip-link trace before issuing the
        # final Home command.  The actual scratch subset is thick/bright;
        # approach, unload and Home are retained as a thin reference path.
        try:
            self.record_actual_tip_trace()
            # Keep target/palm diagnostics, but remove the large planning arcs
            # once execution is complete so the viewport reflects the motion
            # that actually happened.
            planned_finger_root = V100_DEBUG_ROOT + "/FingerArcs"
            if self.stage.GetPrimAtPath(planned_finger_root).IsValid():
                self.stage.RemovePrim(planned_finger_root)
            target = None
            if isinstance(self.current_mode, dict):
                target = self.current_mode.get("anchor")
            draw_actual_tip_trajectories(self.stage, self._actual_tip_trace, target)
        except Exception as exc:
            print("[V103 ACTUAL TIP TRACE DRAW FAILED]", repr(exc))
        self.executing = False
        self.active_fingers = []
        self.current_authorized_contact_fingers = []
        self.planned_working_fingers = []
        self.current_stage = "IDLE"
        self.current_segment_alpha = 0.0
        self.segment_arrival_wait_elapsed = 0.0
        self.segment_settle_elapsed = 0.0
        self.segment_wait_log_elapsed = 0.0
        self.contact_j5_lift_deg = 0.0
        self.deep_adjust_latched = {
            finger: False
            for finger in LONG_FINGERS
        }

        # Final guaranteed command = upright Home + open fingers.
        self.command(self.home_q, empty_fingers())

        print("")
        print("==============================================")
        print("SCRATCH CYCLE COMPLETE")
        print("UNLOAD -> OPEN -> RETREAT -> HOME complete.")
        print("Waiting for terminal repeat decision.")
        print("==============================================")

        self.begin_console_prompt("repeat")

    def release_signed_feedback(self):
        result = {}
        for f in LONG_FINGERS:
            tip = get_world_position(self.stage, FINGER_TIP_PATHS[f])
            if tip is None:
                raise RuntimeError('RELEASE_TIP_FEEDBACK_MISSING|' + f)
            result[f] = float(final_surface_relation(tip, self.back_mesh_query,
                self.back_points, self.back_normals, self.back_tree)['signed'])
        if not all(math.isfinite(v) for v in result.values()):
            raise RuntimeError('RELEASE_TIP_FEEDBACK_NONFINITE')
        return result

    def release_exact_sweep(self, q0, hand0, q1, hand1, authorized, require_clear=False):
        """Check the actual 12 joint path, not a scalar curl-fit approximation."""
        model = HAND_COLLISION_MODEL
        if model is None:
            return False, 'RELEASE_GEOMETRY_UNAVAILABLE'
        chains = getattr(self, '_depth_hand_chains', None)
        if chains is None:
            chains = self._depth_hand_chains = HAND_MODEL.hand_chains(COLLISION_URDF)
        qspan = float(np.max(np.abs(np.degrees(q1-q0))))
        hspan = max(float(np.max(np.abs(hand1[f]-hand0[f]))) for f in LONG_FINGERS)
        # Match the collision sampler's sub-degree resolution, with a bound
        # that rejects rather than silently coarsening a large route.
        count = max(18, int(math.ceil(max(qspan, hspan) / .5)))
        if count > 720:
            return False, 'RELEASE_SWEEP_TOO_LARGE'
        prior = None
        first_depth = None
        for i, alpha in enumerate(np.linspace(0., 1., count+1)):
            q = q0 + alpha*(q1-q0)
            hand = {f: hand0[f]+alpha*(hand1[f]-hand0[f]) for f in LONG_FINGERS}
            if (not np.all(np.isfinite(q)) or np.any(q < self.lower) or np.any(q > self.upper)
                    or any(not np.all(np.isfinite(v)) or np.any(v < -1.)
                           or np.any(v > np.asarray(HAND_MODEL.LIMITS_DEG)+1.) for v in hand.values())):
                return False, 'RELEASE_JOINT_LIMIT'
            joints = self.depth_joint_values(hand)
            p, r = self.solver.compute_forward_kinematics(PALM_FRAME, q)
            depths = {}
            for f in LONG_FINGERS:
                tip = np.asarray(p) + np.asarray(r) @ HAND_MODEL.tip_local(chains[f], joints)
                rel = final_surface_relation(tip, self.back_mesh_query,
                    self.back_points, self.back_normals, self.back_tree)
                signed = float(rel['signed'])
                depth = depths[f] = -signed
                cap = max(.004, (first_depth or {}).get(f, .004)) if f in authorized else 0.
                if not math.isfinite(depth) or depth > cap + 1e-6 or depth > .008:
                    return False, 'RELEASE_DEPTH|' + f
                if require_clear and i == count and signed < V103_SAFETY_CLEARANCE_M:
                    return False, 'RELEASE_ENDPOINT_NOT_CLEAR|' + f
            if first_depth is None:
                first_depth = depths
            probes = model.world_probe_points(self.solver, q, {}, actual_joint_values=joints)
            args = (self.back_points, self.back_normals, self.back_tree, self.back_mesh_query)
            if prior is None:
                safe, reason = model.check_static_points(probes, *args,
                    allowed_contact_fingers=set(authorized))
            else:
                safe, reason = model.check_swept_points(prior, probes, *args,
                    step_index=i, path_fraction=float(alpha), allowed_contact_fingers=set(authorized))
            if not safe:
                return False, reason
            prior = probes
        return True, 'PASS'

    def start_measured_release_tail(self, reason, cycle, signed, contact):
        """Latch the stop once; discard remaining curls and recheck an exit.

        Reuse only unload poses as proposals. The hand is held at the
        measured twelve angles until lift is complete, then opens in free space.
        No new scratch, no start_execution() call that erases earned telemetry.
        """
        self.hold_actual_state()
        q0 = np.asarray(self.read_actual_arm_q(), dtype=float)
        hand0 = {f: np.asarray(v, dtype=float) for f, v in self.read_actual_finger_triplets_deg().items()}
        record = dict(reason=reason, source='ACTUAL_TIP_MARKERS' if contact is None else contact['source'],
            signed_mm={f:v*1000. for f,v in signed.items()}, contact=contact,
            actual_q_deg=np.degrees(q0).tolist(),
            actual_joint_deg={f:v.tolist() for f,v in hand0.items()})
        if reason == 'BASE_CONTACT':
            self.base_contact_stop_records[cycle] = record
        else:
            self.early_exit_joint_records[cycle] = record
        # A no-jump physical hand model must agree with the observed markers.
        chains = HAND_MODEL.hand_chains(COLLISION_URDF)
        joints = self.depth_joint_values(hand0)
        p0, r0 = self.solver.compute_forward_kinematics(PALM_FRAME, q0)
        for f in LONG_FINGERS:
            predicted = np.asarray(p0) + np.asarray(r0) @ HAND_MODEL.tip_local(chains[f], joints)
            if np.linalg.norm(predicted-get_world_position(self.stage, FINGER_TIP_PATHS[f])) > .001:
                raise RuntimeError('RELEASE_MARKER_FK_MISMATCH|' + f)
        opened = {f: np.asarray(HAND_MODEL.joint_targets(0.), dtype=float) for f in LONG_FINGERS}
        tail = None
        last = 'NO_UNLOAD_CANDIDATE'
        standalone_home_tail = False
        seen_lifts = set()
        for candidate in RELEASE_POLICY.tail_candidates(self.trajectory, self.segment_index):
            lift = np.asarray(candidate['q'], dtype=float).copy()
            key = tuple(lift.tolist())
            if key in seen_lifts:
                continue
            seen_lifts.add(key)
            resume_index = RELEASE_POLICY.resume_open_index(self.trajectory, self.segment_index, lift)
            # A measured stop can occur after a deep-adjust or an overlapped
            # inter-cycle reset whose planned OPEN_AFTER_UNLOAD checkpoint is
            # not at exactly the candidate lift pose.  That bookkeeping gap
            # must not turn an otherwise safe release into a batch failure.
            # In that case we build and verify a standalone open -> retreat ->
            # Home tail below instead of trying to splice back into a stale
            # multi-cycle trajectory.
            standalone = resume_index is None
            ok, last = self.release_exact_sweep(q0, hand0, lift, hand0, LONG_FINGERS, True)
            if not ok:
                continue
            ok, last = self.release_exact_sweep(lift, hand0, lift, opened, (), True)
            if not ok:
                continue
            tail = [
                dict(
                    stage='UNLOAD_STEP_MEASURED_RELEASE_CYCLE_'+str(cycle),
                    q=lift,
                    hand=hand0,
                    duration=arm_move_duration(q0, lift, .3),
                    authorized=list(LONG_FINGERS),
                ),
                dict(
                    stage='OPEN_AFTER_UNLOAD_CYCLE_'+str(cycle),
                    q=lift,
                    hand=opened,
                    duration=OPEN_AFTER_UNLOAD_TIME_S,
                    authorized=[],
                ),
            ]
            if standalone:
                mode = self.current_mode if isinstance(self.current_mode, dict) else {}

                def check_retreat_segment(a, b):
                    return self.release_exact_sweep(
                        np.asarray(a, dtype=float), opened,
                        np.asarray(b, dtype=float), opened,
                        (), True,
                    )

                def local_lift(a):
                    normal = mode.get('normal')
                    if normal is None:
                        return None
                    p, r = self.solver.compute_forward_kinematics(PALM_FRAME, a)
                    return solve_offset_pose(
                        self.solver,
                        p,
                        r,
                        normal,
                        SAFE_RETREAT_OFFSET_M / 10.0,
                        a,
                        self.home_q,
                        self.lower,
                        self.upper,
                    )

                return_route, retreat_method = v103_plan_retreat_route(
                    lift,
                    [],
                    self.home_q,
                    self.lower,
                    self.upper,
                    check_retreat_segment,
                    local_lift,
                    prefer_direct_home=True,
                )
                for route_index, retreat_q in enumerate(return_route):
                    is_home = route_index == len(return_route) - 1
                    tail.append(
                        dict(
                            stage='RETURN_HOME' if is_home else 'SAFE_RETREAT',
                            q=np.asarray(retreat_q, dtype=float),
                            hand=opened,
                            duration=arm_move_duration(
                                lift,
                                retreat_q,
                                RETURN_HOME_MIN_TIME_S if is_home else .30,
                            ),
                            authorized=[],
                        )
                    )
                    lift = np.asarray(retreat_q, dtype=float)
                tail.append(
                    dict(
                        stage='HOME_WAIT',
                        q=np.asarray(self.home_q, dtype=float),
                        hand=opened,
                        duration=.20,
                        authorized=[],
                    )
                )
                standalone_home_tail = True
            break
        if tail is None:
            raise RuntimeError('RELEASE_STOP_HELD_NO_VERIFIED_EXIT|' + last)
        self._release_tail = tail
        self._release_tail_previous = dict(q=q0, hand=hand0)
        self._release_tail_elapsed = self._release_tail_wait = self._release_tail_settle = 0.
        self._release_tail_active = True
        prefix = self.trajectory[:self.segment_index]
        # Delete just the stopped release portion. The same open checkpoint
        # rejoins the previously checked suffix, including all later cycles
        # and Home. Never silently shorten a multi-cycle dataset trial.
        if standalone_home_tail:
            self._release_resume_trajectory = []
            self._release_resume_index = 0
        else:
            self._release_resume_trajectory = prefix + self.trajectory[resume_index:]
            self._release_resume_index = len(prefix)+1
        self.trajectory = prefix
        self.contact_j5_lift_deg = 0.
        self.current_stage = 'UNLOAD_STEP_MEASURED_RELEASE_CYCLE_' + str(cycle)
        self.active_fingers = list(LONG_FINGERS)
        self.current_authorized_contact_fingers = list(LONG_FINGERS)
        print('[V103 RELEASE STOP LATCHED]', record, 'remaining_curl_discarded=True; exact_12DOF_exit_checked=True')
        self.report_dynamic_depth_cycle(cycle)

    def update_measured_release_tail(self, dt):
        if not self._release_tail:
            self._release_tail_active = False
            self.trajectory = self._release_resume_trajectory
            self.segment_index = self._release_resume_index
            self.segment_elapsed = self.segment_arrival_wait_elapsed = 0.
            self.segment_settle_elapsed = self.segment_wait_log_elapsed = 0.
            self.active_fingers = self.current_authorized_contact_fingers = []
            if self.segment_index >= len(self.trajectory):
                self.finish_cycle()
            return
        target = self._release_tail[0]
        previous = self._release_tail_previous
        # Check actual clearance BEFORE emitting any opening drive target.
        # An external disturbance after the lift must not open into the back.
        if target['stage'].startswith('OPEN_AFTER_UNLOAD') and any(
                s < V103_SAFETY_CLEARANCE_M for s in self.release_signed_feedback().values()):
            raise RuntimeError('MEASURED_RELEASE_OPEN_NOT_CLEAR')
        self._release_tail_elapsed += float(dt)
        alpha = smoothstep(min(1., self._release_tail_elapsed / max(target['duration'], 1e-5)))
        q = previous['q'] + alpha*(target['q']-previous['q'])
        hand = {f: previous['hand'][f]+alpha*(target['hand'][f]-previous['hand'][f]) for f in LONG_FINGERS}
        for i, name in enumerate(ARM_JOINTS):
            set_drive_deg(self.arm_drives[name], float(np.degrees(q[i])))
        for f in LONG_FINGERS:
            for drive, value in zip(self.finger_triplet_drives[f], hand[f]):
                set_drive_deg(drive, float(value))
        self.current_arm_command = q.copy()
        self.current_finger_commands = {f: HAND_MODEL.progress_from_triplet(hand[f])[0] for f in LONG_FINGERS}
        self.current_stage = target['stage']
        self.current_authorized_contact_fingers = list(target['authorized'])
        self.active_fingers = list(target['authorized'])
        self.current_segment_alpha = alpha
        self.write_status('unloading', 'Measured release tail: ' + self.current_stage)
        if alpha >= 1.:
            self._release_tail_wait += float(dt)
            actual = self.read_actual_finger_triplets_deg()
            arrived = (np.max(np.abs(np.degrees(self.read_actual_arm_q()-target['q']))) <= KEYFRAME_ARM_ARRIVAL_TOL_DEG
                       and all(np.max(np.abs(actual[f]-target['hand'][f])) <= 2. for f in LONG_FINGERS))
            self._release_tail_settle = self._release_tail_settle + float(dt) if arrived else 0.
            if self._release_tail_wait > 10.:
                raise RuntimeError('MEASURED_RELEASE_TRACKING_TIMEOUT|' + self.current_stage)
            if self._release_tail_settle >= .1:
                if self.current_stage.startswith('UNLOAD_STEP_') and any(
                        s < V103_SAFETY_CLEARANCE_M for s in self.release_signed_feedback().values()):
                    raise RuntimeError('MEASURED_RELEASE_LIFT_NOT_CLEAR')
                self._release_tail_previous = self._release_tail.pop(0)
                self._release_tail_elapsed = self._release_tail_wait = self._release_tail_settle = 0.

    def _apply_wave_pressure_guard(
        self,
        stage,
        previous_q_effective,
        target_q_effective,
        previous_finger_commands,
        target_finger_commands,
        authorized_fingers,
        focus_name=None,
    ):
        """Prevent shared wrist pressure from burying an already-deep core.

        Planned keyframes use command-FK, while the real PhysX fingertips can
        lag by several degrees. Once any authorized live contact reaches the
        guard band, impose a positive-J5 lift floor immediately. The current
        focus is prevented from receiving more curl, but non-focus fingers are
        still allowed to continue their curl-away/background motion so the
        handoff does not become another stop-and-wait sequence.
        """
        stage = str(stage)
        if not stage.startswith("SCRATCH_WAVE_ASYNC_"):
            self._wave_pressure_guard_cycle = None
            self._wave_pressure_guard_fingers = set()
            self._wave_pressure_guard_logged = set()
            self._wave_pressure_guard_j5_floor = None
            self._wave_pressure_guard_j5_hard = False
            return

        cycle = self.scratch_cycle_from_stage(stage)
        if cycle != getattr(self, "_wave_pressure_guard_cycle", None):
            self._wave_pressure_guard_cycle = cycle
            self._wave_pressure_guard_fingers = set()
            self._wave_pressure_guard_logged = set()
            self._wave_pressure_guard_j5_floor = None
            self._wave_pressure_guard_j5_hard = False

        try:
            actual_q, actual_hand, depths = self.depth_feedback()
            actual_q = np.asarray(actual_q, dtype=np.float64)
            if actual_q.shape[0] <= 4 or not np.all(np.isfinite(actual_q)):
                return
        except Exception:
            return

        authorized = set(authorized_fingers or ())
        if not authorized:
            return
        previous_guard = set(getattr(self, "_wave_pressure_guard_fingers", set()))
        guard_fingers = set()
        for finger in LONG_FINGERS:
            if finger not in authorized:
                continue
            depth = float(depths.get(finger, float("nan")))
            if not math.isfinite(depth):
                continue
            if (depth >= SCRATCH_WAVE_PRESSURE_GUARD_ENTER_M
                    or (finger in previous_guard
                        and depth >= SCRATCH_WAVE_PRESSURE_GUARD_EXIT_M)):
                guard_fingers.add(finger)
        self._wave_pressure_guard_fingers = guard_fingers
        if not guard_fingers:
            # Once the old finger has curled back below the hysteresis band,
            # release the temporary lift floor so the wrist may support the
            # next finger again.
            self._wave_pressure_guard_j5_floor = None
            self._wave_pressure_guard_j5_hard = False
            return

        actual_j5 = float(actual_q[4])
        deepest = max(
            (float(depths.get(f, float("nan"))) for f in guard_fingers),
            default=float("nan"),
        )
        hard_any = math.isfinite(deepest) and deepest >= SCRATCH_WAVE_PRESSURE_GUARD_HARD_M
        lift_deg = (SCRATCH_WAVE_PRESSURE_GUARD_HARD_LIFT_DEG
                    if hard_any else SCRATCH_WAVE_PRESSURE_GUARD_LIFT_DEG)
        floor = getattr(self, "_wave_pressure_guard_j5_floor", None)
        if floor is None:
            floor = actual_j5 + math.radians(lift_deg)
        else:
            floor = max(float(floor), actual_j5)
        if hard_any and not bool(getattr(self, "_wave_pressure_guard_j5_hard", False)):
            floor = max(
                floor,
                actual_j5 + math.radians(SCRATCH_WAVE_PRESSURE_GUARD_HARD_LIFT_DEG),
            )
            self._wave_pressure_guard_j5_hard = True
        floor = float(np.clip(floor, float(self.lower[4]), float(self.upper[4])))
        self._wave_pressure_guard_j5_floor = floor
        before_target_j5 = float(target_q_effective[4])
        previous_q_effective[4] = max(float(previous_q_effective[4]), floor)
        target_q_effective[4] = max(float(target_q_effective[4]), floor)

        for finger in sorted(guard_fingers):
            depth = float(depths.get(finger, float("nan")))
            if not math.isfinite(depth):
                continue
            # Only the current owner is curl-held. A completed/non-focus
            # finger must be free to curl away faster than the shared wrist
            # pressure, exactly as the intended scratch handoff requires.
            curl_action = "CONTINUE_NONFOCUS_CURL"
            if finger == focus_name:
                prev = float(previous_finger_commands.get(finger, 0.0))
                target = float(target_finger_commands.get(finger, prev))
                hold = prev
                try:
                    measured_progress = float(HAND_MODEL.progress_from_triplet(
                        actual_hand[finger])[0])
                    if math.isfinite(measured_progress):
                        hold = max(prev, min(target, measured_progress))
                except Exception:
                    pass
                target_finger_commands[finger] = min(target, hold)
                curl_action = "HOLD_FOCUS_CURL"

            hard = depth >= SCRATCH_WAVE_PRESSURE_GUARD_HARD_M
            log_key = (cycle, finger, "hard" if hard else "guard")
            if log_key not in getattr(self, "_wave_pressure_guard_logged", set()):
                print(
                    "[V103 WAVE PRESSURE GUARD]",
                    "cycle=", cycle,
                    "finger=", finger,
                    "focus=", focus_name,
                    "depth_mm=", round(depth * 1000.0, 3),
                    "actual_j5_deg=", round(math.degrees(actual_j5), 3),
                    "target_j5_before_deg=", round(math.degrees(before_target_j5), 3),
                    "target_j5_floor_deg=", round(math.degrees(floor), 3),
                    "curl_action=", curl_action,
                    "pressure_action=", "J5_HARD_LIFT" if hard else "J5_HOLD_LIFT",
                    "runtime_only=True",
                )
                self._wave_pressure_guard_logged.add(log_key)

    def update_execution(self, dt):
        if not self.executing:
            return
        if getattr(self, '_release_tail_active', False):
            try:
                self.update_measured_release_tail(dt)
            except Exception as exc:
                self.emergency_stop('MEASURED_RELEASE_EXIT_FAILED|' + repr(exc))
            return
        if getattr(self, '_depth_recovery', None) is not None:
            self.update_depth_recovery(dt)
            return  # Freeze the original arm AND curl interpolation clocks.
        self.contact_diag_elapsed_s = getattr(self, "contact_diag_elapsed_s", 0.0) + float(dt)

        if self.segment_index >= len(self.trajectory):
            self.finish_cycle()
            return

        previous = self.trajectory[self.segment_index - 1]
        target = self.trajectory[self.segment_index]
        if str(target.get('stage', '')).startswith(V103_BASE_CONTACT_STOP_STAGES):
            try:
                signed = self.release_signed_feedback()
                contact = next(iter(getattr(self, '_base_contact_pairs', {}).values()), None)
                reason = RELEASE_POLICY.release_reason(target['stage'], signed, contact)
                if reason:
                    cycle = self.scratch_cycle_from_stage(target['stage'])
                    if cycle not in self.scratch_completed_records:
                        raise RuntimeError('RELEASE_STOP_BEFORE_SCRATCH_COMPLETED')
                    self.start_measured_release_tail(reason, cycle, signed, contact)
                    return
            except Exception as exc:
                self.emergency_stop('RELEASE_STOP_FAILED|' + repr(exc))
                return

        previous_active = list(previous.get("active", []))
        target_active = list(target.get("active", []))
        previous_authorized = set(
            previous.get("authorized_contact_fingers", previous_active)
        )
        target_authorized = set(
            target.get("authorized_contact_fingers", target_active)
        )

        duration = max(float(target["duration"]), 1e-5)
        # Advance the shared arm/finger clock only while the last commanded
        # pose is tracked. Endpoint-only waits allow PIP/DIP phase divergence
        # during pressure transfer even when the planned sweep is valid.
        tracking_stage = str(target.get('stage', ''))
        wave_clock_scale = 1.0
        if tracking_stage.startswith('SCRATCH_WAVE_ASYNC_'):
            tracking_key = (int(self.segment_index), tracking_stage)
            if tracking_key != getattr(self, '_wave_tracking_hold_key', None):
                self._wave_tracking_hold_key = tracking_key
                self._wave_tracking_wait_s = 0.0
                self._wave_tracking_hold_wall_started = None
                self._wave_tracking_recover_s = 0.0
                self._wave_tracking_log_s = -1.0
            try:
                arm_error = float(np.max(np.abs(np.degrees(
                    self.read_actual_arm_q()-self.current_arm_command))))
                errors = HAND_MODEL.triplet_errors(
                    self.read_actual_finger_triplets_deg(),
                    checked_finger_commands(self.current_finger_commands))
                # Prefer the frame's measured dynamic focus.  A completed
                # finger is held during serial handoff; letting its stale
                # drive residual win the global max would pause the next
                # finger and recreate a false cross-finger stall.  If the
                # frame has no focus (legacy/release), retain the full-hand
                # safety check.
                focus_name = target.get('dynamic_focus')
                if focus_name in errors:
                    error_fingers = [focus_name]
                else:
                    error_fingers = list(errors)
                worst = max(error_fingers, key=lambda f: float(np.max(errors[f])))
                finger_error = float(np.max(errors[worst]))
                if not np.isfinite(arm_error+finger_error):
                    raise RuntimeError('Nonfinite tracking error')
            except Exception as exc:
                self.emergency_stop('WAVE_FEEDBACK_INVALID|'+repr(exc))
                return
            # Moderate drive lag must not freeze the shared wave clock.  A
            # global pause makes every other finger wait for the slowest one.
            # Keep the hard hold only for a genuinely unsafe tracking loss;
            # the intermediate band is handled as a bounded slowdown below.
            safety_hold = bool(
                arm_error > SCRATCH_WAVE_TRACKING_HOLD_ARM_DEG * 2.0
                or finger_error > SCRATCH_WAVE_TRACKING_HOLD_FINGER_DEG * 1.5
            )
            tracking_lag = bool(
                arm_error > SCRATCH_WAVE_TRACKING_HOLD_ARM_DEG
                or finger_error > SCRATCH_WAVE_TRACKING_HOLD_FINGER_DEG
            )
            soft_lag = bool(
                arm_error > SCRATCH_WAVE_TRACKING_SLOW_ARM_DEG
                or finger_error > SCRATCH_WAVE_TRACKING_SLOW_FINGER_DEG
            )
            wave_clock_scale = 1.0
            if safety_hold:
                now = time.perf_counter()
                started = getattr(self, '_wave_tracking_hold_wall_started', None)
                if started is None:
                    started = now
                    self._wave_tracking_hold_wall_started = started
                wait = max(0.0, now - started)
                self._wave_tracking_wait_s = wait
                self._wave_tracking_recover_s = 0.0
                if wait-float(getattr(self, '_wave_tracking_log_s', -1.)) >= .5:
                    print('[V103 WAVE TRACKING HOLD]', 'finger=', worst,
                          'finger_error_deg=', round(finger_error,3),
                          'arm_error_deg=', round(arm_error,3), 'wait_wall_s=',round(wait,3))
                    self._wave_tracking_log_s = wait
                if wait >= SCRATCH_WAVE_TRACKING_RECOVERY_WALL_S:
                    self.recover_wave_tracking(worst, arm_error, finger_error)
                return
            if soft_lag:
                finger_span = max(
                    SCRATCH_WAVE_TRACKING_HOLD_FINGER_DEG
                    - SCRATCH_WAVE_TRACKING_SLOW_FINGER_DEG, 1e-6)
                arm_span = max(
                    SCRATCH_WAVE_TRACKING_HOLD_ARM_DEG
                    - SCRATCH_WAVE_TRACKING_SLOW_ARM_DEG, 1e-6)
                severity = max(
                    (finger_error-SCRATCH_WAVE_TRACKING_SLOW_FINGER_DEG)/finger_span,
                    (arm_error-SCRATCH_WAVE_TRACKING_SLOW_ARM_DEG)/arm_span,
                    0.0,
                )
                wave_clock_scale = max(
                    SCRATCH_WAVE_TRACKING_MIN_CLOCK_SCALE,
                    1.0-(1.0-SCRATCH_WAVE_TRACKING_MIN_CLOCK_SCALE)*min(severity,1.0),
                )
            # Crossing the old hold threshold is a slowdown, not a global
            # pause. The next finger and the releasing finger keep advancing
            # while the lagging finger catches up; depth/collision guards still
            # remain authoritative.
            if tracking_lag and not safety_hold:
                wave_clock_scale = min(wave_clock_scale, 0.50)
            # Require a short continuous in-tolerance interval before clearing
            # a hold.  This filters the one-frame "good" samples that were
            # previously resetting wait_s to zero while PIP/DIP were still
            # lagging, which let the wave keep advancing into over-depth.
            recovery = getattr(self, '_wave_tracking_recover_s', 0.) + float(dt)
            self._wave_tracking_recover_s = recovery
            if getattr(self, '_wave_tracking_wait_s', 0.) > 0.0 and recovery < 0.20:
                return
            self._wave_tracking_hold_wall_started = None
            self._wave_tracking_wait_s = 0.0
            self._wave_tracking_recover_s = 0.0
            self._wave_tracking_log_s = -1.0
            # Live depth may outrun command-FK even while joint tracking looks
            # acceptable. Slow (do not stop) the shared wave clock before the
            # hard 4 mm gate so old-finger curl-away can catch up with wrist
            # pressure. The J5 floor below provides the geometric relief.
            try:
                _, _, live_depths = self.depth_feedback()
                guard_depth = max(
                    [float(live_depths.get(f, float("nan")))
                     for f in (set(previous_active) | set(target_active)
                               | previous_authorized | target_authorized)
                     if math.isfinite(float(live_depths.get(f, float("nan"))))]
                    or [0.0]
                )
                if guard_depth >= SCRATCH_WAVE_PRESSURE_GUARD_HARD_M:
                    wave_clock_scale = min(wave_clock_scale, 0.25)
                elif guard_depth >= SCRATCH_WAVE_PRESSURE_GUARD_ENTER_M:
                    wave_clock_scale = min(wave_clock_scale, 0.50)
            except Exception:
                pass
        else:
            self._wave_tracking_hold_key = None
            self._wave_tracking_wait_s = 0.0
            self._wave_tracking_recover_s = 0.0
            self._wave_tracking_log_s = -1.0
        self.segment_elapsed += float(dt) * wave_clock_scale

        raw_alpha = self.segment_elapsed / duration
        stage = str(target.get("stage", "UNKNOWN"))
        if target.get('scratch_optional_fingers'):
            self.optional_scratch_fingers[self.scratch_cycle_from_stage(stage)] = list(
                target['scratch_optional_fingers']
            )
        if target.get('scratch_relaxed_fingers') is not None:
            self.relaxed_scratch_fingers[self.scratch_cycle_from_stage(stage)] = list(
                target.get('scratch_relaxed_fingers', [])
            )
        if target.get('scratch_primary_finger') in LONG_FINGERS:
            self.primary_scratch_fingers[self.scratch_cycle_from_stage(stage)] = str(
                target['scratch_primary_finger']
            )
        # Contact-wave keyframes are samples of one continuous demonstration,
        # not independent point-to-point moves.  Linear interpolation preserves
        # velocity across their boundaries; per-segment smoothstep used to
        # force velocity to zero at every sample and taught a stop-go motion.
        alpha = (
            clamp(raw_alpha, 0.0, 1.0)
            if stage.startswith("SCRATCH_WAVE_ASYNC_")
            else smoothstep(raw_alpha)
        )
        self.current_stage = stage
        self.current_segment_alpha = clamp(raw_alpha, 0.0, 1.0)
        if stage.startswith(("UNLOAD_STEP_", "UNLOAD_J5_LIFT_")):
            cycle_index = self.scratch_cycle_from_stage(stage)
            if cycle_index is not None:
                self.report_dynamic_depth_cycle(cycle_index)

        # Contact assist is intentionally simple: joint5 alone lifts by small
        # positive increments.  Apply the same offset to every contact frame
        # so interpolation and feedback tracking use the actual commanded pose.
        previous_q_effective = np.asarray(previous["q"], dtype=np.float64).copy()
        target_q_effective = np.asarray(target["q"], dtype=np.float64).copy()
        previous_finger_commands = {
            finger: float(previous["fingers"].get(finger, 0.0))
            for finger in LONG_FINGERS
        }
        target_finger_commands = {
            finger: float(target["fingers"].get(finger, 0.0))
            for finger in LONG_FINGERS
        }
        contact_j5_lift_rad = math.radians(float(self.contact_j5_lift_deg))
        entering_pressure_wave = bool(
            stage.startswith("SCRATCH_WAVE_ASYNC_")
            and not str(previous.get("stage", "")).startswith(
                "SCRATCH_WAVE_ASYNC_"
            )
        )
        if self.frame_uses_contact_j5_lift(previous):
            previous_q_effective[4] += contact_j5_lift_rad
        if self.frame_uses_contact_j5_lift(target):
            # The positive J5 stall-assist used during acquisition must fade
            # out as the planned negative-J5 pressure wave begins; otherwise a
            # +1 degree lift cancels most of the planned skin pressure.
            if not entering_pressure_wave:
                target_q_effective[4] += contact_j5_lift_rad

        self._apply_wave_pressure_guard(
            stage,
            previous_q_effective,
            target_q_effective,
            previous_finger_commands,
            target_finger_commands,
            set(previous_active) | set(target_active)
            | previous_authorized | target_authorized,
            focus_name=target.get("dynamic_focus"),
        )
        q = previous_q_effective + alpha * (
            target_q_effective - previous_q_effective
        )

        fingers = empty_fingers()

        for finger in LONG_FINGERS:
            a = float(previous_finger_commands[finger])
            b = float(target_finger_commands[finger])
            fingers[finger] = a + alpha * (b - a)

        # Offline bounding boxes are not measured base contact evidence.
        self.command(q, fingers)

        # A target keyframe's newly acquired finger is not active at the start
        # of the interpolation. Keep only the previous keyframe's contacts
        # latched until the target is actually reached. Planned transition
        # authorization is tracked separately and remains lane-gated in the
        # runtime safety check.
        self.active_fingers = previous_active
        self.current_authorized_contact_fingers = list(
            previous_authorized | target_authorized
        )

        if (
            stage.startswith("PRE_FLEX_")
            or stage.startswith("JOIN_CONTACT_")
            or stage.startswith("SYNC_")
            or stage.startswith("ACQUIRE_")
            or stage.startswith("ARC_PRECONTACT_")
            or stage.startswith("ARC_CONTACT_")
            or stage.startswith("ARC_SYNC_")
            or stage.startswith("ARC_ASYNC_")
            or stage.startswith("DEEP_ADJUST_")
        ):
            self.write_status("scratching", stage)
        elif stage == "PRE_FOLD_INACTIVE":
            self.write_status("moving", "Folding non-working fingers before approach")
        elif stage == "PRE_SHAPE_HAND":
            self.write_status(
                "moving",
                "Folding inactive fingers in free space; working fingers stay open"
            )
        elif stage.startswith("APPROACH_") or stage.startswith("ARC_APPROACH_"):
            self.write_status(
                "moving",
                "Palm-aligned normal approach: "
                +
                stage
            )
        elif stage.startswith("ARC_HANDOFF_PRECURVE"):
            self.write_status(
                "moving",
                "Late finger-arc handoff with compliant terminal contact: " + stage
            )
        elif stage == "FULL_SCRATCH" or stage.startswith("FULL_SCRATCH_CONTACT_CYCLE_"):
            self.write_status("scratching", "Full scratch contact")
        elif stage.startswith("SCRATCH_CURL_"):
            self.write_status("scratching", "Completing four-finger curl: " + stage)
        elif stage.startswith("SCRATCH_WAVE_ASYNC_"):
            self.write_status("scratching", "Overlapping shallow-to-deep comfort wave: " + stage)
        elif stage.startswith("FINGER_CURL_RELEASE_"):
            self.write_status("unloading", "Curling fingers independently away from back: " + stage)
        elif stage.startswith("UNLOAD_J5_LIFT_"):
            self.write_status(
                "unloading",
                "Releasing contact with Joint5 wrist lift: " + stage,
            )
        elif stage.startswith("UNLOAD_STEP_"):
            self.write_status(
                "unloading",
                "Whole-hand unload with fingers flexed: "
                +
                stage
            )
        elif stage.startswith("UNLOAD_ALL"):
            self.write_status(
                "unloading",
                "Mesh clearance confirmed - safe to open"
            )
        elif stage.startswith("OPEN_AFTER_UNLOAD"):
            self.write_status("unloading", "Opening fingers after clearance")
        elif stage == "SAFE_RETREAT":
            self.write_status("moving", "Safe retreat from back")
        elif stage == "RETURN_HOME":
            self.write_status("returning_home", "Returning to upright Home")

        if raw_alpha >= 1.0:
            self.command(target_q_effective, target["fingers"])
            self.segment_arrival_wait_elapsed += float(dt)
            self.segment_wait_log_elapsed += float(dt)

            try:
                actual_q = self.read_actual_arm_q()
                actual_triplets = self.read_actual_finger_triplets_deg()
            except Exception as exc:
                self.emergency_stop(
                    "KEYFRAME_FEEDBACK_READ_FAIL_"
                    + str(target.get("stage", "UNKNOWN"))
                    + "_" + repr(exc)
                )
                return

            arm_error_deg = np.abs(
                np.degrees(actual_q - target_q_effective)
            )
            triplet_errors = HAND_MODEL.triplet_errors(
                actual_triplets, checked_finger_commands(target["fingers"])
            )
            finger_error_deg = {finger: float(np.max(errors)) for finger, errors in triplet_errors.items()}
            max_arm_error_deg = float(np.max(arm_error_deg))
            worst_arm_joint = int(np.argmax(arm_error_deg))
            worst_finger = max(finger_error_deg, key=finger_error_deg.get)
            max_finger_error_deg = float(finger_error_deg[worst_finger])
            # Once a fingertip has established planned contact, a small steady
            # position-drive error is expected because the BackSurface pushes
            # back.  Use the existing contact tolerance for every keyframe that
            # holds, acquires or advances a committed contact.  Previously the
            # ARC_SYNC_CONTACT_* stages accidentally fell through to the strict
            # free-space 3-degree rule and stopped at a harmless 3.53 degrees.
            contact_tracking_stage = bool(
                previous_active
                or target_active
                or stage.startswith("ARC_SYNC_CONTACT_")
                or stage.startswith("ARC_ASYNC_CONTACT_")
                or stage.startswith("SCRATCH_WAVE_ASYNC_")
                or stage.startswith("SCRATCH_CURL_")
                or stage.startswith("FINGER_CURL_RELEASE_")
                or stage.startswith("FULL_SCRATCH_CONTACT_CYCLE_")
            )
            acquisition_tracking_stage = stage.startswith(
                ("ARC_SYNC_", "ARC_ASYNC_")
            )
            full_curl_tracking_stage = stage.startswith("SCRATCH_CURL_")
            pressure_wave_tracking_stage = stage.startswith(
                "SCRATCH_WAVE_ASYNC_"
            )
            is_full_curl_endpoint = bool(
                (pressure_wave_tracking_stage or full_curl_tracking_stage)
                and all(abs(float(target["fingers"].get(f, 0)) - SCRATCH_FULL_CURL_DEG) < 1e-6
                        for f in target.get('scratch_required_fingers', LONG_FINGERS))
            )
            finger_arrival_tolerance_deg = (
                SCRATCH_ACQUISITION_FINGER_ARRIVAL_TOL_DEG
                if acquisition_tracking_stage
                else (
                    SCRATCH_WAVE_FINGER_ARRIVAL_TOL_DEG
                    if pressure_wave_tracking_stage
                    else (
                        SCRATCH_CONTACT_FINGER_ARRIVAL_TOL_DEG
                        if contact_tracking_stage
                        else KEYFRAME_FINGER_ARRIVAL_TOL_DEG
                    )
                )
            )
            if is_full_curl_endpoint or target.get('early_exit'):
                # All twelve joints must reach the end pose, not just four MCPs.
                finger_arrival_tolerance_deg = KEYFRAME_FINGER_ARRIVAL_TOL_DEG
            arrival_timeout_s = (
                SCRATCH_CONTACT_KEYFRAME_TIMEOUT_S
                if (contact_tracking_stage or full_curl_tracking_stage)
                else KEYFRAME_ARRIVAL_TIMEOUT_S
            )
            required_settle_time_s = (
                SCRATCH_WAVE_KEYFRAME_SETTLE_TIME_S
                if pressure_wave_tracking_stage
                else (
                    SCRATCH_KEYFRAME_SETTLE_TIME_S
                    if contact_tracking_stage
                    else KEYFRAME_SETTLE_TIME_S
                )
            )
            # Do not leave an 8--10 degree deadband between arrival and relief.
            # Wait for actual stagnation, not just normal moving-drive lag.
            lift_eligible = bool(
                contact_tracking_stage
                and self.frame_uses_contact_j5_lift(target)
                and not pressure_wave_tracking_stage
                and max_arm_error_deg <= KEYFRAME_ARM_ARRIVAL_TOL_DEG
                and self.segment_arrival_wait_elapsed < arrival_timeout_s
                and self.contact_j5_lift_deg
                < CONTACT_J5_LIFT_MAX_DEG - 1e-9
            )
            self.contact_lift_stall_state, lift_due = contact_lift_stall_update(
                getattr(self, "contact_lift_stall_state", None),
                (self.segment_index, worst_finger, int(np.argmax(triplet_errors[worst_finger]))),
                self.segment_arrival_wait_elapsed, max_finger_error_deg,
                finger_arrival_tolerance_deg, lift_eligible,
            )
            if lift_due and self.contact_lift_geometry_allows(worst_finger, target_q_effective):
                self.contact_j5_lift_deg = min(
                    CONTACT_J5_LIFT_MAX_DEG,
                    self.contact_j5_lift_deg + CONTACT_J5_LIFT_STEP_DEG,
                )
                # Keep the TOTAL arrival wait: repeated assists cannot extend
                # this frame indefinitely or hide a persistent obstruction.
                self.contact_lift_stall_state = None
                self.segment_settle_elapsed = 0.0
                self.segment_wait_log_elapsed = 0.0
                print(
                    "[V102 J5 CONTACT LIFT]",
                    "stage=", stage,
                    "finger=", worst_finger,
                    "finger_joint=", HAND_MODEL.SUFFIXES[int(np.argmax(triplet_errors[worst_finger]))],
                    "finger_err_deg=", round(max_finger_error_deg, 3),
                    "arrival_tol_deg=", round(finger_arrival_tolerance_deg, 3),
                    "reason=STALLED_ABOVE_ARRIVAL_TOL",
                    "total_wait_s=", round(self.segment_arrival_wait_elapsed, 3),
                    "step_deg=", CONTACT_J5_LIFT_STEP_DEG,
                    "j5_lift_deg=", round(self.contact_j5_lift_deg, 3),
                )
                return

            arrived = bool(
                max_arm_error_deg <= KEYFRAME_ARM_ARRIVAL_TOL_DEG
                and max_finger_error_deg <= finger_arrival_tolerance_deg
            )

            if arrived:
                self.segment_settle_elapsed += float(dt)
            else:
                self.segment_settle_elapsed = 0.0

            if self.segment_wait_log_elapsed >= KEYFRAME_WAIT_LOG_PERIOD_S:
                self.log_stall_diagnostics(worst_finger, target["fingers"][worst_finger], "ARRIVAL_WAIT")
                self.segment_wait_log_elapsed = 0.0
                print(
                    "[V100 KEYFRAME ARRIVAL WAIT]",
                    "stage=", target.get("stage", "UNKNOWN"),
                    "arm_err_deg=", round(max_arm_error_deg, 3),
                    "arm_joint=", worst_arm_joint,
                    "finger_err_deg=", round(max_finger_error_deg, 3),
                    "finger=", worst_finger,
                    "finger_joint=", HAND_MODEL.SUFFIXES[int(np.argmax(triplet_errors[worst_finger]))],
                    "target_MCP_PIP_DIP_deg=", np.round(
                        HAND_MODEL.joint_targets(target["fingers"][worst_finger]), 3).tolist(),
                    "effective_MCP_PIP_DIP_deg=", [
                        round(float(d.GetTargetPositionAttr().Get()), 3)
                        for d in self.finger_triplet_drives[worst_finger]],
                    "actual_MCP_PIP_DIP_deg=", np.round(actual_triplets[worst_finger], 3).tolist(),
                    "finger_tol_deg=", round(finger_arrival_tolerance_deg, 3),
                    "settle_required_s=", round(required_settle_time_s, 3),
                    "timeout_s=", round(arrival_timeout_s, 3),
                    "contact_tracking=", contact_tracking_stage,
                    "j5_lift_deg=", round(self.contact_j5_lift_deg, 3),
                    "settle_s=", round(self.segment_settle_elapsed, 3),
                    "wait_s=", round(self.segment_arrival_wait_elapsed, 3),
                )

            if not keyframe_ready_to_advance(
                arrived, self.segment_settle_elapsed, required_settle_time_s
            ):
                if self.segment_arrival_wait_elapsed > arrival_timeout_s:
                    self.log_stall_diagnostics(worst_finger, target["fingers"][worst_finger], "TIMEOUT", force=True)
                    self.emergency_stop(
                        "KEYFRAME_TRACKING_TIMEOUT_"
                        + str(target.get("stage", "UNKNOWN"))
                        + "_arm_err_deg=" + str(round(max_arm_error_deg, 2))
                        + "_arm_joint=" + str(worst_arm_joint)
                        + "_finger_err_deg=" + str(round(max_finger_error_deg, 2))
                        + "_finger=" + str(worst_finger)
                    )
                return

            if target.get('early_exit'):
                clearances = {}
                for f in LONG_FINGERS:
                    tip = get_world_position(self.stage,FINGER_TIP_PATHS[f])
                    if tip is None:
                        self.emergency_stop('EARLY_EXIT_ACTUAL_TIP_UNAVAILABLE')
                        return
                    clearances[f] = float(final_surface_relation(tip,self.back_mesh_query,
                                      self.back_points,self.back_normals,self.back_tree)['signed'])
                joint_values = {('right_%s_%s'%(f,s)):math.radians(float(v))
                                for f in LONG_FINGERS for s,v in zip(HAND_MODEL.SUFFIXES,actual_triplets[f])}
                gap,link = HAND_COLLISION_MODEL.base_guard.clearance(
                    actual_q,target['fingers'],actual_joint_values=joint_values)
                base_contact_stop = False  # Planned metadata cannot prove contact.
                # A planned all-finger reserve is verified against the
                # experiment's +20 mm release target. The 1.5 mm value is
                # only a free-space safety margin and must not validate an
                # early-exit endpoint.
                required_exit_clearance = (
                    float(V103_ACTUAL_EXIT_ACCEPTANCE_M)
                    if target.get('early_exit', {}).get('all_finger_reserve')
                    else V103_SAFETY_CLEARANCE_M
                )
                blocked = ([] if base_contact_stop else [f for f,s in clearances.items()
                           if not math.isfinite(s) or s < required_exit_clearance])
                base_blocked = (
                    False
                    if (base_contact_stop
                        or bool(globals().get('BASE_PLANNING_GUARD_DISABLED', False)))
                    else (not math.isfinite(gap) or gap < V103_BASE_MIN_CLEARANCE_M)
                )
                record = dict(signed_mm={f:round(s*1000.,3) for f,s in clearances.items()},
                    required_skin_mm=required_exit_clearance*1000.,blocked_fingers=blocked,
                    base_gap_mm=round(gap*1000.,3),closest_base_link=link,base_blocked=base_blocked,
                    base_contact_stop=base_contact_stop,
                    actual_q_deg=np.round(np.degrees(actual_q),3).tolist(),
                    actual_joint_deg={f:np.round(actual_triplets[f],3).tolist() for f in LONG_FINGERS})
                print('[V103 ACTUAL EXIT CHECK]',record)
                if blocked or base_blocked:
                    self.emergency_stop('EARLY_EXIT_ACTUAL_CLEARANCE_NOT_VERIFIED|'+str(record))
                    return
                cycle_index = self.scratch_cycle_from_stage(stage)
                self.early_exit_joint_records[cycle_index] = {
                    'actual_joint_deg':{f:np.round(actual_triplets[f],3).tolist() for f in LONG_FINGERS},
                    'skin_clearance_mm':min(clearances.values())*1000.,'base_clearance_mm':gap*1000.,
                    'per_finger_signed_mm':record['signed_mm'],
                    'base_contact_stop':base_contact_stop}
                print('[V103 ACTUAL EARLY EXIT VERIFIED]',self.early_exit_joint_records[cycle_index])
            if stage.startswith('SCRATCH_WAVE_ASYNC_') and (
                    target.get('scratch_complete') or target.get('early_exit') or is_full_curl_endpoint):
                # Include the actually reached endpoint even for a short
                # 50-ms wave. Do not rely on a slower periodic logger firing.
                for f in LONG_FINGERS:
                    tip = get_world_position(self.stage, FINGER_TIP_PATHS[f])
                    if tip is None:
                        self.emergency_stop('WAVE_ENDPOINT_MARKER_MISSING|' + f)
                        return
                    rel = final_surface_relation(tip, self.back_mesh_query,
                        self.back_points, self.back_normals, self.back_tree)
                    lane = lane_relation_for_world_point(tip, self.current_lane_spec, f)
                    self.update_dynamic_depth_stats(f, rel['signed'], rel['distance'], lane, target['fingers'][f])
                cycle = self.scratch_cycle_from_stage(stage)
                self.scratch_completed_records[cycle] = dict(stage=stage, reached=True)
            removed_active = set(previous_active) - set(target_active)
            if is_full_curl_endpoint:
                cycle_index = self.scratch_cycle_from_stage(stage)
                record = {f: np.round(actual_triplets[f], 3).tolist() for f in LONG_FINGERS}
                self.full_curl_joint_records[cycle_index] = record
                print("[V102 ACTUAL FULL CURL MCP/PIP/DIP]", "cycle=", cycle_index,
                      "target_deg=", HAND_MODEL.ENDPOINT_DEG, "actual_deg=", record,
                      "max_error_deg=", round(max_finger_error_deg, 3))
            if set(target_active) != set(previous_active):
                print(
                    "[V100 ACTIVE COMMIT]",
                    "stage=",
                    target.get("stage", "UNKNOWN"),
                    "alpha=",
                    round(float(raw_alpha), 3),
                    "previous=",
                    previous_active,
                    "active=",
                    target_active,
                )
            self.active_fingers = target_active
            self.current_authorized_contact_fingers = list(target_authorized)
            for finger in removed_active:
                self.deep_adjust_latched[finger] = False
                self.contact_loss_time[finger] = 0.0
            self.segment_index += 1
            self.contact_lift_stall_state = None
            self.segment_elapsed = 0.0
            self.segment_arrival_wait_elapsed = 0.0
            self.segment_settle_elapsed = 0.0
            self.segment_wait_log_elapsed = 0.0
            if entering_pressure_wave:
                self.contact_j5_lift_deg = 0.0
            if not self.frame_uses_contact_j5_lift(target):
                self.contact_j5_lift_deg = 0.0

    def depth_feedback(self):
        q = self.read_actual_arm_q()
        hand = self.read_actual_finger_triplets_deg()
        depths = {}
        for finger in LONG_FINGERS:
            tip = get_world_position(self.stage, FINGER_TIP_PATHS[finger])
            if tip is None:
                raise RuntimeError('DEEP_ADJUST_MISSING_MARKER|' + finger)
            relation = final_surface_relation(tip, self.back_mesh_query, self.back_points,
                                              self.back_normals, self.back_tree)
            depths[finger] = -float(relation['signed'])
        return q, hand, depths

    @staticmethod
    def depth_joint_values(hand):
        return {'right_%s_%s' % (f, s): math.radians(float(v))
                for f in LONG_FINGERS for s, v in zip(HAND_MODEL.SUFFIXES, hand[f])}

    def depth_recovery_segment(self, q0, hand0, q1, hand1, limit, relief):
        """Measured 12-DOF start -> bounded wrist/rejoin segment; never curl-fit.

        Samples preserve tip-local marker offsets verified at the actual start.
        Full hand STL sweeps and base clearance remain mandatory, including the
        inactive pinky. This is geometric verification, not a force guarantee.
        """
        model = HAND_COLLISION_MODEL
        if model is None or not hasattr(model, 'base_guard') or self.current_lane_spec is None:
            return False, 'RECOVERY_GEOMETRY_UNAVAILABLE', {}
        if (not np.all(np.isfinite(q1)) or np.any(q1 < self.lower) or np.any(q1 > self.upper)
                or any(not np.all(np.isfinite(v)) or np.any(np.asarray(v) < 0.)
                       or np.any(np.asarray(v) > HAND_MODEL.LIMITS_DEG) for v in hand1.values())):
            return False, 'RECOVERY_JOINT_LIMIT', {}
        q_span = float(np.max(np.abs(np.degrees(q1-q0))))
        h_span = max(float(np.max(np.abs(hand1[f]-hand0[f]))) for f in LONG_FINGERS)
        if q_span > 3. or h_span > 3.:
            return False, ('RECOVERY_REJOIN_TOO_FAR|arm_delta_deg='
                +str(np.round(np.degrees(q1-q0),3).tolist())+'|hand_delta_deg='
                +str({f:np.round(hand1[f]-hand0[f],3).tolist() for f in LONG_FINGERS})), {}
        chains = getattr(self, '_depth_hand_chains', None)
        if chains is None:
            chains = self._depth_hand_chains = HAND_MODEL.hand_chains(COLLISION_URDF)
        p0, r0 = self.solver.compute_forward_kinematics(PALM_FRAME, q0)
        p0, r0 = np.asarray(p0), np.asarray(r0)
        joints0 = self.depth_joint_values(hand0)
        bias = {}
        for f in LONG_FINGERS:
            actual = get_world_position(self.stage, FINGER_TIP_PATHS[f])
            if actual is None:
                return False, 'RECOVERY_MARKER_MISSING|' + f, {}
            bias[f] = r0.T @ (np.asarray(actual)-p0) - HAND_MODEL.tip_local(chains[f], joints0)
            if np.linalg.norm(bias[f]) > .001:
                return False, 'RECOVERY_MARKER_FK_MISMATCH|' + f, {}
        authorized = self._depth_recovery.authorized
        prior_probes = None
        first_depths = None
        n = max(7, int(math.ceil(max(q_span, h_span)/.15)) + 1)
        for index, alpha in enumerate(np.linspace(0., 1., n)):
            q = q0 + alpha*(q1-q0)
            hand = {f: hand0[f]+alpha*(hand1[f]-hand0[f]) for f in LONG_FINGERS}
            joints = self.depth_joint_values(hand)
            if not BASE_PLANNING_GUARD_DISABLED:
                gap, link = model.base_guard.clearance(q, {}, actual_joint_values=joints)
                if not math.isfinite(gap) or gap < V103_BASE_MIN_CLEARANCE_M:
                    return False, 'RECOVERY_BASE_MARGIN|' + str(link), {}
            p, r = self.solver.compute_forward_kinematics(PALM_FRAME, q)
            p, r = np.asarray(p), np.asarray(r)
            if not PALM_POLICY.angle_ok(palm_back_alignment(
                    self.arcs, r, self.current_lane_spec['normal'])['angle_deg']):
                return False, 'RECOVERY_PALM_ANGLE', {}
            root_state = root_base_axis_alignment(self.arcs, p, r)
            root_angle = float(root_state['angle_deg'])
            if (not root_state.get('available', False) or not math.isfinite(root_angle)
                    or root_angle > ROOT_BASE_AXIS_HARD_DEG):
                return False, 'RECOVERY_ROOT_AXIS', {}
            depths = {}
            for f in LONG_FINGERS:
                tip = p + r @ (HAND_MODEL.tip_local(chains[f], joints) + bias[f])
                rel = final_surface_relation(tip, self.back_mesh_query, self.back_points,
                                             self.back_normals, self.back_tree)
                depth = depths[f] = -float(rel['signed'])
                cap = limit if f in authorized else -INACTIVE_PENETRATION_M
                if not math.isfinite(depth) or depth > cap:
                    return False, 'RECOVERY_DEPTH|' + f, depths
                if f in authorized:
                    lane = lane_relation_for_world_point(tip, self.current_lane_spec, f)
                    if not lane['valid']:
                        return False, 'RECOVERY_LANE|' + f, depths
                    if relief and first_depths is not None and depth > max(.004, first_depths[f]) + .0001:
                        return False, 'RECOVERY_INCREASES_DEPTH|' + f, depths
            if first_depths is None:
                first_depths = depths.copy()
            probes = model.world_probe_points(self.solver, q, {}, actual_joint_values=joints)
            args = (self.back_points, self.back_normals, self.back_tree, self.back_mesh_query)
            if prior_probes is None:
                safe, reason = model.check_static_points(probes, *args,
                                                       allowed_contact_fingers=authorized)
            else:
                safe, reason = model.check_swept_points(prior_probes, probes, *args,
                    step_index=index, path_fraction=float(alpha), allowed_contact_fingers=authorized)
            if not safe:
                return False, reason, depths
            prior_probes = probes
        return True, 'PASS', depths

    def command_depth_recovery(self, q, hand):
        # Physical triplets deliberately bypass the scalar curl model here.
        for i, name in enumerate(ARM_JOINTS):
            set_drive_deg(self.arm_drives[name], float(np.degrees(q[i])))
        for finger, values in hand.items():
            for drive, value in zip(self.finger_triplet_drives[finger], values):
                set_drive_deg(drive, float(value))
        self.current_arm_command = np.array(q, copy=True)
        print('[V103 WRIST RECOVERY COMMAND]', 'q_deg=', np.round(np.degrees(q), 3),
              'phase=', self._depth_recovery.phase, 'curl_clock_paused=True')

    def recover_wave_tracking(self, finger, arm_error, finger_error):
        # Only a small residual at an already authorized contact is eligible.
        # A free-space stall or large tracking error must not trigger motion.
        key = (int(self.segment_index), str(self.trajectory[self.segment_index]['stage']))
        authorized = set(self.active_fingers) | set(self.current_authorized_contact_fingers)
        if (finger not in authorized
                or arm_error > SCRATCH_WAVE_TRACKING_HOLD_ARM_DEG * 2.0
                or finger_error > SCRATCH_WAVE_TRACKING_HOLD_FINGER_DEG * 1.5
                or getattr(self, '_wave_recovery_attempt_key', None) == key):
            self.emergency_stop('WAVE_TRACKING_STALLED|finger='+finger)
            return
        try:
            _, _, depths = self.depth_feedback()
            if not math.isfinite(depths[finger]) or not 0. <= depths[finger] <= .004:
                self.emergency_stop('WAVE_TRACKING_STALLED|finger='+finger)
                return
            # Capture BEFORE hold_actual_state changes the command fields.
            snapshot = (np.asarray(self.current_arm_command).copy(),
                        dict(self.current_finger_commands), float(self.segment_elapsed))
            self._wave_recovery_attempt_key = key
            self.begin_depth_recovery(finger, tracking_snapshot=snapshot)
        except Exception as exc:
            self.emergency_stop('WAVE_TRACKING_RECOVERY_ERROR|' + repr(exc))

    def begin_depth_recovery(self, triggering_finger, tracking_snapshot=None):
        if getattr(self, '_depth_recovery', None) is not None:
            return
        count = getattr(self, '_depth_recovery_count', 0) + 1
        if count > 3:
            stage = str(self.current_stage)
            cycle = self.scratch_cycle_from_stage(stage)
            scratch_stage = stage.startswith((
                'ARC_SYNC_', 'ARC_ASYNC_', 'SCRATCH_WAVE_ASYNC_',
                'SCRATCH_CURL_', 'FINGER_CURL_RELEASE_',
            ))
            if scratch_stage:
                key = (stage, cycle, str(triggering_finger))
                self._deep_adjust_record_only_keys.add(key)
                # Start a fresh counter for a later stage; this stage/finger
                # is suppressed by the key above, so it cannot spin.
                self._depth_recovery_count = 0
                print(
                    '[V103 DEEP_ADJUST LIMIT RECORDED CONTINUE]',
                    'stage=', stage,
                    'cycle=', cycle,
                    'finger=', triggering_finger,
                    'action=RECORD_ONLY_CONTINUE_TRAJECTORY',
                )
            else:
                self.emergency_stop('DEEP_ADJUST_REPEATED_LIMIT')
            return
        try:
            q, hand, _ = self.depth_feedback()
            authorized = (set(self.active_fingers) | set(self.current_authorized_contact_fingers)
                          | {triggering_finger})
            frozen = {f: np.clip(v, 0., HAND_MODEL.LIMITS_DEG) for f, v in hand.items()}
            self._depth_recovery = DEPTH_CONTROL.DepthRecovery(
                q, frozen, authorized, force_relief=tracking_snapshot is not None,
                joint_limits=(self.lower,self.upper,np.asarray(HAND_MODEL.LIMITS_DEG)),
                triggering_finger=triggering_finger)
            # Route the independent module's diagnostics through the active
            # bridge logger, which only replaces this script's global print.
            DEPTH_CONTROL.__dict__['print'] = print
            for f, drives in self.finger_triplet_drives.items():
                for j, drive in enumerate(drives):
                    prim = drive.GetPrim()
                    velocity = prim.GetAttribute('physxJoint:maxJointVelocity')
                    print('[V103 RECOVERY DRIVE]', f, j,
                          'stiffness=',drive.GetStiffnessAttr().Get(),
                          'damping=',drive.GetDampingAttr().Get(),
                          'maxForce=',drive.GetMaxForceAttr().Get(),
                          'maxJointVelocity=',velocity.Get() if velocity else None)
            self._depth_recovery.tracking_snapshot = tracking_snapshot
            self._depth_recovery_count = count
            self.hold_actual_state()
            print('[V103 WRIST RECOVERY START]', 'event=', count,
                  'actual_12_joints_held=True; original trajectory clock paused')
        except Exception as exc:
            self._depth_recovery = None
            self.emergency_stop('DEEP_ADJUST_START_ERROR|' + repr(exc))

    def update_depth_recovery(self, dt):
        def rejoin():
            snapshot = getattr(self._depth_recovery, 'tracking_snapshot', None)
            if snapshot is not None:
                # The snapshot's finger values are the commands that were
                # being interpolated when the over-depth was detected.  They
                # can be ahead of the measured physical triplets by several
                # degrees.  Replaying them during rejoin was the source of a
                # relief/rejoin oscillation: the old target immediately drove
                # ring back over 4 mm and eventually exhausted the step limit.
                # Keep the measured curl frozen and rejoin only the arm/wrist;
                # the normal trajectory clock remains paused until this pose
                # is tracked and the runtime planner cap supplies margin for
                # the subsequent interpolation.
                recovery_hand = getattr(self._depth_recovery, 'hand', {})
                return (snapshot[0].copy(),
                        {f: np.asarray(recovery_hand.get(
                            f, HAND_MODEL.joint_targets(snapshot[1][f])),
                            dtype=float).copy() for f in LONG_FINGERS})
            target = self.trajectory[self.segment_index]
            return (np.asarray(target['q']).copy(),
                    {f: np.asarray(HAND_MODEL.joint_targets(target['fingers'][f])) for f in LONG_FINGERS})
        try:
            recovery = self._depth_recovery
            if recovery.goal is not None:
                # Read resolved USD targets before feedback: distinguish an
                # overwritten drive from a commanded joint failing to track.
                goal_q, goal_hand = recovery.goal
                for i, name in enumerate(ARM_JOINTS):
                    value = self.arm_drives[name].GetTargetPositionAttr().Get()
                    expected = float(np.degrees(goal_q[i]))
                    if value is None or not math.isfinite(float(value)) or abs(float(value)-expected) > 1.e-4:
                        raise RuntimeError('RECOVERY_DRIVE_TARGET_CHANGED|'+name+'|expected='+str(expected)+'|actual='+str(value))
                for finger, targets in goal_hand.items():
                    for index, (drive, expected) in enumerate(zip(self.finger_triplet_drives[finger], targets)):
                        value = drive.GetTargetPositionAttr().Get()
                        if value is None or not math.isfinite(float(value)) or abs(float(value)-float(expected)) > 1.e-4:
                            raise RuntimeError('RECOVERY_DRIVE_TARGET_CHANGED|'+finger+'|joint_index='+str(index)+'|expected='+str(expected)+'|actual='+str(value))
            now = time.perf_counter()
            previous_wall = getattr(recovery, '_adapter_wall_at', None)
            recovery._adapter_wall_at = now
            wall_dt = float(dt) if previous_wall is None else max(0.0, now-previous_wall)
            result = recovery.tick(dt, self.depth_feedback, self.depth_recovery_segment,
                                   self.command_depth_recovery, rejoin, wall_dt=wall_dt)
            if result == 'DONE':
                target = self.trajectory[self.segment_index]
                snapshot = getattr(recovery, 'tracking_snapshot', None)
                # A pressure-wave recovery that has already returned to the
                # measured safe band must not replay the old wave command. It
                # was the source of RECOVERY_DEPTH rejoin rejections: the
                # stale target immediately pushed ring back over 4 mm. The
                # scratch depth endpoint has already been recorded, so switch
                # directly to the exact measured release tail and then Home.
                cycle = self.scratch_cycle_from_stage(self.current_stage)
                if (snapshot is not None and cycle is not None and
                        cycle in self.scratch_completed_records):
                    q_now, hand_now, depths_now = self.depth_feedback()
                    signed_now = {f: -float(depths_now[f]) for f in LONG_FINGERS}
                    self._depth_recovery = None
                    print('[V103 WRIST RECOVERY SAFE EXIT]',
                          'cycle=', cycle,
                          'depth_mm=', {f: round(float(depths_now[f]) * 1000., 3)
                                        for f in LONG_FINGERS},
                          'action=MEASURED_RELEASE_TAIL')
                    self.start_measured_release_tail(
                        'RECOVERY_SAFE_EXIT', cycle, signed_now, None)
                    return
                self.current_finger_commands = dict(target['fingers'] if snapshot is None else snapshot[1])
                if snapshot is None:
                    self.contact_j5_lift_deg = 0.
                self.segment_elapsed = float(target['duration']) if snapshot is None else snapshot[2]
                self.segment_arrival_wait_elapsed = self.segment_settle_elapsed = 0.
                self._wave_tracking_hold_key = None
                self._depth_recovery = None
                print('[V103 WRIST RECOVERY RESUME]', 'steps=', recovery.steps,
                      'actual_rejoin_verified=True; raw depth peaks retained')
        except Exception as exc:
            # In a scratch wave, an attempted depth correction can be
            # impossible even though the measured trajectory is still useful.
            # Record that local correction failure and resume the same
            # trajectory; do not turn it into an emergency stop for this point.
            # The stage/finger key suppresses an immediate retry loop while
            # preserving the raw over-depth in the samples and diagnostics.
            reason = str(exc)
            recovery = getattr(self, '_depth_recovery', None)
            stage = str(getattr(self, 'current_stage', ''))
            triggering_finger = str(getattr(recovery, 'triggering_finger', '') or '')
            cycle = self.scratch_cycle_from_stage(stage)
            record_only_key = (stage, cycle, triggering_finger)
            record_only_stage = stage.startswith((
                'SCRATCH_WAVE_ASYNC_', 'ARC_SYNC_', 'ARC_ASYNC_',
            ))
            if ('DEEP_ADJUST_NO_SAFE_WRIST_STEP' in reason and
                    triggering_finger and record_only_stage):
                self._deep_adjust_record_only_keys.add(record_only_key)
                self._depth_recovery = None
                print(
                    '[V103 DEEP_ADJUST NO SAFE STEP RECORDED CONTINUE]',
                    'stage=', stage,
                    'cycle=', cycle,
                    'finger=', triggering_finger,
                    'action=RESUME_TRAJECTORY',
                )
                return
            self._depth_recovery = None
            self.emergency_stop('DEEP_ADJUST_FAILED|' + repr(exc))

    def contact_lift_geometry_allows(self, finger, target_q_effective):
        """Relief is for an authorized near-surface finger, not a free-air stall.

        This geometric guard is not a contact-force measurement. Existing
        runtime penetration/lane checks still apply to every finger.
        """
        tip = get_world_position(self.stage, FINGER_TIP_PATHS[finger])
        if tip is None or self.current_lane_spec is None:
            return False
        relation = final_surface_relation(
            tip, self.back_mesh_query, self.back_points, self.back_normals, self.back_tree)
        lane = lane_relation_for_world_point(tip, self.current_lane_spec, finger)
        distance, signed = float(relation["distance"]), float(relation["signed"])
        if (not math.isfinite(distance) or not math.isfinite(signed)
                or not lane["valid"] or signed < RUNTIME_HARD_PENETRATION_M
                or not self.transition_contact_is_authorized(finger, lane, distance)):
            return False
        step = min(CONTACT_J5_LIFT_STEP_DEG, CONTACT_J5_LIFT_MAX_DEG - self.contact_j5_lift_deg)
        joint = UsdPhysics.RevoluteJoint(self.arm_drives["joint5"].GetPrim())
        next_deg = float(np.degrees(target_q_effective[4])) + step
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        return bool(lower is not None and upper is not None and lower <= next_deg <= upper)

    def transition_contact_is_authorized(self, finger, lane_state, distance):
        """Unify offline frame authorization with real PhysX contact timing.

        PhysX may report distal contact one interpolation frame before the
        point-cloud sampler labels the fingertip as active.  During the known
        synchronized scratch arc, treat a planned working finger as acquiring
        only when its real tip is already in its own target region and within
        the 12 mm near-surface band.  This is not a blanket collision waiver.
        """
        frame_authorized = finger in set(
            getattr(self, "current_authorized_contact_fingers", [])
        )
        synchronized_acquisition = self.current_stage.startswith(
            (
                "ARC_SYNC_PRECONTACT_",
                "ARC_SYNC_CONTACT_",
                "ARC_ASYNC_PRECONTACT_",
                "ARC_ASYNC_CONTACT_",
                "SCRATCH_WAVE_ASYNC_",
            )
        )
        planned_sync_finger = bool(
            synchronized_acquisition
            and finger in set(getattr(self, "planned_working_fingers", []))
        )
        return bool(
            (frame_authorized or planned_sync_finger)
            and lane_state is not None
            and bool(lane_state["valid"])
            and float(distance) <= CONTACT_NEAR_OFFSET_M
        )

    @staticmethod
    def frame_uses_contact_j5_lift(frame):
        stage = str(frame.get("stage", ""))
        return bool(
            stage.startswith("ARC_SYNC_")
            or stage.startswith("ARC_ASYNC_")
            or stage.startswith("SCRATCH_WAVE_ASYNC_")
            or stage.startswith("SCRATCH_CURL_")
            or stage.startswith("FINGER_CURL_RELEASE_")
            or stage == "FULL_SCRATCH"
            or stage.startswith("FULL_SCRATCH_CONTACT_CYCLE_")
        )

    @staticmethod
    def scratch_cycle_from_stage(stage):
        marker = "_CYCLE_"
        stage = str(stage)
        if marker not in stage:
            return None
        tail = stage.rsplit(marker, 1)[1]
        digits = ""
        for char in tail:
            if not char.isdigit():
                break
            digits += char
        return int(digits) if digits else None

    def update_dynamic_depth_stats(
        self,
        finger,
        signed,
        distance,
        lane_state,
        command,
    ):
        """Track actual PhysX fingertip depth during each comfort-wave cycle."""
        stage = str(self.current_stage)
        if not stage.startswith("SCRATCH_WAVE_ASYNC_"):
            return
        cycle_index = self.scratch_cycle_from_stage(stage)
        if cycle_index is None:
            return
        if not math.isfinite(float(signed)) or not math.isfinite(float(distance)):
            return
        if finger != "pinky" and (
            lane_state is None
            or not bool(lane_state.get("valid", False))
            or float(distance) > CONTACT_NEAR_OFFSET_M
        ):
            return

        cycle_stats = self.dynamic_depth_stats.setdefault(cycle_index, {})
        stats = cycle_stats.setdefault(
            finger,
            {
                "min_signed": float("inf"),
                "stage": None,
                "command_deg": None,
                "samples": 0,
                "last_logged_depth_mm": 0.0,
            },
        )
        stats["samples"] += 1
        if float(signed) >= float(stats["min_signed"]):
            return

        stats["min_signed"] = float(signed)
        stats["stage"] = stage
        stats["command_deg"] = float(command)
        depth_mm = max(0.0, -float(signed)) * 1000.0
        if depth_mm >= float(stats["last_logged_depth_mm"]) + 0.25:
            stats["last_logged_depth_mm"] = depth_mm
            print(
                "[V102 ACTUAL DEPTH NEW MAX]",
                "cycle=", int(cycle_index),
                "finger=", finger,
                "depth_mm=", round(depth_mm, 2),
                "stage=", stage,
                "command_progress_pct=", round(float(command), 2),
            )
            try:
                commanded_q = self.current_arm_command
                actual_q = self.read_actual_arm_q()
                actual_triplets = self.read_actual_finger_triplets_deg()
                predicted = finger_surface_relation_at_state(
                    self.solver, self.arcs, commanded_q, self.current_finger_commands, finger,
                    self.back_points, self.back_normals, self.back_tree, self.back_mesh_query)
                print('[V103 DEPTH TRACKING]', 'finger=', finger,
                      'commanded_marker_depth_mm=', round(max(0.,-float(predicted['signed']))*1000.,3),
                      'actual_marker_depth_mm=', round(depth_mm,3),
                      'arm_error_deg=', np.round(np.degrees(actual_q-commanded_q),3).tolist(),
                      'command_triplet_deg=', HAND_MODEL.joint_targets(command),
                      'actual_triplet_deg=', actual_triplets[finger],
                      'scope=command_FK_vs_actual_marker_on_rest_surface')
            except Exception as exc:
                print('[V103 DEPTH TRACKING UNAVAILABLE]', repr(exc))

    def record_actual_tip_trace(self):
        """Capture live USD fingertip marker positions for the current frame.

        The planner's FK arcs are only a proposal.  During execution the
        actual drive state can lag, collide, or be corrected by the wrist, so
        the trustworthy path is the world pose of each live tip-link prim.
        Samples are collected at the existing safety-check cadence and later
        drawn by ``draw_actual_tip_trajectories``.
        """
        if not self.executing:
            return
        sim_time = float(getattr(self, "telemetry_sim_time", 0.0))
        if sim_time <= float(getattr(self, "_actual_tip_trace_last_sim", -1.0)) + 1e-9:
            return
        stage_name = str(getattr(self, "current_stage", "UNKNOWN"))
        scratch = _actual_trace_is_scratch_stage(stage_name)
        active = set(getattr(self, "active_fingers", ()))
        for finger in LONG_FINGERS:
            tip = get_world_position(self.stage, FINGER_TIP_PATHS[finger])
            if tip is None:
                continue
            point = np.asarray(tip, dtype=np.float64)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                continue
            self._actual_tip_trace.setdefault(finger, []).append({
                "sim_time_s": sim_time,
                "stage": stage_name,
                "world_xyz_m": [float(v) for v in point],
                "scratch": bool(scratch),
                "active": bool(finger in active),
            })
        self._actual_tip_trace_last_sim = sim_time

    def record_actual_contact_location(self, finger, tip, elapsed):
        cycle = self.scratch_cycle_from_stage(self.current_stage)
        if cycle is None:
            return
        relation = final_surface_relation(tip,self.back_mesh_query,self.back_points,
                                          self.back_normals,self.back_tree)
        records = getattr(self,'_actual_contact_locations',None)
        if records is None:
            records = self._actual_contact_locations = {}
        row = records.setdefault((cycle,finger),{'weighted':np.zeros(3),'time':0.,'previous':None})
        point = np.asarray(tip,dtype=float)
        previous = row['previous']
        row['previous'] = point.copy()
        # Geometric moving contact, not a force-sensor contact claim.
        if previous is None or elapsed <= 0 or float(relation['signed']) >= 0:
            return
        normal = np.asarray(self.current_lane_spec['normal'])
        motion = point-previous
        tangent = motion-np.dot(motion,normal)*normal
        if np.linalg.norm(tangent)/elapsed < .001:
            return
        row['weighted'] += point*elapsed
        row['time'] += elapsed

    def report_dynamic_depth_cycle(self, cycle_index):
        cycle_index = int(cycle_index)
        if cycle_index in self.dynamic_depth_reported_cycles:
            return
        cycle_stats = self.dynamic_depth_stats.get(cycle_index, {})
        if not cycle_stats:
            return

        max_depth_mm = {}
        minimum_signed_mm = {}
        deepest_stage = {}
        sample_count = {}
        for finger in LONG_FINGERS:
            stats = cycle_stats.get(finger)
            if stats is None:
                max_depth_mm[finger] = None
                minimum_signed_mm[finger] = None
                deepest_stage[finger] = None
                sample_count[finger] = 0
                continue
            max_depth_mm[finger] = round(
                max(0.0, -float(stats["min_signed"])) * 1000.0, 2
            )
            minimum_signed_mm[finger] = round(
                float(stats["min_signed"]) * 1000.0, 3
            )
            deepest_stage[finger] = stats["stage"]
            sample_count[finger] = int(stats["samples"])

        optional = set(getattr(self,'optional_scratch_fingers',{}).get(cycle_index,[]))
        relaxed = set(getattr(self,'relaxed_scratch_fingers',{}).get(cycle_index,[])) | optional
        required = list(LONG_FINGERS)
        primary = getattr(self, 'primary_scratch_fingers', {}).get(cycle_index)
        quality_required = [f for f in required if f in V103_MANDATORY_QUALITY_FINGERS]
        measured = [max_depth_mm[f] for f in quality_required if max_depth_mm[f] is not None]
        spread_mm = (
            round(max(measured) - min(measured), 2)
            if measured and len(measured) == len(quality_required)
            else None
        )
        print("")
        print("==============================================")
        print("V103 ACTUAL DYNAMIC DEPTH SUMMARY")
        print("Measurement: tip-link marker vs cached REST BackSurface; wave-only partial/full samples.")
        print("Not deformable-skin indentation or collision-manifold penetration.")
        print("==============================================")
        print("Cycle:", cycle_index)
        print("Maximum penetration mm:", max_depth_mm)
        print("Closest signed surface value mm:", minimum_signed_mm)
        print("Four-finger max-depth spread mm (diagnostic):", spread_mm)
        print("Primary depth anchor (optimizer preference only):", primary)
        print("Mandatory quality fingers:", quality_required)
        print("Relaxed quality fingers (trajectory still contains them):", sorted(relaxed))
        print("Deepest stage:", deepest_stage)
        print("Near-surface samples:", sample_count)
        actual_locations = {}
        for finger in required:
            row = getattr(self,'_actual_contact_locations',{}).get((cycle_index,finger))
            if row is not None and row['time'] > 0:
                center = row['weighted']/row['time']
                delta = center-np.asarray(self.current_lane_spec['anchor'])
                normal = np.asarray(self.current_lane_spec['normal'])
                tangent = delta-np.dot(delta,normal)*normal
                actual_locations[finger] = dict(world_xyz_m=center.tolist(),
                    tangent_offset_xyz_mm=(tangent*1000.).tolist(),
                    error_mm=float(np.linalg.norm(tangent)*1000.))
        middle_error = actual_locations.get('middle',{}).get('error_mm')
        strict_location_ok = (not V103_STRICT_POINT_TEST or
                              (middle_error is not None and middle_error <= V103_MIDDLE_ANCHOR_MAX_M*1000.))
        base_stop_verified = cycle_index in getattr(self, 'base_contact_stop_records', {})
        print('[V103 ACTUAL CONTACT LOCATION]',actual_locations,
              'strict_point_pass=',strict_location_ok,
              'scope=time_weighted_moving_marker_vs_rest_surface; not force contact')
        raw_depths = {f: max(0., -float(row['min_signed']))
                      if math.isfinite(float(row['min_signed'])) else float('nan')
                      for f, row in cycle_stats.items()}
        quality_angle_deg = float('nan')
        try:
            if self.current_lane_spec is not None:
                _, quality_palm_r = self.solver.compute_forward_kinematics(
                    PALM_FRAME, self.read_actual_arm_q()
                )
                quality_angle_deg = float(palm_back_alignment(
                    self.arcs, np.asarray(quality_palm_r, dtype=np.float64),
                    np.asarray(self.current_lane_spec['normal'], dtype=np.float64),
                )['angle_deg'])
        except Exception:
            quality_angle_deg = float('nan')
        quality_minimums_m = {
            finger: v103_quality_min_depth_m(quality_angle_deg, finger)
            for finger in V103_MANDATORY_QUALITY_FINGERS
        }
        scratch_completed = cycle_index in getattr(self, 'scratch_completed_records', {})
        imitation_valid = bool(strict_location_ok and scratch_completed
                               and RELEASE_POLICY.quality(
                                   raw_depths, sample_count, relaxed,
                                   minimums_m=quality_minimums_m))
        quality_reason = (
            "PASS"
            if imitation_valid
            else (
                ("SCRATCH_COMPLETION_OR_DEPTH_BAND_OR_STRICT_LOCATION_NOT_MET")
                + "|scratch_completed=" + str(scratch_completed)
                + "|minimum_depth_mm="
                + str(round(min(quality_minimums_m.values()) * 1000.0, 3)) + '|maximum_depth_mm='
                + str(round(V103_QUALITY_DEPTH_MAX_M * 1000.0, 3))
                + "|maximum_spread_mm="
                + 'diagnostic_only'
                + "|full_curl_verified=" + str(cycle_index in self.full_curl_joint_records)
                + "|early_exit_verified=" + str(cycle_index in getattr(self,'early_exit_joint_records',{}))
                + "|base_contact_stop_verified=" + str(base_stop_verified)
            )
        )
        self.imitation_cycle_quality[cycle_index] = {
            "valid": imitation_valid,
            "scratch_completed": scratch_completed,
            "raw_max_depth_m": raw_depths,
            "reason": quality_reason,
            "max_depth_mm": dict(max_depth_mm),
            "spread_mm": spread_mm,
            "required_fingers": list(required),
            "optional_fingers": sorted(optional),
            "relaxed_fingers": sorted(relaxed),
            "quality_required_fingers": list(quality_required),
            "primary_depth_anchor": primary,
            "hand_model_id": HAND_MODEL.MODEL_ID,
            "actual_full_curl_mcp_pip_dip_deg": self.full_curl_joint_records.get(cycle_index),
            "actual_early_exit": getattr(self,'early_exit_joint_records',{}).get(cycle_index),
            "actual_base_contact_stop": getattr(self,'base_contact_stop_records',{}).get(cycle_index),
        }
        print(
            "V103 IMITATION SAMPLE QUALITY:",
            "VALID" if imitation_valid else "REJECT",
            quality_reason,
        )
        print("Negative signed depth means penetration; summary reports positive mm.")
        self.dynamic_depth_reported_cycles.add(cycle_index)

    def safety_check(self, elapsed):
        if not self.executing:
            return

        # Record the live USD marker path before any safety response can
        # freeze the pose.  This path is later rendered separately from the
        # planned V100 arcs and is the source used for imitation data review.
        try:
            self.record_actual_tip_trace()
        except Exception as exc:
            if not self._actual_tip_trace_error_logged:
                print("[V103 ACTUAL TIP TRACE UNAVAILABLE]", repr(exc))
                self._actual_tip_trace_error_logged = True

        # Measured rates use accumulated physics dt; wall rate exposes slow
        # simulation separately. Reset at stage transitions, never bridge pauses.
        try:
            stamp = float(getattr(self, 'telemetry_sim_time', 0.))
            wall = time.perf_counter()
            triplets = self.read_actual_finger_triplets_deg()
            angles = np.concatenate([np.degrees(self.read_actual_arm_q())] + [
                np.asarray(triplets[f]) for f in LONG_FINGERS])
            previous = getattr(self, '_speed_sample', None)
            if previous is not None and previous[0] == self.current_stage:
                delta = stamp-previous[1]
                if delta > 0 and wall > previous[2]:
                    rates = (angles-previous[3])/delta
                    if wall-getattr(self,'_speed_log_wall',0.) >= .5:
                        print('[V103 MEASURED SPEED]', 'stage=',self.current_stage,
                              'dt_sim_s=',round(delta,4),
                              'sim_per_wall=',round(delta/(wall-previous[2]),3),
                              'arm_deg_s=',np.round(rates[:5],2).tolist(),
                              'finger_MCP_PIP_DIP_deg_s=',np.round(rates[5:].reshape(4,3),2).tolist())
                        self._speed_log_wall=wall
            self._speed_sample=(self.current_stage,stamp,wall,angles)
        except Exception as exc:
            if not getattr(self,'_speed_error_logged',False):
                print('[V103 MEASURED SPEED UNAVAILABLE]',repr(exc))
                self._speed_error_logged=True

        actual_contact_palm_angle = None
        for finger in LONG_FINGERS:
            tip = get_world_position(
                self.stage,
                FINGER_TIP_PATHS[
                    finger
                ],
            )

            if tip is None:
                continue

            if self.current_stage.startswith('SCRATCH_WAVE_ASYNC_') and self.current_lane_spec is not None:
                self.record_actual_contact_location(finger, tip, elapsed)

            relation = final_surface_relation(
                tip,
                self.back_mesh_query,
                self.back_points,
                self.back_normals,
                self.back_tree,
            )

            signed = float(
                relation[
                    "signed"
                ]
            )

            distance = float(
                relation[
                    "distance"
                ]
            )

            source = str(
                relation[
                    "source"
                ]
            )

            command = float(
                self.current_finger_commands.get(
                    finger,
                    0.0,
                )
            )

            is_active = (finger in self.active_fingers)

            lane_state = None
            if self.current_lane_spec is not None:
                lane_state = lane_relation_for_world_point(
                    tip,
                    self.current_lane_spec,
                    finger,
                )

            self.update_dynamic_depth_stats(
                finger,
                signed,
                distance,
                lane_state,
                command,
            )

            # A new finger may enter compliant contact during interpolation.
            # Runtime geometry, rather than an offline frame label alone,
            # decides whether the early physical contact is intended.
            transition_contact_authorized = self.transition_contact_is_authorized(
                finger,
                lane_state,
                distance,
            )
            contact_authorized = bool(
                is_active or transition_contact_authorized
            )

            if (contact_authorized and self.current_lane_spec is not None
                    and distance <= WARM_CONTACT_MAX_DISTANCE_M):
                if actual_contact_palm_angle is None:
                    _, actual_palm_r = self.solver.compute_forward_kinematics(
                        PALM_FRAME, self.read_actual_arm_q(),
                    )
                    actual_contact_palm_angle = float(palm_back_alignment(
                        self.arcs, actual_palm_r, self.current_lane_spec["normal"],
                    )["angle_deg"])
                # Palm/back alignment is deliberately not a runtime motion
                # veto.  Earlier V103 treated the selected 60-degree tier as
                # a hard emergency limit, so a harmless 60.023-degree sample
                # could abort an otherwise valid four-finger scratch.  The
                # live safety authority is PhysX collision/contact plus the
                # measured target-lane and release checks below.  Keep the
                # angle as telemetry for diagnosis; do not stop on a soft
                # alignment preference.
                if (math.isfinite(actual_contact_palm_angle)
                        and actual_contact_palm_angle > palm_alignment_limit_deg()):
                    print(
                        "[V103 PALM ALIGNMENT DIAGNOSTIC] angle=",
                        round(actual_contact_palm_angle, 3),
                        "limit=", palm_alignment_limit_deg(),
                        "runtime_veto=DISABLED",
                    )

            if (
                is_active
                and lane_state is not None
                and distance <= WARM_CONTACT_MAX_DISTANCE_M
                and not (self.current_stage.startswith('SCRATCH_WAVE_ASYNC_')
                         and signed > V103_SAFETY_CLEARANCE_M)
            ):
                if not lane_state["valid"]:
                    if lane_state["patch_radius"] > TARGET_PATCH_RADIUS_HARD_M:
                        reason = (
                            "ACTIVE_FINGER_LEFT_TARGET_PATCH_"
                            + finger
                            + "_radius="
                            + str(round(lane_state["patch_radius"] * 1000.0, 2))
                            + "mm_limit="
                            + str(round(TARGET_PATCH_RADIUS_HARD_M * 1000.0, 1))
                            + "mm_cross="
                            + str(round(lane_state["cross_error"] * 1000.0, 2))
                            + "mm"
                        )
                    else:
                        reason = (
                            "ACTIVE_FINGER_LEFT_TARGET_LANE_"
                            + finger
                            + "_cross=" + str(round(lane_state["cross_error"] * 1000.0, 2))
                            + "mm_click_distance_diag=" + str(round(lane_state["patch_radius"] * 1000.0, 2))
                            + "mm"
                        )
                    self.emergency_stop(
                        reason
                    )
                    return

            hard_limit = (
                RUNTIME_HARD_PENETRATION_M
                if contact_authorized
                else INACTIVE_PENETRATION_M
            )
            # Allow a small collection-mode overlap only after the scratch
            # stroke, while unloading/opening.  This never applies to the
            # acquisition, contact, or full-curl stages.
            if (
                not contact_authorized
                and self.current_stage.startswith(UNLOAD_RUNTIME_TOLERANCE_STAGES)
            ):
                hard_limit = UNLOAD_RUNTIME_PENETRATION_TOLERANCE_M

            if signed < hard_limit:
                self.emergency_stop(
                    (
                        "OVER_COMPRESSION_"
                        if contact_authorized
                        else "UNPLANNED_FINGER_COLLISION_"
                    )
                    + finger
                    + "_signed="
                    + str(round(signed * 1000.0, 2))
                    + "mm_dist="
                    + str(round(distance * 1000.0, 2))
                    + "mm_src="
                    + source
                )
                return

            deep_adjust_key = (
                str(self.current_stage),
                self.scratch_cycle_from_stage(self.current_stage),
                str(finger),
            )
            if (contact_authorized and signed < DEEP_ADJUST_ENTER_SIGNED_M
                    and self.current_stage.startswith(('SCRATCH_WAVE_ASYNC_', 'ARC_SYNC_', 'ARC_ASYNC_'))
                    and deep_adjust_key not in getattr(
                        self, '_deep_adjust_record_only_keys', set())):
                # Rejoin the exact interpolated pose that was being tracked
                # when measured depth crossed 4 mm.  Passing no snapshot made
                # ``update_depth_recovery`` use the *endpoint* of the current
                # keyframe instead; on 997 that incorrectly requested another
                # ~7.5 degrees of ring curl and advanced index/middle before
                # their shared trajectory clock had reached those commands.
                # The recovery may relieve/rejoin this pose, but must never
                # turn a safety correction into a shortcut to a future pose.
                depth_snapshot = (
                    np.asarray(self.current_arm_command).copy(),
                    dict(self.current_finger_commands),
                    float(self.segment_elapsed),
                )
                self.begin_depth_recovery(finger, tracking_snapshot=depth_snapshot)
                if not self.executing:
                    return

            if finger in self.active_fingers:
                was_latched = bool(self.deep_adjust_latched.get(finger, False))
                now_latched = update_deep_adjust_latch(was_latched, signed)
                self.deep_adjust_latched[finger] = now_latched

                if now_latched and not was_latched:
                    print(
                        "[V4.12-ARC4 RUNTIME DEEP_ADJUST ENTER]",
                        finger,
                        "depth_mm=",
                        round(-signed * 1000.0, 2),
                    )

                elif was_latched and not now_latched:
                    print(
                        "[V4.12-ARC4 RUNTIME DEEP_ADJUST EXIT]",
                        finger,
                        "depth_mm=",
                        round(-signed * 1000.0, 2),
                    )

            if (
                finger
                in
                self.active_fingers
                and
                command
                >=
                MIN_MAIN_PHASE_DEG
                and getattr(self, '_depth_recovery', None) is None
                and not self.current_stage.startswith(
                    ("SCRATCH_CURL_", "FINGER_CURL_RELEASE_", "UNLOAD_STEP_")
                )
            ):
                # Euclidean surface distance is a better "lost contact" signal
                # than an infinitely extended tangent-plane signed value.
                if (
                    distance
                    >
                    CONTACT_LOSS_DISTANCE_M
                ):
                    self.contact_loss_time[
                        finger
                    ] += float(
                        elapsed
                    )
                else:
                    self.contact_loss_time[
                        finger
                    ] = 0.0

                if (
                    self.contact_loss_time[
                        finger
                    ]
                    >
                    CONTACT_LOSS_TIMEOUT_S
                ):
                    self.emergency_stop(
                        "ACTIVE_CONTACT_LOST_"
                        +
                        finger
                        +
                        "_distance="
                        +
                        str(
                            round(
                                distance
                                *
                                1000.0,
                                2,
                            )
                        )
                        +
                        "mm"
                    )
                    return

            else:
                self.contact_loss_time[
                    finger
                ] = 0.0

    def report_contact_sources(self):
        """Inspect reporting configuration without enabling/changing physics."""
        try:
            robot = self.stage.GetPrimAtPath(ROBOT_ROOT)
            reporters = [str(p.GetPath()) for p in Usd.PrimRange(robot)
                         if "PhysxContactReportAPI" in p.GetAppliedSchemas()] if robot else []
            surfaces = {}
            for root in BACK_CONTACT_ROOTS:
                prim = self.stage.GetPrimAtPath(root)
                surfaces[root] = dict(present=bool(prim),
                    collision_enabled=prim.GetAttribute("physics:collisionEnabled").Get() if prim else None,
                    reporter="PhysxContactReportAPI" in prim.GetAppliedSchemas() if prim else False)
            print("[V102 CONTACT SOURCES]", "robot_reporters=", reporters, "surfaces=", surfaces,
                  "Note: subscription/schema presence does not prove deformable events are supported/delivered.")
        except Exception as exc:
            print("[V102 CONTACT SOURCES UNAVAILABLE]", repr(exc))

    def report_contact_diagnostic_coverage(self, reason):
        counts = getattr(self, "contact_diag_counts", {})
        print("[V102 CONTACT COVERAGE]", "reason=", reason, "event_counts=", counts,
              "V2_delivery=", "OBSERVED" if counts.get("/World/DeformableBackV2", 0) else
              "UNVERIFIED_NO_EVENTS (not proof of no collision)")

    def log_stall_diagnostics(self, finger, progress, reason, force=False):
        """Read existing handles only. Never initialize, set gains or apply APIs."""
        key = (self.segment_index, finger)
        logged = getattr(self, "stall_diag_logged", set())
        if not force and key in logged:
            return
        logged.add(key)
        self.stall_diag_logged = logged
        record = dict(stage=self.current_stage, reason=reason, finger=finger,
                      planned_triplet_deg=HAND_MODEL.joint_targets(progress),
                      event_counts=dict(getattr(self, "contact_diag_counts", {})))
        try:
            drives = self.finger_triplet_drives[finger]
            record["usd"] = [dict(target_deg=d.GetTargetPositionAttr().Get(),
                                  stiffness=d.GetStiffnessAttr().Get(), damping=d.GetDampingAttr().Get(),
                                  max_force=d.GetMaxForceAttr().Get(), drive_type=d.GetTypeAttr().Get())
                             for d in drives]
            art = self.articulation
            # get_gains() falls back to USD and can Apply DriveAPI if a handle
            # is invalid. Guard it, so diagnostics never enter that fallback.
            if not art.handles_initialized:
                record["live_status"] = "UNAVAILABLE_PHYSICS_HANDLE"
            else:
                ids = np.asarray(self.finger_triplet_indices[finger], dtype=int)
                view = art._articulation_view
                if not view.is_physics_handle_valid():
                    raise RuntimeError("Physics handle invalid; no fallback reads")
                kps, kds = view.get_gains(joint_indices=ids)
                efforts = view.get_max_efforts(joint_indices=ids)
                record["live_status"] = "READ_FROM_PHYSICS_HANDLE"
                record["live_kp"] = np.asarray(kps).reshape(-1).tolist()
                record["live_kd"] = np.asarray(kds).reshape(-1).tolist()
                record["live_max_effort"] = np.asarray(efforts).reshape(-1).tolist()
                record["actual_triplet_deg"] = np.degrees(art.get_joint_positions(joint_indices=ids)).tolist()
                record["actual_velocity_deg_s"] = np.degrees(art.get_joint_velocities(joint_indices=ids)).tolist()
                for name in ("live_kp", "live_kd", "live_max_effort", "actual_triplet_deg", "actual_velocity_deg_s"):
                    values = np.asarray(record[name], dtype=float)
                    if values.shape != (3,) or not np.all(np.isfinite(values)):
                        raise RuntimeError("Missing/nonfinite live field: " + name)
                record["units_note"] = "Angular live gains use radians; USD angular gains use degrees. Not measured motor torque."
        except Exception as exc:
            record["live_status"] = "UNAVAILABLE_OR_PARTIAL"
            record["read_error"] = repr(exc)
        now = getattr(self, "contact_diag_elapsed_s", 0.0)
        record["recent_contact_pairs"] = [dict(pair=r, age_s=round(now-r["sim_time_s"], 3))
                                          for r in getattr(self, "contact_diag_latest", {}).values()]
        print("[V102 STALL DRIVE SNAPSHOT]", record)

    def record_back_contact(self, header, contact_data, paths, back_root, lost):
        """Log first/repeated pair at bounded frequency; always process safety."""
        now = getattr(self, "contact_diag_elapsed_s", 0.0)
        counts = getattr(self, "contact_diag_counts", {})
        counts[back_root] = counts.get(back_root, 0) + 1
        self.contact_diag_counts = counts
        pair = tuple(paths)
        latest = getattr(self, "contact_diag_latest", {})
        last_logs = getattr(self, "contact_diag_last_log", {})
        # Bound history even for generated collider paths.
        if pair not in latest and len(latest) >= 32:
            oldest = next(iter(latest))
            latest.pop(oldest)
            last_logs.pop(oldest, None)
        record = dict(stage=self.current_stage, sim_time_s=now, back_root=back_root,
                      event="LOST" if lost else str(header.type), paths=list(paths),
                      points=contact_sample_for_log(header, contact_data))
        previous = latest.get(pair)
        latest[pair] = record
        if (previous is None or previous["stage"] != self.current_stage
                or previous["event"] != record["event"]
                or now-last_logs.get(pair, -1e9) >= CONTACT_DIAGNOSTIC_LOG_INTERVAL_S):
            print("[V102 BACK CONTACT]", record)
            last_logs[pair] = now
        self.contact_diag_latest, self.contact_diag_last_log = latest, last_logs

    def contact_report(self, headers, contact_data):
        if not self.executing:
            return

        try:
            for header in headers:
                paths = []

                for value in [
                    header.actor0,
                    header.actor1,
                    header.collider0,
                    header.collider1,
                ]:
                    try:
                        paths.append(str(PhysicsSchemaTools.intToSdfPath(value)))
                    except Exception as exc:
                        paths.append('')  # Preserve actor/collider side ordering.
                        if not getattr(self, "contact_diag_error_logged", False):
                            print("[V102 CONTACT PATH UNAVAILABLE]", repr(exc))
                            self.contact_diag_error_logged = True

                finger = RELEASE_POLICY.base_finger_pair(paths, BASE_PATH, ROBOT_ROOT)
                if finger is not None:
                    pair = tuple(paths)
                    if header.type == ContactEventType.CONTACT_LOST:
                        self._base_contact_pairs.pop(pair, None)
                    else:
                        separation = RELEASE_POLICY.touching_separation(header, contact_data)
                        if separation is not None:
                            self._base_contact_pairs[pair] = dict(source='PHYSX_CONTACT_REPORT',
                                finger=finger, paths=paths, separation_m=separation,
                                sim_time_s=getattr(self, 'telemetry_sim_time', 0.))
                            if self.current_stage.startswith(V103_BASE_CONTACT_STOP_STAGES):
                                # This callback runs after a physics step. Hold
                                # now, before another step can drive farther in.
                                self.hold_actual_state()
                        else:
                            self._base_contact_pairs.pop(pair, None)
                    continue
                back_root = next((root for root in BACK_CONTACT_ROOTS
                                  if any(path_is_within(p, root) for p in paths)), None)
                if back_root is None or not any(path_is_within(p, ROBOT_ROOT) for p in paths):
                    continue
                lost = header.type == ContactEventType.CONTACT_LOST
                try:
                    self.record_back_contact(header, contact_data, paths, back_root, lost)
                except Exception as exc:
                    if not getattr(self, "contact_diag_error_logged", False):
                        print("[V102 CONTACT DIAGNOSTIC UNAVAILABLE]", repr(exc))
                        self.contact_diag_error_logged = True
                if lost:
                    continue  # Separation event is not a new unauthorized touch.

                authorized_fingers = set(self.active_fingers)
                # During an acquisition transition, the new finger is allowed
                # to touch only after its real tip has entered its own lane and
                # the 12 mm near-surface band. Planned synchronized fingers are
                # considered here even when PhysX contact arrives one frame
                # before the offline sampler changes PRECONTACT to CONTACT.
                acquiring_fingers = set()
                for finger in LONG_FINGERS:
                    synchronized_acquisition_stage = self.current_stage.startswith(
                        (
                            "ARC_SYNC_PRECONTACT_",
                            "ARC_SYNC_CONTACT_",
                            "ARC_ASYNC_PRECONTACT_",
                            "ARC_ASYNC_CONTACT_",
                            "SCRATCH_WAVE_ASYNC_",
                        )
                    )
                    acquisition_stage = (
                        self.current_stage.startswith(
                            "ARC_PRECONTACT_" + finger.upper()
                        )
                        or self.current_stage.startswith(
                            "ARC_CONTACT_" + finger.upper()
                        )
                        or synchronized_acquisition_stage
                    )
                    intended_finger = bool(
                        finger
                        in set(getattr(self, "current_authorized_contact_fingers", []))
                        or (
                            synchronized_acquisition_stage
                            and finger
                            in set(getattr(self, "planned_working_fingers", []))
                        )
                    )
                    if intended_finger and acquisition_stage:
                        tip = get_world_position(self.stage, FINGER_TIP_PATHS[finger])
                        if tip is None or self.current_lane_spec is None:
                            continue
                        lane_state = lane_relation_for_world_point(
                            tip,
                            self.current_lane_spec,
                            finger,
                        )
                        relation = final_surface_relation(
                            tip,
                            self.back_mesh_query,
                            self.back_points,
                            self.back_normals,
                            self.back_tree,
                        )
                        distance = float(relation["distance"])
                        if self.transition_contact_is_authorized(
                            finger,
                            lane_state,
                            distance,
                        ):
                            acquiring_fingers.add(finger)
                            authorized_fingers.add(finger)

                planned_tip_paths = [
                    FINGER_TIP_PATHS[finger]
                    for finger in authorized_fingers
                    if finger in FINGER_TIP_PATHS
                ]

                # V10 compliant-contact semantics.  Open gross approach is
                # still strict.  Once the measured arc handoff/contact phase
                # begins, the terminal/distal phalanx of the corresponding
                # planned working finger is also an intended skin-contact body.
                terminal_contact_stage = (
                    self.current_stage.startswith("ARC_HANDOFF_PRECURVE")
                    or self.current_stage.startswith("ARC_PRECONTACT_")
                    or self.current_stage.startswith("ARC_CONTACT_")
                    or self.current_stage.startswith("ARC_SYNC_")
                    or self.current_stage.startswith("ARC_ASYNC_")
                    or self.current_stage.startswith("SCRATCH_WAVE_ASYNC_")
                    or self.current_stage.startswith("SYNC_")
                    or self.current_stage.startswith("SCRATCH_CURL_")
                    or self.current_stage.startswith("FINGER_CURL_RELEASE_")
                    or self.current_stage.startswith(("UNLOAD_STEP_", "UNLOAD_J5_LIFT_"))
                    or self.current_stage.startswith("DEEP_ADJUST_")
                    or self.current_stage == "FULL_SCRATCH"
                    or self.current_stage.startswith("FULL_SCRATCH_CONTACT_CYCLE_")
                )

                authorized_terminal_fingers = set(authorized_fingers)
                # Only active fingers and runtime-verified acquiring fingers
                # may use compliant distal contact semantics. Their real tips
                # are lane-, patch- and near-surface-gated above.
                authorized_terminal_fingers.update(acquiring_fingers)

                planned_distal_paths = [
                    FINGER_DISTAL_PATHS[finger]
                    for finger in authorized_terminal_fingers
                    if finger in FINGER_DISTAL_PATHS and terminal_contact_stage
                ]

                # Prefer the actual collider to its actor: fixed links can
                # share an actor, but an allowed tip must not authorize a palm.
                robot_colliders = [str(PhysicsSchemaTools.intToSdfPath(v))
                                   for v in (header.collider0, header.collider1)]
                robot_colliders = [p for p in robot_colliders if path_is_within(p, ROBOT_ROOT)]
                if not robot_colliders:
                    robot_colliders = [p for p in paths if path_is_within(p, ROBOT_ROOT)]
                allowed = any(
                    path_is_within(collider, path)
                    for path in (planned_tip_paths + planned_distal_paths)
                    for collider in robot_colliders
                )

                if not allowed:
                    print(
                        "[V4.12-ARC CONTACT STOP]",
                        "state=",
                        self.state,
                        "active=",
                        list(self.active_fingers),
                        "planned=",
                        list(self.planned_working_fingers),
                        "stage=",
                        self.current_stage,
                        "paths=",
                        paths,
                    )
                    self.emergency_stop("UNPLANNED_ROBOT_BACK_CONTACT")
                    return

        except Exception as exc:
            # Do not silently turn unavailable contact evidence into "clear".
            if not getattr(self, "contact_diag_error_logged", False):
                print("[V102 CONTACT REPORT UNAVAILABLE]", repr(exc),
                      "geometric checks remain active; contact safety not fully verified")
                self.contact_diag_error_logged = True

    def emergency_stop(self, reason):
        self.report_contact_diagnostic_coverage(reason)
        for cycle_index in sorted(self.dynamic_depth_stats):
            self.report_dynamic_depth_cycle(cycle_index)
        try:
            self.hold_actual_state()
        except Exception:
            pass

        self.executing = False
        self.console_prompt_mode = None
        self.console_input_buffer = ""
        self.saved_execution = None
        self.active_fingers = []
        self.current_authorized_contact_fingers = []
        self.planned_working_fingers = []
        self.current_stage = "STOPPED"
        self.current_segment_alpha = 0.0
        self.current_lane_spec = None
        self.segment_arrival_wait_elapsed = 0.0
        self.segment_settle_elapsed = 0.0
        self.segment_wait_log_elapsed = 0.0
        self.contact_j5_lift_deg = 0.0
        self.deep_adjust_latched = {
            finger: False
            for finger in LONG_FINGERS
        }

        self.write_status("error", "SAFETY STOP: " + str(reason))

        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("V4 SAFETY STOP")
        print(reason)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def shutdown(self):
        if not getattr(
            self,
            "alive",
            False,
        ):
            return

        self.alive = False
        monitor = getattr(self, "deformation_monitor", None)
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                print("[V102 LIVE DEFORMATION CLEANUP ERROR]", repr(exc))
        self.executing = False
        self.console_prompt_mode = None
        self.console_input_buffer = ""
        self.saved_execution = None
        self.active_fingers = []
        self.current_authorized_contact_fingers = []
        self.segment_arrival_wait_elapsed = 0.0
        self.segment_settle_elapsed = 0.0
        self.segment_wait_log_elapsed = 0.0
        self.contact_j5_lift_deg = 0.0

        for attr_name in [
            "physics_subscription",
            "contact_subscription",
        ]:
            subscription = getattr(
                self,
                attr_name,
                None,
            )

            if subscription is not None:
                for method_name in [
                    "unsubscribe",
                    "release",
                ]:
                    method = getattr(
                        subscription,
                        method_name,
                        None,
                    )

                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass

                try:
                    setattr(
                        self,
                        attr_name,
                        None,
                    )
                except Exception:
                    pass

        print(
            "[V4.2] Previous scratch runtime shut down."
        )


    def physics_step(self, dt):
        if not getattr(
            self,
            "alive",
            False,
        ):
            return

        dt = float(dt)
        self.telemetry_sim_time = getattr(self,'telemetry_sim_time',0.) + dt

        monitor = getattr(self, "deformation_monitor", None)
        if monitor is not None:
            monitor.update(dt, self.current_stage)

        self.poll_console_prompt()

        self.request_timer += dt
        self.safety_timer += dt

        if self.request_timer >= REQUEST_CHECK_PERIOD_S:
            self.request_timer = 0.0
            self.check_request()

        if self.executing:
            with PALM_POLICY.scope((self.current_mode or {}).get("palm_alignment_limit_deg", 30.0)):
                self.update_execution(dt)

                if self.safety_timer >= SAFETY_CHECK_PERIOD_S:
                    elapsed = self.safety_timer
                    self.safety_timer = 0.0
                    self.safety_check(elapsed)

    def start(self):
        # Session-only diagnostics: do not save the USD or change collision
        # filters/self-collision, drive gains, or the physical time step.
        try:
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                base = self.stage.GetPrimAtPath(BASE_PATH)
                if not base:
                    raise RuntimeError('BASE_LINK_MISSING')
                reporter = PhysxSchema.PhysxContactReportAPI.Apply(base)
                reporter.CreateThresholdAttr(0.)
            print('[V103 BASE CONTACT SOURCE] PhysX base/finger pairs; separation<=0; session-only reporter')
            root = self.stage.GetPrimAtPath(ROBOT_ROOT)
            articulation_settings = [
                (str(p.GetPath()), p.GetAttribute('physxArticulation:enabledSelfCollisions').Get())
                for p in Usd.PrimRange(root)
                if p.GetAttribute('physxArticulation:enabledSelfCollisions')
            ] if root else []
            print('[V103 BASE CONTACT FILTER AUDIT]', articulation_settings,
                  'filters unchanged; disabled/filtered self-collisions may suppress base events;',
                  'no events is NOT proof of no base contact')
        except Exception as exc:
            print('[V103 BASE CONTACT SOURCE UNAVAILABLE]', repr(exc),
                  'base stop NOT verified; no bounding-box contact fallback')
        self.physics_subscription = (
            omni.physx
            .get_physx_interface()
            .subscribe_physics_step_events(
                self.physics_step
            )
        )

        try:
            self.contact_subscription = (
                omni.physx
                .get_physx_simulation_interface()
                .subscribe_contact_report_events(
                    self.contact_report
                )
            )
        except Exception as exc:
            self.contact_subscription = None
            print("[V4] Contact-report warning:", repr(exc))

        print("")
        print("==============================================")
        print("V103 NEAR-BASE 4F ASYNCHRONOUS COMFORT-WAVE READY")
        print("[V103 BASE POLICY] planning_guard=DISABLED; base is telemetry only during approach/planning/replay")
        print("[V103 BASE RELEASE POLICY] after scratch: curl until BackSurface clearance or stop at measured base contact (both exits accepted)")
        print("V103 BUILD: CONTACT-WORKSPACE-v22-MINIMA-CENTROID-MCP-ROW-DEPTH-IK-ROLLING-WAVE-CENTER15; CONTACT-STATE-v5; SAFE-PLANE-NORMAL-APPROACH-V1; PREPOSITION-SWEEP-OVERRIDE-V1; DIAGNOSTIC-DEPTH-CAP-OVERRIDE-V1; Cmin=four fixed-palm natural-minimum centroid, MCP row perpendicular BASE->Cmin, target center hard 15mm")
        print('[V103 POINT TEST]', 'strict=',V103_STRICT_POINT_TEST,
              'finger_anchor=NONE',
              'failed search is NOT a reachability proof')
        print("V102 UNITS: finger commands/phases (including legacy *_deg labels) are PROGRESS PERCENT; arm q/J1-J5 and MCP/PIP/DIP feedback remain physical angles.")
        print("==============================================")
        print("Startup drive targets synchronized to ACTUAL robot state.")
        print("Historical CENTER drive targets are NOT used.")
        print("GUI click becomes TARGET_PATCH center; working fingers use parallel local lanes, not one shared point.")
        print("V100 FINITE-REGION FIX: every contacting fingertip stays inside the 75 mm GUI patch; executable plans use a 65 mm radius reserve and an 8 mm lane-cross reserve.")
        print("V100 COORDINATE CHECK: GUI candidate ID/XYZ, robot-base world pose, CSV-to-BackSurface sample alignment and selected-target mesh alignment are verified/logged.")
        print("Direct-q refinement, endpoint validation and ARC contact acquisition are target-lane locked.")
        print("V8 direct-q fallback is explicitly rechecked against TARGET_PATCH center + every active finger lane before VALID.")
        print("No fixed open-tip standoff exists; actual no-contact geometry decides the approach.")
        print("The 0.20 mm non-tip static-contact epsilon is numerical only; it is not an approach standoff.")
        print("V103 PALM POLICY: prefer parallel; search full paths under 30, then 45, then 60 deg caps. These are limits, NOT fixed poses. No whole-hand collision or penetration gates are relaxed.")
        print("Working fingers stay open until flex acquisition; all commands obey the limited coupled curl range.")
        print("TRAINING POLICY: 4F only; pinky prefers 3 mm and may finish at a safe local optimum; index/middle/ring use the relaxed 1-4 mm gate; 3F/2F/1F disabled.")
        print("V100 AUTH FIX: ARC handoff/precontact planned-finger authorization is carried into final trajectory validation; palm/proximal/middle links remain strict.")
        print("V102 DIRECT ENDPOINT IK: J1-J5 move once to the selected scratch palm pose while all four fingers remain straight.")
        print("V103 SERIAL DYNAMIC ACQUISITION: real mesh distance orders shallow-to-deep contact; one focus advances until contact, then the next handoff begins; no overlap.")
        print("V102 DYNAMIC ARC MINIMUM: every candidate measures each finger minimum from the current palm pose/back normal; no fixed template phase or 8-degree repair window is used.")
        print("V102 NONEMPTY 4F WAVE: contact progress is scaled to remaining stroke; safety margin is not a completion gate; index/middle/ring use 1-4 mm with 3 mm nominal target, pinky uses a safe local optimum if needed. Runtime contact quality is checked separately.")
        print("V102 CONTINUOUS DEMONSTRATION: no stationary full-contact keyframe, no per-wave settle dwell, and no smoothstep zero-velocity reset between wave samples.")
        print("[V104 CONTACT ACCEPTANCE] acquisition+wave sampled continuously; airborne curl excluded; "
              "hard planning gate=required-finger depth 1-4mm with target 3mm; "
              "slide/moving-contact/tangent-ratio=DIAGNOSTIC_ONLY; "
              "geometric quality only, not measured skin comfort.")
        print("[V102 DRIVE TARGET V2] layer=session; resolved target verified after every write; "
              "MCP/PIP/DIP feedback logged on arrival wait.")
        print("V103 IMITATION QUALITY GATE: four-finger trajectory is mandatory; index/middle/ring independently use 1-4 mm actual peak depth, pinky targets 3 mm but may use a safe local optimum <=4 mm; pairwise spread is diagnostic only.")
        print("V102 FAST DATA PLANNING: reduced Lula/direct seeds, first-valid balance exit, and two-valid-endpoint stage stop avoid redundant optimization for imitation-data batches.")
        print("V102 STAGED CURL: 100% -> MCP/PIP/DIP=70/85/45 deg; hard limits=70/90/45. Wrist unload before opening; cycles=" + str(int(SCRATCH_CYCLE_COUNT)))
        print("V102 ACQUISITION REPAIR: fast path first, J4/J5 local fallback then bounded J1-J5 support; all swept safety gates retained.")
        print("V102 COMFORT PRESSURE: J5 reduces first, J4 trims second, and J1-J3 assist only if wrist-only geometry is insufficient.")
        print("V102 TARGET POLICY: centre accuracy is relaxed inside the same finite scratch region; lane and 8 mm penetration safety remain hard.")
        print("V102 TERMINAL GATE: enter y before first motion; after completion, n stops repetition and any other input repeats the saved trajectory.")
        print("V100 NO-OP ASSIST FIX: an endpoint-to-endpoint interpolation is never reported as arm motion.")
        print("V100 RUNTIME FIX: active contact is committed only after keyframe arrival; synchronized acquiring fingers are independently tip/lane-gated before distal authorization.")
        print("V100 NO-JUMP FIX: trajectory start preserves the measured Play-time finger angles; no synthetic 0-degree reset is sent.")
        print("V100 GEOMETRIC HANDOFF FIX: all working fingers stay straight through gross approach; the low-phase target-centered handoff is solved from measured four-tip centroid geometry.")
        print("V100 ACTIVE-LANE FIX: existing contacts are lane-constrained during optimization and every interpolated offline validation sample.")
        print("V100 LANE-RESERVE FIX: executable plans keep active fingertip cross-track error within 8 mm; the 14 mm runtime boundary remains an emergency limit.")
        print("V100 ARRIVAL-GATE FIX: a keyframe advances only after measured arm/finger joints are inside tolerance and settled; tracking timeout stops explicitly instead of drifting into the next finger.")
        print("V100 CONTACT-LIFECYCLE FIX: active/authorized contact is released at mesh-confirmed UNLOAD_ALL; opening, retreat and Home are verified free-space states.")
        print("Depth policy DEPTH-1TO4-v1: target 1-4 mm; planned cap 3.6 mm; actual >4 mm freezes curl and validates wrist relief; >8 mm STOP; relief exit <=3.5 mm; validated rejoin required.")
        print("If no safe 4F endpoint/full trajectory exists, this target terminates; no 3F replan is attempted.")
        print("")
        print("After each scratch:")
        print("  UNLOAD_ALL")
        print("  -> OPEN_AFTER_UNLOAD")
        print("  -> SAFE_RETREAT")
        print("  -> RETURN_HOME")
        print("  -> terminal repeat prompt (n returns to GUI-target idle)")
        print("")
        print("Run scratch_target_gui.py in a SEPARATE Python process.")
        print("==============================================")


# ============================================================
# STOP OLD / LEGACY RUNTIME INSTANCES
# ============================================================

def cleanup_legacy_scratch_runtimes():
    """
    Isaac Script Editor executes files in temporary script namespaces.
    A callback from an older V4 can therefore survive even when the newer
    script's globals() cannot see it.

    V4.13 uses two cleanup layers:
      1. builtins registry for all V4.2+ runs.
      2. best-effort GC scan for older IntegratedScratchV4 objects.
    """
    stopped = 0

    registered = getattr(
        builtins,
        "_DOFBOT_SCRATCH_V4_RUNTIME",
        None,
    )

    if registered is not None:
        shutdown = getattr(
            registered,
            "shutdown",
            None,
        )

        if callable(shutdown):
            try:
                shutdown()
                stopped += 1
            except Exception:
                pass

    try:
        objects = list(
            gc.get_objects()
        )
    except Exception:
        objects = []

    for obj in objects:
        if obj is registered:
            continue

        try:
            if (
                type(obj).__name__
                != "IntegratedScratchV4"
            ):
                continue
        except Exception:
            continue

        try:
            obj.alive = False
        except Exception:
            pass

        try:
            obj.executing = False
        except Exception:
            pass

        for attr_name in [
            "physics_subscription",
            "contact_subscription",
        ]:
            try:
                subscription = getattr(
                    obj,
                    attr_name,
                    None,
                )
            except Exception:
                subscription = None

            if subscription is None:
                continue

            for method_name in [
                "unsubscribe",
                "release",
            ]:
                method = getattr(
                    subscription,
                    method_name,
                    None,
                )

                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

            try:
                setattr(
                    obj,
                    attr_name,
                    None,
                )
            except Exception:
                pass

        stopped += 1

    gc.collect()

    if stopped:
        print(
            "[V4.2] Legacy runtime cleanup count:",
            stopped,
        )


cleanup_legacy_scratch_runtimes()


# ============================================================
# CREATE ONE RUNTIME
# ============================================================

_INTEGRATED_SCRATCH_V4 = IntegratedScratchV4()

builtins._DOFBOT_SCRATCH_V4_RUNTIME = (
    _INTEGRATED_SCRATCH_V4
)

_INTEGRATED_SCRATCH_V4.start()
