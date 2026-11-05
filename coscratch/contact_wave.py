"""Quality-driven wrist/finger rollout. Pure planning, no simulator commands."""
import time
import numpy as np
from scipy.optimize import least_squares

BUILD = 'CONTACT-WORKSPACE-v22-DEPTH-CONDITIONED-IK-ROLLING-WAVE-CENTER15'
SUPPORT_DEG = np.array([3., 3., 3., 6., 10.])
DYNAMIC_MAX_DELAY_DEG = 18.0
DYNAMIC_PROBE_DEG = 1.0
DYNAMIC_MIN_SENSITIVITY_M_PER_DEG = 0.00002
# Once a finger has completed its measured depth handoff, the next finger
# may still use J1-J5 to solve its own target, but it must not drive the
# completed finger materially deeper.  This is a handoff invariant, not a
# replacement for the global 1--4 mm depth policy.
HANDOFF_DEPTH_DRIFT_M = 0.0006
# Command-rate reserve for a completed finger leaving the surface while the
# next focus finger advances. The physical lift is still checked from fresh
# signed-depth finite differences and by the full swept ``safe`` callback.
HANDOFF_RELEASE_RATE_RATIO = 1.35
# Minimum overlap cadence while the successor is still approaching. The
# measured derivative and swept ``safe`` callback remain the hard limits.
HANDOFF_RELEASE_MIN_RATE_RATIO = 0.75
# Release-only frames are still swept-checked, but may advance faster than a
# contact focus so the final +20 mm clearance does not look like a long hold.
RELEASE_TAIL_MAX_INCREMENT_PCT = 3.0
# During a handoff, map the next finger's measured 0->target-depth progress
# onto the completed finger's target-depth->positive-skin-clearance release.
# A completed finger must leave the skin, not remain at a residual penetration
# while the next finger enters. Positive signed distance is clearance outside
# the BackSurface; the requested handoff target is 20 mm clear because the
# fingertip rises during curl while more proximal links can remain lower.
# The old/completed finger must finish the handoff at the requested 20 mm
# positive clearance.  This is deliberately separate from the 1--4 mm
# scratching-quality band: 1--4 mm qualifies contact, while +20 mm is the
# release/clearance endpoint before the wave may finish.
HANDOFF_RELEASE_TARGET_SIGNED_M = 0.020
# An optional finger may have a safe FK increment that produces no measured
# normal-depth progress (for example, a tangential tip on a shallow patch).
# Do not let that same assistant consume the whole optional patience budget;
# after a few unchanged measurements, hand off to the next assistant at the
# same pose.  The local-optimum exit remains the final bounded fallback.
OPTIONAL_FOCUS_STALL_STEPS = 3
OPTIONAL_FOCUS_STALL_TOLERANCE_M = 0.0001

# A curl step is not generally normal to the local back surface.  When the
# fingertip moves mostly tangentially, the same joint increment produces much
# less signed penetration than the nominal FK preview suggests.  Use a bounded
# geometric gain for the *next* bounded step; the measured signed depth and the
# full swept safety callback remain authoritative.
NORMAL_ALIGNMENT_FLOOR = 0.50
NORMAL_COMPENSATION_MAX = 1.50

# Core quality/handoff policy. 3 mm is a preferred optimizer setpoint only;
# the live handoff gate for mandatory fingers is the measured 1--4 mm band.
LIVE_CORE_DEPTH_MIN_M = 0.0010
LIVE_CORE_DEPTH_MAX_M = 0.0040
# Natural-minimum scan is prediction, not a stop command. The focus keeps
# curling through the minimum while J5/J4 are allowed to take over the normal
# component.  Background followers approach the minimum but do not consume
# their post-minimum scratch stroke before they become focus.
MINIMUM_SCAN_COARSE_STEP_PCT = 4.0
MINIMUM_SCAN_FINE_STEP_PCT = 0.5
MINIMUM_SCAN_FINE_RADIUS_PCT = 4.0
EARLY_WRIST_LOOKAHEAD_PCT = 5.0
EARLY_WRIST_SENSITIVITY_M_PER_PCT = 0.00005
# V19 rolling prediction. Later owners do not wait for a fixed look-ahead
# threshold: at every accepted frame we rescan the current-palm natural
# minimum, predict the remaining 3 mm deficit, finite-difference the wrist
# depth gain, and distribute only the required wrist angle over the remaining
# command-to-minimum time. The first frozen owner is explicitly exempt.
ROLLING_MINIMUM_GAP_STEPS = 1.0
ROLLING_WRIST_PROBE_DEG = 0.25
ROLLING_WRIST_J5_MAX_STEP_DEG = 0.80
ROLLING_WRIST_J4_MAX_STEP_DEG = 0.55
ROLLING_WRIST_LEAD_STEPS = 0.50
ROLLING_WRIST_MIN_DEFICIT_M = 0.00005
ROLLING_RELEASE_OUTWARD_MARGIN_M = 0.00008
# After a finger reaches its measured minimum, curl it away promptly before
# allowing the shared wrist to press the successor.  This is an action policy,
# not a derivative/sensitivity test; the full swept ``safe()`` check below is
# still the authority for depth and collision limits.
ROLLING_RELEASE_FAST_CURL_STEP_DEG = 8.0
# A release step is still validated by the full swept ``safe()`` check below.
# The old 6 deg bookkeeping cap was smaller than the curl-away required by
# near-base poses (909 commonly needs 7--11 deg), so it rejected every safe
# continuous handoff before the real collision/depth validator ran.  Keep a
# bounded single-window step, but let the measured release outrun the wrist.
ROLLING_RELEASE_MAX_STEP_DEG = 12.0
# V20: a safe finger-only command that produces essentially no additional
# live penetration must not consume the rollout budget for a mandatory owner.
# Once the owner is below the 1 mm live gate, lack of measured inward progress
# forces the same-frame arm/wrist branches to be evaluated.
MANDATORY_FINGER_ONLY_MIN_PROGRESS_M = 0.00005
BACKGROUND_RATE_BY_RANK = (0.65, 0.42, 0.25)
BACKGROUND_DEPTH_SLOW_M = 0.0024
BACKGROUND_DEPTH_HOLD_M = 0.0034
# Pending fingers remain moving, but later ranks must keep normal-depth headroom
# for the current owner.  These are coordination reserves, not final quality
# gates.  If a follower is already deeper than its reserve, it may hold or curl
# outward, but a candidate may not drive it materially deeper while another
# finger owns the handoff.
PENDING_DEPTH_RESERVE_BY_RANK_M = (0.0018, 0.0010, 0.0005)
PENDING_DEPTH_GROWTH_TOL_M = 0.00015
# If an old completed finger is close to the 3.6 mm planned cap, its curl-away
# must outrun common wrist pressure even before the successor has reached 1 mm.
HANDOFF_RELEASE_URGENT_START_M = 0.0028
HANDOFF_RELEASE_URGENT_FULL_M = 0.0034
HANDOFF_RELEASE_URGENT_MAX_RATIO = 2.0
# An optional pinky is a real wave channel, but it cannot consume the remaining
# rollout budget while a mandatory core finger is still pending.  It gets a
# short bounded ownership attempt in frozen order, then yields without being
# declared complete.
OPTIONAL_BEFORE_MANDATORY_MAX_ACCEPTED_STEPS = 2
# Once a finger owns the frozen completion turn, live dynamic delay may still
# inform geometry/wrist support but must not throttle that owner back to the
# old 0.25% crawl.  Keep a bounded 1.5--2.0% focus cadence; swept safety and
# depth guards remain authoritative.
FROZEN_FOCUS_ACQUIRE_MAX_INCREMENT_PCT = 2.0
FROZEN_FOCUS_CONTACT_MAX_INCREMENT_PCT = 1.5


class _OptionalRolloutBudget(RuntimeError):
    """Internal signal for a safe optional-channel budget stop.

    A pinky rollout may run out of its local search budget after the mandatory
    fingers already have measured quality evidence.  That condition is not a
    geometry failure and must be converted into a bounded four-channel local
    optimum by ``roll``.  Keeping it as a private exception prevents the
    signal from being mistaken for a normal candidate rejection elsewhere.
    """

    def __init__(self, focus):
        self.focus = str(focus)
        super().__init__(self.focus)


def _normal_compensation(state, probe_state, finger):
    """Return a bounded 1/cos(theta)-style gain for a curl probe.

    ``measure`` may provide the actual fingertip and local surface normal.  If
    an older caller does not provide them, keep the neutral gain so this helper
    never changes legacy behavior or invents a geometric estimate.
    """
    try:
        tip0 = np.asarray(state.get("tip", {}).get(finger), dtype=float)
        tip1 = np.asarray(probe_state.get("tip", {}).get(finger), dtype=float)
        normal0 = np.asarray(state.get("normal", {}).get(finger), dtype=float)
        normal1 = np.asarray(probe_state.get("normal", {}).get(finger), dtype=float)
        delta = tip1 - tip0
        delta_norm = float(np.linalg.norm(delta))
        normal = normal0 + normal1
        normal_norm = float(np.linalg.norm(normal))
        if (delta.shape != (3,) or normal.shape != (3,)
                or not np.all(np.isfinite(delta))
                or not np.all(np.isfinite(normal))
                or delta_norm <= 1e-9 or normal_norm <= 1e-9):
            return 1.0, None
        alignment = abs(float(np.dot(delta / delta_norm, normal / normal_norm)))
        if not np.isfinite(alignment):
            return 1.0, None
        gain = float(np.clip(
            1.0 / max(alignment, NORMAL_ALIGNMENT_FLOOR),
            1.0,
            NORMAL_COMPENSATION_MAX,
        ))
        return gain, alignment
    except Exception:
        return 1.0, None


def _surface_error_m(row):
    """Distance outside the surface, or penetration depth once in contact."""
    signed = float(row.get('signed', float('nan')))
    distance = float(row.get('distance', float('nan')))
    if not np.isfinite(signed) or not np.isfinite(distance):
        return float('inf')
    return max(0.0, -signed) if signed <= 0.0 else max(0.0, distance)


def precontact_increment(signed_at, commands, finger, maximum):
    """Advance toward first skin contact without crossing it.

    This compatibility entry point is used by the fixed-palm acquisition
    bridge in ``controller.py``. It only proposes a bounded curl increment; the
    caller still authorizes the full swept wrist/finger segment.
    """
    previous = float(signed_at(commands))
    if not np.isfinite(previous) or previous <= 0.0 or maximum <= 0.0:
        return 0.0
    accepted = 0.0
    for increment in np.linspace(0.0, float(maximum), 17)[1:]:
        trial = dict(commands)
        trial[finger] = float(commands[finger]) + float(increment)
        signed = float(signed_at(trial))
        if not np.isfinite(signed) or signed < 0.0 or signed > previous + 1e-9:
            break
        accepted, previous = float(increment), signed
    return accepted


def adaptive_overlap_increment(*, next_signed, next_minimum_signed,
                               next_command, next_minimum_command,
                               current_command, current_minimum_command,
                               current_step, at_minimum=False,
                               bootstrap=False, during_transfer=False):
    """Compatibility scheduler for legacy diagnostics.

    Production handoff uses the rolling minimum controller below.  This
    bounded helper remains available to older tests and callers without
    changing the production ordering or safety decisions.
    """
    remaining_next = max(0.0, float(next_minimum_command) - float(next_command))
    if remaining_next <= 1e-9:
        return 0.0
    base = min(3.0 if at_minimum else 5.0, remaining_next)
    ns, nmin = float(next_signed), float(next_minimum_signed)
    if (bootstrap or not np.isfinite(ns) or not np.isfinite(nmin)
            or ns <= 0.012):
        return base
    span = ns - nmin
    if span <= 0.0002:
        return base
    sensitivity = span / max(remaining_next, 1e-9)
    if not (0.00002 <= sensitivity <= 0.004):
        return base
    target = 0.008 if during_transfer else 0.003
    needed = min(remaining_next, max(0.0, ns - target) / sensitivity)
    current_remaining = max(0.0, float(current_minimum_command) - float(current_command))
    steps_left = max(1, int(np.ceil(current_remaining / max(0.5, float(current_step)))))
    cap = 12.0 if during_transfer else 24.0
    return min(remaining_next, max(base, min(cap, needed / steps_left)))


def scan_minimum_profile(measure, q, commands, finger, *,
                         max_command=100.0, steps=24):
    """Return the natural signed-distance minimum over the remaining curl."""
    start = float(commands.get(finger, 0.0))
    stop = min(float(max_command), 100.0)
    if stop <= start + 1e-9:
        signed = float(measure(q, commands)['signed'][finger])
        return dict(finger=finger, start_command=start,
                    minimum_command=start, minimum_signed=signed,
                    natural_depth=max(0.0, -signed), sensitivity=0.0,
                    slope_before=0.0, slope_after=0.0, samples=1)

    def sample(values):
        rows = []
        for command in values:
            trial = dict(commands)
            trial[finger] = float(command)
            signed = float(measure(q, trial)['signed'][finger])
            if np.isfinite(signed):
                rows.append((float(command), signed))
        return rows

    rows = sample(np.linspace(start, stop, max(12, int(steps)) + 1))
    if not rows:
        raise RuntimeError('WAVE_MINIMUM_SCAN_NONFINITE|' + str(finger))
    best = int(np.argmin([row[1] for row in rows]))
    left = rows[max(0, best - 1)][0]
    right = rows[min(len(rows) - 1, best + 1)][0]
    if right > left + 1e-9:
        rows = sorted(rows + sample(np.linspace(left, right, 17)), key=lambda x: x[0])
        unique = {}
        for command, signed in rows:
            unique[round(command, 12)] = signed
        rows = [(float(command), float(signed)) for command, signed in unique.items()]
        rows.sort()
        best = int(np.argmin([row[1] for row in rows]))
    minimum_command, minimum_signed = rows[best]
    before = ((minimum_signed - rows[best - 1][1]) /
              max(minimum_command - rows[best - 1][0], 1e-12)) if best else 0.0
    after = ((rows[best + 1][1] - minimum_signed) /
             max(rows[best + 1][0] - minimum_command, 1e-12)) if best + 1 < len(rows) else 0.0
    return dict(finger=finger, start_command=start,
                minimum_command=float(minimum_command),
                minimum_signed=float(minimum_signed),
                natural_depth=max(0.0, -float(minimum_signed)),
                sensitivity=0.5 * (before + after),
                slope_before=before, slope_after=after, samples=len(rows))


def precontact_increment_to_target(signed_at, commands, finger, maximum,
                                   target_signed, tolerance=0.00008):
    """Choose a bounded curl increment that approaches positive prep clearance."""
    previous = float(signed_at(commands))
    target = float(target_signed)
    if not np.isfinite(previous) or maximum <= 0.0 or previous <= target + tolerance:
        return 0.0
    accepted = 0.0
    for increment in np.linspace(0.0, float(maximum), 17)[1:]:
        trial = dict(commands)
        trial[finger] = float(commands[finger]) + float(increment)
        signed = float(signed_at(trial))
        if not np.isfinite(signed) or signed < target - tolerance or signed > previous + 1e-9:
            break
        accepted, previous = float(increment), signed
    return accepted


def precontact_adjust_to_target(signed_at, commands, finger, maximum,
                                target_signed, tolerance=0.00010):
    """Move either direction within a bounded window to positive prep height."""
    current = float(commands[finger])
    maximum = max(0.0, float(maximum))
    target = max(0.0, float(target_signed))
    if maximum <= 1e-12:
        return 0.0
    initial = float(signed_at(commands))
    if not np.isfinite(initial) or abs(initial - target) <= tolerance:
        return 0.0
    best_command, best_error, best_motion = current, abs(initial - target), float('inf')
    for command in np.linspace(max(0., current - maximum),
                               min(100., current + maximum), 33):
        trial = dict(commands)
        trial[finger] = float(command)
        signed = float(signed_at(trial))
        if not np.isfinite(signed) or signed < -tolerance:
            continue
        error, motion = abs(signed - target), abs(float(command) - current)
        if error + 1e-12 < best_error or (abs(error - best_error) <= 1e-12 and motion < best_motion):
            best_command, best_error, best_motion = float(command), error, motion
    return best_command - current


def continuous_touch_increment(signed_at, commands, finger, maximum,
                               continuation_deg=0.65):
    """Cross signed zero and continue curling in the same motion window."""
    maximum = max(0.0, float(maximum))
    if maximum <= 0.0:
        return 0.0, False
    start = float(signed_at(commands))
    if not np.isfinite(start):
        return 0.0, False
    if start <= 0.0:
        return min(maximum, float(continuation_deg)), True
    for increment in np.linspace(0.0, maximum, 25)[1:]:
        trial = dict(commands)
        trial[finger] = float(commands[finger]) + float(increment)
        if float(signed_at(trial)) <= 0.0:
            return min(maximum, float(increment) + max(0.0, float(continuation_deg))), True
    return maximum, False


def solve_wrist_feedforward(measure, q, commands, fingers, next_finger,
                            profile, lower, upper, scan_steps=24):
    """Small compatibility feed-forward solve using measured joint signs."""
    q = np.asarray(q, dtype=float).copy()
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    target = float(profile.get('target_depth', 0.003))
    minimum_command = float(profile.get('minimum_command', commands.get(next_finger, 0.0)))
    trial_commands = dict(commands)
    trial_commands[next_finger] = minimum_command
    base = float(measure(q, trial_commands)['signed'][next_finger])
    best = None
    for joint in (4, 3):
        for sign in (-1.0, 1.0):
            probe = q.copy()
            delta = np.radians(0.25) * sign
            probe[joint] = np.clip(probe[joint] + delta, lower[joint], upper[joint])
            actual = probe[joint] - q[joint]
            if abs(actual) <= 1e-12:
                continue
            signed = float(measure(probe, trial_commands)['signed'][next_finger])
            gain = (signed - base) / actual
            if not np.isfinite(gain) or abs(gain) < 1e-9:
                continue
            desired_signed = -target
            raw = (desired_signed - base) / gain
            q_trial = q.copy()
            q_trial[joint] = np.clip(q_trial[joint] + raw, lower[joint], upper[joint])
            predicted = scan_minimum_profile(measure, q_trial, commands, next_finger,
                                              max_command=100., steps=scan_steps)
            error = abs(predicted['natural_depth'] - target)
            candidate = (error, joint, q_trial, predicted)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return dict(delta_deg={}, q=q.copy(), predicted_profile=profile)
    _, joint, q_trial, predicted = best
    return dict(delta_deg={int(joint): float(np.degrees(q_trial[joint] - q[joint]))},
                q=q_trial, predicted_profile=predicted)


def _scan_natural_minimum(measure, q, commands, finger):
    """Scan the remaining curl arc at the current arm pose.

    The returned command is the command with the lowest signed BackSurface
    distance (largest inward reach) at this *current* wrist/palm pose. It is
    used only to predict when wrist pressure should take over. The focus is
    deliberately allowed to keep curling after this command so the motion
    remains a scratch instead of turning into a static press.
    """
    start = float(commands.get(finger, 0.0))
    if not np.isfinite(start):
        return None
    start = float(np.clip(start, 0.0, 100.0))
    coarse = list(np.arange(start, 100.0 + 1e-9, MINIMUM_SCAN_COARSE_STEP_PCT))
    if not coarse or coarse[-1] < 100.0 - 1e-9:
        coarse.append(100.0)
    rows = []
    for command in coarse:
        trial = dict(commands)
        trial[finger] = float(command)
        state = measure(q, trial)
        signed = float(state['signed'][finger])
        if np.isfinite(signed):
            rows.append((float(command), signed))
    if not rows:
        return None
    coarse_command, coarse_signed = min(rows, key=lambda row: row[1])
    lo = max(start, coarse_command - MINIMUM_SCAN_FINE_RADIUS_PCT)
    hi = min(100.0, coarse_command + MINIMUM_SCAN_FINE_RADIUS_PCT)
    fine_rows = []
    for command in np.arange(lo, hi + 1e-9, MINIMUM_SCAN_FINE_STEP_PCT):
        trial = dict(commands)
        trial[finger] = float(command)
        state = measure(q, trial)
        signed = float(state['signed'][finger])
        if np.isfinite(signed):
            fine_rows.append((float(command), signed))
    if fine_rows:
        minimum_command, minimum_signed = min(fine_rows, key=lambda row: row[1])
    else:
        minimum_command, minimum_signed = coarse_command, coarse_signed
    current_state = measure(q, commands)
    current_signed = float(current_state['signed'][finger])
    return dict(
        command_pct=float(minimum_command),
        signed_m=float(minimum_signed),
        depth_m=max(0.0, -float(minimum_signed)),
        current_signed_m=current_signed,
        current_depth_m=max(0.0, -current_signed),
        command_to_minimum_pct=float(minimum_command - start),
    )



def _rolling_wrist_compensation_plan(
    measure,
    q,
    commands,
    finger,
    minimum_info,
    lower,
    upper,
    target_depth_m,
    focus_increment,
):
    """Predict the wrist angle still needed at the current natural minimum.

    The plan is recomputed every rollout frame. Positive ``step_deg`` means
    increasing the selected arm joint; negative means decreasing it. J5 is
    preferred, J4 is used only when its local inward depth gain is stronger.
    """
    if minimum_info is None or finger is None:
        return None
    minimum_command = float(minimum_info['command_pct'])
    minimum_commands = dict(commands)
    minimum_commands[finger] = minimum_command
    base_state = measure(q, minimum_commands)
    base_signed = float(base_state['signed'][finger])
    if not np.isfinite(base_signed):
        return None
    minimum_depth = max(0.0, -base_signed)
    deficit = max(0.0, float(target_depth_m) - minimum_depth)
    remaining_command = max(0.0, minimum_command - float(commands[finger]))
    focus_step = max(0.25, float(focus_increment))
    eta_steps = max(1.0, remaining_command / focus_step)
    if deficit <= ROLLING_WRIST_MIN_DEFICIT_M:
        return dict(
            required=False,
            deficit_m=deficit,
            minimum_depth_m=minimum_depth,
            remaining_command_pct=remaining_command,
            eta_steps=eta_steps,
            joint=None,
            gain_m_per_deg=0.0,
            total_deg=0.0,
            step_deg=0.0,
        )

    best = None
    q = np.asarray(q, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    for joint, max_step in ((4, ROLLING_WRIST_J5_MAX_STEP_DEG),
                            (3, ROLLING_WRIST_J4_MAX_STEP_DEG)):
        for direction in (-1.0, 1.0):
            probe_deg = direction * float(ROLLING_WRIST_PROBE_DEG)
            probe = q.copy()
            probe[joint] += np.radians(probe_deg)
            if probe[joint] < lower[joint] - 1e-12 or probe[joint] > upper[joint] + 1e-12:
                continue
            state = measure(probe, minimum_commands)
            signed = float(state['signed'][finger])
            if not np.isfinite(signed):
                continue
            inward_gain = max(0.0, base_signed - signed) / abs(probe_deg)
            if inward_gain <= 1e-9:
                continue
            total_deg = direction * deficit / inward_gain
            # Finish the predicted compensation slightly before the natural
            # minimum. Because the full estimate is recomputed after every
            # accepted frame, this is rolling prediction rather than a fixed
            # speed or one-shot angle command.
            denominator = max(1.0, eta_steps - ROLLING_WRIST_LEAD_STEPS)
            step_deg = total_deg / denominator
            step_deg = float(np.clip(step_deg, -max_step, max_step))
            row = dict(
                required=True,
                deficit_m=deficit,
                minimum_depth_m=minimum_depth,
                remaining_command_pct=remaining_command,
                eta_steps=eta_steps,
                joint=int(joint),
                gain_m_per_deg=float(inward_gain),
                total_deg=float(total_deg),
                step_deg=float(step_deg),
            )
            if best is None or row['gain_m_per_deg'] > best['gain_m_per_deg']:
                best = row
    return best


def _couple_completed_release_to_arm(
    measure,
    q,
    cq,
    commands,
    proposed,
    completed_fingers,
    focus,
    current_state,
    cleared=(),
):
    """Curl completed fingers away promptly before shared wrist pressure.

    The previous implementation required a positive local outward-sensitivity
    finite difference before it would release a completed finger. That made
    wrist assistance unavailable exactly when a local probe was flat or noisy.
    The policy is now direct: once a prior finger is complete, append a bounded
    fast curl-away command and let the full swept ``safe()`` check decide
    whether the coupled wrist/finger segment is physically acceptable.
    """
    result = dict(proposed)
    if np.max(np.abs(np.asarray(cq, dtype=float) - np.asarray(q, dtype=float))) <= 1e-10:
        return result, True, {}
    cleared = set(cleared or ())
    diagnostics = {}
    for finger in completed_fingers:
        if finger == focus or finger in cleared:
            continue
        current_signed = float(current_state['signed'][finger])
        if not np.isfinite(current_signed):
            continue
        current_command = float(result[finger])
        if current_command >= 100.0 - 1e-9:
            diagnostics[finger] = 0.0
            continue
        extra = min(
            float(ROLLING_RELEASE_FAST_CURL_STEP_DEG),
            float(ROLLING_RELEASE_MAX_STEP_DEG),
            max(0.0, 100.0 - current_command),
        )
        result[finger] = min(100.0, current_command + extra)
        diagnostics[finger] = float(result[finger] - commands[finger])
    return result, True, diagnostics


def cooperative_pinky(q0, commands0, q1, commands1, measure, safe, target_depth=.003):
    """Try bounded companion curl without changing an accepted core solution.

    The original four-finger swept-safe proposal remains the fallback. Never
    replace an optimized pinky command by a larger forced shared increment.
    """
    original = dict(commands1)
    if 'pinky' not in original:
        return original
    state = measure(q1, original)
    best = original
    score = abs(float(state['signed']['pinky']) + target_depth)
    core_delta = max(0., *(commands1[f]-commands0[f] for f in ('index','middle','ring')))
    if core_delta <= 1e-9:
        return best
    for scale in (1., .5, .25):
        trial = dict(original)
        trial['pinky'] = min(100., max(original['pinky'], commands0['pinky']+scale*core_delta))
        if trial['pinky'] == original['pinky']:
            continue
        state = measure(q1, trial)
        signed = float(state['signed']['pinky'])
        candidate_score = abs(signed + target_depth)
        if (not state['valid'] or not np.isfinite(signed) or signed < -.004
                or candidate_score >= score-1e-12):
            continue
        if safe(q0, commands0, q1, trial)[0]:
            best, score = trial, candidate_score
    return best


def _dynamic_schedule(measure, q, commands, fingers, tie_order=None,
                      max_delay_deg=DYNAMIC_MAX_DELAY_DEG,
                      current_state=None):
    """Recompute contact order and command delays from the current geometry.

    The finite difference is FK guidance only.  The caller's ``safe`` callback
    remains the authorization for every proposed swept segment.
    """
    tie_order = list(tie_order or fingers)
    tie_rank = {finger: index for index, finger in enumerate(tie_order)}
    state = current_state if current_state is not None else measure(q, commands)
    rows = {}
    for finger in fingers:
        metric = _surface_error_m({
            'signed': state['signed'][finger],
            'distance': state['distance'][finger],
        })
        probe_commands = dict(commands)
        current = float(commands.get(finger, 0.0))
        probe = min(100.0, current + DYNAMIC_PROBE_DEG)
        delta = probe - current
        sensitivity = 0.0
        signed_sensitivity = 0.0
        normal_gain = 1.0
        normal_alignment = None
        if delta > 1e-9:
            probe_commands[finger] = probe
            probe_state = measure(q, probe_commands)
            probe_metric = _surface_error_m({
                'signed': probe_state['signed'][finger],
                'distance': probe_state['distance'][finger],
            })
            if np.isfinite(probe_metric) and np.isfinite(metric):
                sensitivity = abs(probe_metric - metric) / delta
            signed_sensitivity = (
                float(probe_state['signed'][finger])
                - float(state['signed'][finger])
            ) / delta
            normal_gain, normal_alignment = _normal_compensation(
                state, probe_state, finger
            )
        rows[finger] = {
            'metric_m': metric,
            'sensitivity_m_per_deg': max(sensitivity, DYNAMIC_MIN_SENSITIVITY_M_PER_DEG),
            # Positive means curling raises signed clearance (away from the
            # surface); negative means this command direction would deepen it.
            'signed_sensitivity_m_per_deg': float(signed_sensitivity),
            'normal_gain': float(normal_gain),
            'normal_alignment': normal_alignment,
        }
    finite = [row['metric_m'] for row in rows.values() if np.isfinite(row['metric_m'])]
    reference = min(finite) if finite else 0.0
    for row in rows.values():
        error = max(0.0, row['metric_m'] - reference) if np.isfinite(row['metric_m']) else 0.0
        row['error_m'] = error
        row['delay_deg'] = min(float(max_delay_deg), error / row['sensitivity_m_per_deg'])
    order = sorted(
        fingers,
        key=lambda finger: (
            rows[finger]['metric_m'] if np.isfinite(rows[finger]['metric_m']) else float('inf'),
            tie_rank.get(finger, len(tie_rank)),
        ),
    )
    return state, order, {finger: rows[finger]['delay_deg'] for finger in fingers}, rows


def prepare_exit(*, q, commands, fingers, lower, upper, measure, safe,
                 base_clearance, skin_clearance=.006, base_margin=.015,
                 max_lift_deg=15., max_curl_extra_pct=12.,
                 base_guard_enabled=True, stop_on_base=False,
                 base_contact_tolerance=.00030):
    """After earned contact/release, curl until surface clearance or base stop.

    Base clearance is an optional legacy planning guard, NOT physical contact.
    stop_on_base is retained for call compatibility but cannot waive clearance.
    A physical base stop is handled exclusively by the runtime event callback.
    Every candidate is checked over its whole BackSurface sweep.
    """
    q = np.asarray(q, dtype=float)
    commands = dict(commands)
    initial = measure(q, commands)
    lagging = [f for f in fingers if initial['signed'][f] < skin_clearance]
    last_failure = 'NO_CLEAR_ENDPOINT'
    evaluated = 0
    # Keep the already-clear fingers fixed, allowing late fingers to finish
    # their own release instead of stopping all fingers at the first release.
    extras = [v for v in (3., 6., 9., 12.) if v <= max_curl_extra_pct] + [0.]
    for j4_deg in (0., 1., -1., 2., -2., 4., -4., 6., -6.):
        for lift_deg in np.arange(0., max_lift_deg+.001, .5):
            cq = q.copy()
            cq[3] += np.radians(j4_deg)
            cq[4] += np.radians(lift_deg)
            if np.any(cq < np.asarray(lower)-1e-10) or np.any(cq > np.asarray(upper)+1e-10):
                continue
            cq = np.clip(cq, lower, upper)
            for extra in extras:
                # ``fingers`` is the required scratch-quality set and may omit
                # an optional near-base pinky.  Collision/base checks still
                # require a complete hand command dictionary.  Preserve every
                # command and modify only required lagging fingers.
                cf = dict(commands)
                for f in fingers:
                    cf[f] = min(100.,commands[f]+extra) if f in lagging else commands[f]
                state = measure(cq, cf)
                evaluated += 1
                gap, link = base_clearance(cq, cf)
                if not state['valid'] or not all(
                        np.isfinite(state['signed'][f]) and state['signed'][f] >= skin_clearance
                        for f in fingers):
                    continue
                if (base_guard_enabled
                        and (not np.isfinite(gap) or gap < base_margin)):
                    last_failure = 'BASE_RESERVE'
                    continue
                ok, reason = safe(q, commands, cq, cf)
                if not ok:
                    last_failure = str(reason)
                    continue
                state = dict(state, base_gap=gap, base_link=link)
                return dict(q=cq, commands=cf, state=state, evaluated=evaluated,
                            j5_lift_deg=float(lift_deg), j4_trim_deg=float(j4_deg),
                            extra_curl_pct={f: cf[f]-commands[f] for f in fingers},
                            base_contact_stop=False, base_contact_link=None)
    raise RuntimeError('ALL_FINGER_EXIT_NO_SAFE_RESERVE|last='+last_failure
                       +'|evaluated='+str(evaluated)
                       +'|initial_signed_mm='+str({f:round(initial['signed'][f]*1000.,3) for f in fingers}))


def roll(*, q, commands, fingers, lower, upper, measure, qualified,
         accept, safe, speed=120., arm_speed=30., max_steps=96, seconds=25.,
         finish=None, depth_max=None, released=None, base_clearance=None,
         acquiring=None, contact_objective=None, target_depth=None, cooperate=None,
         optional_fingers=(), optional_patience_steps=18, optional_patience_s=18.0,
         stop_when_qualified=False, stop_after_qualification=False,
         base_guard_enabled=True,
         sequential_handoff=False, priority_order=None, primary_finger=None,
         assist_target_depth=None, assist_quality=None, mandatory_fingers=(),
         freeze_priority_order=False):
    """Advance one measured contact focus at a time and preserve handoffs.

    measure supplies signed surface distance and hard geometric checks. accept
    records contact motion for diagnostics, while the caller decides the hard
    per-finger completion rule. safe checks an entire segment before accept.
    No depth or collision failure is converted into an airborne full-curl
    success. With ``sequential_handoff=True`` the dynamically selected focus
    stays locked until it qualifies; during scratch handoff a completed finger
    curls away at a rate matched to the focus down-pressure, and the caller's
    release tail handles any remaining final curl.
    """
    started = time.perf_counter()
    deadline = started + seconds
    # ``qualified`` owns the caller's hard completion rule. target_depth is a
    # preferred optimizer setpoint only; it must not turn the nominal 3 mm
    # value into a harder gate than the caller's 1--4 mm band.
    if target_depth is None:
        target_depth_m = min(0.0015, float(depth_max)) if depth_max is not None else 0.0015
    else:
        target_depth_m = float(target_depth)
    if not np.isfinite(target_depth_m) or target_depth_m <= 0.0:
        raise ValueError('CONTACT_TARGET_DEPTH_INVALID')
    if depth_max is not None and target_depth_m > float(depth_max):
        raise ValueError('CONTACT_TARGET_DEPTH_EXCEEDS_CAP')
    if assist_target_depth is None:
        assist_target_depth_m = target_depth_m
    else:
        assist_target_depth_m = float(assist_target_depth)
    if (not np.isfinite(assist_target_depth_m)
            or assist_target_depth_m <= 0.0
            or (depth_max is not None and assist_target_depth_m > float(depth_max))):
        raise ValueError('CONTACT_ASSIST_TARGET_DEPTH_INVALID')
    anchor = np.asarray(q, dtype=float).copy()
    q = anchor.copy()
    commands = dict(commands)
    command_names = tuple(commands)
    optional_fingers = frozenset(optional_fingers)
    mandatory_fingers = frozenset(mandatory_fingers)
    if not mandatory_fingers:
        mandatory_fingers = frozenset(f for f in fingers if f not in optional_fingers)
    optional_only_steps = 0
    optional_only_started = None
    raw_measure = measure
    measurement_cache = {}
    def measure(cq, cf):
        # Exact keys only: never quantize the finite differences used by the
        # optimizer. The contact state can change on accept, so cache is local
        # to a single rollout step, not reused across steps or live states.
        key = (np.asarray(cq, dtype=float).tobytes(),
               tuple(float(cf[f]) for f in command_names))
        if key not in measurement_cache:
            measurement_cache[key] = raw_measure(cq, cf)
        return measurement_cache[key]
    physical_lower, physical_upper = np.asarray(lower).copy(), np.asarray(upper).copy()
    support_tiers = (SUPPORT_DEG, np.array([6.,6.,6.,10.,15.]),
                     np.array([12.,12.,12.,15.,20.]))
    support_tier = 0
    deferred = set()
    lower = np.maximum(physical_lower, anchor-np.radians(support_tiers[support_tier]))
    upper = np.minimum(physical_upper, anchor+np.radians(support_tiers[support_tier]))
    initial = measure(q, commands)
    # The first 1 mm crossing is only an acquisition handoff.  It protects the
    # first contacting finger from being driven too deeply while other fingers
    # approach, but it is not the final depth target.  Every rollout step
    # recomputes both contact quality and target-depth completion from current
    # BackSurface geometry.
    initially_acquiring = {
        f: bool(acquiring(f)) if acquiring is not None else False for f in fingers
    }
    initial, measured_order, initial_delays, initial_schedule = _dynamic_schedule(
        measure, q, commands, fingers, tie_order=fingers, current_state=initial)
    # The staging/acquisition order is the cycle ownership order. Geometry is
    # still re-measured every step for sensitivity/delay/minimum prediction,
    # but wrist motion must not reshuffle who completes next.
    tie_order = [f for f in (priority_order or measured_order) if f in fingers]
    for f in measured_order:
        if f not in tie_order:
            tie_order.append(f)
    order = list(tie_order if freeze_priority_order else measured_order)
    first_owner = order[0] if order else None
    if primary_finger not in order:
        primary_finger = next((f for f in order if f in mandatory_fingers),
                              order[0] if order else None)
    def is_qualified(finger):
        # Historical/contact-integrated quality may be used for final evidence,
        # but only the pinky may use the relaxed assist predicate. All three
        # core fingers keep the caller's normal 1--4 mm depth quality rule.
        if finger in optional_fingers and assist_quality is not None:
            return bool(assist_quality(finger))
        return bool(qualified(finger))
    def goal_depth(finger):
        return assist_target_depth_m if finger in optional_fingers else target_depth_m
    def live_handoff_ready(finger, state):
        signed = float(state['signed'][finger])
        if not np.isfinite(signed):
            return False
        depth = max(0.0, -signed)
        if finger in mandatory_fingers:
            return bool(is_qualified(finger) and
                        depth >= LIVE_CORE_DEPTH_MIN_M - 1e-12 and
                        depth <= LIVE_CORE_DEPTH_MAX_M + 1e-12)
        # Optional pinky still tries the normal 1--4 mm band first. Its local
        # optimum is handled only after every mandatory finger is complete.
        return bool(is_qualified(finger) and depth <= LIVE_CORE_DEPTH_MAX_M + 1e-12)

    def handoff_release_target_reached(state):
        """Return whether every completed finger has reached +20 mm clear.

        ``released()`` is intentionally a contact-state predicate (it becomes
        true just after the fingertip leaves the skin).  It is therefore not a
        suitable endpoint for the continuous handoff release: using it as the
        ``cleared`` test used to remove an old finger at roughly +0.5 mm and
        let the rollout return while that finger was still near the surface.
        Keep the two notions separate and require the explicit release target
        here.  The normal ``safe``/swept checks remain authoritative for every
        step toward the target.
        """
        if not handoff_depth_limits:
            return True
        signed = state.get('signed', {}) if isinstance(state, dict) else {}
        for finger in handoff_depth_limits:
            try:
                value = float(signed[finger])
            except (KeyError, TypeError, ValueError):
                return False
            if not np.isfinite(value) or value < HANDOFF_RELEASE_TARGET_SIGNED_M - 1e-6:
                return False
        return True
    print('[V103 PRIMARY DEPTH ANCHOR]',
          'finger=', primary_finger,
          'target_mm=', round(target_depth_m*1000., 3),
          'assist_target_mm=', round(assist_target_depth_m*1000., 3),
          'assist_gate=', 'PINKY_ONLY_RELAXED' if assist_quality is not None else 'CALLER_GATE',
          'core_live_handoff_mm=1_to_4', 'order_frozen=', bool(freeze_priority_order))
    if any(initially_acquiring.values()):
        print('[V103 PENDING ACQUISITION PRIORITY]',
              'order=', order,
              'untouched=', [f for f in order if initially_acquiring[f]],
              'measured_error_mm=', {f:round(initial_schedule[f]['metric_m']*1000.,3) for f in order},
              'dynamic_delay_deg=', {f:round(initial_delays[f],2) for f in order},
              'fixed_phase_offset=DISABLED',
              'safety_gates_unchanged=True')
    frames, focus_log = [], []
    focus = None
    optional_focus_stall_steps = {}
    optional_focus_last_depth = {}
    optional_focus_attempt_steps = {}
    # ``order`` is intentionally re-measured every step, but the active
    # finger is locked until it has reached the caller's qualification gate.
    # Without this lock a tiny FK/noise change could reorder two near-equal
    # tips between steps, causing the next finger to start before the current
    # one had finished its depth handoff.  That was the source of the old
    # ring->middle overlap and the large ring tracking correction at runtime.
    locked_focus = None
    completed_handoffs = []
    # ``handoff_depth_limits`` is populated only after the focus has passed
    # the caller's measured qualification gate.  Until then the focus is
    # allowed to approach its own target normally.
    handoff_depth_limits = {}
    handoff_transfer = None
    if sequential_handoff and first_owner is not None:
        # Bootstrap is allowed to establish only the first, terrain-derived
            # owner.  A later finger that happens to enter the 1--4 mm band
        # because of the shared palm pose is not completed: it must still go
        # through PREPARING -> SCRATCHING -> MINIMUM -> RELEASING.  Registering
        # every qualified finger here was the source of skipped handoffs and
        # one-finger-looking trajectories in the old hybrid path.
        if live_handoff_ready(first_owner, initial):
            depth = max(0.0, -float(initial['signed'][first_owner]))
            handoff_depth_limits[first_owner] = min(
                float(depth_max) if depth_max is not None else float('inf'),
                depth + HANDOFF_DEPTH_DRIFT_M,
            )
            completed_handoffs.append(first_owner)
            print('[V103 HANDOFF BASELINE]',
                  'completed=', completed_handoffs,
                  'depth_mm=', {
                      first_owner: round(
                          max(0., -float(initial['signed'][first_owner])) * 1000., 3)
                  },
                  'held_max_mm=', {
                      first_owner: round(handoff_depth_limits[first_owner] * 1000., 3)
                  },
                  'bootstrap_only_first_owner=True')

    def pinky_local_optimum_result(pending, state):
        """Finish a 4F wave without making assist misses block it.

        Every finger remains in ``fingers`` and therefore in every command
        frame. This is a bounded local-best exit, not a 3F fallback: the
        terrain-derived primary finger must already be qualified, and the
        existing safe/depth-cap checks still authorize every accepted segment.
        """
        relaxed = [f for f in pending if f in optional_fingers]
        print('[V103 ASSIST LOCAL OPTIMUM]',
              'pending=', relaxed,
              'signed_mm=', {f: round(float(state['signed'][f])*1000., 3) for f in relaxed},
              'commands=', {f: round(float(commands[f]), 2) for f in fingers},
              'primary_required_around_3mm=True',
              'assist_preferred_around_3mm=True',
              'assist_normal_gate_mm=1_to_4',
              'three_finger_fallback=DISABLED')
        return dict(q=q, commands=commands, order=order, frames=frames,
                    seconds=time.perf_counter()-started, focuses=focus_log,
                    depth_complete=True, local_optimum=True,
                    relaxed_fingers=relaxed)

    def mandatory_quality_ready(state):
        """Return whether the core fingers already have safe quality evidence.

        A local budget stop for the optional pinky is allowed only after every
        mandatory finger has earned the caller's measured quality predicate.
        The predicate is cumulative (``stats``), while ``state`` supplies a
        fresh safety check at the current pose so a stale quality record can
        never authorize an over-deep or invalid endpoint.
        """
        if not mandatory_fingers or not isinstance(state, dict):
            return False
        if not bool(state.get('valid', True)):
            return False
        signed_map = state.get('signed', {})
        for finger in mandatory_fingers:
            if not is_qualified(finger):
                return False
            try:
                signed = float(signed_map[finger])
            except (KeyError, TypeError, ValueError):
                return False
            if not np.isfinite(signed):
                return False
            # Quality may have been earned on an earlier accepted contact
            # segment and the finger may now be releasing.  Only the current
            # hard cap and validity are required here; the final replay still
            # validates the complete four-channel trajectory.
            if depth_max is not None and -signed > float(depth_max) + 1e-9:
                return False
        return True

    def optional_budget_local_optimum(pending, state, reason):
        """Convert an optional-channel budget stop into a bounded 4F result."""
        optional_pending = [f for f in pending if f in optional_fingers]
        if not optional_pending or not mandatory_quality_ready(state):
            return None
        print('[V103 OPTIONAL BUDGET LOCAL OPTIMUM]',
              'focus=', focus,
              'pending=', optional_pending,
              'reason=', reason,
              'mandatory_quality_ready=True',
              'action=HOLD_OPTIONAL_CHANNEL_CONTINUE_4F')
        return pinky_local_optimum_result(optional_pending, state)

    for step in range(max_steps):
        measurement_cache.clear()
        current_state, measured_order, dynamic_delays, dynamic_schedule = _dynamic_schedule(
            measure, q, commands, fingers, tie_order=tie_order,
        )
        # Geometry is fresh every step, but completion ownership follows the
        # frozen staging order. Later fingers may already be moving/contacting;
        # they cannot complete ahead of the current focus.
        if freeze_priority_order:
            order = list(tie_order)
            measured_for_log = list(measured_order)
        else:
            order = list(measured_order)
            measured_for_log = list(measured_order)
        pending = [f for f in order if f not in completed_handoffs]
        release_tail_pending = bool(
            sequential_handoff
            and not pending
            and handoff_depth_limits
            and not handoff_release_target_reached(current_state)
        )
        optional_only_now = bool(
            pending and set(pending).issubset(optional_fingers)
        )
        if time.perf_counter() >= deadline:
            budget_focus = focus if focus in pending else (pending[0] if pending else None)
            if (budget_focus in optional_fingers
                    and (focus is None or focus in optional_fingers)):
                local = optional_budget_local_optimum(
                    pending, current_state, 'step_deadline')
                if local is not None:
                    return local
            if optional_only_now:
                print('[V103 OPTIONAL BUDGET LOCAL OPTIMUM]',
                      'pending=', pending,
                      'action=ACCEPT_SAFE_LOCAL_OPTIMUM')
                return pinky_local_optimum_result(pending, current_state)
            raise RuntimeError('CONTACT_ROLLOUT_BUDGET|' + str(focus))
        if focus is not None and pending:
            print('[V103 FROZEN ORDER UPDATE]' if freeze_priority_order else '[V103 DYNAMIC ORDER UPDATE]',
                  'focus_locked=', focus if focus in pending else None,
                  'completion_order=', order,
                  'measured_order=', measured_for_log,
                  'measured_error_mm=', {
                      f: round(float(dynamic_schedule[f]['metric_m']) * 1000., 3)
                      for f in order
                  },
                  'handoff_reselect=', 'OFF' if freeze_priority_order else 'ON')
        # A V103 caller can use qualification as a phase handoff: return the
        # measured qualified pose before this controller starts its legacy
        # release-tail/J5 compensation.  The caller then appends the real
        # four-finger curl continuation and performs measured release itself.
        if (not pending and stop_when_qualified and stop_after_qualification):
            print('[V103 QUALIFICATION HANDOFF]',
                  'release_tail_skipped=True',
                  'next=CALLER_FULL_FOUR_FINGER_CURL')
            return dict(q=q, commands=commands, order=order, frames=frames,
                        seconds=time.perf_counter()-started, focuses=focus_log,
                        depth_complete=True)
        # Scratch completion is the measured depth requirement, not a 100%
        # finger-command target.  The caller may deliberately end the contact
        # stroke here and plan a separate curl-away -> arm-lift -> open tail.
        if (not pending and stop_when_qualified
                and not release_tail_pending and finish is None):
            return dict(q=q, commands=commands, order=order, frames=frames,
                        seconds=time.perf_counter()-started, focuses=focus_log,
                        depth_complete=True)
        if pending and set(pending).issubset(optional_fingers):
            if optional_only_started is None:
                optional_only_started = time.perf_counter()
                optional_only_steps = 0
            if (optional_only_steps >= int(optional_patience_steps)
                    or time.perf_counter() - optional_only_started >= float(optional_patience_s)):
                return pinky_local_optimum_result(pending, current_state)
            optional_only_steps += 1
        else:
            optional_only_started = None
            optional_only_steps = 0
        # A safe accepted step is not necessarily useful contact progress.
        # Track measured penetration for the active optional assistant and
        # yield to another pending assistant after a short no-progress run.
        # This is deliberately separate from the global optional patience
        # budget: the latter is the final local-best bound, while this handoff
        # keeps all four channels from being starved by one flat/tangential tip.
        if focus in optional_fingers and focus in pending:
            depth = max(0.0, -float(current_state['signed'][focus]))
            previous = optional_focus_last_depth.get(focus)
            if previous is None or depth > previous + OPTIONAL_FOCUS_STALL_TOLERANCE_M:
                optional_focus_stall_steps[focus] = 0
            else:
                optional_focus_stall_steps[focus] = (
                    optional_focus_stall_steps.get(focus, 0) + 1
                )
            optional_focus_last_depth[focus] = depth
            if optional_focus_stall_steps[focus] >= OPTIONAL_FOCUS_STALL_STEPS:
                deferred.add(focus)
                optional_focus_stall_steps[focus] = 0
                print('[V103 OPTIONAL FOCUS STALL]',
                      'focus=', focus,
                      'depth_mm=', round(depth * 1000., 3),
                      'stall_steps=', OPTIONAL_FOCUS_STALL_STEPS,
                      'action=TRY_NEXT_ASSIST')
        if not pending and frames and finish is not None:
            # ``finish`` is an evidence/quality check, not permission to cut
            # the release tail short.  A completed contact may already be
            # outside the skin (the caller's ``released`` predicate), while
            # still being far below the requested +20 mm clearance.  Keep
            # rolling release-only frames until the explicit target is met.
            if release_tail_pending:
                completion = None
                print('[V103 HANDOFF RELEASE HOLD]',
                      'waiting_for_clearance_mm=', round(
                          HANDOFF_RELEASE_TARGET_SIGNED_M * 1000., 3),
                      'current_signed_mm=', {
                          f: round(float(current_state['signed'].get(f, float('nan'))) * 1000., 3)
                          for f in handoff_depth_limits
                      })
            else:
                completion = finish(q, commands)
            if completion:
                return dict(q=q,commands=commands,order=order,frames=frames,
                            seconds=time.perf_counter()-started,focuses=focus_log,
                            early_exit=completion)
        if not pending and all(commands[f] >= 100.-1e-8 for f in fingers):
            return dict(q=q, commands=commands, order=order, frames=frames,
                        seconds=time.perf_counter()-started, focuses=focus_log)
        # A pinky that appears before a mandatory core finger in the frozen
        # geometry order still gets a real ownership attempt, but it must not
        # consume the remaining rollout budget and starve that mandatory core
        # finger.  After a short bounded attempt, defer pinky at this pose; it
        # keeps moving as a background channel and will be revisited after the
        # mandatory fingers qualify.
        if pending:
            first_pending = pending[0]
            mandatory_after = any(
                f in mandatory_fingers for f in pending[1:]
            )
            if (first_pending in optional_fingers and mandatory_after
                    and optional_focus_attempt_steps.get(first_pending, 0)
                        >= OPTIONAL_BEFORE_MANDATORY_MAX_ACCEPTED_STEPS):
                deferred.add(first_pending)
                print('[V103 OPTIONAL BEFORE CORE YIELD]',
                      'optional=', first_pending,
                      'accepted_steps=', optional_focus_attempt_steps.get(first_pending, 0),
                      'next_mandatory=', next((f for f in pending[1:] if f in mandatory_fingers), None))

        # Failed focus choices are deferred only at this unchanged pose.
        # Re-selecting pending[0] here used to undo the fallback below and
        # solve the same failed finger repeatedly until the step budget ran out.
        # The dynamic order chooses the next finger only at a handoff.  While
        # the current finger is still pending, keep it as focus even if a
        # later measurement says another tip is marginally shallower.  A
        # failed branch may explicitly defer that focus below, which is the
        # only normal way to yield at an unchanged pose.
        if sequential_handoff and freeze_priority_order:
            if pending:
                available = [f for f in pending if f not in deferred]
                new_focus = available[0] if available else pending[0]
            else:
                new_focus = None
        elif sequential_handoff and locked_focus in pending and locked_focus not in deferred:
            new_focus = locked_focus
        else:
            available = [f for f in pending if f not in deferred]
            new_focus = available[0] if available else (pending[0] if pending else None)
        focus_changed = new_focus != focus
        if focus_changed and new_focus is not None:
            focus_log.append(new_focus)
        focus = new_focus
        if sequential_handoff:
            locked_focus = focus
        # Finished fingers may now curl away under the rate-matched handoff
        # branch below. Waiting fingers keep their chance to scratch; they are
        # never marched past their minimum unqualified.  ``cleared`` means the
        # explicit +20 mm handoff endpoint, not merely leaving the skin.
        proposed = dict(commands)
        cleared = {
            f for f in handoff_depth_limits
            if float(current_state['signed'].get(f, float('-inf'))) >=
                HANDOFF_RELEASE_TARGET_SIGNED_M - 1e-6
        }
        if release_tail_pending:
            # Once the last successor has qualified, keep a synthetic
            # release-only transfer alive.  This reuses the same measured
            # derivative/swept-safety path as normal handoffs instead of
            # jumping directly to the caller's short generic exit reserve.
            release_fingers = [
                f for f in handoff_depth_limits if f not in cleared
            ]
            handoff_transfer = {
                'focus': None,
                'focus_start_depth': 0.0,
                'release_start_signed': {
                    f: float(current_state['signed'][f])
                    for f in release_fingers
                },
                'release_fingers': release_fingers,
                'release_tail': True,
            }
            print('[V103 HANDOFF RELEASE TAIL]',
                  'release_fingers=', release_fingers,
                  'target_clearance_mm=', round(
                      HANDOFF_RELEASE_TARGET_SIGNED_M * 1000., 3))
        if sequential_handoff and focus_changed and focus is not None:
            release_fingers = [
                f for f in handoff_depth_limits
                if f != focus and f not in pending and f not in cleared
            ]
            handoff_transfer = {
                'focus': focus,
                'focus_start_depth': max(0., -float(current_state['signed'][focus])),
                'release_start_signed': {
                    f: float(current_state['signed'][f])
                    for f in release_fingers
                },
                'release_fingers': release_fingers,
            }
            if release_fingers:
                print('[V103 HANDOFF TRANSFER START]',
                      'focus=', focus,
                      'focus_start_mm=', round(handoff_transfer['focus_start_depth']*1000., 3),
                      'release_fingers=', release_fingers,
                      'release_start_signed_mm=', {
                          f: round(handoff_transfer['release_start_signed'][f]*1000., 3)
                          for f in release_fingers
                      },
                      'target_depth_mm=', round(goal_depth(focus)*1000., 3),
                      'release_target_clearance_mm=',
                      round(HANDOFF_RELEASE_TARGET_SIGNED_M*1000., 3))
        all_tracked_cleared = bool(
            not handoff_depth_limits
            or all(f in cleared for f in handoff_depth_limits)
        )
        hold_cleared = bool(pending or not all_tracked_cleared)
        clearance_only = bool(base_guard_enabled and not pending and all_tracked_cleared
                              and finish is not None and base_clearance is not None
                              and min(current_state['signed'].values()) < .0015
                              and base_clearance(q, commands)[0] < .030)
        focus_is_acquiring = bool(
            focus is not None and acquiring is not None and acquiring(focus)
        )
        # Re-scan every pending finger at the CURRENT palm pose. The first
        # owner has already established the fixed-palm ~3 mm reference during
        # acquisition; later owners use these scans for rolling ETA and wrist
        # deficit prediction. No fixed phase delay or fixed wrist speed is
        # carried from an earlier frame.
        minimum_info_by_finger = {}
        for candidate in pending:
            info = _scan_natural_minimum(measure, q, commands, candidate)
            if info is not None:
                minimum_info_by_finger[candidate] = info
        minimum_info = minimum_info_by_finger.get(focus)
        first_owner_free_phase = bool(
            focus is not None
            and focus == first_owner
            and focus not in completed_handoffs
        )

        focus_increment = 0.0
        if focus is not None:
            available_focus = max(0.0, 100.0 - float(commands[focus]))
            max_focus_step = (
                FROZEN_FOCUS_ACQUIRE_MAX_INCREMENT_PCT
                if focus_is_acquiring else FROZEN_FOCUS_CONTACT_MAX_INCREMENT_PCT
            )
            if minimum_info is not None:
                remaining_to_min = max(
                    0.0,
                    float(minimum_info['command_pct']) - float(commands[focus]),
                )
                if remaining_to_min > 1e-9:
                    # Keep the current owner moving continuously toward its
                    # minimum, but do not jump over it in one command frame.
                    nominal_steps = max(
                        1.0,
                        np.ceil(remaining_to_min / max(max_focus_step, 1e-6)),
                    )
                    focus_increment = min(
                        max_focus_step,
                        remaining_to_min,
                        max(0.25, remaining_to_min / nominal_steps),
                    )
                else:
                    # After the minimum the finger CONTINUES curling. This is
                    # the actual scratching/release stroke, not a static press.
                    focus_increment = min(
                        max_focus_step,
                        max(0.75, 0.04 * available_focus),
                    )
            else:
                focus_increment = min(
                    max_focus_step,
                    max(0.75, 0.04 * available_focus),
                )

        if release_tail_pending:
            # There is no successor focus during the final release-only tail,
            # so the normal focus cadence would be zero and completed fingers
            # would never receive another curl command.  Use a bounded,
            # safety-checked release cadence; the per-finger branch below
            # still limits each finger by its measured lift derivative and
            # the full swept validator.
            focus_increment = min(
                RELEASE_TAIL_MAX_INCREMENT_PCT,
                max(0.75, 0.04 * max(
                    0.0,
                    max((100.0 - float(commands[f]))
                        for f in handoff_depth_limits),
                )),
            )

        rolling_wrist_plan = None
        early_wrist_support = False
        predicted_minimum_requires_arm = False
        if (focus is not None and minimum_info is not None
                and not first_owner_free_phase):
            rolling_wrist_plan = _rolling_wrist_compensation_plan(
                measure, q, commands, focus, minimum_info,
                lower, upper, goal_depth(focus), focus_increment,
            )
            predicted_minimum_requires_arm = bool(
                focus in mandatory_fingers
                and float(minimum_info.get('depth_m', 0.0))
                    < LIVE_CORE_DEPTH_MIN_M - 1e-9
            )
            early_wrist_support = bool(
                (rolling_wrist_plan is not None
                 and rolling_wrist_plan.get('required', False)
                 and abs(float(rolling_wrist_plan.get('step_deg', 0.0))) > 1e-6)
                or predicted_minimum_requires_arm
            )

        if minimum_info is not None:
            print('[V103 MINIMUM SCAN]',
                  'focus=', focus,
                  'first_owner_free_fixed_palm=', first_owner_free_phase,
                  'live_depth_mm=', round(minimum_info['current_depth_m']*1000., 3),
                  'minimum_command_pct=', round(minimum_info['command_pct'], 3),
                  'minimum_depth_mm=', round(minimum_info['depth_m']*1000., 3),
                  'command_to_minimum_pct=', round(minimum_info['command_to_minimum_pct'], 3),
                  'focus_increment_pct=', round(float(focus_increment), 3),
                  'rolling_wrist=', None if rolling_wrist_plan is None else {
                      'required': bool(rolling_wrist_plan.get('required', False)),
                      'deficit_mm': round(float(rolling_wrist_plan.get('deficit_m',0.0))*1000.,3),
                      'joint': rolling_wrist_plan.get('joint'),
                      'gain_mm_per_deg': round(float(rolling_wrist_plan.get('gain_m_per_deg',0.0))*1000.,4),
                      'total_deg': round(float(rolling_wrist_plan.get('total_deg',0.0)),3),
                      'step_deg': round(float(rolling_wrist_plan.get('step_deg',0.0)),3),
                      'eta_steps': round(float(rolling_wrist_plan.get('eta_steps',1.0)),2),
                  },
                  'predicted_minimum_requires_arm=', bool(predicted_minimum_requires_arm))
        handoff_release_schedule = {}
        handoff_release_targets = {}
        handoff_progress = None
        handoff_live_depth_m = None
        if (sequential_handoff and handoff_transfer is not None
                and (handoff_transfer.get('focus') == focus
                     or handoff_transfer.get('release_tail', False))
                and (focus is None or focus in dynamic_schedule)):
            # Release is driven ONLY by the successor's measured live depth.
            # Do not predict the next depth from curl sensitivity: that old
            # prediction released ring/middle before index had actually made
            # contact on 2444.  A small 5% target motion is allowed while the
            # successor is still at 0 mm so the old finger can counter common
            # wrist pressure without disappearing from the skin.
            if handoff_transfer.get('release_tail', False):
                # The successor has already completed; release all tracked
                # fingers directly toward +20 mm.  There is no new focus whose
                # live depth should throttle this tail.
                handoff_live_depth_m = LIVE_CORE_DEPTH_MIN_M
                handoff_progress = 1.0
            else:
                current_focus_depth = max(0., -float(current_state['signed'][focus]))
                handoff_live_depth_m = current_focus_depth
                # The live core floor is 1 mm in the relaxed policy. Keep a
                # single monotonic ramp from no measured contact to that floor;
                # the previous two-band formula became a dead branch and a
                # near-zero denominator when the floor was lowered from 2 mm.
                if current_focus_depth <= 0.0:
                    target_progress = 0.05
                elif current_focus_depth < LIVE_CORE_DEPTH_MIN_M:
                    target_progress = 0.05 + 0.95 * (
                        current_focus_depth / max(LIVE_CORE_DEPTH_MIN_M, 1e-9)
                    )
                else:
                    target_progress = 1.0
                handoff_progress = float(np.clip(target_progress, 0.0, 1.0))
            for f in handoff_transfer.get('release_fingers', ()):
                release_start = float(handoff_transfer['release_start_signed'][f])
                handoff_release_targets[f] = (
                    release_start * (1. - handoff_progress)
                    + HANDOFF_RELEASE_TARGET_SIGNED_M * handoff_progress
                )
        for f in fingers:
            available = max(0.0, 100.0 - float(commands[f]))
            if f == focus:
                increment = focus_increment
            elif (sequential_handoff and f in pending and f != focus):
                # Rolling ETA follower control. Every pending finger has been
                # moving since the beginning of the scratch. Do not use a
                # fixed rank speed; recompute how much command remains to its
                # current-palm natural minimum and choose a per-frame speed so
                # successive minima arrive only ~one keyframe apart.
                try:
                    focus_rank = order.index(focus) if focus in order else 0
                    follower_rank = order.index(f)
                    rank_ahead = max(1, follower_rank - focus_rank)
                except Exception:
                    rank_ahead = 3
                follower_info = minimum_info_by_finger.get(f)
                focus_info = minimum_info_by_finger.get(focus)
                signed_rate = float(dynamic_schedule.get(f, {}).get(
                    'signed_sensitivity_m_per_deg', 0.0))
                depth = max(0.0, -float(current_state['signed'][f]))
                if follower_info is not None and focus_info is not None:
                    focus_remaining = max(
                        0.0,
                        float(focus_info['command_pct']) - float(commands[focus]),
                    )
                    focus_eta = max(
                        1.0,
                        focus_remaining / max(float(focus_increment), 0.25),
                    )
                    target_eta = max(
                        1.0,
                        focus_eta + float(rank_ahead) * ROLLING_MINIMUM_GAP_STEPS,
                    )
                    follower_remaining = max(
                        0.0,
                        float(follower_info['command_pct']) - float(commands[f]),
                    )
                    if follower_remaining > 1e-9:
                        increment = min(
                            4.0,
                            follower_remaining,
                            max(0.10, follower_remaining / target_eta),
                        )
                    else:
                        # If a later finger reached/passed its minimum a little
                        # early, keep it moving rather than freezing it. Post-
                        # minimum positive signed sensitivity means continued
                        # curl is naturally lifting/unloading that fingertip.
                        increment = (
                            min(2.5, max(0.65, float(focus_increment)))
                            if signed_rate >= 0.0 else 0.0
                        )
                else:
                    increment = min(2.0, max(0.10, float(focus_increment) * 0.5))

                # Keep only the real planned-depth headroom as a hard normal
                # limit. This is not a rank reserve and does not stop a future
                # finger merely because it is already in the useful 1--4 mm
                # band.
                if signed_rate < -1e-12:
                    room = max(0.0, (float(depth_max) if depth_max is not None else 0.004)
                               - 0.00015 - depth)
                    if room <= 1e-9:
                        increment = 0.0
                    else:
                        increment = min(increment, room / (-signed_rate))
            elif (sequential_handoff and f in handoff_depth_limits
                  and f not in pending and f not in cleared):
                # The next focus finger is pressing down while this completed
                # finger curls away. Match physical depth rates using fresh
                # signed-depth finite differences. If positive curl does not
                # lift this finger at the current pose, do not guess: keep it
                # still and let the swept/depth guards reject unsafe branches.
                lift_rate = max(
                    0.0,
                    float(dynamic_schedule.get(f, {}).get(
                        'signed_sensitivity_m_per_deg', 0.0)),
                )
                release_target = handoff_release_targets.get(f)
                current_signed = float(current_state['signed'][f])
                release_needed = (
                    max(0., float(release_target) - current_signed)
                    if release_target is not None else 0.0
                )
                phase_release = release_needed / lift_rate if lift_rate > 1e-9 else 0.0
                # Couple release speed to the SUCCESSOR'S LIVE depth. When
                # the successor is still airborne/at 0 mm, old fingers only
                # curl away slowly enough to counter common wrist pressure.
                # Once the successor actually reaches 1 mm, full fast release
                # is allowed. Never use predicted successor progress here.
                live_fraction = float(np.clip(
                    (handoff_live_depth_m or 0.0) / LIVE_CORE_DEPTH_MIN_M,
                    0.0, 1.0,
                ))
                old_depth = max(0.0, -current_signed)
                urgent_fraction = float(np.clip(
                    (old_depth - HANDOFF_RELEASE_URGENT_START_M)
                    / max(HANDOFF_RELEASE_URGENT_FULL_M
                          - HANDOFF_RELEASE_URGENT_START_M, 1e-9),
                    0.0, 1.0,
                ))
                release_drive = max(live_fraction, urgent_fraction)
                rate_cap = focus_increment * (
                    HANDOFF_RELEASE_MIN_RATE_RATIO
                    + (HANDOFF_RELEASE_RATE_RATIO - HANDOFF_RELEASE_MIN_RATE_RATIO) * live_fraction
                    + (HANDOFF_RELEASE_URGENT_MAX_RATIO
                       - HANDOFF_RELEASE_RATE_RATIO) * urgent_fraction
                )
                if release_target is not None and lift_rate > 1e-9:
                    target_room = max(0., float(release_target) - current_signed)
                    target_increment = (
                        target_room / lift_rate if target_room > 0. else 0.0
                    )
                    release_increment = min(target_increment, rate_cap)
                else:
                    release_increment = 0.0
                # Keep the old finger moving continuously but bounded. The
                # successor's real 0->1 mm contact depth controls how quickly
                # the handoff opens, preventing old fingers from vanishing
                # before the new contact is established.
                increment = min(6.0, max(0.0, release_increment))
                handoff_release_schedule[f] = increment
            elif f not in pending and not sequential_handoff:
                # Legacy non-serial mode retains its historical moving release.
                delay_fraction = min(1.0, float(dynamic_delays[f]) / DYNAMIC_MAX_DELAY_DEG)
                nominal_increment = min(6.0, max(1.0, 0.08 * available))
                increment = nominal_increment * (1.0 - 0.50 * delay_fraction)
            else:
                increment = 0.0
            if (hold_cleared or clearance_only) and f not in pending and f in cleared:
                increment = 0.
            proposed[f] = min(100., commands[f]+increment)
        if focus and proposed[focus] <= commands[focus]+1e-9:
            # An optional finger that cannot make even a bounded increment is
            # not a reason to end the four-finger wave immediately.  Defer it
            # at this unchanged pose and give the other assistants a chance;
            # only when every pending assistant is blocked do we emit the
            # bounded local-optimum result.  This keeps the 4F command frame
            # intact while preventing one pinky/middle branch from vetoing the
            # remaining fingers.
            if (focus in optional_fingers
                    and not any(f not in optional_fingers for f in pending)):
                deferred.add(focus)
                alternatives = [f for f in pending if f not in deferred]
                if alternatives:
                    focus = alternatives[0]
                    focus_log.append(focus)
                    support_tier = 0
                    lower = np.maximum(
                        physical_lower, anchor-np.radians(support_tiers[support_tier]))
                    upper = np.minimum(
                        physical_upper, anchor+np.radians(support_tiers[support_tier]))
                    print('[V103 CONTACT ORDER FALLBACK] next=',focus,
                          'deferred=',sorted(deferred),
                          'reason=NO_OPTIONAL_INCREMENT')
                    continue
                return pinky_local_optimum_result(pending, current_state)
            raise RuntimeError('CONTACT_UNQUALIFIED_FULL_CURL|' + focus)

        print('[V103 DYNAMIC CONTACT SCHEDULE]',
              'step=', step,
              'order=', order,
              'error_mm=', {f:round(dynamic_schedule[f]['metric_m']*1000.,3) for f in order},
              'delay_deg=', {f:round(dynamic_delays[f],2) for f in order},
              'sensitivity_mm_per_deg=', {
                  f:round(dynamic_schedule[f]['sensitivity_m_per_deg']*1000.,4)
                  for f in order
              },
              'signed_sensitivity_mm_per_deg=', {
                  f:round(dynamic_schedule[f].get('signed_sensitivity_m_per_deg',0.)*1000.,4)
                  for f in order
              },
              'normal_alignment=', {
                  f: (None if dynamic_schedule[f].get('normal_alignment') is None
                      else round(float(dynamic_schedule[f]['normal_alignment']),3))
                  for f in order
              },
              'normal_gain=', {
                  f: round(float(dynamic_schedule[f].get('normal_gain',1.0)),3)
                  for f in order
              },
              'handoff_release_deg=', {
                  f:round(v,3) for f,v in handoff_release_schedule.items()
              },
              'handoff_progress=', None if handoff_progress is None else round(handoff_progress,3),
              'successor_live_depth_mm=', None if handoff_live_depth_m is None else round(handoff_live_depth_m*1000.,3),
              'handoff_release_target_mm=', {
                  f: round(v*1000.,3) for f,v in handoff_release_targets.items()
              },
              'fixed_phase_offset=DISABLED',
              'handoff=FROZEN_ORDER_COUPLED_BACKGROUND' if (sequential_handoff and freeze_priority_order) else ('handoff=SOFT_FOCUS_LOCKED_UNTIL_QUALIFIED_DYNAMIC_RESELECT' if sequential_handoff else
              'handoff=LEGACY_DYNAMIC'))
        solutions = []
        failures = []
        # Full triangle clearance is expensive. Its optimization residual is
        # identically zero far from base, so evaluate it only in the near zone.
        # Independent swept collision checks still run on EVERY proposal.
        near_base = False
        if base_guard_enabled and base_clearance is not None:
            gaps = [base_clearance(q, cf)[0] for cf in (commands, proposed)]
            if not np.all(np.isfinite(gaps)):
                raise RuntimeError('BASE_CLEARANCE_NONFINITE')
            near_base = min(gaps) < .025
        # J5 is the primary pressure/depth actuator.  Try it alone first so a
        # valid pressure step cannot be replaced by a numerically nicer J4
        # roll.  J4 is an assist for the same wrist step; the full arm is the
        # final fallback only after both wrist-only tiers fail.
        nominal = dict(proposed)
        moving = [f for f in fingers if nominal[f] > commands[f]+1e-8
                  and (f != focus or nominal[f]-commands[f] > .25+1e-8)]
        attempts = [(ids, False, not pending) for ids in (
            ((3, 4), (0, 1, 2, 3, 4)) if pending or clearance_only
            else ((), (3, 4), (0, 1, 2, 3, 4)))]
        if pending and sequential_handoff:
            # The first frozen owner is a hard fixed-palm phase. If it somehow
            # reaches rollout still unqualified, no wrist/arm branch may rescue
            # it: the IK/preposition itself must be rejected instead.
            if first_owner_free_phase:
                attempts = [((), False, True)]
                if moving:
                    attempts.append(((), True, True))
            else:
                # Later owners are rolling-predictive. Finger motion is always
                # available, but when the current natural-minimum prediction
                # says a 3 mm deficit remains, the finite-difference wrist plan
                # is evaluated in the SAME frame rather than waiting until the
                # finger bottoms out. Old completed fingers are coupled to that
                # arm candidate below so they curl away fast enough to avoid
                # being re-pressed.
                attempts = [((), False, True)]
                if moving:
                    attempts.append(((), True, True))
                attempts.append(((4,), False, True))
                attempts.append(((3, 4), False, True))
                if moving:
                    attempts.append(((3, 4), True, True))
                attempts.append(((0, 1, 2, 3, 4), False, True))
                if moving:
                    attempts.append(((0, 1, 2, 3, 4), True, True))
        elif pending:
            # Non-serial preview callers retain their historical wrist-first
            # search; the J5-primary handoff policy is for the production
            # sequential scratch path only.
            attempts.append(((3, 4), False, True))
            if moving:
                attempts.append(((3, 4), True, True))
            attempts.append(((0, 1, 2, 3, 4), False, True))
            if moving:
                attempts.append(((0, 1, 2, 3, 4), True, True))
        for indices, adaptive, shape_sweep in attempts:
            # Away from the natural minimum, keep the old fast J5-first early
            # exit. Near/after the minimum, continue evaluating J4/J5/full-arm
            # support so the controller does not accept a shallow J5-only step
            # merely because it was safe.
            if pending and shape_sweep and solutions and not early_wrist_support:
                break
            proposed = dict(nominal)
            lo = np.maximum(lower, q-np.radians([.6, .6, .6, 1.5, 2.]))
            hi = np.minimum(upper, q+np.radians([.6, .6, .6, 1.5, 2.]))
            rolling_target_q = None
            rolling_joint = None
            if (rolling_wrist_plan is not None
                    and rolling_wrist_plan.get('required', False)
                    and rolling_wrist_plan.get('joint') is not None):
                rolling_joint = int(rolling_wrist_plan['joint'])
                step_deg = float(rolling_wrist_plan.get('step_deg', 0.0))
                if rolling_joint in indices and abs(step_deg) > 1e-8:
                    rolling_target_q = q.copy()
                    rolling_target_q[rolling_joint] += np.radians(step_deg)
                    if step_deg > 0.0:
                        lo[rolling_joint] = max(lo[rolling_joint], q[rolling_joint])
                        hi[rolling_joint] = min(
                            hi[rolling_joint],
                            q[rolling_joint] + np.radians(max(0.10, abs(step_deg) * 1.25)),
                        )
                    else:
                        lo[rolling_joint] = max(
                            lo[rolling_joint],
                            q[rolling_joint] - np.radians(max(0.10, abs(step_deg) * 1.25)),
                        )
                        hi[rolling_joint] = min(hi[rolling_joint], q[rolling_joint])
            indices = tuple(i for i in indices if hi[i]-lo[i] > 1e-10)
            base_gap_at_q = 0.
            base_gradient = np.zeros(5)
            if base_guard_enabled and near_base:
                base_gap_at_q = base_clearance(q, proposed)[0]
                # Local finite-difference guidance avoids hundreds of full
                # STL queries inside least_squares. This approximation NEVER
                # authorizes motion: safe() still checks the real swept mesh.
                for i in indices:
                    h = min(np.radians(.05), hi[i]-q[i])
                    if h < 1e-8:
                        h = -min(np.radians(.05), q[i]-lo[i])
                    if abs(h) < 1e-8:
                        continue
                    sample_q = q.copy()
                    sample_q[i] += h
                    gap = base_clearance(sample_q, proposed)[0]
                    if not np.isfinite(gap):
                        raise RuntimeError('BASE_CLEARANCE_NONFINITE')
                    base_gradient[i] = (gap-base_gap_at_q)/h
            def decode(x):
                value = q.copy()
                if indices:
                    value[list(indices)] = x[:len(indices)]
                return value

            def decode_fingers(x):
                cf = dict(nominal)
                if adaptive:
                    for f, scale in zip(moving, x[len(indices):]):
                        cf[f] = float(commands[f]+scale*(nominal[f]-commands[f]))
                return cf

            def residual(x):
                if time.perf_counter() >= deadline:
                    # least_squares may cross the wall-clock budget while
                    # probing an optional pinky.  Let the caller convert this
                    # private signal into a bounded four-finger local optimum
                    # only when the mandatory fingers already have quality
                    # evidence; core-focus budgets remain hard failures.
                    if focus in optional_fingers and mandatory_quality_ready(current_state):
                        raise _OptionalRolloutBudget(focus)
                    if not optional_only_now:
                        raise RuntimeError('CONTACT_ROLLOUT_BUDGET|' + str(focus))
                cq = decode(x)
                cf = decode_fingers(x)
                state = measure(cq, cf)
                values = []
                for f in fingers:
                    signed = state['signed'][f]
                    # Prefer 3 mm while acquiring contact quality. Acceptance
                    # remains the caller's band, so a safe 2.5 mm scratch can
                    # complete without another wrist correction.
                    values.append((signed+goal_depth(focus))/.001 if f == focus else 0.)
                    if clearance_only:
                        # All fingers have earned contact and released. Gain
                        # retreat clearance with the arm, not another curl
                        # toward base. This earns no new scratching credit.
                        values.append(max(0., .0017-signed)/.001*3.)
                    # Preference only: leave 0.4 mm numerical/tracking reserve
                    # below the caller's hard cap (3.6 mm for the 4 mm policy).
                    soft_max = max(0., depth_max-.0004) if depth_max is not None else .005
                    values.append(max(0., -soft_max-signed)/.001 * 8.)
                    if (sequential_handoff and focus in order and f in pending
                            and f != focus and f in order):
                        rank_ahead = order.index(f) - order.index(focus)
                        if rank_ahead > 0:
                            reserve = PENDING_DEPTH_RESERVE_BY_RANK_M[
                                min(rank_ahead, len(PENDING_DEPTH_RESERVE_BY_RANK_M)) - 1
                            ]
                            follower_depth = max(0.0, -float(signed))
                            # Soft only: discourage spending future headroom,
                            # but never reject a 1--4 mm-safe future contact.
                            values.append(max(0.0, follower_depth - reserve) / .001 * 1.5)
                    near = state['distance'][f] <= .004 or f == focus
                    values.append(max(0., state['cross'][f]-.007)/.002*3. if near else 0.)
                    values.append(max(0., state['radius'][f]-.064)/.004*3. if near else 0.)
                    if sequential_handoff and f != focus and f not in pending:
                        # Preserve a completed contact while the next finger
                        # is acquired.  The hard envelope below rejects a
                        # proposal that makes it materially deeper; this
                        # residual simply guides least_squares away from the
                        # boundary before that rejection is needed.
                        values.append((signed + goal_depth(f)) / .001 * .65)
                        held_max = handoff_depth_limits.get(f)
                        if held_max is not None:
                            values.append(max(0., -held_max-signed)/.001*16.)
                # Optimize inside the same angular bounds the validator uses.
                # A weak 59-degree aesthetic cost previously let the 1 mm
                # pressure objective spend the last available wrist margin.
                values.extend([max(0., state['angle']-(state.get('angle_limit',60.)-.2))*8.,
                               max(0., state['root_angle']-(state.get('root_angle_limit',15.)-.2))*8.])
                # Reserve space BEFORE the hard swept guard rejects a branch.
                # The conservative geometry is identical to the final guard;
                # this is a preference for 18 mm, never a waiver of 10 mm.
                if base_guard_enabled and near_base:
                    gap = base_gap_at_q + np.dot(base_gradient,cq-q)
                    values.append(max(0., .018-gap)/.002*.3)
                    # Keep the old conservative near-base preference as a
                    # residual only.  It must never be evaluated when the
                    # target is far from the base (there is no ``gap`` in
                    # that branch), and it is not a collision authorization.
                    values.append(max(0., .0115-gap)/.001*8.)
                # Both contact transfer and release can hide an interior depth
                # maximum behind valid endpoints. Optimize interior samples;
                # safe() independently checks a denser, full-hand sweep.
                if shape_sweep:
                    for alpha in (.25, .5, .75):
                        intermediate = measure(q+alpha*(cq-q), {
                            f: commands[f]+alpha*(cf[f]-commands[f]) for f in commands})
                        for f in fingers:
                            near = intermediate['signed'][f] <= .0015
                            values.append(max(0., intermediate['cross'][f]-.007)/.001*6. if near else 0.)
                            values.append(max(0., intermediate['radius'][f]-.064)/.002*6. if near else 0.)
                            interior_max = max(0., depth_max-.0004) if depth_max is not None else .005
                            values.append(max(0., -interior_max-intermediate['signed'][f])/.001*8.)
                if adaptive:
                    # Progress is a preference, not permission to cross the
                    # surface too deeply. Waiting fingers remain unchanged.
                    values.extend((1.-x[len(indices):])*.25)
                if rolling_target_q is not None and rolling_joint in indices:
                    denom_deg = max(0.10, abs(float(rolling_wrist_plan.get('step_deg', 0.0))))
                    values.append(
                        (np.degrees(cq[rolling_joint]-rolling_target_q[rolling_joint])
                         / denom_deg) * 0.8
                    )
                values.extend((np.degrees(cq-q)/np.array([.4,.4,.4,3.,4.])*.12).tolist())
                if contact_objective is not None:
                    values.extend(contact_objective(q, commands, cq, cf))
                return np.asarray(values)

            if indices or adaptive:
                initial_x = np.clip(q[list(indices)], lo[list(indices)], hi[list(indices)])
                xlo, xhi = lo[list(indices)], hi[list(indices)]
                if rolling_target_q is not None and rolling_joint in indices:
                    initial_x = np.asarray(initial_x, dtype=float)
                    local_index = list(indices).index(rolling_joint)
                    initial_x[local_index] = np.clip(
                        rolling_target_q[rolling_joint],
                        xlo[local_index], xhi[local_index],
                    )
                if adaptive:
                    min_scale = [min(1., .25/(nominal[f]-commands[f])) if f == focus else 0.
                                 for f in moving]
                    initial_x = np.r_[initial_x, np.full(len(moving), .5)]
                    xlo, xhi = np.r_[xlo, min_scale], np.r_[xhi, np.ones(len(moving))]
                    initial_x = np.clip(initial_x, xlo, xhi)
                try:
                    result = least_squares(residual, initial_x,
                                           bounds=(xlo, xhi),
                                           max_nfev=30 if adaptive else 20, diff_step=1e-3,
                                           xtol=2e-4, ftol=2e-4, gtol=2e-4)
                except _OptionalRolloutBudget as exc:
                    local = optional_budget_local_optimum(
                        pending, current_state, 'solver_deadline')
                    if local is not None:
                        return local
                    raise RuntimeError('CONTACT_ROLLOUT_BUDGET|' + str(exc.focus))
                cq = decode(result.x)
                proposed = decode_fingers(result.x)
            else:
                cq = q.copy()
            proposed, release_ok, release_diag = _couple_completed_release_to_arm(
                measure, q, cq, commands, proposed, completed_handoffs, focus,
                current_state, cleared=cleared,
            )
            if not release_ok:
                failures.append('OLD_RELEASE_CANNOT_OUTRUN_WRIST:' + str(release_diag))
                continue
            if release_diag and any(float(v) > 1e-9 for v in release_diag.values() if isinstance(v, (int, float))):
                print('[V103 ROLLING RELEASE COUPLING]',
                      'focus=', focus,
                      'release_step_deg=', {k: round(float(v),3) for k,v in release_diag.items()
                                           if isinstance(v,(int,float))},
                      'arm_delta_deg=', np.round(np.degrees(cq-q),3).tolist())
            state = measure(cq, proposed)
            if sequential_handoff and handoff_depth_limits:
                # A completed finger is protected during the next finger's
                # handoff, but a small transient increase is not by itself a
                # failed trajectory.  The old hard rejection made a valid
            # 1--4 mm individual-depth solution disappear whenever the
                # coupled J1--J5 solve moved a completed tip by a fraction of
                # a millimetre.  Keep the drift in the residual as a soft
                # preference and let the per-finger 4 mm cap plus swept mesh
                # check remain the actual safety gates.
                drift = {
                    f: max(0., -float(state['signed'][f]) - float(limit))
                    for f, limit in handoff_depth_limits.items()
                    if f != focus
                }
                drift = {f: d for f, d in drift.items() if d > 1e-8}
                if drift:
                    print(
                        '[V103 HANDOFF DEPTH SOFT]',
                        'drift_mm=', {f: round(d * 1000., 3)
                                      for f, d in drift.items()},
                        'hard_gate=PER_FINGER_DEPTH_MAX_4MM',
                    )
            # Future-finger reserve is a COORDINATION preference, not a
            # safety limit.  V17 made it a hard rejection and therefore killed
            # 909 even when ring was predicted around 3.2 mm -- inside the
            # genuine 2--4 mm quality band and below the 3.6 mm planned cap.
            # Keep the real depth cap below as the hard guard.  The residual
            # already prefers headroom, and the new finger-only branch can hold
            # or unload a future finger without forcing common wrist pressure.
            if depth_max is not None and any(-s > depth_max for s in state['signed'].values()):
                failures.append('BALANCED_DEPTH_CAP')
                continue
            # Fail early if a pending finger advances into an airborne stroke.
            # Allow up to 0.5 mm clearance while closing the remaining gap.
            if focus and state['signed'][focus] > .0005:
                current_focus_state = measure(q, commands)
                current_signed = current_focus_state['signed'][focus]
                current_distance = current_focus_state['distance'][focus]
                # Early acquisition hands off before every finger touches.
                # A pending focus may approach through air only while closing
                # its gap; use either signed clearance or nearest-surface
                # distance because a curved STL can improve the latter before
                # its signed normal value changes. This earns no contact credit
                # and cannot finish the finger by itself.
                if depth_max is None or (
                    state['signed'][focus] >= current_signed-.00005
                    and state['distance'][focus] >= current_distance-.00005
                ):
                    failures.append('FOCUS_SHALLOW:' + focus)
                    continue
            if not state['valid']:
                failures.append(state.get('reason', 'GEOMETRY'))
                continue
            if (len(indices) == 0 and focus in mandatory_fingers
                    and focus in pending and not first_owner_free_phase):
                current_focus_depth = max(
                    0.0, -float(current_state['signed'][focus])
                )
                next_focus_depth = max(0.0, -float(state['signed'][focus]))
                if (current_focus_depth < LIVE_CORE_DEPTH_MIN_M - 1e-9
                        and next_focus_depth
                            <= current_focus_depth + MANDATORY_FINGER_ONLY_MIN_PROGRESS_M):
                    failures.append(
                        'FINGER_ONLY_NO_LIVE_PROGRESS:' + str(focus)
                        + ':current_mm=' + str(round(current_focus_depth * 1000.0, 3))
                        + ':next_mm=' + str(round(next_focus_depth * 1000.0, 3))
                    )
                    continue
            if clearance_only and min(state['signed'].values()) < .0015 and (
                min(state['signed'].values()) <= min(current_state['signed'].values())+.00005
            ):
                failures.append('RELEASE_CLEARANCE_NO_PROGRESS')
                continue
            ok, reason = safe(q, commands, cq, proposed)
            if not ok:
                failures.append(str(reason))
                continue
            if len(indices) == 0:
                solver_level = 'FINGER_ONLY'
            elif tuple(indices) == (4,):
                solver_level = 'J5_PRIMARY'
            elif len(indices) <= 2:
                solver_level = 'J4_J5_ASSIST'
            else:
                solver_level = 'J1_TO_J5'
            if adaptive:
                solver_level += '_ADAPTIVE'
            if first_owner_free_phase and len(indices) == 0:
                solver_level += '_FIRST_OWNER_FIXED_PALM'
            elif (rolling_wrist_plan is not None
                    and rolling_wrist_plan.get('required', False)
                    and len(indices) > 0):
                solver_level += '_ROLLING_PREDICTIVE'
            solutions.append((cq, state, solver_level, dict(proposed)))
            if not focus or state['signed'][focus] <= -goal_depth(focus):
                break
        if not solutions:
            # A missing optional branch is not an immediate local optimum.
            # First widen the bounded wrist support envelope, then defer this
            # assistant and try the other pending assistants.  The local-best
            # exit is allowed only after all optional branches at this pose
            # have been exhausted; no 3F re-plan is introduced.
            if support_tier+1 < len(support_tiers):
                support_tier += 1
                lower = np.maximum(physical_lower, anchor-np.radians(support_tiers[support_tier]))
                upper = np.minimum(physical_upper, anchor+np.radians(support_tiers[support_tier]))
                print('[V103 CONTACT RANGE EXPAND] focus=',focus,'support_deg=',support_tiers[support_tier].tolist())
                continue
            # A failed proposal has not called accept(): keep q, fingers and
            # all contact evidence unchanged while trying a different focus.
            if focus is not None:
                if freeze_priority_order and focus in mandatory_fingers:
                    raise RuntimeError('CONTACT_COORDINATION_NO_BRANCH|focus=' + str(focus) + '|' + str(failures)
                                       + '|frozen_order=' + str(order)
                                       + '|q_deg=' + str(np.round(np.degrees(q),3).tolist())
                                       + '|progress=' + str({f:round(commands[f],3) for f in fingers})
                                       + '|signed_mm=' + str({f:round(current_state['signed'][f]*1000,3) for f in fingers}))
                deferred.add(focus)
                alternatives = [f for f in pending if f not in deferred]
                if alternatives:
                    focus = alternatives[0]
                    focus_log.append(focus)
                    support_tier = 0
                    lower = np.maximum(physical_lower, anchor-np.radians(support_tiers[support_tier]))
                    upper = np.minimum(physical_upper, anchor+np.radians(support_tiers[support_tier]))
                    print('[V103 CONTACT ORDER FALLBACK] next=',focus,'deferred=',sorted(deferred))
                    continue
                if (focus in optional_fingers
                        and not any(f not in optional_fingers for f in pending)):
                    return pinky_local_optimum_result(pending, current_state)
            raise RuntimeError('CONTACT_COORDINATION_NO_BRANCH|focus=' + str(focus) + '|' + str(failures)
                               + '|q_deg=' + str(np.round(np.degrees(q),3).tolist())
                               + '|progress=' + str({f:round(commands[f],3) for f in fingers})
                               + '|signed_mm=' + str({f:round(current_state['signed'][f]*1000,3) for f in fingers}))
        # Select the safe branch nearest the preferred target. When rolling
        # prediction says a later owner needs wrist compensation, prefer a
        # safe arm-coupled solution now instead of repeatedly taking a
        # finger-only step and discovering the same deficit at the minimum.
        candidate_pool = solutions
        force_arm_support = bool(
            (rolling_wrist_plan is not None
             and rolling_wrist_plan.get('required', False))
            or predicted_minimum_requires_arm
        )
        if force_arm_support:
            arm_solutions = [row for row in solutions
                             if not str(row[2]).startswith('FINGER_ONLY')]
            if arm_solutions:
                candidate_pool = arm_solutions
        cq, state, level, proposed = min(candidate_pool, key=lambda row:
            abs(row[1]['signed'][focus]+goal_depth(focus)) if focus else 0.)
        if cooperate is not None:
            proposed = cooperate(q, commands, cq, proposed)
            ok, reason = safe(q, commands, cq, proposed)
            if not ok:
                raise RuntimeError('COOPERATIVE_SWEEP_REJECT|' + str(reason))
            state = measure(cq, proposed)
        max_progress = max(proposed[f]-commands[f] for f in fingers)
        duration = max(.028, max_progress / speed,
                       float(np.max(np.abs(np.degrees(cq-q)))) / arm_speed)
        if base_clearance is not None:
            state['base_gap'], state['base_link'] = base_clearance(cq, proposed)
        accept(q, commands, cq, proposed, duration, focus, level, state)
        frames.append(dict(q=cq.copy(), fingers=dict(proposed), duration=duration, focus=focus, level=level))
        if focus in optional_fingers and any(
                f in mandatory_fingers for f in pending if f != focus):
            optional_focus_attempt_steps[focus] = (
                optional_focus_attempt_steps.get(focus, 0) + 1
            )
        q, commands = cq.copy(), dict(proposed)
        deferred.clear()  # A changed pose may make an earlier focus feasible.
        optional_focus_stall_steps.clear()
        optional_focus_last_depth.clear()
        if (sequential_handoff and focus is not None
                and focus not in completed_handoffs
                and live_handoff_ready(focus, state)):
            completed_handoffs.append(focus)
            focus_depth = max(0., -float(state['signed'][focus]))
            # Keep the envelope inside the global cap.  A contact at 2.0 mm
            # therefore gets at most 2.6 mm during the next handoff; a contact
            # already at 3.5 mm can use the remaining 0.5 mm but never cross
            # the 4 mm hard cap.
            handoff_depth_limits[focus] = min(
                float(depth_max) if depth_max is not None else float('inf'),
                focus_depth + HANDOFF_DEPTH_DRIFT_M,
            )
            locked_focus = None
            print('[V103 DYNAMIC HANDOFF]', 'completed=', focus,
                  'depth_mm=', round(focus_depth * 1000., 3),
                  'held_max_mm=', round(handoff_depth_limits[focus] * 1000., 3),
                  'next_order=', order,
                  'completed_sequence=', completed_handoffs,
                  'next_finger=frozen_order_next' if freeze_priority_order else 'next_finger=measured_at_next_step')
    if (focus in optional_fingers
            and not any(f not in optional_fingers for f in pending)):
        return pinky_local_optimum_result(pending, current_state)
    raise RuntimeError('CONTACT_COORDINATION_STEP_BUDGET|' + str(focus))
