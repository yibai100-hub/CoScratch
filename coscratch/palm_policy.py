"""Per-plan palm tilt limits. No Isaac imports, global physics changes or motion."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

TIERS_DEG = (30.0, 45.0, 60.0)
SUPPORTED_CAPS_DEG = TIERS_DEG + (75.0, 90.0)
_limit = ContextVar('v103_palm_limit', default=30.0)
_candidate_deadline = ContextVar('v103_candidate_deadline', default=None)
CANDIDATE_PATH_BUDGET_S = 300.0
EQUIVALENT_FAILURE_LIMIT = 3


class CandidatePathTimeout(BaseException):
    """Bypass broad geometry-exception handlers, but not batch cancellation."""


def check_candidate_budget():
    deadline = _candidate_deadline.get()
    if deadline is not None and time.perf_counter() >= deadline:
        raise CandidatePathTimeout('CANDIDATE_PATH_BUDGET_EXHAUSTED')


def validate_bounded(validate, candidate, emit=print):
    started = time.perf_counter()
    token = _candidate_deadline.set(started + CANDIDATE_PATH_BUDGET_S)
    try:
        result = validate(candidate)
        check_candidate_budget()
        return result
    except CandidatePathTimeout:
        return False, None, 'CANDIDATE_PATH_BUDGET_EXHAUSTED'
    finally:
        _candidate_deadline.reset(token)
        emit('[V103 CANDIDATE PATH TIME]', 'seconds=', round(time.perf_counter()-started, 3),
             'budget_s=', CANDIDATE_PATH_BUDGET_S, 'cooperative_geometry_checkpoints=True')


def _failure_family(reason):
    """Collapse numeric details without treating a search miss as a safety pass."""
    text = str(reason)
    for marker in (
        # A rollout budget miss means the complete contact controller failed
        # to finish for this endpoint.  Numeric focus details vary between
        # otherwise equivalent seeds, so collapse them into one family and
        # stop retesting the remainder of that tier after the existing
        # conservative three-strike threshold.  This does not authorize a
        # path or relax any collision/depth gate.
        'CONTACT_ROLLOUT_BUDGET',
        # Do not collapse acquisition failures into an equivalent family.
        # The acquisition owner/order is measured from each endpoint pose, so
        # two candidates with the same top-level failure can have different
        # useful first-owner geometry (909 is the regression case).  Let the
        # bounded endpoint pool test them instead of stopping after 3.
        'HAND_BASE_MARGIN_REPLAY',
        'HAND_BASE_MARGIN',
        'CANDIDATE_PATH_BUDGET_EXHAUSTED',
        # The same endpoint often produces a transient >4 mm sample during
        # the serial handoff.  Treat these as one equivalent path family so
        # the tier does not spend minutes replaying near-identical candidates.
        'BALANCED_DEPTH_REPLAY_REJECT',
    ):
        if marker in text:
            return marker
    return None


def three_finger_fallback_reason(result, radial_distance, radial_limit, clearance,
                                 near_base_gap=0.018, boundary_band=0.005,
                                 force_replan=False):
    """Authorize a fresh RMI search, never authorize a trajectory or relax a gate.

    ``force_replan`` is the V103 production policy: after the complete 4F
    endpoint/swept-path search returns no usable plan, every target gets one
    fresh three-finger search.  The historical radial-distance checks remain
    available only to legacy offline diagnostics that do not opt into the
    production policy; they are not a runtime reachability certificate.
    """
    if result.get('best') is not None:
        return None
    if force_replan:
        return 'FAILED_4F_REPLAN'
    if math.isfinite(radial_distance) and 0 <= radial_distance <= radial_limit:
        return 'NEAR_BASE_TARGET'
    if (math.isfinite(radial_distance) and math.isfinite(boundary_band)
            and 0 <= boundary_band <= .005
            and radial_limit < radial_distance <= radial_limit + boundary_band):
        return 'NEAR_BASE_SEARCH_BOUNDARY_BAND'
    for candidate in result.get('candidates', []):
        reason = str(candidate.get('trajectory_reason', ''))
        if not ('HAND_BASE_MARGIN' in reason or 'focus=pinky' in reason
                or 'ACQUISITION_FAIL' in reason):
            continue
        # Measure the actual candidate hand envelope, not only the GUI radius.
        # These samples establish proximity only; full swept validation remains
        # mandatory for every newly searched three-finger path.
        for fraction in (0., .25, .5, .75, 1.):
            try:
                commands = {f: float(p)*fraction
                            for f,p in candidate['template']['phases'].items()}
                gap, _ = clearance(candidate['q'], commands)
                if math.isfinite(gap) and gap <= near_base_gap:
                    return 'FAILED_4F_HAND_NEAR_BASE'
            except Exception:
                continue  # unavailable geometry is not evidence
    return None


def limit_deg():
    return _limit.get()


def angle_ok(angle, limit=None):
    cap = limit_deg() if limit is None else float(limit)
    value = float(angle)
    return math.isfinite(value) and 0.0 <= value <= cap + 1e-6 and cap in SUPPORTED_CAPS_DEG


@contextmanager
def scope(cap):
    cap = float(cap)
    if cap not in SUPPORTED_CAPS_DEG:
        raise ValueError('Unsupported V103 palm tier: '+str(cap))
    token = _limit.set(cap)
    try:
        yield
    finally:
        _limit.reset(token)


def select_plan(search, validate, emit=print, tiers=TIERS_DEG, active_count=4,
                max_valid_candidates=None):
    """Exhaust returned 4F trajectories per tier, not merely endpoint validity.

    Search retains its bounded seed/template budget; failure is not a proof
    that a mathematical IK solution cannot exist. No angle is forced to a tier.
    """
    all_candidates, rejects, attempts = [], [], []
    for cap in tiers:
        started = time.perf_counter()
        emit('[V103 PALM TIER] max_deg=', cap, 'preference=parallel; J1-J5 free')
        with scope(cap):
            candidates = list(search(cap, all_candidates))
            all_candidates.extend(candidates)
            valid = [c for c in candidates
                     if (c.get('valid') or c.get('path_validation_eligible'))
                     and len(c['template']['active']) == int(active_count)]
            valid_total = len(valid)
            # Endpoint search can return many near-duplicate 4F poses.  Full
            # swept validation is the expensive part, so production callers
            # may request a deterministic shortlist.  The default remains
            # exhaustive for offline diagnostics and regression tests.
            shortlist_skipped = 0
            if max_valid_candidates is not None:
                limit = max(1, int(max_valid_candidates))
                if len(valid) > limit:
                    shortlist_skipped = len(valid) - limit
                    valid = valid[:limit]
                    emit('[V103 CANDIDATE SHORTLIST]',
                         'tier_deg=', cap,
                         'tested=', len(valid),
                         'valid_total=', valid_total,
                         'skipped_near_duplicates=', shortlist_skipped,
                         'full_path_validation=True')
            row = dict(limit_deg=cap, evaluated=len(candidates),
                       endpoint_valid=len(valid), endpoint_valid_total=valid_total,
                       strict_endpoint_valid=sum(bool(c.get('valid')) for c in valid),
                       provisional_endpoint_valid=sum(not bool(c.get('valid')) for c in valid),
                       trajectories_tested=0, trajectory_rejects=[],
                       shortlist_skipped=shortlist_skipped)
            attempts.append(row)
            repeated_family = None
            repeated_family_count = 0
            for candidate in valid:
                angle = candidate.get('geometry', {}).get('palm_back_angle_deg', float('nan'))
                if not angle_ok(angle):
                    row['trajectory_rejects'].append('ENDPOINT_ANGLE_INCONSISTENT')
                    continue
                candidate['mode'] = dict(
                    candidate['mode'],
                    palm_alignment_limit_deg=cap,
                    path_validation_mode=candidate.get('path_validation_mode', 'STRICT_ENDPOINT'),
                )
                candidate['palm_alignment_limit_deg'] = cap
                row['trajectories_tested'] += 1
                safe, trajectory, reason = validate_bounded(validate, candidate, emit)
                # Validation may replace a 4F candidate in place with a freshly
                # validated 3F candidate. Rebind the selected tier to that new
                # candidate, its execution mode and frames before publication.
                if safe and trajectory:
                    angle = candidate.get('geometry', {}).get('palm_back_angle_deg', float('nan'))
                    if not angle_ok(angle):
                        safe, trajectory, reason = False, None, 'REPLACEMENT_ENDPOINT_ANGLE_INCONSISTENT'
                if safe and trajectory:
                    candidate['palm_alignment_limit_deg'] = cap
                    candidate['mode'] = dict(
                        candidate['mode'], palm_alignment_limit_deg=cap,
                        path_validation_mode=candidate.get('path_validation_mode', 'STRICT_ENDPOINT'),
                    )
                    for frame in trajectory:
                        frame['palm_alignment_limit_deg'] = cap
                    row.update(seconds=round(time.perf_counter()-started, 3), selected=True)
                    emit('[V103 PALM SELECTED] limit_deg=', cap, 'actual_endpoint_deg=', round(angle, 3),
                         'complete_trajectory=PASS')
                    return dict(best=candidate, trajectory=trajectory, attempts=attempts,
                                candidates=all_candidates, rejects=rejects)
                reason = str(reason) if not safe else 'EMPTY_TRAJECTORY'
                rejects.append((candidate['template']['name'], reason))
                row['trajectory_rejects'].append(reason)
                family = _failure_family(reason)
                if family is not None and family == repeated_family:
                    repeated_family_count += 1
                else:
                    repeated_family = family
                    repeated_family_count = 1 if family is not None else 0
                if (family is not None
                        and repeated_family_count >= EQUIVALENT_FAILURE_LIMIT):
                    skipped = max(0, len(valid) - row['trajectories_tested'])
                    row['equivalent_family_pruned'] = dict(
                        family=family,
                        consecutive=repeated_family_count,
                        skipped=skipped,
                    )
                    emit('[V103 EQUIVALENT CANDIDATE FAMILY STOP]',
                         'family=', family,
                         'consecutive=', repeated_family_count,
                         'skipped=', skipped,
                         'safety_thresholds_unchanged=True')
                    break
            row.update(seconds=round(time.perf_counter()-started, 3), selected=False)
            emit('[V103 PALM TIER REJECTED]', row,
                 'no complete safe path in searched candidates; safety gates retained')
    return dict(best=None, trajectory=None, attempts=attempts,
                candidates=all_candidates, rejects=rejects)
