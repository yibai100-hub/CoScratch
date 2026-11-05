"""Bounded FK-first endpoint search. Never commands motion or accepts a path.

Coarse scores are numerical warm starts only. Every returned candidate is built
by the production geometry gates; the caller must validate the complete path.
"""
import time
import numpy as np
from scipy.optimize import minimize

BUILD = 'FK-FIRST-DIVERSE-BRANCHES-v12'


def path_eligible(candidate):
    return bool(candidate.get('valid') or candidate.get('path_validation_eligible'))


def eligibility_rank(candidate):
    if candidate.get('valid'):
        return 0
    if candidate.get('path_validation_eligible'):
        return 1
    return 2


class SearchDeadline(Exception):
    pass


def radical_inverse(index, base):
    value, fraction = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * fraction
        fraction /= base
    return value


def joint_samples(lower, upper, seeds=(), count=1536):
    """Deterministic full-limit coverage, not a fixed J1/J4/J5 heading."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (5,) or upper.shape != (5,) or not np.all(upper > lower):
        raise ValueError('Invalid five-joint interior bounds')
    seen = set()
    for q in list(seeds) + [lower + (upper-lower) * np.array([
            radical_inverse(i, b) for b in (2, 3, 5, 7, 11)])
            for i in range(1, count+1)]:
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            continue
        q = np.clip(q, lower, upper)
        key = tuple(np.round(q, 5))
        if key not in seen:
            seen.add(key)
            yield q.copy()


def diverse_best(rows, count, separation_deg=12.):
    selected = []
    for row in sorted(rows, key=lambda item: item[0]):
        if all(np.linalg.norm(np.degrees(row[1]-old[1])) >= separation_deg
               for old in selected):
            selected.append(row)
        if len(selected) >= count:
            break
    return selected


def nearby_seeds(q, lower, upper):
    """Bounded local alternatives, not target-specific stored successes."""
    q = np.asarray(q)
    offsets = []
    for delta in (.75, 1.5):
        for axis in range(5):
            for sign in (-1., 1.):
                offset = np.zeros(5)
                offset[axis] = sign*delta
                offsets.append(offset)
    # Heading and wrist pitch together can preserve a lane while clearing base.
    for first in (-1., 1.):
        for fourth in (-1., 1.):
            offset = np.zeros(5)
            offset[0], offset[3] = first, fourth
            offsets.append(offset)
    for offset in offsets:
        candidate = q+np.radians(offset)
        if np.all(candidate >= lower) and np.all(candidate <= upper):
            yield candidate


def retain_valid_endpoints(ordered, count=6):
    """Preserve small, strictly validated alternatives within a bounded list."""
    kept = []
    # A stored physical-success seed is a replay branch, not an ordinary
    # near-duplicate.  Two endpoint poses can be less than 0.5 degrees apart
    # yet produce a different frozen finger order / acquisition result.  Keep
    # validated priority seeds before applying the ordinary diversity filter.
    priority = [candidate for candidate in ordered
                if candidate.get('_priority_seed') and candidate.get('valid')]
    for candidate in priority:
        if len(kept) >= count:
            break
        if all(np.linalg.norm(np.degrees(candidate['q']-other['q'])) >= .05
               for other in kept):
            kept.append(candidate)
    for candidate in ordered:
        if not candidate.get('valid'):
            continue
        if any(candidate is kept_candidate for kept_candidate in kept):
            continue
        if all(np.linalg.norm(np.degrees(candidate['q']-other['q'])) >= .5 for other in kept):
            kept.append(candidate)
        if len(kept) == count:
            break
    return kept


def retain_path_branches(ordered, count=12):
    """Keep the proven local options, then reserve slots for other arm branches.

    Strict endpoints are kept first. Narrowly bounded provisional endpoints may
    use remaining slots, but still require the exact same complete path check.
    Diversity uses the arm joints as well as wrist joints.
    """
    kept = retain_valid_endpoints(ordered, min(6, count))
    strict_pool = [c for c in ordered if c.get('valid') and not any(c is k for k in kept)]
    provisional_pool = [c for c in ordered
                        if c.get('path_validation_eligible') and not c.get('valid')]
    if not kept and provisional_pool:
        # A near-base target may have no strict static endpoint at all. Seed the
        # diversity calculation with the best bounded provisional candidate;
        # this grants path review only, never motion acceptance.
        kept.append(provisional_pool.pop(0))
    scale = np.array([1., 1., 1., .5, .5])
    # Exhaust spatially distinct strict branches before provisional review.
    for pool in (strict_pool, provisional_pool):
        while pool and len(kept) < count:
            distances = [min(np.linalg.norm(np.degrees(c['q']-k['q'])*scale)
                             for k in kept) for c in pool]
            index = int(np.argmax(distances))
            if distances[index] < 3.:
                break
            kept.append(pool.pop(index))
    return kept


class ArcMinima:
    def __init__(self, arcs, fingers, minimum, maximum):
        self.fingers = tuple(fingers)
        self.grid = np.linspace(float(minimum), float(maximum), 201)
        self.points = np.array([np.column_stack([
            np.interp(self.grid, arcs[f]['angles'], arcs[f]['points'][:, k])
            for k in range(3)]) for f in self.fingers])

    def at(self, rotation, normal):
        normal = np.asarray(normal, dtype=float)
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length < 1e-10:
            raise ValueError('Invalid surface normal')
        height = self.points @ (np.asarray(rotation).T @ (normal / length))
        indices = np.argmin(height, axis=1)
        points = self.points[np.arange(len(self.fingers)), indices]
        return dict(zip(self.fingers, self.grid[indices])), points


def search(*, fk, arcs, fingers, minimum, maximum, lower, upper,
           anchor, normal, palm_normal, root_angle, evaluate, seeds=(),
           seconds=120., coarse_count=1536, emit=print, palm_limit_deg=60.,
           anchor_finger=None, priority_seeds=(), priority_only=False):
    """Use cheap FK/arc projections first, then strictly evaluated refinements.

    Deadline applies to endpoint search, checked between evaluations (an external
    FK/evaluation call cannot be preempted). Full swept-path validation is separate.
    """
    started = time.perf_counter()
    deadline = started + float(seconds)
    anchor, normal = np.asarray(anchor), np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    minima = ArcMinima(arcs, fingers, minimum, maximum)
    anchor_index = None
    if anchor_finger is not None:
        if anchor_finger not in minima.fingers:
            raise ValueError('anchor_finger is not active: ' + str(anchor_finger))
        anchor_index = minima.fingers.index(anchor_finger)
    candidates, coarse, refined = [], [], []
    class PriorityReviewComplete(Exception):
        pass
    stats = dict(build=BUILD, fk_evaluations=0, strict_evaluations=0,
                 deadline_reached=False, endpoint_budget_s=float(seconds))
    bounds = list(zip(lower, upper))

    def check():
        if time.perf_counter() >= deadline:
            raise SearchDeadline()

    def pose(q):
        check()
        p, r = fk(np.asarray(q))
        p, r = np.asarray(p), np.asarray(r)
        if p.shape != (3,) or r.shape != (3, 3) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise ValueError('Invalid FK')
        stats['fk_evaluations'] += 1
        phases, local = minima.at(r, normal)
        return p, r, phases, local

    def cheap(q):
        p, r, phases, local = pose(q)
        tips = p + local @ r.T
        target = anchor - .002 * normal
        delta = tips.mean(axis=0) - target
        anchor_delta = None if anchor_index is None else tips[anchor_index] - target
        heights = (tips-anchor) @ normal + .002
        cosine = np.clip(abs(float(np.dot(r @ palm_normal, normal))), 0., 1.)
        angle = np.degrees(np.arccos(cosine))
        # Parallel is preferred, but 30 degrees is not a search wall.
        tilt = (angle/45.)**2 + 40.*(max(0., angle-palm_limit_deg)/15.)**2
        collinear = float(root_angle(p, r))
        cost = float(np.dot(delta, delta)/.025**2 + np.mean(heights**2)/.010**2
                     + tilt + 4.*(collinear/15.)**2)
        # In the near-base three-finger fallback the clicked point belongs to
        # the middle finger, not to an unconstrained four-tip centroid.  Keep a
        # smaller centroid term for the neighbouring index/ring coverage while
        # making the middle fingertip the primary optimizer target.
        if anchor_delta is not None:
            cost += float(4.0 * np.dot(anchor_delta, anchor_delta) / .018**2)
        # A minimum at the open endpoint has no inward scratch segment.
        cost += sum((max(0., 8.-v)/8.)**2 * 6. for v in phases.values())
        return cost

    def strict(q):
        p, r, phases, _ = pose(q)
        candidate = evaluate(np.asarray(q), phases, p, r)
        stats['strict_evaluations'] += 1
        candidates.append(candidate)
        return float(candidate.get('search_cost', candidate['geometry']['cost']))

    try:
        emit('[V103 FK SEARCH] coarse_samples=', coarse_count, 'endpoint_budget_s=', seconds)
        # Never let coarse ranking or local minimization erase a known branch.
        # These are untrusted hints: run the same strict evaluator first and
        # retain its result. Global alternatives still run for path fallback.
        for seed in priority_seeds:
            q = np.asarray(seed, dtype=float)
            if q.shape != (5,) or not np.all(np.isfinite(q)) or np.any(q < lower) or np.any(q > upper):
                emit('[V103 PRIORITY SEED] rejected invalid/out-of-bounds hint')
                continue
            strict(q)
            # Preserve this candidate through the final diversity filter.  The
            # same-target historic success must be replayed before nearby
            # optimizer branches are allowed to replace it.
            candidates[-1]['_priority_seed'] = True
            emit('[V103 PRIORITY SEED]', 'q_deg=', np.round(np.degrees(q), 3),
                 'endpoint_valid=', bool(candidates[-1].get('valid')),
                 'reason=', candidates[-1]['geometry'].get('reject'),
                 'full_path_still_required=True')
        if priority_only:
            raise PriorityReviewComplete()
        for q in joint_samples(lower, upper, seeds, coarse_count):
            coarse.append((cheap(q), q))
        # Keep different arm branches rather than many nearly identical seeds.
        for _, q in diverse_best(coarse, 20):
            check()
            result = minimize(cheap, q, method='Powell', bounds=bounds,
                              options=dict(maxiter=12, maxfev=450, xtol=2e-3, ftol=2e-3))
            refined.append((cheap(result.x), result.x.copy()))
        emit('[V103 FK SEARCH] coarse_refinement_complete; strict branches=', len(refined))
        # Evaluate all branch starts before spending the budget polishing one.
        starts = diverse_best(refined+coarse, 10, separation_deg=8.)
        for _, q in starts:
            strict(q)
        for _, q in starts:
            check()
            minimize(strict, q, method='Powell', bounds=bounds,
                     options=dict(maxiter=8, maxfev=180, xtol=8e-4, ftol=2e-3))
            eligible = [c for c in candidates if path_eligible(c)]
            if len(diverse_best([(eligibility_rank(c), c['q']) for c in eligible], 3, 5.)) >= 3:
                break
        # A 3-degree duplicate radius erased useful near-base alternatives.
        # Strictly check a small neighborhood of up to two existing solutions;
        # these do not bypass endpoint gates or the shared search deadline.
        eligible = [c for c in candidates if path_eligible(c)]
        anchors = diverse_best([(eligibility_rank(c) * 1e6 + c['geometry']['cost'],c['q']) for c in eligible],2,1.)
        for _, q in anchors:
            for nearby in nearby_seeds(q,lower,upper):
                strict(nearby)
    except PriorityReviewComplete:
        stats['priority_only'] = True
    except SearchDeadline:
        stats['deadline_reached'] = True
        emit('[V103 FK SEARCH BUDGET] keeping strictly checked candidates; no motion approval implied')
    # De-duplicate by q AND phases, preserving the best evaluation of a state.
    unique = {}
    for c in candidates:
        key = (tuple(np.round(c['q'], 5)), tuple(round(c['template']['phases'][f], 3) for f in fingers))
        old = unique.get(key)
        rank = lambda x: (eligibility_rank(x), x['geometry']['cost'])
        if old is None or rank(c) < rank(old):
            unique[key] = c
    ordered = sorted(unique.values(), key=lambda c: (eligibility_rank(c), c['geometry']['cost']))
    # Limit subsequent expensive path attempts; rejected endpoints kept for diagnostics.
    valid = [c for c in ordered if c.get('valid')]
    provisional = [c for c in ordered if path_eligible(c) and not c.get('valid')]
    kept_valid = retain_path_branches(ordered)
    stats.update(seconds=round(time.perf_counter()-started, 3),
                 endpoint_valid=len(valid),
                 returned_valid=sum(bool(c.get('valid')) for c in kept_valid),
                 path_review_eligible=len(provisional),
                 returned_provisional=sum(not c.get('valid') for c in kept_valid),
                 returned_invalid=min(12, sum(not path_eligible(c) for c in ordered)))
    emit('[V103 FK SEARCH SUMMARY]', stats)
    return kept_valid + [c for c in ordered if not path_eligible(c)][:12], stats
