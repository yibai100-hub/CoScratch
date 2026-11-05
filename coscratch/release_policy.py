"""Pure release/quality policy. No simulator imports and no drive commands.

Base proximity from the offline bounding-box model is never contact evidence.
Only a PhysX event naming opposite base/finger actors can latch a base stop.
"""
import math

FINGERS = ('index', 'middle', 'ring', 'pinky')
CORE = FINGERS[:3]
RELEASE_STAGES = ('SCRATCH_CURL_', 'FINGER_CURL_RELEASE_')
# Surface clearance is a release-completion target, not merely a contact-loss
# threshold.  The scratch experiment targets roughly +20 mm above the back.
RELEASE_CLEARANCE_TARGET_M = 0.020


def quality(depths_m, samples, relaxed=(), minimums_m=None):
    """Use unrounded depths with optional per-finger tilt-aware floors.

    ``minimums_m`` only changes the lower quality floor (for the selected
    palm tilt).  The four millimetre upper safety cap and sample requirements
    remain hard.
    """
    relaxed = set(relaxed)
    minimums_m = minimums_m or {}
    if not relaxed.issubset(set(FINGERS)):
        return False
    for finger in FINGERS:
        depth = depths_m.get(finger)
        if (depth is None or not math.isfinite(depth)
                or samples.get(finger, 0) <= 0):
            # A relaxed assist channel may remain airborne on a difficult
            # local slope. It is still present in the command trajectory; the
            # primary anchor and all hard geometry/swept checks remain strict.
            if finger in relaxed:
                continue
            return False
        if depth > .004 + 1e-12:
            return False
        minimum = float(minimums_m.get(finger, .001))
        if finger not in relaxed and depth < minimum - 1e-12:
            return False
    return True


def within(path, root):
    return path == root or path.startswith(root + '/')


def base_finger_pair(paths, base_path, robot_root):
    """paths are [actor0, actor1, collider0, collider1], never an unordered set."""
    if len(paths) != 4:
        return None
    sides = ((paths[0], paths[2]), (paths[1], paths[3]))
    for base_side, hand_side in (sides, sides[::-1]):
        if not any(within(p, base_path) for p in base_side):
            continue
        for finger in FINGERS:
            # Match physical link boundaries, not 'right_base_link' (palm).
            for suffix in ('proximal_link', 'middle_link', 'distal_link', 'tip_link'):
                link = robot_root + '/right_' + finger + '_' + suffix
                if any(within(p, link) for p in hand_side):
                    return finger
    return None


def touching_separation(header, contact_data):
    """Positive-separation contact-offset pairs and corrupt ranges aren't touch."""
    offset, count = int(header.contact_data_offset), int(header.num_contact_data)
    if offset < 0 or count <= 0 or offset + count > len(contact_data):
        return None
    values = [float(contact_data[i].separation) for i in range(offset, offset+count)]
    if not all(math.isfinite(v) for v in values):
        return None
    value = min(values)
    return value if value <= 0. else None


def release_reason(stage, signed_m, contact):
    if not str(stage).startswith(RELEASE_STAGES):
        return None
    if contact and contact.get('source') == 'PHYSX_CONTACT_REPORT':
        return 'BASE_CONTACT'
    # A 1--4 mm contact is only a qualified scratch contact. During the
    # explicit SCRATCH_CURL_* continuation the fingers must still execute the
    # complete curl trajectory; surface clearance must not short-circuit it.
    # Only a real PhysX base-contact event may stop that continuation.
    if str(stage).startswith('SCRATCH_CURL_'):
        return None
    if (set(signed_m) == set(FINGERS)
            and all(math.isfinite(v) and v >= RELEASE_CLEARANCE_TARGET_M
                    for v in signed_m.values())):
        return 'SURFACE_CLEARANCE'
    return None


def tail_candidates(trajectory, start):
    """Only unload endpoints from this cycle; never replay remaining curls."""
    for index in range(start, len(trajectory)):
        stage = str(trajectory[index].get('stage', ''))
        if index > start and stage.startswith(('ARC_ASYNC_', 'SCRATCH_WAVE_ASYNC_')):
            break
        if stage.startswith((
                'UNLOAD_ALL',
                'UNLOAD_STEP_',
                'UNLOAD_J5_LIFT_',
                # The inter-cycle reset lifts the wrist and opens the hand
                # in one checked segment, so it is also a valid release-tail
                # candidate when an actual release stop is latched.
                'UNLOAD_OPEN_OVERLAPPED_')):
            yield trajectory[index]


def resume_open_index(trajectory, start, lift_q):
    """Rejoin only this cycle's original open-hand checkpoint at the same arm pose."""
    for index in range(start, len(trajectory)):
        frame = trajectory[index]
        stage = str(frame.get('stage', ''))
        if index > start and stage.startswith(('ARC_ASYNC_', 'SCRATCH_WAVE_ASYNC_')):
            break
        # A combined inter-cycle reset already contains the verified open
        # hand state.  Treat it as the checkpoint itself; otherwise a release
        # stop during overlap is incorrectly downgraded to a standalone tail.
        if (stage.startswith(('OPEN_AFTER_UNLOAD', 'UNLOAD_OPEN_OVERLAPPED_'))
                and len(frame['q']) == len(lift_q)
                and all(abs(float(a)-float(b)) < 1e-10 for a,b in zip(frame['q'], lift_q))
                and set(frame.get('fingers', {})) == set(FINGERS)
                and all(float(v) == 0. for v in frame['fingers'].values())):
            return index
    return None
