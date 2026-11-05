"""Bounded feedback recovery. No Isaac import, scene write or automatic motion.

The adapter must validate every commanded segment using measured 12-DOF hand
geometry. Recovery does not turn a 4..8 mm excursion into normal-quality data.
"""
import numpy as np

TARGET_M = .002
NORMAL_MAX_M = .004
RECOVER_EXIT_M = .0035
HARD_MAX_M = .008
# J5 is the wrist lift joint and its Lula limit is +/-90 deg.  When the
# positive wrist direction has no remaining room, a small joint2 shoulder
# motion is a bounded upper-arm relief fallback.  It is tried only after the
# normal J4/J5 recovery candidates have failed, and every trial still goes
# through the caller's full swept/depth validator.
J5_UPPER_RELIEF_MARGIN_DEG = 3.0
J2_UPPER_ARM_RELIEF_STEPS_DEG = (0.5, 1.0, 1.5, 2.0)


class DepthRecovery:
    """Freeze curl, relieve with small wrist steps, then validate a rejoin.

    Tracking deadlines use wall time so a slow PhysX/render loop cannot turn a
    short recovery into minutes. Geometry settling still uses physics samples.
    No teleport or home reset.
    """
    def __init__(self, q, hand, authorized, force_relief=False, joint_limits=None,
                 triggering_finger=None):
        self.authorized = set(authorized)
        if not self.authorized:
            raise ValueError('DEPTH_RECOVERY_REQUIRES_AUTHORIZED_CONTACT')
        self.q = np.array(q, dtype=float, copy=True)
        self.hand = {f: np.array(v, dtype=float, copy=True) for f, v in hand.items()}
        self.elapsed = self.wait = self.settled = 0.
        self.steps = 0
        self.phase = 'relief'
        self.goal = None
        self.goal_start_q = None
        self.goal_start_hand = None
        self.force_relief = bool(force_relief)
        # During a live wave-tracking recovery this identifies the finger that
        # crossed the depth gate.  It is only used by the emergency release
        # fallback after every wrist-only candidate has failed; normal recovery
        # still freezes all finger triplets.
        self.triggering_finger = (str(triggering_finger)
                                  if triggering_finger in self.authorized else None)
        self.rejoin_target = None
        self.rejoin_steps = 0
        self.rejoin_final = False
        self.feedback_log_at = -1.
        self.joint_limits = joint_limits

    def _finger_release_candidates(self, hand):
        """Yield bounded physical triplets for the triggering contact finger.

        The sign of curl-away is pose-dependent, so test both signs and a few
        small magnitudes.  This is deliberately an emergency fallback: the
        caller still runs every candidate through the full measured swept
        validator before issuing a command.
        """
        finger = self.triggering_finger
        if not self.force_relief or finger is None or finger not in hand:
            return
        current = np.asarray(hand[finger], dtype=float)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            return
        limits = None
        if self.joint_limits is not None:
            limits = np.asarray(self.joint_limits[2], dtype=float)
        # Prefer the common curl-away move, then test the opposite local sign.
        directions = (
            np.array([1., 1., 1.]), np.array([0., 1., 1.]),
            np.array([1., 0., 0.]), np.array([-1., -1., -1.]),
            np.array([0., -1., -1.]), np.array([-1., 0., 0.]),
        )
        for magnitude in (.25, .5, 1., 1.5, 2.):
            for direction in directions:
                candidate = {f: np.asarray(v, dtype=float).copy()
                             for f, v in hand.items()}
                value = current + magnitude * direction
                if limits is not None:
                    value = np.minimum(value, limits)
                value = np.maximum(value, 0.)
                if np.max(np.abs(value-current)) <= 1.e-9:
                    continue
                candidate[finger] = value
                yield finger, magnitude, direction, candidate

    def _j5_near_upper_limit(self, q):
        """Return whether positive wrist lift is within the configured limit."""
        if self.joint_limits is None:
            return False
        try:
            _, upper, _ = self.joint_limits
            q = np.asarray(q, dtype=float)
            upper = np.asarray(upper, dtype=float)
            if q.shape != upper.shape or q.size < 5:
                return False
            remaining_deg = float(np.degrees(upper[4] - q[4]))
            return bool(np.isfinite(remaining_deg) and
                        remaining_deg <= J5_UPPER_RELIEF_MARGIN_DEG + 1e-9)
        except Exception:
            return False

    def tick(self, dt, sample, validate, command, rejoin, wall_dt=None):
        """Return RUNNING/DONE; reject on missing feedback or unsafe recovery.

        sample -> (actual q, actual triplets, depth map); validate ->
        (safe, reason, endpoint depths). command accepts physical triplets, not
        fitted curl percentages. rejoin -> (q, physical triplets).
        """
        deadline_dt = float(dt if wall_dt is None else wall_dt)
        # ``elapsed`` is diagnostic only.  Do not impose one wall-clock cap on
        # the whole recovery: rendered PhysX can need more than eight wall
        # seconds while still making measured progress through several safe
        # relief/rejoin waypoints.  Boundedness comes from the independent
        # two-second no-tracking deadline for each command plus the six relief
        # and twelve rejoin step limits below.  The measured 8 mm hard-depth
        # stop remains unconditional.
        self.elapsed += deadline_dt
        q, hand, depths = sample()
        values = np.array([depths[f] for f in self.authorized], dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError('DEEP_ADJUST_INVALID_DEPTH')
        if float(values.max()) > HARD_MAX_M:
            raise RuntimeError('DEEP_ADJUST_OVER_8MM')
        if self.goal is not None:
            goal_q, goal_hand = self.goal
            arm_errors = np.degrees(q-goal_q)
            hand_errors = {f: np.asarray(hand[f])-goal_hand[f] for f in goal_hand}
            if not (np.all(np.isfinite(arm_errors)) and
                    all(np.all(np.isfinite(v)) for v in hand_errors.values())):
                raise RuntimeError('DEEP_ADJUST_INVALID_JOINT_FEEDBACK')
            detail = ('phase='+self.phase+'|arm_error_deg='+str(np.round(arm_errors,3).tolist())
                      +'|hand_error_deg='+str({f:np.round(v,3).tolist() for f,v in hand_errors.items()}))
            if self.elapsed-self.feedback_log_at >= .25:
                self.feedback_log_at = self.elapsed
                print('[V103 RECOVERY FEEDBACK]', detail, 'depth_mm=',
                      {f:round(float(depths[f])*1000,3) for f in self.authorized})
            # A rejoin waypoint can overshoot the 4 mm normal band by a small
            # amount because PhysX drives settle after the wrist command was
            # issued.  Do not spend the entire tracking deadline waiting for
            # that already-unsafe rejoin target: demote it back to the
            # validated relief state and issue another measured wrist step.
            # The hard 8 mm stop remains unconditional, and the relief path
            # still requires an independently swept depth improvement.
            if self.phase == 'rejoin' and float(values.max()) > NORMAL_MAX_M:
                print('[V103 RECOVERY REJOIN OVERDEPTH -> RELIEF]',
                      'depth_mm=', round(float(values.max()) * 1000., 3))
                self.phase = 'relief'
                self.rejoin_final = False
            # A one-degree waypoint cannot use a one-degree arrival tolerance:
            # that would accept a hand which has not moved at all.
            hand_tolerance = .2 if self.phase == 'rejoin' and not self.rejoin_final else 1.
            arm_residual = float(np.max(np.abs(np.degrees(q-goal_q))))
            hand_residual = max(
                float(np.max(np.abs(hand[f]-goal_hand[f]))) for f in goal_hand)
            moved = bool(
                self.goal_start_q is not None and
                (np.max(np.abs(np.degrees(q-self.goal_start_q))) > .05 or
                 any(np.max(np.abs(hand[f]-self.goal_start_hand[f])) > .05
                     for f in goal_hand))
            )
            exact_tracked = (arm_residual <= .2 and
                             all(np.max(np.abs(hand[f]-goal_hand[f])) <= hand_tolerance
                                 for f in goal_hand))
            # A fingertip that is already on the surface may hold a small
            # position-drive error.  Do not call that a failed recovery when
            # the measured depth is safe; the independent swept/depth check
            # remains authoritative.  This band is only for rejoin waypoints,
            # never for the initial wrist-relief command.
            compliant_rejoin = bool(
                self.phase == 'rejoin' and not exact_tracked and
                arm_residual <= .5 and
                hand_residual <= 2.0 and
                float(values.max()) <= NORMAL_MAX_M and
                moved)
            # A relief command deliberately freezes the hand at the measured
            # contact pose and moves only J4/J5.  PhysX can therefore leave a
            # several-degree finger-drive residual even after the wrist has
            # reached its validated waypoint.  That residual is not a reason
            # to abort recovery: requiring it to be <=2 degrees made real
            # runs (notably ring) hit TRACKING_TIMEOUT while the arm was already
            # settled and measured depth was improving.  In relief, arm
            # tracking plus demonstrable motion is the authoritative arrival
            # test; hand residual remains diagnostic only.  Rejoin waypoints
            # still use the stricter hand-aware criteria below.
            relief_tracked = bool(
                self.phase == 'relief' and not exact_tracked and
                arm_residual <= .5 and moved)
            tracked = exact_tracked or compliant_rejoin or relief_tracked
            # ``wait`` is the deadline for reaching a safe tracking band, not
            # a deadline for both reaching it and accumulating the required
            # physics-settle samples.  With a slow rendered Isaac loop the old
            # shared timer could enter the contact-compliant band just before
            # two wall seconds, then stop before 0.10 simulation seconds of
            # settling had elapsed.  Once tracking is safe, the independent
            # step-count bounds remain active while physics settling
            # completes.  If tracking is lost again, start a fresh bounded
            # acquisition window rather than inheriting an already-expired one.
            if tracked:
                self.wait = 0.
                self.settled += float(dt)
            else:
                self.wait += deadline_dt
                self.settled = 0.
                if self.wait > 2.0:
                    raise RuntimeError('DEEP_ADJUST_TRACKING_TIMEOUT|'+detail)
            if self.settled < .10:
                return 'RUNNING'
            if (self.phase == 'relief' and
                    getattr(self, 'tracking_snapshot', None) is not None and
                    float(values.max()) <= RECOVER_EXIT_M):
                print('[V103 RECOVERY SAFE EXIT]',
                      'depth_mm=', round(float(values.max()) * 1000., 3),
                      'rejoin=SKIPPED_MEASURED_RELEASE_TAIL')
                self.goal = None
                return 'DONE'
            if self.phase == 'rejoin':
                if float(values.max()) > NORMAL_MAX_M:
                    raise RuntimeError('DEEP_ADJUST_REJOIN_STILL_DEEP')
                if compliant_rejoin:
                    print('[V103 RECOVERY CONTACT-COMPLIANT]',
                          'residual_hand_error_deg=',
                          {f:round(float(np.max(np.abs(hand[f]-goal_hand[f]))),3)
                          for f in goal_hand}, 'depth_mm=', round(float(values.max())*1000.,3))
                if self.rejoin_final:
                    return 'DONE'
            elif relief_tracked:
                print('[V103 RECOVERY RELIEF WAYPOINT TRACKED]',
                      'residual_hand_error_deg=',
                      {f:round(float(np.max(np.abs(hand[f]-goal_hand[f]))),3)
                       for f in goal_hand},
                      'depth_mm=', round(float(values.max())*1000.,3),
                      'next=', ('REJOIN' if float(values.max()) <= RECOVER_EXIT_M
                               else 'ANOTHER_VALIDATED_RELIEF_STEP'))
                # A tracking-triggered recovery is already inside an active
                # pressure wave.  Once measured depth is back below the
                # hysteresis exit, replaying the old wave endpoint is unsafe:
                # it can immediately drive the same finger over 4 mm again.
                # Let the adapter take the measured pose into its verified
                # release tail instead of forcing a rejoin to stale commands.
            self.goal = None
        # The hand remains at the captured actual pose throughout relief.
        if self.phase == 'rejoin' or (float(values.max()) <= RECOVER_EXIT_M and not (self.force_relief and self.steps == 0)):
            if float(values.max()) > NORMAL_MAX_M:
                raise RuntimeError('DEEP_ADJUST_REJOIN_STILL_DEEP')
            if self.rejoin_target is None:
                final_q, final_hand = rejoin()
                self.rejoin_target = (np.array(final_q, copy=True),
                    {f: np.array(v, copy=True) for f, v in final_hand.items()})
            final_q, final_hand = self.rejoin_target
            if self.joint_limits is not None:
                lower, upper, hand_upper = self.joint_limits
                if (np.any(final_q < lower) or np.any(final_q > upper) or
                        any(np.any(v < 0.) or np.any(v > hand_upper) for v in final_hand.values())):
                    raise RuntimeError('DEEP_ADJUST_FINAL_TARGET_OUTSIDE_LIMITS')
                # Only tolerate small measured solver overshoot. Never hide a
                # large violation or alter the measured sweep start.
                arm_over = np.max(np.abs(np.degrees(q-np.clip(q,lower,upper))))
                hand_over = max(float(np.max(np.abs(v-np.clip(v,0.,hand_upper)))) for v in hand.values())
                if max(arm_over,hand_over) > .2:
                    raise RuntimeError('DEEP_ADJUST_MEASURED_LIMIT_EXCURSION|arm_deg='+str(arm_over)+'|hand_deg='+str(hand_over))
            arm_delta = np.degrees(final_q-q)
            hand_delta = {f: final_hand[f]-hand[f] for f in final_hand}
            if not (np.all(np.isfinite(arm_delta)) and
                    all(np.all(np.isfinite(v)) for v in hand_delta.values())):
                raise RuntimeError('DEEP_ADJUST_INVALID_REJOIN')
            span = max(float(np.max(np.abs(arm_delta))),
                       max(float(np.max(np.abs(v))) for v in hand_delta.values()))
            if self.rejoin_steps >= 12:
                raise RuntimeError('DEEP_ADJUST_REJOIN_STEP_LIMIT')
            # Recompute from measured joints, not from the previous command.
            # Keep the existing 3-degree segment validator; command <=1 degree.
            alpha = min(1., 1./max(span, 1.e-12))
            target_q = q + alpha*(final_q-q)
            target_hand = {f: hand[f]+alpha*hand_delta[f] for f in final_hand}
            if self.joint_limits is not None:
                target_q = np.clip(target_q, lower, upper)
                target_hand = {f:np.clip(v,0.,hand_upper) for f,v in target_hand.items()}
            self.rejoin_final = alpha == 1.
            print('[V103 RECOVERY REJOIN STEP]', self.rejoin_steps+1,
                  'arm_delta_deg=', np.round(arm_delta,3).tolist(),
                  'hand_delta_deg=', {f:np.round(v,3).tolist() for f,v in hand_delta.items()},
                  'fraction=', round(alpha,4), 'final=', self.rejoin_final)
            ok, reason, _ = validate(q, hand, target_q, target_hand,
                                     NORMAL_MAX_M, False)
            if not ok:
                raise RuntimeError('DEEP_ADJUST_REJOIN_REJECTED|' + str(reason))
            self.phase = 'rejoin'
            self.rejoin_steps += 1
        else:
            if self.steps >= 6:
                raise RuntimeError('DEEP_ADJUST_STEP_LIMIT')
            target_q = target_hand = None
            candidate_diagnostics = []
            # J5 first; use progressively larger bounded wrist increments when
            # PhysX has already overshot.  The old fixed +/-0.5 degree list
            # could make every candidate look depth-neutral and abort before
            # any measured relief was possible.  Every candidate still passes
            # the full 12-DOF swept validator and remains below its 3-degree
            # segment bound; these are not blind commands.
            for j4, j5 in ((0, .5), (0, 1.0), (0, 1.5), (0, 2.0),
                           (0, -.5), (0, -1.0),
                           (.5, 0), (-.5, 0), (1.0, 0), (-1.0, 0),
                           (.5, .5), (-.5, .5), (.5, 1.0), (-.5, 1.0)):
                trial = np.array(q, copy=True)
                trial[3:5] += np.radians([j4, j5])
                ok, reason, predicted = validate(q, hand, trial, self.hand,
                                                  HARD_MAX_M, True)
                if predicted:
                    candidate_diagnostics.append({
                        'j4_deg': j4, 'j5_deg': j5, 'ok': bool(ok),
                        'reason': str(reason),
                        'predicted_mm': {f: round(float(v) * 1000., 3)
                                         for f, v in predicted.items()}})
                if (ok and max(predicted[f] for f in self.authorized)
                        < float(values.max()) - .00001):
                     target_q, target_hand = trial, self.hand
                     break
            if target_q is None and self._j5_near_upper_limit(q):
                # The wrist is already at (or within 3 deg of) its positive
                # 90-deg limit.  Let the upper arm (joint2) lift the palm a
                # little instead of declaring NO_SAFE_WRIST_STEP.  Test both
                # signs because the USD/URDF axis convention is model-specific;
                # the measured validator selects only a depth-improving safe
                # direction.  This branch is never used while J5 has room.
                j2_diagnostics = []
                for magnitude in J2_UPPER_ARM_RELIEF_STEPS_DEG:
                    for delta_deg in (float(magnitude), -float(magnitude)):
                        trial = np.array(q, copy=True)
                        trial[1] += np.radians(delta_deg)
                        if self.joint_limits is not None:
                            lower, upper, _ = self.joint_limits
                            if trial[1] < lower[1] - 1e-9 or trial[1] > upper[1] + 1e-9:
                                continue
                        ok, reason, predicted = validate(q, hand, trial, self.hand,
                                                         HARD_MAX_M, True)
                        row = {
                            'j2_deg': delta_deg,
                            'ok': bool(ok),
                            'reason': str(reason),
                            'predicted_mm': ({f: round(float(v) * 1000., 3)
                                              for f, v in predicted.items()}
                                             if predicted else {}),
                        }
                        j2_diagnostics.append(row)
                        if (ok and predicted and
                                max(predicted[f] for f in self.authorized)
                                < float(values.max()) - .00001):
                            target_q, target_hand = trial, self.hand
                            print('[V103 RECOVERY J2 UPPER-ARM CANDIDATE]',
                                  'delta_deg=', delta_deg,
                                  'j5_remaining_deg=', round(
                                      float(np.degrees(np.asarray(self.joint_limits[1])[4] - q[4])), 3),
                                  'predicted_depth_mm=', row['predicted_mm'])
                            break
                    if target_q is not None:
                        break
                if target_q is None:
                    print('[V103 RECOVERY J2 UPPER-ARM NO SAFE CANDIDATE]',
                          'j5_near_upper_limit=True',
                          'candidates=', j2_diagnostics)
            if target_q is None:
                # A live wave-tracking recovery may have crossed the 4 mm
                # gate because the active finger continued curling while the
                # wrist was settling.  If no wrist-only step can improve the
                # measured maximum, try a tiny physical curl-away move on the
                # triggering finger.  This is not a relaxed path: every move
                # is still checked by the same full 12-DOF swept validator,
                # lane/patch checks and hard depth cap.  Normal (non-wave)
                # recovery deliberately keeps the old freeze-curl behaviour.
                release_diagnostics = []
                for finger, magnitude, direction, candidate_hand in self._finger_release_candidates(hand):
                    ok, reason, predicted = validate(q, hand, q, candidate_hand,
                                                      HARD_MAX_M, True)
                    row = {
                        'finger': finger,
                        'delta_deg': np.round(magnitude * direction, 3).tolist(),
                        'ok': bool(ok),
                        'reason': str(reason),
                        'predicted_mm': ({f: round(float(v) * 1000., 3)
                                         for f, v in predicted.items()}
                                         if predicted else {}),
                    }
                    release_diagnostics.append(row)
                    if not (ok and predicted and
                            all(np.isfinite(float(predicted[f])) for f in self.authorized) and
                            max(float(predicted[f]) for f in self.authorized)
                            < float(values.max()) - .00001):
                        continue
                    target_q = np.array(q, copy=True)
                    target_hand = candidate_hand
                    self.hand = {f: np.asarray(v, dtype=float).copy()
                                 for f, v in candidate_hand.items()}
                    print('[V103 RECOVERY FINGER RELEASE CANDIDATE]',
                          'finger=', finger,
                          'delta_deg=', row['delta_deg'],
                          'predicted_depth_mm=', row['predicted_mm'])
                    break
                if target_q is not None:
                    self.steps += 1
                    self.goal_start_q = np.array(q, copy=True)
                    self.goal_start_hand = {f: np.array(hand[f], copy=True)
                                            for f in target_hand}
                    self.goal = (np.array(target_q, copy=True),
                                 {f: np.array(v, copy=True)
                                  for f, v in target_hand.items()})
                    self.wait = self.settled = 0.
                    command(*self.goal)
                    print('[V103 RECOVERY FINGER RELEASE]',
                          'finger=', self.triggering_finger,
                          'steps=', self.steps,
                          'depth_mm=', round(float(values.max()) * 1000., 3))
                    return 'RUNNING'
                print('[V103 RECOVERY NO WRIST STEP CANDIDATES]',
                      'depth_mm=', round(float(values.max()) * 1000., 3),
                      'candidates=', candidate_diagnostics,
                      'finger_release_candidates=', release_diagnostics)
                raise RuntimeError('DEEP_ADJUST_NO_SAFE_WRIST_STEP')
            self.steps += 1
        self.goal_start_q = np.array(q, copy=True)
        self.goal_start_hand = {f: np.array(hand[f], copy=True) for f in target_hand}
        self.goal = (np.array(target_q, copy=True),
                     {f: np.array(v, copy=True) for f, v in target_hand.items()})
        self.wait = self.settled = 0.
        command(*self.goal)
        return 'RUNNING'
