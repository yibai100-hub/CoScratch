"""PC policy and hybrid observations. No robot imports, sockets or motion on import."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'frozen'))
from act_relative_geometry_v1 import PalmGeometry
from hand import measured_runtime as hand
from act_finger_branch_mapper_continuous_v1 import BranchMapper

FINGERS = hand.FINGERS
ARM = tuple('j' + str(i) for i in range(1, 6))
CHANNELS = ARM + FINGERS
MEASURED = ('j1', 'j3', 'j4', 'j5')


def finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + ': incorrect shape or non-finite value')
    return result


def verify_sources():
    rows = json.loads((ROOT / 'reference/source_manifest.json').read_text())
    for row in rows:
        p = ROOT / row['path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Frozen asset changed: ' + row['path'])
    return len(rows)


class Features:
    def __init__(self):
        self.contract = json.loads((ROOT / 'config/model_contract.json').read_text())
        self.palm = PalmGeometry(ROOT / 'assets/robot_kinematics.urdf',
                                 self.contract['base_p'], self.contract['base_r'])

    def build(self, actual17, previous17, geometry24):
        a = finite(actual17, (17,), 'actual17')
        p = finite(previous17, (17,), 'previous17')
        g = finite(geometry24, (24,), 'geometry24')
        # These zero placeholders are exclusively the frozen FK helper signature.
        # They do not enter the policy. Geometry24 is an encoded target descriptor,
        # not a camera XYZ triple padded with zeros.
        original = np.r_[a, p, g, np.zeros(13)]
        return finite(np.r_[original[:58], self.palm.features(original)], (64,), 'features')


class Policy:
    def __init__(self, device='cpu'):
        import torch
        from act_withdrawal_expert_v1 import ACT
        verify_sources()
        torch.set_num_threads(1)
        self.torch = torch
        self.device = device
        ck = torch.load(ROOT / 'weights/final.pt', map_location='cpu', weights_only=True)
        if ck['architecture'] != 'withdrawal_expert_v1' or ck['memory_stride'] != 12:
            raise ValueError('Wrong architecture or memory cadence')
        self.model = ACT(chunk=24, input_dim=64).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.model.use_specialist = True
        with np.load(ROOT / 'weights/normalization.npz') as n:
            self.sm, self.ss = n['obs_mean'].copy(), n['obs_std'].copy()
            self.am, self.astd = n['action_mean'].copy(), n['action_std'].copy()
        if not (np.isfinite(self.ss).all() and (self.ss > 0).all()):
            raise ValueError('Invalid normalization')
        self.reset()

    def reset(self):
        self.hidden = None
        self.last_sample = None
        self.memory_ticks = 0
        self.phase_id = None

    def tensor(self, x):
        v = finite(x, (64,), 'input64')
        return self.torch.as_tensor((v - self.sm) / self.ss,
                                   dtype=self.torch.float32, device=self.device)[None]

    def observe(self, x, sample_index):
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError('Nonnegative integer control sample required')
        if self.last_sample is not None and sample_index != self.last_sample + 12:
            raise ValueError('Memory requires consecutive 12-control-sample intervals; no invented feedback')
        with self.torch.inference_mode():
            self.hidden, logits = self.model.observe(self.tensor(x), self.hidden)
            prob = logits.softmax(-1)[0].cpu().numpy()
        finite(prob, (14,), 'phase probabilities')
        self.phase_id = int(prob.argmax())
        self.last_sample = sample_index
        self.memory_ticks += 1
        return prob

    def predict(self, x, previous17):
        if self.hidden is None:
            raise ValueError('Observe before prediction')
        if self.phase_id == 13:
            raise RuntimeError('Model declared DONE; do not restart automatically')
        with self.torch.inference_mode():
            delta = self.model.predict(self.tensor(x), self.hidden)[0].cpu().numpy()
            route = self.model.route_probabilities(self.hidden)[0].cpu().numpy()
        chunk = delta * self.astd + self.am + finite(previous17, (17,), 'previous17')
        return finite(chunk, (24, 17), 'action chunk'), route


class Ledger:
    """Update only from gateway dispatch receipts; measurements remain separate.

    This is a last-command estimate, not a dynamics observer or shaft feedback.
    SDK success means the SDK call returned, not that a PWM servo arrived.
    """
    def __init__(self, session):
        self.session = str(session)
        self.commands = {}
        self.feedback = {}
        self.pending = None
        self.last_seq = -1
        self.faulted = False

    def stage(self, seq, command):
        if self.faulted or self.pending is not None:
            raise RuntimeError('Outstanding or failed dispatch; reconcile first')
        if type(seq) is not int or seq <= self.last_seq:
            raise ValueError('Non-increasing sequence')
        validate_command(command)
        self.pending = (seq, copy.deepcopy(command))

    def acknowledge(self, receipt):
        if self.pending is None:
            raise ValueError('No matching dispatch')
        seq, command = self.pending
        if receipt.get('session') != self.session or receipt.get('seq') != seq:
            raise ValueError('Wrong session or sequence')
        writes = receipt.get('writes')
        if not isinstance(writes, list) or len(writes) != len(CHANNELS):
            raise ValueError('One receipt for each actuator required')
        names = [r.get('name') for r in writes]
        if names != list(CHANNELS):
            raise ValueError('Receipt order/identity mismatch')
        updates = {}
        any_failure = False
        for r in writes:
            n = r['name']
            if type(r.get('success')) is not bool:
                raise ValueError('Missing dispatch result')
            if r['success']:
                if r.get('source') not in ('mock_dispatch', 'sdk_dispatch'):
                    raise ValueError('Unsupported state source')
                if r.get('requested') != command[n]:
                    raise ValueError('Receipt is not for staged command')
                v = r['effective']
                validate_value(n, v)
                updates[n] = copy.deepcopy(v)
            else:
                any_failure = True
        self.commands.update(updates)
        self.last_seq = seq
        self.pending = None
        self.faulted = any_failure

    def ingest_measurements(self, readings):
        updates = {}
        for name, row in readings.items():
            if name not in MEASURED or row.get('source') != 'serial_readback':
                raise ValueError('J2/finger commands cannot be labeled measurements')
            q = float(row['model_rad']); stamp = float(row['received_monotonic_s'])
            if not np.isfinite([q, stamp]).all():
                raise ValueError('Invalid measurement')
            if name in self.feedback and stamp <= self.feedback[name]['received_monotonic_s']:
                raise ValueError('Repeated or out-of-order measurement')
            updates[name] = copy.deepcopy(row)
        self.feedback.update(updates)

    def previous17(self):
        if set(self.commands) != set(CHANNELS):
            raise RuntimeError('No complete dispatch history; startup pose must not be assumed')
        fingers = []
        for f in FINGERS:
            v = self.commands[f]
            fingers.extend(np.radians(hand.joint_targets(hand.Progress(v['progress'], v['branch']), f)))
        return np.r_[[self.commands[a] for a in ARM], fingers]

    def observation17(self, now, max_age_s):
        if self.faulted:
            raise RuntimeError('Partial/failed write; state requires reconciliation')
        if not np.isfinite([now, max_age_s]).all() or max_age_s <= 0:
            raise ValueError('Invalid feedback freshness budget')
        actual = self.previous17().copy()
        source = ['command_estimate'] * 17
        ages = {}
        for a in MEASURED:
            row = self.feedback.get(a)
            if row is None:
                raise RuntimeError('Missing serial readback: ' + a)
            age = now - row['received_monotonic_s']
            if not 0 <= age <= max_age_s:
                raise RuntimeError('Stale serial readback: ' + a)
            actual[int(a[1:])-1] = row['model_rad']
            source[int(a[1:])-1] = 'serial_readback'
            ages[a] = age
        return actual, dict(source=source, serial_age_s=ages,
                            estimated_indices=[1] + list(range(5,17)))


def validate_value(name, value):
    if name in ARM:
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError('Invalid arm target')
    else:
        if not isinstance(value, dict) or set(value) != {'progress', 'branch'}:
            raise ValueError('Finger requires coupled progress and branch')
        p = float(value['progress']); b = value['branch']
        if not np.isfinite(p) or not 0 <= p <= 100 or b not in ('home','curl','release'):
            raise ValueError('Outside calibrated finger curve domain')
        if b == 'home' and p != 0:
            raise ValueError('Home branch only has one calibrated point')


def validate_command(command):
    if not isinstance(command, dict) or set(command) != set(CHANNELS):
        raise ValueError('Exactly five arm joints and four coupled fingers required')
    for n in CHANNELS:
        validate_value(n, command[n])


class ActuatorProjection:
    """Same calibrated inverse curves and per-control-sample speed as final sim.

    Projected packets are plans only. They never update Ledger without a receipt.
    A partially dispatched packet invalidates its unexecuted suffix.
    """
    def __init__(self):
        self.contract = json.loads((ROOT/'config/model_contract.json').read_text())
        lookup = {}
        for f in FINGERS:
            choices = [hand.Progress(0, 'home')]
            choices += [hand.Progress(float(v), branch) for branch in ('curl','release')
                        for v in np.linspace(0,100,1001)]
            values = np.radians(np.asarray([hand.joint_targets(c,f) for c in choices]))
            lookup[f] = (choices, values)
        self.mapper = BranchMapper(hand, lookup)

    def project(self, chunk, previous_command, prefix=24):
        validate_command(previous_command)
        finite(chunk, (24,17), 'action chunk')
        if prefix not in (12,24):
            raise ValueError('Supported execution prefix 12 or 24')
        q = np.asarray([previous_command[a] for a in ARM])
        f = {n: hand.Progress(previous_command[n]['progress'], previous_command[n]['branch']) for n in FINGERS}
        output=[]
        for row in chunk[:prefix]:
            limit=np.radians(self.contract['arm_speed_deg_s'])/120
            q=q+np.clip(row[:5]-q,-limit,limit)
            f=self.mapper.map(row,f,self.contract['finger_settings'])
            output.append(dict(**{a:float(q[i]) for i,a in enumerate(ARM)},
                               **{n:dict(progress=float(f[n]),branch=hand.phase_of(f[n])) for n in FINGERS}))
        return output
