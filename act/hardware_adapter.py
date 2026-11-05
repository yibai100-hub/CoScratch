"""Commissioning library for an injected Arm_Lib-compatible device.

No SDK import/connection or motion occurs automatically. The active pending
binding rejects construction. Coordinate transforms are explicit and reversible.
This module is not a 120 Hz hardware scheduler or a collision-monitor substitute.
"""
import contextlib
import copy
import io
import math
import threading
import time

ARM=tuple('j'+str(i)for i in range(1,6))
FINGERS=('index','middle','ring','pinky')


def scalar(v):
    if type(v) not in (int,float) or not math.isfinite(v):
        raise ValueError('Unverified/non-finite mapping parameter')
    return float(v)


class ArmAdapter:
    def __init__(self, binding, device):
        if binding.get('commissioned') is not True:
            raise RuntimeError('Physical zero/sign/channel binding is not commissioned')
        self.binding=copy.deepcopy(binding);self.device=device
        self.lock=threading.Lock()
        for n in ARM:
            c=self.binding['arm'][n]
            if c['protocol']!=('pwm' if n=='j2' else 'serial'):
                raise ValueError('Protocol mismatch')
            self._check_affine(c,'model_to_physical')
            self._check_affine(c,'write')
            if n!='j2':self._check_affine(c,'read')
            if type(c['channel'])is not int:raise ValueError('Channel required')
        for n in FINGERS:
            c=self.binding['fingers'][n]
            self._check_affine(c,'curve_motor_to_hardware')
        serial=[self.binding['arm'][n]['channel']for n in ARM if n!='j2']
        pwm=[self.binding['arm']['j2']['channel']]+[self.binding['fingers'][f]['channel']for f in FINGERS]
        if len(set(serial))!=4 or len(set(pwm))!=5:
            raise ValueError('Duplicate physical actuator route')
        if any(n not in (1,2,3,4,5,6)for n in serial) or set(pwm)!={1,2,3,4,5}:
            raise ValueError('Unsupported physical channel')

    @staticmethod
    def _check_affine(c,prefix):
        if scalar(c[prefix+'_sign'])not in (-1.,1.):raise ValueError('Direction must be +/-1')
        scalar(c[prefix+'_offset_deg'])

    def model_to_physical(self,n,q):
        c=self.binding['arm'][n]
        return c['model_to_physical_sign']*math.degrees(scalar(q))+c['model_to_physical_offset_deg']

    def physical_to_model(self,n,v):
        c=self.binding['arm'][n]
        return math.radians((scalar(v)-c['model_to_physical_offset_deg'])/c['model_to_physical_sign'])

    def _call(self,fn,*args):
        capture=io.StringIO()
        with self.lock,contextlib.redirect_stdout(capture):
            result=fn(*args)
        if capture.getvalue().strip() or result is False:
            raise RuntimeError('SDK reported failure: '+capture.getvalue().strip())
        return result

    def write(self,name,value,duration_ms=100):
        # duration is a hardware transport parameter, not a task phase timeout.
        if type(duration_ms)is not int or not 1<=duration_ms<=65535:
            raise ValueError('Invalid SDK move duration')
        if name in ARM:
            c=self.binding['arm'][name];physical=self.model_to_physical(name,value)
            api=c['write_sign']*physical+c['write_offset_deg']
            protocol=c['protocol'];ch=c['channel']
            maximum=270 if protocol=='serial' and ch==5 else 180
            if not 0<=api<=maximum:
                raise ValueError('Outside SDK representable command range; not clipped')
            if protocol=='pwm':
                api=int(round(api));self._call(self.device.Arm_PWM_servo_write,ch,api)
                effective_physical=(api-c['write_offset_deg'])/c['write_sign']
            else:
                angle=180-api if ch in (3,4)else api
                span,origin,extent=(3320,380,270)if ch==5 else (2200,900,180)
                count=int(span*angle/extent+origin)
                decoded=(count-origin)*extent/span
                decoded=180-decoded if ch in (3,4)else decoded
                self._call(self.device.Arm_serial_servo_write,ch,float(api),duration_ms)
                effective_physical=(decoded-c['write_offset_deg'])/c['write_sign']
            return self.physical_to_model(name,effective_physical),'sdk_dispatch'
        if name not in FINGERS:raise ValueError('Unknown actuator')
        c=self.binding['fingers'][name];p=scalar(value['progress']);b=value['branch']
        if not 0<=p<=100 or b not in ('home','curl','release')or(b=='home'and p!=0):
            raise ValueError('Invalid finger curve command')
        curve_motor=1+1.78*p
        api=int(round(c['curve_motor_to_hardware_sign']*curve_motor+c['curve_motor_to_hardware_offset_deg']))
        if not 0<=api<=180:raise ValueError('Outside PWM SDK domain; not clipped')
        effective_motor=(api-c['curve_motor_to_hardware_offset_deg'])/c['curve_motor_to_hardware_sign']
        effective_progress=(effective_motor-1)/1.78
        if not 0<=effective_progress<=100:
            raise ValueError('Quantized target outside calibrated curve; no extrapolation')
        self._call(self.device.Arm_PWM_servo_write,c['channel'],api)
        return dict(progress=effective_progress,branch=b),'sdk_dispatch'

    def read(self,name):
        if name not in ('j1','j3','j4','j5'):
            raise ValueError('No physical feedback for J2/fingers')
        c=self.binding['arm'][name]
        raw=self._call(self.device.Arm_serial_servo_read,c['channel'])
        if raw is None:raise RuntimeError('Missing serial measurement')
        physical=c['read_sign']*scalar(raw)+c['read_offset_deg']
        return dict(model_rad=self.physical_to_model(name,physical),
                    received_monotonic_s=time.monotonic(),source='serial_readback')
