import copy
import json
import math
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import (ARM,FINGERS,CHANNELS,MEASURED,Features,Policy,Ledger,
                  ActuatorProjection,hand,verify_sources,validate_command)
from gateway import Dispatcher,MockBackend,exchange
from hardware_adapter import ArmAdapter
from frames import transform_target


def home_command():
    q=np.radians([0,55,-156,0,0])
    return dict(**{a:float(q[i])for i,a in enumerate(ARM)},
                **{f:dict(progress=0.,branch='home')for f in FINGERS})


def packet(command=None,seq=0,session='test',token='x'*32,kind='command'):
    return dict(schema='act_dispatch_v1',token=token,session=session,seq=seq,
                kind=kind,command=home_command()if command is None else command)


def initialized():
    ledger=Ledger('test');dispatch=Dispatcher('x'*32,MockBackend())
    command=home_command();ledger.stage(0,command)
    ledger.acknowledge(dispatch.handle(packet(command)))
    readings={n:dict(model_rad=float(command[n])+.001,received_monotonic_s=10.,
                    source='serial_readback')for n in MEASURED}
    ledger.ingest_measurements(readings)
    return ledger,dispatch


def fake_binding():
    c=json.loads((ROOT/'config/hardware_binding.pending.json').read_text())
    # Synthetic fixture only, NOT commissioned physical calibration.
    c['commissioned']=True
    for i,n in enumerate(ARM):
        c['arm'][n].update(model_to_physical_sign=1.,
                          model_to_physical_offset_deg=[88,105,157,89,89][i],
                          write_sign=1.,write_offset_deg=0.,read_sign=1.,read_offset_deg=0.)
    for f in FINGERS:
        c['fingers'][f].update(curve_motor_to_hardware_sign=1.,curve_motor_to_hardware_offset_deg=0.)
    return c


class FakeDevice:
    def __init__(self):self.calls=[];self.bad=False
    def Arm_PWM_servo_write(self,*args):
        self.calls.append(('pwm',args))
        if self.bad:raise IOError('Bus error')
    def Arm_serial_servo_write(self,*args):self.calls.append(('serial',args))
    def Arm_serial_servo_read(self,ch):return 90.


class ContractTests(unittest.TestCase):
    def test_camera_base_model_frames_and_units(self):
        base_camera=np.eye(4);base_camera[:3,3]=[.1,.2,.3]
        model_base=np.eye(4);model_base[:3,3]=[0.,-.13583169877529144,.22712773084640503]
        p,r=transform_target([.2,0.,.4],np.eye(3),base_camera,model_base)
        np.testing.assert_allclose(p,[.3,.06416830122470856,.927127730846405])
        np.testing.assert_array_equal(r,np.eye(3))
        reflected=np.eye(4);reflected[0,0]=-1
        with self.assertRaises(ValueError):transform_target([0,0,0],np.eye(3),reflected,model_base)

    def test_frozen_assets_and_model_identity(self):
        self.assertEqual(verify_sources(),22)

    def test_no_estimate_before_dispatch_receipt(self):
        l=Ledger('test');l.stage(0,home_command())
        with self.assertRaises(RuntimeError):l.previous17()

    def test_mixed_measurements_and_estimates(self):
        l,d=initialized();actual,meta=l.observation17(10.1,.2);previous=l.previous17()
        np.testing.assert_allclose(actual[[0,2,3,4]],previous[[0,2,3,4]]+.001)
        np.testing.assert_array_equal(actual[[1]+list(range(5,17))],previous[[1]+list(range(5,17))])
        self.assertEqual(meta['source'].count('serial_readback'),4)
        self.assertEqual(meta['source'].count('command_estimate'),13)

    def test_failed_write_does_not_forge_rest_of_packet(self):
        l,d=initialized();new=home_command();new['j1']+=.01;new['j2']+=.02
        d.backend.fail_on='j2';l.stage(1,new);receipt=d.handle(packet(new,1));l.acknowledge(receipt)
        self.assertEqual(l.commands['j1'],new['j1'])
        self.assertEqual(l.commands['j2'],home_command()['j2'])
        self.assertTrue(l.faulted)
        self.assertEqual(receipt['writes'][2]['error'],'not_attempted_after_failure')
        with self.assertRaises(RuntimeError):l.observation17(10.1,.2)

    def test_wrong_ack_cannot_update_ledger(self):
        l,d=initialized();old=l.previous17();c=home_command();c['j2']+=.1;l.stage(1,c)
        r=d.handle(packet(c,1));r['session']='other'
        with self.assertRaises(ValueError):l.acknowledge(r)
        np.testing.assert_array_equal(old,l.previous17())

    def test_stale_or_falsely_labeled_feedback_rejected(self):
        l,d=initialized()
        with self.assertRaises(RuntimeError):l.observation17(11.,.2)
        with self.assertRaises(ValueError):
            l.ingest_measurements({'j2':dict(model_rad=0,received_monotonic_s=12,source='serial_readback')})
        with self.assertRaises(ValueError):
            l.ingest_measurements({'j1':dict(model_rad=0,received_monotonic_s=9,source='serial_readback')})

    def test_packet_identity_nan_shape_and_stop(self):
        d=Dispatcher('x'*32,MockBackend());d.handle(packet())
        for invalid in (packet(seq=0),packet(seq=1,session='other'),packet(seq=1,token='z'*32)):
            with self.assertRaises(ValueError):d.handle(invalid)
        c=home_command();c['j1']=float('nan')
        with self.assertRaises(ValueError):d.handle(packet(c,1))
        c=home_command();del c['ring']
        with self.assertRaises(ValueError):d.handle(packet(c,1))
        count=len(d.backend.writes);d.handle(packet(seq=1,kind='stop'))
        with self.assertRaises(RuntimeError):d.handle(packet(seq=2))
        self.assertEqual(len(d.backend.writes),count)

    def test_pending_hardware_binding_cannot_drive(self):
        dev=FakeDevice();binding=json.loads((ROOT/'config/hardware_binding.pending.json').read_text())
        with self.assertRaises(RuntimeError):ArmAdapter(binding,dev)
        self.assertEqual(dev.calls,[])

    def test_mapping_roundtrip_and_pwm_quantized_estimate(self):
        dev=FakeDevice();a=ArmAdapter(fake_binding(),dev)
        for n in ARM:
            for q in [-.2,0.,.3]:
                self.assertAlmostEqual(q,a.physical_to_model(n,a.model_to_physical(n,q)),12)
        value,source=a.write('j2',math.radians(20.2))
        self.assertAlmostEqual(math.degrees(value),20.)
        self.assertEqual(source,'sdk_dispatch')
        effective,_=a.write('ring',dict(progress=63.21,branch='curl'))
        self.assertAlmostEqual(effective['progress'],(round(1+1.78*63.21)-1)/1.78)
        self.assertNotEqual(effective['progress'],63.21)

    def test_sdk_failure_and_unavailable_feedback(self):
        dev=FakeDevice();a=ArmAdapter(fake_binding(),dev);dev.bad=True
        with self.assertRaises(IOError):a.write('ring',dict(progress=50.,branch='curl'))
        with self.assertRaises(ValueError):a.read('j2')
        r=a.read('j1');self.assertEqual(r['source'],'serial_readback')

    def test_projection_preserves_one_actuator_per_finger(self):
        a=ActuatorProjection();home=home_command()
        l,d=initialized();q=l.previous17();chunk=np.tile(q,(24,1))
        chunk[:,:5]+=.3
        for i,f in enumerate(FINGERS):
            chunk[:,5+3*i:8+3*i]=np.radians(hand.joint_targets(hand.Progress(80,'curl'),f))
        out=a.project(chunk,home)
        self.assertEqual(len(out),24)
        previous=home
        for c in out:
            validate_command(c);self.assertEqual(set(c),set(CHANNELS))
            for n in ARM:self.assertLessEqual(abs(c[n]-previous[n]),math.radians(18)/120+1e-12)
            for f in FINGERS:
                self.assertLessEqual(abs(c[f]['progress']-previous[f]['progress']),120/120/1.78+1e-9)
            previous=c
        self.assertEqual(l.commands,home)  # Prediction/projection never acknowledges itself.

    def test_two_process_network_dispatch_and_stop(self):
        with tempfile.TemporaryDirectory()as td:
            td=Path(td);token=secrets.token_hex(24);(td/'token').write_text(token)
            child=subprocess.Popen([sys.executable,str(ROOT/'gateway.py'),'--token-file',str(td/'token'),
                                    '--ready-file',str(td/'ready')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+10
                while not (td/'ready').exists():
                    if child.poll()is not None:raise RuntimeError(child.communicate()[1].decode())
                    if time.monotonic()>deadline:raise TimeoutError('Mock gateway startup')
                    time.sleep(.03)
                port=json.loads((td/'ready').read_text())['port']
                l=Ledger('test')
                with socket.create_connection(('127.0.0.1',port),timeout=3)as sock:
                    with sock.makefile('rwb')as stream:
                        for seq in range(8):
                            c=home_command();c['ring']=dict(progress=seq*5.,branch='curl')
                            l.stage(seq,c);l.acknowledge(exchange(stream,packet(c,seq,token=token)))
                        r=exchange(stream,packet(seq=8,kind='stop',token=token));self.assertTrue(r['stopped'])
                self.assertEqual(l.commands['ring']['progress'],35.)
                child.wait(timeout=5);self.assertEqual(child.returncode,0)
            finally:
                if child.poll()is None:child.terminate();child.wait(timeout=5)


class NumericalReplay(unittest.TestCase):
    def test_full_causal_reference_replay(self):
        policy=Policy();features=Features()
        maxima=dict(action_rad=0.,features=0.,phase_probability=0.,route_probability=0.)
        counts=dict(memory=0,inferences=0,phase_argmax_mismatches=0,route_argmax_mismatches=0)
        last_route=None;phases=[];latencies=[]
        for line in (ROOT/'fixtures/reference_5367.jsonl').open():
            row=json.loads(line)
            if row['kind']=='memory_feedback':
                prob=policy.observe(row['input64'],row['physics_step'])
                reference=np.asarray(row['phase_probabilities'])
                maxima['phase_probability']=max(maxima['phase_probability'],float(abs(prob-reference).max()))
                counts['phase_argmax_mismatches']+=int(prob.argmax()!=reference.argmax())
                phases.append(int(prob.argmax()));counts['memory']+=1
            elif row['kind']=='expert_route':last_route=row
            elif row['kind']=='inference':
                x=features.build(row['actual17'],row['previous17'],row['network_input'][34:58])
                maxima['features']=max(maxima['features'],float(abs(x-np.asarray(row['network_input'])).max()))
                t=time.perf_counter();chunk,route=policy.predict(x,row['previous17']);latencies.append((time.perf_counter()-t)*1000)
                maxima['action_rad']=max(maxima['action_rad'],float(abs(chunk-np.asarray(row['raw_absolute_chunk'])).max()))
                maxima['route_probability']=max(maxima['route_probability'],float(abs(route-np.asarray(last_route['probabilities'])).max()))
                counts['route_argmax_mismatches']+=int(route.argmax()!=last_route['selected_expert'])
                counts['inferences']+=1
        report=dict(status='NUMERICAL_REPLAY',counts=counts,max_absolute_errors=maxima,
                    action_max_error_deg=math.degrees(maxima['action_rad']),
                    cpu_prediction_latency_ms=dict(median=float(np.median(latencies)),p95=float(np.percentile(latencies,95))),
                    tolerance=dict(features=1e-8,action_rad=2e-4,probabilities=2e-4),
                    hardware_commands_sent=False,new_physics_executed=False,
                    scope='Original simulation observations replayed; not hybrid-state hardware closed-loop success')
        (ROOT/'reports/numerical_replay.json').write_text(json.dumps(report,indent=2))
        self.assertGreater(counts['memory'],100)
        self.assertLessEqual(maxima['features'],1e-8)
        self.assertLessEqual(maxima['action_rad'],2e-4)
        self.assertLessEqual(maxima['phase_probability'],2e-4)
        self.assertLessEqual(maxima['route_probability'],2e-4)
        self.assertEqual(counts['phase_argmax_mismatches'],0)
        self.assertEqual(counts['route_argmax_mismatches'],0)
        self.assertEqual(phases[-1],13)
        with self.assertRaises(RuntimeError):policy.predict(x,row.get('previous17',np.zeros(17)))
        report['status']='PASS';(ROOT/'reports/numerical_replay.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':unittest.main()
