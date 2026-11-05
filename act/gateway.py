"""Standard-library-only Jetson command/receipt boundary; CLI is MOCK ONLY.

The physical SDK adapter is separately commissioned. This server deliberately
does not impersonate a real controller or label mock positions as readback.
"""
import argparse
import copy
import hmac
import json
import math
import socket
import time
from pathlib import Path

ARM = tuple('j'+str(i) for i in range(1,6))
FINGERS = ('index','middle','ring','pinky')
CHANNELS = ARM + FINGERS
MAX_LINE = 65536


def validate(command):
    if not isinstance(command,dict) or set(command) != set(CHANNELS):
        raise ValueError('Exactly nine actuator targets required')
    for n in ARM:
        if type(command[n]) not in (float,int) or not math.isfinite(command[n]):
            raise ValueError('Finite arm radians required')
    for f in FINGERS:
        v=command[f]
        if not isinstance(v,dict) or set(v)!={'progress','branch'}:
            raise ValueError('Coupled finger progress/branch required')
        p=v['progress'];b=v['branch']
        if type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=100:
            raise ValueError('Outside calibrated curve domain')
        if b not in ('home','curl','release') or (b=='home' and p!=0):
            raise ValueError('Invalid calibration branch')


class MockBackend:
    def __init__(self, fail_on=None):
        self.writes=[];self.fail_on=fail_on

    def write(self, name, value):
        if name==self.fail_on:
            raise RuntimeError('Injected write failure')
        self.writes.append((name,copy.deepcopy(value)))
        return copy.deepcopy(value), 'mock_dispatch'


class Dispatcher:
    """No queue, no retry, one authenticated session, strictly ordered packets."""
    def __init__(self, token, backend):
        if not isinstance(token,str) or len(token)<24:
            raise ValueError('Session token must have at least 24 characters')
        self.token=token;self.backend=backend
        self.session=None;self.last_seq=-1;self.stopped=False

    def handle(self, packet):
        if not isinstance(packet,dict) or packet.get('schema')!='act_dispatch_v1':
            raise ValueError('Invalid schema')
        supplied=packet.get('token')
        if not isinstance(supplied,str) or not hmac.compare_digest(supplied,self.token):
            raise ValueError('Wrong token')
        session=packet.get('session');seq=packet.get('seq');kind=packet.get('kind')
        if not isinstance(session,str) or not 1<=len(session)<=128:
            raise ValueError('Invalid session')
        if self.session is not None and self.session!=session:
            raise ValueError('Different session cannot take over')
        if type(seq) is not int or seq<=self.last_seq:
            raise ValueError('Duplicate/out-of-order packet; never auto-retry motor commands')
        if kind not in ('command','stop'):
            raise ValueError('Unknown packet type')
        if kind=='command':
            if self.stopped:
                raise RuntimeError('Session stopped; no automatic restart')
            validate(packet.get('command'))
        self.session=session;self.last_seq=seq
        if kind=='stop':
            self.stopped=True
            return dict(session=session,seq=seq,stopped=True,
                        scope='No further commands; this is not physical emergency braking')
        rows=[];failed=False
        for name in CHANNELS:
            row=dict(name=name,requested=packet['command'][name],success=False,
                     dispatch_monotonic_s=time.monotonic())
            if failed:
                row['error']='not_attempted_after_failure'
            else:
                try:
                    value,source=self.backend.write(name,packet['command'][name])
                    row.update(success=True,effective=value,source=source)
                except Exception as exc:
                    failed=True;self.stopped=True;row['error']=str(exc)
            rows.append(row)
        return dict(session=session,seq=seq,writes=rows,stopped=self.stopped)


def exchange(stream, packet):
    encoded=(json.dumps(packet,allow_nan=False)+'\n').encode()
    if len(encoded)>MAX_LINE:
        raise ValueError('Oversized packet')
    stream.write(encoded);stream.flush()
    raw=stream.readline(MAX_LINE+1)
    if not raw or len(raw)>MAX_LINE or not raw.endswith(b'\n'):
        raise RuntimeError('Missing/invalid acknowledgement; dispatch outcome unknown, do not retry')
    response=json.loads(raw)
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response


def serve(port, token, ready_file):
    # Loopback default is intentional; deployment can use an authenticated SSH
    # tunnel after connection. No robot SDK is loaded by this validation server.
    backend=MockBackend();dispatcher=Dispatcher(token,backend)
    with socket.socket() as server:
        server.bind(('127.0.0.1',port));server.listen(1);server.settimeout(30)
        Path(ready_file).write_text(json.dumps(dict(port=server.getsockname()[1],mode='MOCK_ONLY')))
        connection,_=server.accept()
        with connection:
            connection.settimeout(10)
            with connection.makefile('rwb') as stream:
                while True:
                    raw=stream.readline(MAX_LINE+1)
                    if not raw:break
                    try:
                        if len(raw)>MAX_LINE or not raw.endswith(b'\n'):
                            raise ValueError('Oversized/truncated packet')
                        response=dispatcher.handle(json.loads(raw))
                    except Exception as exc:
                        response={'error':str(exc)}
                        dispatcher.stopped=True
                    stream.write((json.dumps(response,allow_nan=False)+'\n').encode());stream.flush()
                    if dispatcher.stopped:break
    return len(backend.writes)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Mock gateway: never imports or drives Arm_Lib')
    p.add_argument('--port',type=int,default=0);p.add_argument('--token-file',required=True)
    p.add_argument('--ready-file',required=True)
    a=p.parse_args()
    serve(a.port,Path(a.token_file).read_text().strip(),a.ready_file)
