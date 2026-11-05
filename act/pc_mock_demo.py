"""One real-model packet through an independent MOCK gateway; never hardware."""
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import numpy as np
from core import ROOT,ARM,FINGERS,MEASURED,Features,Policy,Ledger,ActuatorProjection
from gateway import exchange


def run():
    contract=json.loads((ROOT/'config/model_contract.json').read_text())
    q=np.radians(contract['model_home_deg'])
    command=dict(**{a:float(q[i])for i,a in enumerate(ARM)},
                 **{f:dict(progress=0.,branch='home')for f in FINGERS})
    # A stored target descriptor is only a computer test fixture, not a real goal.
    for line in (ROOT/'fixtures/reference_5367.jsonl').open():
        row=json.loads(line)
        if row['kind']=='inference':geometry=row['network_input'][34:58];break
    model=Policy();features=Features();projection=ActuatorProjection()
    session=secrets.token_hex(12);token=secrets.token_hex(24);ledger=Ledger(session)
    with tempfile.TemporaryDirectory()as td:
        td=Path(td);(td/'token').write_text(token)
        child=subprocess.Popen([sys.executable,str(ROOT/'gateway.py'),'--token-file',str(td/'token'),
                                '--ready-file',str(td/'ready')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+10
            while not (td/'ready').exists():
                if child.poll()is not None:raise RuntimeError(child.communicate()[1].decode())
                if time.monotonic()>deadline:raise TimeoutError('Mock startup')
                time.sleep(.03)
            port=json.loads((td/'ready').read_text())['port'];receipts=[]
            with socket.create_connection(('127.0.0.1',port),timeout=5)as sock:
                with sock.makefile('rwb')as stream:
                    def send(seq,c):
                        ledger.stage(seq,c)
                        receipt=exchange(stream,dict(schema='act_dispatch_v1',token=token,session=session,
                                                     kind='command',seq=seq,command=c))
                        ledger.acknowledge(receipt);receipts.append(receipt)
                    send(0,command)
                    # No fabricated readings are passed into the measured-state API.
                    # Synthetic mock state is explicitly selected for this local test.
                    x=features.build(ledger.previous17(),ledger.previous17(),geometry)
                    phase=model.observe(x,0);chunk,route=model.predict(x,ledger.previous17())
                    for seq,c in enumerate(projection.project(chunk,command),1):send(seq,c)
                    exchange(stream,dict(schema='act_dispatch_v1',token=token,session=session,kind='stop',seq=25))
            child.wait(timeout=5)
            if child.returncode!=0:raise RuntimeError('Mock process failed')
        finally:
            if child.poll()is None:child.terminate();child.wait(timeout=5)
    report=dict(status='PASS',mode='ONE_PACKET_MOCK_ONLY',model_phase=int(phase.argmax()),
                expert=int(route.argmax()),packets_acknowledged=len(receipts),actuator_write_receipts=sum(len(r['writes'])for r in receipts),
                final_estimated17=ledger.previous17().tolist(),
                hardware_commands_sent=False,real_feedback_used=False,
                scope='Model to projection to TCP to mock dispatch to command ledger. Not physical dynamics, timing or task validation.')
    (ROOT/'reports/mock_end_to_end.json').write_text(json.dumps(report,indent=2))
    (ROOT/'reports/mock_dispatch_receipts.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in receipts))
    print(json.dumps({k:v for k,v in report.items()if k!='final_estimated17'},indent=2))
    return report


if __name__=='__main__':run()
