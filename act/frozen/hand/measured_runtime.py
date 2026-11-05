"""Runtime adapter: measured independent fingers and explicit hysteresis branches."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from hand import measured_model as measured
# Only URDF geometry helpers are shared; no reference angle/curve is used.
from hand.reference_model import hand_chains, tip_local, file_hash
from project_paths import ROOT
MODEL_ID=measured.MODEL_ID
FINGERS=measured.FINGERS
SUFFIXES=measured.SUFFIXES
CONTROL_MAX=measured.CONTROL_MAX
MOTOR_DEG_PER_PROGRESS=measured.MOTOR_DEG_PER_PROGRESS
motor_command_deg=measured.motor_command_deg
HOME_DEG=measured.HOME_DEG
LIMITS_BY_FINGER_DEG=measured.LIMITS_BY_FINGER_DEG
ARC_FILENAME='assets/measured_curl_arcs.csv'
audit_scene=measured.audit_scene

class Progress(float):
    def __new__(cls,value,phase='curl'):
        if phase not in ('curl','release','home'): raise ValueError(phase)
        obj=super().__new__(cls,value);obj.phase=phase;return obj
    def __reduce__(self): return (Progress,(float(self),self.phase))

def phase_of(value): return getattr(value,'phase','curl')
def phased(previous, target):
    result={}
    for f in FINGERS:
        a,b=previous.get(f,0.),target.get(f,0.)
        phase=('curl' if b>a+1e-9 else 'release' if b<a-1e-9 else phase_of(a))
        result[f]=Progress(b,phase)
    return result

def blend(previous,target,alpha):
    end=phased(previous,target)
    return {f:Progress(float(previous.get(f,0.))+alpha*(float(end[f])-float(previous.get(f,0.))),phase_of(end[f])) for f in FINGERS}

def joint_targets(progress,finger,phase=None):
    phase=phase or phase_of(progress)
    if phase=='home': return tuple(HOME_DEG[finger])
    return measured.joint_targets(finger,phase,float(progress))

def joint_values_rad(commands):
    return {'right_%s_%s'%(f,s):math.radians(v) for f in FINGERS for s,v in zip(SUFFIXES,joint_targets(commands.get(f,Progress(0,'home')),f))}

def limits_ok(finger,values,tolerance=1.):
    v=np.asarray(values);bounds=np.asarray(LIMITS_BY_FINGER_DEG[finger])
    return bool(np.isfinite(v).all() and (v>=bounds[:,0]-tolerance).all() and (v<=bounds[:,1]+tolerance).all())

def progress_from_triplet(actual,finger):
    actual=np.asarray(actual);best=(np.inf,Progress(0,'home'))
    error=float(np.max(np.abs(actual-HOME_DEG[finger])))
    if error<1.5: return Progress(0,'home'),error
    for phase in ('curl','release'):
        grid=np.linspace(0,100,201);poses=np.array([joint_targets(p,finger,phase) for p in grid]);i=int(np.argmin(np.sum((poses-actual)**2,axis=1)))
        lo,hi=grid[max(0,i-1)],grid[min(200,i+1)]
        objective=lambda p:float(np.sum((np.array(joint_targets(p,finger,phase))-actual)**2))
        result=minimize_scalar(objective,bounds=(lo,hi),method='bounded')
        p=min((lo,float(result.x),hi),key=objective)
        if objective(p)<best[0]:best=(objective(p),Progress(p,phase))
    return best[1],float(np.max(np.abs(np.array(joint_targets(best[1],finger))-actual)))

def triplet_errors(actual,commands):
    return {f:np.abs(np.asarray(actual[f])-joint_targets(commands[f],f)) for f in FINGERS}

def generate_arcs(urdf):
    chains=hand_chains(urdf)
    for phase in ('curl','release'):
        path=ROOT/('assets/measured_'+phase+'_arcs.csv')
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['finger','progress_pct','local_x_m','local_y_m','local_z_m','mcp_deg','pip_deg','dip_deg','model_id'])
            for f in FINGERS:
                for p in np.linspace(0,100,201):
                    q=joint_targets(p,f,phase);values={'right_%s_%s'%(f,s):math.radians(v) for s,v in zip(SUFFIXES,q)}
                    writer.writerow([f,p,*tip_local(chains[f],values),*q,MODEL_ID])
        path.with_suffix('.json').write_text(json.dumps({'model_id':MODEL_ID,'phase':phase,'calibration_sha256':file_hash(measured.DATA_PATH),'initial_pose_sha256':file_hash(measured.DATA_PATH.parent/'initial_pose.json'),'urdf_sha256':file_hash(urdf),'csv_sha256':file_hash(path)}))

def verify_arc_csv(path,urdf):
    for phase in ('curl','release'):
        p=ROOT/('assets/measured_'+phase+'_arcs.csv');m=json.loads(p.with_suffix('.json').read_text())
        if (m['model_id']!=MODEL_ID or m['phase']!=phase or m['csv_sha256']!=file_hash(p) or m['urdf_sha256']!=file_hash(urdf) or m['calibration_sha256']!=file_hash(measured.DATA_PATH) or m['initial_pose_sha256']!=file_hash(measured.DATA_PATH.parent/'initial_pose.json')):raise RuntimeError('MEASURED_ARC_SOURCE_MISMATCH')
    return True
