"""ACT-only predictions through the user-authorized nonblocking demo executor.

Inputs: measured joints, previous commands, static task geometry, recurrent actual-feedback memory; no phase input.
No expert trajectory, expert timing, expert endpoint or teacher action is loaded.
"""
import ast,hashlib,json,os,pathlib,time,types,traceback
import numpy as np
from act_closed_loop_common import FINGERS,PHASES,runtime_stage
from act_demo_sweep_prefix_v1 import select_prefix

def act_curve_sample(a,b,u):
 curve=b.get('_act_predicted_curve')
 if curve is None:return ORIGINAL_MOTION(a,b,u)
 z=float(np.clip(u,0,1))*(len(curve)-1);i=min(int(z),len(curve)-2);v=z-i;l=curve[i];r=curve[i+1]
 q=(1-v)*np.asarray(l['q'])+v*np.asarray(r['q']);f={}
 for name in FINGERS:
  value=(1-v)*float(l['fingers'][name])+v*float(r['fingers'][name]);f[name]=ACT_HAND_MODEL.Progress(value,ACT_HAND_MODEL.phase_of(r['fingers'][name]))
 return q,f

def run_policy(app,rt,g,out,project,program_name):
 global NATIVE_MOTION, ACT_HAND_MODEL, ORIGINAL_MOTION
 ORIGINAL_MOTION=g['motion_sample'];ACT_HAND_MODEL=g['HAND_MODEL'];NATIVE_MOTION=act_curve_sample;g['motion_sample']=act_curve_sample
 import torch,omni.physx,omni.usd
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
 from pxr import UsdGeom
 from scipy.spatial.transform import Rotation
 from gpu_plan.plan_codec import loads
 from act_withdrawal_expert_v1 import ACT
 from act_relative_geometry_v1 import PalmGeometry
 root=pathlib.Path(__file__).resolve().parent.parent
 specpath=pathlib.Path(os.environ['ACT_EVAL_SPEC']);spec=json.loads(specpath.read_text())
 checkpoint=pathlib.Path(spec['checkpoint']);checkpoint_bytes=checkpoint.read_bytes();ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 norm=np.load(checkpoint.parent/'normalization.npz');sm=norm['obs_mean'];ss=norm['obs_std'];am=norm['action_mean'];astd=norm['action_std'];assert sm.shape==(64,) and ck['architecture']=='withdrawal_expert_v1'
 torch.manual_seed(24024);torch.cuda.manual_seed_all(24024);torch.set_num_threads(1);model=ACT(chunk=24,input_dim=64).cuda();model.load_state_dict(ck['model']);model.use_specialist=bool(spec['use_specialist']);model.eval()
 memory_stride=int(ck['memory_stride']);assert memory_stride==12
 (out/'evaluated_checkpoint.pt').write_bytes(checkpoint_bytes)
 (out/'evaluation_spec.json').write_text(json.dumps(spec,indent=2))
 context=root/'sources'/os.environ['ACT_REPLAY_SOURCE']/'contexts'/os.environ['ACT_REPLAY_POSE']/'runtime_original_context.npz'
 with np.load(context) as ctx:palm= PalmGeometry(project/'assets/robot_kinematics.urdf',ctx['robot_base_p'],ctx['robot_base_r'])
 from act_actual_approach_geometry_v2 import GeometryAdmission
 geometry_config=json.loads((out/'executed_geometry_config.json').read_text())
 assert geometry_config['checkpoint_sha256']==hashlib.sha256(checkpoint_bytes).hexdigest()
 assert geometry_config['angle_limit_enforced'] is False
 admission=GeometryAdmission(geometry_config)
 assert np.max(abs(np.asarray(g['get_palm_local_normal'](rt.arcs))-admission.axes['normal']))<1e-9
 assert np.max(abs(np.asarray(g['get_palm_local_forward'](rt.arcs))-admission.axes['forward']))<1e-9
 tasks={}
 for tid in spec['targets']:
  p=root/'act_measured_v1/eval_tasks'/(str(tid)+'.typed.json')
  raw=json.loads(p.read_text());assert set(raw)=={'mode','lane_spec','geometry','target','context_sha256'}
  task=loads(p.read_text(),g['HAND_MODEL'].Progress,g['TASK'].FrozenGeometry);tasks[int(tid)]=task
 assert len(spec['targets'])==1
 tid=int(spec['targets'][0])
 native_update=rt.update_execution;native_command=rt.command;native_stop=rt.emergency_stop
 protected=['command','update_execution','contact_report','safety_check','recover_wave_tracking','release_exact_sweep']
 source=(project/'scratch/controller.py').read_text();tree=ast.parse(source)
 proof={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name in protected}
 assert all(k in proof for k in protected)
 (out/'native_protection_provenance.json').write_text(json.dumps({'method_ast_sha256':proof,'native_executor_called_unchanged':False,'native_packet_arrival_wait_enforced':False,'native_packet_tracking_timeout_enforced':False,'physical_joint_angle_limits_enforced':True,'articulation_self_collision_enabled':True,'physics_hz':240,'control_hz':120,'physical_substeps_per_command':2,'no_expert_actions_loaded':True,'checkpoint_sha256':hashlib.sha256(checkpoint_bytes).hexdigest()},indent=2))
 from act_geometry_bounded_query_v1 import install_and_preflight
 install_and_preflight(rt,g,out,root/'physx_bundle_v1/act_withdrawal_profile_p24_3672_v1')
 state={'inferences':0,'complete':False,'last_progress':time.perf_counter(),'phase':'approach','phase_id':0,'hidden':None,'memory_ticks':0,'phase_probabilities':[],'last_memory_step':-1}
 from act_crosspoint_physics_diagnostic_v1 import PhysicsDiagnostic
 physics_diagnostic=PhysicsDiagnostic(rt,g,out)
 samples=[];events=[];begin=time.perf_counter();start_sim=rt.telemetry_sim_time
 live=(out/'raw_physx_samples.live.jsonl').open('w',buffering=1);steps=(out/'policy_steps.live.jsonl').open('w',buffering=1)
 trace=(out/'inference_and_gate_trace.jsonl').open('w',buffering=1)
 _prof_infer=[];_prof_packets=[];_prof_steps=[] # PROFILE_ONLY
 def event(kind,**kw):
  row=dict(type=kind,time=rt.telemetry_sim_time,**kw);events.append(row);print('ACT_EVENT',json.dumps(row),flush=True)
 def stop(self,reason):event('stop',reason=str(reason));native_stop(reason)
 rt.emergency_stop=types.MethodType(stop,rt)
 def joints(q,f):return np.r_[q,np.radians([v for k in FINGERS for v in g['HAND_MODEL'].joint_targets(f[k],k)])]
 # Projection onto the SAME actuator hand curves; does not change the geometry or native guards.
 lookup={}
 for f in FINGERS:
  choices=[g['HAND_MODEL'].Progress(0,'home')]
  choices += [g['HAND_MODEL'].Progress(float(v),branch)for branch in ('curl','release')for v in np.linspace(0,100,1001)]
  values=np.radians(np.array([g['HAND_MODEL'].joint_targets(v,f)for v in choices]));lookup[f]=(choices,values)
 import importlib.util
 ms=importlib.util.spec_from_file_location('frozen_mapping_helper',out/'executed_mapping_helper.py');mm=importlib.util.module_from_spec(ms);ms.loader.exec_module(mm)
 branch_mapper=mm.BranchMapper(g['HAND_MODEL'],lookup)
 mapping_log=(out/'mapping_decisions.jsonl').open('w',buffering=1)
 def current():
  hand=rt.read_actual_finger_triplets_deg();q=np.asarray(rt.read_actual_arm_q());return q,hand,np.r_[q,np.radians([v for f in FINGERS for v in hand[f]])]
 def features():
  q,hand,actual=current();previous=joints(rt.current_arm_command,rt.current_finger_commands)
  # The13 zero placeholders are only for the frozen FK helper signature. They
  # are excluded from the64 model features; no phase label is input to ACT.
  geomfeat=np.r_[actual,previous,tasks[tid]['geometry'],np.zeros(13)]
  continuous=np.r_[geomfeat[:58],palm.features(geomfeat)]
  assert continuous.shape==(64,) and np.isfinite(continuous).all()
  return continuous,actual,previous
 def observe_memory(sample_index):
  assert sample_index!=state['last_memory_step'];continuous,actual,previous=features()
  x=torch.tensor((continuous-sm)/ss,dtype=torch.float32,device='cuda')[None]
  with torch.no_grad():hidden,logits=model.observe(x,state['hidden']);prob=logits.softmax(-1)[0].cpu().numpy()
  assert np.isfinite(prob).all();predicted=int(np.argmax(prob));name=PHASES[predicted]if predicted<13 else 'done'
  old=state['phase'];state.update(hidden=hidden,memory_ticks=state['memory_ticks']+1,last_memory_step=sample_index,phase=name,phase_id=predicted,phase_probabilities=prob.tolist())
  if name!=old:event('model_phase_change',previous=old,phase=name,physics_step=sample_index,probabilities=prob.tolist(),source='model_argmax_no_threshold_no_sequence_whitelist')
  trace.write(json.dumps(dict(kind='memory_feedback',physics_step=sample_index,sim_time=rt.telemetry_sim_time,model_phase=name,phase_probabilities=prob.tolist(),input64=continuous.tolist(),memory_ticks=state['memory_ticks'],source='actual feedback; no phase input'))+'\n')
  if predicted==13:
   state['complete']=True;rt.executing=False;event('model_declared_done',physics_step=sample_index,actual17=actual.tolist())
 def infer():
  _prof_i0=time.perf_counter() # PROFILE_ONLY
  continuous,actual,previous=features();x=torch.tensor((continuous-sm)/ss,dtype=torch.float32,device='cuda')[None]
  _prof_net0=time.perf_counter() # PROFILE_ONLY
  assert state['hidden']is not None
  with torch.no_grad():
   route_prob=model.route_probabilities(state['hidden'])[0].cpu().numpy();chunk=model.predict(x,state['hidden'])[0].cpu().numpy()*astd+am+previous
  _prof_net1=time.perf_counter() # PROFILE_ONLY
  trace.write(json.dumps(dict(kind='expert_route',sim_time=rt.telemetry_sim_time,probabilities=route_prob.tolist(),selected_expert=int(route_prob.argmax())if model.use_specialist else 0,specialist_enabled=model.use_specialist,source='learned_binary_argmax_no_phase_input_no_manual_gate'))+'\n')
  assert chunk.shape==(24,17)and np.isfinite(chunk).all()
  state['trace_id']=state.get('trace_id',0)+1
  trace.write(json.dumps(dict(kind='inference',trace_id=state['trace_id'],sim_time=rt.telemetry_sim_time,target=tid,phase=state['phase'],phase_probabilities=state['phase_probabilities'],input_dim=64,network_input=continuous.tolist(),actual17=actual.tolist(),previous17=previous.tolist(),raw_absolute_chunk=chunk.tolist(),memory_ticks=state['memory_ticks'],phase_input=False,phase_gate=False))+'\n')
  _prof_infer.append(dict(sim_time=rt.telemetry_sim_time,features_ms=(_prof_net0-_prof_i0)*1000,network_and_readback_ms=(_prof_net1-_prof_net0)*1000,infer_total_ms=(time.perf_counter()-_prof_i0)*1000)) # PROFILE_ONLY
  return chunk,None,actual
 def allowed(q,f,phase):
  if phase in ('approach','home','open2'):return []
  p,rot=rt.solver.compute_forward_kinematics(g['PALM_FRAME'],q);answer=[]
  for k in FINGERS:
   v=np.asarray(p)+np.asarray(rot)@g['interpolate_arc'](rt.arcs,k,f[k])[0]
   rel=g['final_surface_relation'](v,rt.back_mesh_query,rt.back_points,rt.back_normals,rt.back_tree);lane=g['lane_relation_for_world_point'](v,rt.current_lane_spec,k)
   if lane['valid']and rel['distance']<=g['CONTACT_NEAR_OFFSET_M']:answer.append(k)
  return answer
 def set_task():
  rt.target_id=tid;rt.current_mode=tasks[tid]['mode'];rt.current_lane_spec=tasks[tid]['lane_spec']
 def next_packet(initial=False):
  _prof_p0=time.perf_counter() # PROFILE_ONLY
  if state['hidden']is None:observe_memory(0)
  if state['complete']:return
  phase=state['phase'];cycle=2 if phase in ('reapproach2','precontact2','wave2','curl2','unload2','open2') else 1
  rt.current_stage=runtime_stage(phase)
  chunk,goal,actual=infer()
  _prof_p1=time.perf_counter() # PROFILE_ONLY
  full_chunk=chunk.copy();chunk=select_prefix(full_chunk)
  raw=chunk[-1];duration=len(chunk)/120
  pq=rt.current_arm_command.copy();pf=dict(rt.current_finger_commands)
  curve=[dict(q=pq.copy(),fingers=dict(pf))];previous_q=pq.copy();previous_f=dict(pf)
  for predicted in chunk:
   # Piecewise linear curve: its peak velocity is delta/dt, not smoothstep1.5*delta/dt.
   limit=np.radians(g['ARM_MAX_SPEED_DEG_S'])/120
   cq=np.clip(previous_q+np.clip(predicted[:5]-previous_q,-limit,limit),rt.lower,rt.upper);guessed={}
   if spec['mapping_mode']=='fixed':
    cf=branch_mapper.map(predicted,previous_f,g['MOTOR_EXECUTION_SETTINGS']);decision=branch_mapper.last
   else:
    guessed={};decision=[]
    for j,f in enumerate(FINGERS):
     choices,values=lookup[f];choice=choices[int(np.argmin(np.sum((values-predicted[5+3*j:8+3*j])**2,axis=1)))];old=float(previous_f[f]);new=float(choice)
     speed=g['MOTOR_EXECUTION_SETTINGS']['release_speed_deg_s']if new<old else g['MOTOR_EXECUTION_SETTINGS']['speed_deg_s']
     maximum=speed/120/g['HAND_MODEL'].MOTOR_DEG_PER_PROGRESS
     guessed[f]=g['HAND_MODEL'].Progress(float(np.clip(new,old-maximum,old+maximum)),g['HAND_MODEL'].phase_of(choice))
     decision.append(dict(finger=f,previous_progress=old,previous_branch=g['HAND_MODEL'].phase_of(previous_f[f]),inverse_winner_branch=g['HAND_MODEL'].phase_of(choice),inverse_winner_progress=float(choice),raw_joint_deg=np.degrees(predicted[5+3*j:8+3*j]).tolist(),max_progress_step=maximum))
    cf=g['HAND_MODEL'].phased(previous_f,guessed)
    for row,f in zip(decision,FINGERS):row.update(retained_branch=g['HAND_MODEL'].phase_of(cf[f]),limited_progress=float(cf[f]),mapped_joint_deg=list(g['HAND_MODEL'].joint_targets(cf[f],f)))
   mapping_log.write(json.dumps(dict(trace_id=state['trace_id'],phase=phase,horizon=len(curve),mode=spec['mapping_mode'],fingers=decision))+'\n')
   curve.append(dict(q=cq.copy(),fingers=cf));previous_q=cq;previous_f=cf
  q=curve[-1]['q'];fingers=curve[-1]['fingers']
  auth0=allowed(pq,pf,phase);auth1=allowed(q,fingers,phase)
  a=dict(q=pq,fingers=pf,active=auth0,authorized_contact_fingers=auth0,duration=0,stage=runtime_stage(phase))
  b=dict(q=q,fingers=fingers,active=auth1,authorized_contact_fingers=auth1,duration=duration,stage=runtime_stage(phase),scratch_required_fingers=list(FINGERS),_act_predicted_curve=curve)
  # Additional original native swept geometry checks, without broad contact exemptions.
  def sampler(u,a=a,b=b):return NATIVE_MOTION(a,b,u)
  _prof_p2=time.perf_counter() # PROFILE_ONLY
  with g['TASK'].working_scope(FINGERS),g['TASK'].request_cache(g['TASK'].settings()['planning_cache_bytes']) as geometry_request_cache:
   geometry_request_cache.register_immutable(rt.back_points,'back_points',id(rt.back_points))
   geometry_request_cache.register_immutable(rt.back_normals,'back_normals',id(rt.back_normals))
   ok,reason=g['segment_mesh_safe'](rt.solver,rt.arcs,pq,pf,q,fingers,rt.back_points,rt.back_normals,rt.back_tree,rt.back_mesh_query,allowed_contact_fingers=set(auth0+auth1),sampler=sampler)
   _prof_p3=time.perf_counter() # PROFILE_ONLY
   if ok:
    aq,ah,_=current();target_hand={f:np.asarray(g['HAND_MODEL'].joint_targets(fingers[f],f))for f in FINGERS}
    ok,reason=rt.release_exact_sweep(aq,ah,q,target_hand,set(auth0+auth1))
  _prof_p4=time.perf_counter() # PROFILE_ONLY
  steps.write(json.dumps(dict(inference_trace_id=state['trace_id'],target=tid,phase=phase,cycle=cycle,raw_act_joint17=raw.tolist(),raw_act_chunk_joint17=full_chunk.tolist(),predicted_horizon=len(full_chunk),executed_prefix=len(chunk),executed_curve_joint17=[joints(k['q'],k['fingers']).tolist()for k in curve],applied_target_joint17=joints(q,fingers).tolist(),learned_endpoint_joint17=None,duration=duration,geometry_passed=bool(ok),geometry_reason=str(reason)))+'\n')
  _prof_packets.append(dict(trace_id=state['trace_id'],sim_time=rt.telemetry_sim_time,initial=initial,prefix=len(chunk),inference_ms=(_prof_p1-_prof_p0)*1000,mapping_and_contact_setup_ms=(_prof_p2-_prof_p1)*1000,planned_sweep_ms=(_prof_p3-_prof_p2)*1000,actual_sweep_ms=(_prof_p4-_prof_p3)*1000,total_blocking_ms=(time.perf_counter()-_prof_p0)*1000,geometry_passed=bool(ok))) # PROFILE_ONLY
  if not ok:rt.emergency_stop('ACT_PREDICTION_REJECTED|'+str(reason));return
  state['inferences']+=1
  if initial:rt.start_execution([a,b],tasks[tid]['mode'],planned_working_fingers=FINGERS,lane_spec=tasks[tid]['lane_spec'])
  else:
   # Previous packet has consumed its commanded samples; packet arrival waits remain OFF.
   assert rt.segment_index>=len(rt.trajectory)
   rt.trajectory=[a,b];rt.segment_index=1
 control_tick={'substep':0}
 def update(self,dt):
  if not self.executing:return
  control_tick['substep']+=1
  assert abs(float(dt)-1/240)<1e-7,dt
  if control_tick['substep']%2==0:return
  dt=float(dt)*2
  try:
   if self.segment_index>=len(self.trajectory):next_packet()
   if self.executing:native_update(dt)
  except Exception:
   event('exception',error=traceback.format_exc());self.emergency_stop('ACT_CONTROL_ERROR')
 native_command(np.asarray(rt.read_actual_arm_q()),rt.read_actual_fingers_deg())
 set_task();next_packet(initial=True)
 rt.update_execution=types.MethodType(update,rt);rt.console_prompt_mode=None
 iface=omni.physx.get_physx_interface();stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache();offsets={};robot=g['ROBOT_ROOT']
 for f in FINGERS:
  d=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot+'/right_'+f+'_distal_link'))).T;t=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot+'/right_'+f+'_tip_link'))).T;offsets[f]=np.linalg.solve(d,t)
 captured={'index':0};last_observed=0;physical_tick={'index':0}
 def before(dt):
  if not rt.executing:return
  physical_tick['index']+=1
  if physical_tick['index']%2==0:return
  dt=float(dt)*2
  phase=state['phase'];cycle=2 if phase in ('reapproach2','precontact2','wave2','curl2','unload2','open2')else 1
  _prof_steps.append(time.perf_counter()) # PROFILE_ONLY
  captured.update(index=captured['index']+1,dt=float(dt),start=rt.telemetry_sim_time,q=rt.current_arm_command.copy(),f=dict(rt.current_finger_commands),target=tid,phase=phase,cycle=cycle)
 def after(dt):
  nonlocal last_observed
  if physical_tick['index']%2!=0:return
  if captured['index']==last_observed:return
  try:
   last_observed=captured['index'];q,hand,actual=current();tips={}
   for f in FINGERS:
    v=iface.get_rigidbody_transformation(robot+'/right_'+f+'_distal_link');assert v.get('ret_val')
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(v['rotation']).as_matrix();T[:3,3]=v['position'];tips[f]=(T@offsets[f])[:3,3]
   rel={f:g['final_surface_relation'](v,rt.back_mesh_query,rt.back_points,rt.back_normals,rt.back_tree)for f,v in tips.items()}
   row=dict(physics_step_index=captured['index'],feedback_sim_time=rt.telemetry_sim_time,applied_command_start_sim_s=captured['start'],applied_command_end_sim_s=captured['start']+captured['dt'],target=captured['target'],reference_phase=captured['phase'],cycle=captured['cycle'],q_actual=q.tolist(),fingers_actual={f:np.asarray(hand[f]).tolist()for f in FINGERS},command17_rad=joints(captured['q'],captured['f']).tolist(),tips={f:v.tolist()for f,v in tips.items()},tip_signed_m={f:float(v['signed'])for f,v in rel.items()},tip_normals={f:np.asarray(v['normal']).tolist()for f,v in rel.items()},tip_lane_valid={f:bool(g['lane_relation_for_world_point'](v,tasks[captured['target']]['lane_spec'],f)['valid'])for f,v in tips.items()})
   assert abs(row['feedback_sim_time']-row['applied_command_end_sim_s'])<1e-6
   if captured['phase'] in ('approach','reapproach2','precontact1','precontact2'):
    measured_palm=iface.get_rigidbody_transformation(robot+'/right_base_link')
    # A rigid link pose is preferred. Some PhysX versions do not expose the
    # fixed palm as a separate body; verified full-chain FK from CURRENT
    # measured joints remains actual-state feedback, never command feedback.
    fk_T=palm.pose(q)
    if measured_palm.get('ret_val'):
     measured_T=np.eye(4);measured_T[:3,:3]=Rotation.from_quat(measured_palm['rotation']).as_matrix();measured_T[:3,3]=measured_palm['position']
     palm_source='PHYSX_RIGID_BODY'
    else:
     measured_T=fk_T;palm_source='VERIFIED_FK_FROM_CURRENT_ACTUAL_JOINTS'
    evidence=admission.evaluate(measured_T,tips,rel,hand,captured['phase'])
    evidence.update(feedback_sim_time=rt.telemetry_sim_time,physics_step=captured['index'],palm_feedback_source=palm_source,palm_PhysX_FK_delta_mm=float(np.linalg.norm(measured_T[:3,3]-fk_T[:3,3])*1000))
    state['actual_admission']=evidence
    row['actual_palm_world']=measured_T.tolist();row['actual_approach_conditions']=evidence
    (out/'latest_actual_approach.json').write_text(json.dumps(evidence,indent=2))
   # Observe at a fixed actual-feedback cadence for BOTH execution prefixes.
   # This updates learned memory and semantic labels, never an external phase
   # counter, arrival vote, completion classifier, or85% rule.
   if captured['index']%memory_stride==0 and not state['complete']:observe_memory(captured['index'])
   row['model_phase_after_feedback']=state['phase'];row['phase_prediction_source']='recurrent_model_argmax'
   samples.append(row);live.write(json.dumps(row)+'\n');physics_diagnostic.after()
  except Exception:
   event('exception',error=traceback.format_exc());rt.emergency_stop('ACT_OBSERVATION_ERROR')
 sub0=iface.subscribe_physics_on_step_events(before,True,1000);sub1=iface.subscribe_physics_on_step_events(after,False,100)
 while rt.executing:
  if rt.telemetry_sim_time-start_sim>=float(spec.get('observation_budget_sim_s',240.)):rt.emergency_stop('EXPERIMENT_OBSERVATION_RESOURCE_TRUNCATION');break
  if rt.telemetry_sim_time-start_sim>=240 or captured['index']>=28800:rt.emergency_stop('ACT_COMMON_SESSION_SIM_LIMIT');break
  if time.perf_counter()-begin>3600:rt.emergency_stop('ACT_COMMON_WALL_RESOURCE_LIMIT');break
  app.update()
  if time.perf_counter()-state['last_progress']>10:
   progress=dict(samples=len(samples),physics_step=captured['index'],sim_s=rt.telemetry_sim_time-start_sim,inferences=state['inferences'],phase=[tid,state['phase'],None],wall_s=time.perf_counter()-begin)
   (out/'progress.json').write_text(json.dumps(progress));print('ACT_PROGRESS',json.dumps(progress),flush=True);state['last_progress']=time.perf_counter()
 (out/'packet_timing.json').write_text(json.dumps(dict(inferences=_prof_infer,packets=_prof_packets,physics_callback_wall_intervals_ms=(np.diff(_prof_steps)*1000).tolist(),measurement_scope='synchronous real wall time in the simulation process; no control edits'),indent=2)) # PROFILE_ONLY
 live.close();steps.close();trace.close();mapping_log.close();sub0=None;sub1=None
 q,hand,actual=current();armerr=float(np.max(np.abs(np.degrees(q-rt.home_q))));handerr=max(float(np.max(np.abs(hand[f]-g['HAND_MODEL'].HOME_DEG[f])))for f in FINGERS)
 result=dict(status='MODEL_DECLARED_DONE'if state['complete']else'ACT_STOPPED',model_declared_done=state['complete'],trajectory_complete=False,task_qualified=False,physics_executed=bool(samples),policy_controls_simulation=True,hardware_executed=False,original_collection_modified=False,control_hz=120,physics_hz=240,physical_substeps=physical_tick['index'],action_chunk_steps=24,executed_actions_per_inference=int(spec['execution_prefix']),inferences=state['inferences'],feedback_samples=len(samples),sim_s=rt.telemetry_sim_time-start_sim,checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),events=events,stop_events=[e for e in events if e['type']in('stop','exception')],final_home_check=dict(max_arm_error_deg=armerr,max_finger_error_deg=handerr,passed=bool(armerr<=g['KEYFRAME_ARM_ARRIVAL_TOL_DEG']and handerr<=g['KEYFRAME_FINGER_ARRIVAL_TOL_DEG']),diagnostic_only_not_phase_gate=True),phase_input=False,phase_schedule=False,phase_argmax_no_threshold=True,endpoint_gate=False,completion_classifier_used=False,ring85_gate=False,feedback_votes_used=False,memory_stride_samples=memory_stride,memory_ticks=state['memory_ticks'],geometry_approach_admission=False,expert_or_IK_commands_used=False,executor='Recurrent ACT actions and model semantic phase; no external phase sequence or completion gate. Physical limits, arm selfcollision, finite/rate/swept checks preserved. Back collision remains OFF demo setting.',no_expert_actions_loaded=True)

 return result,samples
