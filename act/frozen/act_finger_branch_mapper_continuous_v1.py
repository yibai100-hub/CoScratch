"""Stateful inverse measured-hand mapping; no physical/safety thresholds changed.

Three consecutive 120 Hz predicted targets must choose a different measured
branch before committing it. Reproject onto retained branch while undecided.
Never infer hysteresis branch solely from the sign of a progress increment.
"""
import numpy as np
from scipy.optimize import minimize_scalar
class BranchMapper:
 def __init__(self,hand,lookup,confirm_samples=3):
  assert confirm_samples==3
  self.hand=hand;self.lookup=lookup;self.confirm=confirm_samples;self.state={};self.last=[]
 def map(self,predicted,previous,settings):
  result={};self.last=[]
  for j,f in enumerate(self.hand.FINGERS):
   choices,values=self.lookup[f];target=np.asarray(predicted[5+3*j:8+3*j]);cost=np.sum((values-target)**2,axis=1);winner=int(np.argmin(cost));candidate=self.hand.phase_of(choices[winner]);old=float(previous[f])
   st=self.state.setdefault(f,dict(branch=self.hand.phase_of(previous[f]),pending=None,count=0))
   if candidate==st['branch']:st.update(pending=None,count=0)
   else:
    st['count']=st['count']+1 if st['pending']==candidate else 1;st['pending']=candidate
    if st['count']>=self.confirm:st.update(branch=candidate,pending=None,count=0)
   ids=[i for i,c in enumerate(choices)if self.hand.phase_of(c)==st['branch']];selected=min(ids,key=lambda i:cost[i]);new=float(choices[selected])
   # The original grid still chooses the hysteresis branch. Only its retained
   # branch progress is refined continuously on the SAME calibrated curve.
   # This preserves sub-grid model commands instead of rounding them to zero.
   grid_progress=new;grid_error=float(cost[selected])
   if len(ids)>1:
    position=ids.index(selected);lo=float(choices[ids[max(0,position-1)]]);hi=float(choices[ids[min(len(ids)-1,position+1)]])
    def objective(p):
     q=np.radians(np.asarray(self.hand.joint_targets(self.hand.Progress(float(p),st['branch']),f)))
     return float(np.sum((q-target)**2))
    if hi>lo:
     fit=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-9,'maxiter':40})
     candidates=[grid_progress,lo,hi]
     if fit.success and np.isfinite(fit.x):candidates.append(float(fit.x))
     new=min(candidates,key=objective)
    refined_error=objective(new)
   else:refined_error=grid_error
   speed=settings['release_speed_deg_s']if new<old else settings['speed_deg_s'];maximum=speed/120/self.hand.MOTOR_DEG_PER_PROGRESS
   limited=float(np.clip(new,old-maximum,old+maximum));result[f]=self.hand.Progress(limited,st['branch'])
   self.last.append(dict(finger=f,previous_progress=old,previous_branch=self.hand.phase_of(previous[f]),raw_joint_deg=np.degrees(target).tolist(),inverse_winner_branch=candidate,inverse_winner_progress=float(choices[winner]),retained_branch=st['branch'],pending=st['pending'],confirmation_count=st['count'],requested_progress=new,grid_selected_progress=grid_progress,continuous_projection=True,projection_squared_error_before_rad2=grid_error,projection_squared_error_after_rad2=refined_error,limited_progress=limited,max_progress_step=maximum,mapped_joint_deg=list(self.hand.joint_targets(result[f],f))))
  return result
