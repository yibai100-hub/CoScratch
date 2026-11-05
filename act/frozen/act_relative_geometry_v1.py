import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
class PalmGeometry:
 def __init__(self,urdf,base_p,base_r):
  self.bp=np.asarray(base_p,dtype=np.float64);self.br=np.asarray(base_r,dtype=np.float64)
  assert self.bp.shape==(3,) and self.br.shape==(3,3)
  assert np.max(abs(self.br.T@self.br-np.eye(3)))<1e-8 and abs(np.linalg.det(self.br)-1)<1e-8
  parents={j.find('child').get('link'):j for j in ET.parse(urdf).getroot().findall('joint')};chain=[];link='right_base_link'
  while link!='base_link':
   j=parents[link];n=j.get('name');kind=j.get('type');org=j.find('origin');T=np.eye(4)
   if org is not None:
    T[:3,3]=np.fromstring(org.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(org.get('rpy','0 0 0'),sep=' ')).as_matrix()
   idx=int(n[5:])-1 if n in ['joint'+str(i)for i in range(1,6)] else None
   assert kind=='fixed' or idx is not None
   node=j.find('axis');axis=np.fromstring(node.get('xyz','1 0 0')if node is not None else '1 0 0',sep=' ');axis/=np.linalg.norm(axis)
   chain.append((idx,T,axis));link=j.find('parent').get('link')
  self.chain=list(reversed(chain));assert sorted(i for i,t,a in self.chain if i is not None)==list(range(5))
 def pose(self,q):
  q=np.asarray(q,dtype=np.float64);one=q.ndim==1;q=np.atleast_2d(q);assert q.shape[1]>=5 and np.isfinite(q).all()
  T=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();T[:,:3,:3]=self.br;T[:,:3,3]=self.bp
  for i,org,axis in self.chain:
   T=T@org
   if i is not None:
    A=np.broadcast_to(np.eye(4),T.shape).copy();A[:,:3,:3]=Rotation.from_rotvec(q[:,i,None]*axis).as_matrix();T=T@A
  return T[0] if one else T
 def features(self,obs):
  obs=np.asarray(obs,dtype=np.float64);one=obs.ndim==1;x=np.atleast_2d(obs);assert x.shape[1]==71
  T=self.pose(x[:,:5]);p=x[:,34:37]*.5;n=x[:,37:46].reshape(-1,3,3)[:,:,2];n=n/np.linalg.norm(n,axis=1)[:,None]
  rot=T[:,:3,:3];d=np.einsum('nji,nj->ni',rot,p-T[:,:3,3]);normal=np.einsum('nji,nj->ni',rot,n)
  result=np.concatenate([d,normal],axis=1);assert np.isfinite(result).all()
  return result[0]if one else result
