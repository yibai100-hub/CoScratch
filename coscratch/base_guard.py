"""Conservative hand/base clearance from full URDF collision bounds.

Uses complete STL triangles, not sparse surface probes. Oriented hand boxes
are tested against actual base triangles; base holes are not filled by a box.
All bounds are compared in the fixed robot-base frame, independent of world pose.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


def corners(lo, hi):
    return np.array(list(itertools.product(*zip(lo, hi))), dtype=float)


def box_gap(a, b):
    gap = np.maximum(0., np.maximum(a[0]-b[1], b[0]-a[1]))
    return float(np.linalg.norm(gap))


class BaseGuard:
    def __init__(self, model, hand_model):
        self.model, self.hand_model = model, hand_model
        self._cache = {}
        self.bounds = {}
        self.base_triangles = []
        for link in model.root.findall('link'):
            name = link.get('name', '')
            if name != 'base_link' and not name.startswith('right_'):
                continue
            points = []
            for c in link.findall('collision'):
                g = c.find('geometry')
                if g is None:
                    raise RuntimeError('BASE_GUARD_MISSING_GEOMETRY:'+name)
                if g.find('mesh') is not None:
                    mesh = g.find('mesh')
                    path = model._resolve_mesh(mesh.get('filename'))
                    triangles = model._stl_triangles(path) if path else None
                    if triangles is None:
                        raise RuntimeError('BASE_GUARD_MESH_UNAVAILABLE:'+name)
                    scale = np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                    v = np.asarray(triangles).reshape(-1,3)*scale
                elif g.find('box') is not None:
                    size = np.fromstring(g.find('box').get('size'),sep=' ')
                    v = corners(-size/2,size/2)
                elif g.find('sphere') is not None:
                    r = float(g.find('sphere').get('radius'))
                    v = corners(np.full(3,-r),np.full(3,r))
                elif g.find('cylinder') is not None:
                    r = float(g.find('cylinder').get('radius'))
                    h = float(g.find('cylinder').get('length'))/2
                    v = corners([-r,-r,-h],[r,r,h])
                else:
                    raise RuntimeError('BASE_GUARD_UNSUPPORTED_GEOMETRY:'+name)
                origin = c.find('origin')
                xyz = np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
                rpy = np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
                points.append(v @ Rotation.from_euler('xyz',rpy).as_matrix().T + xyz)
                if name == 'base_link':
                    if g.find('mesh') is None:
                        raise RuntimeError('BASE_GUARD_REQUIRES_BASE_TRIANGLE_MESH')
                    self.base_triangles.append(points[-1].reshape(-1,3,3))
            if points:
                v = np.vstack(points)
                if not np.all(np.isfinite(v)):
                    raise RuntimeError('BASE_GUARD_NONFINITE:'+name)
                self.bounds[name] = corners(v.min(axis=0),v.max(axis=0))
        if 'base_link' not in self.bounds or len(self.bounds) < 14:
            raise RuntimeError('BASE_GUARD_INCOMPLETE_HAND_OR_BASE')
        base = self.bounds.pop('base_link')
        self.base = (base.min(axis=0),base.max(axis=0))
        self.triangles = np.concatenate(self.base_triangles)
        self.tri_lo = self.triangles.min(axis=1)
        self.tri_hi = self.triangles.max(axis=1)
        centers = self.triangles.mean(axis=1)
        radii = np.linalg.norm(self.triangles-centers[:,None,:],axis=2).max(axis=1)
        self.groups = []
        last = -1.
        for bound in (.005,.02,.05,float('inf')):
            ids = np.where((radii>last)&(radii<=bound))[0]
            if len(ids):
                self.groups.append((ids,cKDTree(centers[ids]),float(radii[ids].max())))
            last = bound

    @staticmethod
    def triangle_box_separation(tri, lo, hi):
        center, half = (lo+hi)/2,(hi-lo)/2
        tri = tri-center
        edges = np.roll(tri,-1,axis=1)-tri
        axes = [np.broadcast_to(np.eye(3),(len(tri),3,3)),
                np.cross(edges[:,0],edges[:,1])[:,None,:]]
        axes.extend(np.cross(edges,np.broadcast_to(axis,edges.shape)) for axis in np.eye(3))
        axes = np.concatenate(axes,axis=1)
        norm = np.linalg.norm(axes,axis=2,keepdims=True)
        axes = axes/np.maximum(norm,1e-15)
        projections = np.einsum('nvc,nac->nav',tri,axes)
        radius = np.sum(np.abs(axes)*half,axis=2)
        gaps = np.maximum(0.,np.maximum(projections.min(axis=2)-radius,
                                       -radius-projections.max(axis=2)))
        return gaps.max(axis=1)

    def clearance(self, q, commands, actual_joint_values=None):
        key = (tuple(np.asarray(q,dtype=float)), tuple(sorted(commands.items())),
               None if actual_joint_values is None else tuple(sorted(actual_joint_values.items())))
        if key in self._cache:
            return self._cache[key]
        result = self._clearance(q,commands,actual_joint_values)
        if len(self._cache) >= 8192:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _clearance(self, q, commands, actual_joint_values=None):
        arm = {'joint%d'%(i+1):float(v) for i,v in enumerate(q)}
        fingers = (self.hand_model.joint_values_rad(commands) if actual_joint_values is None
                   else actual_joint_values)
        links, joints = {}, {}
        closest = (float('inf'),None)
        for name, local in self.bounds.items():
            t = self.model._link_transform(name,arm,fingers,links,joints)
            p = local @ t[:3,:3].T + t[:3,3]
            # Broad phase uses base triangle boxes, not the whole base box.
            # Distances above 50 mm are capped; this remains a lower bound.
            center = p.mean(axis=0)
            radius = np.linalg.norm(p-center,axis=1).max()
            nearby = [ids[tree.query_ball_point(center,radius+.050+tri_radius)]
                      for ids,tree,tri_radius in self.groups]
            ids = np.concatenate(nearby)
            broad = np.linalg.norm(np.maximum(0.,np.maximum(p.min(axis=0)-self.tri_hi[ids],
                                            self.tri_lo[ids]-p.max(axis=0))),axis=1)
            ids = ids[broad < .050]
            distance = .050
            if len(ids):
                tri_local = (self.triangles[ids]-t[:3,3]) @ t[:3,:3]
                distance = min(.050,float(self.triangle_box_separation(
                                      tri_local,local.min(axis=0),local.max(axis=0)).min()))
            if distance < closest[0]:
                closest = (distance,name)
        return closest
