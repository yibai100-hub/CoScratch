"""Explicit metric camera/base/model transforms, separate from policy features."""
import numpy as np


def rigid(value):
    t=np.asarray(value,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all():raise ValueError('Expected finite 4x4 transform')
    if not np.allclose(t[3],[0,0,0,1],atol=1e-9,rtol=0):raise ValueError('Not homogeneous')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-7,rtol=0)or abs(np.linalg.det(r)-1)>1e-7:
        raise ValueError('Rotation must be right-handed and orthonormal')
    return t


def transform_target(point_camera_m,frame_camera,base_from_camera,model_from_base):
    """For a wrist camera, base_from_camera must be recomputed from current pose.

    No saved legacy hand-eye transform is selected implicitly. This returns
    metric geometry, not a fabricated/padded 24-dimensional task descriptor.
    """
    t=rigid(model_from_base)@rigid(base_from_camera)
    p=np.asarray(point_camera_m,dtype=float);r=np.asarray(frame_camera,dtype=float)
    if p.shape!=(3,)or not np.isfinite(p).all():raise ValueError('Metric XYZ required')
    temp=np.eye(4);temp[:3,:3]=r;rigid(temp)
    return t[:3,:3]@p+t[:3,3],t[:3,:3]@r
