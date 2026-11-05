"""Read V2 tensor nodes or observed CPU USD writeback; own only a display mesh.

Static targeting data, physical nodes, materials and joint commands are untouched.
Failure restores original visibility; never display USD rest points as live data.
"""
import time
import numpy as np

BODY = '/World/DeformableBackV2'
DISPLAY = '/World/V102_LiveDeformationDisplay'
CLOUD = '/World/DenseBackCandidates'
OWNER = 'v102_live_deformation_v1'
UPDATE_WALL_S = 0.1
LOG_WALL_S = 1.0
COLOR_RANGE_MM = 5.0


def host_array(value, dtype=float):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def validate_nodes(value, count):
    points = host_array(value)
    if points.shape != (1, count, 3) or not np.all(np.isfinite(points)):
        raise RuntimeError('Live node shape/nonfinite mismatch: '+str(points.shape))
    return points[0].copy()


def front_normals(rest, faces, count):
    front = faces[np.all(faces < count, axis=1)]
    if not len(front):
        raise RuntimeError('No front surface triangles')
    normals = np.zeros((count, 3))
    tri = rest[front]
    area = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    for j in range(3):
        np.add.at(normals, front[:, j], area)
    if np.sum(normals[:, 2]) < 0:
        normals *= -1  # V2 front was explicitly oriented toward world +Z.
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise RuntimeError('Invalid front normals; cannot label inward displacement')
    return normals / lengths[:, None]


def deformation_metrics(rest, current, normals, count):
    if current.shape != rest.shape or not np.all(np.isfinite(current)):
        raise RuntimeError('Invalid current nodes')
    delta = current[:count]-rest[:count]
    inward_mm = -np.sum(delta*normals, axis=1)*1000.
    displacement_mm = np.linalg.norm(delta, axis=1)*1000.
    colors = np.tile([.62, .76, .65], (len(rest), 1))
    amount = np.clip(np.abs(inward_mm)/COLOR_RANGE_MM, 0., 1.)[:, None]
    destination = np.where((inward_mm >= 0)[:, None], [.95, .16, .08], [.12, .38, .95])
    colors[:count] = colors[:count]*(1-amount)+destination*amount
    stats = dict(max_inward_mm=round(max(0., float(np.max(inward_mm))), 3),
                 max_outward_mm=round(max(0., float(-np.min(inward_mm))), 3),
                 max_displacement_mm=round(float(np.max(displacement_mm)), 3),
                 inward_over_1mm_nodes=int(np.count_nonzero(inward_mm >= 1.)))
    return stats, colors.astype(np.float32)


class Monitor:
    def __init__(self, stage):
        self.stage = stage
        self.ready = False
        self.failed = False
        self.mesh = None
        self.saved_visibility = {}
        self.last_update = -1e9
        self.last_log = -1e9
        self.sim_time = 0.
        self.source_prepared = False
        self.points_notice = None
        self.points_revision = 0
        self.last_points_write_sim = None
        self.last_wait_log = -1e9

    def on_points_changed(self, notice, sender):
        if self.Sdf.Path(BODY+'.points') in notice.GetChangedInfoOnlyPaths():
            self.points_revision += 1
            self.last_points_write_sim = self.sim_time

    def wait_for_writeback(self, reason):
        now = time.monotonic()
        if now-self.last_wait_log >= 5.:
            self.last_wait_log = now
            print('[V102 LIVE DEFORMATION WAITING]', reason,
                  'source=USD_POINTS; no live depth claimed; physics/renderer settings unchanged')

    def read_current(self):
        if self.source == 'PHYSX_SIMULATION_NODES':
            if not self.body_view.check():
                raise RuntimeError('Live deformable handle invalid')
            return validate_nodes(self.body_view.get_simulation_nodal_positions(), len(self.rest))
        prim = self.stage.GetPrimAtPath(BODY)
        current = validate_nodes(host_array(prim.GetAttribute('points').Get())[None], len(self.rest))
        # A pre-existing USD points value is NOT evidence of live simulation.
        # PhysX may suppress USD notices while writing back, so also detect
        # actual coordinate changes between physics-step reads.
        changed = not np.array_equal(current, self.usd_previous)
        self.usd_previous = current.copy()
        if changed:
            self.last_points_write_sim = self.sim_time
        if not self.usd_verified:
            if not changed:
                self.wait_for_writeback('waiting for a changed V2 points write after monitor startup')
                return None
            self.usd_verified = True
        age = self.sim_time-self.last_points_write_sim
        if age > 1.:
            self.wait_for_writeback('no fresh points write for %.2fs simulation time; last display may be stale/sleeping' % age)
            return None
        return current

    def initialize(self):
        if self.source_prepared:
            current = self.read_current()
            if current is not None:
                self.create_display(current)
            return
        import omni.physics.tensors as tensors
        from pxr import Usd, UsdGeom, Sdf, Vt, Tf
        self.Usd, self.UsdGeom, self.Sdf, self.Vt = Usd, UsdGeom, Sdf, Vt
        body = self.stage.GetPrimAtPath(BODY)
        if not body:
            raise RuntimeError('V2 body missing')
        # V2 stores the nodal rest/points arrays in the body prim's local
        # coordinates.  The original single-environment scene has an identity
        # body transform, but a prepared dual scene intentionally translates
        # env_1.  A translation-only transform is therefore safe: the mesh,
        # points and generated display all remain in the same local frame.  A
        # rotation/scale/shear would change that correspondence and is still
        # rejected rather than silently mixing frames.
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(body)
        linear = np.asarray(matrix, dtype=float)[:3, :3]
        if not np.allclose(linear, np.eye(3), atol=1e-7, rtol=0):
            raise RuntimeError('V2 body transform has rotation/scale; mapping unverified')
        self.body_world_offset = np.asarray(matrix.ExtractTranslation(), dtype=float)
        if float(np.linalg.norm(self.body_world_offset)) > 1e-9:
            print('[V102 LIVE DEFORMATION FRAME] translation-only body transform accepted;',
                  'local rest/points frame preserved; offset_mm=',
                  np.round(self.body_world_offset * 1000., 3).tolist())
        self.rest = host_array(body.GetAttribute('omniphysics:restShapePoints').Get())
        self.front_count = int(body.GetAttribute('v2:frontVertexCount').Get())
        if self.rest.ndim != 2 or self.rest.shape[1] != 3 or not 0 < self.front_count < len(self.rest):
            raise RuntimeError('Invalid V2 rest geometry/front count')
        self.faces = host_array(body.GetAttribute('surfaceFaceVertexIndices').Get(), np.int64)
        usd_tets = host_array(body.GetAttribute('omniphysics:restTetVtxIndices').Get(), np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or np.min(self.faces) < 0 or np.max(self.faces) >= len(self.rest):
            raise RuntimeError('Invalid V2 boundary faces')
        # Warp supports either pipeline; selecting a frontend does not switch
        # the physics backend. Do not change suppressReadback or GPU settings.
        self.sim_view = tensors.create_simulation_view('warp')
        self.sim_view.set_subspace_roots('/')
        if self.sim_view.device_ordinal == -1:
            # 107.3 CPU deformable tensor getters are not all implemented.
            # Read the single-prim V2 simulation mesh's PhysX USD writeback.
            import carb
            if not carb.settings.get_settings().get('/physics/updateToUsd'):
                raise RuntimeError('CPU tensor backend and USD writeback disabled; no verified live source. No settings changed.')
            if body.GetTypeName() != 'TetMesh' or not body.HasAPI('OmniPhysicsVolumeDeformableSimAPI'):
                raise RuntimeError('CPU writeback requires V2 single-prim simulation TetMesh')
            authored_tets = host_array(body.GetAttribute('tetVertexIndices').Get(), np.int64)
            if not np.array_equal(authored_tets, usd_tets):
                raise RuntimeError('USD simulation/rest topology differs')
            self.source = 'USD_POINTS_WRITEBACK_OBSERVED'
            self.usd_initial = validate_nodes(host_array(body.GetAttribute('points').Get())[None], len(self.rest))
            self.usd_previous = self.usd_initial.copy()
            self.usd_verified = False
            self.points_notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.on_points_changed, self.stage)
        else:
            self.source = 'PHYSX_SIMULATION_NODES'
            self.body_view = self.sim_view.create_volume_deformable_body_view(BODY)
            if self.body_view.count != 1 or list(self.body_view.prim_paths) != [BODY] or not self.body_view.check():
                raise RuntimeError('Live deformable view unavailable or wrong body')
            live_rest = validate_nodes(self.body_view.get_rest_nodal_positions(), len(self.rest))
            live_tets = host_array(self.body_view.get_simulation_element_indices(), np.int64)
            if live_tets.shape != (1, len(usd_tets), 4) or not np.array_equal(live_tets[0], usd_tets):
                raise RuntimeError('Live/authoring topology mapping differs')
            if not np.allclose(live_rest, self.rest, atol=1e-6, rtol=0):
                raise RuntimeError('Live/authoring rest coordinates differ')
        self.normals = front_normals(self.rest, self.faces, self.front_count)
        self.source_prepared = True
        current = self.read_current()
        if current is None:
            return
        self.create_display(current)

    def create_display(self, current):
        Usd, UsdGeom, Sdf, Vt = self.Usd, self.UsdGeom, self.Sdf, self.Vt
        prior = self.stage.GetPrimAtPath(DISPLAY)
        if prior:
            raise RuntimeError('Display path occupied; refusing to overwrite: '+DISPLAY)
        self.vis_backup = Sdf.Layer.CreateAnonymous('v102_visibility_backup')
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.mesh = UsdGeom.Mesh.Define(self.stage, DISPLAY)
            self.mesh.GetPrim().SetCustomDataByKey('owner', OWNER)
            self.mesh.CreateSubdivisionSchemeAttr('none')
            self.mesh.CreateDoubleSidedAttr(True)
            self.mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(self.faces), 3, np.int32)))
            self.mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(self.faces.astype(np.int32).ravel()))
            self.mesh.CreateDisplayOpacityAttr(Vt.FloatArray([1.]))
            self.mesh.GetDisplayColorPrimvar().SetInterpolation('vertex')
            self.write_display(current)
            # Visibility edits only. No cloud points, topology, or physics API.
            for path in (CLOUD, BODY):
                prim = self.stage.GetPrimAtPath(path)
                if prim:
                    prop = path+'.visibility'
                    existed = bool(self.stage.GetSessionLayer().GetAttributeAtPath(prop))
                    if existed:
                        Sdf.CreatePrimInLayer(self.vis_backup, path)
                        Sdf.CopySpec(self.stage.GetSessionLayer(), prop, self.vis_backup, prop)
                    self.saved_visibility[path] = existed
                    UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
        self.ready = True
        print('[V102 LIVE DEFORMATION READY] source='+self.source+'; reference=USD_REST_SHAPE;',
              'green=rest red=inward blue=outward; color scale=5mm; geometry is NOT exaggerated;',
              'reference cloud preserved, temporarily hidden; no collision on display mesh')

    def write_display(self, current):
        stats, colors = deformation_metrics(self.rest, current, self.normals, self.front_count)
        with self.Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.mesh.GetPointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(current.astype(np.float32)))
            self.mesh.GetDisplayColorAttr().Set(self.Vt.Vec3fArray.FromNumpy(colors))
            bounds = np.array([current.min(axis=0), current.max(axis=0)], dtype=np.float32)
            self.mesh.GetExtentAttr().Set(self.Vt.Vec3fArray.FromNumpy(bounds))
        return stats

    def update(self, dt, stage_name):
        if self.failed:
            return
        self.sim_time += float(dt)
        now = time.monotonic()
        if now-self.last_update < UPDATE_WALL_S:
            return
        self.last_update = now
        try:
            if not self.ready:
                self.initialize()
            if not self.ready:
                return
            current = self.read_current()
            if current is None:
                return
            stats = self.write_display(current)
            if now-self.last_log >= LOG_WALL_S:
                self.last_log = now
                print('[V102 LIVE DEFORMATION]', 'stage=', stage_name, 'sim_time_s=', round(self.sim_time, 3),
                      stats, 'source='+self.source,
                      'scope=front_surface_vs_rest; NOT fingertip penetration or measured force')
        except Exception as exc:
            self.failed = True
            print('[V102 LIVE DEFORMATION UNAVAILABLE]', repr(exc),
                  'no static fallback; restoring reference visibility; rerun v102 to retry')
            try:
                self.close()
            except Exception as cleanup_exc:
                print('[V102 LIVE DEFORMATION CLEANUP ERROR]', repr(cleanup_exc))

    def close(self):
        if self.points_notice is not None:
            self.points_notice.Revoke()
            self.points_notice = None
        if not hasattr(self, 'Usd'):
            return
        try:
            with self.Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                for path, existed in self.saved_visibility.items():
                    prim = self.stage.GetPrimAtPath(path)
                    if not prim or prim.GetAttribute('visibility').Get() != 'invisible':
                        continue  # Preserve any intervening user visibility change.
                    prop = path+'.visibility'
                    if existed:
                        self.Sdf.CopySpec(self.vis_backup, prop, self.stage.GetSessionLayer(), prop)
                    else:
                        prim.RemoveProperty('visibility')
                if self.mesh is not None:
                    prim = self.stage.GetPrimAtPath(DISPLAY)
                    if prim and prim.GetCustomDataByKey('owner') == OWNER:
                        self.stage.RemovePrim(DISPLAY)  # Only this monitor's generated display.
        finally:
            self.saved_visibility.clear()
            self.mesh = None
            self.ready = False
            self.source_prepared = False
            self.body_view = self.sim_view = None
