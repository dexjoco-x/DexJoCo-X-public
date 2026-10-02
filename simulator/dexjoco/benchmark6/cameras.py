"""A common wrist camera rig and a limited, explicit self-occlusion check."""
import mujoco
import os
import numpy as np
from scipy.spatial.transform import Rotation

LEGACY_RIG=dict(name='common_tcp_side85_v1',position_tcp=[.14,.22,.10],
               look_at_tcp=[.16,0,.18],vertical_fov_deg=85.)
OFFICIAL_RIG=dict(name='dexjoco_official_allegro60_hanoi_front_v1',
    position_tcp=[-.14,-.03,.04],vertical_fov_deg=60.,
    reference_tcp_site_quat_wxyz=[0.,0.,0.,1.],
    reference_attachment_quat_single_wxyz=[-.353553,.612372,-.612372,.352553],
    reference_attachment_quat_dual_wxyz=[-.353553,.612372,-.612372,.353553],
    hanoi_main_camera='front',cross_hand_mapping='same attachment_site/TCP extrinsics',
    real_world_calibrated=False)
PROFILE=os.environ.get('BENCHMARK7_CAMERA_PROFILE',OFFICIAL_RIG['name'])
if PROFILE not in (OFFICIAL_RIG['name'],LEGACY_RIG['name']):
    raise ValueError(f'Unknown BENCHMARK7_CAMERA_PROFILE: {PROFILE}')
RIG=LEGACY_RIG if PROFILE==LEGACY_RIG['name'] else OFFICIAL_RIG


def apply_wrist_rig(env):
    m=env._model
    for side,sid in env._tcp_sites.items():
        body=int(m.site_bodyid[sid]);r=Rotation.from_quat(m.site_quat[sid],scalar_first=True).as_matrix()
        position=m.site_pos[sid]+r@RIG['position_tcp']
        if PROFILE==LEGACY_RIG['name']:
            target=m.site_pos[sid]+r@RIG['look_at_tcp']
            back=position-target;back/=np.linalg.norm(back)
            right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
            rotation=np.column_stack((right,up,back))
        else:
            # Official XML cameras are in the attachment BODY frame. Convert
            # through its Z180 TCP site before mapping to another hand's TCP.
            variant='single' if len(env._tcp_sites)==1 else 'dual'
            ref=Rotation.from_quat(RIG[f'reference_attachment_quat_{variant}_wxyz'],scalar_first=True).as_matrix()
            tcp=Rotation.from_quat(RIG['reference_tcp_site_quat_wxyz'],scalar_first=True).as_matrix()
            rotation=r@tcp.T@ref
        quat=Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        ids=[i for i in range(m.ncam) if m.cam_bodyid[i]==body]
        if len(ids)!=1:raise ValueError(f'Expected one wrist camera at {side} attachment')
        m.cam_pos[ids[0]]=position;m.cam_quat[ids[0]]=quat;m.cam_fovy[ids[0]]=RIG['vertical_fov_deg']
    if PROFILE==OFFICIAL_RIG['name'] and env.task=='single_hand_hanoi':
        cid=m.camera('front').id
        m.cam_pos[cid]=[1.3,0.,1.6]
        q=np.array([.5963678,.3799282,.3799282,.5963678])
        m.cam_quat[cid]=q/np.linalg.norm(q);m.cam_fovy[cid]=45.
        env.camera_id=(cid,*env.camera_id[1:])
    mujoco.mj_forward(m,env._data)


def rigid_hand_geoms(model):
    """Fixed adapter/base/forearm before the first hand joint, for all hands."""
    rigid=set()
    for i in range(model.nbody):
        if not model.body(i).name.startswith('benchhand_'):continue
        parent=int(model.body_parentid[i])
        if model.body_jntnum[i]==0 and (parent in rigid or not model.body(parent).name.startswith('benchhand_')):
            rigid.add(i)
    return np.array([i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in rigid])


def segmentation_renderer(model, *, height, width):
    """Independent audit context: class IDs must never be MSAA color averages.

    A four-sample framebuffer can produce nonexistent encoded segmentation IDs
    at silhouettes (reproduced in Sharpa photograph). Keep ordinary RGB MSAA
    unchanged; only the separate geometry-ID framebuffer has zero samples.
    The temporary construction setting is restored even on renderer failure.
    """
    samples=int(model.vis.quality.offsamples)
    try:
        model.vis.quality.offsamples=0
        return mujoco.Renderer(model,height=height,width=width)
    finally:
        model.vis.quality.offsamples=samples


def obstruction_fraction(renderer,geom_ids):
    flags=renderer.scene.flags.copy()
    renderer.enable_segmentation_rendering()
    try:seg=renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
        # MuJoCo's renderer does not restore flags when ID decoding raises.
        renderer.scene.flags[:]=flags
    return float(np.mean((seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:,:,0],geom_ids)))


def camera_quality(samples):
    """Reject sustained mounting obstruction, not brief natural finger contact.

    This does NOT prove task visibility or semantics; representative RGB still
    needs visual inspection. Geometry segmentation is audit-only, not VLA input.
    """
    for name,values in samples.items():
        values=np.asarray(values,dtype=float)
        if values.ndim!=1 or not values.size or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError(f'Invalid wrist obstruction samples: {name}')
    result={name:dict(samples=len(values),median_rigid_occlusion=float(np.median(values)),
                     max_rigid_occlusion=float(max(values))) for name,values in samples.items()}
    if not result or any(v['median_rigid_occlusion']>.65 for v in result.values()):
        raise ValueError(f'Wrist camera persistently obstructed by its mounting/hand base: {result}')
    return dict(passed=True,wrist_views=result,median_rigid_occlusion_limit=.65,
                limitation='Checks mounting occlusion only, not guaranteed object visibility in every frame.')
