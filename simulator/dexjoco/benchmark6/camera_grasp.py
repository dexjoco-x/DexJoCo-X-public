"""Fit a physical camera grasp from a successful contact example.

The source simulator is a separate FK scratch model. The active task receives
only native actions; camera geometry, mass, friction and success are untouched.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
import zarr
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .models import XML
from .retarget import TargetFK, GeometricRetargeter


def propose(env, source,thin=False):
    source=Path(source);meta=json.loads((source/'summary.json').read_text())
    if not meta.get('success') or meta.get('stable_success_steps',0)<10:
        raise ValueError('Grasp example must be a physically successful pilot')
    # Pre-versioned development pilots belong to the preserved v1 models.
    # This assumption is reported and is only used to propose a new grasp.
    task=meta['task']
    if task!=env.task or task not in ('bimanual_photograph','bimanual_unlock_ipad'):
        raise ValueError('Contact example must match the current object/task')
    if thin and task!='bimanual_photograph':raise ValueError('Thin-camera geometry is not an iPad template')
    version=meta.get('version','benchmark6_v1')
    model=mujoco.MjModel.from_xml_path(str(XML/f"{version}_{task}_{meta['hand']}.xml"))
    data=mujoco.MjData(model)
    z=zarr.open(str(source/'privileged.zarr'),mode='r')['data']
    object_name='ipad_body' if task=='bimanual_unlock_ipad' else 'link_19'
    camera=model.body(object_name).id
    if task=='bimanual_unlock_ipad':
        ready=next((p for p in meta.get('task_metrics',{}).get('phase_history',[]) if p.get('phase')=='ready'),None)
        if ready is None:raise ValueError('iPad example has no verified preparation phase')
        index=int(np.ceil(float(ready['time'])/.05))
        if not 0<=index<len(z['qpos']):raise ValueError('Invalid source preparation index')
    else:
        root=int(model.joint('camera_root').qposadr[0])
        heights=z['qpos'][:,root+2];lifted=np.flatnonzero(heights>heights[0]+.06)
        if not len(lifted):raise ValueError('No lifted-camera grasp in source')
        index=int(lifted[0])
    data.qpos[:]=z['qpos'][index];mujoco.mj_forward(model,data)
    reference=TargetFK(meta['hand'],'left');target=TargetFK(env.hand,'left')
    names=[reference.model.body(i).name for i in reference.bodies]
    points=np.array([data.body(n).xpos+data.body(n).xmat.reshape(3,3)@p
                     for n,p in zip(names,reference.points)])
    object_r=data.xmat[camera].reshape(3,3)
    local=(points-data.xpos[camera])@object_r
    active=env._data.body(object_name);world=local@active.xmat.reshape(3,3).T+active.xpos
    n=min(len(world),len(target.bodies));world=world[:n]
    reference_path=XML/f"{version}_{task}_{meta['hand']}.json"
    source_meta=json.loads(reference_path.read_text())['sides']['left']
    qsource=np.array([z['ctrl'][index,model.actuator(name).id] for name in source_meta['actuator_names']])
    teacher=GeometricRetargeter(env.hand,side='left',source_hand=meta['hand'])
    qprior=teacher.sequence(qsource[None])[0][0]
    rotation=active.xmat.reshape(3,3)@object_r.T@data.site('attachment_site_left').xmat.reshape(3,3)@teacher.alignment[:3,:3]
    if thin:
        # Grasp the SAME camera across its front/back thickness, away from its
        # lens, instead of requiring the old example's 10 cm top/bottom span.
        # Targets are fingertip pad centres with small physical closing preload.
        gid=env._model.geom('camera_body_collision').id
        center=env._data.geom_xpos[gid];box_r=env._data.geom_xmat[gid].reshape(3,3)
        half=env._model.geom_size[gid];n=len(target.bodies)
        local_targets=np.zeros((n,3));local_targets[0]=[-.055,0,-half[2]-.005]
        local_targets[1:,0]=np.linspace(-half[0]+.019,-.039,n-1)
        local_targets[1:,2]=half[2]+.005
        world=local_targets@box_r.T+center
        rotation=active.xmat.reshape(3,3)@Rotation.from_euler('x',np.pi/2).as_matrix()@object_r.T@data.site('attachment_site_left').xmat.reshape(3,3)@teacher.alignment[:3,:3]
    position=world.mean(0)-rotation@target.evaluate(qprior)[:n].mean(0)
    def predict(value):
        rot=Rotation.from_rotvec(value[-3:]).as_matrix()@rotation
        return target.evaluate(value[:-6])[:n]@rot.T+position+value[-6:-3]
    def residual(value):
        return np.r_[(predict(value)-world).ravel()*80,(value[:-6]-qprior)*.02,
                     value[-6:-3]*.5,value[-3:]*.12]
    fit=least_squares(residual,np.r_[qprior,np.zeros(6)],
        bounds=(np.r_[target.low,[-.12]*3,[-1.]*3],np.r_[target.high,[.12]*3,[1.]*3]),
        max_nfev=180)
    pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(fit.x[-3:]).as_matrix()@rotation
    pose[:3,3]=position+fit.x[-6:-3]
    closed=fit.x[:-6];opened=np.clip(.15*closed+.85*target.home,target.low,target.high)
    if thin:
        names=env.hand_metadata['sides']['left']['actuator_names']
        for i,name in enumerate(names):
            if 'WRJ' in name:opened[i]=closed[i]
        if env.hand=='inspire':opened[0]=closed[0]
        if env.hand in ('allegro','leap'):opened[12:14]=closed[12:14]
    return pose,opened,closed,dict(source_step=index,source_model_version=version,
        legacy_source_version_assumed='version' not in meta,
        grasp_strategy='common camera front/back faces' if thin else 'source contact points',
        tip_fit_m=np.linalg.norm(predict(fit.x)-world,axis=1).tolist())
