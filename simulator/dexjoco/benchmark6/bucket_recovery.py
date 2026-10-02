"""Diagnostic contact-target grasp proposal for a real, unscaled basket handle.

Only the offline FK scratch model is assigned joint states. The task simulation
receives native arm/hand actuator commands, including every failed attempt.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .retarget import TargetFK
from .success_policy import required_success_steps, success_policy
from ..autocollect.linker_teacher import snapshot
from ..autocollect.retarget import _pose_error6,_matrix_to_pose7


def recover(env,rows,result,grip_prior=None,rotation_prior=None):
    corners=env._data.site_xpos[[env._model.site(f"bucket_ref_{i}").id for i in range(8)]]
    food=env._data.body("boxed_food_0").xpos
    if not (np.all(food>=corners.min(0)) and np.all(food<=corners.max(0))):
        result["recovery_skipped"]="payload not inside basket";return rows,result
    fk=TargetFK(env.hand);gid=env._model.geom("handle1").id
    center=env._data.geom_xpos[gid].copy();object_R=env._data.geom_xmat[gid].reshape(3,3).copy()
    normal=object_R[:,0];along=object_R[:,2];n=len(fk.bodies)
    targets=np.array([center-normal*.012]+[center+normal*.012+along*y for y in np.linspace(.035,-.035,n-1)])
    qprior=np.clip(env._data.ctrl[env._native_ctrl_ids["right"]] if grip_prior is None else grip_prior,fk.low,fk.high)
    Rprior=env.tcp()[:3,:3].copy() if rotation_prior is None else np.asarray(rotation_prior).copy()
    pprior=center-Rprior@fk.evaluate(qprior).mean(0)
    def predict(value):
        rot=Rotation.from_rotvec(value[-3:]).as_matrix()@Rprior
        return fk.evaluate(value[:-6])@rot.T+pprior+value[-6:-3]
    def residual(value):
        return np.r_[(predict(value)-targets).ravel()*70,(value[:-6]-qprior)*.015,
                     value[-6:-3]*.2,value[-3:]*.15]
    fitted=least_squares(residual,np.r_[qprior,np.zeros(6)],
        bounds=(np.r_[fk.low,[-.15]*3,[-1.2]*3],np.r_[fk.high,[.15]*3,[1.2]*3]),
        max_nfev=180,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    closed=fitted.x[:-6].copy();opened=np.clip(.25*closed+.75*fk.home,fk.low,fk.high)
    if env.hand=="allegro":
        # Add closing preload through native force-limited actuators once the
        # pads meet the thin handle; geometric touching alone is not a grasp.
        flexion=[1,2,3,5,6,7,9,10,11,13,14,15]
        closed[flexion]=np.minimum(closed[flexion]+.18,fk.high[flexion])
    pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(fitted.x[-3:]).as_matrix()@Rprior
    pose[:3,3]=pprior+fitted.x[-6:-3]
    result["recovery_tip_fit_m"]=np.linalg.norm(predict(fitted.x)-targets,axis=1).tolist()
    result["handle_contact_plan"]=dict(pose=pose.tolist(),closed=closed.tolist(),opened=opened.tolist(),
        handle_center=center.tolist(),grip_prior=qprior.tolist(),rotation_prior=Rprior.tolist())
    required=required_success_steps(env.task)
    stable=0;info={};stopped=False
    def step(target,grip):
        nonlocal stable,info
        obs=env._compute_observation();error=_pose_error6(obs["state"]["tcp_pose"],_matrix_to_pose7(target))
        action=np.r_[np.clip(error[:3]*.35,-.004,.004),np.clip(error[3:]*.3,-.03,.03),grip]
        before=snapshot(env);state=np.r_[obs["state"]["tcp_pose"],obs["state"]["gripper_pose"]]
        _,_,terminated,truncated,info=env.step(action)
        rows.append(dict(state=state,action=action,truth=before,info=info))
        stable=stable+1 if info.get("succeed",False) else 0
        return (terminated or truncated) and not info.get("succeed",False)
    # Open above the handle, approach, close around it, then lift through contact.
    # The hinged handle first unfolds; that motion is not basket-bottom lift.
    # Leave enough arm travel for BOTH unfolding and the unchanged 15 cm lift.
    for phase,count,height in (("approach",90,.10),("descend",70,0),("close",40,0),("lift",160,.38)):
        target=pose.copy();target[2,3]+=height
        for i in range(count):
            alpha=(i+1)/count if phase=="close" else (1 if phase=="lift" else 0)
            grip=(1-alpha)*opened+alpha*closed
            stopped=step(target,grip)
            if stopped or stable>=required:break
        if stopped or stable>=required:break
    result.update(success_policy=success_policy(env.task),success=bool(info.get("succeed",False)),stable_success_steps=stable,
                  steps=len(rows),final_truth=snapshot(env),recovery_attempted=True)
    return rows,result
