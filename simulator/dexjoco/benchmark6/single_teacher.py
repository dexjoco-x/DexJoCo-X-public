"""Independent benchmark single-arm teacher, initialized from legacy workflow.

Adds explicit reference-frame/phase diagnostics; legacy L20/XHand collectors are
not edited. Control is bounded native actuation, with no object-state writes.
"""
from types import SimpleNamespace
from .success_policy import required_success_steps, success_policy
import numpy as np
from scipy.spatial.transform import Rotation
from ..autocollect.linker_teacher import snapshot,body_pose
from ..autocollect.retarget import _pose7_to_matrix,_matrix_to_pose7,_pose_error6
from ..autocollect.hammer_retarget import infer_hammer_phase_plan,_interpolate_rigid_transform


def payload_ready_for_handle(env):
    """Controller stage gate, NOT a replacement task-success condition."""
    m,d=env._model,env._data
    food=d.body('boxed_food_0');fid=m.body('boxed_food_0').id
    corners=d.site_xpos[[m.site(f'bucket_ref_{i}').id for i in range(8)]]
    if not (np.all(food.xpos>=corners.min(0)) and np.all(food.xpos<=corners.max(0))):return False
    # Require actual support on the basket bottom, no hand-food contact, and
    # a cleared TCP. This does not accept an object merely passing over the rim.
    supported=False
    for contact in d.contact:
        gids=(contact.geom1,contact.geom2);bodies=[int(m.geom_bodyid[g]) for g in gids]
        if fid not in bodies:continue
        other=gids[1-bodies.index(fid)];name=m.geom(other).name or ''
        bodyname=m.body(int(m.geom_bodyid[other])).name or ''
        if bodyname.startswith('benchhand_'):return False
        supported |= name=='bucket_bottom'
    return supported and np.linalg.norm(env.tcp()[:3,3]-food.xpos)>.12


def rollout(env,obs,episode,hand_targets,task,tcp_offset=(0,0,0),frame_mode="two",tip_phase=False,handle_handoff=False):
    if handle_handoff and task!='bucket':raise ValueError('Handle handoff requires bucket')
    required=required_success_steps(task)
    source=episode.states
    names=(("hammer_body",slice(13,20)),("nail",slice(20,27))) if task=="hammer" else (
        ("boxed_food_0",slice(20,27)),("bucket",slice(13,20)))
    frames=[_pose7_to_matrix(body_pose(env,name))@np.linalg.inv(_pose7_to_matrix(source[0,sl])) for name,sl in names]
    if frame_mode=="first":frames[1]=frames[0].copy()
    if task=="hammer":phase=infer_hammer_phase_plan(episode)
    else:
        distances=np.linalg.norm(source[:int(len(source)*.65),:3]-source[0,20:23],axis=1)
        begin=int(np.argmin(distances))+10
        phase=SimpleNamespace(transition_start_step=begin,transition_end_step=begin+35)
    if tip_phase:
        from .retarget import TargetFK
        fk=TargetFK(env.hand);centers=[]
        for state,q in zip(source,hand_targets):
            tcp=_pose7_to_matrix(state[:7]);local=fk.evaluate(q)[:2].mean(0)+np.array(tcp_offset)
            centers.append(tcp[:3,3]+tcp[:3,:3]@local)
        point=source[0,names[0][1]][:3]
        begin=int(np.argmin(np.linalg.norm(np.array(centers)[:int(len(source)*.65)]-point,axis=1)))+10
        phase=SimpleNamespace(transition_start_step=begin,transition_end_step=begin+35)
    rows=[];stable=0;info={};stopped=False;ready_steps=0;handoff_step=None
    def step(action):
        nonlocal obs,info,stable
        action=np.clip(action,env.action_space.low,env.action_space.high)
        before=snapshot(env);state=np.r_[obs["state"]["tcp_pose"],obs["state"]["gripper_pose"]]
        obs,_,terminated,truncated,info=env.step(action)
        success=bool(info.get("succeed",False));stable=stable+1 if success else 0
        rows.append(dict(state=state,action=action,truth=before,info=info))
        return (terminated or truncated) and not success
    for i,action in enumerate(episode.actions):
        alpha=np.clip((i-phase.transition_start_step)/max(1,phase.transition_end_step-phase.transition_start_step-1),0,1)
        transform=_interpolate_rigid_transform(*frames,float(alpha))
        desired=transform@_pose7_to_matrix(source[i,:7]);desired[:3,3]+=desired[:3,:3]@np.asarray(tcp_offset)
        error=_pose_error6(obs["state"]["tcp_pose"],_matrix_to_pose7(desired))
        delta=np.r_[transform[:3,:3]@action[:3],transform[:3,:3]@action[3:6]]+error*np.array([.35]*3+[.25]*3)
        stopped=step(np.r_[delta,hand_targets[i]])
        if stopped or stable>=required:break
        if handle_handoff:
            ready_steps=ready_steps+1 if payload_ready_for_handle(env) else 0
            if ready_steps>=5:
                handoff_step=len(rows)
                break
    if task=="hammer" and stable<required and not stopped:
        hold_rotation=_pose7_to_matrix(obs["state"]["tcp_pose"])[:3,:3];grip=hand_targets[-1]
        gid=env._model.geom("face").id
        for cycle in range(5):
            for count,height in ((50,.085),(30,-.008)):
                for _ in range(count):
                    face=env._data.geom_xpos[gid];axis=env._data.geom_xmat[gid].reshape(3,3)[:,2]
                    radius,half=env._model.geom_size[gid,:2]
                    extent=radius*np.sqrt(max(0,1-axis[2]**2))+half*abs(axis[2])
                    desired=env._data.geom("nail_head").xpos.copy()+[0,0,extent+.0048+height]
                    delta=np.clip((desired-face)*.35,-.005,.005)
                    actual=_pose7_to_matrix(obs["state"]["tcp_pose"])[:3,:3]
                    dr=Rotation.from_matrix(hold_rotation@actual.T).as_rotvec()*.25
                    stopped=step(np.r_[delta,dr,grip])
                    if stopped or stable>=required:break
                if stopped or stable>=required:break
            if stopped or stable>=required:break
    if task=="bucket" and stable<required and not stopped and not handle_handoff:
        corners=env._data.site_xpos[[env._model.site(f"bucket_ref_{i}").id for i in range(8)]]
        food=env._data.body("boxed_food_0").xpos
        inside=np.all(food>=corners.min(0)) and np.all(food<=corners.max(0))
        if inside:
            target=_pose7_to_matrix(obs["state"]["tcp_pose"])
            if not info.get("succeed",False):target[2,3]+=.12
            for _ in range(100):
                error=_pose_error6(obs["state"]["tcp_pose"],_matrix_to_pose7(target))
                stopped=step(np.r_[np.clip(error[:3]*.35,-.004,.004),error[3:]*.25,hand_targets[-1]])
                if stopped or stable>=required:break
    result=dict(success_policy=success_policy(task),success=bool(info.get("succeed",False)),stable_success_steps=stable,steps=len(rows),
        nail_depth=float(env.get_nail_depth()) if task=="hammer" else None,final_truth=snapshot(env),
        frame_mode=frame_mode,tip_phase=tip_phase,transition_start=phase.transition_start_step,
        transition_end=phase.transition_end_step)
    if handle_handoff:
        result.update(handle_handoff=True,handle_handoff_step=handoff_step,
                      handle_handoff_gate='payload_inside_bottom_supported_no_hand_contact_tcp_clear_5_steps')
        if handoff_step is not None:
            from .bucket_recovery import recover
            rotation=(frames[1]@_pose7_to_matrix(source[-1,:7]))[:3,:3]
            rows,result=recover(env,rows,result,grip_prior=hand_targets[-1],rotation_prior=rotation)
        else:result['handle_handoff_skipped']='payload support/clearance gate not reached'
    return rows,result
