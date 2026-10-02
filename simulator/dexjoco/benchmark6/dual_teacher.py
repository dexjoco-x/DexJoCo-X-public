"""Bounded bimanual source-guided controller with shared task-object frames."""
import numpy as np
from scipy.spatial.transform import Rotation
from .sources import RESET_ATTRS, NH
from .environments import TIP_SUFFIXES
from .retarget import POINTS,GeometricRetargeter,AdaptiveRetargeter
from ..autocollect.retarget import _pose7_to_matrix, _matrix_to_pose7, _pose_error6
from ..autocollect.hammer_retarget import _interpolate_rigid_transform
from ..autocollect.linker_teacher import snapshot


# Screen-local +Y points up; local +Z faces the common robot side. Aligning
# only the normal leaves an uncontrolled in-plane roll (e.g. a sideways iPad).
IPAD_SCREEN_ROTATION=np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])


def _ipad_prepared(env):
    if not env._ipad_gate.ready:return False
    actual=env._data.geom_xmat[env._model.geom('ipad_screen').id].reshape(3,3)
    return Rotation.from_matrix(IPAD_SCREEN_ROTATION@actual.T).magnitude()<np.deg2rad(10.)


def _index_position(env):
    body=env._data.body("benchhand_right_native_"+TIP_SUFFIXES[env.hand][1])
    return body.xpos+body.xmat.reshape(3,3)@POINTS[env.hand][1]


def _pointing_grip(env,previous):
    """Embodiment-specific joint posture, independent of task/object size."""
    if env.hand=="leap":
        value=np.array([0,0,0,0, 0,1.3,1.5,1.3, 0,1.3,1.5,1.3, 1.1,1.,.8,.8])
    elif env.hand=="allegro":
        value=np.array([0,0,0,0, 0,1.3,1.5,1.3, 0,1.3,1.5,1.3, 1.1,1.,.8,.8])
    elif env.hand=="inspire":value=np.array([.8,.4,0,1.2,1.2,1.2])
    else:
        value=GeometricRetargeter(env.hand).sequence(np.array([[.8,.4,0,1.2,1.2,1.2]]))[0][0]
    bounds=env._model.actuator_ctrlrange[env._native_ctrl_ids["right"]]
    return np.clip(value,bounds[:,0],bounds[:,1])


def _index_axis(env):
    body=env._data.body("benchhand_right_native_"+TIP_SUFFIXES[env.hand][1])
    local={"leap":[0,-1,0],"allegro":[0,0,1],"inspire":[0,1,0],
           "wuji":[-1,0,0],"shadow":[0,0,1],"xhand":[0,0,-1],"sharpa_wave":[1,0,0],
           "linkerhand_l20":[0,0,1]}[env.hand]
    return body.xmat.reshape(3,3)@local


def _point_tcp(env,target,normal,rotation_step=.10):
    axis=_index_axis(env);cross=np.cross(axis,-normal)
    angle=np.arctan2(np.linalg.norm(cross),np.dot(axis,-normal))
    rot=Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-9)*min(angle,rotation_step)).as_matrix()
    tcp=env.tcp("right");result=tcp.copy()
    result[:3,:3]=rot@tcp[:3,:3]
    result[:3,3]=target+rot@(tcp[:3,3]-_index_position(env))
    return result


def _scalar_info(info):
    return {k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()
            if isinstance(v,(bool,str,int,float,np.generic))}


def rollout(env,demo,*,gain=.5,repeat=1,feedback=True,pose_source="action",camera_grasp=False,
            translation_cap=None,rotation_cap=None,adaptive_wrist=False,thin_camera_grasp=False,
            grasp_squeeze=0.,ipad_hold_offset=(0.,0.,0.)):
    # Teacher speed limits may be stricter than the common action space. This
    # never changes the scene, robot dynamics or task success rule.
    tcap=env._translation_bound if translation_cap is None else float(translation_cap)
    rcap=env._rotation_bound if rotation_cap is None else float(rotation_cap)
    if not 0<tcap<=env._translation_bound or not 0<rcap<=env._rotation_bound:
        raise ValueError('Teacher caps must fit the common action space')
    if not 0<=grasp_squeeze<=.03:raise ValueError('Grasp squeeze must be 0..3 cm')
    ipad_hold_offset=np.asarray(ipad_hold_offset,dtype=float)
    if ipad_hold_offset.shape!=(3,) or not np.isfinite(ipad_hold_offset).all() or np.linalg.norm(ipad_hold_offset)>.25:
        raise ValueError('Held-tablet workspace offset must be a finite vector of at most 25 cm')
    if np.any(ipad_hold_offset) and env.task!='bimanual_unlock_ipad':
        raise ValueError('Held-tablet workspace offset is only defined for iPad')
    if demo.hand!=env.hand and demo.hand not in ("inspire","wuji","allegro","leap","xhand","sharpa_wave","linkerhand_l20"):
        raise NotImplementedError("Source transmission conversion is not implemented")
    retargeted={};alignments={};offsets={};dynamic_alignments={}
    for si,side in enumerate(("right","left")):
        alignments[side]=np.eye(4);offsets[side]=np.zeros(3)
        if demo.hand!=env.hand or grasp_squeeze>0:
            offset=(0,0,-.008) if env.hand=="xhand" else (0,0,0)
            teacher=(AdaptiveRetargeter if adaptive_wrist else GeometricRetargeter)(env.hand,offset=offset,
                squeeze=grasp_squeeze if grasp_squeeze>0 else .015,side=side,source_hand=demo.hand)
            commands=demo.actions[:,si*(7+NH[demo.hand])+7:(si+1)*(7+NH[demo.hand])]
            retargeted[side]=teacher.sequence(commands)[0]
            alignments[side]=teacher.alignment;offsets[side]=np.array(offset)
            if adaptive_wrist:dynamic_alignments[side]=teacher.transforms
    nh=NH[demo.hand];source_objects=demo.objects
    target_objects=np.array([getattr(env,name) for name in RESET_ATTRS[env.task]])
    if env.task=="bimanual_photograph":
        # The stored logo pose is LOCAL to the table. Convert both to world
        # before transforming world-frame robot trajectories.
        table=env._model.body("table").pos.copy()
        target_objects[1,:3]+=table
        source_objects=source_objects.copy()
        source_table=table.copy();source_table[2]+=demo.states[0,-1]-env.delta_h
        source_objects[1,:3]+=source_table
    frames=[_pose7_to_matrix(t)@np.linalg.inv(_pose7_to_matrix(s)) for s,t in zip(source_objects,target_objects)]
    if env.task=="bimanual_hanoi":
        # Transfer in one fixed base frame, without assuming missing source
        # disk poses. Success must be produced and checked in fresh physics.
        frames=[frames[0],frames[0]]
    source_positions=demo.states[:int(len(demo.states)*.8),:3]
    nearest_index=int(np.argmin(np.linalg.norm(source_positions-source_objects[0,:3],axis=1)))
    begin=nearest_index+10;end=begin+35
    rows=[];success_count=0;info={};obs=env._compute_observation();stopped=False
    ipad = env.task == 'bimanual_unlock_ipad'
    parked_right = env.tcp('right').copy() if ipad else None
    parked_grip = env.hold_action()['right'][6:].copy() if ipad else None
    if ipad:
        # Keep the inactive hand outside the lifted tablet's swept volume.
        # Move the arm through the ordinary controller, never change its reset.
        parked_right[1,3]-=.20
    best_left = None
    def step(action):
        nonlocal obs,info,success_count,best_left
        before=snapshot(env)
        public_state=np.r_[obs["state"]["tcp_pose"],obs["state"]["gripper_pose"]]
        obs,_,terminated,truncated,info=env.step(action)
        success=bool(info.get("succeed",False));success_count=success_count+1 if success else 0
        rows.append(dict(state=public_state,action=np.r_[action["right"],action["left"]],truth=before,info=info))
        if ipad:
            evidence=env._ipad_measurements.last
            score=(int(evidence['grasp_opposed']),evidence['left_contact_branches'],evidence['left_contact_points'])
            if score[-1] and (best_left is None or score>best_left[0]):
                best_left=(score,env.tcp('left').copy(),np.asarray(action['left'][6:]).copy())
        return (terminated or truncated) and not success
    def drive(desired,grips,translation_gain=gain):
        action={}
        for side in ("right","left"):
            error=_pose_error6(_matrix_to_pose7(env.tcp(side)),_matrix_to_pose7(desired[side]))
            action[side]=np.r_[np.clip(error[:3]*translation_gain,-tcap,tcap),
                              np.clip(error[3:]*gain,-rcap,rcap),grips[side]]
        return step(action)
    grasp_metrics={}
    if camera_grasp or thin_camera_grasp:
        if env.task not in ('bimanual_photograph','bimanual_unlock_ipad'):raise ValueError('Unsupported object grasp task')
        from .camera_grasp import propose
        pose,opened,closed,grasp_metrics=propose(env,demo.source,thin=thin_camera_grasp)
        desired={s:env.tcp(s) for s in ('right','left')}
        grips={s:env.hold_action()[s][6:] for s in ('right','left')}
        if ipad:desired['right']=parked_right.copy()
        for phase,count,height in (('clearance',70,.20),('approach',100,.20),('descend',100,0),('close',80,0),('lift',120,.18)):
            target=pose.copy();target[2,3]+=height
            if phase=='clearance':target=desired['left'].copy();target[2,3]=max(target[2,3],pose[2,3]+.20)
            for i in range(count):
                desired['left']=target
                alpha=(i+1)/count if phase=='close' else 1. if phase=='lift' else 0.
                grips['left']=(1-alpha)*opened+alpha*closed
                stopped=drive(desired,grips,translation_gain=.5)
                if stopped or success_count>=10:break
            if stopped or success_count>=10:break
    for i,source_action in enumerate([] if camera_grasp or thin_camera_grasp else demo.actions):
        alpha=float(np.clip((i-begin)/max(1,end-begin),0,1))
        blend=_interpolate_rigid_transform(frames[0],frames[1],alpha)
        desired={};grips={}
        for si,side in enumerate(("right","left")):
            segment=source_action[si*(7+nh):(si+1)*(7+nh)]
            frame=blend
            if env.task=="bimanual_microwave_cook" and side=="left":frame=frames[1]
            if env.task=="bimanual_unlock_ipad":frame=frames[1]
            pose=demo.states[i,si*7:(si+1)*7] if pose_source=="state" else segment[:7]
            alignment=dynamic_alignments[side][i] if side in dynamic_alignments else alignments[side]
            desired[side]=frame@_pose7_to_matrix(pose)@alignment
            if side not in dynamic_alignments:desired[side][:3,3]+=desired[side][:3,:3]@offsets[side]
            limits=env._model.actuator_ctrlrange[env._native_ctrl_ids[side]]
            commands=retargeted[side][i] if side in retargeted else segment[7:]
            grips[side]=np.clip(commands,limits[:,0],limits[:,1])
        if ipad:
            # Source motion proposes ONLY acquisition with the left hand.
            # Never execute a source's premature right-hand taps on the stand.
            desired['right']=parked_right.copy();grips['right']=parked_grip.copy()
        for _ in range(repeat):
            stopped=drive(desired,grips)
            if stopped or success_count>=10:break
        if ipad and _ipad_prepared(env):break
        if stopped or success_count>=10:break
    if ipad and not stopped and not _ipad_prepared(env) and best_left is not None:
        from .ipad_task import RULES
        evidence=env._ipad_measurements.last
        has_grasp=bool(evidence['grasp_opposed'])
        already_held=(evidence['grasp_opposed'] and evidence['lift_m']>=RULES.lift_m
                      and evidence['support_clearance_m']>=RULES.support_clearance_m)
        # Recover the best physically observed contact posture using normal
        # actuator commands ONLY if acquisition failed. A contact-rich posture
        # near the stand must not replace an already lifted, securely held pose.
        # This is NOT restoring a simulator snapshot.
        if not has_grasp:
            desired['left']=best_left[1].copy();grips['left']=best_left[2].copy()
            for _ in range(60):
                stopped=drive(desired,grips)
                if stopped or _ipad_prepared(env):break
        if not already_held and not stopped and not _ipad_prepared(env):
            desired['left']=env.tcp('left').copy();desired['left'][2,3]+=.28
            for _ in range(120):
                stopped=drive(desired,grips)
                if stopped or _ipad_prepared(env):break
        if not stopped and not _ipad_prepared(env):
            # Reorient the actual held tablet toward the common robot side.
            # Rotation is around its observed centre, applied via LEFT TCP.
            gid=env._model.geom('ipad_screen').id
            held_center=env._data.geom_xpos[gid].copy()
            held_center[2]=max(held_center[2],env._ipad_measurements.initial_screen_z+.32)
            # Search a reachable holding location via ordinary arm control.
            # This moves neither reset objects nor geometry/contact parameters.
            held_center+=ipad_hold_offset
            for _ in range(240):
                metric=env._ipad_measurements.last
                if metric['grasp_opposed'] and metric['lift_m']>=RULES.lift_m:
                    center=env._data.geom_xpos[gid].copy()
                    actual=env._data.geom_xmat[gid].reshape(3,3)
                    error=Rotation.from_matrix(IPAD_SCREEN_ROTATION@actual.T).as_rotvec()
                    rotation=Rotation.from_rotvec(error*min(1.,.25/max(np.linalg.norm(error),1e-9))).as_matrix()
                    left=env.tcp('left');desired['left']=left.copy()
                    desired['left'][:3,:3]=rotation@left[:3,:3]
                    desired['left'][:3,3]=held_center+rotation@(left[:3,3]-center)
                stopped=drive(desired,grips)
                if stopped or _ipad_prepared(env):break
    if feedback and success_count<10 and not stopped and env.task=="bimanual_microwave_cook":
        if env.hot_dog_inside_microwave(margin=.01) and abs(env._data.joint("microjoint").qpos[0])<.01:
            grips["right"]=_pointing_grip(env,grips["right"])
            gid=env._model.geom("start_button").id
            # The keypad front is the microwave's local -Y face.
            for i in range(240):
                normal=env._data.geom_xmat[gid].reshape(3,3)@np.array([0.,-1.,0.])
                clearance=.065 if i<60 else -.002
                target=env._data.geom_xpos[gid]+normal*(env._model.geom_size[gid,1]+clearance)
                desired["right"]=_point_tcp(env,target,normal)
                stopped=drive(desired,grips,translation_gain=.5)
                if stopped or success_count>=10:break
    if feedback and success_count<10 and not stopped and env.task=="bimanual_photograph":
        # Correct the held camera through the LEFT hand, then servo the RIGHT
        # index to its real shutter. No object-state writes or relaxed criteria.
        d=env._data;grips["right"]=_pointing_grip(env,grips["right"])
        for i in range(400):
            axis=d.geom_xmat[env._camera_axis_z_geom_id].reshape(3,3)[:,2]
            target_axis=np.array([1.,0.,0.]);cross=np.cross(axis,target_axis)
            angle=np.arctan2(np.linalg.norm(cross),np.dot(axis,target_axis))
            rot=Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-9)*min(angle,.12)).as_matrix()
            view=d.site_xpos[env._view_site_id].copy()
            target_view=d.geom_xpos[env._target_region_geom_id].copy()
            left=env.tcp("left");desired["left"]=left.copy()
            desired["left"][:3,:3]=rot@left[:3,:3]
            desired["left"][:3,3]=view+np.clip(target_view-view,-.015,.015)+rot@(left[:3,3]-view)
            shutter=d.geom_xpos[env._shutter_geom_id].copy()
            normal=d.geom_xmat[env._shutter_geom_id].reshape(3,3)[:,2]
            # A cylinder's +Z axis is not necessarily its outward face. The
            # camera asset's cap is rotated so +Z points INTO the camera.
            if np.dot(normal,shutter-d.geom_xpos[env._model.geom("camera_body_collision").id])<0:
                normal=-normal
            # Approach before pressing; rebuild targets from current geometry.
            clearance=.055 if i<100 else .02 if i<150 else -.004
            desired["right"]=_point_tcp(env,shutter+clearance*normal,normal)
            stopped=drive(desired,grips,translation_gain=.25)
            if stopped or success_count>=10:break
    if feedback and success_count<10 and not stopped and ipad and _ipad_prepared(env) and not env._unlock_failed:
        # Touch 1,2,3 in order with explicit releases. The environment, not this
        # controller, determines which physical contacts count as button taps.
        d=env._data;last_index=env._unlock_index;release=35
        # Preserve the final holding SETPOINT. Replacing it by the measured
        # TCP discards the load-compensating tracking error and can lose a grasp.
        grips["right"]=_pointing_grip(env,grips["right"])
        aligned_steps=0;pressing=False
        ipad_body=d.body('ipad_body')
        held_object=_pose7_to_matrix(np.r_[ipad_body.xpos,ipad_body.xquat])
        holding_tcp=desired['left'].copy()
        for i in range(600):
            # Hold the OBJECT steady against tapping forces through normal
            # left-arm commands. A fixed wrist alone does not reject slip.
            actual_object=_pose7_to_matrix(np.r_[ipad_body.xpos,ipad_body.xquat])
            desired['left']=held_object@np.linalg.inv(actual_object)@holding_tcp
            current=env._unlock_index
            if current!=last_index:
                release=50;last_index=current;aligned_steps=0;pressing=False
            digit=(1,2,3)[min(current,2)];gid=env._model.geom(f"btn{digit}_cyl").id
            normal=d.geom_xmat[gid].reshape(3,3)[:,2]
            difference=_index_position(env)-d.geom_xpos[gid]
            gap=float(difference@normal)
            lateral=float(np.linalg.norm(difference-gap*normal))
            aligned=(lateral<.012 and .07<gap<.16 and float(_index_axis(env)@(-normal))>np.cos(.20))
            aligned_steps=aligned_steps+1 if aligned else 0
            if release<=0 and aligned_steps>=5:pressing=True
            releasing=release>0 or env._screen_unlocked or env._button_release_required
            # First orient and align in free space, THEN descend to the digit.
            # A fixed elapsed counter alone cannot establish finger alignment.
            target=d.geom_xpos[gid]+normal*(.12 if releasing or not pressing else -.003)
            desired["right"]=_point_tcp(env,target,normal)
            if release>0:release-=1
            stopped=drive(desired,grips,translation_gain=.2)
            if stopped or success_count>=10 or env._unlock_failed:break
    if success_count<10 and not stopped:
        # Hold the last targets, not the last velocity command.
        for _ in range(100):
            if drive(desired,grips) or success_count>=10:break
    metrics={}
    if env.task=="bimanual_hanoi":
        metrics=dict(env._compute_success_metrics()[1])
    if env.task=="bimanual_microwave_cook":
        metrics=dict(door_angle=float(env._data.joint("microjoint").qpos[0]),
                     food_inside=bool(env.hot_dog_inside_microwave(margin=.01)),
                     start_contact=bool(env._geom_in_contact("start_button")))
    if ipad:metrics=env.ipad_task_metrics()
    return rows,dict(success=bool(info.get("succeed",False)),stable_success_steps=success_count,
                     steps=len(rows),nail_depth=None,final_truth=snapshot(env),
                     final_info=_scalar_info(info),task_metrics=metrics,grasp_metrics=grasp_metrics)
