"""Experimental privileged cross-hand teacher; no object attachment or VLA.

FK-only models used for hand retargeting never advance or modify task physics.
Task environments receive only bounded native-delta actions.
"""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..sim.envs.linkerhand_l20_specs import JOINT_NAMES
from ..sim.envs.panda_linkerhand_l20_env import PandaHammerNailLinkerL20Env, PandaPickBucketLinkerL20Env
from .retarget import _pose7_to_matrix, _matrix_to_pose7, _pose_error6
from .hammer_retarget import infer_hammer_phase_plan, _interpolate_rigid_transform

XML = Path(__file__).resolve().parents[1] / 'sim/envs/xmls'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


class HandFK:
    def __init__(self, source=False):
        self.model = mujoco.MjModel.from_xml_path(str(XML / ('panda_inspire_right.xml' if source else 'panda_linkerhand_l20_right.xml')))
        self.data = mujoco.MjData(self.model)
        if source:
            from dexjoco.sim.envs.model_profiles.inspire import INSPIRE_RIGHT_ACTIVE_JOINT_NAMES
            names = INSPIRE_RIGHT_ACTIVE_JOINT_NAMES
            self.bodies = [self.model.body('inspire_' + f + ('_distal' if f == 'thumb' else '_intermediate')).id for f in FINGERS]
            self.points = np.array([[.018, .003, -.006]] + [[.003, .04, -.005]]*4)
        else:
            names = JOINT_NAMES
            self.bodies = [self.model.body('l20_' + f + '_distal').id for f in FINGERS]
            self.points = np.array([[0, 0, .025]] + [[-.006, 0, .02]]*4)
        self.jids = np.array([self.model.joint(n).id for n in names])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        self.low, self.high = self.model.jnt_range[self.jids].T.copy()
        self.mimics = []
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                child, parent = self.model.eq_obj1id[i], self.model.eq_obj2id[i]
                if parent in self.jids:
                    factor, offset = self.model.eq_data[i, 1], self.model.eq_data[i, 0]
                    self.mimics.append((child, parent, factor, offset))
                    ix = list(self.jids).index(parent)
                    if factor > 0:
                        self.high[ix] = min(self.high[ix], (self.model.jnt_range[child, 1]-offset)/factor)
        self.site = self.model.site('attachment_site').id

    def evaluate(self, q, jacobian=False):
        m, d = self.model, self.data
        d.qpos[self.qids] = q
        for child, parent, factor, offset in self.mimics:
            d.qpos[m.jnt_qposadr[child]] = d.qpos[m.jnt_qposadr[parent]] * factor + offset
        mujoco.mj_kinematics(m,d)
        mujoco.mj_comPos(m,d)
        rot = d.site_xmat[self.site].reshape(3,3)
        world = np.array([d.xpos[b] + d.xmat[b].reshape(3,3) @ p for b,p in zip(self.bodies,self.points)])
        local = (world-d.site_xpos[self.site]) @ rot
        if not jacobian:
            return local
        jacs=[]
        for b,point in zip(self.bodies,world):
            jp=np.zeros((3,m.nv)); mujoco.mj_jac(m,d,jp,None,point,b)
            reduced=jp[:,self.vids].copy()
            for child,parent,factor,_ in self.mimics:
                reduced[:,list(self.jids).index(parent)] += factor * jp[:,m.jnt_dofadr[child]]
            jacs.append(rot.T @ reduced)
        return local, np.concatenate(jacs,axis=0)


class FingerRetargeter:
    def __init__(self, tcp_offset=(0,0,0), squeeze_m=0., *, target_hand=None, initial_pose=None):
        self.source, self.target = HandFK(True), HandFK(False) if target_hand is None else target_hand
        self.offset = np.asarray(tcp_offset,float)
        self.squeeze_m = float(squeeze_m)
        self.initial_pose = np.array([.7,.6,.3,.3]+[0,.3,.3]*4) if initial_pose is None else np.asarray(initial_pose,float)
        if self.initial_pose.shape != self.target.low.shape:
            raise ValueError('Initial hand pose does not match the target hand dimension.')

    def sequence(self, source_actions):
        hand = self.target
        previous = np.clip(self.initial_pose, hand.low,hand.high)
        result, errors = [], []
        for command in source_actions:
            desired = self.source.evaluate(np.clip(command,self.source.low,self.source.high)) - self.offset
            # Retarget fingertip centers, compensating for different pad shapes.
            # Apply only during source closure; open/release poses are unchanged.
            amount=self.squeeze_m*np.clip((np.mean(command[2:])-.15)/.3,0,1)
            directions=np.vstack([desired[1:].mean(0)-desired[0], desired[0]-desired[1:]])
            desired += amount*directions/np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-8)
            def fun(q):
                return np.r_[(hand.evaluate(q)-desired).ravel()*50, (q-previous)*.015]
            def jac(q):
                return np.vstack([hand.evaluate(q,True)[1]*50, np.eye(len(previous))*.015])
            fit=least_squares(fun,previous,jac=jac,bounds=(hand.low,hand.high),max_nfev=35,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            previous=fit.x
            result.append(fit.x.copy());errors.append(np.linalg.norm(hand.evaluate(fit.x)-desired,axis=1))
        return np.array(result),np.array(errors)


def make_env(task, seed, randomize=True, hand_type='linkerhand_l20_right'):
    if task not in ('hammer','bucket'):
        raise ValueError(f'Unsupported collection task: {task}')
    if hand_type == 'linkerhand_l20_right':
        cls = PandaHammerNailLinkerL20Env if task == 'hammer' else PandaPickBucketLinkerL20Env
    elif hand_type == 'xhand_right':
        from ..sim.envs.panda_xhand_env import PandaHammerNailXHandEnv, PandaPickBucketXHandEnv
        cls = PandaHammerNailXHandEnv if task == 'hammer' else PandaPickBucketXHandEnv
    else:
        raise ValueError(f'Unsupported collection hand: {hand_type}')
    env=cls(seed=seed,randomize=randomize,max_episode_steps=2000)
    env.hz=1000000  # wall-clock throttle only; simulation stays at 20 Hz
    env._prime_rgb_array_renderer=lambda:None
    env.render=lambda:[np.zeros((1,1,3),np.uint8)]*4
    obs,_=env.reset(seed=seed)
    return env,obs


def body_pose(env,name):
    b=env._data.body(name)
    return np.r_[b.xpos,b.xquat].copy()


def snapshot(env):
    d=env._data
    return {k:np.array(getattr(d,k)).copy() for k in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','time')}


def rollout(env, obs, episode, hand_targets, task, tcp_offset=(0,0,0), frame_capture=None):
    """Transfer demonstrated arm motion and geometrically retargeted fingers."""
    source=episode.states
    if task == 'hammer':
        frames=[_pose7_to_matrix(body_pose(env,n)) @ np.linalg.inv(_pose7_to_matrix(source[0,s]))
                for n,s in [('hammer_body',slice(13,20)),('nail',slice(20,27))]]
        phase=infer_hammer_phase_plan(episode)
    else:
        frames=[_pose7_to_matrix(body_pose(env,n)) @ np.linalg.inv(_pose7_to_matrix(source[0,s]))
                for n,s in [('boxed_food_0',slice(20,27)),('bucket',slice(13,20))]]
        # First payload pickup then deposit/basket acquisition. Locate the
        # closest TCP-to-payload approach before blending toward the basket.
        distances=np.linalg.norm(source[:int(len(source)*.65),:3]-source[0,20:23],axis=1)
        begin=int(np.argmin(distances))+10
        phase=type('Phase',(),{'transition_start_step':begin,'transition_end_step':begin+35})()
    rows=[];success_count=0
    info = {}
    def step(action):
        nonlocal obs, info, success_count
        action=np.clip(action,env.action_space.low,env.action_space.high)
        before=snapshot(env)
        state=np.r_[obs['state']['tcp_pose'],obs['state']['gripper_pose']]
        if frame_capture is not None: frame_capture(env,len(rows))
        obs,reward,terminated,truncated,info=env.step(action)
        success=bool(info.get('succeed',False))
        success_count=success_count+1 if success else 0
        rows.append(dict(state=state,action=action,truth=before,info=info))
        if not np.isfinite(env._data.qpos).all():raise RuntimeError('nonfinite simulation')
        return terminated and not success
    for i,source_action in enumerate(episode.actions):
        alpha=np.clip((i-phase.transition_start_step)/max(1,phase.transition_end_step-phase.transition_start_step-1),0,1)
        transform=_interpolate_rigid_transform(*frames,float(alpha))
        desired=transform @ _pose7_to_matrix(source[i,:7])
        desired[:3,3] += desired[:3,:3] @ np.asarray(tcp_offset)
        error=_pose_error6(obs['state']['tcp_pose'],_matrix_to_pose7(desired))
        delta=np.r_[transform[:3,:3]@source_action[:3],transform[:3,:3]@source_action[3:6]] + error*np.array([.35]*3+[.25]*3)
        action=np.clip(np.r_[delta,hand_targets[i]],env.action_space.low,env.action_space.high)
        stopped=step(action)
        if success_count>=10:break
        if stopped:break
    if task == 'hammer' and success_count < 10:
        # Close the loop on the actual tool face after transferring the grasp.
        # The source grasp-to-tool transform differs between hand morphologies.
        hold_rotation=_pose7_to_matrix(obs['state']['tcp_pose'])[:3,:3]
        grip=hand_targets[-1]
        gid=env._model.geom('face').id
        for cycle in range(5):
            for phase_name, count, height in [('align',50,.085),('strike',30,-.008)]:
                for _ in range(count):
                    face=env._data.geom_xpos[gid]
                    axis=env._data.geom_xmat[gid].reshape(3,3)[:,2]
                    radius,half=env._model.geom_size[gid,:2]
                    extent=radius*np.sqrt(max(0,1-axis[2]**2))+half*abs(axis[2])
                    nail=env._data.geom('nail_head').xpos.copy()
                    desired_face=nail+np.array([0,0,extent+.0048+height])
                    delta=(desired_face-face)*.35
                    delta=np.clip(delta,-.005,.005)
                    actual_rotation=_pose7_to_matrix(obs['state']['tcp_pose'])[:3,:3]
                    dr=Rotation.from_matrix(hold_rotation@actual_rotation.T).as_rotvec()*.25
                    step(np.r_[delta,dr,grip])
                    if success_count>=10:break
                if success_count>=10:break
            if success_count>=10:break
    if task == 'bucket' and success_count < 10:
        corners=env._data.site_xpos[[env._model.site(f'bucket_ref_{i}').id for i in range(8)]]
        food=env._data.body('boxed_food_0').xpos
        inside=np.all(food>=corners.min(0)) and np.all(food<=corners.max(0))
        if inside:
            target=_pose7_to_matrix(obs['state']['tcp_pose'])
            # Finish the demonstrated lift if needed, then keep the grasp for
            # at least half a second under the unchanged task success check.
            if not info.get('succeed',False):target[2,3]+=.12
            for _ in range(100):
                error=_pose_error6(obs['state']['tcp_pose'],_matrix_to_pose7(target))
                command=np.r_[np.clip(error[:3]*.35,-.004,.004),error[3:]*.25,hand_targets[-1]]
                step(command)
                if success_count>=10:break
    return rows,dict(success=bool(info.get('succeed',False)),stable_success_steps=success_count,
                     steps=len(rows),nail_depth=float(env.get_nail_depth()) if task=='hammer' else None,
                     final_truth=snapshot(env))
