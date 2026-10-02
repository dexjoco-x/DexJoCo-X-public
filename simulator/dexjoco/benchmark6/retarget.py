"""Offline geometric hand retargeting; never writes the task simulator's qpos."""
from __future__ import annotations
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .models import path_for
from .environments import TIP_SUFFIXES
from ..autocollect.linker_teacher import HandFK
from ..autocollect.replay import ReplayEpisode
from ..autocollect.retarget import _pose7_to_matrix, _matrix_to_pose7

POINTS = {
    "allegro": np.array([[0,0,.035]]+[[0,0,.019]]*3),
    "leap": np.array([[.004,-.066,-.014]]+[[-.0006,-.054,.0145]]*3),
    "xhand": np.zeros((5,3)),
    "inspire": np.array([[.018,.003,-.006]]+[[.003,.04,-.005]]*4),
    "wuji": np.array([[-.021,.001,.0002]]+[[-.0185,.00105,0]]*4),
    "shadow": np.array([[0,.0008,.025]]+[[0,.0004,.023]]*4),
    # Native URDF fingertip markers expressed in distal DP coordinates.
    "sharpa_wave": np.array([[.027611,0,0]]+[[.0260092574,0,0]]*4),
    "linkerhand_l20": np.array([[0,0,.025]]+[[-.006,0,.02]]*4),
}


def reference_points(hand,side='right'):
    """Body-local anatomical reference points, not task-specific object offsets."""
    if side not in ('right','left'):raise ValueError(side)
    points=POINTS[hand].copy()
    if side=='left' and hand=='inspire':points[:,1]*=-1
    # Genuine left Wuji CAD reverses distal local axes; right-hand coordinates
    # put the references 37--42 mm outside the intended left fingertips.
    if side=='left' and hand=='wuji':points*=-1
    return points


class TargetFK:
    """Analytic control-space FK, with explicit mimic/tendon Jacobian reduction.

    Shadow distal fixed tendons are split equally ONLY in this kinematic
    approximation. In the task rollout their native physical tendon is retained.
    """
    def __init__(self, hand, side="right"):
        path=path_for("hammer_nail" if side=="right" else "bimanual_photograph",hand)
        meta=json.loads(path.with_suffix(".json").read_text())["sides"][side]
        self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model)
        m=self.model
        self.jids=np.array([m.joint(n).id for n in meta["state_joint_names"]])
        self.qids=m.jnt_qposadr[self.jids];self.vids=m.jnt_dofadr[self.jids]
        self.low,self.high=np.array(meta["actuator_ranges"]).T
        if np.any(self.high<=self.low):raise ValueError("Invalid position-control bounds")
        self.home=np.array(meta["home"])
        self.B=np.zeros((len(self.jids),len(self.low)));self.offset=np.zeros(len(self.jids))
        indices={m.joint(j).name:i for i,j in enumerate(self.jids)}
        for col,terms in enumerate(meta["transmissions"]):
            total=sum(float(v) for _,v in terms)
            for name,coefficient in terms:self.B[indices[name],col]=1./total
        for _ in range(m.neq+1):
            for eq in range(m.neq):
                if m.eq_type[eq]!=mujoco.mjtEq.mjEQ_JOINT:continue
                child,parent=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq])
                if child not in self.jids or parent not in self.jids:continue
                ci=np.flatnonzero(self.jids==child)[0];pi=np.flatnonzero(self.jids==parent)[0]
                if np.any(m.eq_data[eq,2:5]):raise ValueError("Nonlinear mimic not implemented")
                self.B[ci]=m.eq_data[eq,1]*self.B[pi]
                self.offset[ci]=m.eq_data[eq,1]*self.offset[pi]+m.eq_data[eq,0]
        self.site=m.site("attachment_site" if side=="right" else "attachment_site_left").id
        suffixes=TIP_SUFFIXES[hand]
        if side=="left":
            if hand in ("allegro","inspire"):suffixes=tuple(s+"_left" for s in suffixes)
            elif hand in ("xhand","sharpa_wave"):suffixes=tuple(s.replace("right","left") for s in suffixes)
            elif hand=="shadow":suffixes=tuple(s.replace("shadow_rh_","shadow_left_lh_") for s in suffixes)
        self.bodies=[m.body(f"benchhand_{side}_native_"+s).id for s in suffixes]
        self.points=reference_points(hand,side)

    def evaluate(self,q,jacobian=False):
        m,d=self.model,self.data;d.qpos[self.qids]=self.B@q+self.offset
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        rot=d.site_xmat[self.site].reshape(3,3)
        world=np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@p for b,p in zip(self.bodies,self.points)])
        points=(world-d.site_xpos[self.site])@rot
        if not jacobian:return points
        jac=[]
        for body,point in zip(self.bodies,world):
            jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,point,body)
            jac.append(rot.T@jp[:,self.vids]@self.B)
        return points,np.concatenate(jac,0)


class GeometricRetargeter:
    def __init__(self,hand,offset=(0,0,0),squeeze=.0,side="right",source_hand="inspire"):
        self.source_hand=source_hand
        self.source=HandFK(source=True) if side=="right" and source_hand=="inspire" else TargetFK(source_hand,side)
        self.target=TargetFK(hand,side);self.hand=hand
        self.offset=np.array(offset);self.squeeze=squeeze
        self.alignment=np.eye(4)
        # Shadow includes a long physical forearm and a different palm axis.
        # Align by a rigid transform (NO scaling), applied to arm targets only.
        if hand=="shadow" or source_hand!="inspire":
            x=self.target.evaluate(self.target.home)
            y=self.source.evaluate(np.clip(np.zeros(len(self.source.low)),self.source.low,self.source.high))
            n=min(len(x),len(y));x=x[:n];y=y[:n]
            u,_,v=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
            correction=np.eye(3);correction[-1,-1]=np.linalg.det(v.T@u.T)
            rot=v.T@correction@u.T
            self.alignment[:3,:3]=rot;self.alignment[:3,3]=y.mean(0)-rot@x.mean(0)

    def closure(self,command):
        # Inspire has a dedicated thumb pair followed by positive flexion
        # controls. Other hands may flex toward NEGATIVE joint angles.
        if self.source_hand=='inspire':return float(np.mean(command[2:]))
        bounded=np.clip(command,self.source.low,self.source.high)
        return float(np.mean((bounded-self.source.low)/(self.source.high-self.source.low)))

    def sequence(self,commands):
        hand=self.target;q=np.clip(hand.home,hand.low,hand.high)
        result=[];errors=[];n=min(len(hand.bodies),len(self.source.evaluate(self.source.low)));rot=self.alignment[:3,:3]
        for source_command in commands:
            desired=self.source.evaluate(np.clip(source_command,self.source.low,self.source.high))[:n]-self.offset
            closure=self.closure(source_command)
            amount=self.squeeze*np.clip((closure-.15)/.3,0,1)
            direction=np.vstack([desired[1:].mean(0)-desired[0],desired[0]-desired[1:]])
            desired+=amount*direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
            desired=(desired-self.alignment[:3,3])@rot
            prior=q.copy()
            def fun(value):return np.r_[(hand.evaluate(value)[:n]-desired).ravel()*50,(value-prior)*.015]
            def jac(value):return np.vstack([hand.evaluate(value,True)[1][:3*n]*50,np.eye(len(value))*.015])
            fit=least_squares(fun,q,jac=jac,bounds=(hand.low,hand.high),max_nfev=45,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            q=fit.x;result.append(q.copy());errors.append(np.linalg.norm(hand.evaluate(q)[:n]-desired,axis=1))
        return np.array(result),np.array(errors)


def align_native_episode(episode,alignment):
    """Rigidly change the wrist reference; task-object coordinates stay intact."""
    states=episode.states.copy();actions=episode.actions.copy()
    for i in range(len(actions)):
        pose=_pose7_to_matrix(episode.states[i,:7])
        states[i,:7]=_matrix_to_pose7(pose@alignment)
        dr=Rotation.from_rotvec(episode.actions[i,3:6]).as_matrix()
        actions[i,:3]+=(dr-np.eye(3))@pose[:3,:3]@alignment[:3,3]
    return ReplayEpisode(episode.source,actions,states,episode.timestamps,
                         {"tcp_alignment":alignment.tolist()})


class AdaptiveRetargeter(GeometricRetargeter):
    """Fit fingers AND a slowly changing wrist frame; never scale task objects.

    This is a diagnostic kinematic proposal generator, not a grasp guarantee.
    Wrist regularizers keep the underdetermined tip fitting problem bounded.
    """
    def sequence(self,commands):
        hand=self.target;q=np.clip(hand.home,hand.low,hand.high)
        alignment=self.alignment.copy();result=[];errors=[];transforms=[]
        n=min(len(hand.bodies),len(self.source.evaluate(self.source.low)))
        for command in commands:
            desired=self.source.evaluate(np.clip(command,self.source.low,self.source.high))[:n]-self.offset
            amount=self.squeeze*np.clip((self.closure(command)-.15)/.3,0,1)
            direction=np.vstack([desired[1:].mean(0)-desired[0],desired[0]-desired[1:]])
            desired+=amount*direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
            prior=q.copy();base=alignment.copy()
            def predict(value):
                r=Rotation.from_rotvec(value[-3:]).as_matrix()@base[:3,:3]
                p=base[:3,3]+value[-6:-3]
                return hand.evaluate(value[:-6])[:n]@r.T+p
            def fun(value):
                return np.r_[(predict(value)-desired).ravel()*70,(value[:-6]-prior)*.04,
                             value[-6:-3]*12,value[-3:]*.18,
                             (base[:3,3]+value[-6:-3]-self.alignment[:3,3])*2]
            initial=np.r_[q,np.zeros(6)]
            fit=least_squares(fun,initial,bounds=(np.r_[hand.low,[-.008]*3,[-.08]*3],
                np.r_[hand.high,[.008]*3,[.08]*3]),max_nfev=35,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            q=fit.x[:-6];alignment[:3,3]+=fit.x[-6:-3]
            alignment[:3,:3]=Rotation.from_rotvec(fit.x[-3:]).as_matrix()@base[:3,:3]
            result.append(q.copy());errors.append(np.linalg.norm(predict(fit.x)-desired,axis=1))
            transforms.append(alignment.copy())
        self.transforms=np.array(transforms)
        return np.array(result),np.array(errors)


def align_adaptive_episode(episode,transforms):
    from ..autocollect.retarget import _pose_error6
    states=episode.states.copy();actions=episode.actions.copy()
    for i,(state,action) in enumerate(zip(episode.states,episode.actions)):
        pose=_pose7_to_matrix(state[:7]);desired=pose.copy()
        desired[:3,3]+=action[:3]
        desired[:3,:3]=Rotation.from_rotvec(action[3:6]).as_matrix()@pose[:3,:3]
        actual=pose@transforms[i]
        target=desired@transforms[min(i+1,len(transforms)-1)]
        states[i,:7]=_matrix_to_pose7(actual)
        actions[i,:6]=_pose_error6(_matrix_to_pose7(actual),_matrix_to_pose7(target))
    return ReplayEpisode(episode.source,actions,states,episode.timestamps,{"adaptive_wrist":True})
