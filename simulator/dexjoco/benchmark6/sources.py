"""Read-only source demonstration inventory; old success labels are not acceptance."""
from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import zarr
from scipy.spatial.transform import Rotation
from ..autocollect.replay import ReplayEpisode
from ..autocollect.retarget import _pose_error6

ROOT=Path(__file__).resolve().parents[2]
NH={"allegro":16,"leap":16,"inspire":6,"wuji":20,"xhand":12,"shadow":24,"sharpa_wave":22,"linkerhand_l20":16}
RESET_ATTRS={"hammer_nail":("hammer_ori_pose","nail_ori_pose"),
             "pick_bucket":("bucket_ori_pose","box_food_ori_pose"),
             "water_plant":("_spray_ori_pose","_plant_ori_pose"),
             "click_mouse":("mouse_ori_pose","mousepad_ori_pose"),
             "bimanual_microwave_cook":("_hot_dog_ori_pose","_microwave_ori_pose"),
             "bimanual_photograph":("camera_ori_pose","logo_ori_pose"),
             "bimanual_unlock_ipad":("_stand_ori_pose","_ipad_ori_pose")}
RESET_ATTRS["bimanual_hanoi"] = ("hanoi_base_ori_pose",)


def leading_idle_frames(poses, fingers, command_poses=None, reset_fields=None):
    """Only skip a stationary prefix; never skip approach, pauses, or grasp stages.

    This chooses SOURCE samples for a new physical rollout, not saved simulator
    states or a video-only cut. Three pre-motion samples preserve the transition.
    """
    poses=np.asarray(poses,float);fingers=np.asarray(fingers,float)
    if poses.ndim==2:poses=poses[:,None,:]
    if poses.ndim!=3 or poses.shape[-1]!=7 or len(poses)!=len(fingers) or not len(poses):
        raise ValueError('Unaligned source poses/fingers')
    if not np.isfinite(poses).all() or not np.isfinite(fingers).all():raise ValueError('Nonfinite source motion')
    def pose_active(values):
        values=np.asarray(values,float)
        if values.shape!=poses.shape or not np.isfinite(values).all():raise ValueError('Invalid command poses')
        q=values[:,:,3:];norm=np.linalg.norm(q,axis=2)
        if np.any(norm<1e-9):raise ValueError('Invalid source quaternion')
        q=q/norm[:,:,None]
        angle=2*np.arccos(np.clip(np.abs(np.sum(q*q[0],axis=2)),0,1))
        return np.any(np.linalg.norm(values[:,:,:3]-values[0,:,:3],axis=2)>.002,axis=1)|np.any(angle>.01,axis=1)
    active=pose_active(poses)|np.any(np.abs(fingers-fingers[0])>.02,axis=1)
    if command_poses is not None:active|=pose_active(command_poses)
    if reset_fields is not None:
        fields=np.asarray(reset_fields,float)
        if len(fields)!=len(poses) or not np.isfinite(fields).all():raise ValueError('Invalid reset fields')
        active|=np.any(np.abs(fields-fields[0])>1e-8,axis=1)
    indexes=np.flatnonzero(active)
    first=int(indexes[0]) if len(indexes) else None
    skipped=max(0,first-3) if first is not None and first>=10 else 0
    # Preserve a short genuine initialization segment. Some native hands move
    # toward a constant initial command before a much longer operator wait.
    warmup=min(20,skipped)
    return skipped,dict(policy='source_stationary_prefix_v1',removed_source_frames=skipped-warmup,
        original_source_frames=len(poses),first_motion_source_frame=first,kept_pre_motion_frames=3,
        kept_initial_settling_frames=warmup,
        position_threshold_m=.002,rotation_threshold_rad=.01,finger_threshold_rad=.02,
        meaning='source-sample trim before fresh reset+bounded control; not video-only cutting or simulator state restoration',
        source_fps_not_assumed=True)


def trim_demo_idle(demo):
    """Preserve source identity and reset metadata while trimming raw templates."""
    n=len(demo.states);arms=demo.arms;nh=NH[demo.hand]
    poses=demo.states[:,:7*arms].reshape(n,arms,7)
    targets=demo.actions.reshape(n,arms,7+nh)
    # Command changes mark deliberate finger motion; measured initial finger
    # settling must not disguise a subsequent 10-second stationary wait.
    fingers=targets[:,:,7:].reshape(n,-1)
    start,record=leading_idle_frames(poses,fingers,targets[:,:,:7],demo.states[:,arms*(7+nh):])
    warmup=record['kept_initial_settling_frames'];record['removed_source_frames']=start-warmup
    record['resume_source_frame']=start
    if start==warmup:return demo,record
    indices=np.r_[np.arange(warmup),np.arange(start,n)]
    return replace(demo,states=demo.states[indices].copy(),actions=demo.actions[indices].copy()),record


def trim_episode_idle(episode):
    """Single-arm native-delta template variant; original recordings stay intact."""
    start,record=leading_idle_frames(episode.states[:,:7],
        episode.actions[:,6:],reset_fields=episode.states[:,13:])
    warmup=record['kept_initial_settling_frames'];record['removed_source_frames']=start-warmup
    record['resume_source_frame']=start
    if start==warmup:return episode,record
    indices=np.r_[np.arange(warmup),np.arange(start,len(episode.states))]
    # This template's timestamps cannot establish historical physics timing.
    # Fresh generated data uses the actual common simulator control clock.
    return ReplayEpisode(episode.source,episode.actions[indices].copy(),episode.states[indices].copy(),None,
                         dict(episode.metadata,source_preprocessing=record)),record


@dataclass
class Demonstration:
    source:Path
    task:str
    hand:str
    actions:np.ndarray
    states:np.ndarray
    derived_fields:tuple=()

    @property
    def arms(self):return 2 if self.task.startswith("bimanual") else 1

    def retarget_finger_samples(self, mode="command"):
        """Choose geometry evidence, not copied target-hand actuator commands.

        Measured joints include source contact/servo lag; commands describe the
        controller setpoint, which can lie beyond an object surface. Both are
        proposals for a fresh physical rollout, never simulator-state writes.
        """
        if self.arms != 1:
            raise ValueError("Finger sample selection currently supports single-arm sources only")
        if mode not in ("command", "measured"):
            raise ValueError(mode)
        values = self.actions[:, 7:] if mode == "command" else self.states[:, 7:7+NH[self.hand]]
        if values.shape != (len(self.actions), NH[self.hand]) or not np.isfinite(values).all():
            raise ValueError("Invalid source finger samples")
        return values.copy()

    @property
    def objects(self):
        offset=self.arms*(7+NH[self.hand])
        if self.task=="bimanual_hanoi":
            # Legacy Hanoi records only fixed-base XYZ, not disk poses or
            # starting tower topology. Never invent the missing disk state.
            return np.r_[self.states[0,offset:offset+3],1.,0.,0.,0.].reshape(1,7)
        return self.states[0,offset:offset+14].reshape(2,7)

    def legacy_single_episode(self,delta_mode="target_error"):
        if self.arms!=1:raise ValueError("Not a single-arm source")
        # Compatibility with two-object reference transfer (only TCP and object
        # fields are used). Native finger targets are passed separately.
        states=np.c_[self.states[:,:7],np.zeros((len(self.states),6)),self.states[:,7+NH[self.hand]:]]
        if delta_mode=="target_error":
            deltas=np.array([_pose_error6(s[:7],a[:7]) for s,a in zip(self.states,self.actions)])
        elif delta_mode=="measured":
            # An old absolute setpoint error is NOT a measured step velocity.
            # This mode tracks observed motion with separate bounded feedback.
            deltas=np.array([_pose_error6(s[:7],self.states[min(i+1,len(self.states)-1),:7])
                             for i,s in enumerate(self.states)])
        else:raise ValueError(delta_mode)
        actions=np.c_[deltas,np.zeros((len(deltas),6))]
        return ReplayEpisode(self.source,actions,states,None,{"source_hand":self.hand})


def load_sources(task,hand,limit=None):
    if hand=="allegro":roots=[ROOT/"DexJoCo-Datasets-Raw/dexjoco_raw_datasets"/task]
    else:
        layout="dual_arm" if task.startswith("bimanual") else "single_arm"
        directory="shadow_hand_e3m5" if hand=="shadow" else hand
        canonical=ROOT/"demos_cross_embodiment"/directory/layout/"ob_jrandom"
        candidates=sorted(p for p in canonical.glob("*/replay.zarr") if p.parent.name.startswith(task+"_"))
        roots=[] if candidates else [ROOT/"demos"/directory/layout/"ob_jrandom"]
    if hand=="allegro" or roots:
        candidates=sorted(p for root in roots for p in root.glob("*/replay.zarr")
                          if hand=="allegro" or p.parent.name.startswith(task+"_"))
    result=[]
    for path in candidates:
        z=zarr.open(str(path),mode="r")["data"]
        actions=np.asarray(z["action"],float).reshape(len(z["action"]),-1)
        states=np.asarray(z["state"],float).reshape(len(z["state"]),-1)
        arms=2 if task.startswith("bimanual") else 1
        derived=()
        if task=="bimanual_hanoi":
            derived=("Hanoi source records only base XYZ; base orientation is fixed identity; initial disk topology is unavailable",)
        if task=='click_mouse' and states.shape[1]==arms*(7+NH[hand])+8:
            offset=arms*(7+NH[hand]);pad=states[:,offset:offset+7].copy()
            pad[:,1]-=.25;pad[:,2]=.92+states[:,-1];pad[:,3:]=[1,0,0,0]
            states=np.c_[states[:,:-1],pad,states[:,-1]]
            derived=('legacy fixed mousepad pose inferred from mouse reset XY, shared -0.25m Y offset and table height',)
        if task=="bimanual_microwave_cook" and states.shape[1]==arms*(7+NH[hand])+8:
            # Explicit legacy schema: TCP, hand, hot-dog pose, table height.
            # Its fixed microwave was not observed. Reconstruct the TEMPLATE
            # frame from the local legacy asset + recorded table delta, and
            # label the inference. This is never a collected ground-truth label.
            asset=ROOT/"dexjoco/dexjoco/sim/envs/xmls/robocasa_microwave.xml"
            body=ET.parse(asset).find(".//body[@name='microwave_object']")
            pose=np.r_[np.fromstring(body.get("pos"),sep=" "),np.fromstring(body.get("quat"),sep=" ")]
            poses=np.tile(pose,(len(states),1));poses[:,2]+=states[:,-1]
            states=np.c_[states[:,:-1],poses,states[:,-1]]
            derived=("microwave template pose inferred from legacy fixed asset plus recorded table delta",)
        extra=4 if task=="bimanual_hanoi" else 15
        if actions.shape[1] != arms*(7+NH[hand]) or states.shape[1] != arms*(7+NH[hand])+extra:
            continue # An unsupported historical schema cannot be guessed.
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):continue
        result.append(Demonstration(path.parent,task,hand,actions,states,derived))
        if limit is not None and len(result)>=limit:break
    if not result:raise ValueError(f"No supported source demonstrations for {task}/{hand}")
    return result


def nearest(env,demos,rank=0):
    target=np.array([getattr(env,name) for name in RESET_ATTRS[env.task]])
    def score(demo):
        poses=demo.objects
        translation=np.linalg.norm((poses[:,:3]-target[:,:3])/np.array([.08,.08,.04]))
        rotation=sum(Rotation.from_quat(a[3:],scalar_first=True).inv().__mul__(
            Rotation.from_quat(b[3:],scalar_first=True)).magnitude() for a,b in zip(poses,target))
        return translation+rotation
    return sorted(demos,key=score)[rank]


def load_pilot_source(directory):
    """Read a physically successful pilot as a NEW controller template.

    This does not promote old data or inherit its success in the target hand.
    Reconstruct absolute setpoints from the stored actual TCP + world delta,
    and preserve 20 Hz action timing. Only supported native source hands can
    be decoded; Shadow's old proxy-state format is intentionally excluded.
    """
    import json
    from .environments import make_env
    from ..autocollect.retarget import _pose7_to_matrix, _matrix_to_pose7
    directory=Path(directory);meta=json.loads((directory/'summary.json').read_text())
    from .success_policy import required_success_steps
    if not meta.get('success') or meta.get('stable_success_steps',0)<required_success_steps(meta['task']):
        raise ValueError('Pilot source must have a physical successful trajectory')
    hand=meta['hand']
    if hand not in ('inspire','wuji','allegro','leap','xhand','sharpa_wave','linkerhand_l20'):
        raise ValueError('Unsupported pilot source hand transmission')
    task={'hammer':'hammer_nail','bucket':'pick_bucket'}.get(meta['task'],meta['task'])
    if task=='bimanual_hanoi':
        raise ValueError('Hanoi pilot templates need explicit initial tower topology support')
    z=zarr.open(str(directory/'replay.zarr'),mode='r')['data']
    states=np.asarray(z['state'],float);deltas=np.asarray(z['action'],float)
    arms=2 if task.startswith('bimanual') else 1;nh=NH[hand]
    if states.shape[1]!=arms*(7+nh) or deltas.shape[1]!=arms*(6+nh):
        raise ValueError('Unexpected pilot array schema')
    actions=[]
    for state,delta in zip(states,deltas):
        row=[]
        for side in range(arms):
            pose=_pose7_to_matrix(state[side*7:(side+1)*7]);d=delta[side*(6+nh):(side+1)*(6+nh)]
            pose[:3,3]+=d[:3]
            pose[:3,:3]=Rotation.from_rotvec(d[3:6]).as_matrix()@pose[:3,:3]
            row.extend(np.r_[_matrix_to_pose7(pose),d[6:]])
        actions.append(row)
    env=make_env(task,hand,meta['seed'])
    try:
        objects=np.concatenate([getattr(env,n) for n in RESET_ATTRS[task]])
        extra=np.tile(np.r_[objects,env.delta_h],(len(states),1))
    finally:env.close()
    actions=np.array(actions);states=np.c_[states,extra]
    derived=('template absolute targets reconstructed from 20 Hz physical pilot; initial object frames from its recorded reset seed',)
    if task=='bimanual_unlock_ipad':
        prepared=next((event for event in meta.get('task_metrics',{}).get('phase_history',[])
            if event.get('phase')=='ready'),None)
        if prepared is not None:
            # The target teacher uses only the LEFT acquisition demonstration,
            # then performs its own right-hand input. Replaying the source's
            # complete input tail wastes the target's finite task horizon.
            time=float(prepared['time'])
            ends=np.asarray(z['timestamp'],float)+.05
            index=int(np.searchsorted(ends,time,side='left'))+1
            if not np.isfinite(time) or not 0<index<=len(actions):
                raise ValueError('Invalid recorded iPad preparation boundary')
            actions=actions[:index];states=states[:index]
            derived+=(f'left acquisition template ends at verified source preparation boundary: {index} steps; source input tail not replayed',)
    return Demonstration(directory,task,hand,actions,states,derived)
