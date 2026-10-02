"""A physically single, centered right arm; native Hanoi object and tower logic."""
import json,hashlib
import numpy as np
import mujoco
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from ..sim.envs.panda_bimanual_hanoi_env import PandaBimanualHanoiGymEnv as NativeHanoi, _PANDA_HOME
from ..sim.envs.arm_profiles import configure_bimanual_arm
from ..sim.controllers import apply_arm_pose_control
from .models import XML
from .cameras import apply_wrist_rig, RIG
from .success_policy import success_policy
HANDS=('allegro','inspire','wuji','leap','sharpa_wave','xhand','linkerhand_l20')
TASK='single_hand_hanoi'
class SingleHandHanoi(NativeHanoi):
    task=TASK
    def __init__(self,hand,seed=81000,pixels=False):
        if hand not in HANDS:raise ValueError(hand)
        self.hand=hand;self.scene_seed=seed;self.pixels=pixels;self.env=self
        path=XML/f'benchmark7_v3_single_hand_hanoi_inspire_scene_{hand}.xml'
        self.hand_metadata=json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==self.hand_metadata['xml_sha256']
        self._model=mujoco.MjModel.from_xml_path(str(path));self._data=mujoco.MjData(self._model)
        self._model.opt.timestep=.002;self._control_dt=.05;self._physics_dt=.002;self._n_substeps=25
        self._viewer=None;self._renderer=None;self.render_mode='rgb_array';self.image_obs=pixels
        self._max_episode_steps=2000;self._translation_bound=.08;self._rotation_bound=.30
        self._arm_profile_right,self._panda_right_dof_ids,self._panda_right_ctrl_ids=configure_bimanual_arm(self._model,'right')
        self._arm_home_right=_PANDA_HOME.copy()
        spec=self.hand_metadata['sides']['right']
        self._hand_reset_ids=np.array([int(self._model.joint(n).qposadr) for n in spec['joint_names']])
        self._hand_home=np.asarray(spec['home'])
        self._full_hand_qpos={'right':np.array([int(self._model.joint(n).qposadr) for n in spec['state_joint_names']])}
        self._native_ctrl_ids={'right':np.array([self._model.actuator(n).id for n in spec['actuator_names']])}
        self._tcp_sites={'right':self._model.site('attachment_site_right').id}
        self._mocap_right_id=int(self._model.body('target_right').mocapid)
        self.camera_id=tuple(self._model.camera(n).id for n in ('back','left','right','handcam_rgb_right'))
        self._post_assignment_tol=.04;self._placement_height_tol=.012;self._static_qvel_tol=.35
        self._success_trigger_target=20;self._illegal_trigger_target=10
        # Scene objects
        self._base_body_id = self._model.body("hanoi_base").id
        self._base_init_pos = self._model.body_pos[self._base_body_id].copy()
        self._base_init_quat = self._model.body_quat[self._base_body_id].copy()
        self._base_body_z0 = float(self._base_init_pos[2])

        self._disk_names = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        self._disk_size_rank = {
            "hanoi_disk_large": 3,
            "hanoi_disk_medium": 2,
            "hanoi_disk_small": 1,
        }
        self._disk_qpos_adr = {}
        self._disk_qvel_adr = {}
        self._disk_body_id = {}
        self._disk_init_pos = {}
        self._disk_init_quat = {}
        self._disk_body_z0 = {}
        self._disk_half_height = {}
        self._disk_mass0 = {}
        self._disk_mass_mul = (0.75, 1.25)

        for disk_name in self._disk_names:
            joint_name = f"{disk_name}_joint"
            joint_id = self._model.joint(joint_name).id
            body_id = self._model.body(disk_name).id
            self._disk_qpos_adr[disk_name] = int(self._model.jnt_qposadr[joint_id])
            self._disk_qvel_adr[disk_name] = int(self._model.jnt_dofadr[joint_id])
            self._disk_body_id[disk_name] = body_id
            self._disk_init_pos[disk_name] = self._model.body_pos[body_id].copy()
            self._disk_init_quat[disk_name] = self._model.body_quat[body_id].copy()
            self._disk_body_z0[disk_name] = float(self._disk_init_pos[disk_name][2])
            self._disk_mass0[disk_name] = float(self._model.body_mass[body_id])

            top_site = self._model.site(f"{disk_name}_top_site").id
            bottom_site = self._model.site(f"{disk_name}_bottom_site").id
            half_height = 0.5 * (
                float(self._model.site_pos[top_site][2]) - float(self._model.site_pos[bottom_site][2])
            )
            self._disk_half_height[disk_name] = half_height

        self._post_site_names = {
            "A": "hanoi_post_a_site",
            "B": "hanoi_post_b_site",
            "C": "hanoi_post_c_site",
        }
        self._post_site_ids = {name: self._model.site(site_name).id for name, site_name in self._post_site_names.items()}

        self._table_body_id = self._model.body("table").id
        self._table_z = float(self._model.body("table").pos[2])
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._table_body_id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self._model.geom_size[gid, 1]) for gid in self._table_leg_geom_ids
        }

        self._base_upper_geom_id = self._model.geom("hanoi_base_upper_collision").id
        self._base_upper_top_local_z = float(
            self._model.geom_pos[self._base_upper_geom_id, 2] + self._model.geom_size[self._base_upper_geom_id, 2]
        )

        self.object_bodies=set(self._disk_body_id.values())|{self._base_body_id}
        for b in range(self._model.nbody):
            p=b
            while p and p not in self.object_bodies:p=int(self._model.body_parentid[p])
            if p in self.object_bodies:self.object_bodies.add(b)
        assert self._model.body('base_right').pos[1]==0
        assert not any('benchhand_left' in (self._model.body(i).name or '') or self._model.body(i).name=='base_left' for i in range(self._model.nbody))
        self.left_bodies=set();self.left_violation=False
        self.contract=dict(version='single_hand_hanoi_inspire_scene_v3',task=TASK,hand=hand,
            object_source='arena_arm_hand_bimanual_hanoi_inspire.xml',
            model_xml=str(path),model_sha256=self.hand_metadata['xml_sha256'],
            object_source_hashes=self.hand_metadata['object_source_hashes'],
            active_arms=['right'],left_arm_present=False,arm_base_position=self._model.body('base_right').pos.tolist(),
            control_dt=.05,physics_dt=.002,max_episode_steps=2000,
            action='world delta TCP rotvec6 + native actuator position targets',
            translation_action_bound=.08,rotation_action_bound=.30,
            initial_towers={'A':['small'],'B':['medium'],'C':['large']},
            goal_towers={'A':[],'B':[],'C':['large','medium','small']},
            success_policy=success_policy(TASK),release_steps=20,
            post_assignment_tol=.04,placement_height_tol=.012,static_qvel_tol=.35,
            success_trigger_target=20,illegal_trigger_target=10,auto_reset=False,
            wrist_camera_rig=dict(RIG),model_mutations_during_rollout=False,
            arm_pos_gain=400.,arm_ori_gain=200.,arm_damping_ratio=4.)
        ids=self._native_ctrl_ids['right'];limits=self._model.actuator_ctrlrange[ids]
        self.action_space=spaces.Box(np.r_[[-.08]*3,[-.30]*3,limits[:,0]],np.r_[[.08]*3,[.30]*3,limits[:,1]],dtype=np.float64)
        apply_wrist_rig(self)
        obs,_=self.reset(seed=seed)
        self.observation_space=spaces.Dict({k:spaces.Box(-np.inf,np.inf,shape=np.asarray(v).shape,dtype=np.float64) for k,v in obs.items()})
    @property
    def hanoi_base_ori_pose(self):return np.r_[self.base_ori_pos,self._model.body('hanoi_base').quat]
    def tcp(self,side='right'):
        sid=self._tcp_sites[side];pose=np.eye(4);pose[:3,3]=self._data.site_xpos[sid];pose[:3,:3]=self._data.site_xmat[sid].reshape(3,3);return pose
    def observe(self):
        return dict(tcp=self.tcp(),hand_qpos=self._data.qpos[self._full_hand_qpos['right']].copy(),
            hand_command=self._data.ctrl[self._native_ctrl_ids['right']].copy(),
            disk_poses=np.stack([self._body_pose(n) for n in self._disk_names]),base_pose=self.hanoi_base_ori_pose.copy(),table_height=float(self.delta_h))
    def reset(self,seed=None,**kwargs):
        self.scene_seed=self.scene_seed if seed is None else int(seed)
        rng=np.random.RandomState(self.scene_seed);m=self._model;d=self._data
        mujoco.mj_resetData(m,d)
        self.delta_h=float(rng.uniform(0,.05));m.body('table').pos[2]=self._table_z+self.delta_h
        for gid in self._table_leg_geom_ids:m.geom_size[gid,1]=self._table_leg_half_len0[gid]+self.delta_h
        xy=rng.uniform([-.25,0],[-.2,0]);self.base_ori_pos=np.r_[xy,self._base_body_z0+self.delta_h];m.body_pos[self._base_body_id]=self.base_ori_pos
        steps,towers=self._sample_reset_tower_state();self._apply_reset_tower_state(towers,xy-self._base_init_pos[:2])
        d.qpos[self._panda_right_dof_ids]=self._arm_home_right;d.qpos[self._hand_reset_ids]=self._hand_home
        # Coupled hands have more physical joints than native commands; derive reset ctrl from transmission.
        from .environments import Common
        names=self.hand_metadata['sides']['right']
        if len(self._native_ctrl_ids['right'])==len(self._hand_home):cmd=self._hand_home
        else:
            mujoco.mj_forward(m,d);cmd=d.actuator_length[self._native_ctrl_ids['right']].copy()
        d.ctrl[self._native_ctrl_ids['right']]=cmd
        mujoco.mj_forward(m,d);tcp=self.tcp();d.mocap_pos[self._mocap_right_id]=tcp[:3,3];d.mocap_quat[self._mocap_right_id]=Rotation.from_matrix(tcp[:3,:3]).as_quat(scalar_first=True)
        mujoco.mj_forward(m,d)
        self.env_step=0;self._success_trigger_count=0;self._illegal_trigger_count=0;self._illegal_trigger_reason='';self._pending_illegal_reset=False
        self._hanoi_metrics_time=None;self.release_count=0;self.left_violation=False;self.done=False;self.trace=[]
        return self.observe(),dict(task=TASK,active_arms=['right'],reset_steps_remaining=steps,reset_tower_state=towers)
    def hold_action(self):return np.r_[np.zeros(6),self._data.ctrl[self._native_ctrl_ids['right']]]
    def _compute_success_metrics(self):
        if self._hanoi_metrics_time!=float(self._data.time):
            self._hanoi_metrics=NativeHanoi._compute_success_metrics(self);self._hanoi_metrics_time=float(self._data.time)
        return self._hanoi_metrics
    def step(self,action):
        if self.done:raise RuntimeError('Episode ended; reset required')
        if isinstance(action,dict):raise ValueError('Only the right-arm vector is accepted')
        action=np.asarray(action,dtype=float)
        if action.shape!=self.action_space.shape or not np.isfinite(action).all():raise ValueError('Invalid action')
        if np.any(action<self.action_space.low-1e-8) or np.any(action>self.action_space.high+1e-8):raise ValueError('Action outside limits')
        m=self._model;d=self._data;tcp=self.tcp();mid=self._mocap_right_id
        d.mocap_pos[mid]=tcp[:3,3]+action[:3]
        d.mocap_quat[mid]=Rotation.from_matrix(Rotation.from_rotvec(action[3:6]).as_matrix()@tcp[:3,:3]).as_quat(scalar_first=True)
        for _ in range(self._n_substeps):
            d.ctrl[self._panda_right_ctrl_ids]=apply_arm_pose_control(model=m,data=d,profile=self._arm_profile_right,site_id=self._tcp_sites['right'],dof_ids=self._panda_right_dof_ids,ctrl_ids=self._panda_right_ctrl_ids,pos=d.mocap_pos[mid],ori=d.mocap_quat[mid],joint=self._arm_home_right,gravity_comp=True,pos_gains=(400.,)*3,damping_ratio=4,nullspace_stiffness=.5)
            d.ctrl[self._native_ctrl_ids['right']]=action[6:];mujoco.mj_step(m,d)
        self.env_step+=1;native,metrics=self._compute_success_metrics();contacts=[]
        for c in d.contact:
            bodies=[int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            for disk,other in (bodies,bodies[::-1]):
                if disk in self._disk_body_id.values() and m.body(other).name.startswith('benchhand_right'):contacts.append(m.body(other).name)
        self.release_count=self.release_count+1 if native and not contacts else 0
        success=self.release_count>=20;self.done=bool(success or metrics['illegal_state'] or self.env_step>=self._max_episode_steps)
        info=dict(metrics,succeed=success,native_succeed=native,single_hand_valid=True,left_task_contacts=[],released_stable_steps=self.release_count,hand_disk_contacts=sorted(set(contacts)),success_stable_count=self._success_trigger_count,illegal_stable_count=self._illegal_trigger_count,auto_reset=False,task=TASK)
        self.trace.append(info);return self.observe(),float(success),self.done,False,info
    def render(self):
        if self._renderer is None:self._renderer=mujoco.Renderer(self._model,480,640)
        frames=[]
        for camera in self.camera_id:self._renderer.update_scene(self._data,camera=camera);frames.append(self._renderer.render().copy())
        return frames
    def close(self):
        if self._renderer is not None:self._renderer.close();self._renderer=None
    def object_signature(self):
        m=self._model;rows=[]
        for b in sorted(self.object_bodies):
            # World placement is sampled separately; local shape/physics must match.
            rows.append([m.body(b).name,m.body_mass[b].item(),m.body_inertia[b].tolist()])
            for g in np.where(m.geom_bodyid==b)[0]:
                rows.append([m.geom(g).name,int(m.geom_type[g]),m.geom_size[g].tolist(),m.geom_pos[g].tolist(),m.geom_quat[g].tolist(),m.geom_friction[g].tolist(),int(m.geom_contype[g]),int(m.geom_conaffinity[g])])
        return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
