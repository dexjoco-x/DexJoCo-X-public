"""Versioned, unnormalized 80-D policy boundary over the frozen v8 simulator."""
import os,sys,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ['BENCHMARK7_CAMERA_PROFILE']='dexjoco_official_allegro60_hanoi_front_v1'
REPOSITORY_ROOT=P.parent
sys.path[:0]=[str(REPOSITORY_ROOT/'simulator'),str(P/'codecs')]
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from dexjoco.benchmark6.environments import make_env
PROTOCOL=json.loads((P/'eval_protocol.json').read_text())
CONTRACTS=json.loads((P/'contracts.json').read_text())
LAYOUTS=('grouped80_v1','arm40_v1')
DEFAULT_CODEC_SHA='73b1c80497a493835e58ace3e3abe509906d63916a4c263f5191d585a38bbd7b'
_CODEC_CACHE={}

def codec_for(hand,representation,side):
 if representation=='native':return None
 key=(hand,representation,side)
 if key not in _CODEC_CACHE:
  contracts=json.loads((P/'codecs/faas7_project_v1.json').read_text())['hands']
  if representation=='faas':
   from faas7 import FAAS7
   _CODEC_CACHE[key]=FAAS7(contracts[hand]['sides'][side])
  elif representation=='dexlatent':
   import torch
   from dexlatent7 import SidedDexLatent7,HandHead
   torch.set_num_threads(2)
   path=Path(os.environ.get('BENCHMARK7_DEXLATENT_CHECKPOINT',P/'codecs/selected.pt'))
   expected_sha=os.environ.get('BENCHMARK7_DEXLATENT_SHA256',DEFAULT_CODEC_SHA)
   assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha
   payload=torch.load(path,map_location='cpu',weights_only=True)
   assert isinstance(payload.get('step'),int) and payload['step']>0
   canonical={h:c['sides'][c['canonical_side']] for h,c in contracts.items()}
   assert payload['contracts']==canonical
   heads=torch.nn.ModuleDict({h:HandHead(len(c['joint_names'])) for h,c in canonical.items()});heads.load_state_dict(payload['heads'])
   for h,entry in contracts.items():
    for s,side_contract in entry['sides'].items():
     _CODEC_CACHE[(h,representation,s)]=SidedDexLatent7(canonical[h],side_contract,heads[h])
  else:raise ValueError(representation)
 return _CODEC_CACHE[key]

class Runtime:
 def __init__(self,task,hand,seed,layout,representation='native',render=True):
  if layout not in LAYOUTS:raise ValueError('An explicit supported layout is required')
  if task not in PROTOCOL['tasks'] or hand not in PROTOCOL['hands']:raise ValueError((task,hand))
  self.layout=layout;self.representation=representation;self.task=task;self.hand=hand
  self.e=make_env(task,hand,int(seed),pixels=False);self.e.reset(seed=int(seed))
  self.m=self.e._model;self.d=self.e._data;self.shadow=mujoco.MjData(self.m)
  self.contract=CONTRACTS[task+'/'+hand]
  self.codecs={side:codec_for(hand,representation,side) for side in self.e._native_ctrl_ids}
  self.renderer=mujoco.Renderer(self.m,height=256,width=256) if render else None
  self.cameras=self.contract['cameras'];self._check_contract()
 def _check_contract(self):
  expected_revision=self.contract.get('model_revision')
  revision=self.e.hand_metadata.get('mount_revision',{}).get('id')
  if self.contract.get('evaluation_requires_matching_mount') and revision!=expected_revision:
   raise RuntimeError(f'Model mount mismatch for {self.task}/{self.hand}: expected {expected_revision}, got {revision}')
  binding=self.contract.get('model_binding')
  if binding:
   body=self.m.body(binding['body'])
   if not np.allclose(body.pos,binding['position'],atol=1e-12,rtol=0):
    raise RuntimeError(f'Model mount position mismatch for {self.task}/{self.hand}')
   expected=Rotation.from_quat(binding['quaternion_wxyz'],scalar_first=True).as_matrix()
   actual=Rotation.from_quat(body.quat,scalar_first=True).as_matrix()
   if not np.allclose(actual,expected,atol=1e-12,rtol=0):
    raise RuntimeError(f'Model mount orientation mismatch for {self.task}/{self.hand}')
  self.model_binding={'xml_sha256':self.e.hand_metadata['xml_sha256'],'mount_revision':revision}
  for side,ids in self.e._native_ctrl_ids.items():
   assert [self.m.actuator(int(i)).name for i in ids]==self.contract['actuator_names'][side]
   assert self.e.hand_metadata['sides'][side]['state_joint_names']==self.contract['joint_names'][side]
  for name,c in self.cameras.items():
   i=self.m.camera(c['camera_name']).id
   np.testing.assert_allclose(self.m.cam_pos[i],c['position_body_m'],atol=1e-12,rtol=0)
   np.testing.assert_allclose(self.m.cam_quat[i],c['quaternion_body_wxyz'],atol=1e-12,rtol=0)
   assert self.m.cam_fovy[i]==c['vertical_fov_deg']
  assert self.m.opt.timestep==.002
 def slots(self,arm,state=False):
  if self.layout=='arm40_v1':return arm*40,arm*40+6,6
  return (arm*7,14+arm*32,7) if state else (arm*6,12+arm*32,6)
 def masks(self,state=False):
  out=np.zeros(80,bool)
  for arm,side in enumerate(('right','left')):
   if side not in self.e._native_ctrl_ids:continue
   a,h,n=self.slots(arm,state);out[a:a+n]=True
   if self.representation=='native':count=len(self.e._full_hand_qpos[side] if state else self.e._native_ctrl_ids[side]);out[h:h+count]=True
   elif self.representation=='faas':out[h:h+32]=self.codecs[side].mask
   else:out[h:h+32]=True
  return out
 def observation(self):
  # Recompute FK for images/state on separate data, leaving physics integration caches intact.
  
  for name in ('qpos','qvel','act','ctrl','mocap_pos','mocap_quat','qacc_warmstart'):
   getattr(self.shadow,name)[:]=getattr(self.d,name)
  self.shadow.time=self.d.time
  mujoco.mj_forward(self.m,self.shadow)
  out=np.zeros(80,np.float32)
  for arm,side in enumerate(('right','left')):
   if side not in self.e._native_ctrl_ids:continue
   sid=self.e._tcp_sites[side];a,h,n=self.slots(arm,True);out[a:a+3]=self.shadow.site_xpos[sid]
   rot=Rotation.from_matrix(self.shadow.site_xmat[sid].reshape(3,3));out[a+3:a+n]=rot.as_quat(scalar_first=True) if n==7 else rot.as_rotvec()
   codec=self.codecs[side];q=self.shadow.qpos[self.e._full_hand_qpos[side]];v=q if codec is None else codec.state(q);out[h:h+len(v)]=v
  obs={'state':out,'state_mask':self.masks(True),'action_mask':self.masks(False),'prompt':self.contract['instruction'],
       'task':self.task,'hand':self.hand}
  image_mask=[]
  for name,key in [('front','base'),('wrist_left','wrist_left'),('wrist_right','wrist_right')]:
   exists=name in self.cameras;image_mask.append(exists)
   if exists and self.renderer:
    self.renderer.update_scene(self.shadow,camera=self.cameras[name]['camera_name']);im=self.renderer.render().copy()
   else:im=np.zeros((256,256,3),np.uint8)
   obs[key]=im
  obs['image_mask']=np.asarray(image_mask,bool)
  return obs
 def encode_action(self,commands):
  if not isinstance(commands,dict):commands={'right':commands}
  out=np.zeros(80,np.float64)
  for arm,side in enumerate(('right','left')):
   if side not in self.e._native_ctrl_ids:continue
   a,h,_=self.slots(arm);v=np.asarray(commands[side]);codec=self.codecs[side];out[a:a+6]=v[:6];q=v[6:] if codec is None else codec.action(v[6:]);out[h:h+len(q)]=q
  return out
 def decode_action(self,action,clip=True):
  x=np.array(action,dtype=np.float64,copy=True)
  if x.shape!=(80,) or not np.isfinite(x).all():raise ValueError('Expected finite unnormalized action[80]')
  result={};clipped=0
  for arm,side in enumerate(('right','left')):
   if side not in self.e._native_ctrl_ids:continue
   a,h,_=self.slots(arm);ids=self.e._native_ctrl_ids[side];codec=self.codecs[side];q=x[h:h+len(ids)] if codec is None else codec.decode(x[h:h+32]);u=np.r_[x[a:a+6],q]
   limits=self.m.actuator_ctrlrange[ids];lo=np.r_[[-self.e._translation_bound]*3,[-self.e._rotation_bound]*3,limits[:,0]];hi=np.r_[[self.e._translation_bound]*3,[self.e._rotation_bound]*3,limits[:,1]]
   bounded=np.clip(u,lo,hi);clipped+=int(np.count_nonzero(bounded!=u));result[side]=bounded if clip else u
  return result,clipped
 def step(self,action):
  commands,clipped=self.decode_action(action);_,reward,term,trunc,info=self.e.step(commands if len(commands)>1 or self.task!='single_hand_hanoi' else commands['right'])
  return bool(info.get('succeed',False)),bool(term),bool(trunc),clipped,info
 def close(self):
  if self.renderer:self.renderer.close()
  self.e.close()
