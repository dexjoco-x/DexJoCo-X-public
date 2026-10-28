"""Portable Native data conversion and masked frame loading.

No simulator dependency. Being-H uses a project native200 layout, NOT FAAS or
DexLatent, with right/left arm slots disjoint. Native finger channels are retained.
"""
import hashlib,json
import copy
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
try:
 from .normalization import normalize,denormalize
except ImportError:
 from normalization import normalize,denormalize

VIEWS=('front','wrist_right','wrist_left')
PI_KEYS={'front':'base_0_rgb','wrist_right':'right_wrist_0_rgb','wrist_left':'left_wrist_0_rgb'}
def bind_action_bounds(stats,contract):
 """Bind physical ranges without narrowing them to the demonstration quantiles."""
 result=copy.deepcopy(stats);lower=np.zeros(200);upper=np.zeros(200)
 cart=contract['native_cartesian_bounds']
 for side,base in [('right',0),('left',40)]:
  if side not in contract['actuator_ranges']:continue
  ranges=np.asarray(contract['actuator_ranges'][side],np.float64)
  width=6+len(ranges)
  lower[base:base+width]=np.r_[[-cart['translation_m']]*3,[-cart['rotation_rad']]*3,ranges[:,0]]
  upper[base:base+width]=np.r_[[cart['translation_m']]*3,[cart['rotation_rad']]*3,ranges[:,1]]
 result['physical_bounds']={'minimum':lower.tolist(),'maximum':upper.tolist()}
 result['label_quantization']='inward_float32_v1'
 return result

def load_training_stats(root,layout):
 root=Path(root);record=json.loads((root/'prepared'/(layout+'_train_stats.json')).read_text())
 actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'bundle/manifests').glob('*.json')}
 if record['manifest_hashes']!=actual:raise ValueError('Dataset changed; recompute and freeze training-only statistics')
 return record['groups']
def encode(x,m,key,layout):
 x=np.asarray(x);m=np.asarray(m,dtype=bool)
 if layout!='beingh05':raise ValueError(layout)
 y=np.zeros((*x.shape[:-2],200),dtype=x.dtype);valid=np.zeros_like(y,dtype=bool)
 for arm,base in enumerate((0,40)):
  active=m[...,arm,0];y[...,base:base+3]=x[...,arm,:3];valid[...,base:base+6]=active[...,None]
  if key=='state':
   q=x[...,arm,3:7].copy();q=np.where(active[...,None],q,np.array([1,0,0,0]))
   rot=Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_rotvec().reshape(*x.shape[:-2],3);offset=7
  else:rot=x[...,arm,3:6];offset=6
  y[...,base+3:base+6]=rot
  n=x.shape[-1]-offset;y[...,base+6:base+6+n]=x[...,arm,offset:];valid[...,base+6:base+6+n]=m[...,arm,offset:]
 return np.where(valid,y,0),valid
def decode(x,mask,layout):
 x=np.asarray(x)
 if layout=='beingh05':y=np.stack([x[...,0:28],x[...,40:68]],axis=-2)
 else:raise ValueError(layout)
 return np.where(mask,y,0)
def saved_training_labels(arrays,row,stats,selection=slice(None)):
 """Read collected labels verbatim, bound to the decoder used during execution."""
 digest=hashlib.sha256(json.dumps(stats,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if digest!=row.get('training_statistics_sha256'):raise ValueError('Saved labels require their original frozen action statistics')
 labels=arrays['training_labels']
 if labels.dtype!=np.float32 or labels.shape!=(len(arrays['action']),200):raise ValueError('Invalid saved training label shape or dtype')
 labels=labels[selection]
 raw=arrays['action'][selection];native_mask=arrays['action_mask'][selection]
 _,mask=encode(raw,native_mask,'action','beingh05')
 if not np.isfinite(labels).all() or np.any(labels[~mask]):raise ValueError('Invalid saved training label values or padding')
 restored=decode(denormalize(labels,mask,stats),native_mask,'beingh05')
 np.testing.assert_array_equal(restored,raw)
 return labels
def statistics(values,masks):
 x=np.concatenate(values);m=np.concatenate(masks);d=x.shape[-1]
 out={k:np.zeros(d) for k in ('mean','std','min','max','q01','q99','count')}
 for i in range(d):
  v=x[:,i][m[:,i]]
  if len(v):
   out['mean'][i]=v.mean();out['std'][i]=v.std();out['min'][i]=v.min();out['max'][i]=v.max();out['q01'][i],out['q99'][i]=np.quantile(v,[.01,.99]);out['count'][i]=len(v)
  else:out['q01'][i]=-1;out['q99'][i]=1;out['std'][i]=1
 return {k:v.tolist() for k,v in out.items()}
class PortableDataset:
 def __init__(self,root,layout='beingh05',split='train',horizon=16,stats=None,holdout_hand=None):
  self.root=Path(root);self.layout=layout;self.horizon=horizon;self.holdout_hand=holdout_hand
  allrows=[]
  for p in sorted((self.root/'manifests').glob('*.json')):allrows.extend(json.loads(p.read_text()))
  self.allrows=allrows
  self.rows=[r for r in allrows if r['split']==split and (not holdout_hand or (r['hand']==holdout_hand if split=='test' else r['hand']!=holdout_hand))]
  self.ends=np.cumsum([r['steps'] for r in self.rows]);self.stats=stats
  self._cache={}
 def __len__(self):return int(self.ends[-1]) if len(self.ends) else 0
 def arrays(self,row):
  key=row['tensor']
  if key not in self._cache:
   if len(self._cache)>=8:self._cache.pop(next(iter(self._cache)))
   with np.load(self.root/key,allow_pickle=False) as z:self._cache[key]={k:z[k] for k in z.files}
  return self._cache[key]
 def fit_stats(self):
  groups={}
  for r in self.allrows:
   if r['split']!='train' or r['hand']==self.holdout_hand:continue
   key=r['task']+'/'+r['hand'];group=groups.setdefault(key,{'state':[],'state_mask':[],'action':[],'action_mask':[]})
   z=self.arrays(r)
   for k in ('state','action'):
    v,m=encode(z[k],z[k+'_mask'],k,self.layout);group[k].append(v);group[k+'_mask'].append(m)
  self.stats={g:{k:statistics(a[k],a[k+'_mask']) for k in ('state','action')} for g,a in groups.items()}
  return self.stats
 def frame(self,row,t):
  if not 0<=t<row['steps']:raise IndexError(t)
  z=self.arrays(row);st,sm=encode(z['state'][t],z['state_mask'][t],'state',self.layout)
  count=min(self.horizon,row['steps']-t);a,am=encode(z['action'][t:t+count],z['action_mask'][t:t+count],'action',self.layout)
  a=np.pad(a,((0,self.horizon-count),(0,0)));am=np.pad(am,((0,self.horizon-count),(0,0)))
  stats=self.stats[row['task']+'/'+row['hand']]
  if 'training_labels' in z:
   labels=saved_training_labels(z,row,stats['action'],slice(t,t+count))
   labels=np.pad(labels,((0,self.horizon-count),(0,0)))
  else:labels=normalize(a,am,stats['action'])
  images={};masks={}
  for v in VIEWS:
   present=v in row['videos'];masks[PI_KEYS[v]]=present
   if present:
    cap=cv2.VideoCapture(str(self.root/row['videos'][v]['path']));cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,img=cap.read();cap.release()
    if not ok:raise ValueError('Video frame decode failed')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);images[PI_KEYS[v]]=cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
   else:images[PI_KEYS[v]]=np.zeros((224,224,3),np.uint8)
  contract=json.loads((self.root/row['contract']).read_text())
  return {'state':normalize(st,sm,stats['state']),'state_mask':sm,'images':images,'image_masks':masks,
   'actions':labels,'action_mask':am,'prompt':f"Embodiment: {row['hand']}. {contract['instruction']}"}
 def __getitem__(self,index):
  if not 0<=index<len(self):raise IndexError(index)
  i=int(np.searchsorted(self.ends,index,side='right'));t=index-(int(self.ends[i-1]) if i else 0)
  return self.frame(self.rows[i],t)
 def native_actions(self,prediction,row):
  z=self.arrays(row);_,mask=encode(z['action'][0],z['action_mask'][0],'action',self.layout)
  s=self.stats[row['task']+'/'+row['hand']]['action'];return decode(denormalize(prediction,mask,s),z['action_mask'][0],self.layout)


