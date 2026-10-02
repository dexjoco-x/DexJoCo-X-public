"""Register portable native200 data without modifying the existing Being-H tree.

The Parquet tensors are already normalized with bound training-only statistics.
Always import this module before the training dataset registry is constructed.
"""
import numpy as np
from configs.data_config import BaseDataConfig, ModalityDef, DATA_CONFIG_MAP
from BeingH.dataset.transform.base import ComposedModalityTransform
from BeingH.dataset.transform.state_action import StateActionToTensor
from BeingH.dataset.datasets.vla_dataset import LeRobotIterableDataset

class Native200(BaseDataConfig):
 VIDEO_KEYS=['video.front','video.wrist_right','video.wrist_left']
 VIDEO_SOURCE_COLUMNS={k:'observation.images.'+k.split('.')[1] for k in VIDEO_KEYS}
 STATE_KEYS=['state.native'];ACTION_KEYS=['action.native'];LANGUAGE_KEYS=['language.instruction']
 UNIFIED_MAPPING={'state.native':(0,200),'action.native':(0,200)}
 action_normalization_modes={};state_normalization_modes={}
 def define_modalities(self):
  return self.add_video_modality({'language.instruction':ModalityDef(source_column='task_index',start=0,end=0),
   'state.native':ModalityDef(source_column='observation.state',start=0,end=200),
   'action.native':ModalityDef(source_column='action',start=0,end=200),
   'mask.action':ModalityDef(source_column='action.valid_mask',start=0,end=200,continuous=False)})
 def get_sampling_indices(self):
  d=super().get_sampling_indices();d['mask.action']=self.action_indices;return d
 def get_transforms(self):return ComposedModalityTransform(transforms=[StateActionToTensor(apply_to=self.STATE_KEYS+self.ACTION_KEYS)])
 def get_feature_meta(self):return {'state.native':('project native200 normalized robot state',200),'action.native':('project native200 normalized actuator action',200)}
class Native200Single(Native200):
 VIDEO_KEYS=['video.front','video.wrist_right']
 VIDEO_SOURCE_COLUMNS={k:'observation.images.'+k.split('.')[1] for k in VIDEO_KEYS}
DATA_CONFIG_MAP['benchmark7_native200_dual']=Native200
DATA_CONFIG_MAP['benchmark7_native200_single']=Native200Single

# The upstream loader repeats the final row beyond the episode boundary. Apply
# a temporal mask so those artificial tail targets never contribute to loss.
_original=LeRobotIterableDataset._extract_and_sample_step
if not getattr(_original,'benchmark7_tail_mask',False):
 def _extract(self,df,base_index,file_path,modality_defs,sampling_indices,fps=30):
  data=_original(self,df,base_index,file_path,modality_defs,sampling_indices,fps)
  if 'action.native' in modality_defs and 'mask.action' in data:
   indices=base_index+np.asarray(sampling_indices['mask.action']);valid=(indices>=0)&(indices<len(df))
   data['mask.action']=np.asarray(data['mask.action'],bool)&valid[:,None]
   data['action.native']=np.where(data['mask.action'],data['action.native'],0)
  return data
 _extract.benchmark7_tail_mask=True
 LeRobotIterableDataset._extract_and_sample_step=_extract
