"""Convert Native episodes to 200-D Parquet/video datasets, sharing video bytes."""
import argparse,json,os,re
from pathlib import Path
import numpy as np
import pandas as pd
from adapters import PortableDataset,encode,normalize,bind_action_bounds,saved_training_labels
from data_io import write,sha,link_or_copy
from normalization import AFFINE,AFFINE_V2,LEGACY,VERSIONS,bind,version,stabilize_scales

def main(root,normalization_version=AFFINE,output_name=None,task=None,hand=None,scene=None,statistics_file=None,reader='beingh'):
 root=Path(root);bundle=root/'bundle';output=root/'prepared';output.mkdir(exist_ok=True)
 if reader not in ('beingh','lerobot'):raise ValueError(reader)
 prefix='beingh05' if reader=='beingh' else 'lerobot'
 suffix='_affine_v3' if normalization_version==AFFINE else '_affine_v2' if normalization_version==AFFINE_V2 else '_legacy_v1'
 name=output_name or (prefix+suffix)
 if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Output name must be one plain directory name')
 statsfile=output/(name+'_train_stats.json')
 if statsfile.exists() or (output/name).exists():raise FileExistsError('Choose a new output name; existing training exports are never overwritten')
 ds=PortableDataset(bundle,layout='beingh05')
 manifests={p.name:sha(p) for p in (bundle/'manifests').glob('*.json')}
 if statistics_file:
  frozen=json.loads(Path(statistics_file).read_text());stats=frozen['groups']
  if any(version(s)!=normalization_version for g in stats.values() for s in g.values()):raise ValueError('Frozen statistics use another normalization version')
  stats_origin={'source_sha256':sha(statistics_file),'fit_manifest_hashes':frozen.get('statistics_origin',{}).get('fit_manifest_hashes',frozen['manifest_hashes'])}
 else:
  stats={g:{k:bind(s,normalization_version) for k,s in group.items()} for g,group in ds.fit_stats().items()}
  if normalization_version!=LEGACY:
   for g in stats:
    row=next(r for r in ds.allrows if r['task']+'/'+r['hand']==g)
    stats[g]['action']=bind_action_bounds(stats[g]['action'],json.loads((bundle/row['contract']).read_text()))
  if normalization_version==AFFINE:
   stats={g:{k:stabilize_scales(s) for k,s in group.items()} for g,group in stats.items()}
  stats_origin={'fit_manifest_hashes':manifests}
 rows_selected=[r for r in ds.allrows if (not task or r['task'] in task) and (not hand or r['hand'] in hand) and (not scene or r['scene_id'] in scene)]
 if not rows_selected:raise ValueError('No episodes selected')
 for r in rows_selected:
  if r['task']+'/'+r['hand'] not in stats:raise ValueError('Frozen statistics do not cover selected episodes')
 write(statsfile,{'layout':'beingh05','normalization_version':normalization_version,'label_dtype':'float32','decode_dtype':'float64' if normalization_version!=LEGACY else 'legacy',
  'scope':'per task/hand; frozen training statistics','manifest_hashes':manifests,'statistics_origin':stats_origin,'groups':stats})
 print('STATS',name,len(stats),flush=True)
 ds.layout='beingh05';ds.stats=stats
 groups={}
 for r in rows_selected:groups.setdefault((r['task'],r['hand'],r['split']),[]).append(r)
 manifest=[]
 for (task,hand,split),rows in sorted(groups.items()):
  dest=output/name/split/task/hand;meta=dest/'meta';meta.mkdir(parents=True,exist_ok=True)
  total=0;episodes=[];contract=json.loads((bundle/rows[0]['contract']).read_text());prompt=f"Embodiment: {hand}. {contract['instruction']}"
  views=list(rows[0]['videos']);normalization=stats[task+'/'+hand]
  for idx,r in enumerate(rows):
   if list(r['videos'])!=views:raise ValueError('Within-group camera presence drift')
   z=ds.arrays(r);n=r['steps'];arrays={}
   for k in ['state','action']:
    v,m=encode(z[k],z[k+'_mask'],k,'beingh05');arrays[k]=normalize(v,m,normalization[k]);arrays[k+'_mask']=m
    if k=='action' and 'training_labels' in z:
     arrays[k]=saved_training_labels(z,r,normalization[k])
   df=pd.DataFrame({'observation.state':list(arrays['state']),'action':list(arrays['action']),
    'observation.state_mask':list(arrays['state_mask']),'action.valid_mask':list(arrays['action_mask']),
    'timestamp':np.arange(n,dtype=np.float32)/20,'frame_index':np.arange(n,dtype=np.int64),
    'episode_index':idx,'index':np.arange(total,total+n,dtype=np.int64),'task_index':0})
   p=dest/'data/chunk-000'/f'episode_{idx:06d}.parquet';p.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(p,index=False)
   episode={'episode_index':idx,'tasks':[prompt],'length':n,'dataset_from_index':total,'dataset_to_index':total+n,'source_scene':r['scene_id'],'source_seed':r['seed'],'source_split':split,'data/chunk_index':0,'data/file_index':idx}
   for view in views:
    key='observation.images.'+view;video=dest/'videos'/key/'chunk-000'/f'episode_{idx:06d}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    if not video.exists():link_or_copy(bundle/r['videos'][view]['path'],video)
    episode.update({f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':idx,f'videos/{key}/from_timestamp':0.,f'videos/{key}/to_timestamp':n/20})
   episodes.append(episode);total+=n
  features={k:{'dtype':'float32','shape':[200]} for k in ['observation.state','action']}
  features.update({k:{'dtype':'bool','shape':[200]} for k in ['observation.state_mask','action.valid_mask']})
  for key in ['timestamp','frame_index','episode_index','index','task_index']:features[key]={'dtype':'float32' if key=='timestamp' else 'int64','shape':[1],'names':None}
  for view in views:features['observation.images.'+view]={'dtype':'video','shape':[256,256,3],'names':['height','width','channel'],'info':{'video.fps':20,'video.height':256,'video.width':256,'video.channels':3,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}
  paths=('chunk_index','file_index') if reader=='lerobot' else ('episode_chunk','episode_index')
  write(meta/'info.json',{'codebase_version':'v3.0' if reader=='lerobot' else 'v2.1','reader':reader,'robot_type':hand,'total_episodes':len(rows),'total_frames':total,'total_tasks':1,'chunks_size':1000,'fps':20,
   'data_path':'data/chunk-{'+paths[0]+':03d}/episode_{'+paths[1]+':06d}.parquet','video_path':'videos/{video_key}/chunk-{'+paths[0]+':03d}/episode_{'+paths[1]+':06d}.mp4','features':features,
   'splits':{split:f'0:{len(rows)}'},'representation':'benchmark7_project_native200_v1','normalization_version':normalization_version,
   'decode_dtype':'float64' if normalization_version!=LEGACY else 'legacy','training_stats_sha256':sha(statsfile),'pre_normalized':True,'object_pose_input':False,
   'physical_replay_verified':False})
  # Identity normalization metadata: actual per-group raw stats are separately
  # bound and used before writing Parquet; the model config does not renormalize.
  identity={k:[v]*200 for k,v in [('mean',0.),('std',1.),('min',-1.),('max',1.),('q01',-1.),('q99',1.)]}
  write(meta/'stats.json',{'observation.state':identity,'action':identity,'action.valid_mask':identity})
  write(meta/'raw_train_statistics.json',normalization)
  pd.DataFrame({'task_index':[0]},index=pd.Index([prompt])).to_parquet(meta/'tasks.parquet')
  (meta/'tasks.jsonl').write_text(json.dumps({'task_index':0,'task':prompt})+'\n')
  ep=meta/'episodes/chunk-000';ep.mkdir(parents=True,exist_ok=True);pd.DataFrame(episodes).to_parquet(ep/'file-000.parquet',index=False)
  (meta/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
  manifest.append({'task':task,'hand':hand,'split':split,'path':str(dest.relative_to(root)),'episodes':len(rows),'frames':total,'config':'benchmark7_native200_dual' if len(views)==3 else 'benchmark7_native200_single'})
  print('PREPARED',task,hand,split,len(rows),flush=True)
 write(output/(name+'_datasets.json'),manifest)
 # JSON is also valid YAML; the upstream training config loader accepts it.
 recipe={}
 for g in manifest:
  if g['split']!='train':continue
  recipe['benchmark7_'+g['task']+'_'+g['hand']]={'dataset_names':[g['hand']],'data_config_names':[g['config']],
   'embodiment_tags':['new_embodiment'],'sampling_strategy':'step','video_backend':'decord',
   'vit_transform_args':{'transform_type':'default','normalize_type':'imagenet','use_color_jitter':True},
   'frame_step_size':[1],'num_used_episodes_per_task':[-1],'num_used_episodes_per_dataset':[-1],'num_used_frames_per_dataset':[-1],
   # `sampling_strategy=step` samples inside a selected dataset.  Weighting
   # every task/hand group equally would still oversample short groups, so use
   # the actual number of train frames for a globally frame-proportional mix.
   'weight':g['frames']}
 write(output/(name+'_train.yaml'),recipe)
 write(output/(name+'_preparation.json'),{'status':'converted_pending_loader_and_physical_replay_checks','episodes':len(rows_selected),'normalization_version':normalization_version,
  'training_started':False,'video_storage':'hardlinked once; no duplicated video bytes','physical_replay_verified':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--normalization',choices=VERSIONS,default=AFFINE)
 p.add_argument('--output-name');p.add_argument('--task',action='append');p.add_argument('--hand',action='append');p.add_argument('--scene',action='append')
 p.add_argument('--statistics-file',type=Path,help='Reuse frozen statistics when exporting newly collected trajectories')
 p.add_argument('--reader',choices=['beingh','lerobot'],default='beingh',help='Being-H episode paths or official LeRobot v3 file-index paths; numeric labels are identical')
 a=p.parse_args();main(a.root,a.normalization,a.output_name,a.task,a.hand,a.scene,a.statistics_file,a.reader)
