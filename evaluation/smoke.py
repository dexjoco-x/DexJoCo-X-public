"""42-cell environment/renderer/ABI validation; never runs the held-out test scenes."""
import argparse,json,sys,time,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
from runtime import Runtime,PROTOCOL,LAYOUTS,codec_for

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);reports=[];states={}
 for ti,task in enumerate(PROTOCOL['tasks']):
  for hi,hand in enumerate(PROTOCOL['hands']):
   key=task+'__'+hand;start=time.time();r=Runtime(task,hand,PROTOCOL['smoke_seed_base']+ti*100+hi,'grouped80_v1')
   try:
    obs=r.observation();checks=[]
    assert obs['state'].shape==(80,) and not np.any(obs['state'][~obs['state_mask']])
    for name in ('base','wrist_left','wrist_right'):
     assert obs[name].shape==(256,256,3) and obs[name].dtype==np.uint8
    assert obs['base'].std()>1 and obs['wrist_right'].std()>1
    if len(r.e._native_ctrl_ids)==1:assert not obs['image_mask'][1] and not obs['wrist_left'].any()
    states[key+'__initial']=np.r_[r.d.qpos.copy(),r.d.qvel.copy(),r.d.ctrl.copy(),r.d.mocap_pos.ravel(),r.d.mocap_quat.ravel()]
    hold=r.e.hold_action()
    if not isinstance(hold,dict):hold={'right':hold}
    for layout in LAYOUTS:
     r.layout=layout
     for rep in ('native','faas','dexlatent'):
      r.representation=rep
      r.codecs={side:codec_for(hand,rep,side) for side in r.e._native_ctrl_ids}
      x=r.encode_action(hold);cmd,_=r.decode_action(x,clip=False);err=max(float(np.max(np.abs(cmd[s]-hold[s]))) for s in hold)
      if rep!='dexlatent':assert err<1e-8,(key,rep,err)
      else:assert np.isfinite(err)
      ob=r.observation();assert not np.any(ob['state'][~ob['state_mask']]);assert not np.any(x[~r.masks()])
      checks.append(dict(layout=layout,representation=rep,action_roundtrip_max_abs=err,active_action_dims=int(r.masks().sum()),active_state_dims=int(r.masks(True).sum())))
    r.layout='grouped80_v1';r.representation='native'
    r.codecs={side:None for side in r.e._native_ctrl_ids}
    for _ in range(4):r.step(r.encode_action(r.e.hold_action()))
    after=r.observation();states[key+'__after4']=np.r_[r.d.qpos.copy(),r.d.qvel.copy(),r.d.ctrl.copy(),r.d.mocap_pos.ravel(),r.d.mocap_quat.ravel()]
    assert np.isfinite(states[key+'__after4']).all()
    np.savez_compressed(out/(key+'.npz'),initial=states[key+'__initial'],after4=states[key+'__after4'])
    if hand=='xhand':
     import imageio.v2 as imageio
     imageio.imwrite(out/(key+'.jpg'),np.concatenate([after[k] for k in ('base','wrist_left','wrist_right')],axis=1))
    report=dict(task=task,hand=hand,passed=True,cameras=list(r.cameras),render_shapes=[256,256,3],physics_steps=4,checks=checks,seconds=time.time()-start)
    reports.append(report);print(key,'PASS',flush=True)
    (out/'progress.json').write_text(json.dumps(dict(completed=len(reports),total=42,last=key),indent=2))
   finally:r.close()
 versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','mujoco','gymnasium','torch','zarr','numcodecs']}
 (out/'report.json').write_text(json.dumps(dict(passed=True,cells=len(reports),groups=reports,versions=versions,full_policy_model_tested=False,heldout_scenes_executed=False,training_started=False),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
