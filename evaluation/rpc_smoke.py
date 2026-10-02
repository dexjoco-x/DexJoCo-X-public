"""Exercise the 30x80 WebSocket boundary with a hold-action test policy."""
import argparse,json,threading
import numpy as np
from websockets.sync.server import serve
from websockets.sync.client import connect
from serve80 import handler_for
import msgpack_numpy as mp
from runtime import Runtime,LAYOUTS,PROTOCOL

def run(output,representations=('native','faas')):
 reports=[];horizon=int(PROTOCOL['action_horizon'])
 for layout in LAYOUTS:
  for rep in dict.fromkeys(representations):
   r=Runtime('bimanual_photograph','xhand',21000001,layout,rep)
   class HoldPolicy:
    def infer(self,obs,noise=None):
     assert noise.shape==(horizon,80) and obs['base'].shape==(256,256,3)
     assert set(obs).issuperset({'state','state_mask','action_mask','prompt','base','wrist_left','wrist_right'})
     return {'actions':np.repeat(r.encode_action(r.e.hold_action())[None,:],horizon,axis=0)}
   try:
    with serve(handler_for(HoldPolicy(),layout,rep,horizon,{'test_policy':True}),'127.0.0.1',0,compression=None,max_size=None) as server:
     thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
     try:
      with connect('ws://127.0.0.1:'+str(server.socket.getsockname()[1]),compression=None,max_size=None) as ws:
       meta=mp.unpackb(ws.recv(timeout=20));assert meta['layout']==layout and meta['representation']==rep and meta['action_horizon']==horizon
       obs=r.observation();obs['initial_noise']=np.random.default_rng(17).standard_normal((horizon,80)).astype(np.float32)
       ws.send(mp.packb(obs));response=mp.unpackb(ws.recv(timeout=30));assert response['actions'].shape==(horizon,80)
       r.step(response['actions'][0]);assert np.isfinite(r.d.qpos).all()
     finally:server.shutdown();thread.join(timeout=5)
    reports.append(dict(layout=layout,representation=rep,passed=True));print(layout,rep,'RPC_PASS',flush=True)
   finally:r.close()
 from pathlib import Path
 Path(output).write_text(json.dumps(dict(passed=True,checks=reports,test_policy='hold_action',action_horizon=horizon,full_policy_forward_tested=False,formal_scenes_executed=False),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True)
 p.add_argument('--representations',nargs='+',choices=('native','faas','dexlatent'),default=['native','faas'])
 args=p.parse_args();run(args.output,args.representations)
