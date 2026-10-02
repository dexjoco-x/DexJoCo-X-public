"""Physical-unit 80-D policy server boundary with explicit flow noise."""
import argparse, importlib
import numpy as np
import msgpack_numpy as mp
from websockets.sync.server import serve

def handler_for(policy,layout,representation,action_horizon=30,extra=None):
    meta=dict(schema=f"benchmark7_eval80_h{action_horizon}_async_v2",layout=layout,representation=representation,
              action_dim=80,action_horizon=action_horizon,action_units="physical_unnormalized",
              explicit_flow_noise=True,**(extra or {}))
    def handler(ws):
        ws.send(mp.packb(meta))
        for message in ws:
            try:
                obs=mp.unpackb(message);noise=np.asarray(obs.pop("initial_noise"),np.float32)
                if noise.shape!=(action_horizon,80) or not np.isfinite(noise).all():raise ValueError("Bad explicit noise")
                result=policy.infer(obs,noise=noise);actions=np.asarray(result["actions"])
                if actions.shape!=(action_horizon,80) or not np.isfinite(actions).all():
                    raise ValueError("Policy must return a finite physical-unit action chunk")
                ws.send(mp.packb({"actions":actions}))
            except Exception as error:
                ws.send(str(error));raise
    return handler

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory",required=True,help="Installed user module:function; called with checkpoint path")
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--layout",choices=["grouped80_v1","arm40_v1"],required=True)
    parser.add_argument("--representation",choices=["native","faas","dexlatent"],required=True)
    parser.add_argument("--action-horizon",type=int,default=30);parser.add_argument("--port",type=int,default=8200);parser.add_argument("--bind",default="127.0.0.1")
    args=parser.parse_args()
    module,separator,name=args.factory.partition(':')
    if not separator or not module or not name:parser.error('--factory must be module:function')
    policy=getattr(importlib.import_module(module),name)(args.checkpoint)
    with serve(handler_for(policy,args.layout,args.representation,args.action_horizon,{"factory":args.factory}),args.bind,args.port,compression=None,max_size=None) as server:server.serve_forever()
