"""Seven-hand/six-task evaluator using three independent 50-rollout repeats."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from websockets.sync.client import connect
import msgpack_numpy as mp
from execution_contract import rollout, EXECUTION_VERSION
from runtime import Runtime, PROTOCOL, P
from audit_native_replay import environment_fingerprint
from reference_environment import reference_mismatches
from provenance import evaluation_provenance, validate_episode_count

def main(a):
    diagnostic=bool(getattr(a,"diagnostic",False))
    validate_episode_count(a.episodes, diagnostic)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    tasks=PROTOCOL["tasks"] if a.task=="all" else [a.task]
    hands=PROTOCOL["hands"] if a.hand=="all" else [a.hand]
    manifest_sha=hashlib.sha256((P/"models.json").read_bytes()).hexdigest();results=[]
    provenance=evaluation_provenance(P.parent)
    max_steps=getattr(a,"max_steps",None)
    if max_steps is not None and (not diagnostic or max_steps<1):
        raise ValueError('--max-steps requires --diagnostic and a positive value')
    requested_action=getattr(a,"action_horizon",None);requested_execution=getattr(a,"execution_horizon",None)
    action_horizon=int(PROTOCOL["action_horizon"] if requested_action is None else requested_action)
    execute=int(PROTOCOL["execution_horizon"] if requested_execution is None else requested_execution)
    if not 1<=execute<=action_horizon:
        raise ValueError('Execution horizon must be positive and no longer than the action chunk')
    protocol_sha=hashlib.sha256((P/"eval_protocol.json").read_bytes()).hexdigest()
    numerical_environment=environment_fingerprint()
    mismatches=reference_mismatches(numerical_environment)
    if mismatches and not diagnostic:
        raise ValueError(f'Non-reference numerical environment: {mismatches}. Follow docs/REPLAY_REPAIR.md; use --diagnostic for other environments.')
    environment_record=dict(fingerprint=numerical_environment,reference_matches=not mismatches,
                            mismatches=mismatches,diagnostic_only=diagnostic)
    environment_path=out/'numerical_environment.json'
    if environment_path.exists():raise FileExistsError('Use a fresh output directory')
    environment_path.write_text(json.dumps(environment_record,indent=2)+'\n')
    environment_sha=hashlib.sha256(environment_path.read_bytes()).hexdigest()
    with connect(a.server,compression=None,max_size=None,open_timeout=30) as ws:
        meta=mp.unpackb(ws.recv(timeout=30))
        if meta.get("diagnostic_only") and not diagnostic:
            raise ValueError('A diagnostic policy cannot produce formal evaluation results')
        expected=dict(schema=f"benchmark7_eval80_h{action_horizon}_async_v2",layout=a.layout,
                      representation=a.representation,action_dim=80,
                      action_horizon=action_horizon,action_units="physical_unnormalized",
                      explicit_flow_noise=True)
        for key,value in expected.items():
            if meta.get(key)!=value:raise ValueError("Server contract mismatch: "+key)
        for task in tasks:
            for hand in hands:
                for index in range(a.episodes):
                    path=out/f"{task}__{hand}__{index:03}.json"
                    if path.exists():raise FileExistsError("Use a fresh output directory")
                    seed=PROTOCOL["test_scene_seeds"][task][index]
                    policy_seed=PROTOCOL["policy_seeds"][task][index]
                    rng=np.random.default_rng(policy_seed)
                    runtime=Runtime(task,hand,seed,a.layout,a.representation);start=time.time();writer=None
                    if a.video:
                        import imageio.v2 as imageio
                        writer=imageio.get_writer(out/f"{task}__{hand}__{index:03}.mp4",fps=20,codec="libx264",ffmpeg_params=["-threads","1"])
                    try:
                        def infer(obs):
                            ws.send(mp.packb(obs));raw=ws.recv(timeout=180)
                            if isinstance(raw,str):raise RuntimeError(raw)
                            return np.asarray(mp.unpackb(raw)["actions"])
                        limit=int(runtime.e.contract["max_episode_steps"])
                        if max_steps is not None:limit=min(limit,max_steps)
                        outcome=rollout(runtime,infer,rng,limit,
                                        writer,action_horizon,execute)
                        if not np.isfinite(runtime.d.qpos).all() or not np.isfinite(runtime.d.qvel).all():
                            raise ValueError('Non-finite simulator state after rollout')
                        row=dict(task=task,hand=hand,index=index,repeat=1+index//50,
                                 scene_seed=seed,policy_seed=policy_seed,seconds=time.time()-start,
                                 layout=a.layout,representation=a.representation,server_metadata=meta,
                                 environment_manifest_sha256=manifest_sha,
                                 environment_manifest_file="evaluation/models.json",protocol_sha256=protocol_sha,
                                 numerical_environment_file="numerical_environment.json",
                                 numerical_environment_sha256=environment_sha,
                                 diagnostic_only=diagnostic,provenance=provenance,**outcome)
                        row['action_horizon']=action_horizon;row['execution_horizon']=execute
                        path.write_text(json.dumps(row,indent=2)+"\n");results.append(row)
                        print(task,hand,index,outcome["success"],outcome["steps"],flush=True)
                    finally:
                        if writer:writer.close()
                        runtime.close()
    mode=f'synchronous_delta_h{action_horizon}_execute{execute}_v1'
    summary=dict(episodes=len(results),successes=sum(row["success"] for row in results),
                 success_rate=sum(row["success"] for row in results)/len(results),
                 execution_mode=mode,diagnostic_only=diagnostic,results=results)
    summary['numerical_environment_file']='numerical_environment.json'
    summary['numerical_environment_sha256']=environment_sha
    summary['provenance']=provenance
    summary['episodes_per_hand_task']=a.episodes
    summary['repeats_per_hand_task']=3 if not diagnostic else None
    summary['rollouts_per_repeat']=50 if not diagnostic else None
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--server",default="ws://127.0.0.1:8200")
    parser.add_argument("--layout",choices=["grouped80_v1","arm40_v1"],required=True)
    parser.add_argument("--representation",choices=["native","faas","dexlatent"],required=True)
    parser.add_argument("--task",choices=PROTOCOL["tasks"]+["all"],default="all")
    parser.add_argument("--hand",choices=PROTOCOL["hands"]+["all"],default="all")
    parser.add_argument("--episodes",type=int,default=150)
    parser.add_argument("--output",required=True);parser.add_argument("--video",action="store_true")
    parser.add_argument("--diagnostic",action="store_true",help="Label all outputs as integration checks")
    parser.add_argument("--max-steps",type=int,help="Optional short horizon for --diagnostic only")
    parser.add_argument("--action-horizon",type=int,help="Physical time steps per chunk; Ego-pi pairs 50 tokens into 25 steps")
    parser.add_argument("--execution-horizon",type=int,help="Actions executed before re-inference (default: protocol value)")
    args=parser.parse_args();main(args)
