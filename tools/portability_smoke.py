#!/usr/bin/env python3
"""Initialize and step every benchmark hand-task cell from this checkout."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"evaluation"))
from runtime import Runtime,PROTOCOL

def main():
    groups=[]
    for task in PROTOCOL["tasks"]:
        for hand in PROTOCOL["hands"]:
            runtime=Runtime(task,hand,0,"grouped80_v1","native",render=False)
            try:
                obs=runtime.observation()
                hold=runtime.e.hold_action()
                action=runtime.encode_action(hold)
                success,terminated,truncated,clipped,info=runtime.step(action)
                assert obs["state"].shape==(80,)
                assert np.isfinite(obs["state"]).all() and np.isfinite(action).all()
                groups.append({
                    "task":task,"hand":hand,"passed":True,
                    "active_state_dims":int(obs["state_mask"].sum()),
                    "active_action_dims":int(obs["action_mask"].sum()),
                    "physics_step_finite":bool(np.isfinite(runtime.d.qpos).all()),
                })
                print(f"{task}/{hand}: passed",flush=True)
            finally:
                runtime.close()
    result={"passed":len(groups)==42,"cells":len(groups),
            "checks":["environment_init","contract","observation","native_action_encode","physics_step"],
            "groups":groups}
    (ROOT/"PORTABILITY_SMOKE.json").write_text(json.dumps(result,indent=2)+"\n")
    assert result["passed"]

if __name__=="__main__":main()
