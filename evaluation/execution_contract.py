"""Frozen synchronous action-chunk execution used for Table I."""
import hashlib
from pathlib import Path
import time
import numpy as np

EXECUTION_VERSION = "synchronous_delta_h30_execute8_v1"
CONTRACT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def rollout(runtime, infer, rng, horizon, writer=None, action_horizon=30, execute=8):
    steps = clipped = requests = 0
    success = terminated = truncated = False
    latencies = []
    while steps < horizon:
        obs = runtime.observation()
        obs["initial_noise"] = rng.standard_normal((action_horizon, 80)).astype(np.float32)
        start = time.monotonic()
        actions = np.asarray(infer(obs))
        latencies.append(time.monotonic() - start)
        requests += 1
        if actions.shape != (action_horizon, 80) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite physical action chunk ({action_horizon},80)")
        for action in actions[:min(execute, horizon - steps)]:
            if writer is not None:
                writer.append_data(runtime.observation()["base"])
            success, terminated, truncated, count, _ = runtime.step(action)
            steps += 1
            clipped += count
            if success or terminated or truncated:
                break
        if success or terminated or truncated:
            break
    return dict(success=bool(success), terminated=bool(terminated), truncated=bool(truncated),
                steps=steps, clipped_coordinates=clipped, hold_steps=0,
                inference_requests=requests, inference_latencies=latencies,
                execution_mode=f'synchronous_delta_h{action_horizon}_execute{execute}_v1', execution_contract_sha256=CONTRACT_SHA256)
