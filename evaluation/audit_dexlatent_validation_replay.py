#!/usr/bin/env python3
"""Replay all frozen validation demonstrations before admitting DexLatent.

Each episode is run twice from the same seed: once with the original native
commands and once after the exact sided DexLatent action encode/decode bridge.
This is a codec acceptance test, not learned-policy evaluation.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent


def initial_error(runtime, state):
    errors = []
    for arm, side in enumerate(("right", "left")):
        if side not in runtime.e._native_ctrl_ids:
            continue
        pose = runtime.e.tcp(side)
        expected_rot = Rotation.from_quat(state[arm, 3:7][[1, 2, 3, 0]]).as_matrix()
        errors.append(float(np.max(np.abs(pose[:3, 3] - state[arm, :3]))))
        errors.append(float(np.max(np.abs(pose[:3, :3] - expected_rot))))
        q = runtime.d.qpos[runtime.e._full_hand_qpos[side]]
        errors.append(float(np.max(np.abs(q - state[arm, 7:7 + len(q)]))))
    return max(errors, default=0.0)


def commands(runtime, action):
    result = {}
    for arm, side in enumerate(("right", "left")):
        if side in runtime.e._native_ctrl_ids:
            n = len(runtime.e._native_ctrl_ids[side])
            result[side] = np.asarray(action[arm, :6 + n], dtype=np.float64)
    return result


def step(runtime, value):
    payload = value if len(value) > 1 or runtime.task != "single_hand_hanoi" else value["right"]
    result = runtime.e.step(payload)
    return result[-1], bool(result[2]), bool(result[3])


def run_episode(job):
    import sys
    sys.path.insert(0, str(HERE))
    from runtime import Runtime

    bundle, row = Path(job[0]), job[1]
    with np.load(bundle / row["tensor"]) as data:
        states = data["state"]
        actions = data["action"]
    result = {"task": row["task"], "hand": row["hand"], "scene": row["scene_id"],
              "seed": row["seed"], "steps": len(actions)}
    for representation in ("native", "dexlatent"):
        runtime = Runtime(row["task"], row["hand"], row["seed"], "arm40_v1",
                          representation=representation, render=False)
        first_success = None
        max_error = 0.0
        try:
            init = initial_error(runtime, states[0])
            if init > 2e-5:
                raise ValueError((row["task"], row["hand"], row["scene_id"], "initial", init))
            for index, raw in enumerate(actions):
                native = commands(runtime, raw)
                if representation == "dexlatent":
                    encoded = runtime.encode_action(native)
                    # Match the formal evaluator: decoded actuator targets are
                    # bounded to the environment contract before execution.
                    reconstructed, _ = runtime.decode_action(encoded, clip=True)
                    max_error = max(max_error, *(float(np.max(np.abs(reconstructed[s] - native[s])))
                                                 for s in native))
                    native = reconstructed
                info, terminated, truncated = step(runtime, native)
                if bool(info.get("succeed", False)) and first_success is None:
                    first_success = index
                if terminated or truncated:
                    break
            result[representation] = {"passed": first_success is not None,
                                      "first_success_step": first_success,
                                      "maximum_action_error": max_error}
        finally:
            runtime.close()
    result["preserved"] = result["native"]["passed"] == result["dexlatent"]["passed"]
    return result


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.checkpoint:
        os.environ['BENCHMARK7_DEXLATENT_CHECKPOINT'] = str(args.checkpoint.resolve())
        os.environ['BENCHMARK7_DEXLATENT_SHA256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    rows = []
    for path in sorted((args.release / "bundle/manifests").glob("*.json")):
        rows.extend(x for x in json.loads(path.read_text()) if x["split"] == "validation")
    assert len(rows) == 210
    jobs = [(args.release / "bundle", row) for row in rows]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_episode, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(record["task"], record["hand"], record["scene"],
                  record["native"]["passed"], record["dexlatent"]["passed"], flush=True)
    records.sort(key=lambda x: (x["task"], x["hand"], x["scene"]))
    native = sum(x["native"]["passed"] for x in records)
    latent = sum(x["dexlatent"]["passed"] for x in records)
    regressions = sum(x["native"]["passed"] and not x["dexlatent"]["passed"] for x in records)
    # Some delivered tensors are shorter than their accepted source trajectory
    # because video-alignment trimming removed the terminal success tail.  Use
    # replayed native successes as the paired denominator; the codec must not
    # regress any of those episodes.
    report = {"passed": regressions == 0,
              "scope": "all frozen validation demonstrations; no test split; codec replay only",
              "episodes": 210, "checkpoint": os.environ.get('BENCHMARK7_DEXLATENT_CHECKPOINT', 'package default'),
              "checkpoint_sha256": os.environ.get('BENCHMARK7_DEXLATENT_SHA256'), "native_successes": native,
              "dexlatent_successes": latent, "native_to_dexlatent_regressions": regressions,
              "records": records}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
