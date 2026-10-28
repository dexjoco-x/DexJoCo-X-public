#!/usr/bin/env python3
"""Measure saved-action sensitivity to float32 casting via the policy interface.

This diagnostic changes command precision. It is neither original-action replay
certification nor a learned-policy evaluation. Dataset files are never changed.
Use the reference environment described in docs/REPLAY_REPAIR.md.
"""
import argparse
import ctypes
import gc
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from audit_native_replay import environment_fingerprint, sha256
from audit_dexlatent_validation_replay import commands, initial_error
from episode_validation import load_validated_episode
from reference_environment import reference_mismatches
from runtime import Runtime, CONTRACTS


def probe(job):
    bundle, row = job
    result = {key: row[key] for key in ('task', 'hand', 'scene_id', 'seed', 'tensor_sha256')}
    runtime = None
    try:
        states, saved = load_validated_episode(bundle, row, CONTRACTS[row['task']+'/'+row['hand']])
        actions = saved.astype(np.float32)
        runtime = Runtime(row['task'], row['hand'], row['seed'], 'arm40_v1', 'native', False)
        error = initial_error(runtime, states[0])
        result.update(initial_error=error, saved_action_dtype=str(saved.dtype))
        if not np.isfinite(error) or error > 2e-5:
            raise ValueError('Initial robot state does not match the recording')
        first, clipped = None, 0
        for index, action in enumerate(actions):
            success, terminated, truncated, count, info = runtime.step(runtime.encode_action(commands(runtime, action)))
            clipped += count
            if success and first is None:
                first = index
            if terminated or truncated:
                break
        result.update(passed=first is not None, first_success_step=first,
                      executed_steps=index+1, saved_steps=len(actions),
                      clipped_coordinates=clipped,
                      stop_reason='terminated' if terminated else 'truncated' if truncated else 'actions_exhausted',
                      error=None)
    except Exception as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}')
    finally:
        if runtime is not None:
            runtime.close()
        gc.collect()
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim(0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    for key in ('task', 'hand', 'scene'):
        parser.add_argument('--'+key, action='append')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.release if (args.release/'manifests').is_dir() else args.release/'bundle'
    manifests = sorted((bundle/'manifests').glob('*.json'))
    rows = [row for path in manifests for row in json.loads(path.read_text())
            if (not args.task or row['task'] in args.task)
            and (not args.hand or row['hand'] in args.hand)
            and (not args.scene or row['scene_id'] in args.scene)]
    if not rows or args.workers < 1:
        parser.error('Select at least one episode and a positive worker count')
    keys = [(r['task'], r['hand'], r['scene_id']) for r in rows]
    if len(set(keys)) != len(keys):
        parser.error('Duplicate episode keys in manifests')
    environment = environment_fingerprint()
    mismatches = reference_mismatches(environment)
    if mismatches:
        parser.error(f'Use the reference environment: {mismatches}')
    progress = args.output.with_suffix('.episodes.jsonl')
    if args.output.exists() or progress.exists():
        parser.error('Choose a new output path to preserve previous results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with progress.open('x') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(probe, (bundle, row)) for row in rows]):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            print(f'{len(records)}/{len(rows)} {record["task"]}/{record["hand"]}/{record["scene_id"]}: {record["passed"]}', flush=True)
    records.sort(key=lambda row: (row['task'], row['hand'], row['scene_id']))
    report = dict(diagnostic_only=True, mode='float32_cast_policy_interface',
                  episodes=len(records), successes=sum(r['passed'] for r in records),
                  execution_errors=sum(r['error'] is not None for r in records),
                  script_sha256=sha256(__file__), environment=environment,
                  manifest_sha256={path.name: sha256(path) for path in manifests}, records=records)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('diagnostic_only', 'mode', 'episodes', 'successes', 'execution_errors')}, indent=2))
    return 0 if all(r['passed'] for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
