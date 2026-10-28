"""Replay actual exported training labels through their versioned decoder.

Task success and agreement with source recorded states are reported separately.
Optional candidate snapshots are new executions, not a certified dataset export.
"""
import os
os.environ.update(OPENBLAS_CORETYPE='Haswell', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
import argparse
import ctypes
import gc
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
from runtime import Runtime, CONTRACTS
from audit_native_replay import environment_fingerprint
from reference_environment import reference_mismatches
from episode_validation import load_validated_episode
from audit_dexlatent_validation_replay import commands, initial_error
from dataset.tools.adapters import encode
from collection.training_action_codec import TrainingActionCodec
from collection.training_rollout import PHYSICS, snapshot, robot_state, state_error, physical_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return str(value)


def run(job):
    row, group, episode, data_root, candidate_root = job
    root = Path(data_root); runtime = None; started = time.time()
    result = {k: row[k] for k in ('task', 'hand', 'scene_id', 'seed', 'tensor_sha256')}
    try:
        bundle = root / 'bundle'
        expected, raw = load_validated_episode(bundle, row, CONTRACTS[row['task'] + '/' + row['hand']])
        with np.load(bundle / row['tensor'], allow_pickle=False) as z:
            state_mask, action_mask = z['state_mask'].copy(), z['action_mask'].copy()
        assert np.all(action_mask == action_mask[0])
        folder = root / group['path']
        stats_path = folder / 'meta/raw_train_statistics.json'
        stats = json.loads(stats_path.read_text())['action']
        contract = json.loads((bundle / row['contract']).read_text())
        codec = TrainingActionCodec(stats, contract, action_mask[0])
        path = folder / 'data/chunk-000' / f"episode_{episode['episode_index']:06d}.parquet"
        df = pd.read_parquet(path)
        labels = np.stack(df.action)
        assert len(labels) == row['steps'] == episode['length']
        assert labels.dtype == np.float32 and labels.shape[1] == 200
        _, mask = encode(raw, action_mask, 'action', 'beingh05')
        np.testing.assert_array_equal(np.stack(df['action.valid_mask']), mask)
        np.testing.assert_array_equal(df.frame_index.to_numpy(), np.arange(len(labels)))
        runtime = Runtime(row['task'], row['hand'], row['seed'], 'arm40_v1', 'native', False)
        start_error = initial_error(runtime, expected[0]); assert start_error <= 2e-5
        first = None; drift = 0.; maximum_action_error = 0.; clips = 0; strikes = []
        states = []; actions = []; private = []; trace = hashlib.sha256()
        for i, label in enumerate(labels):
            action = codec.decode_labels(label)
            state = robot_state(runtime)
            drift = max(drift, state_error(state, expected[i], state_mask[i]))
            maximum_action_error = max(maximum_action_error, float(np.max(np.abs(action - raw[i]))))
            if candidate_root:
                states.append(state); actions.append(action); private.append(snapshot(runtime))
            depth = runtime.e.get_nail_depth() if row['task'] == 'hammer_nail' else None
            success, term, trunc, clipped, info = runtime.step(runtime.encode_action(commands(runtime, action)))
            clips += clipped
            if depth is not None and runtime.e.get_nail_depth() > depth + 1e-10:
                strikes.append(float(np.degrees(np.arccos(np.clip(-runtime.d.geom('face').xmat.reshape(3, 3)[2, 2], -1, 1)))))
            trace.update(physical_hash(runtime).encode())
            if success and first is None: first = i
            if term or trunc: break
        assert clips == 0, 'Final float32 labels must decode inside physical limits'
        quality = (bool(strikes) and max(strikes) <= 35) if row['task'] == 'hammer_nail' else True
        result.update(task_success=first is not None, first_success_step=first, executed_steps=i + 1,
            saved_steps=len(labels), recorded_state_fidelity_passed=drift <= 2e-5,
            passed=first is not None and drift <= 2e-5 and quality,
            max_source_state_error=drift, max_action_error=maximum_action_error,
            initial_error=start_error, runtime_clipped_coordinates=clips, physical_trace_sha256=trace.hexdigest(),
            motion_quality_pass=quality, max_strike_tilt_deg=max(strikes, default=None),
            stop_reason='terminated' if term else 'truncated' if trunc else 'actions_exhausted',
            parquet_sha256=sha(path), statistics_sha256=sha(stats_path), final_info=info, error=None)
        if candidate_root and first is not None and quality:
            dest = Path(candidate_root) / row['task'] / row['hand'] / row['scene_id']
            dest.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(dest / 'candidate.npz', state=states, action=actions, labels=labels[:i + 1],
                state_mask=state_mask[:i + 1], action_mask=action_mask[:i + 1],
                **{k: np.asarray([r[k] for r in private]) for k in PHYSICS})
            candidate = dict(result, source_method='continuous_execution_of_exported_float32_labels',
                independent_replay_checked=False, rgb_generated=False, release_ready=False,
                normalization_version=stats['normalization_version'], action_statistics=stats)
            (dest / 'collection.json').write_text(json.dumps(candidate, indent=2, default=jsonable) + '\n')
            result['candidate'] = str(dest)
    except Exception as exc:
        result.update(passed=False, task_success=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        if runtime is not None: runtime.close()
        runtime = None; gc.collect(); ctypes.CDLL(None).malloc_trim(0)
    result['elapsed_seconds'] = time.time() - started
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--prepared-name', default='beingh05_affine_v3')
    p.add_argument('--workers', default=4, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--candidate-dir', type=Path)
    for key in ('task', 'hand', 'scene'): p.add_argument('--' + key)
    args = p.parse_args()
    fingerprint = environment_fingerprint()
    mismatch = reference_mismatches(fingerprint)
    if mismatch: raise RuntimeError(f'Reference environment mismatch: {mismatch}')
    bundle = args.data_root / 'bundle'
    lookup = {(r['task'], r['hand'], r['scene_id']): r for path in (bundle / 'manifests').glob('*.json') for r in json.loads(path.read_text())}
    groups = json.loads((args.data_root / 'prepared' / (args.prepared_name + '_datasets.json')).read_text())
    jobs = []
    for group in groups:
        folder = args.data_root / group['path']
        for line in (folder / 'meta/episodes.jsonl').read_text().splitlines():
            episode = json.loads(line)
            row = lookup[group['task'], group['hand'], episode['source_scene']]
            if any(value and row[key] != value for key, value in [('task', args.task), ('hand', args.hand), ('scene_id', args.scene)]): continue
            jobs.append((row, group, episode, str(args.data_root), str(args.candidate_dir) if args.candidate_dir else None))
    if not jobs: raise ValueError('No episodes selected')
    jobs.sort(key=lambda j: (j[0]['scene_id'], j[0]['hand'], j[0]['task']))
    progress = args.output.with_suffix('.episodes.jsonl')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if progress.exists(): raise FileExistsError('Use a new report path for an independent verification')
    records = []
    with progress.open('x') as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            r = future.result(); records.append(r)
            log.write(json.dumps(r, default=jsonable) + '\n'); log.flush()
            print(f"{len(records)}/{len(jobs)} {r['task']}/{r['hand']}/{r['scene_id']}: task={r['task_success']} states={r.get('recorded_state_fidelity_passed')} {r['error'] or ''}", flush=True)
    report = dict(episodes=len(records), task_successes=sum(r['task_success'] for r in records),
        source_state_matches=sum(r.get('recorded_state_fidelity_passed', False) for r in records),
        errors=sum(r['error'] is not None for r in records), passed=all(r['passed'] for r in records),
        scope='Actual Parquet labels, fixed decoder, continuous policy interface execution; no trained model.',
        environment=fingerprint, script_sha256=sha(__file__), records=records)
    args.output.write_text(json.dumps(report, indent=2, default=jsonable) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'environment')}, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
