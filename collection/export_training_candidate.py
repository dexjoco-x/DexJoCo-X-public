"""Export synchronized observations during an independent saved-label replay."""
import os
os.environ.update(OPENBLAS_CORETYPE='Haswell', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
import argparse
import gc
import ctypes
import hashlib
import json
import sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
from runtime import Runtime
from audit_dexlatent_validation_replay import commands
from collection.training_action_codec import TrainingActionCodec
from collection.training_rollout import PHYSICS, robot_state, state_error, physical_hash
from collection.collect_training_labels import jsonable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats_sha(stats):
    return hashlib.sha256(json.dumps(stats, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def export(source, source_bundle, row, output_bundle):
    source, source_bundle, output_bundle = map(Path, (source, source_bundle, output_bundle))
    meta = json.loads((source / 'collection.json').read_text())
    if not meta.get('task_success') or not meta.get('motion_quality_pass'):
        raise ValueError('Candidate must complete the task and pass motion checks')
    dest = output_bundle / 'episodes' / row['task'] / row['hand'] / row['scene_id']
    if dest.exists(): raise FileExistsError(dest)
    with np.load(source / 'candidate.npz', allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    with np.load(source_bundle / row['tensor'], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    contract = json.loads((source_bundle / row['contract']).read_text())
    codec = TrainingActionCodec(meta['action_statistics'], contract, saved['action_mask'][0])
    runtime = Runtime(row['task'], row['hand'], row['seed'], 'arm40_v1', 'native', True)
    writers = {}; drift = 0.; state_drift = 0.; first = None; trace = hashlib.sha256()
    dest.mkdir(parents=True)
    try:
        for view in row['videos']:
            writers[view] = imageio.get_writer(str(dest / (view + '.mp4')), fps=20, codec='libx264',
                quality=None, pixelformat='yuv420p', macro_block_size=None,
                ffmpeg_params=['-crf', '16', '-preset', 'fast', '-threads', '1'])
        for i, label in enumerate(saved['labels']):
            action = codec.decode_labels(label)
            np.testing.assert_array_equal(action, saved['action'][i])
            for name in PHYSICS:
                actual = np.asarray(getattr(runtime.d, name))
                if actual.size: drift = max(drift, float(np.max(np.abs(actual - saved[name][i]))))
            state_drift = max(state_drift, state_error(robot_state(runtime), saved['state'][i], saved['state_mask'][i]))
            obs = runtime.observation()
            for view, writer in writers.items(): writer.append_data(obs['base' if view == 'front' else view])
            success, term, trunc, clipped, info = runtime.step(runtime.encode_action(commands(runtime, action)))
            assert clipped == 0
            trace.update(physical_hash(runtime).encode())
            if success and first is None: first = i
            if term or trunc:
                if i + 1 != len(saved['labels']): raise ValueError('Rendering replay ended before saved labels')
                break
        if first is None or drift != 0 or state_drift > 1e-12:
            raise ValueError(f'Independent label/render replay failed: success={first}, physics={drift}, state={state_drift}')
    finally:
        for writer in writers.values(): writer.close()
        runtime.close()
    n = len(saved['labels'])
    arrays.update(state=saved['state'].astype(np.float32), action=saved['action'], training_labels=saved['labels'])
    for key in ('state_mask', 'action_mask', 'arm_mask'):
        arrays[key] = np.broadcast_to(arrays[key][0], (n, *arrays[key].shape[1:])).copy()
    arrays['timestamp'] = np.arange(n, dtype=np.float64) / 20
    np.savez_compressed(dest / 'native.npz', **arrays)
    (dest / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    videos = {}
    for view in writers:
        path = dest / (view + '.mp4')
        reader = imageio.get_reader(str(path))
        try: count = sum(1 for _ in reader)
        finally: reader.close()
        if count != n: raise ValueError('Video frame count does not match saved labels')
        videos[view] = dict(path=str(path.relative_to(output_bundle)), frames=n, effective_frames=n, fps=20, sha256=sha(path), bytes=path.stat().st_size)
    evidence = dict(task=row['task'], hand=row['hand'], scene_id=row['scene_id'], seed=row['seed'],
        parent_tensor_sha256=row['tensor_sha256'], candidate_sha256=sha(source / 'candidate.npz'),
        normalization_version=meta['normalization_version'], training_statistics_sha256=stats_sha(meta['action_statistics']),
        source_method=meta['source_method'], physical_success=True, first_success_step=first,
        independent_max_physics_error=drift, independent_max_state_error=state_drift,
        physical_trace_sha256=trace.hexdigest(), model_binding=runtime.model_binding,
        pre_action_rgb=True, videos_fully_decoded=True, steps=n, private_physics_excluded=True,
        release_ready=False)
    evidence['source_steps'] = row['steps']
    parameter_keys = ('gain', 'terminal_hold_budget', 'extra_collected_steps', 'terminal_z_offset_m',
                      'rounding_variant', 'label_max_abs_change', 'original_reference_steps',
                      'maximum_extra_strokes', 'stroke_amplitude_m', 'stroke_lower_offset_m',
                      'stroke_down_steps', 'world_targets_from', 'object_pose_control_input')
    evidence['collection_parameters'] = {key: meta[key] for key in parameter_keys if key in meta}
    evpath = output_bundle / 'evidence/training_affine_v2' / row['task'] / row['hand'] / (row['scene_id'] + '.json')
    evpath.parent.mkdir(parents=True, exist_ok=True)
    evpath.write_text(json.dumps(evidence, indent=2, default=jsonable) + '\n')
    newrow = {k: row[k] for k in ('task', 'hand', 'scene_id', 'seed', 'split', 'tensor', 'contract', 'camera_profile')}
    newrow.update(steps=n, source_steps=n, tensor_sha256=sha(dest / 'native.npz'), contract_sha256=sha(dest / 'contract.json'),
        videos=videos, revision='training_affine_v2_candidate', parent_revision=row['revision'], parent_tensor_sha256=row['tensor_sha256'],
        camera_audit_path=str(evpath.relative_to(output_bundle)), camera_audit_sha256=sha(evpath),
        training_statistics_sha256=evidence['training_statistics_sha256'], training_normalization_version=meta['normalization_version'],
        source='collection/training_affine_v2/' + '/'.join(row[k] for k in ('task', 'hand', 'scene_id')))
    if row.get('mount_revision'): newrow['mount_revision'] = row['mount_revision']
    result = dict(row=newrow, evidence=evidence, release_ready=False)
    (dest / 'export.json').write_text(json.dumps(result, indent=2) + '\n')
    gc.collect(); ctypes.CDLL(None).malloc_trim(0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-bundle', type=Path, required=True)
    p.add_argument('--output-bundle', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.source / 'collection.json').read_text())
    row = next(r for r in json.loads((args.source_bundle / 'manifests' / (meta['task'] + '.json')).read_text()) if r['hand'] == meta['hand'] and r['scene_id'] == meta['scene_id'])
    result = export(args.source, args.source_bundle, row, args.output_bundle)
    print(json.dumps(result['evidence'], indent=2))


if __name__ == '__main__': main()
