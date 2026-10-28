"""New physical demonstrations executed through frozen float32 training labels.

Targets come from a successful Raw reference rollout. No simulator state is
restored during collection or independent replay. Failed attempts are retained.
"""
import os
os.environ.update(OPENBLAS_CORETYPE='Haswell', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
from runtime import Runtime, CONTRACTS
from episode_validation import load_validated_episode
from audit_dexlatent_validation_replay import commands, step
from audit_native_replay import environment_fingerprint
from reference_environment import reference_mismatches
from collection.training_action_codec import TrainingActionCodec
from collection.training_rollout import PHYSICS, snapshot, robot_state, state_error, physical_hash


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return str(value)


def target_path(bundle, row, cache):
    """Recover requested world targets using the accepted unmodified source."""
    path = Path(cache) / row['task'] / row['hand'] / (row['scene_id'] + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert str(z['source_sha256']) == row['tensor_sha256']
            return z['action'].copy(), z['positions'].copy(), z['rotations'].copy()
    _, original = load_validated_episode(bundle, row, CONTRACTS[row['task'] + '/' + row['hand']])
    runtime = Runtime(row['task'], row['hand'], row['seed'], 'arm40_v1', 'native', False)
    positions = []; rotations = []; first = None
    try:
        for i, action in enumerate(original):
            pos = np.zeros((2, 3)); rot = np.tile(np.eye(3), (2, 1, 1))
            for arm, side in enumerate(('right', 'left')):
                if side not in runtime.e._native_ctrl_ids: continue
                pose = runtime.e.tcp(side)
                pos[arm] = pose[:3, 3] + action[arm, :3]
                rot[arm] = Rotation.from_rotvec(action[arm, 3:6]).as_matrix() @ pose[:3, :3]
            positions.append(pos); rotations.append(rot)
            info, term, trunc = step(runtime, commands(runtime, action))
            if info.get('succeed') and first is None: first = i
            if term or trunc: break
        if first is None: raise ValueError('Reference source failed its original saved-action replay')
    finally: runtime.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, action=original[:i + 1], positions=positions, rotations=rotations, source_sha256=row['tensor_sha256'])
    return original[:i + 1], np.asarray(positions), np.asarray(rotations)


def verify_candidate(directory, contract):
    """Independent reset and replay from labels read back from disk."""
    directory = Path(directory)
    meta = json.loads((directory / 'collection.json').read_text())
    with np.load(directory / 'candidate.npz', allow_pickle=False) as file:
        saved = {k: file[k].copy() for k in file.files}
    codec = TrainingActionCodec(meta['action_statistics'], contract, saved['action_mask'][0])
    runtime = Runtime(meta['task'], meta['hand'], meta['seed'], 'arm40_v1', 'native', False)
    first = None; drift = 0.; states_drift = 0.; trace = hashlib.sha256(); decoded_equal = True
    try:
        for i, label in enumerate(saved['labels']):
            action = codec.decode_labels(label)
            decoded_equal &= np.array_equal(action, saved['action'][i])
            for name in PHYSICS:
                current = np.asarray(getattr(runtime.d, name))
                if current.size: drift = max(drift, float(np.max(np.abs(current - saved[name][i]))))
            states_drift = max(states_drift, state_error(robot_state(runtime), saved['state'][i], saved['state_mask'][i]))
            success, term, trunc, clipped, info = runtime.step(runtime.encode_action(commands(runtime, action)))
            assert clipped == 0
            trace.update(physical_hash(runtime).encode())
            if success and first is None: first = i
            if term or trunc: break
        return dict(independent_replay_checked=True, independent_replay_success=first is not None,
            first_replay_success_step=first, independent_replay_max_physics_error=drift,
            independent_replay_max_state_error=states_drift, decoded_commands_identical=bool(decoded_equal),
            independent_physical_trace_sha256=trace.hexdigest(), independent_executed_steps=i + 1,
            independent_replay_passed=first is not None and drift == 0 and states_drift < 1e-12 and decoded_equal and i + 1 == len(saved['labels']))
    finally: runtime.close()


def collect(row, bundle, statistics, directory, cache, gain=1., tail=40, terminal_z_offset=0.):
    directory, bundle = Path(directory), Path(bundle)
    if directory.exists(): raise FileExistsError(directory)
    if terminal_z_offset and row['task'] not in ('hammer_nail', 'pick_bucket'):
        raise ValueError('Terminal vertical motion is defined only for hammer/bucket collection')
    if tail < 0 or not np.isfinite(gain) or gain <= 0 or not np.isfinite(terminal_z_offset):
        raise ValueError('Invalid collection parameters')
    contract = json.loads((bundle / row['contract']).read_text())
    with np.load(bundle / row['tensor'], allow_pickle=False) as z:
        action_mask, state_mask = z['action_mask'][0].copy(), z['state_mask'][0].copy()
    codec = TrainingActionCodec(statistics, contract, action_mask)
    reference, positions, rotations = target_path(bundle, row, cache)
    runtime = Runtime(row['task'], row['hand'], row['seed'], 'arm40_v1', 'native', False)
    states = []; actions = []; labels = []; private = []; strikes = []; first = None; requested_clips = 0
    trace = hashlib.sha256()
    try:
        for i in range(len(reference) + tail):
            j = min(i, len(reference) - 1)
            requested = reference[j].copy()
            for arm, side in enumerate(('right', 'left')):
                if side not in runtime.e._native_ctrl_ids: continue
                pose = runtime.e.tcp(side)
                position = positions[j, arm].copy()
                if i >= len(reference): position[2] += terminal_z_offset * (i - len(reference) + 1) / tail
                requested[arm, :6] = gain * np.r_[position - pose[:3, 3], Rotation.from_matrix(rotations[j, arm] @ pose[:3, :3].T).as_rotvec()]
            label, action, clipped = codec.encode_command(requested)
            requested_clips += clipped
            labels.append(label); actions.append(action); states.append(robot_state(runtime)); private.append(snapshot(runtime))
            depth = runtime.e.get_nail_depth() if row['task'] == 'hammer_nail' else None
            success, term, trunc, clipped, info = runtime.step(runtime.encode_action(commands(runtime, action)))
            assert clipped == 0
            if depth is not None and runtime.e.get_nail_depth() > depth + 1e-10:
                strikes.append(float(np.degrees(np.arccos(np.clip(-runtime.d.geom('face').xmat.reshape(3, 3)[2, 2], -1, 1)))))
            trace.update(physical_hash(runtime).encode())
            if success and first is None: first = i
            if term or trunc: break
        quality = (bool(strikes) and max(strikes) <= 35) if row['task'] == 'hammer_nail' else True
        meta = {k: row[k] for k in ('task', 'hand', 'scene_id', 'seed')}
        meta.update(source_tensor_sha256=row['tensor_sha256'], source_method='new_collection_tracking_accepted_world_targets',
            action_statistics=statistics, normalization_version=statistics['normalization_version'], gain=gain,
            terminal_hold_budget=tail, extra_collected_steps=max(0, len(labels) - len(reference)),
            terminal_z_offset_m=terminal_z_offset,
            saved_steps=len(labels), task_success=first is not None, first_success_step=first,
            motion_quality_pass=quality, max_strike_tilt_deg=max(strikes, default=None),
            requested_coordinates_limited_to_physical_range=requested_clips,
            physical_trace_sha256=trace.hexdigest(), model_binding=runtime.model_binding,
            new_physical_collection=True, rgb_generated=False, release_ready=False, final_info=info)
    finally: runtime.close()
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'candidate.npz', state=states, action=actions, labels=labels,
        state_mask=np.broadcast_to(state_mask, (len(labels), *state_mask.shape)),
        action_mask=np.broadcast_to(action_mask, (len(labels), *action_mask.shape)),
        **{k: np.asarray([r[k] for r in private]) for k in PHYSICS})
    (directory / 'collection.json').write_text(json.dumps(meta, indent=2, default=jsonable) + '\n')
    meta.update(verify_candidate(directory, contract))
    meta['accepted_for_observation_export'] = bool(meta['task_success'] and quality and meta['independent_replay_passed'])
    (directory / 'collection.json').write_text(json.dumps(meta, indent=2, default=jsonable) + '\n')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--statistics-file', type=Path, required=True)
    for key in ('task', 'hand', 'scene'): p.add_argument('--' + key, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-cache', type=Path, required=True)
    p.add_argument('--gain', type=float, default=1.)
    p.add_argument('--tail', type=int, default=40)
    p.add_argument('--terminal-z-offset', type=float, default=0.)
    args = p.parse_args()
    mismatch = reference_mismatches(environment_fingerprint())
    if mismatch: raise RuntimeError(mismatch)
    bundle = args.data_root / 'bundle'
    row = next(r for r in json.loads((bundle / 'manifests' / (args.task + '.json')).read_text()) if r['hand'] == args.hand and r['scene_id'] == args.scene)
    stats = json.loads(args.statistics_file.read_text())['groups'][args.task + '/' + args.hand]['action']
    result = collect(row, bundle, stats, args.output, args.target_cache, args.gain, args.tail, args.terminal_z_offset)
    print(json.dumps({k: v for k, v in result.items() if k not in ('action_statistics', 'final_info')}, indent=2, default=jsonable))
    return 0 if result['accepted_for_observation_export'] else 1


if __name__ == '__main__': raise SystemExit(main())
