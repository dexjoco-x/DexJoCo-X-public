"""Rescale collected v2 labels without changing any executed physical command."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from adapters import encode, decode
from data_io import link_or_copy, sha, write
from normalization import AFFINE, AFFINE_V2, version, stabilize_scales, scale_labels, preserve_decoded_scale, denormalize, normalize


def statistics_hash(stats):
    return hashlib.sha256(json.dumps(stats, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main(source, output, statistics_file):
    source, output, statistics_file = map(Path, (source, output, statistics_file))
    if output.exists(): raise FileExistsError('Use a new output bundle for label rescaling')
    frozen = json.loads(statistics_file.read_text())
    previous = frozen['groups']
    if any(version(s) != AFFINE_V2 for group in previous.values() for s in group.values()):
        raise ValueError('Source statistics must be the frozen affine-v2 statistics')
    updated = {key: {kind: stabilize_scales(stats) for kind, stats in group.items()} for key, group in previous.items()}
    output.mkdir(parents=True)
    rows = [r for path in sorted((source / 'manifests').glob('*.json')) for r in json.loads(path.read_text())]
    if not rows: raise ValueError('No source episodes')
    # Adjacent-float collection can contain subnormal labels. Preserve actual
    # decoded values, including tiny zero-centered commands, before freezing scales.
    for row in rows:
        key = row['task'] + '/' + row['hand']
        with np.load(source / row['tensor'], allow_pickle=False) as arrays:
            _, mask = encode(arrays['action'], arrays['action_mask'], 'action', 'beingh05')
            updated[key]['action'] = preserve_decoded_scale(arrays['training_labels'], mask,
                previous[key]['action'], updated[key]['action'])
    output_rows = []; records = []
    for row in rows:
        key = row['task'] + '/' + row['hand']
        old, new = previous[key]['action'], updated[key]['action']
        if row.get('training_normalization_version') != AFFINE_V2 or row.get('training_statistics_sha256') != statistics_hash(old):
            raise ValueError('Source labels and frozen statistics do not match')
        if sha(source / row['tensor']) != row['tensor_sha256']: raise ValueError('Source tensor hash mismatch')
        with np.load(source / row['tensor'], allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in f.files}
        labels = arrays['training_labels']
        _, mask = encode(arrays['action'], arrays['action_mask'], 'action', 'beingh05')
        np.testing.assert_array_equal(decode(denormalize(labels, mask, old), arrays['action_mask'], 'beingh05'), arrays['action'])
        scaled = scale_labels(labels, new, require_label_bits=False)
        restored = decode(denormalize(scaled, mask, new), arrays['action_mask'], 'beingh05')
        np.testing.assert_array_equal(restored.view(np.uint64), arrays['action'].view(np.uint64))
        if not np.isfinite(scaled).all(): raise ValueError('Nonfinite rescaled label')
        arrays['training_labels'] = scaled
        target = output / row['tensor']; target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        shutil.copy2(source / row['contract'], output / row['contract'])
        if sha(output / row['contract']) != row['contract_sha256']: raise ValueError('Contract hash mismatch')
        for video in row['videos'].values():
            if sha(source / video['path']) != video['sha256']: raise ValueError('Source video hash mismatch')
            (output / video['path']).parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(source / video['path'], output / video['path'])
        state, state_mask = encode(arrays['state'], arrays['state_mask'], 'state', 'beingh05')
        normalized_state = normalize(state, state_mask, updated[key]['state'])
        evidence_path = Path('evidence/training_affine_v3') / row['task'] / row['hand'] / (row['scene_id'] + '.json')
        evidence = json.loads((source / row['camera_audit_path']).read_text())
        if sha(source / row['camera_audit_path']) != row['camera_audit_sha256']: raise ValueError('Source evidence hash mismatch')
        evidence.update(collection_normalization_version=AFFINE_V2,
            collection_training_statistics_sha256=statistics_hash(old),
            normalization_version=AFFINE, training_statistics_sha256=statistics_hash(new),
            label_scale_exponents=new['label_scale_exponents'],
            label_bits_preserved=bool(np.array_equal(np.ldexp(scaled.astype(np.float64), np.asarray(new['label_scale_exponents'], np.int32)), labels.astype(np.float64))),
            physical_command_rescaling_exact=True, decoded_commands_bitwise_unchanged=True,
            source_label_tensor_sha256=row['tensor_sha256'], final_training_file_replay_verified=False,
            release_ready=False)
        write(output / evidence_path, evidence)
        newrow = dict(row, tensor_sha256=sha(target), revision='training_affine_v3_candidate',
            training_normalization_version=AFFINE, training_statistics_sha256=statistics_hash(new),
            training_label_parent_tensor_sha256=row['tensor_sha256'],
            camera_audit_path=str(evidence_path), camera_audit_sha256=sha(output / evidence_path))
        write(target.parent / 'export.json', dict(row=newrow, evidence=evidence, release_ready=False))
        output_rows.append(newrow)
        records.append(dict(task=row['task'], hand=row['hand'], scene_id=row['scene_id'], frames=row['steps'],
            decoded_commands_bitwise_unchanged=True,
            maximum_action_label_before=float(np.max(np.abs(labels))),
            maximum_action_label_after=float(np.max(np.abs(scaled))),
            maximum_state_label_after=float(np.max(np.abs(normalized_state)))))
        print(f"{len(records)}/{len(rows)} {key}/{row['scene_id']}: exact commands", flush=True)
    for task in sorted({r['task'] for r in output_rows}):
        write(output / 'manifests' / (task + '.json'), [r for r in output_rows if r['task'] == task])
    frozen.update(groups=updated, normalization_version=AFFINE, label_rescaling_source_statistics_sha256=sha(statistics_file))
    write(output / 'training_statistics.json', frozen)
    catalog = json.loads((source / 'catalog.json').read_text()) if (source / 'catalog.json').exists() else {}
    catalog.update(episodes=len(records), frames=sum(r['frames'] for r in records),
        dataset_version='training-float32-v1-candidate', normalization_version=AFFINE,
        training_statistics_sha256=sha(output / 'training_statistics.json'),
        manifest_sha256={p.name: sha(p) for p in (output / 'manifests').glob('*.json')},
        final_training_file_replay_verified=False, release_ready=False)
    write(output / 'catalog.json', catalog)
    report = dict(passed=True, episodes=len(records), frames=sum(r['frames'] for r in records),
        decoded_commands_bitwise_unchanged=True, normalization_version=AFFINE,
        maximum_action_label=max(r['maximum_action_label_after'] for r in records),
        maximum_state_label=max(r['maximum_state_label_after'] for r in records), records=records)
    write(output / 'label_rescaling_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--statistics-file', required=True, type=Path)
    args = parser.parse_args()
    main(args.source, args.output, args.statistics_file)
