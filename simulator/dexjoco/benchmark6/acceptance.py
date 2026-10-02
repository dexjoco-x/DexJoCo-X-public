"""Complete-count, content integrity and scene/embodiment split acceptance."""
import json
from pathlib import Path
from .audit import digest
from .protocol import load_frozen
from .provenance import file_hash
from .success_policy import required_success_steps, policy_matches


def split_manifests(items,hands):
    standard={split:[x for x in items if x['split']==split] for split in ('train','validation','test')}
    scene_sets={split:{x['scene_id'] for x in rows} for split,rows in standard.items()}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        if scene_sets[a]&scene_sets[b]:raise ValueError('Paired scene leaks between splits')
    loho={}
    for hand in hands:
        loho[hand]=dict(
            train=[x for x in standard['train'] if x['hand']!=hand],
            validation=[x for x in standard['validation'] if x['hand']!=hand],
            test_unseen_hand_unseen_scene=[x for x in standard['test'] if x['hand']==hand],
            test_unseen_hand_seen_scene=[x for x in standard['train'] if x['hand']==hand])
    return dict(joint_training=standard,leave_one_hand_out=loho,
        limitations='Splits are scene-disjoint, not source-template-disjoint. '
                    'Online VLA evaluation still requires executing the learned policy.')


def audit_release(protocol_path,root):
    protocol=load_frozen(protocol_path);root=Path(root);issues=[];items=[];counts={}
    for task in protocol['tasks']:
        for hand in protocol['hands']:
            counts[task+'/'+hand]=0
            for scene in protocol['scenes']:
                directory=root/'accepted'/task/hand/scene['scene_id'];audit_path=directory/'audit.json'
                if not audit_path.exists():issues.append(f'Missing episode {task}/{hand}/{scene["scene_id"]}');continue
                try:
                    a=json.loads(audit_path.read_text());m=json.loads((directory/'metadata.json').read_text())
                except (OSError,ValueError) as exc:
                    issues.append(f'Unreadable episode {directory}: {exc!r}');continue
                expected=(task,hand,scene['seed'],protocol['sha256'])
                actual=(a.get('task'),a.get('hand'),a.get('seed'),a.get('protocol_sha256'))
                actual_meta=tuple(m.get(k) for k in ('task','hand','seed','protocol_sha256'))
                if actual!=expected or actual_meta!=expected or m.get('scene')!=scene or not m.get('training_eligible'):
                    issues.append(f'Wrong scene/hand/protocol label: {directory}');continue
                if a.get('code_sha256')!=digest(protocol['code']) or a.get('model_manifest_sha256')!=protocol['models'][task+'/'+hand]['sha256']:
                    issues.append(f'Wrong code/model provenance: {directory}');continue
                if not all(a.get(k) for k in ('passed','training_eligible','rgb_verified')):
                    issues.append(f'Incomplete physical/RGB verification: {directory}');continue
                if not a.get('camera_quality',{}).get('passed'):
                    issues.append(f'Missing wrist-camera obstruction check: {directory}');continue
                errors=a.get('action_replay_max_error',{})
                if a.get('stable_success_steps',0)<required_success_steps(task) or not policy_matches(task,a) or not policy_matches(task,m) or not errors or max(errors.values())>1e-8:
                    issues.append(f'Failed physics acceptance: {directory}');continue
                if a.get('initial_scene_sha256')!=protocol['paired_scene_sha256'][task][scene['scene_id']]:
                    issues.append(f'Unpaired reset: {directory}');continue
                if not a.get('files'):issues.append(f'Missing file hashes: {directory}');continue
                actual_files={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=audit_path}
                if actual_files!=set(a['files']):
                    issues.append(f'Untracked/missing episode files: {directory}');continue
                corrupted=[name for name,value in a['files'].items()
                    if not (directory/name).is_file() or file_hash(directory/name)!=value]
                if corrupted:issues.append(f'Corrupt episode {directory}: {corrupted[:3]}');continue
                counts[task+'/'+hand]+=1
                items.append(dict(task=task,hand=hand,scene_id=scene['scene_id'],seed=scene['seed'],
                    split=scene['split'],path=str(directory.relative_to(root)),steps=a['steps'],
                    audit_sha256=file_hash(audit_path)))
    expected_paths={str(root/'accepted'/t/h/s['scene_id']) for t in protocol['tasks']
                    for h in protocol['hands'] for s in protocol['scenes']}
    extras=[str(p.parent) for p in (root/'accepted').glob('*/*/*/audit.json') if str(p.parent) not in expected_paths]
    if extras:issues.append(f'Unexpected/duplicate accepted paths: {extras[:5]}')
    scope=protocol.get('scope','suite')
    report=dict(passed=not issues,scope=scope,benchmark_complete=not issues and scope=='suite',
                benchmark_required_count=len(protocol.get('benchmark_tasks',protocol['tasks']))*len(protocol['hands'])*50,
                protocol_sha256=protocol['sha256'],accepted_count=len(items),
                required_count=len(protocol['tasks'])*len(protocol['hands'])*50,counts=counts,issues=issues)
    root.mkdir(parents=True,exist_ok=True)
    (root/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['passed']:
        splits=split_manifests(items,protocol['hands'])
        (root/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
        (root/'episodes.json').write_text(json.dumps(items,indent=2)+'\n')
    return report
