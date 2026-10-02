"""Paired scenes and release gates. A draft is never a frozen benchmark."""
from pathlib import Path
import json
import numpy as np
from . import VERSION, HANDS, CANDIDATE_TASKS
from .audit import digest, nonhand_manifest
from .models import PREFIX
from .success_policy import required_success_steps, success_policy, policy_matches
from .provenance import code_manifest, model_manifest, file_hash


def scene_signature(env):
    m,d=env._model,env._data
    state={}
    for j in range(m.njnt):
        if m.joint(j).name.startswith(PREFIX):continue
        state[m.joint(j).name]=dict(qpos=d.joint(j).qpos.tolist(),qvel=d.joint(j).qvel.tolist())
    for i in range(m.nbody):
        mid=int(m.body_mocapid[i])
        if mid>=0:state['mocap:'+m.body(i).name]=np.r_[d.mocap_pos[mid],d.mocap_quat[mid]].tolist()
    contract={k:v for k,v in env.contract.items() if k!='hand'}
    return digest(dict(model=nonhand_manifest(m),state=state,contract=contract,time=float(d.time)))


def validate_task_set(tasks):
    if len(tasks) not in (5,6) or len(set(tasks))!=len(tasks):raise ValueError('Select 5 or 6 distinct tasks')
    if not set(tasks)<=set(CANDIDATE_TASKS):raise ValueError('Unknown candidate task')
    dual=sum(t.startswith('bimanual_') for t in tasks)
    if abs(dual-(len(tasks)-dual))>1:raise ValueError('Single/dual task counts must be balanced')


def validate_scope(tasks,scope='suite',benchmark_tasks=None):
    if scope=='suite':
        validate_task_set(tasks)
        if benchmark_tasks is not None and list(tasks)!=list(benchmark_tasks):
            raise ValueError('Suite tasks disagree with declared benchmark')
    elif scope=='task_shard':
        if benchmark_tasks is None:raise ValueError('Shard needs the full declared task set')
        validate_task_set(benchmark_tasks)
        if not tasks or len(set(tasks))!=len(tasks) or not set(tasks)<set(benchmark_tasks):
            raise ValueError('Shard must be a nonempty proper subset of the benchmark')
    else:raise ValueError('Unknown release scope')


def planned_scenes(seed=20260909,count=50):
    if count!=50:raise ValueError('Formal protocol requires 50 scenes per task')
    rng=np.random.default_rng(seed)
    seeds=rng.choice(np.arange(100000,200000),size=count,replace=False).tolist()
    # Split BEFORE collection; seed failures may not be silently replaced with
    # easier scenes. The same scene is in the same split for every hand/task.
    return [dict(scene_id=f'scene_{i:03d}',seed=int(s),
                 split='train' if i<40 else 'validation' if i<45 else 'test') for i,s in enumerate(seeds)]


def check_evidence(tasks,evidence):
    required={(task,hand) for task in tasks for hand in HANDS};seen=set();issues=[]
    current_code=digest(code_manifest())
    records=[]
    for directory in evidence:
        path=Path(directory)/'audit.json'
        if not path.exists():issues.append(f'{directory}: no independent RGB/replay audit');continue
        a=json.loads(path.read_text());key=(a.get('task'),a.get('hand'))
        if key not in required:continue
        if not a.get('passed') or not a.get('rgb_verified') or a.get('stable_success_steps',0)<required_success_steps(key[0]) or not policy_matches(key[0],a):
            issues.append(f'{directory}: failed replay/RGB/success gate');continue
        errors=a.get('action_replay_max_error',{})
        if not a.get('camera_quality',{}).get('passed') or not errors or max(errors.values())>1e-8:
            issues.append(f'{directory}: missing camera quality or failed exact replay');continue
        if a.get('version')!=VERSION or a.get('code_sha256')!=current_code:
            issues.append(f'{directory}: stale rules/code version');continue
        if a.get('model_manifest_sha256')!=model_manifest(*key)['sha256']:
            issues.append(f'{directory}: stale model/assets');continue
        # Bind to the actual exported data and videos, not a stand-alone PASS.
        for relative,expected in a.get('files',{}).items():
            actual=Path(directory)/relative
            if not actual.is_file() or file_hash(actual)!=expected:
                issues.append(f'{directory}: changed/missing {relative}');break
        else:
            if not a.get('files'):issues.append(f'{directory}: missing content hashes');continue
            seen.add(key);records.append(dict(path=str(Path(directory).resolve()),audit_sha256=file_hash(path)))
    return records,issues+[f'No current verified witness: {t}/{h}' for t,h in sorted(required-seen)]


def freeze(output,tasks,evidence,seed=20260909,benchmark_tasks=None):
    from .environments import make_env
    scope='task_shard' if benchmark_tasks is not None else 'suite'
    validate_scope(tasks,scope,benchmark_tasks)
    if 'bimanual_unlock_ipad' in tasks:
        from .ipad_task import RULES
        if RULES.status != 'validated':
            raise ValueError('iPad lift/hold/orientation thresholds remain developmental; validate full-motion positive and negative evidence before freezing')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    records,issues=check_evidence(tasks,evidence)
    if issues:raise ValueError('Cannot freeze an unvalidated task set:\n'+'\n'.join(issues))
    scenes=planned_scenes(seed);models={};paired={};contracts={}
    code=code_manifest()
    for task in tasks:
        signatures={}
        for hand in HANDS:
            env=make_env(task,hand,scenes[0]['seed'])
            try:
                models[task+'/'+hand]=model_manifest(task,hand)
                for scene in scenes:
                    env.reset(seed=scene['seed']);actual=scene_signature(env)
                    expected=signatures.setdefault(scene['scene_id'],actual)
                    if actual!=expected:raise RuntimeError(f'Unpaired scene: {task}/{hand}/{scene}')
                contract={k:v for k,v in env.contract.items() if k!='hand'}
                if task in contracts and contracts[task]!=contract:raise RuntimeError('Task rules differ across hands')
                contracts[task]=contract
            finally:env.close()
        paired[task]=signatures
        if len(set(signatures.values()))!=50:
            raise RuntimeError(f'{task}: repeated physical scenes under distinct seeds')
    if code_manifest()!=code:raise RuntimeError('Code changed while freezing')
    protocol=dict(status='frozen',scope=scope,benchmark_tasks=list(benchmark_tasks or tasks),
        version=VERSION,tasks=list(tasks),hands=list(HANDS),
        scenes=scenes,paired_scene_sha256=paired,contracts=contracts,models=models,code=code,
        verified_witnesses=records,target_per_combination=50,camera_size=256,control_hz=20,
        split_policy='paired scene-disjoint 40 train / 5 validation / 5 test per task per hand',
        limitations=['Scene-disjoint does not mean source-template-disjoint.',
            'Joint training on all included hands does not measure unseen-hand generalization.'],
        success_policies={task:success_policy(task) for task in tasks},
        acceptance='native action replay + final state + task-specific success policy (bucket first success; others stable10) + real RGB/videos + hashes + complete paired count')
    protocol['sha256']=digest(protocol)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def load_frozen(path):
    p=json.loads(Path(path).read_text());payload={k:v for k,v in p.items() if k!='sha256'}
    if p.get('status')!='frozen' or p.get('sha256')!=digest(payload):raise ValueError('Not an intact frozen protocol')
    validate_scope(p['tasks'],p.get('scope','suite'),p.get('benchmark_tasks'))
    if p['version']!=VERSION or p['code']!=code_manifest():raise RuntimeError('Frozen code has changed')
    if p['hands']!=list(HANDS) or len(p['scenes'])!=50:raise ValueError('Invalid frozen coverage')
    return p
