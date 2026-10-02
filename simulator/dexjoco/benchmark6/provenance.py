"""Content fingerprints of rules, native models, assets and demonstration data."""
from functools import lru_cache
import hashlib
import json
import importlib.metadata
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
from .audit import digest
from .models import path_for

ROOT=Path(__file__).resolve().parents[2]


@lru_cache(maxsize=20000)
def _file_hash(path,mtime_ns,size):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):result.update(chunk)
    return result.hexdigest()


def file_hash(path):
    path=Path(path).resolve();stat=path.stat()
    return _file_hash(str(path),stat.st_mtime_ns,stat.st_size)


def collector_manifest():
    """Fingerprint the installed public collector and its simulator dependencies."""
    package = ROOT / 'dexjoco'
    files = []
    for directory in ('benchmark6', 'autocollect', 'sim', 'tasks'):
        files.extend((package / directory).rglob('*.py'))
    files.extend((ROOT / 'scripts').glob('import_*.py'))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(files))}


def code_manifest():
    """Replay/release code fingerprint, independent of future teacher recipes.

    The saved native actions, physical model, environment/controller dynamics,
    success rules, observation schema, and all replay checks remain binding.
    Search-only code is recorded separately and cannot alter a saved action.
    """
    teacher_files={'single_teacher.py','dual_teacher.py','replacement_teacher.py',
        'sources.py','retarget.py','bucket_recovery.py','camera_grasp.py'}
    return {name:value for name,value in collector_manifest().items()
        if not name.startswith('scripts/') and not
        (name.startswith('dexjoco/benchmark6/') and Path(name).name in teacher_files)}


def asset_manifest(path):
    path=Path(path)
    tree=ET.parse(path).getroot()
    if tree.findall('include'):raise ValueError('Frozen XML must be fully expanded')
    files={str(path.resolve()),str(path.with_suffix('.json').resolve())}
    metadata=json.loads(path.with_suffix('.json').read_text())
    evidence=dict(metadata.get('surface_friction_policy',{}).get('source_review',{}).get('files_sha256',{}))
    for name,expected in metadata.get('geometry_revision',{}).get('source_files',{}).items():
        if name in evidence and evidence[name]!=expected:raise ValueError('Conflicting source fingerprints')
        evidence[name]=expected
    for name,expected in evidence.items():
        from .contact_policy import resolve_source_file
        source=resolve_source_file(name).resolve()
        if file_hash(source)!=expected:raise ValueError('Contact evidence changed: '+name)
        files.add(str(source))
    for element in tree.iter():
        if element.tag in ('mesh','texture','hfield') and element.get('file'):
            files.add(str((path.parent/element.get('file')).resolve()))
    hashes={name:file_hash(name) for name in sorted(files)}
    return dict(files=hashes,sha256=digest(hashes))


def model_manifest(task,hand):
    return asset_manifest(path_for(task,hand))


def source_manifest(source):
    source=Path(source)
    if not source.exists():return dict(kind='procedural',name=str(source))
    store=source/'replay.zarr'
    if not store.is_dir():raise ValueError('Missing source demonstration store')
    # Historical RGB is not consumed by the privileged geometric teacher.
    files=[]
    for name in ('action','state','timestamp'):
        sub=store/'data'/name
        if sub.exists():files.extend(p for p in sub.rglob('*') if p.is_file())
    for name in ('summary.json','final_truth.json'):
        if (source/name).is_file():files.append(source/name)
    private=source/'privileged.zarr'
    if private.is_dir():files.extend(p for p in private.rglob('*') if p.is_file())
    if not files:raise ValueError('Empty source motion data')
    result=dict(kind='demonstration',path=str(source.resolve()),
                files={str(p.relative_to(source)):file_hash(p) for p in sorted(files)})
    if private.is_dir() and (source/'summary.json').is_file():
        # Contact-grasp proposals additionally consume a source pilot model.
        # Capture preserved legacy models too, not only current target assets.
        from .models import XML
        meta=json.loads((source/'summary.json').read_text())
        task={'hammer':'hammer_nail','bucket':'pick_bucket'}.get(meta['task'],meta['task'])
        version=meta.get('version','benchmark6_v1')
        result['source_model']=asset_manifest(XML/f"{version}_{task}_{meta['hand']}.xml")
        result['legacy_source_version_assumed']='version' not in meta
    return result


def capture(task,hand,source):
    result=dict(code=code_manifest(),collector_code=collector_manifest(),model=model_manifest(task,hand),source=source_manifest(source),
        packages={p:importlib.metadata.version(p) for p in ('mujoco','numpy','scipy','zarr','gymnasium')},
        mujoco_runtime=mujoco.mj_versionString())
    result['sha256']=digest(result)
    return result


def assert_current(provenance,task,hand,source):
    if provenance.get('sha256')!=digest({k:v for k,v in provenance.items() if k!='sha256'}):
        raise RuntimeError('Corrupt provenance record')
    # During replay, a changed search teacher cannot change the saved actions.
    # Source hashes, physics, rules, schema and packages must still match.
    def replay_fields(value):return {k:v for k,v in value.items() if k not in ('sha256','collector_code')}
    if replay_fields(capture(task,hand,source))!=replay_fields(provenance):
        raise RuntimeError('Code/model/assets/source changed since episode start; not training eligible')
