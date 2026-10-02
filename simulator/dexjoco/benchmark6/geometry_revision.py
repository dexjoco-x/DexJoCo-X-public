"""Additive v3: remove local Wuji grasp pads, not its native CAD collision."""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .models import XML
from . import HANDS,CANDIDATE_TASKS
from .audit import digest,nonhand_manifest
from .contact_policy import validate_applied

VERSION='benchmark7_v3'
PARENT_VERSION='benchmark7_v2'
POLICY_ID='wuji_native_collision_only_v1'
# The public repository nests the standalone runtime under ``simulator/``.
# Resolve provenance inputs from that runtime root.
ROOT=XML.parents[3]
PADS={
    'right':('thumb_index_web','thumb_fingertip','index_fingertip','middle_fingertip','ring_fingertip','pinky_fingertip'),
    'left':('thumb_index_web','thumb_fingertip','index_fingertip','middle_fingertip'),
}


def path_for(task,hand):
    if task == 'single_hand_hanoi' and hand in HANDS:
        return XML/f'{VERSION}_single_hand_hanoi_inspire_scene_{hand}.xml'
    if task not in CANDIDATE_TASKS or hand not in HANDS:raise ValueError((task,hand))
    return XML/f'{VERSION}_{task}_{hand}.xml'


def pad_names(task,hand):
    if hand!='wuji':return set()
    sides=('right','left') if task.startswith('bimanual_') else ('right',)
    return {f'benchhand_{s}_native_wuji_{name}_pad' for s in sides for name in PADS[s]}


def canonical_xml(element):
    """Ignore whitespace only, not ordering or any parameter/default/asset."""
    return [element.tag,sorted(element.attrib.items()),(element.text or '').strip(),
            [canonical_xml(child) for child in element]]


def revised_tree(original,task,hand):
    tree=ET.fromstring(original);tree.set('model',f'{VERSION}_{task}_{hand}')
    expected=pad_names(task,hand);removed={}
    for parent in tree.iter():
        for child in list(parent):
            name=child.get('name')
            if child.tag=='geom' and name in expected:
                removed[name]=dict(child.attrib);parent.remove(child)
    if set(removed)!=expected:raise ValueError('Unexpected source pad set')
    return tree,removed


def source_files(hand):
    if hand!='wuji':return {}
    files=[f'teleoperation/wuji/assets/{side}/{name}' for side in ('right','left')
           for name in ('main.urdf','main_mujoco.urdf')]
    files.append('teleoperation/wuji/tools/generate_panda_wuji_xml.py')
    def source_path(name):
        path = ROOT / name
        if not path.exists():
            path = ROOT / 'dexjoco/sim/envs/model_source_provenance' / (name + '.txt')
        return path
    return {n:hashlib.sha256(source_path(n).read_bytes()).hexdigest() for n in files}


def validate_geometry(model,metadata):
    record=metadata.get('geometry_revision',{})
    payload={k:v for k,v in record.items() if k!='sha256'}
    if record.get('id')!=POLICY_ID or record.get('sha256')!=digest(payload):
        raise ValueError('Missing/corrupt geometry revision provenance')
    if record['source_files']!=source_files(metadata['hand']):raise ValueError('Geometry source changed')
    if metadata['hand']=='wuji':
        present={model.geom(i).name for i in range(model.ngeom)}
        forbidden=pad_names(metadata['task'],'wuji')
        if forbidden & present:raise ValueError('Removed grasp pads reintroduced')
        if set(record['removed_geoms'])!=forbidden:raise ValueError('Incorrect geometry removal record')
    return True


def check_pair(parent,path,metadata=None):
    meta=metadata or json.loads(path.with_suffix('.json').read_text())
    original=parent.read_bytes();expected,removed=revised_tree(original,meta['task'],meta['hand'])
    if canonical_xml(expected)!=canonical_xml(ET.parse(path).getroot()):
        raise ValueError('Unexpected XML change beyond declared pad removal')
    if hashlib.sha256(original).hexdigest()!=meta['geometry_revision']['parent']['xml_sha256']:
        raise ValueError('Parent XML changed')
    if hashlib.sha256(parent.with_suffix('.json').read_bytes()).hexdigest()!=meta['geometry_revision']['parent']['metadata_sha256']:
        raise ValueError('Parent metadata changed')
    mujoco.mj_clearCache(mujoco.mj_getCache())
    before=mujoco.MjModel.from_xml_path(str(parent))
    mujoco.mj_clearCache(mujoco.mj_getCache())
    after=mujoco.MjModel.from_xml_path(str(path))
    if before.ngeom-after.ngeom!=len(removed):raise ValueError('Wrong geom count change')
    if (before.nq,before.nv,before.nu,before.nbody,before.njnt,before.nmesh)!=(after.nq,after.nv,after.nu,after.nbody,after.njnt,after.nmesh):
        raise ValueError('Robot or mesh inventory changed')
    unchanged=['qpos0']
    unchanged += ['body_'+k for k in ('pos','quat','ipos','iquat','mass','inertia')]
    unchanged += [k for k in dir(before) if k.startswith(('jnt_','dof_','actuator_','mesh_'))
                  and k not in ('dof_length','mesh_bvhadr')
                  and isinstance(getattr(before,k),np.ndarray)]
    for key in unchanged:
        a=getattr(before,key);b=getattr(after,key)
        if not np.array_equal(a,b,equal_nan=np.issubdtype(a.dtype,np.inexact)):
            raise ValueError('Undeclared native geometry/dynamics change: '+key)
    if nonhand_manifest(before)!=nonhand_manifest(after):raise ValueError('Non-hand model changed')
    length_changes=[]
    for i in np.flatnonzero(before.dof_length!=after.dof_length):
        name=before.joint(int(before.dof_jntid[i])).name
        if meta['hand']!='wuji' or not name.startswith('benchhand_'):
            raise ValueError('Non-Wuji derived length changed')
        length_changes.append(dict(joint=name,before=float(before.dof_length[i]),after=float(after.dof_length[i])))
    validate_applied(after,meta);validate_geometry(after,meta)
    return dict(passed=True,removed_geoms=len(removed),native_arrays_checked=unchanged,
                derived_geometry_length_changes=length_changes,
                excluded_derived_fields=['mesh_bvhadr: reindexed BVH addresses','dof_length: geometry-derived angular length scale'],
                nonhand_equal=True,xml_only_declared_changes=True),after


def build(task,hand):
    if task == 'single_hand_hanoi':
        return json.loads(path_for(task,hand).with_suffix('.json').read_text())
    path=path_for(task,hand)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    parent=XML/f'{PARENT_VERSION}_{task}_{hand}.xml'
    original=parent.read_bytes();oldbytes=parent.with_suffix('.json').read_bytes();meta=json.loads(oldbytes)
    if (meta['version'],meta['task'],meta['hand'])!=(PARENT_VERSION,task,hand):raise ValueError('Wrong parent identity')
    if hashlib.sha256(original).hexdigest()!=meta['xml_sha256']:raise ValueError('Parent changed')
    tree,removed=revised_tree(original,task,hand)
    serialized=ET.tostring(tree,encoding='unicode')+'\n'
    meta=copy.deepcopy(meta);meta['version']=VERSION;meta['status']='candidate_not_task_validated'
    meta['xml_sha256']=hashlib.sha256(serialized.encode()).hexdigest()
    policy=meta['surface_friction_policy']
    for name in removed:del policy['geoms'][name]
    policy['sha256']=digest({k:v for k,v in policy.items() if k!='sha256'})
    geometry=dict(id=POLICY_ID,removed_geoms=removed,source_files=source_files(hand),
        parent=dict(path=str(parent),xml_sha256=hashlib.sha256(original).hexdigest(),
                    metadata_sha256=hashlib.sha256(oldbytes).hexdigest()),
        scope='Remove local Wuji supplemental grasp geoms only. Keep native mesh collisions and wrist adapter.',
        limitations=['Matched local URDF, not independently certified hardware geometry.',
                     'Inherited inertia floors, drives, solver settings and source imperfections unchanged.',
                     'Friction policy source notes describe the imported parent; this record describes current geometry.'])
    geometry['sha256']=digest(geometry);meta['geometry_revision']=geometry
    # Stage only new files; any failure is left visible, never overwrite old data.
    path.write_text(serialized);path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    check_pair(parent,path,meta)
    if parent.read_bytes()!=original or parent.with_suffix('.json').read_bytes()!=oldbytes:
        raise RuntimeError('Parent changed during build')
    return meta
