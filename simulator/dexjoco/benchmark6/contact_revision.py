"""Additive XML-only friction patch; all other compiled arrays must remain equal."""
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from . import HANDS,CANDIDATE_TASKS
from .audit import nonhand_manifest, digest
from .contact_policy import (
    POLICY_ID, PARENT_VERSION, hand_geom_ids, geom_identity, resolve_friction,
    source_review, validate_applied, PREFIX,
)
from .models import XML
VERSION='benchmark7_v2'


def array_changes(before,after):
    """Independent check includes mesh vertices/faces, mass, joints and actuators."""
    changes=[]
    for name in dir(before):
        value=getattr(before,name)
        # Existing meshes can contain undefined derived polygon normals. Match
        # their positions explicitly; NaN != NaN must not invent a new change.
        if isinstance(value,np.ndarray) and not np.array_equal(value,getattr(after,name),
                equal_nan=np.issubdtype(value.dtype,np.inexact)):
            changes.append(name)
    # Only the top-level model label may change, not any object name or order.
    if before.names.split(b'\0',1)[1:]!=after.names.split(b'\0',1)[1:]:changes.append('object_names')
    return changes


def nonfinite_arrays(model):
    result={}
    for name in dir(model):
        value=getattr(model,name)
        if isinstance(value,np.ndarray) and np.issubdtype(value.dtype,np.inexact):
            invalid=~np.isfinite(value)
            if invalid.any():
                result[name]=dict(count=int(invalid.sum()),
                    locations_sha256=hashlib.sha256(invalid.tobytes()).hexdigest())
    return result


def build(task,hand):
    if task not in CANDIDATE_TASKS or hand not in HANDS:raise ValueError((task,hand))
    path=XML/f'{VERSION}_{task}_{hand}.xml'
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    parent=XML/f'{PARENT_VERSION}_{task}_{hand}.xml'
    original=parent.read_bytes();meta_bytes=parent.with_suffix('.json').read_bytes()
    old_meta=json.loads(meta_bytes)
    if old_meta['version']!=PARENT_VERSION or old_meta['task']!=task or old_meta['hand']!=hand:
        raise ValueError('Wrong parent model identity')
    if old_meta['xml_sha256']!=hashlib.sha256(original).hexdigest():raise ValueError('Parent XML changed')
    before=mujoco.MjModel.from_xml_path(str(parent))
    tree=ET.fromstring(original);tree.set('model',f'{VERSION}_{task}_{hand}')
    expected={}
    for gid in hand_geom_ids(before):
        side,key,native=geom_identity(before,gid)
        expected[key]=dict(side=side,original=before.geom_friction[gid].tolist(),
                           resolved=resolve_friction(hand,side,native))
    visited=set()
    def patch_body(body):
        name=body.get('name','')
        ordinal=0
        def visit(container):
            # MJCF frames transform their children but do not create bodies.
            # Geoms inside nested frames share the owning body's ordinal space.
            nonlocal ordinal
            for child in container:
                if child.tag=='geom':
                    key=child.get('name') or f'{name}:geom:{ordinal}'
                    ordinal+=1
                    if key in expected:
                        if not name.startswith(PREFIX):raise ValueError('Refusing non-hand mutation')
                        child.set('friction',' '.join(format(x,'.17g') for x in expected[key]['resolved']['value']))
                        visited.add(key)
                elif child.tag=='body':patch_body(child)
                elif child.tag=='frame':visit(child)
        visit(body)
    patch_body(tree.find('worldbody'))
    if visited!=set(expected):raise ValueError('Could not map all collidable hand geoms')
    serialized=ET.tostring(tree,encoding='unicode')+'\n'
    # All file references in the expanded parent models are absolute.
    after=mujoco.MjModel.from_xml_string(serialized)
    changes=array_changes(before,after)
    if set(changes)-{'geom_friction'}:raise ValueError('Non-friction model change: '+str(changes))
    if nonhand_manifest(before)!=nonhand_manifest(after):raise ValueError('Non-hand physics changed')
    allowed=set(hand_geom_ids(before))
    changed=[i for i in range(before.ngeom) if not np.array_equal(before.geom_friction[i],after.geom_friction[i])]
    if not set(changed)<=allowed:raise ValueError('Object/arm friction changed')
    policy=dict(id=POLICY_ID,scope='surface friction only; not uniform whole-hand physics',
        source_review=source_review(hand),geoms=expected,
        measured_surface_evidence_found=any(e['resolved']['measured'] for e in expected.values()),
        preserved='Every compiled array other than hand geom_friction, including joint frictionloss and elastomer solref.',
        limitations=['Fallback surface parameters are not material measurements.',
            'Existing geometric/postprocessing approximations have not been certified or removed.',
            'Other contact solver parameters are retained, not newly calibrated in this revision.'])
    policy['sha256']=digest(policy)
    meta=dict(old_meta,version=VERSION,status='candidate_not_task_validated',
        xml_sha256=hashlib.sha256(serialized.encode()).hexdigest(),surface_friction_policy=policy,
        parent_model=dict(path=str(parent),xml_sha256=hashlib.sha256(original).hexdigest(),
                          metadata_sha256=hashlib.sha256(meta_bytes).hexdigest()),
        revision_check=dict(changed_arrays=changes,changed_hand_geoms=len(changed),
                            inherited_nonfinite_arrays=nonfinite_arrays(before),
                            nonhand_sha256=digest(nonhand_manifest(after))))
    validate_applied(after,meta)
    if parent.read_bytes()!=original or parent.with_suffix('.json').read_bytes()!=meta_bytes:
        raise RuntimeError('Parent changed during revision')
    path.write_text(serialized)
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    reopened=mujoco.MjModel.from_xml_path(str(path))
    if array_changes(after,reopened):raise RuntimeError('Saved model differs from checked model')
    validate_applied(reopened,meta)
    return meta
