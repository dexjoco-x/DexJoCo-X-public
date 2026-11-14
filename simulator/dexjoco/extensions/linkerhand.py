"""Compose the optional O6/L6/L30 hands with the six released task scenes."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..benchmark6 import TARGET_TASKS
from ..benchmark6 import models, contact_policy
from ..benchmark6.audit import digest
from .linkerhand_mount import add_adapter

HANDS = ('linkerhand_o6', 'linkerhand_l6', 'linkerhand_l30')
VERSION = 'linkerhand_extension_v1'


def _absolute(spec, root):
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str((root / spec.compiler.meshdir / mesh.file).resolve())
    for texture in spec.textures:
        if texture.file:
            texture.file = str((root / spec.compiler.texturedir / texture.file).resolve())
    spec.compiler.meshdir = ''
    spec.compiler.texturedir = ''


def _mounted(hand, side):
    path = models.XML / hand / side / 'hand.xml'
    source = mujoco.MjSpec.from_file(str(path))
    _absolute(source, path.parent)
    out = mujoco.MjSpec()
    out.copy_during_attach = True
    out.compiler.degree = False
    root = out.worldbody.add_body(name='mount')
    child = root.add_frame().attach_body(
        source.body(side[0] + 'h_hand_base_link'), prefix='native_', suffix='')
    child.pos = [-.009, 0, .026]
    child.quat = [np.sqrt(.5), 0, -np.sqrt(.5), 0]
    add_adapter(root, hand)
    out.compile()
    return out


def _fingerprint(task, hand):
    source = models.path_for(task, 'linkerhand_l20')
    # Include the installation path: generated XMLs contain absolute mesh paths.
    payload = {'version': VERSION, 'mujoco': mujoco.__version__,
               'installation': str(models.XML.resolve()), 'task': task, 'hand': hand}
    files = [source, source.with_suffix('.json'), Path(__file__),
             Path(__file__).with_name('linkerhand_mount.py'),
             Path(models.__file__), Path(contact_policy.__file__)]
    files += sorted(p for p in (models.XML / hand).rglob('*') if p.is_file())
    payload['files'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return digest(payload)


def build(task, hand):
    """Return a cached assembled XML; never rewrite the shipped benchmark assets."""
    if task not in TARGET_TASKS or hand not in HANDS:
        raise ValueError((task, hand))
    fingerprint = _fingerprint(task, hand)
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    folder = cache / 'dexjoco-x' / VERSION / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{task}_{hand}.xml'
    with (folder / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and path.with_suffix('.json').exists():
            meta = json.loads(path.with_suffix('.json').read_text())
            if meta.get('build_fingerprint') == fingerprint and meta['xml_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest():
                return path
        return _build(task, hand, path, fingerprint)


def _build(task, hand, path, fingerprint):
    source = models.path_for(task, 'linkerhand_l20')
    source_meta = json.loads(source.with_suffix('.json').read_text())
    tree = ET.fromstring(source.read_text())
    for item in tree.iter():
        for attr, value in list(item.attrib.items()):
            if value.startswith('benchhand_'):
                item.set(attr, 'retired_' + value)
    spec = mujoco.MjSpec.from_string(ET.tostring(tree, encoding='unicode'))
    spec.copy_during_attach = True
    _absolute(spec, models.XML)
    for key in list(spec.keys):
        spec.delete(key)
    sides = ('right', 'left') if task.startswith('bimanual') else ('right',)
    for side in sides:
        name = 'attachment_' + side if len(sides) == 2 or task == 'single_hand_hanoi' else 'attachment'
        root = spec.body(name)
        for item in list(root.bodies) + list(root.geoms):
            spec.delete(item)
        child = _mounted(hand, side)
        root.add_frame().attach_body(child.body('mount'), prefix='benchhand_' + side + '_', suffix='')
        spec.compile()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.modelname = f'{VERSION}_{task}_{hand}'
    spec.compile()
    xml = spec.to_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    meta = dict(version=VERSION, task=task, hand=hand, build_fingerprint=fingerprint,
                reference=str(source), reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                object_source_hashes=source_meta.get('object_source_hashes', {}),
                sides={side: models._hand_metadata(model, side) for side in sides},
                benchmark_track='extension',
                mount_note='Rigid simulation bracket; native wrist transform and hand inertias retained.')
    if hand == 'linkerhand_l30' and 'left' in sides:
        meta['sides']['left']['palm_body_name'] = 'benchhand_left_native_lh_wrist_carpal_link0'
    geoms = {}
    for gid in contact_policy.hand_geom_ids(model):
        side, key, native = contact_policy.geom_identity(model, gid)
        resolved = contact_policy.resolve_friction(hand, side, native)
        if not np.array_equal(model.geom_friction[gid], resolved['value']):
            raise ValueError('Unexpected extension surface friction: ' + key)
        geoms[key] = dict(resolved=resolved)
    policy = dict(id=contact_policy.POLICY_ID, geoms=geoms, source_review=contact_policy.source_review(hand))
    policy['sha256'] = digest(policy)
    meta['surface_friction_policy'] = policy
    # Publish a complete pair under the build lock; interrupted builds are rebuilt.
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary)
        (staged / 'model.xml').write_text(xml)
        (staged / 'model.json').write_text(json.dumps(meta, indent=2) + '\n')
        os.replace(staged / 'model.xml', path)
        os.replace(staged / 'model.json', path.with_suffix('.json'))
    return path
