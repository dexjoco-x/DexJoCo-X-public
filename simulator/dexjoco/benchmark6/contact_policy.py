"""Source-first SURFACE friction revision; joint friction and compliance stay native.

Different hands may have different sourced surface properties. There is no
cross-hand numerical equality requirement. A fallback is an approximation,
not a measured material property. This revision deliberately changes friction
only; it is not a certification of the imported geometry or other parameters.
"""
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICY_ID = 'source_first_surface_friction_v1'
PARENT_VERSION = 'benchmark7_v1'
FALLBACK = (1., .005, .0001)
PREFIX = 'benchhand_'
XML_ROOT = 'dexjoco/dexjoco/sim/envs/xmls/'
SHARPA_ROOT = 'teleoperation/sharpa_wave/sharpa-urdf-usd-xml/'
SHARPA_COMMIT = '0d447b6889e6d993758169dfc0aa75ee9f6ad8d7'
SHARPA_URL = 'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/blob/'+SHARPA_COMMIT

# These reviews concern the local hardware/model revision, not all products
# that a manufacturer may have released. Never interpret no evidence as proof
# that the manufacturer has never performed a measurement.
REVIEWS = {
    'allegro': dict(basis='original_model_simulator_default',
        files=[XML_ROOT+'panda_allegro_copy.xml', XML_ROOT+'panda_allegro_left.xml',
               XML_ROOT+'wonik_allegro/README.md'],
        urls=['https://mujoco.readthedocs.io/en/3.4.0/XMLreference.html#body-geom'],
        note='Original collision classes omit friction; compiled value is the MuJoCo default.'),
    'leap': dict(basis='fallback_no_matched_surface_calibration',
        files=[XML_ROOT+'leap_hand/README.md', XML_ROOT+'leap_hand/right/main.urdf',
               XML_ROOT+'leap_hand/left/main.urdf', XML_ROOT+'panda_leap_right.xml'],
        urls=[], note='URDF joint_properties friction is not surface friction. Local 1.8 pads have no matched material measurement in these sources.'),
    'inspire': dict(basis='fallback_no_matched_surface_calibration',
        files=['teleoperation/inspire/README.md','teleoperation/inspire/assets/right/main_mujoco.urdf',
               'teleoperation/inspire/tools/generate_panda_inspire_xml.py'],
        urls=[], note='Surface calibration not specified in the imported URDF/generator. Existing surface default retained.'),
    'wuji': dict(basis='fallback_local_grasp_patch',
        files=['teleoperation/wuji/README.md','teleoperation/wuji/assets/right/main_mujoco.urdf',
               'teleoperation/wuji/assets/left/main_mujoco.urdf',
               'teleoperation/wuji/tools/generate_panda_wuji_xml.py'],
        urls=['https://github.com/wuji-technology/wuji-description'],
        note='2.0/2.2/2.4 surfaces are set in local grasp postprocessing, not substantiated as calibrated surface values. Geometry patches are separately unresolved and unchanged.'),
    'sharpa_wave': dict(basis='manufacturer_documented_simulator_default',
        files=[SHARPA_ROOT+'README.md',SHARPA_ROOT+'wave_01/right_sharpa_wave/right_sharpa_wave.xml',
               SHARPA_ROOT+'wave_01/left_sharpa_wave/left_sharpa_wave.xml'],
        urls=[SHARPA_URL+'/README.md',SHARPA_URL+'/wave_01/right_sharpa_wave/right_sharpa_wave.xml'],
        note='Manufacturer documents calibrated JOINT dynamics, default rigid-body contact, and softer elastomer solref. Local surface 3.2 is not in that source. Preserve joint friction and solref; not a claim of measured surface friction.'),
    'xhand': dict(basis='fallback_importer_added_surface_settings',
        files=[XML_ROOT+'xhand/right/source.urdf',XML_ROOT+'xhand/left/source.urdf','scripts/import_xhand.py'],
        urls=[], note='Surface friction and adapter friction were added by the local importer, not measured in the source URDF.'),
    'linkerhand_l20': dict(basis='fallback_importer_added_surface_settings',
        files=[XML_ROOT+'linkerhand_l20/right/source.urdf',XML_ROOT+'linkerhand_l20/left/source.urdf',
               'scripts/import_linkerhand_l20.py'],
        urls=[], note='User-supplied L20 V10.1 URDF has no surface calibration; local importer supplies contact settings.'),
}

# Explicit, per-hand/per-side/native-geom records can preserve nondefault
# calibrated values. No qualifying surface measurement is currently registered.
# Changing this registry requires a new model version, never silent rebuilding.
VERIFIED_SURFACES = {}


def resolve_friction(hand, side, native_geom, overrides=None):
    if hand not in REVIEWS or side not in ('right','left'): raise ValueError((hand,side))
    records = VERIFIED_SURFACES if overrides is None else overrides
    record = records.get((hand,side,native_geom))
    if record is None:
        return dict(value=list(FALLBACK),basis=REVIEWS[hand]['basis'],measured=False)
    required=('value','source','hardware_revision','conditions','measurement_reference')
    if 'value' not in record or any(not isinstance(record.get(k),str) or not record[k].strip()
                                    for k in required if k!='value'):
        raise ValueError('Incomplete surface evidence record')
    value=np.asarray(record['value'],dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all() or np.any(value<0):
        raise ValueError('Invalid surface friction vector')
    return dict(value=value.tolist(),basis='documented_surface_measurement',measured=True,
                evidence={k:record[k] for k in required if k!='value'})


def resolve_source_file(name):
    path = ROOT / name
    if not path.exists() and name.startswith('dexjoco/dexjoco/'):
        path = ROOT / ('dexjoco/' + name.removeprefix('dexjoco/dexjoco/'))
    if not path.exists() and name.startswith('teleoperation/'):
        path = ROOT / 'dexjoco/sim/envs/model_source_provenance' / (name + '.txt')
    return path


def source_review(hand):
    review = REVIEWS[hand]
    return dict(**review, files_sha256={
        name: hashlib.sha256(resolve_source_file(name).read_bytes()).hexdigest()
        for name in review['files']
    })


def hand_geom_ids(model):
    return [i for i in range(model.ngeom)
            if model.body(int(model.geom_bodyid[i])).name.startswith(PREFIX)
            and (model.geom_contype[i] or model.geom_conaffinity[i])]


def geom_identity(model, gid):
    body=model.body(int(model.geom_bodyid[gid])).name
    side=body[len(PREFIX):].split('_',1)[0]
    stem=PREFIX+side+'_'
    name=model.geom(gid).name
    ordinal=int(np.count_nonzero(model.geom_bodyid[:gid]==model.geom_bodyid[gid]))
    key=name if name else f'{body}:geom:{ordinal}'
    return side,key,key.removeprefix(stem)


def validate_applied(model, metadata):
    from .audit import digest
    policy=metadata.get('surface_friction_policy',{})
    payload={k:v for k,v in policy.items() if k!='sha256'}
    if policy.get('id')!=POLICY_ID or policy.get('sha256')!=digest(payload):
        raise ValueError('Missing or corrupt surface friction provenance')
    actual={geom_identity(model,i)[1]:model.geom_friction[i].tolist() for i in hand_geom_ids(model)}
    if set(actual)!=set(policy['geoms']):raise ValueError('Hand collision geometry coverage changed')
    for gid in hand_geom_ids(model):
        side,key,native=geom_identity(model,gid)
        expected=resolve_friction(metadata['hand'],side,native)
        if policy['geoms'][key]['resolved']!=expected or actual[key]!=expected['value']:
            raise ValueError('Surface friction differs from registered source policy: '+key)
    if policy['source_review']!=source_review(metadata['hand']):
        raise ValueError('Surface parameter evidence changed after model build')
    return True
