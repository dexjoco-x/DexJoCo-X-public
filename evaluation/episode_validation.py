"""Validate saved data against the runtime contract before executing actions."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_validated_episode(bundle, row, expected_contract):
    bundle = Path(bundle)
    paths = {key: bundle / row[key] for key in ('tensor', 'contract')}
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != row[key + '_sha256']:
            raise ValueError(f'{key} checksum does not match manifest')
    contract = json.loads(paths['contract'].read_text())
    if contract != expected_contract:
        fields = sorted(key for key in set(contract) | set(expected_contract)
                        if contract.get(key) != expected_contract.get(key))
        raise ValueError('Episode contract differs from runtime: ' + ', '.join(fields))
    with np.load(paths['tensor'], allow_pickle=False) as data:
        required = ('state', 'action', 'state_mask', 'action_mask', 'arm_mask', 'timestamp', 'image_mask')
        if not set(required).issubset(data.files):
            raise ValueError('Missing required episode arrays')
        arrays = {key: data[key].copy() for key in required}
    n = row['steps']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('Expected positive integer episode length')
    shapes = {'state': (n, 2, 31), 'action': (n, 2, 28),
              'state_mask': (n, 2, 31), 'action_mask': (n, 2, 28),
              'arm_mask': (n, 2), 'timestamp': (n,), 'image_mask': (3,)}
    for key, value in arrays.items():
        if value.shape != shapes[key] or value.dtype.kind not in 'bifu' or not np.isfinite(value).all():
            raise ValueError(f'Invalid shape, type or finite values: {key}')
    for key in ('state', 'action', 'timestamp'):
        if arrays[key].dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError(f'Expected float32 or float64 values: {key}')
    for key in ('state_mask', 'action_mask', 'arm_mask', 'image_mask'):
        if not np.all((arrays[key] == 0) | (arrays[key] == 1)):
            raise ValueError(f'Expected binary mask: {key}')
    # Raw dataset camera order differs from the policy observation mask order.
    views = ('front', 'wrist_right', 'wrist_left')
    expected_images = np.array([view in contract['cameras'] for view in views])
    if not np.array_equal(arrays['image_mask'], expected_images):
        raise ValueError('Wrong image mask or camera order')
    if set(row['videos']) != set(contract['cameras']):
        raise ValueError('Video views differ from the runtime cameras')
    # Timestamps may be serialized as float32. Allow its rounding error but
    # require the declared sampling grid; zero/reordered/wrong-rate times fail.
    times = arrays['timestamp']
    tolerance = max(1e-9, np.finfo(times.dtype).eps * n * contract['control_dt'] * 2) if times.dtype.kind == 'f' else 1e-9
    if not np.allclose(times, np.arange(n) * contract['control_dt'], atol=tolerance, rtol=0):
        raise ValueError('Timestamps differ from the declared control period')
    states, actions = arrays['state'], arrays['action']
    bounds = contract['native_cartesian_bounds']
    for arm, side in enumerate(('right', 'left')):
        na = len(contract['actuator_names'].get(side, []))
        ns = len(contract['joint_names'].get(side, []))
        active = na > 0
        if not np.all(arrays['arm_mask'][:, arm] == active):
            raise ValueError('Wrong active-arm mask: ' + side)
        for key, width, count in (('action', 28, 6 + na), ('state', 31, 7 + ns)):
            mask = np.arange(width) < (count if active else 0)
            if not np.array_equal(arrays[key + '_mask'][:, arm], np.broadcast_to(mask, (n, width))):
                raise ValueError(f'Wrong {key} mask: {side}')
            if np.any(arrays[key][:, arm, ~mask] != 0):
                raise ValueError(f'Nonzero inactive {key} coordinates: {side}')
        if not active:
            continue
        if not np.allclose(np.linalg.norm(states[:, arm, 3:7], axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError('Invalid state quaternion norm: ' + side)
        limits = np.array([bounds['translation_m']] * 3 + [bounds['rotation_rad']] * 3)
        ranges = np.asarray(contract['actuator_ranges'][side])
        if (np.any(np.abs(actions[:, arm, :6]) > limits + 1e-9)
                or np.any(actions[:, arm, 6:6 + na] < ranges[:, 0] - 1e-7)
                or np.any(actions[:, arm, 6:6 + na] > ranges[:, 1] + 1e-7)):
            raise ValueError('Saved action exceeds the runtime limits: ' + side)
    return states, actions
