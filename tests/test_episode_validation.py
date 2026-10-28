"""Regression fixtures must fail even when their manifest hashes are updated."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from episode_validation import load_validated_episode
from reference_environment import reference_mismatches


class EpisodeValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = json.loads((ROOT / 'evaluation/contracts/hammer_nail/inspire.json').read_text())
        self.row = dict(tensor='native.npz', contract='contract.json', steps=3)
        self.row['videos'] = {view: {} for view in self.contract['cameras']}
        self.arrays = dict(state=np.zeros((3, 2, 31)), action=np.zeros((3, 2, 28)),
                           state_mask=np.zeros((3, 2, 31), bool),
                           action_mask=np.zeros((3, 2, 28), bool),
                           arm_mask=np.zeros((3, 2), bool), timestamp=np.arange(3)*.05,
                           image_mask=np.array([True, True, False]))
        self.arrays['state'][:, 0, 3] = 1
        self.arrays['state_mask'][:, 0, :7+len(self.contract['joint_names']['right'])] = True
        self.arrays['action_mask'][:, 0, :6+len(self.contract['actuator_names']['right'])] = True
        self.arrays['arm_mask'][:, 0] = True

    def save(self, contract=None):
        np.savez_compressed(self.root/'native.npz', **self.arrays)
        (self.root/'contract.json').write_text(json.dumps(self.contract if contract is None else contract))
        for key in ('tensor', 'contract'):
            self.row[key+'_sha256'] = hashlib.sha256((self.root/self.row[key]).read_bytes()).hexdigest()

    def check(self):
        return load_validated_episode(self.root, self.row, self.contract)

    def test_valid_arrays_are_preserved(self):
        self.save()
        states, actions = self.check()
        np.testing.assert_array_equal(actions, self.arrays['action'])
        np.testing.assert_array_equal(states, self.arrays['state'])

    def test_wrong_period_and_limits_rejected_after_rehash(self):
        bad = copy.deepcopy(self.contract)
        bad['control_dt'] = .1
        bad['native_cartesian_bounds']['translation_m'] = 99
        self.save(bad)
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            self.check()

    def test_zero_masks_rejected_after_rehash(self):
        for key in ('state_mask', 'action_mask', 'arm_mask'):
            self.arrays[key][:] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'active-arm mask'):
            self.check()

    def test_zero_timestamps_rejected_after_rehash(self):
        self.arrays['timestamp'][:] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, 'Timestamps'):
            self.check()

    def test_inactive_values_and_invalid_quaternion_rejected(self):
        self.arrays['action'][:, 1, 0] = .001
        self.save()
        with self.assertRaisesRegex(ValueError, 'inactive action'):
            self.check()
        self.arrays['action'][:, 1, 0] = 0
        self.arrays['state'][:, 0, 3] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            self.check()

    def test_wrong_blas_and_binary_rejected(self):
        fp = json.loads((ROOT/'requirements/native-replay-fingerprint.json').read_text())
        self.assertEqual(reference_mismatches(fp), [])
        fp['blas'][0]['architecture'] = 'different CPU kernel'
        fp['scipy_rotation_extension_sha256'] = 'different binary'
        self.assertIn('reference BLAS kernels/thread counts', reference_mismatches(fp))
        self.assertIn('scipy_rotation_extension_sha256', reference_mismatches(fp))

    def test_swapped_wrist_camera_mask_is_rejected(self):
        self.arrays['image_mask'] = np.array([True, False, True])
        self.save()
        with self.assertRaisesRegex(ValueError, 'image mask or camera order'):
            self.check()


if __name__ == '__main__':
    unittest.main()
