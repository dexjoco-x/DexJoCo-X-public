"""A robot-state reset check alone cannot detect an incorrect hand mount."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from runtime import Runtime
from audit_dexlatent_validation_replay import initial_error


class XHandMountBinding(unittest.TestCase):
    def test_wrong_mount_is_rejected_despite_matching_initial_robot_state(self):
        for task in ('bimanual_photograph', 'bimanual_unlock_ipad'):
            with self.subTest(task=task):
                runtime = Runtime(task, 'xhand', 154187, 'arm40_v1', 'native', False)
                try:
                    expected = np.zeros((2, 31), np.float64)
                    for arm, side in enumerate(('right', 'left')):
                        pose = runtime.e.tcp(side)
                        q = runtime.d.qpos[runtime.e._full_hand_qpos[side]]
                        expected[arm, :3] = pose[:3, 3]
                        expected[arm, 3:7] = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
                        expected[arm, 7:7 + len(q)] = q
                    binding = runtime.contract['model_binding']
                    body = runtime.m.body(binding['body'])
                    body.quat[:] = [0.000281768, -0.706543, -0.00028132, 0.70767]
                    mujoco.mj_forward(runtime.m, runtime.d)
                    self.assertLessEqual(initial_error(runtime, expected), 2e-5)
                    with self.assertRaisesRegex(RuntimeError, 'mount orientation mismatch'):
                        runtime._check_contract()
                    body.quat[:] = binding['quaternion_wxyz']
                    runtime.e.hand_metadata['mount_revision']['id'] = 'legacy_wrong_mount'
                    with self.assertRaisesRegex(RuntimeError, 'Model mount mismatch'):
                        runtime._check_contract()
                finally:
                    runtime.close()


if __name__ == '__main__':
    unittest.main()
