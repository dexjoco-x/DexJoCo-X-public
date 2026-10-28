"""Float32-bound rounding must not silently violate the executed limits."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collection.action_precision import float32_native_action


class Float32ActionPrecision(unittest.TestCase):
    def test_boundary_values_round_inward_and_remain_exact_float32(self):
        action = np.zeros((2, 28)); action[0, :3] = .02; action[0, 3:6] = -.03
        action[0, 6:8] = [1.47, -.1]
        limits = {'right': [[0., 1.47], [-.1, .1]]}
        result = float32_native_action(action, limits, .02, .03)
        self.assertGreater(float(np.float32(1.47)), 1.47)
        self.assertLessEqual(float(result[0, 6]), 1.47)
        self.assertGreaterEqual(float(result[0, 7]), -.1)
        self.assertTrue(np.all(np.abs(result[0, :3].astype(float)) <= .02))
        self.assertTrue(np.all(np.abs(result[0, 3:6].astype(float)) <= .03))
        self.assertFalse(result[1].any())
        np.testing.assert_array_equal(result.astype(float).astype(np.float32), result)
        np.testing.assert_array_equal(float32_native_action(result, limits, .02, .03), result)

    def test_invalid_and_unrepresentable_bounds_are_rejected(self):
        action = np.zeros((2, 28))
        with self.assertRaises(ValueError):
            float32_native_action(action, {'right': [[1.47, 1.47]]}, .02, .03)
        action[0, 0] = np.nan
        with self.assertRaises(ValueError):
            float32_native_action(action, {'right': [[0., 1.47]]}, .02, .03)


if __name__ == '__main__':
    unittest.main()
