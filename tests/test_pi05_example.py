"""Verify arm identity, physical-time pairing and the inference-only boundary."""
import unittest
import numpy as np
from examples.pi05.adapter import Native32, images, quantile_transform, validate_statistics


def observation():
    mask = np.zeros(80, bool)
    mask[:28] = mask[40:68] = True
    return dict(state=np.arange(80, dtype=np.float32), state_mask=mask.copy(),
                action_mask=mask, image_mask=np.array([True, False, True]),
                base=np.full((8, 8, 3), 10, np.uint8),
                wrist_left=np.full((8, 8, 3), 20, np.uint8),
                wrist_right=np.full((8, 8, 3), 30, np.uint8))


class Pi05ExampleTests(unittest.TestCase):
    def test_left_right_pairing_preserves_time(self):
        layout = Native32('ego_interleave32', 25)
        obs = observation()
        actions = np.zeros((25, 80), np.float32)
        actions[:, :28] = np.arange(25)[:, None] + 10
        actions[:, 40:68] = np.arange(25)[:, None] + 100
        # Reserved noise is sampled by the evaluator but is not a command.
        noise = actions.copy()
        noise[:, 32:40] = noise[:, 72:80] = 999
        tokens = layout.noise(noise)
        np.testing.assert_equal(tokens[0, :28], 100)
        np.testing.assert_equal(tokens[1, :28], 10)
        np.testing.assert_equal(tokens[48, :28], 124)
        np.testing.assert_equal(tokens[49, :28], 34)
        np.testing.assert_equal(layout.actions(tokens, obs['action_mask']), actions)

    def test_single_arm_rejects_silent_left_arm_truncation(self):
        layout = Native32('single_right32', 30)
        obs = observation()
        with self.assertRaises(ValueError):
            layout.state(obs)
        obs['state_mask'][40:] = obs['action_mask'][40:] = False
        state, mask = layout.state(obs)
        np.testing.assert_equal(state[:28], np.arange(28))
        self.assertFalse(state[28:].any())
        self.assertFalse(mask[28:].any())

    def test_ego_rejects_active_reserved_slots(self):
        obs = observation()
        obs['action_mask'][39] = True
        with self.assertRaises(ValueError):
            Native32('ego_interleave32', 25).masks(obs)

    def test_rgb_and_masks_keep_camera_identity(self):
        rgb, mask = images(observation())
        np.testing.assert_equal(rgb['base_0_rgb'], 10)
        np.testing.assert_equal(rgb['right_wrist_0_rgb'], 30)
        np.testing.assert_equal(rgb['left_wrist_0_rgb'], 0)
        self.assertFalse(mask['left_wrist_0_rgb'])
        self.assertTrue(mask['right_wrist_0_rgb'])

    def test_quantiles_restore_physical_units_and_mask_padding(self):
        stats = dict(q01=[-2, 3, -1], q99=[2, 3, 1])
        mask, value = np.array([True, True, False]), np.array([1, 3.1, 999])
        encoded = quantile_transform(value, mask, stats)
        decoded = quantile_transform(encoded, mask, stats, inverse=True)
        np.testing.assert_allclose(decoded, [1, 3.1, 0], atol=1e-6)

    def test_bad_shapes_and_nonfinite_noise_are_rejected(self):
        layout = Native32('ego_interleave32', 25)
        with self.assertRaises(ValueError):
            layout.actions(np.zeros((25, 32)), np.ones(80, bool))
        noise = np.zeros((25, 80)); noise[0, 0] = np.nan
        with self.assertRaises(ValueError):
            layout.noise(noise)

    def test_bad_statistics_are_rejected(self):
        stats = {key: dict(q01=[0] * 80, q99=[1] * 80) for key in ('state', 'action')}
        validate_statistics(stats, 80)
        stats['state']['q99'][0] = -1
        with self.assertRaises(ValueError):
            validate_statistics(stats, 80)


if __name__ == '__main__':
    unittest.main()
