"""Regression tests for saved-label pairing and exact physical decoding."""
import unittest
import numpy as np

from dataset.tools.adapters import bind_action_bounds
from dataset.tools.normalization import bind, normalize
from dataset.tools.policy_adapter import Native200PolicyAdapter
from examples.pi05.training_v1 import TrainingV1Codec, serving_item


class TrainingV1Tests(unittest.TestCase):
    def setUp(self):
        bounds = dict(actuator_ranges={'right': [[0., 1.47]] * 22, 'left': [[0., 1.47]] * 22},
                      native_cartesian_bounds=dict(translation_m=.02, rotation_rad=.03))
        stats = bind_action_bounds(bind(dict(q01=[-.005] * 200, q99=[.005] * 200)), bounds)
        self.groups = {'task/hand': dict(state=stats, action=stats)}
        mask = np.zeros(80, bool); mask[:28] = mask[40:68] = True
        self.obs = dict(task='task', hand='hand', prompt='Move the object.', state=np.zeros(80, np.float32),
                        state_mask=mask.copy(), action_mask=mask.copy(), image_mask=[True, False, True],
                        base=np.full((256, 256, 3), 11, np.uint8),
                        wrist_left=np.full((256, 256, 3), 22, np.uint8),
                        wrist_right=np.full((256, 256, 3), 33, np.uint8))

    def test_normalized_labels_pair_and_decode_without_rounding_physical_actions(self):
        codec = TrainingV1Codec(self.groups, horizon=3)
        mask = np.tile(np.pad(self.obs['action_mask'], (0, 120)), (3, 1))
        physical = np.zeros((3, 200)); physical[:, 0] = .0161000001; physical[:, 40] = -.011100001
        physical[:, 6:28] = .1234567; physical[:, 46:68] = .9876543
        labels = normalize(physical, mask, self.groups['task/hand']['action'])
        tokens, token_mask = codec.encode_targets(labels, mask)
        np.testing.assert_array_equal(tokens[0::2], labels[:, 40:72])
        np.testing.assert_array_equal(tokens[1::2], labels[:, :32])
        output = codec.actions(tokens, self.obs)
        self.assertEqual(output.dtype, np.float64)
        np.testing.assert_array_equal(output, Native200PolicyAdapter(self.groups).actions(labels, self.obs))
        self.assertFalse(token_mask[:, 28:].any())

    def test_tail_and_inactive_arm_are_excluded_from_loss_mask(self):
        codec = TrainingV1Codec(self.groups, horizon=3)
        labels = np.zeros((3, 200), np.float32); mask = np.zeros_like(labels, bool)
        mask[:2, :28] = True
        _, token_mask = codec.encode_targets(labels, mask)
        self.assertEqual(token_mask.sum(), 56)
        self.assertFalse(token_mask[0::2].any())
        self.assertFalse(token_mask[4:].any())

    def test_single_arm_rejects_left_arm_and_state_padding_is_masked(self):
        codec = TrainingV1Codec(self.groups, mode='single_right32', horizon=3)
        with self.assertRaisesRegex(ValueError, 'Active coordinates'):
            codec.state(self.obs)
        self.obs['state_mask'][40:] = self.obs['action_mask'][40:] = False
        self.assertEqual(codec.state(self.obs).shape, (32,))

    def test_camera_identity_and_prompt_survive_the_serving_boundary(self):
        item = serving_item(TrainingV1Codec(self.groups), self.obs)
        self.assertEqual(item['prompt'], 'Embodiment: hand. Move the object.')
        np.testing.assert_array_equal(item['images']['base_0_rgb'], 11)
        np.testing.assert_array_equal(item['images']['right_wrist_0_rgb'], 33)
        np.testing.assert_array_equal(item['images']['left_wrist_0_rgb'], 0)

    def test_loss_does_not_train_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is only required in the policy training environment')
        from examples.pi05.train import masked_flow_loss
        losses = torch.tensor([[[1., 100.], [3., 100.]]], requires_grad=True)
        mask = torch.tensor([[[True, False], [True, False]]])
        result = masked_flow_loss(losses, mask)
        self.assertEqual(float(result.detach()), 2.)
        result.backward()
        torch.testing.assert_close(losses.grad, torch.tensor([[[.5, 0.], [.5, 0.]]]))


if __name__ == '__main__':
    unittest.main()
