"""Regression cases for physically valid rare actions lost by old exports."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.tools.normalization import AFFINE, AFFINE_V2, LEGACY, bind, denormalize, normalize, stabilize_scales, scale_labels, preserve_decoded_scale


class TrainingNormalizationTest(unittest.TestCase):
    def test_valid_rare_translation_survives_both_directions(self):
        stats = bind({'q01': [-.003], 'q99': [.00409648]})
        action = np.array([[.0161058], [-.025]], dtype=np.float64)
        labels = normalize(action, np.ones_like(action, bool), stats)
        self.assertGreater(float(labels[0, 0]), 1)
        self.assertLess(float(labels[1, 0]), -1)
        reconstructed = denormalize(labels, np.ones_like(action, bool), stats)
        np.testing.assert_allclose(reconstructed, action, atol=2e-9, rtol=0)
        self.assertEqual(reconstructed.dtype, np.float64)

    def test_equal_quantiles_do_not_erase_rare_gripper_motion(self):
        stats = bind({'q01': [1.5, 0], 'q99': [1.5, 0]})
        x = np.array([[1.5, 0], [2.0, .25], [.75, -.125]])
        mask = np.ones_like(x, bool)
        np.testing.assert_array_equal(denormalize(normalize(x, mask, stats), mask, stats), x)

    def test_masked_padding_and_broadcast_masks(self):
        stats = bind({'q01': [-1, 4], 'q99': [1, 4]})
        x = np.array([[.5, np.nan], [3., np.inf]])
        labels = normalize(x, [True, False], stats)
        result = denormalize(labels, [True, False], stats)
        np.testing.assert_array_equal(result, [[.5, 0], [3., 0]])

    def test_unversioned_statistics_keep_existing_checkpoint_behavior(self):
        stats = {'q01': [-1, 3], 'q99': [1, 3]}
        x = np.array([[2., 4.], [-2., 3.]], dtype=np.float64)
        mask = np.ones_like(x, bool)
        expected_labels = np.array([[1., 1.], [-1., 1.]], dtype=np.float32)
        expected_commands = np.array([[1., 3.], [-1., 3.]])
        for s in (stats, bind(stats, LEGACY)):
            np.testing.assert_array_equal(normalize(x, mask, s), expected_labels)
            np.testing.assert_array_equal(denormalize(expected_labels, mask, s), expected_commands)

    def test_float64_inverse_avoids_float32_addition_loss(self):
        stats = bind({'q01': [0.], 'q99': [2.]})
        label = np.array([np.nextafter(np.float32(1), np.float32(2))])
        decoded = denormalize(label, [True], stats)
        self.assertEqual(decoded[0], float(label[0]) + 1.)

    def test_physical_limits_round_inward_without_quantile_clipping(self):
        stats = bind({'q01': [.2, -.005], 'q99': [1.1, .005],
                      'physical_bounds': {'minimum': [0., -.02], 'maximum': [1.47, .02]}})
        values = np.array([[1.47, .02], [0., -.02], [1.4, .0161]])
        labels = normalize(values, np.ones_like(values, bool), stats)
        decoded = denormalize(labels, np.ones_like(values, bool), stats)
        self.assertTrue(np.all(decoded >= [0., -.02]))
        self.assertTrue(np.all(decoded <= [1.47, .02]))
        self.assertTrue(np.all(labels[0] > 1))
        np.testing.assert_allclose(decoded, values, atol=2e-7, rtol=0)
        np.testing.assert_array_equal(normalize(decoded, np.ones_like(values, bool), stats), labels)
        with self.assertRaises(ValueError):
            normalize([[1.6, 0.]], [[True, True]], stats)

    def test_invalid_contracts_fail(self):
        cases = [dict(q01=[2], q99=[1]), dict(q01=[0], q99=[np.nan]),
                 dict(q01=[0, 0], q99=[1, 1])]
        for stats in cases:
            with self.assertRaises(ValueError):
                normalize(np.array([0.]), [True], bind(stats))
        with self.assertRaises(ValueError):
            normalize(np.array([np.nan]), [True], bind(dict(q01=[0], q99=[1])))
        with self.assertRaises(ValueError):
            denormalize([0], [True], dict(q01=[0], q99=[1], normalization_version='unknown'))

    def test_nearly_constant_channel_has_bounded_labels_and_identical_commands(self):
        old = bind(dict(q01=[1.7449999998609003], q99=[1.744999999980638],
                        physical_bounds=dict(minimum=[0.], maximum=[1.745])), AFFINE_V2)
        new = stabilize_scales(old)
        values = np.array([[0.], [1.], [1.7449999999], [1.745]])
        mask = np.ones_like(values, bool)
        previous = normalize(values, mask, old)
        self.assertGreater(np.abs(previous).max(), 1e10)
        labels = normalize(values, mask, new)
        self.assertLessEqual(np.abs(labels).max(), 1.)
        np.testing.assert_array_equal(labels, scale_labels(previous, new))
        np.testing.assert_array_equal(denormalize(labels, mask, new), denormalize(previous, mask, old))

    def test_saved_neighboring_labels_keep_every_bit_after_rescaling(self):
        old = bind(dict(q01=[-.01], q99=[.01], physical_bounds=dict(minimum=[-.05], maximum=[.05])), AFFINE_V2)
        new = stabilize_scales(old)
        labels = np.nextafter(np.array([[3.2], [1e-8]], np.float32), np.float32(-np.inf))
        scaled = scale_labels(labels, new)
        np.testing.assert_array_equal(denormalize(scaled, [[True]], new), denormalize(labels, [[True]], old))
        with self.assertRaises(ValueError):
            scale_labels(np.array([np.nextafter(np.float32(0), np.float32(1))]), new)

    def test_subnormal_collection_commands_survive_the_scale_fit(self):
        previous = bind(dict(q01=[0., 1.745], q99=[0., 1.745],
                             physical_bounds=dict(minimum=[0., 0.], maximum=[1.745, 1.745])), AFFINE_V2)
        labels = np.array([[np.nextafter(np.float32(0), np.float32(1))]*2], np.float32)
        proposed = stabilize_scales(previous)
        adjusted = preserve_decoded_scale(labels, [[True, True]], previous, proposed)
        self.assertEqual(adjusted['label_scale_exponents'], [0, 1])
        scaled = scale_labels(labels, adjusted, require_label_bits=False)
        np.testing.assert_array_equal(denormalize(labels, [[True, True]], previous), denormalize(scaled, [[True, True]], adjusted))


if __name__ == '__main__':
    unittest.main()
