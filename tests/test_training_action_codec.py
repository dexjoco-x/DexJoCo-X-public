import sys
import unittest
import hashlib
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collection.training_action_codec import TrainingActionCodec
from dataset.tools.adapters import bind_action_bounds, saved_training_labels
from dataset.tools.normalization import bind
from dataset.tools.policy_adapter import Native200PolicyAdapter


class TrainingActionCodecTest(unittest.TestCase):
    def setUp(self):
        self.contract = {'actuator_ranges': {'right': [[0., 1.47], [-.1, .1]]},
                         'native_cartesian_bounds': {'translation_m': .02, 'rotation_rad': .03}}
        self.mask = np.zeros((2, 28), bool); self.mask[0, :8] = True
        self.stats = bind_action_bounds(bind({'q01': [-.005]*200, 'q99': [.005]*200}), self.contract)
        self.codec = TrainingActionCodec(self.stats, self.contract, self.mask)

    def test_serialized_float32_labels_produce_exact_same_executed_command(self):
        requested = np.zeros((2, 28)); requested[0, :8] = [.02, -.02, .0161, .03, -.03, 0, 1.47, -.1]
        labels, command, clipped = self.codec.encode_command(requested)
        self.assertEqual(clipped, 0)
        restored = np.frombuffer(labels.tobytes(), dtype=np.float32)
        np.testing.assert_array_equal(self.codec.decode_labels(restored), command)
        self.assertFalse(command[1].any())
        self.assertLessEqual(command[0, 6], 1.47)
        self.assertGreater(command[0, 2], .016)

    def test_collection_applies_only_actual_controller_limits(self):
        requested = np.zeros((2, 28)); requested[0, 0] = .1
        _, command, clipped = self.codec.encode_command(requested)
        self.assertEqual(clipped, 1)
        self.assertLessEqual(command[0, 0], .02)
        self.assertGreater(command[0, 0], .019999)

    def test_mismatched_statistics_and_invalid_serialized_labels_are_rejected(self):
        bad = dict(self.contract, native_cartesian_bounds={'translation_m': .01, 'rotation_rad': .03})
        with self.assertRaises(ValueError): TrainingActionCodec(self.stats, bad, self.mask)
        with self.assertRaises(ValueError): self.codec.decode_labels(np.zeros(200, np.float64))
        labels = np.zeros(200, np.float32); labels[150] = 1
        with self.assertRaises(ValueError): self.codec.decode_labels(labels)

    def test_collected_labels_are_not_renormalized_and_reject_other_statistics(self):
        labels, _, _ = self.codec.encode_command(np.zeros((2, 28)))
        labels[0] = np.nextafter(labels[0], np.float32(-np.inf))
        arrays = dict(training_labels=labels[None], action=self.codec.decode_labels(labels)[None],
                      action_mask=self.mask[None])
        row = dict(training_statistics_sha256=hashlib.sha256(
            json.dumps(self.stats, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        np.testing.assert_array_equal(saved_training_labels(arrays, row, self.stats), labels[None])
        with self.assertRaises(ValueError): saved_training_labels(arrays, {}, self.stats)
        bad = dict(arrays, action=arrays['action'].copy())
        bad['action'][0, 0, 0] += 1e-8
        with self.assertRaises(AssertionError): saved_training_labels(bad, row, self.stats)

    def test_policy_boundary_preserves_the_collected_decoded_command(self):
        requested = np.zeros((2, 28)); requested[0, :8] = [.0161, 0, 0, 0, 0, 0, 1.47, 0]
        labels, actual, _ = self.codec.encode_command(requested)
        mask = np.zeros(80, bool); mask[:8] = True
        observation = dict(task='example', hand='hand', action_mask=mask)
        adapter = Native200PolicyAdapter({'example/hand': dict(state=self.stats, action=self.stats)})
        output = adapter.actions(labels[None], observation)
        self.assertEqual(output.dtype, np.float64)
        np.testing.assert_array_equal(output[0, :28], actual[0])
        np.testing.assert_array_equal(output[0, 40:68], actual[1])
        self.assertFalse(output[0, 28:40].any())
        # Models need not learn values in padded channels; those are ignored.
        labels[150] = np.nan
        np.testing.assert_array_equal(adapter.actions(labels[None], observation), output)
        with self.assertRaises(ValueError): adapter.actions(labels[None], dict(observation, hand='unknown'))


if __name__ == '__main__': unittest.main()
