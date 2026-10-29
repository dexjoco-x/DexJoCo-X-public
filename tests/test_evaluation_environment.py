"""A changed binary build cannot silently produce a formal evaluation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
import evaluate


class EvaluationEnvironment(unittest.TestCase):
    def test_nonreference_build_is_rejected_before_policy_connection(self):
        fingerprint = json.loads((ROOT/'requirements/native-replay-fingerprint.json').read_text())
        fingerprint['mujoco_library_sha256'] = {'libmujoco.so.3.4.0': 'different-build'}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=directory, task='hammer_nail', hand='inspire',
                diagnostic=False, max_steps=None, action_horizon=30, execution_horizon=8,
                server='ws://unused', layout='arm40_v1', representation='native', episodes=150, video=False)
            with patch.object(evaluate, 'environment_fingerprint', return_value=fingerprint), \
                 patch.object(evaluate, 'connect') as connection:
                with self.assertRaisesRegex(ValueError, 'Non-reference numerical environment'):
                    evaluate.main(args)
                connection.assert_not_called()

    def test_short_formal_run_is_rejected_before_output_or_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must_not_exist'
            args = argparse.Namespace(output=str(output), episodes=50, diagnostic=False)
            with patch.object(evaluate, 'connect') as connection:
                with self.assertRaisesRegex(ValueError, 'Formal evaluation requires'):
                    evaluate.main(args)
                connection.assert_not_called()
            self.assertFalse(output.exists())

    def test_episode_counts_and_release_identity(self):
        from provenance import evaluation_provenance, validate_episode_count
        validate_episode_count(150, False)
        validate_episode_count(1, True)
        for episodes in (0, 151):
            with self.assertRaises(ValueError):
                validate_episode_count(episodes, True)
        record = evaluation_provenance(ROOT)
        self.assertEqual(record['benchmark_dataset_version'], 'training-float32-v1')
        self.assertEqual(len(record['benchmark_archive_sha256']), 64)
        self.assertIn('runtime.py', record['evaluation_source_sha256'])


if __name__ == '__main__':
    unittest.main()
