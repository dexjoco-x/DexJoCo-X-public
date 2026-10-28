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
                server='ws://unused', layout='arm40_v1', representation='native', episodes=1, video=False)
            with patch.object(evaluate, 'environment_fingerprint', return_value=fingerprint), \
                 patch.object(evaluate, 'connect') as connection:
                with self.assertRaisesRegex(ValueError, 'Non-reference numerical environment'):
                    evaluate.main(args)
                connection.assert_not_called()


if __name__ == '__main__':
    unittest.main()
