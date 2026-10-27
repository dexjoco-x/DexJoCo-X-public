"""Paired codec preservation must not certify failed Native demonstrations."""
import argparse
from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import audit_dexlatent_validation_replay as audit


class ImmediatePool:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def submit(self, function, job):
        future = Future()
        future.set_result(function(job))
        return future


class CodecReplayAcceptance(unittest.TestCase):
    def test_training_demo_failed_in_both_branches_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / 'bundle/manifests'
            manifests.mkdir(parents=True)
            (manifests / 'hammer_nail.json').write_text(json.dumps([
                dict(task='hammer_nail', hand='inspire', scene_id='scene_000', seed=123, split='train')]))
            args = argparse.Namespace(release=root, output=root / 'report.json', checkpoint=None,
                                      workers=1, split=None, task=None, hand=None)
            record = dict(task='hammer_nail', hand='inspire', scene='scene_000',
                          native={'passed':False}, dexlatent={'passed':False})
            with patch.object(audit, 'ProcessPoolExecutor', ImmediatePool), \
                 patch.object(audit, 'run_episode', return_value=record), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    audit.main(args)
            self.assertEqual(caught.exception.code, 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(report['episodes'], 1)
            self.assertEqual(report['native_to_dexlatent_regressions'], 0)
            self.assertTrue(report['codec_preserves_native_successes'])
            self.assertFalse(report['native_task_replay_passed'])
            self.assertFalse(report['passed'])


if __name__ == '__main__':
    unittest.main()
