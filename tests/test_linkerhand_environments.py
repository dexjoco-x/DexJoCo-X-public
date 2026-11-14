"""Exercise the shipped task entry point, action API, cache and shared-wrist grasp."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from dexjoco.benchmark6 import HANDS as BENCHMARK_HANDS, TARGET_TASKS
from dexjoco.benchmark6 import models
from dexjoco.benchmark6.environments import make_env
from dexjoco.extensions.linkerhand import HANDS, build


class LinkerHandEnvironments(unittest.TestCase):
    def test_task_entry_points_and_reset(self):
        for hand in HANDS:
            for task in TARGET_TASKS:
                with self.subTest(hand=hand, task=task):
                    env = make_env(task, hand, seed=123, pixels=False)
                    try:
                        self.assertEqual(env.hand_metadata['benchmark_track'], 'extension')
                        for side in env._native_ctrl_ids:
                            prefix = 'benchhand_' + side + '_adapter_'
                            for part in ('flange', 'bridge', 'wrist_plate'):
                                geom = env._model.geom(prefix + part + '_collision')
                                self.assertEqual(geom.contype, 1)
                                self.assertEqual(geom.conaffinity, 1)
                        if task == 'single_hand_hanoi':
                            self.assertEqual(set(env._native_ctrl_ids), {'right'})
                        env.reset(seed=123)
                        before = env._data.time
                        for _ in range(5):env.step(env.hold_action())
                        self.assertAlmostEqual(env._data.time - before, .25)
                        self.assertFalse(any(env._data.warning.number))
                        env.reset(seed=124)
                        if task == 'bimanual_unlock_ipad' and hand == 'linkerhand_l30':
                            branches = env._ipad_measurements.branches
                            ids = []
                            for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                                b = env._model.body('benchhand_left_native_lh_' + finger + '_distal').id
                                ids.append(next(branches[g] for g in branches if env._model.geom_bodyid[g] == b))
                            self.assertEqual(len(set(ids)), 5)
                    finally:
                        env.close()

    def test_cache_integrity_and_benchmark_scope(self):
        self.assertEqual(set(BENCHMARK_HANDS), {'allegro', 'inspire', 'leap', 'linkerhand_l20', 'sharpa_wave', 'wuji', 'xhand'})
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {'XDG_CACHE_HOME': cache}):
            path = build('hammer_nail', 'linkerhand_l30')
            self.assertTrue(path.is_relative_to(cache))
            xml = path.read_bytes()
            self.assertEqual(build('hammer_nail', 'linkerhand_l30'), path)
            path.write_text('broken cache')
            self.assertEqual(build('hammer_nail', 'linkerhand_l30').read_bytes(), xml)
            self.assertTrue(models.path_for('hammer_nail', 'allegro').is_relative_to(models.XML))
        with self.assertRaises(ValueError):build('not_a_task', 'linkerhand_l30')
        with self.assertRaises(ValueError):build('hammer_nail', '../other')


if __name__ == '__main__':
    unittest.main()
