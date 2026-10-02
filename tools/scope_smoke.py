#!/usr/bin/env python3
"""Check the 42-cell registry, model fingerprints and six diagnostic replay stores."""
import argparse
import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulator'))

import mujoco
import numpy as np
import zarr

from dexjoco.benchmark6 import HANDS, TARGET_TASKS
from dexjoco.benchmark6.dataset import pack_action, public_observation, unpack_action
from dexjoco.benchmark6.environments import CLASSES, make_env
from dexjoco.benchmark6.models import DONORS
from dexjoco.benchmark6.provenance import collector_manifest, model_manifest
from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.resolver import get_task_config_class

TRUTH = ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='dexjoco_scope_'))
    output.mkdir(parents=True, exist_ok=True)
    expected = {f'{task}/{hand}' for task in TARGET_TASKS for hand in HANDS}
    assert set(CONFIG_MAPPING) == expected
    assert set(CLASSES) == set(TARGET_TASKS)
    assert set(DONORS) == set(HANDS)
    for invalid in ('pick_bucket_oymotion', 'hammer_nail_shadow_hand_e3m5', 'water_plant', 'bimanual_hanoi'):
        try:
            get_task_config_class(invalid)
        except KeyError:
            pass
        else:
            raise AssertionError(f'Historical task remains registered: {invalid}')
    provenance = collector_manifest()
    assert provenance
    for task in TARGET_TASKS:
        for hand in HANDS:
            model_manifest(task, hand)
    reports = []
    for task in TARGET_TASKS:
        env = CONFIG_MAPPING[f'{task}/allegro']().get_environment(render_mode='none', seed=0)
        renderer = None
        try:
            renderer = mujoco.Renderer(env._model, height=64, width=64)
            observations, actions, images, references = [], [], [], []
            for _ in range(2):
                observations.append(public_observation(env))
                packed = pack_action(env, env.hold_action())
                actions.append(packed)
                references.append({k: getattr(env._data, k).copy() for k in TRUTH})
                renderer.update_scene(env._data, camera=env.camera_id[0])
                images.append(renderer.render().copy())
                env.step(unpack_action(env, packed['action'], packed['action_mask']))
            terminal = {k: getattr(env._data, k).copy() for k in TRUTH}
            store = zarr.open_group(str(output / task / 'diagnostic.zarr'), mode='w')
            store.attrs.update(training_eligible=False, diagnostic_only=True, task=task, hand='allegro')
            for name in observations[0]:
                store.create_dataset(name, data=np.stack([r[name] for r in observations]))
            for name in actions[0]:
                store.create_dataset(name, data=np.stack([r[name] for r in actions]))
            store.create_dataset('images', data=np.stack(images))
            store.create_dataset('timestamp', data=np.arange(2) * .05)
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
        replay = make_env(task, 'allegro', seed=0, pixels=False)
        maximum = 0.
        try:
            for index, reference in enumerate(references):
                for name in TRUTH:
                    maximum = max(maximum, float(np.max(np.abs(getattr(replay._data, name) - reference[name]))))
                replay.step(unpack_action(replay, store['action'][index], store['action_mask'][index]))
            for name in TRUTH:
                maximum = max(maximum, float(np.max(np.abs(getattr(replay._data, name) - terminal[name]))))
        finally:
            replay.close()
        assert maximum < 1e-8
        assert store['images'].shape == (2, 64, 64, 3) and np.ptp(store['images'][:]) > 0
        assert store['state'].shape[0] == store['action'].shape[0] == store['images'].shape[0] == 2
        reports.append(dict(task=task, passed=True, frames=2, replay_max_state_error=maximum))
        print(f'{task}: diagnostic store and deterministic replay passed', flush=True)
    result = dict(passed=True, registered_cells=len(expected), model_fingerprints=42,
                  collector_files=len(provenance), historical_task_ids_rejected=True,
                  diagnostic_only=True, trajectories=reports)
    (ROOT / 'SCOPE_SMOKE.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
