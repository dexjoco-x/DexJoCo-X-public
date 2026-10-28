"""Check exported files with the installed official LeRobot dataset reader."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_group(root, group, horizon, video_backend):
    folder = root / group['path']
    windows = {key: [i / 20 for i in range(horizon)] for key in ('action', 'action.valid_mask')}
    dataset = LeRobotDataset('dexjoco-x/local-validation', root=folder,
                            video_backend=video_backend, delta_timestamps=windows)
    episodes = list(map(json.loads, (folder / 'meta/episodes.jsonl').read_text().splitlines()))
    records = []
    for episode in episodes:
        index = episode['episode_index']
        file = folder / dataset.meta.get_data_file_path(index)
        df = pd.read_parquet(file)
        df = df[df.episode_index == index]
        start, end = episode['dataset_from_index'], episode['dataset_to_index']
        assert end - start == len(df) == episode['length']
        actual = dataset.hf_dataset[start:end]
        for key in ('action', 'observation.state', 'action.valid_mask', 'observation.state_mask'):
            np.testing.assert_array_equal(np.stack(actual[key]), np.stack(df[key]))
        assert np.stack(actual['action']).dtype == np.float32
        image_error = 0
        for local in sorted({0, len(df) // 2, len(df) - 1}):
            sample = dataset[start + local]
            count = min(horizon, len(df) - local)
            np.testing.assert_array_equal(sample['action'][:count], np.stack(df.action)[local:local + count])
            expected_padding = np.arange(horizon) >= count
            np.testing.assert_array_equal(sample['action_is_pad'], expected_padding)
            np.testing.assert_array_equal(sample['action.valid_mask_is_pad'], expected_padding)
            effective_mask = sample['action.valid_mask'] & ~sample['action_is_pad'][:, None]
            assert not effective_mask[count:].any()
            for view in dataset.meta.video_keys:
                path = folder / dataset.meta.get_video_file_path(index, view)
                reader = cv2.VideoCapture(str(path))
                reader.set(cv2.CAP_PROP_POS_FRAMES, local)
                ok, frame = reader.read(); reader.release()
                if not ok: raise ValueError(f'Video frame missing: {path} frame {local}')
                expected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                observed = np.rint(sample[view].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                assert observed.shape == expected.shape
                error = int(np.max(np.abs(observed.astype(int) - expected.astype(int))))
                # Different H.264 decoder implementations can differ by one level.
                if error > 1: raise ValueError(f'Wrong RGB frame: {path} frame {local}, error {error}')
                image_error = max(image_error, error)
        records.append(dict(task=group['task'], hand=group['hand'], scene_id=episode['source_scene'],
                            frames=len(df), parquet_sha256=sha(file), labels_exact=True,
                            temporal_padding_correct=True, maximum_rgb_level_error=image_error))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--prepared-name', default='lerobot_affine_v2')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--video-backend', default='pyav')
    args = parser.parse_args()
    if args.horizon < 1: raise ValueError('Horizon must be positive')
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(1)
    groups = json.loads((args.data_root / 'prepared' / (args.prepared_name + '_datasets.json')).read_text())
    records = []
    for group in groups:
        records.extend(check_group(args.data_root, group, args.horizon, args.video_backend))
        print(f"{group['task']}/{group['hand']}: {len(records)} episodes checked", flush=True)
    if not records: raise ValueError('No episodes selected')
    report = dict(passed=True, episodes=len(records), frames=sum(r['frames'] for r in records),
                  lerobot_version=importlib.metadata.version('lerobot'), video_backend=args.video_backend,
                  physical_replay_checked=False, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__': main()
