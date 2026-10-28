#!/usr/bin/env python3
"""Validate raw episodes, converted Native data, masks and real video frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset/tools'))
from data_io import write
from normalization import AFFINE,version


def video_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError('Video frame decode failed')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def reference(raw, kind):
    """Physical field layout independent of the conversion entry point."""
    values = np.zeros((len(raw[kind]), 200), np.float64)
    masks = np.zeros_like(values, bool)
    for arm, base in enumerate((0, 40)):
        active = raw[kind + '_mask'][:, arm, 0].astype(bool)
        values[:, base:base+3] = raw[kind][:, arm, :3]
        masks[:, base:base+6] = active[:, None]
        if kind == 'state':
            values[active, base+3:base+6] = Rotation.from_quat(
                raw[kind][active, arm, 3:7][:, [1, 2, 3, 0]]).as_rotvec()
            offset = 7
        else:
            values[:, base+3:base+6] = raw[kind][:, arm, 3:6]
            offset = 6
        width = raw[kind].shape[-1] - offset
        values[:, base+6:base+6+width] = raw[kind][:, arm, offset:]
        masks[:, base+6:base+6+width] = raw[kind+'_mask'][:, arm, offset:]
    return np.where(masks, values, 0), masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepared-name', default='beingh05_affine_v2')
    args = parser.parse_args()
    bundle = args.data_root / 'bundle'
    rows = {}
    for path in sorted((bundle/'manifests').glob('*.json')):
        for row in json.loads(path.read_text()):
            rows[(row['task'], row['hand'], row['scene_id'])] = row
    groups = json.loads((args.data_root/'prepared'/(args.prepared_name+'_datasets.json')).read_text())
    reports = []
    for group in groups:
        directory = args.data_root / group['path']
        info = json.loads((directory/'meta/info.json').read_text())
        stats_path = directory/'meta/raw_train_statistics.json'
        stats = json.loads(stats_path.read_text())
        episodes = [json.loads(line) for line in (directory/'meta/episodes.jsonl').read_text().splitlines()]
        assert info['pre_normalized'] and info['representation'] == 'benchmark7_project_native200_v1'
        checked, frames, error, videos = 0, 0, 0., 0
        for episode in episodes:
            row = rows[(group['task'], group['hand'], episode['source_scene'])]
            with np.load(bundle/row['tensor'], allow_pickle=False) as file:
                raw = {key: file[key] for key in file.files}
            index = episode['episode_index']
            frame = pd.read_parquet(directory/'data/chunk-000'/f'episode_{index:06d}.parquet')
            assert len(frame) == row['steps'] == episode['length']
            np.testing.assert_array_equal(frame['frame_index'], np.arange(len(frame)))
            np.testing.assert_array_equal(frame['episode_index'], index)
            np.testing.assert_allclose(frame['timestamp'], np.arange(len(frame))/20, atol=1e-4, rtol=0)
            for kind, value_key, mask_key in (
                ('state', 'observation.state', 'observation.state_mask'),
                ('action', 'action', 'action.valid_mask'),
            ):
                value = np.stack(frame[value_key].to_list())
                mask = np.stack(frame[mask_key].to_list()).astype(bool)
                assert value.shape == mask.shape == (len(frame), 200)
                assert np.isfinite(value).all()
                assert not value[~mask].any()
                physical, expected_mask = reference(raw, kind)
                np.testing.assert_array_equal(mask, expected_mask)
                lo, hi = np.asarray(stats[kind]['q01']), np.asarray(stats[kind]['q99'])
                if version(stats[kind]) == AFFINE:
                    assert info['normalization_version'] == AFFINE
                    span = hi-lo
                    restored = np.where(mask, np.where(span != 0, (value.astype(np.float64)+1)/2*span+lo, value.astype(np.float64)+lo), 0)
                    expected = np.where(mask, physical, 0)
                else:
                    assert np.max(np.abs(value)) <= 1.000001
                    restored = np.where(mask, (value+1)/2*(hi-lo)+lo, 0)
                    expected = np.where(mask, np.clip(physical, lo, hi), 0)
                current = float(np.max(np.abs(restored-expected)))
                assert current < 1e-5, (group['task'], group['hand'], kind, current)
                error = max(error, current)
            if index == 0:
                for view, video in row['videos'].items():
                    prepared_video = directory/'videos'/('observation.images.'+view)/'chunk-000'/f'episode_{index:06d}.mp4'
                    for offset in (0, row['steps']-1):
                        rgb = video_frame(bundle/video['path'], offset)
                        prepared_rgb = video_frame(prepared_video, offset)
                        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
                        np.testing.assert_array_equal(prepared_rgb, rgb)
                        videos += 1
            frames += len(frame)
            checked += 1
        assert checked == group['episodes'] == info['total_episodes']
        assert frames == group['frames'] == info['total_frames']
        reports.append(dict(task=group['task'], hand=group['hand'], episodes=checked, frames=frames,
                            maximum_physical_inverse_error=error, decoded_video_frames=videos,
                            statistics_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(), passed=True))
        print(group['task'], group['hand'], checked, 'episodes passed', flush=True)
    assert reports
    result = dict(passed=True, groups=len(reports), episodes=sum(r['episodes'] for r in reports),
                  source_episodes=len(rows), complete_source_coverage=sum(r['episodes'] for r in reports)==len(rows),
                  physical_replay_checked=False,
                  frames=sum(row['frames'] for row in reports), representation='native',
                  model_training_required=False, groups_checked=reports)
    write(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'groups_checked'}, indent=2))


if __name__ == '__main__':
    main()
