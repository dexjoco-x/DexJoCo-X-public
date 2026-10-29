"""Shared Training v1 preprocessing for 32-D pi0.5 training and serving."""
import json
from pathlib import Path

import numpy as np

from dataset.tools.adapters import PortableDataset, load_training_stats
from dataset.tools.policy_adapter import Native200PolicyAdapter
from .adapter import Native32, images
from .serve import ROOT, sha256

SCHEMA = 'dexjocox_pi05_training_v1_native32'


def package_identity():
    package = json.loads((ROOT / 'dataset/PACKAGES.json').read_text())
    return {key: package[key] for key in ('dataset_version', 'sha256')}


class TrainingV1Codec:
    def __init__(self, groups, mode='ego_interleave32', horizon=25):
        self.native = Native200PolicyAdapter(groups)
        self.layout = Native32(mode, horizon)

    def encode_targets(self, labels, mask):
        labels, mask = np.asarray(labels), np.asarray(mask, bool)
        if labels.dtype != np.float32 or labels.shape != (self.layout.horizon, 200) or mask.shape != labels.shape:
            raise ValueError('Expected saved float32 labels and masks with shape [horizon, 200]')
        active = mask.any(axis=0)
        if active[80:].any():
            raise ValueError('Active labels exceed the arm40 interface')
        self.layout.masks(dict(state_mask=active[:80], action_mask=active[:80]))
        return self.layout.noise(labels[:, :80]), self.layout.noise(mask[:, :80]).astype(bool)

    def state(self, observation):
        self.layout.masks(observation)
        state, _ = self.native.state(observation)
        return state[:self.layout.state_dim]

    def actions(self, tokens, observation):
        _, mask = self.layout.masks(observation)
        labels80 = self.layout.actions(tokens, mask)
        labels200 = np.pad(labels80, ((0, 0), (0, 120)))
        return self.native.actions(labels200, observation)


class TrainingV1Dataset(PortableDataset):
    """Read the released saved labels verbatim, with episode-tail loss masking."""
    def __init__(self, root, hand, tasks, mode='ego_interleave32', horizon=25):
        root = Path(root)
        package = json.loads((ROOT / 'dataset/PACKAGES.json').read_text())
        for name, digest in package['source']['manifest_sha256'].items():
            if sha256(root / 'bundle/manifests' / name) != digest:
                raise ValueError('Use the current, unmodified Training v1 dataset')
        groups = load_training_stats(root, 'beingh_final_v3')
        selected = {task + '/' + hand: groups[task + '/' + hand] for task in tasks}
        super().__init__(root / 'bundle', horizon=horizon, stats=selected, image_size=None)
        self.rows = [row for row in self.rows if row['hand'] == hand and row['task'] in tasks]
        self.ends = np.cumsum([row['steps'] for row in self.rows])
        if not self.rows or {row['task'] for row in self.rows} != set(tasks):
            raise ValueError('No training data for the requested hand/tasks')
        if mode == 'single_right32' and any(task.startswith('bimanual_') for task in tasks):
            raise ValueError('Bimanual tasks require ego_interleave32')
        self.codec = TrainingV1Codec(selected, mode, horizon)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['actions'], item['action_mask'] = self.codec.encode_targets(item['actions'], item['action_mask'])
        item['state'] = item['state'][:self.codec.layout.state_dim]
        return item


def model_observation(items, tokenizer, max_length, Observation, device):
    """The same state text, camera order and RGB range in training and serving."""
    import torch
    ids, masks = [], []
    for item in items:
        bins = np.digitize(np.clip(item['state'], -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {item['prompt']}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = tokenizer.encode(text, add_bos=True)
        if len(tokens) > max_length:
            raise ValueError('Prompt/state tokens exceed the configured context length')
        ids.append(tokens + [0] * (max_length - len(tokens)))
        masks.append(np.arange(max_length) < len(tokens))
    keys = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
    return Observation.from_dict(dict(
        image={key: torch.as_tensor(np.stack([i['images'][key] for i in items]), device=device) for key in keys},
        image_mask={key: torch.tensor([i['image_masks'][key] for i in items], device=device) for key in keys},
        state=torch.as_tensor(np.stack([i['state'] for i in items]), device=device),
        tokenized_prompt=torch.tensor(ids, dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.as_tensor(np.stack(masks), device=device),
    ))


def serving_item(codec, observation):
    rgb, mask = images(observation)
    return dict(state=codec.state(observation), images=rgb, image_masks=mask,
                prompt=f"Embodiment: {observation['hand']}. {observation['prompt']}")
