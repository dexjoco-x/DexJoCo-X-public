"""Serve user-owned pi0.5/RLinf weights without importing project training code."""
import argparse
import hashlib
import importlib.machinery
import json
from pathlib import Path
import subprocess
import sys
import threading
import types

import numpy as np

from .adapter import Native32, images, quantile_transform, validate_statistics

ROOT = Path(__file__).resolve().parents[2]
RLINF_COMMIT = 'e1d1bac185fc8d2718c25df9ae8a9089893c46c2'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def import_inference(source):
    """Load only the external model packages, bypassing the RL training registry."""
    source = Path(source).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != RLINF_COMMIT:
        raise ValueError('Use the documented RLinf commit')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--', 'rlinf'], check=True)
    for name in ('rlinf', 'rlinf.models', 'rlinf.models.embodiment',
                 'rlinf.models.embodiment.openpi_rlinf'):
        if name in sys.modules:
            raise RuntimeError('Run this inference server in its own process')
        package = types.ModuleType(name)
        package.__path__ = [str(source.joinpath(*name.split('.')))]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
    from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
    from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation
    return Pi0, Pi0Config, Observation


class Pi05Policy:
    def __init__(self, source, checkpoint, tokenizer, contract, device='cuda'):
        import sentencepiece
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        self.contract = contract
        if contract['schema'] != 'dexjocox_pi05_native32_v1':
            raise ValueError('Unknown checkpoint contract')
        self.layout = Native32(contract['token_layout'], int(contract['action_horizon']))
        if min(contract['max_token_len'], contract['inference_steps']) < 1 or not contract['hands'] or not contract['tasks']:
            raise ValueError('Context length, inference steps and checkpoint scope must be positive/nonempty')
        validate_statistics(contract['statistics'], self.layout.state_dim)
        if sha256(checkpoint) != contract['checkpoint_sha256'] or sha256(tokenizer) != contract['tokenizer_sha256']:
            raise ValueError('Checkpoint or tokenizer differs from the training contract')
        self.device = torch.device(device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        Pi0, Pi0Config, self.Observation = import_inference(source)
        config = Pi0Config(pi05=True, action_dim=32, action_horizon=self.layout.token_horizon,
                          max_token_len=contract['max_token_len'], dtype='bfloat16')
        self.model = Pi0(config, num_steps=contract['inference_steps'], action_env_dim=32,
                         action_chunk=self.layout.token_horizon)
        if Path(checkpoint).suffix == '.safetensors':
            from safetensors.torch import load_file
            weights = load_file(str(checkpoint))
        else:
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.to_bfloat16_for_selected_params('bfloat16')
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    def observation(self, observation):
        torch = self.torch
        state, state_mask = self.layout.state(observation)
        state = quantile_transform(state, state_mask, self.contract['statistics']['state'])
        bins = np.digitize(np.clip(state, -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
        prompt = f"Embodiment: {observation['hand']}. {observation['prompt']}"
        text = f"Task: {prompt}, State: {' '.join(map(str, bins))};\nAction: "
        ids = self.tokenizer.encode(text, add_bos=True)
        length, max_length = len(ids), self.contract['max_token_len']
        if length > max_length:
            raise ValueError('Prompt/state tokens exceed checkpoint context length')
        ids += [0] * (max_length - length)
        rgb, mask = images(observation)
        return self.Observation.from_dict(dict(
            image={key: torch.as_tensor(value[None], device=self.device) for key, value in rgb.items()},
            image_mask={key: torch.tensor([value], device=self.device) for key, value in mask.items()},
            state=torch.as_tensor(state[None], device=self.device),
            tokenized_prompt=torch.tensor([ids], device=self.device),
            tokenized_prompt_mask=torch.arange(max_length, device=self.device)[None] < length,
        ))

    def infer(self, observation, noise):
        if observation['hand'] not in self.contract['hands'] or observation['task'] not in self.contract['tasks']:
            raise ValueError('Hand/task is outside the checkpoint contract')
        _, action_mask = self.layout.masks(observation)
        token_noise = self.layout.noise(noise)
        torch = self.torch
        with self.lock, torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            tokens = self.model.sample_actions(self.observation(observation),
                num_steps=self.contract['inference_steps'],
                noise=torch.as_tensor(token_noise[None], device=self.device)).float().cpu().numpy()[0]
        physical = self.layout.actions(tokens, action_mask)
        physical = quantile_transform(physical, action_mask, self.contract['statistics']['action'], inverse=True)
        return {'actions': physical}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Clean external RLinf checkout')
    parser.add_argument('--checkpoint', required=True, help='User-owned full RLinf state_dict (.pt or .safetensors)')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--contract', required=True, help='Checkpoint-specific JSON, see README')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=8200)
    parser.add_argument('--bind', default='127.0.0.1')
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    policy = Pi05Policy(args.source, args.checkpoint, args.tokenizer, contract, args.device)
    sys.path.insert(0, str(ROOT / 'evaluation'))
    from serve80 import handler_for
    from websockets.sync.server import serve
    metadata = dict(policy_kind='pi05_rlinf', token_layout=contract['token_layout'],
                    trained_hands=contract['hands'], trained_tasks=contract['tasks'],
                    diagnostic_only=bool(contract['diagnostic_only']),
                    checkpoint_sha256=contract['checkpoint_sha256'],
                    contract_sha256=sha256(args.contract), upstream_commit=RLINF_COMMIT)
    with serve(handler_for(policy, 'arm40_v1', 'native', policy.layout.horizon, metadata),
               args.bind, args.port, compression=None, max_size=None) as server:
        print(f'pi0.5 server listening on {args.bind}:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
