"""Serve the model and frozen decoder emitted by examples.pi05.train."""
import argparse
import json
from pathlib import Path
import sys
import threading

from .serve import ROOT, RLINF_COMMIT, sha256
from .train import load_model
from .training_v1 import SCHEMA, TrainingV1Codec, model_observation, package_identity, serving_item


class TrainingV1Policy:
    def __init__(self, source, checkpoint, tokenizer, contract, device='cuda'):
        import sentencepiece
        import torch
        if contract['schema'] != SCHEMA or contract['dataset'] != package_identity():
            raise ValueError('Expected a checkpoint trained with the current Training v1 interface')
        if sha256(checkpoint) != contract['checkpoint_sha256'] or sha256(tokenizer) != contract['tokenizer_sha256']:
            raise ValueError('Checkpoint or tokenizer does not match the serving contract')
        if contract['upstream_commit'] != RLINF_COMMIT:
            raise ValueError('Unexpected model backend')
        if min(contract['max_token_len'], contract['inference_steps']) < 1:
            raise ValueError('Context length and inference steps must be positive')
        self.contract = contract
        self.codec = TrainingV1Codec(contract['statistics'], contract['token_layout'], contract['action_horizon'])
        self.layout = self.codec.layout
        self.device = torch.device(device)
        torch.set_num_threads(4)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        self.model, self.Observation = load_model(source, checkpoint, self.layout.token_horizon,
                                                 contract['max_token_len'], self.device)
        self.model.eval()
        self.lock = threading.Lock()

    def infer(self, observation, noise):
        import torch
        if observation['hand'] not in self.contract['hands'] or observation['task'] not in self.contract['tasks']:
            raise ValueError('Hand/task is outside the checkpoint training scope')
        item = serving_item(self.codec, observation)
        tokens = self.layout.noise(noise)
        with self.lock, torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            obs = model_observation([item], self.tokenizer, self.contract['max_token_len'], self.Observation, self.device)
            predicted = self.model.sample_actions(obs, num_steps=self.contract['inference_steps'],
                noise=torch.as_tensor(tokens[None], device=self.device)).float().cpu().numpy()[0]
        return {'actions': self.codec.actions(predicted, observation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--run', required=True, help='Training output with model.safetensors and contract.json')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=8200)
    parser.add_argument('--bind', default='127.0.0.1')
    args = parser.parse_args()
    run = Path(args.run)
    contract = json.loads((run / 'contract.json').read_text())
    policy = TrainingV1Policy(args.source, run / 'model.safetensors', args.tokenizer, contract, args.device)
    sys.path.insert(0, str(ROOT / 'evaluation'))
    from serve80 import handler_for
    from websockets.sync.server import serve
    metadata = dict(policy_kind='pi05_training_v1', token_layout=contract['token_layout'],
                    trained_hands=contract['hands'], trained_tasks=contract['tasks'],
                    diagnostic_only=contract['diagnostic_only'], training_steps=contract['training_steps'],
                    training_dataset_version=contract['dataset']['dataset_version'],
                    training_archive_sha256=contract['dataset']['sha256'],
                    checkpoint_sha256=contract['checkpoint_sha256'],
                    contract_sha256=sha256(run / 'contract.json'), upstream_commit=RLINF_COMMIT)
    with serve(handler_for(policy, 'arm40_v1', 'native', policy.layout.horizon, metadata),
               args.bind, args.port, compression=None, max_size=None) as server:
        print(f'Training v1 pi0.5 server listening on {args.bind}:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
