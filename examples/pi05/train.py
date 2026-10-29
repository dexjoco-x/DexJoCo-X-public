"""Minimal action-expert fine-tuning on the released Training v1 labels."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .serve import ROOT, RLINF_COMMIT, import_inference, sha256
from .training_v1 import SCHEMA, TrainingV1Dataset, model_observation, package_identity


def load_model(source, checkpoint, horizon, max_token_len, device):
    import torch
    Pi0, Pi0Config, Observation = import_inference(source)
    config = Pi0Config(pi05=True, action_dim=32, action_horizon=horizon,
                      max_token_len=max_token_len, dtype='bfloat16')
    model = Pi0(config, num_steps=5, action_env_dim=32, action_chunk=horizon)
    if Path(checkpoint).suffix == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(str(checkpoint))
    else:
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.to_bfloat16_for_selected_params('bfloat16')
    return model.to(device), Observation


def masked_flow_loss(per_element, mask):
    if per_element.shape != mask.shape or not mask.any():
        raise ValueError('Flow loss requires matching, nonempty action masks')
    return per_element.masked_select(mask).mean()


def main(args):
    import sentencepiece
    import torch
    from safetensors.torch import save_file
    from evaluation.provenance import evaluation_provenance

    if min(args.steps, args.batch_size, args.accumulation_steps, args.action_horizon,
           args.max_token_len, args.inference_steps, args.log_every) < 1 or args.lr <= 0 or args.warmup_steps < 0:
        raise ValueError('Training counts and learning rate must be positive; warmup must be nonnegative')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new training output directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('This training example requires a CUDA GPU with bfloat16 support')
    data = TrainingV1Dataset(args.dataset_root, args.hand, args.tasks,
                             args.token_layout, args.action_horizon)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    model, Observation = load_model(args.source, args.checkpoint, data.codec.layout.token_horizon,
                                    args.max_token_len, device)
    model.train()
    model.freeze_vlm()  # Freeze vision and language expert; train the action expert and projections.
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()  # FP32 optimizer weights; BF16 autocast for compute.
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    configuration = dict(vars(args))
    configuration.update(dataset=package_identity(), upstream_commit=RLINF_COMMIT,
                         initialization_sha256=sha256(args.checkpoint), tokenizer_sha256=sha256(args.tokenizer),
                         training_episodes=len(data.rows), training_frames=len(data),
                         trainable_parameters=sum(p.numel() for p in parameters),
                         trainable_modules='action expert and action/time projections',
                         image_augmentation=False, provenance=evaluation_provenance(ROOT))
    output.mkdir(parents=True)
    (output / 'run.json').write_text(json.dumps(configuration, indent=2) + '\n')
    start = time.monotonic()
    with (output / 'loss.jsonl').open('w') as log:
        for step in range(args.steps):
            warmup = min(args.warmup_steps, args.steps)
            progress = max(0, step - warmup) / max(1, args.steps - warmup - 1)
            factor = (step + 1) / warmup if step < warmup else 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(args.accumulation_steps):
                items = [data[int(index)] for index in rng.integers(len(data), size=args.batch_size)]
                observation = model_observation(items, tokenizer, args.max_token_len, Observation, device)
                actions = torch.as_tensor(np.stack([i['actions'] for i in items]), device=device)
                mask = torch.as_tensor(np.stack([i['action_mask'] for i in items]), device=device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = masked_flow_loss(model.compute_loss(observation, actions, train=False), mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                (loss / args.accumulation_steps).backward()
                total_loss += float(loss.detach()) / args.accumulation_steps
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step + 1, loss=total_loss, grad_norm=float(grad_norm),
                          lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic() - start)
            log.write(json.dumps(record) + '\n'); log.flush()
            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                print(json.dumps(record), flush=True)
    checkpoint = output / 'model.safetensors'
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(checkpoint))
    contract = dict(schema=SCHEMA, token_layout=args.token_layout, action_horizon=args.action_horizon,
                    max_token_len=args.max_token_len, inference_steps=args.inference_steps,
                    hands=[args.hand], tasks=args.tasks, statistics=data.stats,
                    checkpoint_sha256=sha256(checkpoint), tokenizer_sha256=sha256(args.tokenizer),
                    dataset=package_identity(), training_steps=args.steps, diagnostic_only=args.diagnostic,
                    upstream_commit=RLINF_COMMIT)
    (output / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(f'Saved model and serving contract to {output}', flush=True)


if __name__ == '__main__':
    protocol = json.loads((ROOT / 'evaluation/eval_protocol.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Pinned, clean external RLinf checkout')
    parser.add_argument('--checkpoint', required=True, help='Complete RLinf pi0.5 base model state_dict')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--dataset-root', required=True, help='Extracted Training v1 directory containing bundle and prepared')
    parser.add_argument('--hand', choices=protocol['hands'], required=True)
    parser.add_argument('--tasks', nargs='+', choices=protocol['tasks'], default=protocol['tasks'])
    parser.add_argument('--token-layout', choices=['ego_interleave32', 'single_right32'], default='ego_interleave32')
    parser.add_argument('--action-horizon', type=int, default=25)
    parser.add_argument('--max-token-len', type=int, default=384)
    parser.add_argument('--inference-steps', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2.5e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--diagnostic', action='store_true', help='Mark a short integration run as diagnostic only')
    main(parser.parse_args())
