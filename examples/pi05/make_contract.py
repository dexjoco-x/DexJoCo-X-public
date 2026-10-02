"""Bind user-supplied statistics and action layout to an external checkpoint."""
import argparse
import json
from pathlib import Path
from .adapter import Native32, validate_statistics
from .serve import ROOT, sha256


def main():
    protocol = json.loads((ROOT / 'evaluation/eval_protocol.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--statistics', required=True, help='Matching training quantiles; state/action with q01/q99')
    parser.add_argument('--token-layout', required=True, choices=['single_right32', 'ego_interleave32'])
    parser.add_argument('--action-horizon', type=int, required=True, help='Physical steps, not token count')
    parser.add_argument('--max-token-len', type=int, default=384)
    parser.add_argument('--inference-steps', type=int, default=5)
    parser.add_argument('--hands', nargs='+', choices=protocol['hands'], required=True)
    parser.add_argument('--tasks', nargs='+', choices=protocol['tasks'], required=True)
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    layout = Native32(args.token_layout, args.action_horizon)
    statistics = json.loads(Path(args.statistics).read_text())
    validate_statistics(statistics, layout.state_dim)
    if min(args.max_token_len, args.inference_steps) < 1:
        raise ValueError('Context and inference steps must be positive')
    if args.token_layout == 'single_right32' and any(task.startswith('bimanual_') for task in args.tasks):
        raise ValueError('The single-arm token layout cannot serve bimanual tasks')
    result = dict(schema='dexjocox_pi05_native32_v1', token_layout=args.token_layout,
                  action_horizon=args.action_horizon, max_token_len=args.max_token_len,
                  inference_steps=args.inference_steps, hands=args.hands, tasks=args.tasks,
                  checkpoint_sha256=sha256(args.checkpoint), tokenizer_sha256=sha256(args.tokenizer),
                  statistics=statistics, diagnostic_only=args.diagnostic_only)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve previously bound contracts rather than overwriting them.
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
