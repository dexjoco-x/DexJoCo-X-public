#!/usr/bin/env python3
"""Check all local Git refs for excluded training code, mappings and weights."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {
    'tools/pipeline_smoke.py', 'tests/test_foundation_interfaces.py',
    'docs/TRAINING_AND_INFERENCE.md', 'docs/FOUNDATION_MODEL_VALIDATION.json',
    'docs/FOUNDATION_MODEL_VALIDATION.md', 'dataset/tools/check_adapters.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    commits = subprocess.check_output(['git', 'rev-list', '--all'], cwd=ROOT, text=True).splitlines()
    if not commits:
        raise ValueError('Commit the release snapshot before auditing history')
    paths = set()
    errors = []
    roots = []
    for commit in commits:
        row = subprocess.check_output(['git', 'rev-list', '--parents', '-n', '1', commit], cwd=ROOT, text=True).split()
        if len(row) == 1:
            roots.append(commit)
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '-z', commit], cwd=ROOT).decode().split('\0')
        for name in filter(None, names):
            paths.add(name)
            if name.startswith('training/') or name in FORBIDDEN_NAMES:
                errors.append(dict(commit=commit, path=name, reason='Internal model training content'))
            if Path(name).suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors', '.npz', '.onnx', '.tar', '.zip', '.gz'}:
                errors.append(dict(commit=commit, path=name, reason='Weight, episode or archive'))
            if name.startswith('simulator/teleoperation/') and Path(name).suffix == '.py':
                errors.append(dict(commit=commit, path=name, reason='Glove input or mapping code'))
    # The snapshot must have a single independent root and no old dangling objects.
    unreachable = subprocess.check_output(['git', 'fsck', '--full', '--unreachable', '--no-reflogs'],
                                          cwd=ROOT, text=True, stderr=subprocess.STDOUT).splitlines()
    dangling = [line for line in unreachable if line.startswith(('unreachable ', 'dangling '))]
    if len(roots) != 1:
        errors.append(dict(reason='Expected exactly one independent Git root'))
    if dangling:
        errors.append(dict(reason='Unreachable old Git objects', objects=dangling))
    result = dict(passed=not errors, commits_checked=len(commits), roots=roots,
                  historical_paths_checked=len(paths), unreachable_objects=dangling, errors=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
