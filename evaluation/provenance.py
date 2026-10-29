"""Identify the code and benchmark release behind an evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess


def evaluation_provenance(root):
    root = Path(root)
    package = json.loads((root / 'dataset/PACKAGES.json').read_text())
    record = {
        'benchmark_dataset_version': package['dataset_version'],
        'benchmark_archive_sha256': package['sha256'],
        'code_commit': None,
        'code_dirty': None,
    }
    try:
        top = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True,
            stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != root.resolve():
            raise OSError('Source archive is inside a different repository')
        record['code_commit'] = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL).strip()
        record['code_dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain'], text=True,
            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        pass  # Source archives need not contain a .git directory.
    record['evaluation_source_sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / 'evaluation').glob('*.py'))
    }
    return record


def validate_episode_count(episodes, diagnostic):
    if not 1 <= episodes <= 150:
        raise ValueError('--episodes must be between 1 and 150')
    if not diagnostic and episodes != 150:
        raise ValueError('Formal evaluation requires --episodes 150 (3 x 50 per hand/task); '
                         'use --diagnostic for shorter integration checks')
