#!/usr/bin/env python3
"""Check or install user-supplied frozen assets; never download or publish assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('simulator/dexjoco/sim/envs/xmls/', 'simulator/teleoperation/')
PRIVATE_CODECS = {'evaluation/codecs/selected.pt', 'representations/table1_sided_v3/codecs/selected.pt'}


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or not (name.startswith(PREFIXES) or name in PRIVATE_CODECS):
        raise ValueError('Invalid asset path: ' + name)
    return path


def target_path(root, name):
    safe_relative(name)
    path = root / name
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
        raise ValueError('Symlink target is not permitted: ' + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes asset root: ' + name)
    return path


def check(root, manifest):
    missing, changed = [], []
    for row in manifest['files']:
        path = target_path(root, row['path'])
        if not path.is_file():
            missing.append(row['path'])
        elif sha(path) != row['sha256']:
            changed.append(row['path'])
    return dict(passed=not missing and not changed, checked=len(manifest['files']), missing=missing, changed=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'export-private', 'install-local'])
    parser.add_argument('--manifest', type=Path, default=ROOT / 'assets/LOCAL_ASSET_MANIFEST.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    names = [row['path'] for row in manifest['files']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate paths in asset manifest')
    for name in names:
        safe_relative(name)
    root = args.root.resolve()
    if args.command == 'check':
        report = check(root, manifest)
    elif args.command == 'export-private':
        if args.bundle is None:
            parser.error('--bundle is required')
        report = check(root, manifest)
        if not report['passed']:
            raise ValueError('Source assets failed integrity check')
        if args.bundle.exists():
            raise FileExistsError('Refusing to overwrite bundle')
        with tarfile.open(args.bundle, 'w:gz') as archive:
            for row in manifest['files']:
                archive.add(target_path(root, row['path']), arcname=row['path'], recursive=False)
        report = dict(passed=True, exported=len(names), bytes=args.bundle.stat().st_size,
                      private_bundle=True, redistribution_authorized=False)
    else:
        if (args.bundle is None) == (args.source_root is None):
            parser.error('Specify exactly one of --bundle or --source-root')
        # Verify all inputs and existing destinations before writing anything.
        conflicts = [name for name in names if target_path(root, name).exists()
                     and (not target_path(root, name).is_file()
                          or sha(target_path(root, name)) != next(row['sha256'] for row in manifest['files'] if row['path'] == name))]
        if conflicts:
            raise ValueError('Refusing to overwrite different existing assets: ' + ', '.join(conflicts[:5]))
        with tempfile.TemporaryDirectory(prefix='dexjocox-assets-') as directory:
            stage = Path(directory)
            if args.source_root:
                for row in manifest['files']:
                    source = target_path(args.source_root.resolve(), row['path'])
                    if not source.is_file() or sha(source) != row['sha256']:
                        raise ValueError('Source asset missing or changed: ' + row['path'])
                    destination = stage / row['path']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
            else:
                with tarfile.open(args.bundle, 'r:*') as archive:
                    members = archive.getmembers()
                    if len(members) != len(names) or {item.name for item in members} != set(names):
                        raise ValueError('Bundle must contain exactly the manifest files')
                    wanted = {row['path']: row for row in manifest['files']}
                    for member in members:
                        safe_relative(member.name)
                        if not member.isfile() or member.size != wanted[member.name]['bytes']:
                            raise ValueError('Unexpected bundle member: ' + member.name)
                        destination = stage / member.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, destination.open('wb') as output:
                            shutil.copyfileobj(source, output)
            if not check(stage, manifest)['passed']:
                raise ValueError('Staged assets failed hash verification; installation cancelled')
            installed = 0
            for row in manifest['files']:
                destination = target_path(root, row['path'])
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.asset-', dir=destination.parent)
                try:
                    with os.fdopen(fd, 'wb') as output, (stage / row['path']).open('rb') as source:
                        shutil.copyfileobj(source, output)
                    os.replace(temporary, destination)
                finally:
                    if Path(temporary).exists():
                        Path(temporary).unlink()
                installed += 1
            report = {**check(root, manifest), 'installed': installed,
                      'permission_scope': 'Technical verification only; user must have permission for local asset use.'}
    print(json.dumps(report, indent=2))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
