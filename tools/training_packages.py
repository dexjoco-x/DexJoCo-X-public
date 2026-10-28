"""Download and verify the single current benchmark bundle, including both readers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts
            or str(relative) != name or name == '.'):
        raise ValueError('Invalid package path: ' + name)
    root = Path(root).absolute()
    target = root / relative
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('Symlinks are not supported: ' + str(target))
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes data directory: ' + name)
    return target


def inventory(index):
    rows = index['files']
    expected = {r['path']: r for r in rows}
    if len(expected) != len(rows) or 'SHA256SUMS' in expected:
        raise ValueError('Duplicate or reserved inventory path')
    for name, row in expected.items():
        safe_path(Path('/'), name)
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) or row['bytes'] < 0:
            raise ValueError('Invalid file inventory: ' + name)
    return expected


def checksum_text(index):
    return ''.join(row['sha256'] + '  ' + row['path'] + '\n'
                   for row in index['files'])


def verify(root, index):
    missing, changed = [], []
    expected = inventory(index)
    for name, row in expected.items():
        path = safe_path(root, name)
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            changed.append(name)
    checksums = safe_path(root, 'SHA256SUMS')
    if not checksums.is_file():
        missing.append('SHA256SUMS')
    elif checksums.read_text() != checksum_text(index):
        changed.append('SHA256SUMS')
    return dict(passed=not missing and not changed, dataset_version=index['dataset_version'],
                tasks=list(index['tasks']), episodes=index['episodes'], frames=index['frames'],
                files_checked=len(expected) + 1, missing=missing, changed=changed)


def check_destination(destination, index):
    safe_path(destination, 'SHA256SUMS')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        catalog = safe_path(destination, 'bundle/catalog.json')
        expected = next(r for r in index['files'] if r['path'] == 'bundle/catalog.json')
        if not catalog.is_file() or sha(catalog) != expected['sha256']:
            raise FileExistsError('Use a new empty directory for Training v1; existing data is unchanged: '
                                  + str(destination))
        report = verify(destination, index)
        if not report['passed']:
            raise FileExistsError('Existing Training v1 files differ or are incomplete; use a new directory')
        return report
    return None


def extract(archive_path, destination, index):
    destination = Path(destination).absolute()
    previous = check_destination(destination, index)
    if previous:
        return previous
    archive_path = Path(archive_path)
    if (archive_path.is_symlink() or archive_path.stat().st_size != index['bytes']
            or sha(archive_path) != index['sha256']):
        raise ValueError('Archive checksum mismatch: ' + str(archive_path))
    expected = inventory(index)
    accepted = set(expected) | {'SHA256SUMS'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dexjoco_training_', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'data'
        staging.mkdir()
        seen = set()
        with tarfile.open(archive_path, 'r:') as archive:
            for member in archive:
                if member.name in seen or member.name not in accepted:
                    raise ValueError('Unexpected or duplicate archive member: ' + member.name)
                target = safe_path(staging, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                size = (expected[member.name]['bytes'] if member.name in expected
                        else len(checksum_text(index).encode()))
                if member.isfile():
                    if member.size != size:
                        raise ValueError('Archive member size mismatch: ' + member.name)
                    with archive.extractfile(member) as source, target.open('xb') as out:
                        shutil.copyfileobj(source, out, 4 << 20)
                elif member.islnk():
                    # Only links to already extracted, checksum-equivalent ordinary
                    # files are allowed. No symlinks, external or forward links.
                    if (member.name not in expected or member.linkname not in seen
                            or member.linkname not in expected
                            or any(expected[member.name][key] != expected[member.linkname][key]
                                   for key in ('bytes', 'sha256'))):
                        raise ValueError('Invalid hard link: ' + member.name)
                    os.link(safe_path(staging, member.linkname), target)
                else:
                    raise ValueError('Unsupported archive entry: ' + member.name)
                seen.add(member.name)
        if seen != accepted:
            raise ValueError('Missing archive members')
        report = verify(staging, index)
        if not report['passed']:
            raise ValueError('Extracted file checksum mismatch')
        # Preserve an existing data directory if another process populated it.
        if destination.exists():
            destination.rmdir()  # succeeds only for an empty directory
        os.rename(staging, destination)
    return report


def run(args, index):
    hub = index['huggingface']
    if not re.fullmatch(r'[0-9a-f]{40}', hub['revision']):
        raise ValueError('A pinned dataset commit is required')
    if args.command == 'list' or args.dry_run:
        return dict(dry_run=args.dry_run, dataset_version=index['dataset_version'],
                    tasks=list(index['tasks']), episodes=index['episodes'], frames=index['frames'],
                    transport='archive', bytes=index['bytes'], download_files=1,
                    revision=hub['revision'], archive=hub['path_prefix'] + index['archive'],
                    note='Training v1 is one bundle containing all six tasks and both prepared readers.')
    if args.data_root is None:
        raise ValueError('--data-root is required')
    if args.command == 'verify':
        return verify(args.data_root, index)
    if args.command == 'extract':
        if args.archives_root is None:
            raise ValueError('--archives-root is required')
        return extract(args.archives_root / index['archive'], args.data_root, index)
    if args.command != 'download' or args.transport == 'files':
        raise ValueError('Training v1 is distributed as one archive; use --transport auto or archive')
    previous = check_destination(args.data_root.absolute(), index)
    if previous:
        return previous
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install './simulator[download]' first") from error
    cache = args.archives_root or args.data_root.absolute().parent / (args.data_root.name + '_archives')
    filename = hub['path_prefix'] + index['archive']
    safe_path(cache, filename)
    print('Downloading the complete Training v1 bundle (six tasks, float64 + LeRobot/Being-H).', flush=True)
    path = hf_hub_download(repo_id=hub['repo_id'], repo_type='dataset', revision=hub['revision'],
                           filename=filename, local_dir=cache)
    return extract(Path(path), args.data_root, index)
