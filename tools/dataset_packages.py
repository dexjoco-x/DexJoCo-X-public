#!/usr/bin/env python3
"""Download, verify or safely extract the current DexJoCo-X benchmark bundle."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / 'dataset/PACKAGES.json'
HANDS = {'allegro', 'inspire', 'leap', 'linkerhand_l20', 'sharpa_wave', 'wuji', 'xhand'}
TASKS = ('hammer_nail', 'pick_bucket', 'single_hand_hanoi', 'bimanual_microwave_cook',
         'bimanual_photograph', 'bimanual_unlock_ipad')
REVISION = '9f48e1404168227a22586af0cffb8371dbb7e1f1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or str(relative) != name
            or not name.startswith(('bundle/', 'metadata/'))
            or relative.suffix not in {'.json', '.npz', '.mp4', '.md'}):
        raise ValueError('Invalid dataset path: ' + name)
    root = Path(root).absolute()
    target = root / name
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('Symlinks are not supported: ' + name)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes data directory: ' + name)
    return target


def selected_tasks(values):
    tasks = list(TASKS) if 'all' in values else list(dict.fromkeys(values))
    if not tasks or not set(tasks) <= set(TASKS):
        raise ValueError('Select one or more of the six benchmark tasks')
    return tasks


def validate_rows(rows, task):
    if len(rows) != 350 or Counter(r['hand'] for r in rows) != Counter({h: 50 for h in HANDS}):
        raise ValueError('Expected seven hands and 50 demonstrations per hand: ' + task)
    if any(r['task'] != task or r['split'] != 'train' for r in rows):
        raise ValueError('Wrong task or split: ' + task)
    if len({(r['hand'], r['scene_id']) for r in rows}) != 350:
        raise ValueError('Duplicate episode: ' + task)
    if task == 'bimanual_microwave_cook':
        if any(r.get('mount_revision') != 'xhand_left_mount_rx180_v1'
               for r in rows if r['hand'] == 'xhand'):
            raise ValueError('Uncorrected XHand microwave data')


def source_files(source, task, checksum_index):
    manifest = 'bundle/manifests/' + task + '.json'
    rows = read(safe_path(source, manifest))
    validate_rows(rows, task)
    frozen = read(ROOT / 'dataset/catalog.json')['manifest_sha256'][task + '.json']
    if sha(safe_path(source, manifest)) != frozen:
        raise ValueError('Source manifest does not match the frozen corrected dataset')
    expected = {manifest: frozen}
    for row in rows:
        for key in ('tensor', 'contract'):
            expected['bundle/' + row[key]] = row[key + '_sha256']
        for video in row['videos'].values():
            expected['bundle/' + video['path']] = video['sha256']
        if row.get('camera_audit_path'):
            expected['bundle/' + row['camera_audit_path']] = row['camera_audit_sha256']
    catalog = read(source / 'bundle/catalog.json')
    common = ['bundle/catalog.json', 'metadata/VERIFICATION.json', 'metadata/CAMERA_PROFILE.md']
    if catalog.get('native_replay_report'):
        common.append(catalog['native_replay_report'])
    common.extend(catalog.get('quality_reports', []))
    for name in common:
        expected[name] = checksum_index[name]
    if task == 'bimanual_microwave_cook' and catalog.get('package_version', 'v1.0') == 'v1.0':
        for name in ('collection_protocol.json', 'episode_checks.json', 'release.json', 'runtime_manifest.json'):
            relative = 'bundle/evidence/xhand_microwave_corrected_v9/' + name
            expected[relative] = checksum_index[relative]
    files = []
    for name, digest in sorted(expected.items()):
        if checksum_index.get(name) != digest:
            raise ValueError('Frozen checksum records disagree: ' + name)
        path = safe_path(source, name)
        if not path.is_file():
            raise FileNotFoundError(name)
        files.append({'path': name, 'sha256': digest, 'bytes': path.stat().st_size})
    return rows, files


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def build(args, tasks):
    source, output = args.source_root.absolute(), args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    catalog = read(source / 'bundle/catalog.json')
    package_version = catalog.get('package_version', 'v1.0')
    if not re.fullmatch(r'v[0-9]+\.[0-9]+', package_version):
        raise ValueError('Invalid package version in dataset catalog')
    checksums = {}
    for line in (source / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        checksums[name] = digest
    index = {'schema_version': 1, 'dataset_version': catalog.get('dataset_version', 'dataset-v1.0-corrected'),
             'created': catalog.get('release_date', '2026-10-02'), 'episodes': catalog['episodes'], 'frames': catalog['frames'], 'split': 'train',
             'weights_included': False, 'glove_mappings_included': False,
             'prepared_views_included': False,
             'huggingface': {'repo_id': 'darenrenjian/DexJoCo-X', 'repo_type': 'dataset',
                             'revision': args.revision or (REVISION if package_version == 'v1.0' else None)},
             'archives_uploaded_to_hub': False, 'tasks': {}}
    for task in tasks:
        rows, files = source_files(source, task, checksums)
        filename = 'DexJoCo-X_' + task + '_' + package_version + '.tar'
        archive_path = output / filename
        if archive_path.exists():
            raise FileExistsError(archive_path)
        temporary = archive_path.with_suffix('.tar.incomplete')
        # Payloads are already compressed (NPZ/H.264); TAR preserves ordinary
        # files and avoids wasting time recompressing videos or creating links.
        with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT) as archive:
            for number, row in enumerate(files, 1):
                path = safe_path(source, row['path'])
                info = tarfile.TarInfo(row['path'])
                info.size = row['bytes']
                info.mode = 0o644
                with path.open('rb') as stream:
                    reader = HashingReader(stream)
                    archive.addfile(info, reader)
                    if reader.digest.hexdigest() != row['sha256']:
                        raise ValueError('Source checksum mismatch: ' + row['path'])
                if number % 350 == 0:
                    print(task, number, '/', len(files), 'files checked and packed', flush=True)
        temporary.replace(archive_path)
        index['tasks'][task] = {'archive': filename, 'sha256': sha(archive_path),
                                'archive_bytes': archive_path.stat().st_size,
                                'payload_bytes': sum(r['bytes'] for r in files),
                                'episodes': 350, 'hands': sorted(HANDS),
                                'frames': sum(r['steps'] for r in rows), 'files': files}
        write(output / 'PACKAGES.json', index)
        print(task, 'PACKAGE_READY', archive_path.stat().st_size, 'bytes', flush=True)
    (output / 'SHA256SUMS').write_text(''.join(
        value['sha256'] + '  ' + value['archive'] + '\n' for value in index['tasks'].values())
        + sha(output / 'PACKAGES.json') + '  PACKAGES.json\n')
    return {'passed': True, 'tasks': tasks, 'episodes': sum(index['tasks'][t]['episodes'] for t in tasks),
            'archive_bytes': sum(index['tasks'][t]['archive_bytes'] for t in tasks)}


def expected_files(index, tasks):
    files = {}
    for task in tasks:
        for row in index['tasks'][task]['files']:
            if row['path'] in files and files[row['path']] != row:
                raise ValueError('Task package definitions disagree')
            files[row['path']] = row
    return files


def verify(root, index, tasks):
    missing, changed = [], []
    files = expected_files(index, tasks)
    for name, row in files.items():
        path = safe_path(root, name)
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            changed.append(name)
    if not missing and not changed:
        for task in tasks:
            rows = read(safe_path(root, 'bundle/manifests/' + task + '.json'))
            validate_rows(rows, task)
    return {'passed': not missing and not changed, 'tasks': tasks, 'files_checked': len(files),
            'episodes': 350 * len(tasks), 'missing': missing, 'changed': changed}


def extract(args, index, tasks):
    destination = args.data_root.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_files(index, tasks)
    for name, row in expected.items():
        target = safe_path(destination, name)
        if target.exists() and (not target.is_file() or sha(target) != row['sha256']):
            raise FileExistsError('Existing data would change: ' + name)
    # Check every archive before staging any payload or changing the data root.
    for task in tasks:
        entry = index['tasks'][task]
        archive_path = args.archives_root / entry['archive']
        if archive_path.is_symlink() or archive_path.stat().st_size != entry['archive_bytes'] or sha(archive_path) != entry['sha256']:
            raise ValueError('Archive checksum mismatch: ' + entry['archive'])
    with tempfile.TemporaryDirectory(prefix='dexjoco_data_stage_', dir=destination.parent) as temporary:
        staging = Path(temporary)
        for task in tasks:
            entry = index['tasks'][task]
            names = {row['path'] for row in entry['files']}
            with tarfile.open(args.archives_root / entry['archive'], 'r:') as archive:
                members = archive.getmembers()
                if len(members) != len(names) or {m.name for m in members} != names:
                    raise ValueError('Unexpected or duplicate archive members')
                for member in members:
                    if not member.isfile() or member.size != expected[member.name]['bytes']:
                        raise ValueError('Only exact ordinary data files may be extracted')
                    path = safe_path(staging, member.name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as stream, path.open('wb') as target:
                        shutil.copyfileobj(stream, target, 4 << 20)
        result = verify(staging, index, tasks)
        if not result['passed']:
            raise ValueError('Extracted data checksum mismatch')
        for name in expected:
            target = safe_path(destination, name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(safe_path(staging, name), target)
    return result


def download(args, index, tasks):
    files = expected_files(index, tasks)
    use_archives = args.transport == 'archive' or (
        args.transport == 'auto' and index.get('archives_uploaded_to_hub', False))
    if use_archives and not index.get('archives_uploaded_to_hub', False):
        raise ValueError('This index does not contain uploaded task archives; use --transport files')
    hub = index['archive_huggingface'] if use_archives else index['huggingface']
    if not re.fullmatch(r'[0-9a-f]{40}', hub.get('revision') or ''):
        raise ValueError('The package index needs a pinned published dataset commit')
    if args.dry_run:
        return {'dry_run': True, 'tasks': tasks, 'files': len(files),
                'transport': 'archive' if use_archives else 'files',
                'download_files': len(tasks) if use_archives else len(files),
                'bytes': sum(index['tasks'][t]['archive_bytes'] for t in tasks) if use_archives
                         else sum(r['bytes'] for r in files.values()),
                'revision': hub['revision']}
    root = args.data_root.absolute()
    for name, row in files.items():
        target = safe_path(root, name)
        if target.exists() and (not target.is_file() or sha(target) != row['sha256']):
            raise FileExistsError('Existing data would change: ' + name)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install './simulator[download]' first") from error
    if use_archives:
        cache = args.archives_root or root.parent / (root.name + '_archives')
        if any(p.is_symlink() for p in (cache, *cache.parents)):
            raise ValueError('Archive cache must not contain symlink parents')
        prefix = hub['path_prefix']
        if not re.fullmatch(r'task_packages/v[0-9]+\.[0-9]+/', prefix):
            raise ValueError('Unexpected archive directory in the package index')
        for task in tasks:
            entry = index['tasks'][task]
            path = cache / prefix / entry['archive']
            if path.exists() and (path.is_symlink() or not path.is_file()
                                  or path.stat().st_size != entry['archive_bytes']
                                  or sha(path) != entry['sha256']):
                raise FileExistsError('Existing archive checksum conflict: ' + entry['archive'])
        snapshot_download(repo_id=hub['repo_id'], repo_type='dataset', revision=hub['revision'],
                          local_dir=cache, allow_patterns=[prefix + index['tasks'][t]['archive']
                                                          for t in tasks], max_workers=args.workers)
        from types import SimpleNamespace
        report = extract(SimpleNamespace(data_root=root, archives_root=cache / prefix), index, tasks)
        report.update(transport='archive', archive_revision=hub['revision'], archives=len(tasks))
    else:
        snapshot_download(repo_id=hub['repo_id'], repo_type='dataset', revision=hub['revision'],
                          local_dir=root, allow_patterns=list(files), max_workers=args.workers)
        report = verify(root, index, tasks)
        report.update(transport='files', source_revision=hub['revision'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'list', 'verify', 'extract', 'download'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--tasks', nargs='+', choices=(*TASKS, 'all'), default=['all'])
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--revision', help='Published raw-data commit for a newly built package index')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--archives-root', type=Path)
    parser.add_argument('--transport', choices=('auto', 'archive', 'files'), default='auto',
                        help='Download TAR packages and extract, or download individual raw files')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.dry_run and args.command != 'download':
        parser.error('--dry-run is supported by download only')
    tasks = selected_tasks(args.tasks)
    if args.command == 'build':
        if args.source_root is None or args.output is None:
            parser.error('build requires --source-root and --output')
        report = build(args, tasks)
    else:
        index = read(args.index)
        if index.get('schema_version') == 2:
            if __package__:
                from . import training_packages
            else:
                import training_packages
            report = training_packages.run(args, index)
            if args.report:
                write(args.report, report)
            print(json.dumps(report, indent=2))
            if report.get('passed') is False:
                raise SystemExit(1)
            return
        if not set(tasks) <= set(index['tasks']):
            parser.error('Index does not contain all selected tasks')
        if args.command == 'list':
            for task in tasks:
                entry = index['tasks'][task]
                print(f"{task}: {entry['episodes']} episodes, {entry['archive_bytes']/1e9:.3f} GB, {entry['archive']}")
            return
        if args.data_root is None:
            parser.error('A data directory is required: --data-root')
        if args.command == 'verify':
            report = verify(args.data_root, index, tasks)
        elif args.command == 'download':
            report = download(args, index, tasks)
        else:
            if args.archives_root is None:
                parser.error('extract requires --archives-root')
            report = extract(args, index, tasks)
    if args.report:
        write(args.report, report)
    print(json.dumps(report, indent=2))
    if report.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
