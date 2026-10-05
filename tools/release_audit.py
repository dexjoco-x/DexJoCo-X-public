#!/usr/bin/env python3
"""Audit this fresh public source snapshot without claiming asset license clearance."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from runtime_assets import check

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {'RELEASE_MANIFEST.json', 'RELEASE_AUDIT.json'}
EXCEPTIONS = {'evaluation/codecs/faas7_project_v1.json', 'representations/table1_sided_v3/codecs/faas7_project_v3_sided.json'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    paths = sorted(ROOT / name for name in set(filter(None, names)) if Path(name).name not in GENERATED)
    errors = []
    secrets = [re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
               re.compile(r'AKIA[0-9A-Z]{16}'),
               re.compile(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}'),
               re.compile(r'(?i)(?:password|passwd|api[_-]?key|secret)\s*[:=]\s*[\"\'][^\"\']{8,}[\"\']')]
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        if name.startswith('training/') or name in {'tools/pipeline_smoke.py', 'tests/test_foundation_interfaces.py'}:
            errors.append('Internal model training content: ' + name)
        if path.is_symlink(): errors.append('Symlink: ' + name)
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.npz', '.pyc', '.zip', '.gz', '.tar', '.safetensors', '.onnx'}:
            errors.append('Private weight/data or generated archive: ' + name)
        if name.startswith('simulator/teleoperation/') and (path.suffix == '.py' or any(part in {'retarget', 'configs', 'bridge', 'rokoko', 'vive_bridge'} for part in path.relative_to(ROOT / 'simulator/teleoperation').parts)):
            errors.append('Glove mapping or input bridge: ' + name)
        if path.stat().st_size >= 100_000_000: errors.append('Oversized Git object: ' + name)
        try: content = path.read_text()
        except (UnicodeDecodeError, OSError): continue
        if path.suffix == '.py':
            try:
                tree = ast.parse(content, filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or '']
                    else:
                        modules = []
                    if any(module == 'training' or module.startswith('training.') for module in modules):
                        errors.append('Dependency on private training module: ' + name)
                    if isinstance(node, ast.FunctionDef) and node.name in {'expand_pi05_action_head', 'pi05_collate', 'make_pi05_tokenizer'}:
                        errors.append('Legacy model training helper: ' + name)
            except SyntaxError: errors.append('Python syntax: ' + name)
        if path.resolve() != Path(__file__).resolve() and name not in EXCEPTIONS and ('/home/ubuntu' in content or '/data/hdd' in content):
            errors.append('Machine-specific path: ' + name)
        if any(pattern.search(content) for pattern in secrets): errors.append('Potential credential: ' + name)
    assets = check(ROOT, json.loads((ROOT / 'assets/LOCAL_ASSET_MANIFEST.json').read_text()))
    if not assets['passed']: errors.append('Runtime assets missing or changed')
    for model in json.loads((ROOT / 'evaluation/models.json').read_text()):
        for field, hash_field in (('portable_model', 'portable_xml_sha256'), ('contract', 'contract_sha256')):
            path = (ROOT / 'evaluation' / model[field]).resolve()
            if not path.is_file() or digest(path) != model[hash_field]: errors.append('Frozen model/contract mismatch: ' + model['task'] + '/' + model['hand'])
    asset_license = json.loads((ROOT / 'docs/ASSET_RELEASE_REPORT.json').read_text())
    if not asset_license['integrity_passed']: errors.append('Asset provenance integrity')
    decision = json.loads((ROOT / 'docs/PUBLIC_RELEASE_DECISION.json').read_text())
    result = {'passed_static_checks': not errors, 'files': len(paths), 'runtime_asset_integrity': assets,
              'model_training_implementation_included': False,
              'glove_mappings_public': False, 'weights_public': False, 'complete_demonstrations_in_git': False,
              'third_party_license_review_complete': asset_license['runtime_assets_cleared'],
              'asset_license_counts': asset_license['counts'], 'new_upstream_permissions_obtained': False,
              'publication_requested_by_project_owner': decision['publication_requested_by_project_owner'],
              'current_visibility': decision['current_visibility'],
              'publication_paused_by_project_owner': decision['publication_paused_by_project_owner'], 'errors': errors}
    (ROOT / 'RELEASE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'RELEASE_MANIFEST.json').write_text(json.dumps({'algorithm': 'sha256', 'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}}, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
