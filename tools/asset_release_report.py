#!/usr/bin/env python3
"""Inventory files referenced by the 42 frozen models; fail closed on unknown terms."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
XMLS = 'simulator/dexjoco/sim/envs/xmls/'
# These scopes have retained notices and were documented in ASSET_LICENSE_AUDIT.
# This is preparation evidence, not an authorization to publish the repository.
DOCUMENTED = {
    XMLS + 'franka_emika_panda/': ('Apache-2.0', XMLS + 'franka_emika_panda/LICENSE'),
    XMLS + 'wonik_allegro/': ('BSD-2-Clause', XMLS + 'wonik_allegro/LICENSE'),
    XMLS + 'linkerhand_l20/': ('Apache-2.0', XMLS + 'linkerhand_l20/SOURCE_LICENSE'),
    'simulator/teleoperation/sharpa_wave/sharpa-urdf-usd-xml/':
        ('Apache-2.0', 'simulator/teleoperation/sharpa_wave/sharpa-urdf-usd-xml/LICENSE.txt'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/ASSET_RELEASE_REPORT.json')
    parser.add_argument('--require-ready', action='store_true', help='Exit 1 if any runtime asset is uncleared.')
    parser.add_argument('--exclusions-output', type=Path, default=ROOT / 'docs/ASSET_PUBLIC_EXCLUSIONS.json')
    args = parser.parse_args()
    exceptions = {}
    errors = []
    receipts = list((ROOT / 'licenses/third_party').glob('*/VERIFIED_FILES.json'))
    receipts.extend((ROOT / 'licenses/third_party').glob('*/DERIVED_FILES.json'))
    for receipt in sorted(receipts):
        verified = json.loads(receipt.read_text())
        for field in ('license_text', 'upstream_notice'):
            if field in verified and not (ROOT / verified[field]).is_file():
                errors.append('Missing notice: ' + verified[field])
        for path, expected in verified['notice_sha256'].items():
            if not (ROOT / path).is_file() or digest(ROOT / path) != expected:
                errors.append('Notice content changed: ' + path)
        if 'verification_tool' in verified:
            verifier = ROOT / verified['verification_tool']
            if not verifier.is_file() or digest(verifier) != verified['verification_tool_sha256']:
                errors.append('Derivation verifier changed: ' + verified['verification_tool'])
        for item in verified['files']:
            if item['path'] in exceptions:
                errors.append('Duplicate verification: ' + item['path'])
            exceptions[item['path']] = {**item, 'notice': verified['license_text']}
    for item in exceptions.values():
        path = ROOT / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            errors.append('Verified file changed: ' + item['path'])

    source_report = json.loads((ROOT / 'docs/ASSET_SOURCE_MATCHES.json').read_text())
    source_matches = {row['path']: row for row in source_report['files']}
    for path, expected in source_report['notice_sha256'].items():
        if not (ROOT / path).is_file() or digest(ROOT / path) != expected:
            errors.append('Source notice changed: ' + path)
    for rel, receipt in source_matches.items():
        path = ROOT / rel
        if not path.is_file() or digest(path) != receipt['sha256']:
            errors.append('Source-matched asset changed: ' + rel)
    project_report = json.loads((ROOT / 'docs/PROJECT_ASSET_PROVENANCE.json').read_text())
    project_assets = {row['path']: row for row in project_report['files']}
    for path, expected in project_report['notice_sha256'].items():
        if not (ROOT / path).is_file() or digest(ROOT / path) != expected:
            errors.append('Project provenance notice changed: ' + path)
    verifier = ROOT / project_report['verification_tool']
    if not verifier.is_file() or digest(verifier) != project_report['verification_tool_sha256']:
        errors.append('Project texture verifier changed')
    for rel, item in project_assets.items():
        if not (ROOT / rel).is_file() or digest(ROOT / rel) != item['sha256']:
            errors.append('Project asset changed: ' + rel)

    references = {}
    def inspect(path, cell, visited):
        if path in visited:
            return
        visited.add(path)
        tree = ET.parse(path)
        compiler = tree.find('compiler')
        settings = compiler.attrib if compiler is not None else {}
        for element in tree.iter():
            name = element.get('file')
            if not name:
                continue
            directory = settings.get('meshdir', '') if element.tag == 'mesh' else settings.get('texturedir', '') if element.tag == 'texture' else ''
            target = (path.parent / directory / name).resolve()
            try:
                rel = target.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append('Reference outside repository in ' + cell)
                continue
            references.setdefault(rel, set()).add(cell)
            if not target.is_file():
                errors.append('Missing asset: ' + rel)
            elif element.tag == 'include':
                inspect(target, cell, visited)

    models = json.loads((ROOT / 'evaluation/models.json').read_text())
    for model in models:
        inspect((ROOT / 'evaluation' / model['portable_model']).resolve(), model['task'] + '/' + model['hand'], set())
    rows = []
    for rel, cells in sorted(references.items()):
        path = ROOT / rel
        actual = digest(path) if path.is_file() else None
        status, license_name, notice = 'source-or-license-unresolved', None, None
        if rel in exceptions and actual == exceptions[rel]['sha256']:
            status, license_name, notice = exceptions[rel]['decision'], exceptions[rel]['license'], exceptions[rel]['notice']
        else:
            for prefix, (license_id, license_file) in DOCUMENTED.items():
                if rel.startswith(prefix) and (ROOT / license_file).is_file():
                    status, license_name, notice = 'documented-license', license_id, license_file
                    break
        row = dict(path=rel, sha256=actual, bytes=path.stat().st_size if path.is_file() else 0,
                   status=status, license=license_name, notice=notice, used_by=sorted(cells))
        if rel in project_assets and actual == project_assets[rel]['sha256']:
            row.update(status='documented-license' if project_report['project_asset_license'] == 'MIT' else 'project-license-pending',
                       license='MIT' if project_report['project_asset_license'] == 'MIT' else None,
                       notice='LICENSE' if project_report['project_asset_license'] == 'MIT' else None,
                       source_status='project-generated-pixel-exact-recipe', provenance='docs/PROJECT_ASSET_PROVENANCE.json')
        if rel in source_matches:
            row['source_evidence'] = source_matches[rel]['matches']
            row['source_status'] = 'pinned-upstream-content-match'
        rows.append(row)
    counts = dict(Counter(row['status'] for row in rows))
    ready = not errors and not any(counts.get(name, 0) for name in ('source-or-license-unresolved', 'project-license-pending'))
    report = dict(schema_version=1, scope='Asset files referenced by the 42 evaluation/models.json models; not the entire repository or dataset.',
                  models_checked=len(models), integrity_passed=not errors, runtime_assets_cleared=ready,
                  repository_publication_authorized=False,
                  unresolved_source_trace=dict(
                      matched_files=sum(row['status'] == 'source-or-license-unresolved' and 'source_evidence' in row for row in rows),
                      by_upstream=dict(Counter(match['repository'].rsplit('/', 1)[-1].lower()
                                               for row in rows if row['status'] == 'source-or-license-unresolved'
                                               for match in row.get('source_evidence', []))),
                      receipt='docs/ASSET_SOURCE_MATCHES.json',
                      meaning='Source identified; asset license is still pending.'),
                  note='Academic/noncommercial use is not a replacement for a redistribution grant. Unlisted files remain uncleared. Dataset RGB rights and private Git history require separate review.',
                  counts=counts, errors=errors, files=rows)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    excluded = [row for row in rows if row['status'] in ('source-or-license-unresolved', 'project-license-pending')]
    exclusions = dict(
        schema_version=1,
        scope='Uncleared asset files referenced by the 42 frozen formal environments; other repository files still need review.',
        action='License scope unresolved for these shipped files. Preserve source receipts and existing notices; no blanket MIT grant or upstream confirmation is asserted.',
        affected_pairs=sorted({cell for row in excluded for cell in row['used_by']}),
        runtime_available_without_excluded_files=False if excluded else True,
        frozen_model_changes_allowed=False,
        alternative='License review remains incomplete. This direct source release preserves these files and their recorded status at the project owner request; the exclusion list remains a license-review inventory, not a claim that files are absent.',
        files=excluded,
    )
    args.exclusions_output.write_text(json.dumps(exclusions, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ('models_checked', 'integrity_passed', 'runtime_assets_cleared', 'counts', 'unresolved_source_trace', 'errors')}, indent=2))
    return int(bool(errors) or (args.require_ready and not ready))


if __name__ == '__main__':
    raise SystemExit(main())
