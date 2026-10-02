#!/usr/bin/env python3
"""Verify geometry-preserving Inspire conversions against pinned CC-licensed sources.

Requires trimesh, NumPy and SciPy. Original GLBs are supplied separately; no
working model, mesh or dataset file is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'f5e7132f22108164577fea4c25ef99b5cc0e1900'
TOLERANCE = 2e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_check(source, target, reflected=False):
    vertices = np.array(target.vertices)
    if reflected:
        vertices[:, 1] *= -1
    tree = cKDTree(source.vertices)
    distance, mapped = tree.query(vertices)
    reverse, _ = cKDTree(vertices).query(source.vertices)
    # Canonicalize duplicate source vertices before comparing triangle multisets.
    _, canonical = tree.query(source.vertices)
    source_faces = Counter(map(tuple, np.sort(canonical[np.asarray(source.faces)], axis=1)))
    target_faces = Counter(map(tuple, np.sort(mapped[np.asarray(target.faces)], axis=1)))
    error = float(max(distance.max(), reverse.max()))
    if error > TOLERANCE or source_faces != target_faces:
        raise ValueError(f'Geometry conversion mismatch: distance={error}; triangle multisets equal={source_faces == target_faces}')
    return dict(max_bidirectional_vertex_error_m=error, triangle_multiset_equal=True,
                source_triangles=len(source.faces), target_triangles=len(target.faces),
                tolerance_m=TOLERANCE, mirror_axis='Y' if reflected else None)


def mirror_obj_bytes(source):
    lines = []
    for line in source.read_text().splitlines():
        pieces = line.split()
        if line.startswith(('v ', 'vn ')) and len(pieces) >= 4:
            x, y, z = map(float, pieces[1:4])
            lines.append(f'{pieces[0]} {x:.9g} {-y:.9g} {z:.9g}')
        elif line.startswith('f ') and len(pieces) > 3:
            lines.append(' '.join([pieces[0], *reversed(pieces[1:])]))
        else:
            lines.append(line)
    return ('\n'.join(lines) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path, required=True,
                        help='Directory containing the nine right_*.glb files from the pinned upstream revision.')
    parser.add_argument('--output', type=Path, default=ROOT / 'licenses/third_party/inspire/DERIVED_FILES.json')
    args = parser.parse_args()
    sources = json.loads((ROOT / 'licenses/third_party/inspire/DERIVATION_SOURCES.json').read_text())
    original = {}
    for row in sources['files']:
        path = args.upstream_dir / Path(row['upstream_path']).name
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if blob != row['upstream_git_blob'] or sha(path) != row['sha256']:
            raise ValueError('Original GLB does not match pinned upstream: ' + path.name)
        original[row['stem']] = (row, trimesh.load(path, force='scene').to_mesh())

    rows = []
    def record(path, source, check, changes):
        rows.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                         license='CC-BY-NC-SA-4.0', decision='noncommercial-redistribution',
                         source_evidence=source, geometry_verification=check, modifications=changes))

    for stem, (evidence, mesh) in original.items():
        raw = ROOT / f'simulator/teleoperation/inspire/assets/right/meshes__visual__right_{stem}.obj'
        check = geometry_check(mesh, trimesh.load(raw, force='mesh', process=False))
        record(raw, evidence, check, 'Converted upstream GLB scene to OBJ; geometry preserved within documented tolerance.')
        reflected = ROOT / f'simulator/teleoperation/inspire/assets/left_mirrored/meshes__visual__left_mirrored_{stem}.obj'
        if reflected.read_bytes() != mirror_obj_bytes(raw):
            raise ValueError('Mirroring recipe mismatch: ' + reflected.name)
        record(reflected, evidence, geometry_check(mesh, trimesh.load(reflected, force='mesh', process=False), True),
               'Converted upstream right GLB to OBJ; mirrored Y coordinates and normals and reversed face winding for the left hand.')
        for side in ('right', 'left'):
            metadata = ROOT / f'simulator/dexjoco/sim/envs/xmls/benchmark6_assets/inspire/{side}/{stem}.json'
            data = json.loads(metadata.read_text())
            if data['glb_sha256'] != evidence['sha256'] or data['original_obj_sha256'] != sha(raw):
                raise ValueError('Conversion metadata mismatch: ' + str(metadata))
            paths = [ROOT / part['path'].removeprefix('repo://') for part in data['parts']]
            parts = [trimesh.load(path, force='mesh', process=False) for path in paths]
            check = geometry_check(mesh, trimesh.util.concatenate(parts), side == 'left')
            for path in paths:
                record(path, evidence, {**check, 'verified_as_union': [p.relative_to(ROOT).as_posix() for p in paths]},
                       'Split converted GLB into material-specific OBJ parts; preserved colors in model metadata; left parts additionally mirror Y with reversed winding.')

    exact = json.loads((ROOT / 'licenses/third_party/inspire/VERIFIED_FILES.json').read_text())
    for source in exact['files']:
        path = ROOT / source['path']
        target = ROOT / str(path.relative_to(ROOT)).replace('/right/', '/left_mirrored/').replace('__right_', '__left_mirrored_')
        if target.read_bytes() != mirror_obj_bytes(path):
            raise ValueError('Collision mirroring recipe mismatch: ' + target.name)
        record(target, source, dict(exact_mirror_recipe=True, mirror_axis='Y'),
               'Mirrored the upstream right collision OBJ in Y; mirrored normals and reversed face winding.')

    assert len(rows) == 68, len(rows)
    report = dict(repository='https://github.com/dexsuite/dex-urdf', commit=COMMIT,
                  license_text=exact['license_text'], upstream_notice=exact['upstream_notice'],
                  notice_sha256=exact['notice_sha256'], files=rows,
                  verification='Exact original upstream blobs; bidirectional geometry and complete triangle multiset comparison; exact mirror recipe.',
                  verification_tool='tools/verify_inspire_derivatives.py', verification_tool_sha256=sha(Path(__file__)),
                  coverage='Only listed 68 derived meshes; does not assign license to the entire Inspire directory or URDF chain.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, derived_files=len(rows), original_glbs=len(original), max_tolerance_m=TOLERANCE)))


if __name__ == '__main__':
    main()
