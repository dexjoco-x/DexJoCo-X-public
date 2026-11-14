#!/usr/bin/env python3
"""Replay the bundled native-command diagnostics in the installed environments."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

# Match the documented CPU reference setup before importing numerical libraries.
for key, value in {'OPENBLAS_CORETYPE': 'Haswell', 'OPENBLAS_NUM_THREADS': '1',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
    os.environ.setdefault(key, value)
import mujoco
import numpy as np
from dexjoco.benchmark6.environments import make_env

HERE = Path(__file__).resolve().parent


class HammerFaceAudit:
    """Observe each L30 impact without changing the environment's success rule."""
    def __init__(self, env):
        self.impacts = []
        original = env._apply_hammer_nail_interaction

        def observe():
            m, d = env._model, env._data
            face = m.geom('face').id
            rotation = d.geom_xmat[face].reshape(3, 3)
            angle = float(np.degrees(np.arccos(np.clip(-rotation[2, 2], -1, 1))))
            contacts = []
            for contact in d.contact:
                names = {m.geom(contact.geom1).name, m.geom(contact.geom2).name}
                if names.intersection({'nail_head', 'nail_shaft'}):
                    local = rotation.T @ (contact.pos - d.geom_xpos[face])
                    contacts.append((names, local))
            before = float(env._nail_depth)
            value = original()
            if env._nail_depth > before:
                end = any(names == {'face', 'nail_head'} and xyz[2] > .008
                          and np.linalg.norm(xyz[:2]) < .0325 for names, xyz in contacts)
                other = any(names.intersection({'head', 'neck', 'claw'}) for names, _ in contacts)
                self.impacts.append(dict(time=float(d.time), face_angle_deg=angle,
                                         face_end_hit=bool(end and not other and angle < 15)))
            return value
        env._apply_hammer_nail_interaction = observe


def replay(case, video=None):
    path = HERE / case['commands']
    if hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
        raise ValueError('Command checksum mismatch: ' + str(path))
    with np.load(path, allow_pickle=False) as archive:
        commands = {side: archive[side] for side in archive.files}
    if any(len(a) != case['steps'] or not np.isfinite(a).all() for a in commands.values()):
        raise ValueError('Invalid command sequence')
    env = make_env(case['task'], case['hand'], seed=case['seed'], pixels=False)
    renderer = writer = None
    audit = HammerFaceAudit(env) if (case['hand'], case['task']) == ('linkerhand_l30', 'hammer_nail') else None
    success = False
    warnings = set()
    try:
        if video:
            import imageio.v2 as imageio
            video = Path(video)
            video.parent.mkdir(parents=True, exist_ok=True)
            env._model.vis.global_.offwidth = max(768, env._model.vis.global_.offwidth)
            env._model.vis.global_.offheight = max(576, env._model.vis.global_.offheight)
            renderer = mujoco.Renderer(env._model, height=576, width=768)
            camera = 'front' if 'front' in [env._model.camera(i).name for i in range(env._model.ncam)] else 'back'
            writer = imageio.get_writer(str(video), fps=20, codec='libx264', macro_block_size=None,
                                       ffmpeg_params=['-crf', '20', '-movflags', '+faststart'])
        for step in range(case['steps']):
            action = {side: values[step] for side, values in commands.items()}
            payload = action['right'] if len(action) == 1 else action
            _, _, terminated, truncated, info = env.step(payload)
            success = success or bool(info.get('succeed', False))
            warnings.update(np.flatnonzero(env._data.warning.number).tolist())
            if writer:
                renderer.update_scene(env._data, camera=camera)
                writer.append_data(renderer.render().copy())
            if success or terminated or truncated or warnings:
                break
        result = dict(hand=case['hand'], task=case['task'], seed=case['seed'], success=success,
                      steps=step + 1, warnings=sorted(warnings), diagnostic_only=True)
        if audit:
            result['hammer_impacts'] = audit.impacts
            result['face_end_verified'] = bool(audit.impacts and all(i['face_end_hit'] for i in audit.impacts))
        result['passed'] = success and not warnings and result.get('face_end_verified', True)
        return result
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand', default='all', choices=['all', 'linkerhand_l6', 'linkerhand_o6', 'linkerhand_l30'])
    parser.add_argument('--task', default='all')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--video', type=Path, help='MP4 output for a single selected case')
    parser.add_argument('--output', type=Path, default=Path('outputs/linkerhand_replay.json'))
    args = parser.parse_args()
    cases = [c for c in json.loads((HERE / 'manifest.json').read_text())['cases']
             if args.hand in ('all', c['hand']) and args.task in ('all', c['task'])]
    if not cases or args.workers < 1 or (args.video and len(cases) != 1):
        parser.error('Select valid tasks, positive workers, and one case when using --video.')
    if args.workers == 1:
        results = []
        for case in cases:
            result = replay(case, args.video)
            results.append(result)
            print(json.dumps(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(replay, cases))
        for result in results:print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(diagnostic_only=True, cases=results), indent=2) + '\n')
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
