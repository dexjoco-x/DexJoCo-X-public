#!/usr/bin/env python3
"""Run the Native audit in the pinned Linux x86_64 numerical environment."""
import os
import platform
from pathlib import Path
import runpy
import sys

# Set before importing NumPy, SciPy, MuJoCo or the environment.
os.environ.update(OPENBLAS_CORETYPE='Haswell', OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))


def main():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise SystemExit('The reference replay environment requires Linux x86_64.')
    cpuinfo = Path('/proc/cpuinfo').read_text()
    flags = next((set(line.split(':', 1)[1].split()) for line in cpuinfo.splitlines()
                  if line.startswith('flags')), set())
    if not {'avx2', 'fma'}.issubset(flags):
        raise SystemExit('The reference OpenBLAS kernel requires AVX2 and FMA.')
    from audit_native_replay import environment_fingerprint
    from reference_environment import reference_mismatches
    fingerprint = environment_fingerprint()
    errors = reference_mismatches(fingerprint)
    if errors:
        raise SystemExit(f'Reference environment mismatch: {errors}. Install the reference locks in docs/REPLAY_REPAIR.md.')
    runpy.run_path(str(ROOT / 'evaluation/audit_native_replay.py'), run_name='__main__')


if __name__ == '__main__':
    main()
