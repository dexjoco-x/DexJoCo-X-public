"""Shared numerical-environment checks for replay and policy evaluation."""
import json
from pathlib import Path


def reference_mismatches(fingerprint):
    expected = json.loads((Path(__file__).resolve().parents[1] /
                           'requirements/native-replay-fingerprint.json').read_text())
    errors = [key for key in ('packages', 'mujoco_library_sha256', 'controller_math',
                             'numpy_extension_sha256', 'scipy_rotation_extension_sha256')
              if fingerprint.get(key) != expected[key]]
    actual_blas = fingerprint.get('blas') or []
    fields = ('internal_api', 'version', 'threading_layer', 'architecture', 'num_threads')
    actual = {tuple(row.get(key) for key in fields) for row in actual_blas}
    if any(tuple(row.get(key) for key in fields) not in actual for row in expected['blas']):
        errors.append('reference BLAS kernels/thread counts')
    return errors
