"""Reproducible scalar rotation math for the benchmark controller.

The small-argument asin polynomial and evaluation order are adapted from
Intel SVML, as vendored by NumPy 1.26.4. Copyright (C) 2023 Intel Corporation.
SPDX-License-Identifier: BSD-3-Clause. See licenses/third_party/numpy_svml/.
"""
import ctypes
import math

import numpy as np

# Explicit, correctly rounded fused multiply-add avoids NumPy's CPU-dependent
# choice between SVML and the system asin. No AVX-512 or compiler is required.
_libm = ctypes.CDLL('libm.so.6')
_fma = _libm.fma
_fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
_fma.restype = ctypes.c_double
_C = tuple(float.fromhex(v) for v in (
    '0x1.07520c70eb909p-5', '-0x1.0fb17f7dbb0edp-6',
    '0x1.43f44bfbc3baep-6', '0x1.a583395d45ed5p-8',
    '0x1.8f8dc2afccad6p-7', '0x1.c6dbbcb88bd57p-7',
    '0x1.1c6dcf538ad2ep-6', '0x1.6e89cebdefaddp-6',
    '0x1.f1c72e13ad8bep-6', '0x1.6db6db3b445f8p-5',
    '0x1.333333337e0dep-4', '0x1.555555555529cp-3'))
_HALF_PI = float.fromhex('0x1.921fb54442d18p+0')
_HALF_PI_LO = float.fromhex('0x1.1a62633145c07p-54')
_ACOS_THRESHOLD = 1 / math.sqrt(2)
MATH_REVISION = 'portable_svml_asin_v1'


def _asin_small(x):
    """Evaluate the reference polynomial for 0 <= x <= 0.5."""
    t = x*x
    t2 = t*t
    t4 = t2*t2
    p = _fma(t, _C[0], _C[1])
    q = _fma(t, _C[2], _C[3])
    p = _fma(t2, p, q)
    q = _fma(t, _C[4], _C[5])
    s = _fma(t, _C[6], _C[7])
    q = _fma(t2, q, s)
    p = _fma(t4, p, q)
    q = _fma(t, _C[8], _C[9])
    p = _fma(t2, p, q)
    q = _fma(t, _C[10], _C[11])
    p = _fma(t2, p, q)
    return _fma(t*p, x, x)


def portable_asin(value):
    """Scalar asin with the same execution order on all reference CPUs."""
    value = float(value)
    x = abs(value)
    if not x <= 1:
        return math.nan
    if x <= 0.5:
        result = _asin_small(x)
    else:
        reduced = math.sqrt((1-x)*0.5)
        result = _HALF_PI - (2*_asin_small(reduced)-_HALF_PI_LO)
    return math.copysign(result, value)


def _positive_acos(x):
    if not 0 <= x <= 1:
        return math.nan
    if x <= 0.5:
        return _HALF_PI - (_asin_small(x)-_HALF_PI_LO)
    return 2*_asin_small(math.sqrt((1-x)*0.5))


def quaternion_axisangle(quat):
    """Preserve the controller's shortest-angle and small-angle conventions."""
    quat = np.asarray(quat)
    if quat.shape != (4,):
        raise ValueError('Expected one quaternion in wxyz order')
    quat = -quat if quat[0] < 0 else quat
    if quat[0] < _ACOS_THRESHOLD:
        angle = 2*_positive_acos(float(quat[0]))
    else:
        angle = 2*portable_asin(float(np.linalg.norm(quat[1:], axis=-1)))
    if angle < 1e-10:
        return np.zeros(3)
    axis = quat[1:4] / np.linalg.norm(quat[1:4], axis=-1, keepdims=True)
    return axis*angle


def math_fingerprint():
    return {'id': MATH_REVISION,
            'asin_probe_hex': portable_asin(float.fromhex('0x1.5d2886177baa0p-7')).hex()}
