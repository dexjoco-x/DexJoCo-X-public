"""Regression coverage for CPU-dependent rotation-control roundoff."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulator'))
from dexjoco.numerics import portable_asin, quaternion_axisangle, math_fingerprint


class ControllerNumerics(unittest.TestCase):
    def test_first_divergent_rotation_matches_original_reference(self):
        # First divergent input at physics step 50 in Allegro microwave
        # scene_003. System asin and SVML differ by one bit on this input.
        x = float.fromhex('0x1.5d2886177baa0p-7')
        self.assertEqual(portable_asin(x).hex(), '0x1.5d2a371f310bep-7')
        self.assertEqual(math_fingerprint()['asin_probe_hex'], '0x1.5d2a371f310bep-7')

    def test_full_rotation_domain_and_quaternion_sign(self):
        axis = np.array([1., 2., 3.]) / np.linalg.norm([1., 2., 3.])
        for angle in np.r_[0., 1e-12, 1e-7, np.linspace(.01, math.pi, 100)]:
            q = np.r_[math.cos(angle/2), axis*math.sin(angle/2)]
            actual = quaternion_axisangle(q)
            expected = np.zeros(3) if angle < 1e-10 else axis*angle
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=1e-15)
            np.testing.assert_array_equal(actual, quaternion_axisangle(-q))

    def test_asin_domain_boundaries(self):
        for x in [-1., -.999, -.5, -0., 0., .5, .999, 1.]:
            self.assertAlmostEqual(portable_asin(x), math.asin(x), delta=5e-16)
        self.assertEqual(portable_asin(-0.).hex(), '-0x0.0p+0')
        self.assertTrue(math.isnan(portable_asin(1.01)))
        self.assertTrue(math.isnan(portable_asin(math.nan)))

    def test_identical_results_with_numpy_avx512_disabled(self):
        code = '''
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from dexjoco.numerics import portable_asin, quaternion_axisangle
rng = np.random.default_rng(710)
x = np.r_[rng.uniform(-1, 1, 300), float.fromhex('0x1.5d2886177baa0p-7')]
q = rng.normal(size=(80, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
print(json.dumps({'asin': [portable_asin(v).hex() for v in x],
 'rotation': [[float(v).hex() for v in quaternion_axisangle(a)] for a in q]}))
'''
        env = dict(os.environ, OPENBLAS_CORETYPE='Haswell', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('NPY_DISABLE_CPU_FEATURES', None)
        normal = subprocess.check_output([sys.executable, '-c', code, str(ROOT/'simulator')], env=env)
        env['NPY_DISABLE_CPU_FEATURES'] = 'AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'
        restricted = subprocess.check_output([sys.executable, '-c', code, str(ROOT/'simulator')], env=env)
        self.assertEqual(json.loads(normal), json.loads(restricted))


if __name__ == '__main__':
    unittest.main()
