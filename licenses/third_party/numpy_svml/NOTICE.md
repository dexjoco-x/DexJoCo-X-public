The small-argument asin polynomial and fused evaluation order in `simulator/dexjoco/numerics.py` are adapted from Intel SVML, BSD-3-Clause.

Source: https://github.com/numpy/SVML/blob/1b21e453f6b1ba6a6aca392b1d810d9d41576123/linux/avx512/svml_z0_asin_d_la.s

This is the SVML submodule pinned by NumPy v1.26.4. The implementation uses scalar FMA and does not include AVX-512 assembly. Range reduction and the controller adapter are project modifications. The upstream license is retained in LICENSE.
