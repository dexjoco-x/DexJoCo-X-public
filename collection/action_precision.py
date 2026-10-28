"""Representable in-range float32 commands for NEW physical collection.

Quantize before stepping and save that same command. This changes an existing
trajectory's actions; it does not certify a converted old demonstration.
"""
import numpy as np


def float32_native_action(action, actuator_ranges, translation_bound, rotation_bound):
    values = np.asarray(action)
    if values.shape != (2, 28) or not np.isfinite(values).all():
        raise ValueError('Expected finite native action with shape (2, 28)')
    output = np.zeros((2, 28), np.float32)
    for arm, side in enumerate(('right', 'left')):
        if side not in actuator_ranges:
            continue
        ranges = np.asarray(actuator_ranges[side], np.float64)
        if ranges.ndim != 2 or ranges.shape[1] != 2 or len(ranges) > 22:
            raise ValueError('Invalid actuator ranges')
        lo = np.r_[[-translation_bound] * 3, [-rotation_bound] * 3, ranges[:, 0]]
        hi = np.r_[[translation_bound] * 3, [rotation_bound] * 3, ranges[:, 1]]
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise ValueError('Invalid action bounds')
        lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
        lo32 = np.where(lo32.astype(np.float64) < lo, np.nextafter(lo32, np.float32(np.inf)), lo32)
        hi32 = np.where(hi32.astype(np.float64) > hi, np.nextafter(hi32, np.float32(-np.inf)), hi32)
        if np.any(lo32 > hi32):
            raise ValueError('Action interval contains no representable float32 value')
        output[arm, :len(lo)] = np.clip(values[arm, :len(lo)].astype(np.float32), lo32, hi32)
    return output
