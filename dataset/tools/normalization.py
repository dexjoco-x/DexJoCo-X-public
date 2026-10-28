"""Versioned normalization shared by training export and action decoding.

Unversioned statistics retain the original clipped behavior for existing
checkpoints. New exports bind ``quantile_affine_v2`` in each statistics object.
V2 preserves values outside the central quantiles and decodes in float64; its
float32 labels still require independent physical replay validation.
"""
import copy

import numpy as np

LEGACY = 'quantile_clip_v1'
AFFINE = 'quantile_affine_v2'
VERSIONS = (LEGACY, AFFINE)


def version(stats):
    value = stats.get('normalization_version', LEGACY)
    if value not in VERSIONS:
        raise ValueError(f'Unsupported normalization version: {value}')
    return value


def bind(stats, selected=AFFINE):
    if selected not in VERSIONS:
        raise ValueError(f'Unsupported normalization version: {selected}')
    result = copy.deepcopy(stats)
    result['normalization_version'] = selected
    return result


def _inputs(value, mask, stats):
    value = np.asarray(value, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), value.shape)
    lo = np.asarray(stats['q01'], dtype=np.float64)
    hi = np.asarray(stats['q99'], dtype=np.float64)
    if value.ndim < 1 or lo.shape != (value.shape[-1],) or hi.shape != lo.shape:
        raise ValueError('Statistics must match the final feature dimension')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise ValueError('Quantile bounds must be finite and ordered')
    if not np.isfinite(value[mask]).all():
        raise ValueError('Active coordinates must be finite')
    return np.where(mask, value, 0), mask, lo, hi


def _affine(x, lo, hi):
    span = hi - lo
    return np.where(span != 0, 2 * (x - lo) / np.where(span != 0, span, 1) - 1, x - lo)


def _inverse(y, lo, hi):
    y = np.asarray(y, dtype=np.float64)
    span = hi - lo
    return np.where(span != 0, (y + 1) * .5 * span + lo, y + lo)


def representable_label_bounds(stats):
    """Float32 labels whose decoded values stay within physical actuator limits."""
    lo, hi = np.asarray(stats['q01'], np.float64), np.asarray(stats['q99'], np.float64)
    bounds = stats['physical_bounds']
    lower = np.asarray(bounds['minimum'], np.float64)
    upper = np.asarray(bounds['maximum'], np.float64)
    if lower.shape != lo.shape or upper.shape != hi.shape or not np.isfinite([lower, upper]).all() or np.any(lower > upper):
        raise ValueError('Invalid physical action limits')
    low_label = _affine(lower, lo, hi).astype(np.float32)
    high_label = _affine(upper, lo, hi).astype(np.float32)
    for _ in range(4):
        low_label = np.where(_inverse(low_label, lo, hi) < lower, np.nextafter(low_label, np.float32(np.inf)), low_label)
        high_label = np.where(_inverse(high_label, lo, hi) > upper, np.nextafter(high_label, np.float32(-np.inf)), high_label)
    if (not np.isfinite([low_label, high_label]).all() or np.any(low_label > high_label)
            or np.any(_inverse(low_label, lo, hi) < lower) or np.any(_inverse(high_label, lo, hi) > upper)):
        raise ValueError('Physical action interval has no supported float32 label')
    return low_label, high_label


def normalize(value, mask, stats):
    if version(stats) == LEGACY:
        # Preserve the exact old arithmetic, including its constant-channel rule.
        lo, hi = np.array(stats['q01']), np.array(stats['q99'])
        span = hi - lo
        out = 2 * (value - lo) / np.where(span != 0, span, 1) - 1
        out = np.where(span != 0, out, value)
        return np.where(mask, np.clip(out, -1, 1), 0).astype(np.float32)
    x, mask, lo, hi = _inputs(value, mask, stats)
    # Equal quantiles do not imply a constant channel: rare actions must survive.
    y = _affine(x, lo, hi)
    result = np.where(mask, y, 0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Normalized labels cannot be represented as finite float32')
    if 'physical_bounds' in stats:
        lower, upper = (np.asarray(stats['physical_bounds'][k], np.float64) for k in ('minimum', 'maximum'))
        if np.any(mask & ((x < lower) | (x > upper))):
            raise ValueError('Source actions exceed physical actuator limits')
        low_label, high_label = representable_label_bounds(stats)
        result = np.where(mask, np.clip(result, low_label, high_label), 0)
    return result


def denormalize(value, mask, stats):
    if version(stats) == LEGACY:
        lo, hi = np.array(stats['q01']), np.array(stats['q99'])
        return np.where(mask, (np.clip(value, -1, 1) + 1) * .5 * (hi - lo) + lo, 0)
    y, mask, lo, hi = _inputs(value, mask, stats)
    result = np.where(mask, _inverse(y, lo, hi), 0)
    if not np.isfinite(result).all():
        raise ValueError('Decoded actions must be finite')
    return result
