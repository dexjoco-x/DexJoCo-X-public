"""Versioned normalization shared by training export and action decoding.

Unversioned statistics retain the original clipped behavior for existing
checkpoints. V2 removes quantile clipping. V3 additionally scales labels by exact
powers of two, avoiding huge targets in nearly constant channels. Both decode
with float64 arithmetic and require independent physical replay validation.
"""
import copy

import numpy as np

LEGACY = 'quantile_clip_v1'
AFFINE_V2 = 'quantile_affine_v2'
AFFINE = 'quantile_affine_v3'
VERSIONS = (LEGACY, AFFINE_V2, AFFINE)


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
    if selected == AFFINE:
        result.setdefault('label_scale_exponents', [0] * len(result['q01']))
    else:
        result.pop('label_scale_exponents', None)
    return result


def _powers(stats):
    if version(stats) != AFFINE:
        return np.zeros(len(stats['q01']), dtype=np.int32)
    values = np.asarray(stats.get('label_scale_exponents'))
    if (values.shape != (len(stats['q01']),) or values.dtype.kind not in 'iu'
            or np.any(values < 0) or np.any(values > 126)):
        raise ValueError('V3 requires one integer label-scale exponent in [0,126] per coordinate')
    return values.astype(np.int32)


def stabilize_scales(stats):
    """Bound training-label scale without changing the decoded physical commands.

    Power-of-two factors preserve float32 mantissas. Actions use their full
    physical range; other features use training extrema, never validation data.
    """
    result = bind(stats, AFFINE)
    unscaled = bind(stats, AFFINE_V2)
    if 'physical_bounds' in stats:
        lower, upper = (stats['physical_bounds'][k] for k in ('minimum', 'maximum'))
        source = 'physical_control_limits'
    else:
        lower, upper = stats['min'], stats['max']
        source = 'training_extrema'
    values = np.asarray([lower, upper], dtype=np.float64)
    labels = normalize(values, np.ones_like(values, bool), unscaled)
    magnitude = np.max(np.abs(labels.astype(np.float64)), axis=0)
    powers = np.ceil(np.log2(np.maximum(magnitude, 1.))).astype(np.int32)
    result.update(label_scale_exponents=powers.tolist(), label_scale_source=source)
    _powers(result)
    return result


def scale_labels(labels, stats, require_label_bits=True):
    """Rescale labels; optionally require exact recovery of all original label bits."""
    labels = np.asarray(labels)
    if labels.dtype != np.float32: raise ValueError('Label rescaling requires float32 inputs')
    powers = _powers(stats)
    scaled = np.ldexp(labels, -powers)
    if require_label_bits and not np.array_equal(np.ldexp(scaled.astype(np.float64), powers), labels.astype(np.float64)):
        raise ValueError('Power-of-two rescaling would lose a label bit')
    return scaled


def preserve_decoded_scale(labels, mask, previous, proposed):
    """Reduce scaling only where subnormal labels would change a physical value."""
    result = copy.deepcopy(proposed)
    expected = denormalize(labels, mask, previous)
    while True:
        scaled = scale_labels(labels, result, require_label_bits=False)
        restored = denormalize(scaled, mask, result)
        changed = np.any(restored != expected, axis=tuple(range(restored.ndim - 1)))
        if not np.any(changed): return result
        powers = _powers(result)
        if np.any(powers[changed] == 0): raise ValueError('Unable to preserve decoded commands')
        powers[changed] -= 1
        result['label_scale_exponents'] = powers.tolist()


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


def _inverse(y, lo, hi, powers=0):
    y = np.asarray(y, dtype=np.float64)
    y = np.ldexp(y, powers)
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
    powers = _powers(stats)
    low_label = scale_labels(_affine(lower, lo, hi).astype(np.float32), stats)
    high_label = scale_labels(_affine(upper, lo, hi).astype(np.float32), stats)
    for _ in range(4):
        low_label = np.where(_inverse(low_label, lo, hi, powers) < lower, np.nextafter(low_label, np.float32(np.inf)), low_label)
        high_label = np.where(_inverse(high_label, lo, hi, powers) > upper, np.nextafter(high_label, np.float32(-np.inf)), high_label)
    if (not np.isfinite([low_label, high_label]).all() or np.any(low_label > high_label)
            or np.any(_inverse(low_label, lo, hi, powers) < lower) or np.any(_inverse(high_label, lo, hi, powers) > upper)):
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
    result = scale_labels(result, stats)
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
    result = np.where(mask, _inverse(y, lo, hi, _powers(stats)), 0)
    if not np.isfinite(result).all():
        raise ValueError('Decoded actions must be finite')
    return result
