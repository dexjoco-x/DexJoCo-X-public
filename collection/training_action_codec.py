"""Execute exactly the commands decoded from final float32 training labels."""
import numpy as np

from dataset.tools.adapters import bind_action_bounds, decode, encode
from dataset.tools.normalization import AFFINE, denormalize, normalize, version


class TrainingActionCodec:
    def __init__(self, statistics, contract, action_mask):
        if version(statistics) != AFFINE:
            raise ValueError('New collection requires quantile_affine_v2 statistics')
        self.native_mask = np.asarray(action_mask, bool)
        if self.native_mask.shape != (2, 28):
            raise ValueError('Expected native action mask[2, 28]')
        bound = bind_action_bounds(statistics, contract)
        if statistics.get('physical_bounds') != bound['physical_bounds']:
            raise ValueError('Statistics must bind this hand/task physical action contract')
        self.statistics = bound
        _, self.mask = encode(np.zeros((2, 28)), self.native_mask, 'action', 'beingh05')
        self.lower = np.asarray(bound['physical_bounds']['minimum'], np.float64)
        self.upper = np.asarray(bound['physical_bounds']['maximum'], np.float64)

    def encode_command(self, requested):
        values = np.asarray(requested, np.float64)
        if values.shape != (2, 28) or not np.isfinite(values).all():
            raise ValueError('Expected finite native action[2, 28]')
        physical, mask = encode(values, self.native_mask, 'action', 'beingh05')
        bounded = np.clip(physical, self.lower, self.upper)
        clipped = int(np.count_nonzero(mask & (bounded != physical)))
        labels = normalize(bounded, mask, self.statistics)
        actual = self.decode_labels(labels)
        return labels, actual, clipped

    def decode_labels(self, labels):
        labels = np.asarray(labels)
        if labels.shape != (200,) or labels.dtype != np.float32 or not np.isfinite(labels).all():
            raise ValueError('Saved labels must be finite float32[200]')
        if np.any(labels[~self.mask] != 0):
            raise ValueError('Inactive training labels must be zero')
        physical = denormalize(labels, self.mask, self.statistics)
        if np.any(self.mask & ((physical < self.lower) | (physical > self.upper))):
            raise ValueError('Decoded labels exceed physical command limits')
        return decode(physical, self.native_mask, 'beingh05')
