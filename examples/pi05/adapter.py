"""Map Native arm40 observations and noise to a 32-D pi0.5 action head."""
import numpy as np


def checked_vector(value, size, name, dtype=np.float32):
    value = np.asarray(value, dtype=dtype)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite {size}-vector')
    return value


class Native32:
    def __init__(self, mode, horizon):
        if mode not in ('single_right32', 'ego_interleave32') or horizon < 1:
            raise ValueError('Invalid token layout or horizon')
        self.mode, self.horizon = mode, horizon
        self.state_dim = 32 if mode == 'single_right32' else 80
        self.token_horizon = horizon if mode == 'single_right32' else 2 * horizon
        self.indices = np.r_[0:32] if mode == 'single_right32' else np.r_[0:32, 40:72]

    def masks(self, observation):
        state = checked_vector(observation['state_mask'], 80, 'state_mask', bool)
        action = checked_vector(observation['action_mask'], 80, 'action_mask', bool)
        covered = np.zeros(80, bool)
        covered[self.indices] = True
        if (state[~covered] | action[~covered]).any():
            raise ValueError('Active coordinates exceed this token layout; do not truncate the other arm')
        return state, action

    def state(self, observation):
        mask, _ = self.masks(observation)
        state = checked_vector(observation['state'], 80, 'state')
        return np.where(mask, state, 0)[:self.state_dim].copy(), mask[:self.state_dim]

    def noise(self, value):
        value = np.asarray(value, np.float32)
        if value.shape != (self.horizon, 80) or not np.isfinite(value).all():
            raise ValueError('Wrong physical flow noise shape')
        if self.mode == 'single_right32':
            return value[:, :32].copy()
        tokens = np.empty((self.token_horizon, 32), np.float32)
        tokens[0::2], tokens[1::2] = value[:, 40:72], value[:, :32]
        return tokens

    def actions(self, tokens, action_mask):
        tokens = np.asarray(tokens, np.float32)
        if tokens.shape != (self.token_horizon, 32) or not np.isfinite(tokens).all():
            raise ValueError('Wrong model action token shape')
        physical = np.zeros((self.horizon, 80), np.float32)
        if self.mode == 'single_right32':
            physical[:, :32] = tokens
        else:
            physical[:, 40:72], physical[:, :32] = tokens[0::2], tokens[1::2]
        return np.where(action_mask, physical, 0)


def validate_statistics(statistics, state_dim):
    for key, size in (('state', state_dim), ('action', 80)):
        lo = checked_vector(statistics[key]['q01'], size, key + '.q01')
        hi = checked_vector(statistics[key]['q99'], size, key + '.q99')
        if (hi < lo).any():
            raise ValueError(f'{key}: q99 is smaller than q01')


def quantile_transform(value, mask, stats, inverse=False):
    """Ego-pi quantiles, including the midpoint rule for constant coordinates."""
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    span, midpoint = hi - lo, (hi + lo) / 2
    if inverse:
        output = np.where(span < 1e-4, value + midpoint,
                          (value + 1) / 2 * (span + 1e-6) + lo)
    else:
        output = np.where(span < 1e-4, value - midpoint,
                          (value - lo) / (span + 1e-6) * 2 - 1)
    return np.where(mask, output, 0).astype(np.float32)


def images(observation):
    mask = checked_vector(observation['image_mask'], 3, 'image_mask', bool)
    output, masks = {}, {}
    for index, (source, destination) in enumerate((
        ('base', 'base_0_rgb'), ('wrist_left', 'left_wrist_0_rgb'),
        ('wrist_right', 'right_wrist_0_rgb'),
    )):
        image = np.asarray(observation[source])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Images must be RGB uint8 HWC arrays')
        output[destination] = image.copy() if mask[index] else np.zeros_like(image)
        masks[destination] = bool(mask[index])
    return output, masks
