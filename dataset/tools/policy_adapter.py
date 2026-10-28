"""Versioned native200 preprocessing and float64 decoding for arm40_v1 servers."""
import numpy as np
from .normalization import AFFINE, AFFINE_V2, normalize, denormalize, version


class Native200PolicyAdapter:
    """Use with models trained on matching versioned affine labels and statistics.

    The model owns RGB preprocessing. The evaluator owns physical action limits.
    This adapter must not be substituted into an existing legacy checkpoint.
    """

    def __init__(self, groups):
        if (not groups or any(version(s) not in (AFFINE,AFFINE_V2) for g in groups.values() for s in g.values())
                or any('physical_bounds' not in g['action'] for g in groups.values())):
            raise ValueError('This adapter requires versioned native200 affine statistics')
        self.groups = groups

    def _group(self, observation):
        key = observation['task'] + '/' + observation['hand']
        if key not in self.groups: raise ValueError('Hand/task is outside these training statistics')
        return self.groups[key]

    @staticmethod
    def _mask(observation, name):
        mask = np.asarray(observation[name], dtype=bool)
        if mask.shape != (80,): raise ValueError('Expected an arm40_v1 mask of length 80')
        return np.pad(mask, (0, 120))

    def state(self, observation):
        group = self._group(observation)
        state = np.asarray(observation['state'], dtype=np.float32)
        if state.shape != (80,): raise ValueError('Expected an arm40_v1 state of length 80')
        mask = self._mask(observation, 'state_mask')
        return normalize(np.pad(state, (0, 120)), mask, group['state']), mask

    def actions(self, prediction, observation):
        group = self._group(observation)
        labels = np.asarray(prediction, dtype=np.float32)
        if labels.ndim != 2 or labels.shape[1] != 200 or not len(labels):
            raise ValueError('Expected model actions with shape [horizon, 200]')
        mask = self._mask(observation, 'action_mask')
        # denormalize promotes to float64 before arithmetic. Do not cast the
        # decoded values back to float32 before sending them to the evaluator.
        return denormalize(labels, mask, group['action'])[:, :80]
