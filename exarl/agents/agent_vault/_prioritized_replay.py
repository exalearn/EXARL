import random
import numpy as np
import tensorflow as tf
from collections import deque


class PrioritizedReplayBuffer():
    """Class implements Prioritized Experince Replay (PER)
    """

    def __init__(self, maxlen):
        """PER constructor

        Args:
            maxlen (int): buffer length
        """
        self.maxlen = None if maxlen == "none" else maxlen
        self.buffer = deque(maxlen=self.maxlen)
        self.priorities = deque(maxlen=self.maxlen)

    def add(self, experience):
        """Add experiences to buffer

        Args:
            experience (list): state, action, reward, next_state, done

        Returns:
            full_buffer (done): True if buffer is full
        """
        full_buffer = len(self.buffer) == self.maxlen
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
        return full_buffer

    def get_probabilities(self, priority_scale):
        """Get probabilities for experiences

        Args:
            priority_scale (float64): range [0, 1]

        Returns:
            sample_probabilities (numpy array): probabilities assigned to experiences based on weighting factor (scale)
        """
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        """Compute importance

        Args:
            probabilities (numpy array): experience probabilities

        Returns:
            importance_normalized (numpy array): normalized importance
        """
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        """Sample experiences

        Args:
            batch_size (int): size of batch
            priority_scale (float, optional): range = [0, 1]. Defaults to 1.0.

        Returns:
            samples (list): sampled based on probabilities
            importance (numpy array): Importance of samples
            sample_indices (array): Indices of samples
        """
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        """Set priorities to experiences

        Args:
            indices (array): sample indicies
            errors (array): corresponding losses
            offset (float, optional): Small offset. Defaults to 0.1.
        """
        for i, e in zip(indices, errors):
            self.priorities[int(i)] = abs(e) + offset

    def get_buffer_length(self):
        """Get buffer length

        Returns:
            (int): buffer length
        """
        return len(self.buffer)
