import random
import numpy as np
from exarl.base.replay_base import Replay_Base


class PrioritizedReplayBuffer(Replay_Base):
    """
    Class implements Prioritized Experience Replay (PER)
    """

    def __init__(self, capacity, observation_space=None, action_space=None, name="Priority"):
        """ Replay buffer constructor

        Parameters
        ----------
        capacity : int
            Maximum buffer length
        observation_space : gym space (optional)
            Sample of observation space used to init buffer
        action_space : gym space (optional)
            Sample of action space used to init buffer
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity, name=name)
        
        self._alpha = 0.5
        self._alpha_decay_rate = 0.99
        self._beta = 0.5
        self._beta_growth_rate = 1.001

        self.incremental_td_error = 0.0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

        self._samples = 1
        self.compute_weights = False

        # JS: we are going to try a two tiered approach
        # self._data will hold all data
        # self._meta will hold (priority, probability, weight, index)
        # where index is the index into self._data
        self._meta = None
        if observation_space is not None and action_space is not None:
            self._preallocate((observation_space.sample(), 
                               action_space.sample(), 
                               0.0, 
                               observation_space.sample(), 
                               False))

    def _preallocate(self, items):
        super()._preallocate(items)
        # JS: Tuple is (priority, probability, weight, index)
        self._meta = [[0,0,0,i] for i in range(self._capacity)]

    def store(self, state, action, reward, next_state, done):
        """
        Stores data in buffer.  Allocates data if uninitialized.

        Parameters
        ----------
        state : gym space sample
            Current state to store
        action : gym space sample
            Current action to store
        reward : float
            Reward based on action
        next_state : gym space sample
            State after action
        done : bool
            If state is terminal
        """
        data = (state, action, reward, next_state, done)
        if self._data is None:
            self._preallocate(data)
        
        index = self._count % self._capacity
        meta = self._meta[index]

        # JS: This is the full case
        if self._count > self._capacity:
            
            # JS: Subtract our running priority sum
            self.priorities_sum_alpha -= meta[0] ** self._alpha

            # JS: Update the max priority if we are overwriting it
            if meta[0] == self.priorities_max:
                meta[0] = 0
                self.priorities_max = max(self._meta, key=lambda x: x[1])[0]

            # JS: Update the max weight if we are overwriting it
            if self.compute_weights:
                if meta[2] == self.weights_max:
                    meta[2] = 0
                    self.weights_max = max(self._meta, key=lambda x: x[2])[2]

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self._alpha
        probability = priority ** self._alpha / self.priorities_sum_alpha

        # JS: Update meta data
        meta[0] = priority
        meta[1] = probability
        meta[2] = weight

        # JS: Actually add data to buffer
        for slot, item in zip(self._data, data):
            slot[meta[3]] = item
        self._count += 1

    def update_priorities(self, tds, indices):
        N = self.size
        tds = np.absolute(tds) + self.incremental_td_error
        for updated_priority, index in zip(tds, indices):
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                # JS: Annealing the bias 
                updated_weight = ((N * updated_priority) ** (-self._beta)) / self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            old_priority = self._meta[index][0]
            # JS: Update our priority sum
            self.priorities_sum_alpha += (updated_priority ** self._alpha) - (old_priority ** self._alpha)
            # JS: Update our probability
            updated_probability = (updated_priority ** self._alpha) / self.priorities_sum_alpha
            
            self._meta[index][0] = updated_priority
            self._meta[index][1] = updated_probability
            self._meta[index][2] = updated_weight

    def update_parameters(self):
        # JS: Update the hypers
        self._alpha *= self._alpha_decay_rate
        self._beta *= self._beta_growth_rate
        if self._beta > 1:
            self._beta = 1

        N = self.size
        self.priorities_sum_alpha = sum(meta[0] ** self._alpha for meta in self._meta)

        for i, meta in zip(range(N), self._meta):
            meta[1] = meta[0] ** self._alpha / self.priorities_sum_alpha
            if self.compute_weights:
                meta[2] = ((N *  meta[1]) ** (-self._beta)) / self.weights_max
            else:
                meta[2] = 1

    def sample_generator(self, batch_size):
        start = 0
        random_values = []
        while True:
            if start + batch_size <= len(random_values):
                indices = random_values[start : start + batch_size]
                start += batch_size
                yield indices
            else:
                start = 0
                self.update_parameters()
                N = self.size
                random_values = random.choices(range(N), 
                                               [x[1] for _, x in zip(range(N), self._meta)], 
                                               k=batch_size * self._samples)

    def sample(self, batch_size):
        assert self.size > 0, str(self) + " -- empty!"
        indices = next(self.sample_generator(batch_size))
        assert len(indices) == batch_size
        ret = self.get_data_from_indices([self._meta[i][3] for i in indices])
        ret.append(indices)
        return ret

    def get_fake_data(self, batch_size):
        """
        Returns a transposed batch size elements.  Data is garbage, but
        useful for sizing RMA window.

        Parameters
        ----------
        batch_size : int
            batch size to sample

        Returns
        -------
        list :
            List of np arrays for state, action, reward, next_state, done
        """
        assert self._data is not None, "Must have preallocated data with observation and action space in constructor!"
        batch_indices = np.random.choice(self._capacity, batch_size)
        ret = self.get_data_from_indices(batch_indices)
        ret.append(batch_indices)
        return ret