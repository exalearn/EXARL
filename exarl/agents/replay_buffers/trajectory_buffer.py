import numpy as np
from exarl.base.replay_base import Replay_Base
# np.random.seed(0)

class TrajectoryBuffer(Replay_Base):
    """ 
    TODO: WRITE DESCRIPTION HERE...
    """

    def __init__(self, capacity, buffer_trajectory_length=8, observation_space=None, action_space=None, name="Trajectory"):
        """ Replay buffer constructor

        Parameters
        ----------
        capacity : int
            Maximum buffer length
        trajectory_length : int
            Length of max trajectory
        observation_space : gym space (optional)
            Sample of observation space used to init buffer
        action_space : gym space (optional)
            Sample of action space used to init buffer
        """
        super(TrajectoryBuffer, self).__init__(capacity, name=name)
        self._trajectory_length = buffer_trajectory_length
        if observation_space is not None and action_space is not None:
            self._preallocate((observation_space.sample(), 
                               action_space.sample(), 
                               0.0, 
                               observation_space.sample(), 
                               False),
                               capacity = self._capacity + 1)

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
            self._preallocate(data, capacity = self._capacity + 1)
        
        index = self._count % self._capacity
        for slot, item in zip(self._data, data):
            slot[index] = item
        self._count += 1

    def get_padded_indicies(self, end):
        indicies = []
        if self._count == 0:
            indicies = [self._capacity] * self._trajectory_length
        else:
            start = 0 if end < self._trajectory_length else end - self._trajectory_length
            flag = False
            for i in reversed(range(start,end)):
                index = i % self._capacity
                if self._data[-1][index] or flag:
                    indicies.append(self._capacity)
                    flag = True
                else:
                    indicies.append(index)
            indicies.extend([self._capacity] * (self._trajectory_length - len(indicies)))
            indicies.reverse()
        return indicies

    def sample(self, batch_size):
        """
        Returns a transposed batch size elements
        We prepad when based on following paper:
        https://arxiv.org/pdf/1903.07288.pdf

        Parameters
        ----------
        batch_size : int
            batch size to sample

        Returns
        -------
        list :
            List of np arrays for state, action, reward, next_state, done
        """
        assert self.size > 0, str(self) + " -- empty!"
        maxIndex = len(self)
        indices = np.random.choice(maxIndex, batch_size)
        indices = self._count - indices
        # assert (indices <= self._count).sum() == indices.size, str(indices) + " Size: " + str(self._count)
        ret = []
        for i in indices:
            ret.extend(self.get_padded_indicies(i))
        return self.get_data_from_indices(ret)
    
    def last(self):
        indicies = self.get_padded_indicies(self._count)
        return self.get_data_from_indices(indicies)

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
        batch_indices = [self._capacity] * (self._trajectory_length * batch_size)
        # batch_indices = [self._capacity] * batch_size
        return self.get_data_from_indices(batch_indices)