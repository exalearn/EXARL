import numpy as np
from exarl.base.replay_base import Replay_Base
# np.random.seed(0)

class ReplayBuffer(Replay_Base):
    """ 
    Class implements a simple replay buffer
    """

    def __init__(self, capacity, observation_space=None, action_space=None, name="Replay"):
        """ 
        Replay buffer constructor

        Parameters
        ----------
        capacity : int
            Maximum buffer length
        observation_space : gym space (optional)
            Sample of observation space used to init buffer
        action_space : gym space (optional)
            Sample of action space used to init buffer
        """
        super(ReplayBuffer, self).__init__(capacity, name=name)
        if observation_space is not None and action_space is not None:
            self._preallocate((observation_space.sample(), 
                               action_space.sample(), 
                               [0.0], 
                               observation_space.sample(), 
                               False))

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

        for slot, item in zip(self._data, data):
            # print(self._count, self._capacity, len(item), len(slot), type(item), slot.shape, item)
            if type(item) == type([]):
                if len(item) == 1:
                    slot[self._count % self._capacity] = item[0]
                else:
                    slot[self._count % self._capacity] = item
            else:
                slot[self._count % self._capacity] = item
        self._count += 1

    def sample(self, batch_size):
        """
        Returns a transposed batch size elements

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

        batch_indices = np.random.choice(len(self), batch_size)
        return self.get_data_from_indices(batch_indices)
    
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
        return self.get_data_from_indices(batch_indices)

class SimpleBuffer(ReplayBuffer):
    def __init__(self, capacity, observation_space=None, action_space=None, name="Simple"):
        """ 
        TODO: Write

        Parameters
        ----------
        capacity : int
            Maximum buffer length
        observation_space : gym space (optional)
            Sample of observation space used to init buffer
        action_space : gym space (optional)
            Sample of action space used to init buffer
        """
        super(SimpleBuffer, self).__init__(capacity, observation_space=observation_space, action_space=action_space, name=name)

    def sample(self, batch_size):
        """
        Returns a transposed batch size elements

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
        ret = self.get_data_from_indices(range(self.size))
        self._count = 0
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
        return self.get_data_from_indices(range(batch_size))
