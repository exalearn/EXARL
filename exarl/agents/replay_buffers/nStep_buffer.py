import numpy as np
from exarl.agents.replay_buffers.replay_buffer import ReplayBuffer
# np.random.seed(0)

class nStepBuffer(ReplayBuffer):
    """ 
    Class implements a simple replay buffer
    """

    def __init__(self, capacity, horizon, gamma, observation_space=None, action_space=None, name="nStep"):
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
        super(nStepBuffer, self).__init__(capacity, observation_space=observation_space, action_space=action_space, name=name)
        self.horizon = horizon
        self.gamma   = gamma

    def get_data_from_indices(self, indices):
        """
        Returns a list of the of data at given indices
        """
        assert self._data is not None, str(self) + " -- not initialized!"
        return_list = []
        # Append states
        return_list.append(self._data[0][indices])

        # Append actions
        return_list.append(self._data[1][indices])

        # Calculate nStep rewards, next states, and done indicators
        reward_batch   = []
        done_batch     = []
        next_state_ind = []
        for b_start in indices:
            b_end     = np.min([self._count-1, b_start + self.horizon - 1])
            done_ind  = np.where(self._data[4][np.arange(b_start, b_end+1) % self._capacity])[0]
            b_end     = b_end if len(done_ind) == 0 else np.arange(b_start, b_end+1)[done_ind[0]]

            reward_batch.append( np.sum(self._data[2][np.arange(b_start,b_end+1) % self._capacity,0] * self.gamma**np.arange(b_end - b_start + 1)) )
            next_state_ind.append( b_end % self._capacity)
            done_batch.append( 0 if len(done_ind) == 0 else 1)

        # Append nStep rewards
        return_list.append(np.array(reward_batch)[:,None])
        
        # Append nStep next state
        return_list.append(self._data[3][next_state_ind])

        # Append nStep done indicators
        return_list.append(np.array(done_batch))

        return return_list

