import numpy as np
import tensorflow as tf
from exarl.utils.sum_tree import SumTree
from exarl.base.replay_base import Replay
np.random.seed(0)

# TODO: merge _priortized_replay with this file
class ReplayBuffer(Replay):
    """ Class implements a replay buffer
    """

    def __init__(self, max_size, input_size, n_actions):
        """ Replay buffer constructor

        Args:
            max_size (int): maximum buffer length
            input_size (int): dimension of state space
            n_action (int): dimension of action space
        """
        super(ReplayBuffer, self).__init__(max_size)
        self._memory_counter = 0
        # Added the if statement to allow for multidimensional input
        if type(input_size) == type((1,)):
            self._state_buffer      = np.zeros(((max_size,) + input_size))
            self._next_state_buffer = np.zeros(((max_size,) + input_size))
        else:
            self._state_buffer      = np.zeros((max_size, input_size))
            self._next_state_buffer = np.zeros((max_size, input_size))
        self._action_buffer = np.zeros((max_size, n_actions))
        self._reward_buffer = np.zeros((max_size, 1))
        self._done_buffer = np.zeros((max_size, 1))

    def store(self, state, action, reward, next_state, done):
        """ Store experiences in buffer

        Args:
            state (array): array containing current state
            action (array): array of actions
            reward (double): reward for taking action
            next_state (array): array of next state after taking action
            done (bool): indicates episode completion
        """
        self._state_buffer[self._memory_counter] = state
        self._action_buffer[self._memory_counter] = action[0]
        self._reward_buffer[self._memory_counter] = reward
        self._next_state_buffer[self._memory_counter] = next_state
        self._done_buffer[self._memory_counter] = int(done)
        self._memory_counter = (self._memory_counter + 1) % self._memory_size
        if not self.is_full:
            self._mem_length += 1

    def sample_buffer(self, batch_size):
        """ Sample from buffer

        Args:
            batch_size (int): batch size to sample

        Return:
            state_batch (2D array): batch of states
            action_batch (2D array): batch of actions
            reward_batch (array): batch of rewards
            next_state_batch (2D array): batch of next_states
            done_batch (array): batch of done
        """
        record_range = min(len(self), self._memory_size)
        record_range = max(1, record_range)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, batch_size)
        state_batch = self._state_buffer[batch_indices]
        action_batch = self._action_buffer[batch_indices]
        reward_batch = self._reward_buffer[batch_indices]
        next_state_batch = self._next_state_buffer[batch_indices]
        done_batch = self._done_buffer[batch_indices]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def reset_memory(self):
        """Reset memory
        """
        self._memory_counter = 0
        self._mem_length = 0
        self._state_buffer.fill(0)
        self._action_buffer.fill(0)
        self._reward_buffer.fill(0)
        self._next_state_buffer.fill(0)
        self._done_buffer.fill(0)

class nStepBuffer(Replay):
    """ Class implements a replay buffer
    """

    def __init__(self, max_size, input_size, n_actions, horizon, gamma):
        """ Replay buffer constructor

        Args:
            max_size (int): maximum buffer length
            input_size (int): dimension of state space
            n_action (int): dimension of action space
        """
        super(nStepBuffer, self).__init__(max_size)
        self._memory_counter = 0
        self._state_buffer = np.zeros((max_size, input_size))
        self._action_buffer = np.zeros((max_size, n_actions))
        self._reward_buffer = np.zeros((max_size, 1))
        self._next_state_buffer = np.zeros((max_size, input_size))
        self._done_buffer = np.zeros((max_size, 1))
        self._horizon = horizon
        self._gamma   = gamma

    def store(self, state, action, reward, next_state, done):
        """ Store experiences in buffer

        Args:
            state (array): array containing current state
            action (array): array of actions
            reward (double): reward for taking action
            next_state (array): array of next state after taking action
            done (bool): indicates episode completion
        """
        self._state_buffer[self._memory_counter] = state
        self._action_buffer[self._memory_counter] = action[0]
        self._reward_buffer[self._memory_counter] = reward
        self._next_state_buffer[self._memory_counter] = next_state
        self._done_buffer[self._memory_counter] = int(done)
        self._memory_counter = (self._memory_counter + 1) % self._memory_size
        if not self.is_full:
            self._mem_length += 1

    def sample_buffer(self, batch_size):
        """ Sample from buffer

        Args:
            batch_size (int): batch size to sample

        Return:
            state_batch (2D array): batch of states
            action_batch (2D array): batch of actions
            reward_batch (array): batch of rewards
            next_state_batch (2D array): batch of next_states
            done_batch (array): batch of done
        """
        # record_range = min(len(self), self._memory_size)
        # record_range = max(1, record_range)
        record_range = self._memory_counter

        # Randomly sample indices
        batch_indices    = np.random.choice(record_range, batch_size)
        state_batch      = self._state_buffer[batch_indices]
        action_batch     = self._action_buffer[batch_indices]
        # done_batch       = self._done_buffer[batch_indices]
        # reward_batch     = self._reward_buffer[batch_indices]
        # next_state_batch = self._next_state_buffer[batch_indices]

        reward_batch   = []
        next_state_ind = []
        done_batch     = []
        for b_start in batch_indices:
            b_end     = np.min([record_range-1, b_start + self._horizon - 1])
            done_ind  = np.where(self._done_buffer[b_start:(b_end+1)])[0]
            b_end     = b_end if len(done_ind) == 0 else b_start + done_ind[0]
            reward_batch.append( np.sum(self._reward_buffer[b_start:(b_end+1),0] * self._gamma**np.arange(b_end - b_start + 1)) )
            # reward_batch.append( tf.cast( tf.convert_to_tensor(self.reward_buffer[b_start:b_end]), dtype=tf.float32) )
            next_state_ind.append( b_end ) #+ 1 if b_end != record_range else record_range)
            done_batch.append( 0 if len(done_ind) == 0 else 1)
        reward_batch     = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self._next_state_buffer[next_state_ind], dtype=tf.float32)
        done_batch       = tf.convert_to_tensor(done_batch, dtype=tf.float32)


        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def reset_memory(self):
        """Reset memory
        """
        self._memory_counter = 0
        self._mem_length = 0
        self._state_buffer.fill(0)
        self._action_buffer.fill(0)
        self._reward_buffer.fill(0)
        self._next_state_buffer.fill(0)
        self._done_buffer.fill(0)


# class HindsightExperienceReplayMemory(Replay):
#     """This class implements hindsight experience replay
#     (https://arxiv.org/pdf/1707.01495.pdf). The idea is
#     when an agent fails to perform some task, we recognize
#     what was done could be useful in another context or for
#     another task. This is done through caching the trajectories
#     of failed attempts.
#     """

#     def __init__(self, max_size, input_size, n_actions):
#         super(HindsightExperienceReplayMemory, self).__init__(max_size)
#         self._memory_counter = 0
#         self._state_buffer = np.zeros((max_size, input_size))
#         self._action_buffer = np.zeros((max_size, n_actions))
#         self._reward_buffer = np.zeros((max_size, 1))
#         self._next_state_buffer = np.zeros((max_size, input_size))
#         self._done_buffer = np.zeros((max_size, 1))
#         self._goal_buffer = np.zeros((max_size, input_size))

#     def store(self, state, action, reward, next_state, done, goal):
#         # If the counter exceeds the capacity then
#         self._state_buffer[self._memory_counter] = state
#         self._action_buffer[self._memory_counter] = action[0]
#         self._reward_buffer[self._memory_counter] = reward
#         self._next_state_buffer[self._memory_counter] = next_state
#         self._done_buffer[self._memory_counter] = int(done)
#         self._goal_buffer[self._memory_counter] = goal
#         self._memory_counter = (self._memory_counter + 1) % self._memory_size
#         if not self.is_full:
#             self._mem_length += 1

#     def sample_buffer(self, batch_size):

#         record_range = min(len(self), self._memory_size)

#         # Randomly sample indices
#         batch_indices = np.random.choice(record_range, batch_size)
#         state_batch = self._state_buffer[batch_indices]
#         action_batch = self._action_buffer[batch_indices]
#         reward_batch = self._reward_buffer[batch_indices]
#         next_state_batch = self._next_state_buffer[batch_indices]
#         done_batch = self._done_buffer[batch_indices]
#         goal_buffer = self._goal_buffer[batch_indices]

#         return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_buffer

#     def reset_memory(self):
#         self._memory_counter = 0
#         self._mem_length = 0
#         self._state_buffer.fill(0)
#         self._action_buffer.fill(0)
#         self._reward_buffer.fill(0)
#         self._next_state_buffer.fill(0)
#         self._done_buffer.fill(0)
#         self._goal_buffer.fill(0)


# class PrioritedReplayBuffer(Replay):

#     __MIN_EPSILON = 0.01
#     # __ALPHA = 0.6
#     __PER_b_inc_sampling = 0.001
#     __ABSOLUTE_ERROR_UPPER = 1.0

#     def __init__(self, max_size, input_size, n_actions, batch_size, beta=0.4, alpha=0.6):
#         super(PrioritedReplayBuffer, self).__init__(max_size)
#         self.tree = SumTree(max_size)
#         self.input_size = input_size
#         self.n_actions = n_actions
#         self.init_placeholders_data(batch_size)
#         self.beta = beta
#         self.alpha = alpha

#     def init_placeholders_data(self, batch_size):
#         self.state_buffer = np.empty((batch_size, self.input_size))  # place holder to return values
#         self.action_buffer = np.zeros((batch_size, self.n_actions))
#         self.reward_buffer = np.zeros((batch_size, 1))
#         self.next_state_buffer = np.zeros((batch_size, self.input_size))
#         self.done_buffer = np.zeros((batch_size, 1))
#         self.b_idx = np.zeros((batch_size,), dtype=np.int32)
#         self.weights = np.zeros(batch_size,)

#     def store(self, state, action, reward, next_state, done):

#         experience = state, action, reward, next_state, done
#         max_priority = np.max(self.tree.tree[-self.tree.capacity:])
#         if max_priority == 0:
#             max_priority = PrioritedReplayBuffer.__ABSOLUTE_ERROR_UPPER

#         self.tree.add(max_priority, experience)

#         if not self.is_full:
#             self._mem_length += 1

#     def sample_buffer(self, batch_size):  # Include epsilon
#         # TODO: Not efficient
#         # minibatch = np.empty((batch_size, self.tree.data[0].size))

#         priority_segment = self.tree.total_priority / batch_size
#         self.beta = np.min([1, self.beta + PrioritedReplayBuffer.__PER_b_inc_sampling])  # Annealing the beta, replace with epsilon
#         for i in range(batch_size):
#             a, b = priority_segment * i, priority_segment * (i + 1)
#             value = np.random.uniform(a, b)
#             index, priority, data = self.tree.get_priority_values(value)
#             prob = priority / self.tree.total_priority
#             self.b_idx[i] = index
#             self.state_buffer[i] = data[0]  # place holder to return values
#             self.action_buffer[i] = data[1]
#             self.reward_buffer[i] = data[2]
#             self.next_state_buffer[i] = data[3]
#             self.done_buffer[i] = data[4]
#             self.weights[i] = np.power(prob * self.tree.capacity, -self.beta)  # Double check this
#         self.weights = self.weights / np.max(self.weights)
#         return self.state_buffer, self.action_buffer, self.reward_buffer, self.next_state_buffer, self.done_buffer, self.b_idx, self.weights

#     def batch_update(self, tree_index, abs_errors):
#         # self.alpha = np.min([1, self.alpha + PrioritedReplayBuffer.__PER_b_inc_sampling])
#         abs_errors += PrioritedReplayBuffer.__MIN_EPSILON
#         clipped_errors = np.minimum(abs_errors, PrioritedReplayBuffer.__ABSOLUTE_ERROR_UPPER)
#         ps = np.power(clipped_errors, self.alpha)

#         for t_i, prio in zip(tree_index, ps):
#             self.tree.update(t_i, prio)
