import numpy as np
from exarl.utils.sum_tree import SumTree
from exarl.base.replay_base import Replay

class ReplayBuffer(Replay):

    def __init__(self, max_size, input_size, n_actions):
        super(ReplayBuffer, self).__init__(max_size)
        self._memory_counter = 0
        self._state_buffer = np.zeros((max_size, input_size))
        self._action_buffer = np.zeros((max_size, n_actions))
        self._reward_buffer = np.zeros((max_size, 1))
        self._next_state_buffer = np.zeros((max_size, input_size))
        self._done_buffer = np.zeros((max_size, 1))

    def store(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then

        self._state_buffer[self._memory_counter] = state
        self._action_buffer[self._memory_counter] = action
        self._reward_buffer[self._memory_counter] = reward
        self._next_state_buffer[self._memory_counter] = next_state
        self._done_buffer[self._memory_counter] = int(done)
        self._memory_counter = (self._memory_counter + 1) % self._memory_size
        if not self.is_full:
            self._mem_length += 1

    def sample_buffer(self, batch_size):
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
        self._memory_counter = 0
        self._mem_length = 0
        self._state_buffer.fill(0)
        self._action_buffer.fill(0)
        self._reward_buffer.fill(0)
        self._next_state_buffer.fill(0)
        self._done_buffer.fill(0)



class HindsightExperienceReplayMemory(Replay):

    def __init__(self, max_size, input_size, n_actions):
        super(HindsightExperienceReplayMemory, self).__init__(max_size)
        self._memory_counter = 0
        self._state_buffer = np.zeros((max_size, input_size))
        self._action_buffer = np.zeros((max_size, n_actions))
        self._reward_buffer = np.zeros((max_size, 1))
        self._next_state_buffer = np.zeros((max_size, input_size))
        self._done_buffer = np.zeros((max_size, 1))
        self._goal_buffer = np.zeros((max_size, input_size))

    def store(self, state, action, reward, next_state, done, goal):
        # If the counter exceeds the capacity then
        self._state_buffer[self._memory_counter] = state
        self._action_buffer[self._memory_counter] = action
        self._reward_buffer[self._memory_counter] = reward
        self._next_state_buffer[self._memory_counter] = next_state
        self._done_buffer[self._memory_counter] = int(done)
        self._goal_buffer[self._memory_counter] = goal
        self._memory_counter = (self._memory_counter + 1) % self._memory_size
        if not self.is_full:
            self._mem_length += 1

    def sample_buffer(self, batch_size):

        record_range = min(len(self), self._memory_size)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, batch_size)
        state_batch = self._state_buffer[batch_indices]
        action_batch = self._action_buffer[batch_indices]
        reward_batch = self._reward_buffer[batch_indices]
        next_state_batch = self._next_state_buffer[batch_indices]
        done_batch = self._done_buffer[batch_indices]
        goal_buffer = self._goal_buffer[batch_indices]


        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_buffer

    def reset_memory(self):
        self._memory_counter = 0
        self._mem_length = 0
        self._state_buffer.fill(0)
        self._action_buffer.fill(0)
        self._reward_buffer.fill(0)
        self._next_state_buffer.fill(0)
        self._done_buffer.fill(0)
        self._goal_buffer.fill(0)


class PrioritedReplayBuffer(Replay):

    __PER_e = 0.01
    __PER_a = 0.6
    #__PER_b = 0.4
    __PER_b_inc_sampling = 0.001
    __ABSOLUTE_ERROR_UPPER = 1.0

    def __init__(self, max_size, input_size, n_actions, batch_size):
        super(PrioritedReplayBuffer, self).__init__(max_size)
        self.tree = SumTree(max_size)
        self.input_size = input_size
        self.n_actions = n_actions
        self.init_placeholders_data(batch_size)

    def init_placeholders_data(self, batch_size):
        self.state_buffer = np.empty((batch_size, self.input_size)) # place holder to return values
        self.action_buffer = np.zeros((batch_size, self.n_actions))
        self.reward_buffer = np.zeros((batch_size, 1))
        self.next_state_buffer = np.zeros((batch_size, self.input_size))
        self.done_buffer = np.zeros((batch_size, 1))
        self.b_idx = np.zeros((batch_size,), dtype=np.int32)

    def store(self, state, action, reward, next_state, done):

        experience = state, action, reward, next_state, done
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = PrioritedReplayBuffer.__ABSOLUTE_ERROR_UPPER

        self.tree.add(max_priority, experience)

        if not self.is_full:
            self._mem_length += 1

    def sample_buffer(self, batch_size):
        #TODO: Not efficient
        #minibatch = np.empty((batch_size, self.tree.data[0].size))


        priority_segment = self.tree.total_priority/batch_size
        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a,b)
            index, priority, data = self.tree.get_priority_values(value)
            self.b_idx[i] = index
            self.state_buffer[i] = data[0] # place holder to return values
            self.action_buffer[i] = data[1]
            self.reward_buffer[i] = data[2]
            self.next_state_buffer[i] = data[3]
            self.done_buffer[i] = data[4]

        return self.state_buffer, self.action_buffer, self.reward_buffer, self.next_state_buffer, self.done_buffer, self.b_idx

    def batch_update(self, tree_index, abs_errors):
        abs_errors += PrioritedReplayBuffer.__PER_e
        clipped_errors = np.minimum(abs_errors, PrioritedReplayBuffer.__ABSOLUTE_ERROR_UPPER)
        ps = np.power(clipped_errors, PrioritedReplayBuffer.__PER_a)

        for t_i, prio in zip(tree_index, ps):
            self.tree.update(t_i, prio)

