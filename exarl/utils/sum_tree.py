import numpy as np

class SumTree(object):

    def __init__(self, capacity):
        super(SumTree, self).__init__()
        self.capacity = capacity
        self.data_pointer = 0
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    @property
    def total_priority(self):
        return self.tree[0]
    
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        
        self.tree[tree_index] = priority
        
        self._update_tree_difference(tree_index,change)

    def get_priority_values(self, value):
        start_parent_index = 0
        index = self._get_value(start_parent_index, value)
        
        data_index = index - self.capacity + 1
        
        return index, self.tree[index], self.data[data_index]

    def _update_tree_difference(self, tree_index, change):

        while(tree_index != 0):
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def _get_value(self, parent_index, value):

        while True:
            left_index = 2*parent_index + 1
            right_index = left_index + 1

            if left_index >= len(self.tree):
                leaf_index = parent_index
                break
            elif value <= self.tree[left_index]:
                parent_index = left_index
            else:
                value -= self.tree[left_index]
                parent_index = right_index
        
        return leaf_index
