import sys
import numpy as np


class State(object):

    def __init__(self, dim_arr):

        self.arr_state = []       # 1D array contains all elements of
        # multi-dimensional state

        self.arr_each_dim = dim_arr  # 1D arry containts size of each dimension

        self.dim_state = 0
        self.num_elem_state = 1        # the # of total elements for each state

        self.size_arr = []       # 1D helper array to get an get an index

        self.debug = 0                 # TODO: one can pass an argument

        self.set_dim_state()
        self.set_num_elem_state()
        self.allocate_state_space()
        self.set_size_arr()

    # set the dimension of the state, dim_state
    def set_dim_state(self):

        self.dim_state = len(self.arr_each_dim)

        if (self.debug >= 10):
            print("dim_state: ", self.dim_state)

        # TODO: one can add try .. exception
        if self.dim_state < 1:
            print("ERROR: The dimension of the state should be at least 1!")
            sys.exit()

    # set # of total elements of a state
    def set_num_elem_state(self):

        # TODO: one can add try .. exception for each element of arr_each_dim

        for i in range(self.dim_state):
            self.num_elem_state *= self.arr_each_dim[i]

        if (self.debug >= 10):
            print("num_elem_state: ", self.num_elem_state)

    # allocate spapce space using the total # of elements of a state
    def allocate_state_space(self):
        self.arr_state = np.zeros(self.num_elem_state)

    # set 1D hepler array to get an index, size_arr
    def set_size_arr(self):

        self.size_arr = np.zeros(self.dim_state)

        self.size_arr[self.dim_state - 1] = 0
        if (self.dim_state <= 1):
            return

        self.size_arr[self.dim_state -
                      2] = self.arr_each_dim[self.dim_state - 1]
        if (self.dim_state <= 2):
            return

        for i in reversed(range(self.dim_state - 2)):
            self.size_arr[i] = self.arr_each_dim[i + 1] * self.size_arr[i + 1]

        if (self.debug >= 10):
            print("size_arr: ", self.size_arr)

    # return the index from the [i, j, k, ...]-th element of multi-dimensional
    # state
    def idx(self, arr_idx):

        if (self.debug >= 10):
            print("arr_idx: ", arr_idx)

        # TODO: one can add try .. exception for the input of arr_idx
        if self.dim_state != len(arr_idx):
            print(
                "ERROR: The dimension of the idex array should be the same as"
                " the dimension of the state size!")
            sys.exit()

        if (len(arr_idx) == 1):
            index = self.arr_idx[0]
        else:
            index = 0
            for i in range(self.dim_state):
                if (i < self.dim_state - 1):
                    index += self.size_arr[i] * arr_idx[i]
                elif (i == self.dim_state - 1):
                    index += arr_idx[i]

        if (self.debug >= 10):
            print("index: ", int(index))

        return int(index)
