from mult_dim_state import State
import random
import numpy as np

# This script shows how to use the State class

# set the each dimension (i.e. 3 x 4 x 5)
each_dim = [3, 4, 5]

# each_dim = [3, 4, 5, 6, 7]

# call the constructor
state = State(each_dim)

# A user can populate the 1D state array
for i in range(state.num_elem_state):
    state.arr_state[i] = i  # random.random()

# print all elements of the state
print("State: ", state.arr_state)

# Example: print the specified elemens of the state
print("Specified state element: ", state.arr_state[state.idx([1, 1, 4])])
# print("Specified state element: ",
#        state.arr_state[state.idx([2, 2, 4, 5, 6])])
