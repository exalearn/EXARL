#%%
import dotsandboxes as dab
from gym import spaces
#%%
# action_space = spaces.MultiDiscrete([12, 13])
# print(action_space.n)
# print(action_space.sample())
# #%%
# action_space = spaces.MultiBinary([12, 13])
# print(action_space.n)
# print(action_space.sample())
#%%

dab.print()
dab.reset()
print(dab.state())
dab.print()
print(dab.step(3))
dab.print()
print(dab.state(), len(dab.state()))
dab.reset()
dab.reset()
dab.reset()


# %%
import numpy as np
class Space(object):
    def __init__(self, shape=None, dtype=None):
        print("SPACE!!!")
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None

    
class MultiBinary(Space):
    def __init__(self, n):
        self.n = n
        print("Set n", self.n)
        if type(n) in [tuple, list, np.ndarray]:
            input_n = n
        else:
            input_n = (n, )
        super(MultiBinary, self).__init__(input_n, np.int8)

temp = MultiBinary(32)
print(temp.n)
# %%
