import pickle as pk
import numpy as np

f = open('target_weights', 'rb')
target_weights = np.array(pk.load(f), dtype='f')
f.close()

for i in range(np.size(target_weights)):
    target_weights[i].flatten()