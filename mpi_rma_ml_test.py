from mpi4py import MPI
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle as pk

comm = MPI.COMM_WORLD
rank = comm.rank
print('rank',rank)
n = np.zeros (10 , dtype = np.int )

##
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
x = tf.ones((3, 3))
y = model(x)
model.build((3,3))
weights = model.get_weights()
print("Rank[%d] Initial data %s" % (rank, weights[0]))
serial = MPI.pickle.dumps(weights)
nserial = len(serial)
win  = MPI.Win.Allocate(nserial, 1, comm=comm)
buff = win.tomemory()
buff[:] = 0
if rank > 0:
    win.Lock(0)
    win.Put(serial, 0, target=0)
    win.Unlock(0)

comm.Barrier()
if rank == 0:
    f_weights = MPI.pickle.loads(buff)
    print("Rank[%d] final data %s" % (rank, f_weights[0]))

win.Free()
