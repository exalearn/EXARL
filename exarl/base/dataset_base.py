# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
from numpy.core.arrayprint import _none_or_positive_arg
import tensorflow as tf
import numpy as np
import exarl.mpi_settings as mpi_settings
import time

class BufferDataset(tf.data.Dataset):
    def _generator(data_buffer, data_win):
        # actor_idx = np.random.randint(low=1, high=mpi_settings.agent_comm.size, size=1)
        # # Get data buffer from RMA window
        # data_win.Lock(actor_idx)
        # data_win.Get(data_buffer, target_rank=actor_idx)
        # data_win.Unlock(actor_idx)
        data_buffer += 1
        yield (data_buffer,)

    def __new__(cls, data_buffer, data_win):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.float32,
            output_shapes=(None, None),
            args=(data_buffer, data_win,)
        )

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)

def bellman_equation(raw_data):
    return raw_data + 1


data_buffer = np.arange(10)
data_win = 0
benchmark(BufferDataset(data_buffer, data_win).map(bellman_equation))
benchmark(BufferDataset(data_buffer, data_win).prefetch(-1).cache().batch(256).map(bellman_equation))
