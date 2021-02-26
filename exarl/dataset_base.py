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
