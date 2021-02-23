from numpy.core.arrayprint import _none_or_positive_arg
import tensorflow as tf
import numpy as np
from exarl.comm_base import ExaComm
from mpi4py import MPI
import time

class BufferDataset(tf.data.Dataset):
    # def _generator(data_buffer, data_win):
    #     # actor_idx = np.random.randint(low=1, high=mpi_settings.agent_comm.size, size=1)
    #     # # Get data buffer from RMA window
    #     # data_win.Lock(actor_idx)
    #     # data_win.Get(data_buffer, target_rank=actor_idx)
    #     # data_win.Unlock(actor_idx)
    #     data_buffer += 1
    #     yield (data_buffer,)

    def _generator(agent, data_buffer, data_win):
        # Worker method to create samples for training
        # batch_states = np.zeros((agent.batch_size, 1, agent.env.observation_space.shape[0])).astype("float64")
        # batch_target = np.zeros((agent.batch_size, agent.env.action_space.n)).astype("float64")

        # # Return empty batch
        # if len(agent.memory) < agent.batch_size:
        #     yield batch_states, batch_target

        actor_idx = np.random.randint(low=1, high=ExaComm.agent_comm.size, size=1)
        data_win.Lock(actor_idx)
        data_win.Get(data_buffer, target_rank=actor_idx)
        data_win.Unlock(actor_idx)

        try:
            agent_data = MPI.pickle.loads(data_buffer)
        except Exception as e:
            BufferDataset._generator(agent, data_buffer, data_win)

        yield (agent_data,)

    def __new__(cls, agent, data_buffer, data_win):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.float32,
            output_shapes=(None, None),
            args=(agent, data_buffer, data_win,)
        )

# def benchmark(dataset, num_epochs=2):
#     start_time = time.perf_counter()
#     for epoch_num in range(num_epochs):
#         for sample in dataset:
#             # Performing a training step
#             time.sleep(0.01)
#     print("Execution time:", time.perf_counter() - start_time)

# def bellman_equation(raw_data):
#     return raw_data + 1

# data_buffer = np.arange(10)
# data_win = 0
# benchmark(BufferDataset(data_buffer, data_win).map(bellman_equation))
# benchmark(BufferDataset(data_buffer, data_win).prefetch(-1).cache().batch(256).map(bellman_equation))
