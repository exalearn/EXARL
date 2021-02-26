import tensorflow as tf
import numpy as np
import time

class BufferDataset(tf.data.Dataset):

    data_exchange = None
    comm_size = None
    output_type = None

    def _generator():
        agent_data = None
        while agent_data is None:
            actor_idx = 0
            if BufferDataset.comm_size > 1:
                actor_idx = np.random.randint(low=1, high=BufferDataset.comm_size, size=1)
            agent_data = BufferDataset.data_exchange.pop(actor_idx)
            print("Pop", agent_data)
            # tf_data = tf.convert_to_tensor(agent_data, dtype=BufferDataset.output_type)
        yield (agent_data,)

    def __new__(cls, data_exchange, comm_size, output_types=tf.float64, output_shapes=(None, None)):
        if BufferDataset.data_exchange is None and BufferDataset.comm_size is None and BufferDataset.output_type is None:
            BufferDataset.data_exchange = data_exchange
            BufferDataset.comm_size = comm_size
            BufferDataset.output_type = output_types
        else:
            print("Failed attempt to reinitialize BufferDataset")

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
