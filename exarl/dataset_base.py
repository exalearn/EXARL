import tensorflow as tf


class BufferDataset(tf.data.Dataset):
    def _generator(data_buffer, data_win, actor_idx):
        # Get data buffer from RMA window
        data_win.Lock(actor_idx)
        data_win.Get(data_buffer, target_rank=actor_idx)
        data_win.Unlock(actor_idx)
        yield (data_buffer,)

    def __new__(cls, data_buffer, data_win, actor_idx):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            args=(data_buffer, data_win, actor_idx,)
        )
