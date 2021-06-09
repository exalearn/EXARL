from exarl.utils.introspect import introspectTrace
from exarl.base.comm_base import ExaComm
import gc
import numpy as np

import mpi4py.rc

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI


class ExaMPI(ExaComm):
    mpi = MPI

    def __init__(self, comm=MPI.COMM_WORLD, procs_per_env=1, run_length=False):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.run_length = run_length
        self.buffers = {}
        super().__init__(self, procs_per_env)

    def np_type_converter(self, the_type):
        if the_type == float or the_type == np.float64 or the_type == MPI.DOUBLE:
            return np.float64
        if the_type == np.float32 or the_type == MPI.FLOAT:
            return np.float32
        if the_type == int or the_type == np.int64 or the_type == MPI.INT64_T:
            return np.int64
        if the_type == np.int32 or the_type == MPI.INT:
            return np.int32
        print("Failed to convert type", the_type)
        return the_type

    def mpi_type_converter(self, the_type):
        if the_type == np.int32:
            return MPI.INT
        if the_type == np.int64:
            return MPI.INT64_T
        if the_type == np.float32:
            return MPI.FLOAT
        if the_type == np.float64:
            return MPI.DOUBLE
        return the_type

    def encode_type(self, the_type):
        type_map = {
            int: 3,
            float: 4,
            bool: 5,
            np.int32: 6,
            np.int64: 7,
            np.float32: 8,
            np.float64: 9,
        }
        if the_type in type_map:
            return type_map[the_type]
        print("Encode: Type", the_type, "unsupported", flush=True)

    def decode_type(self, the_type, cast=True):
        type_map = {
            3: int,
            4: float,
            5: bool,
            6: np.int32,
            7: np.int64,
            8: np.float32,
            9: np.float64,
        }
        if the_type in type_map:
            if cast:
                return type_map[the_type](the_type)
            else:
                return type_map[the_type]
        print("Decode: Type code", the_type, "unsupported", flush=True)

    def list_like(self, data, prep=True):
        if isinstance(data, range):
            list_flag = True
            cast_flag = True
        elif isinstance(data, tuple):
            list_flag = True
            cast_flag = True
        elif isinstance(data, np.ndarray):
            list_flag = True
            cast_flag = False
        elif isinstance(data, list):
            list_flag = True
            cast_flag = False
        else:
            list_flag = False
            cast_flag = False

        if prep:
            toSend = data
            if cast_flag:
                toSend = list(data)
            elif not list_flag:
                toSend = [data]
            return list_flag, toSend
        return list_flag

    def is_float(self, data):
        list_flag, new_data = self.list_like(data)
        if not list_flag:
            if (
                isinstance(data, float)
                or isinstance(data, np.float32)
                or isinstance(data, np.float64)
            ):
                return True
            return False
        return any([self.is_float(x) for x in data])

    def get_flat_size(self, data):
        list_flag, new_data = self.list_like(data)
        if not list_flag:
            return 1
        return sum([self.get_flat_size(x) for x in data])

    # We use this to encode the np.array shapes
    # We are guarenteing this is an array
    @introspectTrace()
    def encode_int_list(self, data, buff=None, level=0):
        if buff is None:
            buff = []
        # 0 indicates a new list
        # 1 indicates the end of the list
        # # > 2 is the value of the number in list
        # Everything should be type int
        buff.append(0)
        for i, x in enumerate(data):
            if self.list_like(data[i], prep=False):
                self.encode_int_list(data[i], buff=buff, level=level + 1)
            else:
                assert data[i] + 2 > 1
                buff.append(data[i] + 2)
        buff.append(1)
        return buff

    # This decodes the np.array shapes
    # Again we assume we are dealing with a list
    @introspectTrace()
    def decode_int_list(self, buff, data=None, index=0, level=0):
        # Decode is based on the encoding in encode_int_list
        while index < len(buff):
            if buff[index] == 1:
                return index + 1
            elif buff[index] == 0:
                if data is None:
                    data = []
                    index = self.decode_int_list(
                        buff, data=data, index=index + 1, level=level + 1
                    )
                else:
                    data.append([])
                    index = self.decode_int_list(
                        buff, data=data[-1], index=index + 1, level=level + 1
                    )
            else:
                data.append(buff[index] - 2)
                index = index + 1
        if level > 0:
            return index
        return data

    # This is a general for of the encode_int_list above
    # It should support single values
    # Also special support for np arrays -- we generate a list of shapes
    # Ecoding will replace ranges and tuples for lists!
    @introspectTrace()
    def encode_list_format(self, data, buff=None, np_arrays=None, level=0):
        if not self.list_like(data, prep=False):
            return self.encode_type(type(data)), np_arrays

        # Everything after this should be list like!
        if buff is None:
            buff = []
        if np_arrays is None:
            np_arrays = []

        # 0 indicates a new list
        # 2 indecates a new np.array
        # 1 end of list (either list or np.array)
        if isinstance(data, np.ndarray):
            np_arrays.append(np.shape(data))
            data = data.flatten()
            buff.append(2)
        else:
            buff.append(0)
        for i, x in enumerate(data):
            if self.list_like(data[i], prep=False):
                self.encode_list_format(
                    data[i], buff=buff, np_arrays=np_arrays, level=level + 1
                )
            else:
                buff.append(self.encode_type(type(x)))
        buff.append(1)
        return buff, np_arrays

    # This decodes the general purpose list encoding
    # Also supports single values and np.arrays
    # This requires the np.array shapes
    @introspectTrace()
    def decode_list_format(
        self, buff, data=None, is_np=False, np_arrays=None, np_index=0, index=0
    ):
        if not self.list_like(buff, prep=False):
            return self.decode_type(buff)

        # Everything after this should be list like!
        first = data is None
        # Decode is based on the encoding in encode_list_format
        while index < len(buff):
            if buff[index] == 1:
                if is_np:
                    data = np.array(data).reshape(np_arrays[np_index])
                return data, index + 1, np_index + 1
            elif buff[index] == 0 or buff[index] == 2:
                np_array = buff[index]
                if data is None:
                    data = []
                    data, index, np_index = self.decode_list_format(
                        buff,
                        data=data,
                        is_np=np_array,
                        np_arrays=np_arrays,
                        np_index=np_index,
                        index=index + 1,
                    )
                else:
                    data.append([])
                    data[-1], index, np_index = self.decode_list_format(
                        buff,
                        data=data[-1],
                        is_np=np_array,
                        np_arrays=np_arrays,
                        np_index=np_index,
                        index=index + 1,
                    )
            else:
                data.append(self.decode_type(buff[index]))
                index = index + 1

        if first:
            return data
        return data, index, np_index

    # This is an implementation of run length encoding for integers only
    @introspectTrace()
    def run_length_encode(self, data):
        if self.run_length:
            if not self.list_like(data, prep=False):
                return data, 1
            # Lists from here on out!
            # The encoding is number first then the count
            prev_num = data[0]
            encoding = [prev_num]
            count = 1
            for x in data[1:]:
                if x != prev_num:
                    encoding.append(count)
                    encoding.append(x)
                    count = 1
                    prev_num = x
                else:
                    count += 1
            encoding.append(count)
            if len(encoding) < len(data):
                data[: len(encoding)] = encoding
        return data

    # This decodes run_length_encode
    # This only supports ints so the data must be cast if coming from float buffer
    @introspectTrace()
    def run_length_decode(self, data):
        if self.run_length:
            if not self.list_like(data, prep=False):
                return data

            decode = []
            num = None
            for i, x in enumerate(data):
                if i & 1:
                    # Odd
                    decode.extend([num] * int(x))
                else:
                    # Even
                    num = x
            return decode
        return data

    # This allocates a buffer from a pool of buffers
    @introspectTrace(position=2)
    def buffer(self, data_type, data_count, exact=False):
        if data_type in self.buffers:
            temp = [x for x in self.buffers[data_type].keys() if x >= data_count]
            if len(temp):
                size_key = min(temp)
                if (size_key == data_count) or (size_key and not exact):
                    return self.buffers[data_type][size_key]
        else:
            self.buffers[data_type] = {}
        self.buffers[data_type][data_count] = np.empty(data_count, dtype=data_type)
        return self.buffers[data_type][data_count]

    # Deallocate all buffers and call the garbage collection
    @introspectTrace()
    def delete_buffers(self):
        self.buffers = {}
        gc.collect()

    # This does the marshalling of the data into a buffer
    # This should be called last before sending or right before receiving
    @introspectTrace()
    def marshall(self, data, buff, data_type, data_count=None, index=0, first=True):
        # Handles a single value
        if not self.list_like(data, prep=False):
            buff[0] = data_type(data)
            return buff
        # This is a simple copy for only flat lists
        if not index and data_count and data_count == len(data):
            buff[: len(data)] = [data_type(x) for x in data]
            return buff
        # This does marshalling of complex list_like of list_like
        for i, x in enumerate(data):
            if self.list_like(data[i], prep=False):
                index = self.marshall(
                    data[i], buff, data_type, index=index, first=False
                )
            else:
                buff[index] = data_type(data[i])
                index = index + 1
        if first:
            return buff
        return index

    # This demarshalls the data from a marshall...
    # This should be called immediatly after a receive
    # The type is maintained by the data object
    # Be careful to make sure that the types are the same across send/recv
    # If not the underlying buffer type could mismatch and mashalling will fail
    @introspectTrace()
    def demarshall(self, data, buff, data_count=None, index=0, first=True):
        # Handles a single value
        if not self.list_like(data, prep=False):
            data_type = type(buff[0])
            return data_type(buff[0])
        # This is a simple copy for only flat lists
        if not index and data_count and data_count == len(data):
            newData = buff[:data_count]
            data = [type(x)(y) for x, y in zip(data, newData)]
            return data
        # This does demarshalling of complex list_like of list_like
        for i, x in enumerate(data):
            if self.list_like(data[i], prep=False):
                index = self.demarshall(data[i], buff, index=index, first=False)
            else:
                # If there is a buffer mismatch it will break here...
                data_type = type(data[i])
                data[i] = data_type(buff[index])
                index = index + 1
        if first:
            return data
        return index

    # This function allocates the buffer to send/recv
    # Will optionally copy data into the buffer
    @introspectTrace()
    def prep_data(self, data, copy=True, default_buffer_type=np.int64):
        data_count = self.get_flat_size(data)
        # Look for a float, if none then set to default.
        # This can cause a mismatch...
        data_type = np.float64 if self.is_float(data) else default_buffer_type
        if data_count == 1:
            _, new_data = self.list_like(data, prep=True)
            buff = self.buffer(data_type, data_count)
            buff[0] = data_type(new_data[0])
        elif data_count == len(data) and isinstance(data, np.ndarray):
            data_type = data.dtype
            data_count = data.size
            buff = data
        else:
            buff = self.buffer(data_type, data_count)
            if copy:
                buff = self.marshall(data, buff, data_type, data_count=data_count)
        return [buff, data_count, self.mpi_type_converter(data_type)]

    # This will send messages if we do not know what "types" are in the message
    # First it will find the data format and send it along with the size/buffer type of the data
    # Next we send the actuall data
    # Can use compression to decrease message size
    @introspectTrace()
    def send_with_type(self, data, dest, default_buffer_type=np.int64):
        # Get the data properties
        data_count = self.get_flat_size(data)
        data_type = np.float64 if self.is_float(data) else default_buffer_type
        data_shape, np_arrays = self.encode_list_format(data)
        data_shape_len = len(data_shape)

        # Special support for np.array to recover their structure
        np_arrays_flat = self.encode_int_list(np_arrays)
        np_arrays_size = len(np_arrays_flat)

        # Prep message with data and formats
        data_send = [np_arrays_flat, data_shape, data]
        second = self.prep_data(data_send, copy=True, default_buffer_type=data_type)
        second[0] = self.run_length_encode(second[0])
        compress_size = len(second[0])

        # Send first message with sizes and types
        first = np.array(
            [
                np_arrays_size,
                data_shape_len,
                data_count,
                compress_size,
                self.encode_type(data_type),
            ],
            dtype=np.int32,
        )
        self.comm.Send([first, 5, MPI.INT], dest=dest)

        # Send second message with real data
        return self.comm.Send(second, dest=dest)

    @introspectTrace()
    def send(self, data, dest, pack=False, default_buffer_type=np.int64):
        # This is if we do not know the type on both sides of the send/recv
        if pack:
            return self.send_with_type(data, dest)
        toSend = self.prep_data(data)
        return self.comm.Send(toSend, dest=dest)

    # The receives messagse when we don't know the type ahead of time
    # This follows the procedures outline in send_with_type
    @introspectTrace()
    def recv_with_type(self, source, default_buffer_type=np.int64):
        # Recv the message sizes/buffer type
        buff = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        first = [buff, 5, MPI.INT]
        self.comm.Recv(first, source=source)

        # Unpack data properties
        np_arrays_shape_size = buff[0]
        data_shape_len = buff[1]
        data_count = buff[2]
        compress_size = buff[3]
        data_type = self.decode_type(buff[4], cast=False)

        data_total = np_arrays_shape_size + data_shape_len + data_count
        buff = self.buffer(data_type, compress_size)
        second = [buff, compress_size, self.mpi_type_converter(data_type)]
        self.comm.Recv(second, source=source)
        if compress_size < data_total:
            buff = self.run_length_decode(buff)

        # These are the three elements of the message
        buff_np_arrays = [int(x) for x in buff[:np_arrays_shape_size]]
        buff_shape = buff[np_arrays_shape_size: np_arrays_shape_size + data_shape_len]
        buff_data = buff[np_arrays_shape_size + data_shape_len: data_total]

        # Expand the np.arrays
        np_arrays = self.decode_int_list(buff_np_arrays)
        # Expand the data format with np.arrays
        data_shape = self.decode_list_format(buff_shape, np_arrays=np_arrays)
        # Extract the actuall data
        return self.demarshall(data_shape, buff_data, data_count=data_count)

    @introspectTrace()
    def recv(self, data, source=MPI.ANY_SOURCE, default_buffer_type=np.int64):
        # This is if we do not know the type on both sides of the send/recv
        if data is None:
            return self.recv_with_type(source)
        toRecv = self.prep_data(
            data, copy=False, default_buffer_type=default_buffer_type
        )
        self.comm.Recv(toRecv, source=source)
        return self.demarshall(data, toRecv[0][: toRecv[1]], data_count=toRecv[1])

    # Broadcasts must know the data format on both sides
    @introspectTrace()
    def bcast(self, data, root):
        copy = self.rank == root
        newData = self.prep_data(data, copy=copy)
        self.comm.Bcast(newData, root=root)
        if not copy:
            return self.demarshall(
                data, newData[0][: newData[1]], data_count=newData[1]
            )
        return data

    def barrier(self):
        return self.comm.Barrier()

    # TODO: This is only supporting single values
    def reduce(self, arg, op, root):
        ret_type = type(arg)
        np_type = self.np_type_converter(ret_type)
        mpi_type = self.mpi_type_converter(np_type)
        send_buff = np.array(arg, dtype=np_type)
        recv_buff = np.array(arg, dtype=np_type)
        toSend = [send_buff, 1, mpi_type]
        toRecv = [recv_buff, 1, mpi_type]
        converter = {sum: MPI.SUM, max: MPI.MAX, min: MPI.MIN}
        self.comm.Reduce(toSend, toRecv, op=converter[op], root=root)
        return ret_type(toRecv[0])

    # TODO: This is only supporting single values
    def allreduce(self, arg, op=MPI.LAND):
        ret_type = type(arg)
        np_type = self.np_type_converter(ret_type)
        mpi_type = self.mpi_type_converter(np_type)
        send_buff = np.array(arg, dtype=np_type)
        recv_buff = np.array(arg, dtype=np_type)
        toSend = [send_buff, 1, mpi_type]
        toRecv = [recv_buff, 1, mpi_type]
        converter = {sum: MPI.SUM, max: MPI.MAX, min: MPI.MIN}
        self.comm.Allreduce(toSend, toRecv, op=converter[op], root=root)
        return ret_type(toRecv[0])

    def time(self):
        return MPI.Wtime()

    def split(self, procs_per_env):
        # Agent communicator
        agent_color = MPI.UNDEFINED
        if (self.rank == 0) or ((self.rank + procs_per_env - 1) % procs_per_env == 0):
            agent_color = 0
        agent_comm = self.comm.Split(agent_color, self.rank)

        # Environment communicator
        if self.rank == 0:
            env_color = 0
        else:
            env_color = (int((self.rank - 1) / procs_per_env)) + 1
        env_comm = self.comm.Split(env_color, self.rank)

        if agent_color == 0:
            agent_comm = ExaMPI(comm=agent_comm)
        else:
            agent_comm = None
        env_comm = ExaMPI(comm=env_comm)
        return agent_comm, env_comm

    def printBufSize(self):
        print("Printing buffers")
        for i in buffers:
            for j in buffers[i]:
                print(self.rank, "BUFFER:", i, j)
