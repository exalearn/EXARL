import gc
import numpy as np
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
from exarl.utils.introspect import introspectTrace

class ExaMPI(ExaComm):
    """
    This class is built to replacement the pickling of mpi4py.
    The idea is that only numpy arrays will be sent to the
    underlying mpi4py calls thus avoiding pickle.

    Attributes
    ----------
    mpi : MPI
        MPI class used to access comm, sizes, and rank

    comm : MPI.comm
        The underlying communicator

    size : int
        Number of processes in the communicator

    rank : int
        Rank of the current process

    run_length : int
        Flag indicating to use run length encoding

    buffers : map
        These are preallocated buffers for sending and receiving to avoid memory thrashing

    """
    
    MPI = None

    def __init__(self, comm, procs_per_env, num_learners, run_length=False):
        """
        Parameters
        ----------
        comm : MPI Comm, optional
            The base MPI comm to split into sub-comms.  If set to None
            will default to MPI.COMM_WORLD
        procs_per_env : int, optional
            Number of processes per environment (sub-comm)
        num_learners : int, optional
            Number of learners (multi-learner)
        """

        # Singleton
        if ExaMPI.MPI is None:
            mpi4py_rc = True if ExaGlobals.lookup_params('mpi4py_rc') in ["true", "True", 1] else False
            if mpi4py_rc:
                print("Turning mpi4py.rc.threads and mpi4py.rc.recv_mprobe to false!", flush=True)
                import mpi4py.rc
                mpi4py.rc.threads = False
                mpi4py.rc.recv_mprobe = False
            from mpi4py import MPI
            ExaMPI.MPI = MPI

        if comm is None:
            self.comm = ExaMPI.MPI.COMM_WORLD
            self.size = ExaMPI.MPI.COMM_WORLD.Get_size()
            self.rank = ExaMPI.MPI.COMM_WORLD.Get_rank()
        else:
            self.comm = comm
            self.size = comm.size
            self.rank = comm.rank

        self.run_length = run_length
        self.buffers = {}
        super().__init__(self, procs_per_env, num_learners)

    def np_type_converter(self, the_type):
        """
        Gets the appropriate numpy type based on non-numpy types

        Parameters
        ----------
        the_type : type
            Type to convert
        """
        if the_type == float or the_type == np.float64 or the_type == ExaMPI.MPI.DOUBLE:
            return np.float64
        if the_type == np.float32 or the_type == ExaMPI.MPI.FLOAT:
            return np.float32
        if the_type == int or the_type == np.int64 or the_type == ExaMPI.MPI.INT64_T:
            return np.int64
        if the_type == np.int32 or the_type == ExaMPI.MPI.INT:
            return np.int32
        print("Failed to convert type", the_type)
        return the_type

    def mpi_type_converter(self, the_type):
        """
        Gets the appropriate MPI type based on numpy types

        Parameters
        ----------
        the_type : type
            Type to convert
        """
        if the_type == np.int32:
            return ExaMPI.MPI.INT
        if the_type == np.int64:
            return ExaMPI.MPI.INT64_T
        if the_type == np.float32:
            return ExaMPI.MPI.FLOAT
        if the_type == np.float64:
            return ExaMPI.MPI.DOUBLE
        return the_type

    def encode_type(self, the_type):
        """
        Encodes type to int value for sending

        Parameters
        ----------
        the_type : type
            Type to encode
        """
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
        """
        Decodes int to type for receiving, and casts input value as type.

        Parameters
        ----------
        the_type : int
            Int to decode

        cast : bool, optional
            Indicates if we should return type(false) or a value of type
        """
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
        """
        Returns if data is like a list and if not converts to list

        Parameters
        ----------
        data : any
            Data to check if is list/convert

        prep : bool, optional
            Converts data to list and returns as second value if true
        """
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
        """
        Returns if data a float, np.float32, or np.float64

        Parameters
        ----------
        data : any
            Data to check if is float
        """
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
        """
        Gets the flatten size of data

        Parameters
        ----------
        data : any
            Data to check
        """
        list_flag, new_data = self.list_like(data)
        if not list_flag:
            return 1
        return sum([self.get_flat_size(x) for x in data])

    # We use this to encode the np.array shapes
    # We are guaranteeing this is an array
    @introspectTrace()
    def encode_int_list(self, data, buff=None, level=0):
        """
        This converts a list of ints into a flat list while
        encoding the shape.  This is done by flattening
        and adding 2 to each number.

        Parameters
        ----------
        data : list of ints
            Data to encode

        buff : list, optional
            Used internally for recursive call.
            Builds up return list.

        level : int, optional
            Used internally for recursive call.
            Indicates level of nesting.
        """

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
        """
        Reshapes a flat list of ints to list of lists.
        0 and 1 indicate start and end of a list. All
        other values have 2 subtracted from them.

        Parameters
        ----------
        buff: flat list of ints
            Buffer of data to decode

        data : list of list of ints, optional
            Used in recursive call to build up return data list

        index : int, optional
            Current position in decoding buffer

        level : int, optional
            Used internally for recursive call.
            Indicates level of nesting.
        """

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

    @introspectTrace()
    def encode_list_format(self, data, buff=None, np_arrays=None, level=0):
        """
        This call will result in the encoding the type information of a list
        of lists.  Also returns a list of np.array shapes for special np.array
        fast path.

        Parameters
        ----------
        data : any
            The data to encode

        buff : list, optional
            Used internally for recursive call.
            Buffer will hold the typing information.

        np_arrays : np.array, optional
            Used internally for recursive call.
            Array will hold flat data.

        level : int, optional
            Indicates level of nesting.
        """

        if not self.list_like(data, prep=False):
            return self.encode_type(type(data)), np_arrays

        # Everything after this should be list like!
        if buff is None:
            buff = []
        if np_arrays is None:
            np_arrays = []

        # 0 indicates a new list
        # 2 indicates a new np.array
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
        """
        This call decodes the type and shape the originally encoded data.
        np_arrays is for a np.array fast path.

        Parameters
        ----------
        buff : list
            Buffer will hold the typing information.
            Buffer can be supplied for memory reuse.

        data : list
            This is the data that will be return.
            It is also used for internal recursive call.

        is_np : bool
            Fast path for np.array

        np_arrays : np.array, optional
            This is the data that is to be decoded.

        np_index : int, optional
            Used internally for recursive call.
            Indicates level of nesting.

        index : int, optional
            Depth of the recursion.
        """

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
        """
        This implements runlength encoding of ints.
        Runlength encoding takes duplicate values and
        reduces it to two, the number and the number
        of repeating values.

        Parameters
        ----------
        data : list of ints
            This is the data that is to be encoded
        """
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
        """
        This implements runlength encoding of ints.
        Runlength encoding takes duplicate values and
        reduces it to two, the number and the number
        of repeating values. This only supports ints
        so the data must be cast if coming from float buffer

        Parameters
        ----------
        data : list of ints
            This is the data that is to be encoded
        """
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

    @introspectTrace(position=2)
    def buffer(self, data_type, data_count, exact=False):
        """
        Allocates a buffer from a pool of buffers.

        Parameters
        ----------
        data_type : type
            Type of buffer to get

        data_count : int
            Size of buffer

        exact : bool
            Does buffer need to be exact size
        """
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

    @introspectTrace()
    def delete_buffers(self):
        """
        Deallocate all buffers and call the garbage collection
        """
        self.buffers = {}
        gc.collect()

    # This does the marshalling of the data into a buffer
    # This should be called last before sending or right before receiving
    @introspectTrace()
    def marshall(self, data, buff, data_type, data_count=None, index=0, first=True):
        """
        This does the marshalling of the data into a buffer
        This should be called last before sending or right before receiving

        Parameters
        ----------
        data : list
            Data to mashall
        buff : list
            Memory to use for mashalled data
        data_type : type
            Type of data
        data_count : int, optional
            Size of data
        index : int, optional
            Used to keep track of what value is being mashalled within data
        first : bool, optional
            Used to keep track in recursion
        """
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

    @introspectTrace()
    def demarshall(self, data, buff, data_count=None, index=0, first=True):
        """
        This demarshalls the data from a marshall call.
        This should be called immediately after a receive.
        The type is maintained by the data object.
        Be careful to make sure that the types are the same across send/recv.
        If not the underlying buffer type could mismatch and mashalling will fail.

        Parameters
        ----------
        data : list
            data to demarshall

        buff : list
            Size of buffer

        data_cout : int, optional
            Used for flat list fast path

        index : int
            Keeps track of which element we are demarshalling

        first : bool
            Used for recursive calls to track top level call
        """
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

    @introspectTrace()
    def prep_data(self, data, copy=True, default_buffer_type=np.int64):
        """
        This function allocates the buffer to send/recv.
        Will optionally copy data into the buffer.

        Parameters
        ----------
        data : list
            data to prep

        copy : bool, optional
            Indicate if we should marshall data

        default_buffer_type : type, optional
            Buffer type
        """
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
    # Next we send the actual data
    # Can use compression to decrease message size
    @introspectTrace()
    def send_with_type(self, data, dest, default_buffer_type=np.int64):
        """
        This will send messages if we do not know what "types" are in the message.
        First it will find the data format and send it along with the size/buffer type of the data.
        Next we send the actual data.
        Can use compression to decrease message size.

        Parameters
        ----------
        data : list
            data to send

        dest : int
            Rank to send to

        default_buffer_type : type, optional
            Buffer type
        """
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
        self.comm.Send([first, 5, ExaMPI.MPI.INT], dest=dest)

        # Send second message with real data
        return self.comm.Send(second, dest=dest)

    @introspectTrace()
    def send(self, data, dest, pack=False, default_buffer_type=np.int64):
        """
        Point-to-point communication between ranks.  Send must have
        matching recv.

        Parameters
        ----------
        data : any
            Data to be sent
        dest : int
            Rank within comm where data will be sent.
        pack : int, optional
            If we do not know the type on both sides of the send/recv
        """
        # This is if we do not know the type on both sides of the send/recv
        if pack:
            return self.send_with_type(data, dest)
        toSend = self.prep_data(data)
        return self.comm.Send(toSend, dest=dest)

    @introspectTrace()
    def recv_with_type(self, source, default_buffer_type=np.int64):
        """
        The receives messages when we don't know the type ahead of time.
        This follows the procedures outline in send_with_type.

        Parameters
        ----------
        source : int
            Sending rank

        default_buffer_type : type, optional
            Buffer type
        """
        # Recv the message sizes/buffer type
        buff = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        first = [buff, 5, ExaMPI.MPI.INT]
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
        # Extract the actual data
        return self.demarshall(data_shape, buff_data, data_count=data_count)

    @introspectTrace()
    def recv(self, data, source=None, default_buffer_type=np.int64):
        """
        Point-to-point communication between ranks.  Send must have
        matching send.

        Parameters
        ----------
        data : any
            Not used
        dest : int
            Rank within comm where data will be sent. Must have matching recv.
        source : int, optional
            Rank to receive data from.  Default allows data from any source.
        default_buffer_type: type, optional
            Buffer type
        """
        if source is None:
            source = ExaMPI.MPI.ANY_SOURCE
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
        """
        Broadcasts data from the root to all other processes in comm.

        Parameters
        ----------
        data : any
            Data to be broadcast
        root : int
            Indicate which process data comes from
        """
        copy = self.rank == root
        newData = self.prep_data(data, copy=copy)
        self.comm.Bcast(newData, root=root)
        if not copy:
            return self.demarshall(
                data, newData[0][: newData[1]], data_count=newData[1]
            )
        return data

    def barrier(self):
        """
        Block synchronization for the comm.
        """
        return self.comm.Barrier()

    # TODO: This is only supporting single values
    def reduce(self, arg, op, root):
        """
        Data is joined from all processes in comm by doing op.
        Result is placed on root.

        Parameters
        ----------
        arg : any
            Data to reduce
        op : str
            Supports sum, max, and min reductions
        root : int
            Rank the result will end on
        """
        ret_type = type(arg)
        np_type = self.np_type_converter(ret_type)
        mpi_type = self.mpi_type_converter(np_type)
        send_buff = np.array(arg, dtype=np_type)
        recv_buff = np.array(arg, dtype=np_type)
        toSend = [send_buff, 1, mpi_type]
        toRecv = [recv_buff, 1, mpi_type]
        converter = {sum: ExaMPI.MPI.SUM, 
                     max: ExaMPI.MPI.MAX, 
                     min: ExaMPI.MPI.MIN}
        self.comm.Reduce(toSend, toRecv, op=converter[op], root=root)
        return ret_type(toRecv[0])

    # TODO: This is only supporting single values
    def allreduce(self, arg, op=None):
        """
        Data is joined from all processes in comm by doing op.
        Data is put on all processes in comm.

        Parameters
        ----------
        arg : any
            Data to reduce
        op : MPI op, optional
            Operation to perform
        """
        if op is None:
            ExaMPI.MPI.LAND
        ret_type = type(arg)
        np_type = self.np_type_converter(ret_type)
        mpi_type = self.mpi_type_converter(np_type)
        send_buff = np.array(arg, dtype=np_type)
        recv_buff = np.array(arg, dtype=np_type)
        toSend = [send_buff, 1, mpi_type]
        toRecv = [recv_buff, 1, mpi_type]
        converter = {sum: ExaMPI.MPI.SUM, 
                     max: ExaMPI.MPI.MAX, 
                     min: ExaMPI.MPI.MIN}
        self.comm.Allreduce(toSend, toRecv, op=converter[op])
        return ret_type(toRecv[0])

    def time(self):
        """
        Returns MPI wall clock time
        """
        return ExaMPI.MPI.Wtime()

    def split(self, procs_per_env, num_learners):
        """
        This splits the comm into agent, environment, and learner comms.
        Returns three simple sub-comms

        Parameters
        ----------
        procs_per_env : int
            Number of processes per environment comm
        num_learners : int
            Number of processes per learner comm
        """
        if ExaMPI.MPI.COMM_WORLD.Get_size() == procs_per_env:
            assert num_learners == 1, "num_learners should be 1 when global comm size == procs_per_env"
            color = ExaMPI.MPI.UNDEFINED
            if self.rank == 0:
                color = 0
            learner_comm = self.comm.Split(color, self.rank)
            agent_comm = self.comm.Split(color, self.rank)
            if self.rank == 0:
                learner_comm = ExaMPI(comm=learner_comm)
                agent_comm = ExaMPI(comm=agent_comm)
            else:
                learner_comm = None
                agent_comm = None

            env_color = 0
            env_comm = self.comm.Split(env_color, self.rank)
            env_comm = ExaMPI(comm=env_comm)
        else:
            # Agent communicator
            agent_color = ExaMPI.MPI.UNDEFINED
            if (self.rank < num_learners) or ((self.rank - num_learners) % procs_per_env == 0):
                agent_color = 0
            agent_comm = self.comm.Split(agent_color, self.rank)
            if agent_color == 0:
                agent_comm = ExaMPI(comm=agent_comm)
            else:
                agent_comm = None

            # Environment communicator
            if ExaMPI.MPI.COMM_WORLD.Get_size() == procs_per_env:
                env_color = 0
                env_comm = self.comm.Split(env_color, self.rank)
                env_comm = ExaMPI(comm=env_comm)
            else:
                if self.rank < num_learners:
                    env_color = 0
                else:
                    env_color = (int((self.rank - num_learners) / procs_per_env)) + 1
                env_comm = self.comm.Split(env_color, self.rank)
                if env_color > 0:
                    env_comm = ExaMPI(comm=env_comm)
                else:
                    env_comm = None

            # Learner communicator
            learner_color = ExaMPI.MPI.UNDEFINED
            if self.rank < num_learners:
                learner_color = 0
            learner_comm = self.comm.Split(learner_color, self.rank)
            if learner_color == 0:
                learner_comm = ExaMPI(comm=learner_comm)
            else:
                learner_comm = None

        return agent_comm, env_comm, learner_comm

    def raw(self):
        """
        Returns raw MPI comm
        """
        return self.comm

    def printBufSize(self):
        """
        Prints size of internal buffers.
        """
        print("Printing buffers")
        for i in self.buffers:
            for j in self.buffers[i]:
                print(self.rank, "BUFFER:", i, j)
