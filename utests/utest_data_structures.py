import pytest
import numpy as np
from exarl.network.simple_comm import ExaSimple
import exarl.network.data_structures as ds

class TestDataStructure:
    """
    This class contains useful constants, access to MPI networking functionality via simple_comm,
    and helper functions for creating and verifying packets.

    Attributes
    ----------
    comm : exarl.network.simple_comm
        Provides wrapped network comm (MPI communicator)
    packet_size : int
        Standard size of random data to send in bytes
    constructor : dictionary
        Names for easy reference to data structures/constructors
    length_list : list
        Data structure lengths to test
    over_list : list
        Amounts of packets to send past the total length of data structure to test.  This is for testing lost data.
    max_model_lag_list : list
        List of max_model_len values to test
    num_packets_list : list
        List of number of packets for pattern tests
    reps : int
        Number of reps for pattern tests
    max_try: int
        Number of max tries for the pattern free for all tests
    loss_per_rank : number
        This is a number between 0 and 1. Represents the percentage of data that can be lost per rank for pattern tests.
    lossy_length_list : list
        This is the length of the data structure to test for loosing packets.  We invert the numbers because it is easier
        to not loose data for bigger sizes and we want the test to go as far as possible before failing.
    """
    comm = ExaSimple()
    packet_size = 1000
    constructor = {"buff_unchecked": ds.ExaMPIBuffUnchecked,
                   "buff_checked": ds.ExaMPIBuffChecked,
                   "queue_distribute": ds.ExaMPIDistributedQueue,
                   "stack_distribute": ds.ExaMPIDistributedStack}
    # "queue_central": ds.ExaMPICentralizedQueue,
    # "stack_central": ds.ExaMPICentralizedStack

    length_list = [1, 10, 100, 1000]
    over_list = [0, 1, 10, 100]
    max_model_lag_list = [None, 1, 10, 100, 1000]
    num_packets_list = [1, 10, 100]
    reps = 10
    max_try = 1000000
    loss_per_rank = .5
    lossy_length_list = [100000, 10000, 1000, 100, 10]

    def filter(to_remove):
        """
        Convince function to reduce test cases

        Parameters
        ----------
        to_remove : list
            Names to take out of constructors names

        Returns
        -------
        list
            With names removed
        """
        ret = []
        for i in TestDataStructure.constructor:
            for j in to_remove:
                if j not in i:
                    ret.append(i)
        return ret

    def make_packet(self, data_size, seq_num):
        """
        Constructs a standard (testing) packet.
        A packet will contain a sequence number, data size, random data, checksum, and rank.

        Sequence Number: user provided identifier
        Data Size: the size of the random data in bytes
        Data: random bytes
        Checksum: integer sum of all the bytes in random data
        Rank: the rank of the constructing (typically sending) rank

        Parameters
        ----------
        data_size : int
            Size of the data section
        seq_num : int
            Sequence number

        Returns
        -------
        list
            Standard packet
        """
        data = np.random.bytes(data_size)
        checksum = sum([int(x) for x in data])
        return (seq_num, data_size, data, checksum, TestDataStructure.comm.rank)

    def check_packet(self, packet, data_size, name):
        """
        Check a packet conforms to standard construction.  This does not check the sequence number as it is
        a user defined parameter.

        Parameters
        ----------
        packet : tuple
            Packet to check for correctness
        data_size : int
            Size of the data section
        name : string
            Name of the data structure for error reporting
        """
        assert packet is not None, name + " packet is None"
        assert data_size == packet[1], name + " failed packet size should be " + str(data_size) + " but is " + str(packet[1])
        assert data_size == len(packet[2]), name + " size of data should be " + str(data_size) + " but is " + str(len(packet[2]))
        checksum = sum([int(x) for x in packet[2]])
        assert checksum == packet[3], name + " failed packet checksum " + str(checksum) + " but is " + str(packet[3])
        assert packet[4] >= 0 and packet[4] < TestDataStructure.comm.size

    def compare_packet(self, A, B):
        """
        Compares to standard packets element-by-element to check for correctness.
        Returns False if either packet is None.

        Parameters
        ----------
        A : tuple
            First standard packet to compare
        B : tuple
            Second standard packet to compare

        Returns
        -------
        bool
            True if packets match, False otherwise
        """
        if A is None or B is None:
            return False
        if len(A) != len(B):
            return False
        for a, b in zip(A, B):
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            else:
                if a != b:
                    return False
        return True

    def check_order(self, name, seq_num, N):
        """
        Compares the order of sequence numbers popped assuming a given data structure.
        This functions only takes the sequence numbers not a list of standard packets.
        This function assumes that sequence numbers range from 0 to N-1 where N is the
        number of messages.  To use this check, N packets should be pushed into the
        data structure first.  Once all data is pushed, N packets should be popped.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        seq_num : list
            List of sequence numbers in order received from a pop
        N: int
            Number of packets pushed

        Returns
        -------
        bool
            True if packets order is correct, False otherwise
        """
        if "queue" in name:
            return seq_num == list(range(N))
        if "stack" in name:
            return seq_num == list(range(N - 1, -1, -1))
        if "buff_unchecked" in name:
            return seq_num == N * [N - 1]
        if "buff_checked" in name:
            return seq_num == [N - 1]
        return False

    def init_data_structure(self, name, data=None, length=10, rank_mask=True, max_model_lag=None, failPush=False):
        """
        Convenience wrapper for data structure creation.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        data : list
            This should be a standard packet used to initialize the data structure.
            This is required for data structures built on MPI RMA.  The initial
            sequence number of the standard packet should be -1.
        length: int
            Size of the desired data structure
        rank_mask: bool
            What ranks should participate in the data structure.  An example use for this parameter is
            rank_mask=TestDataStructure.comm.rank < 5.  For most tests this parameter should be True to
            include all ranks.

        Returns
        -------
        ExaData :
            The data structure constructed with the requested parameters
        """
        if data is None:
            data = self.make_packet(TestDataStructure.packet_size, -1)
        else:
            assert data[0] == -1, name + " should have an initial standard packet with sequence number -1 but got " + str(data[0])
        return TestDataStructure.constructor[name](TestDataStructure.comm,
                                                   data,
                                                   name=name,
                                                   length=length,
                                                   rank_mask=rank_mask,
                                                   max_model_lag=max_model_lag,
                                                   failPush=failPush)

class TestDataStructureMembers(TestDataStructure):
    """
    This class is a collection of basic tests for data structures.  These include tests to check the basic required
    members and the return values of the methods for popping and pushing data.
    """

    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_name(self, name):
        """
        This test checks the ability to test naming the data structure.
        The name is set by TestDataStructure.constructor in TestDataStructure.init_data_structure.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor.
        """
        data_structure = self.init_data_structure(name)
        assert data_structure.name == name, name + " failed name comparison but got " + data_structure.name
        assert isinstance(data_structure, TestDataStructure.constructor[name])

    @pytest.mark.parametrize("max_model_lag", TestDataStructure.max_model_lag_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_max_model_lag(self, name, max_model_lag):
        """
        This test setting max_model_lag.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        max_model_lag : int
            Value to check
        """
        data_structure = self.init_data_structure(name, max_model_lag=max_model_lag)
        if "buff" in name:
            assert data_structure.max_model_lag is None, name + " max model lag for buffer should be None but is " + data_structure.max_model_lag
        else:
            assert data_structure.max_model_lag == max_model_lag, name + " max model lag not set"

    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_empty(self, name, length):
        """
        This test checks the return of a data structure that is empty.  For an unchecked data structure (e.g. unchecked_buffer)
        the return value should be the data it was initialized with.  Otherwise, a data structure should not return any data.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        """
        data_structure = self.init_data_structure(name, length=length)
        for i in range(length * 2):
            packet = data_structure.pop(TestDataStructure.comm.rank)
            if packet:
                assert "unchecked" in name, name + " should not return packet"
                assert packet[0] == -1, name + " should have -1 sequence number from empty pop " + str(packet[0])
                assert packet[-1] == TestDataStructure.comm.rank, (name +
                                                                   " rank should be " +
                                                                   str(TestDataStructure.comm.rank) +
                                                                   " from empty pop put got "
                                                                   + str(packet[-1]))
                self.check_packet(packet, TestDataStructure.packet_size, name)
            else:
                assert packet is None, name + " should not return a packet"

    # def test_rank_mask(self):
    #     for name in TestDataStructure.constructor:
    #         # data_structure = self.init_data_structure(name, rank_mask=TestClass.comm.rank%2==0)
    #         data_structure = self.init_data_structure(name, rank_mask=False)
    #         for i in range(TestDataStructure.comm.size):
    #             # if i%2==1:
    #             try:
    #                 data_structure.push(self.make_packet(TestDataStructure.packet_size, 0), i)
    #             except:
    #                 print("x")

    @pytest.mark.parametrize("failPush", [True, False])
    @pytest.mark.parametrize("over", TestDataStructure.over_list)
    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_push_return(self, name, length, over, failPush):
        """
        This test checks the return value of the push method.  The push method should return two values:
        Current capacity - what is the current capacity of the data after the push.
        Data Lost - if any data is lost in the transaction.

        The failPush method changes the meaning of lost:
            failPush == True - The data lost value will indicate that the push did not succeed
                0 - successful push
                1 - failed push
            failPush == False - The data lost value in the data structure was overwritten and the push succeeded
                0 - No data was lost
                1 - One entry of data was overwritten

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        over : int
            The amount of pushes to perform over the length of the data structure
        failPush : bool
            Flag to indicate failPush value of data structure
        """
        data_structure = self.init_data_structure(name, length=length, failPush=failPush)
        for i in range(length + over):
            packet = self.make_packet(TestDataStructure.packet_size, i)
            capacity, lost = data_structure.push(packet, TestDataStructure.comm.rank)

            if "buff" in name:
                assert capacity == 1, name + " capacity should be one " + str(capacity)
                if "unchecked" not in name and i > 0:
                    assert lost == 1, name + " can only hold one element so data should be lost on every push" + str(lost)
            else:
                if i < length:
                    assert capacity == i + 1, name + " should have capacity of " + str(i) + " instead of " + str(capacity)
                    assert lost == 0, name + " no data should be lost but got " + str(lost)
                else:
                    assert capacity == length, name + " should have a max capacity of " + str(length) + " instead of " + str(capacity)
                    assert lost == 1, name + " data should be lost but got " + str(lost)

    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_pop_order(self, name, length):
        """
        This test checks the order in which data is popped.  This test is based on the naming convention of
        the data structures found in TestDataStructure.constructor (i.e. queue, stack, buffer)

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        """
        data_structure = self.init_data_structure(name, length=length)

        for i in range(length):
            packet = self.make_packet(TestDataStructure.packet_size, i)
            data_structure.push(packet, TestDataStructure.comm.rank)

        pop_packets = []
        for i in range(length):
            packet = data_structure.pop(TestDataStructure.comm.rank)
            if packet:
                self.check_packet(packet, TestDataStructure.packet_size, name)
                pop_packets.append(packet[0])

        assert self.check_order(name, pop_packets, length), name + " sequence numbers out of order " + str(pop_packets)

    def length_single_rank(self, name, length, over, failPush):
        """
        This function is used to test simple pushing and popping functionality.  Each rank pushes and pops
        from its local data structure.  When over > 0 pops should fail for all checked data structures.
        Further data from pops should correspond to the length of the data structure.  Unchecked data structures
        will return multiple copies of the same data.  We compare the expected received packets with the packets sent.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        over : int
            The amount of pushes to perform over the length of the data structure
        failPush : bool
            Flag to indicate failPush value of data structure
        """
        data_structure = self.init_data_structure(name, length=length, failPush=failPush)

        push_packets = []
        for i in range(length + over):
            packet = self.make_packet(TestDataStructure.packet_size, i)
            data_structure.push(packet, TestDataStructure.comm.rank)
            # Set up the list of packets pushed to compare against
            push_packets.append(packet)
            if i >= data_structure.length:
                if failPush and "buff" not in name:
                    push_packets.pop()
                else:
                    push_packets.pop(0)

        pop_packets = []
        for i in range(length + over):
            packet = data_structure.pop(TestDataStructure.comm.rank)
            if i < data_structure.length:
                self.check_packet(packet, TestDataStructure.packet_size, name)
                pop_packets.append(packet)
            elif "unchecked" not in name:
                assert packet is None, name + " pop should fail"
        assert len(pop_packets) == data_structure.length, (name + " data structure length " +
                                                           str(data_structure.length) +
                                                           " doesn't match " +
                                                           str(len(pop_packets)))

        # For the buffer only the last element should popped
        # In the case of unchecked there will be length copies of the last push
        # For checked buffer there will only be one
        # Reverse sort will capture this
        # The pattern/order is already tested in test_pop_order
        push_packets = sorted(push_packets, key=lambda x: x[0], reverse=True)
        pop_packets = sorted(pop_packets, key=lambda x: x[0], reverse=True)
        for i in range(data_structure.length):
            assert self.compare_packet(push_packets[i], pop_packets[i]), name + " packets don't match " + str(push_packets[i][0]) + " " + str(pop_packets[i][0])

    def length_multiple_ranks(self, name, length, over=0, failPush=False):
        """
        This function is used to test pushing and popping functionality across ranks.  The data structure
        is initialized to a size of length * number of ranks to ensure that when over = 0, no data will be
        lost.  Every rank (including rank 0) pushes length + over packets of data to rank 0.  All ranks
        then hit a barrier.  Rank 0 then pops all the data and checks for the appropriate number of packets.
        If over == 0, we check to see all packets have arrived from each rank.  We are not guaranteed any
        other packets when over > 0 since data will be overwritten.  The last check looks for a packet with
        the last data sequence number.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        over : int
            The amount of pushes to perform over the length of the data structure
        failPush : bool
            Flag to indicate failPush value of data structure
        """
        data_structure = self.init_data_structure(name, length=length * TestDataStructure.comm.size, failPush=failPush)

        for i in range(length + over):
            data_structure.push(self.make_packet(TestDataStructure.packet_size, i), 0)
            TestDataStructure.comm.barrier()

        if TestDataStructure.comm.rank == 0:
            pop_packets = []
            for i in range((length + over) * TestDataStructure.comm.size):
                packet = data_structure.pop(0)
                if i < data_structure.length:
                    self.check_packet(packet, TestDataStructure.packet_size, name)
                    pop_packets.append(packet)
                elif "unchecked" not in name:
                    assert packet is None, name + " pop should fail"
            assert len(pop_packets) == data_structure.length, (name + " data structure length " +
                                                               str(data_structure.length) +
                                                               " doesn't match " +
                                                               str(len(pop_packets)))

            if ((length + over) * TestDataStructure.comm.size) == data_structure.length:
                reduced = sorted([(x[0], x[-1]) for x in pop_packets])
                for i in range(length):
                    ranks = [x[1] for x in reduced if i == x[0]]
                    assert ranks == list(range(TestDataStructure.comm.size)), name + " missing packets " + str(i) + " " + str(ranks)

            # Will have duplicates seq numbers coming from different ranks
            # We can't guarantee the order without more synchronization
            # Instead check to see that the last seq number exists if failPush=False since we are guaranteed someone has to be last
            # Check the inverse is true for failPush=True
            seq_num = set([x[0] for x in pop_packets])
            if failPush and over > 0 and "buff" not in name:
                assert over + length - 1 not in seq_num, name + " sequence number should not be received " + str(over + length - 1)
            else:
                assert over + length - 1 in seq_num, name + " sequence number should be received " + str(over + length - 1)

        TestDataStructure.comm.barrier()

    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_simple_push_pop_single_rank(self, name, length):
        """
        This tests basic pushing and popping to and from local rank.  All pushes and pops
        do not exceed the total size of the data structure's length.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        """
        self.length_single_rank(name, length, over=0, failPush=False)

    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_simple_push_pop_single_multi_rank(self, name, length):
        """
        This tests basic pushing and popping to and from rank 0.  All pushes and pops
        do not exceed the total size of the data structure's length.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        """
        self.length_multiple_ranks(name, length, over=0, failPush=False)

    @pytest.mark.parametrize("failPush", [False, True])
    @pytest.mark.parametrize("over", TestDataStructure.over_list)
    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_failPush_single_rank(self, name, length, over, failPush):
        """
        This tests basic pushing and popping to and from a single rank.  All pushes and pops
        will exceed the total size of the data structure's length to test data loss and failPush.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        over : int
            The amount of pushes to perform over the length of the data structure
        failPush : bool
            Flag to indicate failPush value of data structure
        """
        self.length_single_rank(name, length, over=over, failPush=failPush)

    @pytest.mark.parametrize("failPush", [False, True])
    @pytest.mark.parametrize("over", TestDataStructure.over_list)
    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.constructor.keys())
    def test_failPush_multi_rank(self, name, length, over, failPush):
        """
        TODO: WRITE COMMENT HOW DID I FORGET THIS ONE
        """
        self.length_multiple_ranks(name, length, over=over, failPush=failPush)

class TestMessagePatterClass(TestDataStructure):
    """
    This class is a collection of tests for data structures to test communication patterns used in Exarl.
    """

    @pytest.mark.parametrize("spread", [False, True])
    @pytest.mark.parametrize("num_packets", TestDataStructure.num_packets_list)
    @pytest.mark.parametrize("name", TestDataStructure.filter("buff"))
    def test_sync(self, name, num_packets, spread, reps=TestDataStructure.reps):
        """
        This tests all ranks but one pushing, and rank 0 pops all data.  There is a barrier between
        the pushing and popping phase of the test.  The number of packets and their order is checked.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        num_packets : int
            The number of packets to push per rank
        spread : bool
            True data pushes to local data structure, False pushes data to rank 0
        reps : int
            Number of reps to perform
        """
        if spread:
            data_structure = self.init_data_structure(name, length=num_packets)
        else:
            data_structure = self.init_data_structure(name, length=num_packets * TestDataStructure.comm.size)
        for rep in range(reps):
            # All ranks > 0 will send data
            if TestDataStructure.comm.rank > 0:
                for i in range(num_packets):
                    # Push to my local ds
                    if spread:
                        # Should push to own rank
                        _, loss = data_structure.push(self.make_packet(TestDataStructure.packet_size, i))
                    else:
                        # Pushes to rank 0
                        _, loss = data_structure.push(self.make_packet(TestDataStructure.packet_size, i), 0)
                    assert loss == 0, name + " should not loose any data for sync test"
                # Barrier makes sure all data has been sent
                TestDataStructure.comm.barrier()

            # Rank 0 will read all the data
            else:
                # This keeps track of the seq numbers per rank
                seq_num = [[] for x in range(TestDataStructure.comm.size)]

                # Barrier makes sure we wont pop data until all data is there
                TestDataStructure.comm.barrier()
                for j in range(TestDataStructure.comm.size - 1):
                    for i in range(num_packets):
                        if spread:
                            # Pops from individual rank
                            packet = data_structure.pop(j + 1)
                        else:
                            # Pops from rank 0
                            packet = data_structure.pop(0)
                        self.check_packet(packet, TestDataStructure.packet_size, name)
                        # Update the sequence per rank
                        seq_num[packet[-1]].append(packet[0])

                # Check to see that all packets have arrived and in the correct order
                for i, val in enumerate(seq_num):
                    if i > 0:
                        assert len(val) == num_packets, name + " missing packet " + str(i) + " " + str(len(val)) + " != " + str(num_packets)
                        assert self.check_order(data_structure.name, val, num_packets), (name + " sequence number out of order " +
                                                                                         data_structure.name + " " + str(val))
                    else:
                        assert len(val) == 0, name + " no data should be received from rank 0"

            # Block between iterations
            TestDataStructure.comm.barrier()

    @pytest.mark.parametrize("num_packets", TestDataStructure.num_packets_list)
    @pytest.mark.parametrize("name", TestDataStructure.filter("buff"))
    def test_broadcast(self, name, num_packets, reps=TestDataStructure.reps):
        """
        This tests rank 0 pushing and all other ranks popping data  There is a barrier between
        the pushing and popping phase of the test.  Each packet checks that they receive data from
        rank 0.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        num_packets : int
            The number of packets to push per rank
        reps : int
            Number of reps to perform
        """
        data_structure = self.init_data_structure(name, length=num_packets)
        for rep in range(reps):
            # Rank 0 will send all the data
            if TestDataStructure.comm.rank == 0:
                for j in range(TestDataStructure.comm.size - 1):
                    for i in range(num_packets):
                        # Should push to other ranks
                        _, loss = data_structure.push(self.make_packet(TestDataStructure.packet_size, i), j + 1)
                        assert loss == 0, name + " should not loose any data for broadcast test"
                # Barrier makes sure we wont pop data until all data is there
                TestDataStructure.comm.barrier()

            # All ranks > 0 read data
            else:
                # This keeps track of the seq numbers per rank
                seq_num = [[] for x in range(TestDataStructure.comm.size)]

                # Barrier makes sure all data has been sent
                TestDataStructure.comm.barrier()
                for i in range(num_packets):
                    # Pops from individual rank
                    packet = data_structure.pop(TestDataStructure.comm.rank)
                    self.check_packet(packet, TestDataStructure.packet_size, name)
                    # Update the sequence per rank
                    seq_num[packet[-1]].append(packet[0])

                # Check to see that all packets have arrived and in the correct order
                for i, val in enumerate(seq_num):
                    if i == 0:
                        assert len(val) == num_packets, name + " missing packet " + str(i) + " " + str(len(val)) + " != " + str(num_packets)
                        assert self.check_order(data_structure.name, val, num_packets), (name + " sequence number out of order " +
                                                                                         data_structure.name + " " + str(val))
                    else:
                        assert len(val) == 0, name + " no data should be received from rank 0"

            # Block between iterations
            TestDataStructure.comm.barrier()

    @pytest.mark.parametrize("spread", [False, True])
    @pytest.mark.parametrize("num_packets", TestDataStructure.num_packets_list)
    @pytest.mark.parametrize("length", TestDataStructure.length_list)
    @pytest.mark.parametrize("name", TestDataStructure.filter("buff"))
    def test_free_for_all(self, name, length, num_packets, spread, reps=TestDataStructure.reps, max_try=TestDataStructure.max_try):
        """
        This test has all ranks other than rank 0 pushing data.  At the same time rank 0 will pop data max_try attempts.
        Pushing ranks will continue pushing a given packet until it succeeds.  FailPush=True guaranteeing no data will be
        lost since we push until success.  Buffers cannot be used in the test as they cannot cannot guarantee data will
        not be lost on a push.  We check that all data is received

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        num_packets : int
            The number of packets to push per rank
        spread : bool
            True data pushes to local data structure, False pushes data to rank 0
        reps : int
            Number of reps to perform
        max_try : int
            Number of total read attempts
        """
        data_structure = self.init_data_structure(name, length=length, failPush=True)
        for rep in range(reps):
            # Block between iterations
            TestDataStructure.comm.barrier()

            # All ranks > 0 will send data
            if TestDataStructure.comm.rank > 0:
                for i in range(num_packets):
                    for t in range(max_try):
                        # Push to my local ds
                        if spread:
                            # Should push to own rank
                            capacity, lost = data_structure.push(self.make_packet(TestDataStructure.packet_size, i))
                        else:
                            # Pushes to rank 0
                            capacity, lost = data_structure.push(self.make_packet(TestDataStructure.packet_size, i), 0)
                        # With failPush on this will indicate if the push succeeded
                        if lost == 0:
                            break

            # Rank 0 will read all the data
            else:
                # This keeps track of the seq numbers per rank
                seq_num = [[] for x in range(TestDataStructure.comm.size)]
                # This is the src to read from
                src = 0
                # Total number of received packets
                total_packets = 0
                # Time out based on max_try
                num_try = 0
                while total_packets < (TestDataStructure.comm.size - 1) * num_packets and num_try < max_try:
                    packet = data_structure.pop(src)

                    if packet is not None:
                        self.check_packet(packet, TestDataStructure.packet_size, name)
                        # Update the sequence per rank
                        assert packet[0] > -1, name + " gave invalid packet sequence number " + packet[0]
                        assert not spread or packet[-1] == src, (name + " packet " + str(packet[-1]) + " and src " +
                                                                 str(src) + " do not match for spread: " + spread)
                        seq_num[packet[-1]].append(packet[0])
                        total_packets += 1

                    if spread:
                        src = (src + 1) % TestDataStructure.comm.size
                        src = 1 if src == 0 else src

                    num_try += 1

                # Check to see that all packets have arrived
                assert num_try != max_try, name + " did not collect all packets in less than max_try: " + str(num_try)
                for i, val in enumerate(seq_num):
                    if i > 0:
                        assert len(val) == num_packets, name + " missing packet " + str(i) + " " + str(len(val)) + " != " + str(num_packets)
                        # The order of the data will be based on both the type of data structure and the when the data was pushed/popped
                        # We will just check that all sequence numbers have arrived
                        val = sorted(val)
                        assert self.check_order("queue", val, num_packets), name + " sequence number out of order " + data_structure.name + " " + str(val)
                    else:
                        assert len(val) == 0

        TestDataStructure.comm.barrier()

    @pytest.mark.skip(reason="This test requires manual tuning, but is useful for workflow design")
    @pytest.mark.parametrize("spread", [False, True])
    @pytest.mark.parametrize("failPush", [False, True])
    @pytest.mark.parametrize("num_packets", TestDataStructure.num_packets_list)
    @pytest.mark.parametrize("length", TestDataStructure.lossy_length_list)
    @pytest.mark.parametrize("name", TestDataStructure.filter("buff"))
    def test_lossy_free_for_all(self, name, length, num_packets, failPush, spread,
                                reps=TestDataStructure.reps,
                                max_try=TestDataStructure.max_try,
                                loss_per_rank=TestDataStructure.loss_per_rank):
        """
        This test has all ranks other than rank 0 pushing data.  At the same time rank 0 will pop data max_try attempts.
        Pushing ranks will push a given packet once.  FailPush=True guaranteeing no data will be
        lost since we push until success.  Buffers cannot be used in the test as they cannot cannot guarantee data will
        not be lost on a push.  We check that all data is received.

        This test checks to see how much data is lost in a free for all.  The acceptable amount is not hard and fast.
        We originally set it to 50% of the data as a good approximation, but ultimately the performance of a data
        structure is a combination of number of ranks, length of the data structure, and the number of pop tries by
        rank zero.

        TODO: For RMA workflows consider how this test should be incorporated with the timing of an
        environment vs training.

        Parameters
        ----------
        name : string
            Name of the data structure corresponding to TestDataStructure.constructor
        length : int
            Length of the data structure to initialize
        num_packets : int
            The number of packets to push per rank
        failPush : bool
            Flag to indicate failPush value of data structure
        spread : bool
            True data pushes to local data structure, False pushes data to rank 0
        reps : int
            Number of reps to perform
        max_try : int
            Number of total read attempts
        loss_per_rank : number
            This is a number between 0 and 1. Represents the percentage of data that can be lost per rank.
        """
        data_structure = self.init_data_structure(name, length=length, failPush=failPush)
        for rep in range(reps):
            TestDataStructure.comm.barrier()
            # All ranks > 0 will send data
            if TestDataStructure.comm.rank > 0:
                for i in range(num_packets):
                    # Push to my local ds
                    if spread:
                        # Should push to own rank
                        capacity, lost = data_structure.push(self.make_packet(TestDataStructure.packet_size, i))
                    else:
                        # Pushes to rank 0
                        capacity, lost = data_structure.push(self.make_packet(TestDataStructure.packet_size, i), 0)

            # Rank 0 will read all the data
            else:
                # This keeps track of the seq numbers per rank
                seq_num = [[] for x in range(TestDataStructure.comm.size)]
                # This is the src to read from
                src = 0
                # Total number of received packets
                total_packets = 0
                # Time out based on max_try
                num_try = 0
                while total_packets < (TestDataStructure.comm.size - 1) * num_packets and num_try < max_try:
                    packet = data_structure.pop(src)

                    if packet is not None:
                        self.check_packet(packet, TestDataStructure.packet_size, name)
                        # Update the sequence per rank
                        assert packet[0] > -1, name + " gave invalid packet sequence number " + str(packet[0])
                        assert not spread or packet[-1] == src, (name + " packet " + str(packet[-1]) + " and src " +
                                                                 str(src) + " do not match for spread: " + spread)
                        seq_num[packet[-1]].append(packet[0])
                        total_packets += 1

                    if spread:
                        src = (src + 1) % TestDataStructure.comm.size
                        src = 1 if src == 0 else src

                    num_try += 1

                if TestDataStructure.comm.rank == 0:
                    print("Total Packets:", total_packets)
                    print("Rank", "NumPackets", "NumUniqueSeqNums")
                    for i, val in enumerate(seq_num):
                        print(i, len(val), len(set(val)))
                        print(val)

                # Check how many packets arrived
                for i, val in enumerate(seq_num):
                    if i > 0:
                        val = set(val)
                        assert len(val) >= num_packets * (1 - loss_per_rank), (name + " rank " + str(i) +
                                                                               " did not collect enough packets total unique packets: " +
                                                                               str(len(val)) + " < " + str(num_packets * (1 - loss_per_rank)))
                    else:
                        assert len(val) == 0

        TestDataStructure.comm.barrier()
