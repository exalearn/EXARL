
import numpy as np
import exarl
from exarl.mpi_comm import ExaMPI
import random

def make_list(data):
    new_data = data
    if isinstance(data, tuple) or isinstance(data, range):
        new_data = list(data)
    if isinstance(new_data, list):
        return [make_list(x) for x in new_data]
    return new_data
    
my_comm = ExaMPI(procs_per_env=1)

data = [-1, 0.0, 0]
if my_comm.rank == 0:
    my_comm.send(data, 1)
if my_comm.rank == 1:
    recv_data = my_comm.recv(data, 0)
    print(data)
    print(recv_data)
    assert data == recv_data

# data = [[],[]] #[[[[3], 2], 3], 2]
# if my_comm.rank == 0:
#     my_comm.send(data, 1, pack=True)
# if my_comm.rank == 1:
#     recv_data = my_comm.recv(None, 0)
#     print(data)
#     print(recv_data)
#     assert data == recv_data

# alist = [0, 1, 2, 3, 4]
# list_flag, toSend = my_comm.list_like(alist)
# assert list_flag and alist == toSend

# list_flag, toSend = my_comm.list_like(range(5))
# assert list_flag and alist == toSend

# nplist = np.arange(5)
# list_flag, toSend = my_comm.list_like(nplist)
# assert list_flag and np.array_equal(nplist, toSend)

# value = 10
# list_flag, toSend = my_comm.list_like(value)
# assert not list_flag and [value] == toSend

# alist = [[0, 1], 2, [3, (4, 0.5)], [6, 7, [8]], 9]
# assert my_comm.is_float(alist)
# assert my_comm.get_flat_size(alist) == 10

# alist = [[0, 1], 2, [3, (4, 5)], [6, 7, [8]], 9]
# assert not my_comm.is_float(alist)
# assert my_comm.get_flat_size(alist) == 10

# buff = my_comm.buffer(np.float64, 10, exact=True)
# assert len(buff) == 10 and buff.dtype == np.float64
# assert len(my_comm.buffers[np.float64][10]) == 10 and my_comm.buffers[np.float64][10].dtype == np.float64

# buff = my_comm.buffer(np.float64, 5, exact=True)
# assert len(my_comm.buffers[np.float64][5]) == 5 and my_comm.buffers[np.float64][5].dtype == np.float64
# assert len(buff) == 5 and buff.dtype == np.float64

# buff = my_comm.buffer(np.float64, 2, exact=True)
# assert len(my_comm.buffers[np.float64][2]) == 2 and my_comm.buffers[np.float64][2].dtype == np.float64
# assert len(buff) == 2 and buff.dtype == np.float64

# buff = my_comm.buffer(np.float64, 7, exact=False)
# assert 7 not in my_comm.buffers[np.float64] 
# assert len(buff) == 10 and buff.dtype == np.float64

# buff = my_comm.buffer(np.float64, 3, exact=False)
# assert 3 not in my_comm.buffers[np.float64] 
# assert len(buff) == 5 and buff.dtype == np.float64

# my_comm.delete_buffers()
# assert not my_comm.buffers

# buff = my_comm.buffer(np.int64, 10, exact=True)
# for i in range(10):
#     my_comm.marshall([9-i] * (10-i), buff, np.int64, data_count=(10-i))

# assert all([isinstance(x, np.int64) for x in buff])
# assert all([float(i) == x for i, x in enumerate(buff)])

# for i in range(10):
#     data = my_comm.demarshall(range(i), buff, data_count=i)
#     assert len(data) == i
#     assert all([isinstance(x, int) for x in data])
#     assert all([i == x for i, x in enumerate(data)])

# buff = my_comm.buffer(np.float64, 10, exact=True)
# data = [[0, [True, 2]], (3.0, 4), (5, (6.0, 7), [8, 9])]
# flat = my_comm.marshall(data, buff, np.float64)
# print(flat)
# assert all([float(i) == x for i, x in enumerate(flat)])

# dataFormat = [[0, [False, 0]], (0.0, 0), (0, (0.0, 0), [0, 0])]
# dataCheck = [[0, [True, 2]], [3.0, 4], [5, [6.0, 7], [8, 9]]]
# data = my_comm.demarshall(dataFormat, buff)
# assert data == make_list(dataCheck)

# data = [1, [[]], 1, 1]
# flat = my_comm.marshall(data, buff, np.int64)
# data2 = my_comm.demarshall(data, buff)
# assert data == data2

# buff, data_count, data_type = my_comm.prep_data([False, 1, 2, 3])
# assert len(buff) >= 4 and data_count == 4
# assert buff.dtype == np.int64
# assert all([i==x for i, x in enumerate(buff[:4])])

# buff, data_count, data_type = my_comm.prep_data([False, 1, 2, 3, 4.0])
# assert len(buff) >=5 and data_count == 5
# assert buff.dtype == np.float64
# assert all([i==x for i, x in enumerate(buff[:5])])

# buff, data_count, data_type = my_comm.prep_data(range(10))
# assert len(buff) >=10 and data_count == 10
# assert buff.dtype == np.int64
# assert all([i==x for i, x in enumerate(buff[:10])])

# buff, data_count, data_type = my_comm.prep_data([4])
# assert len(buff) >=1 and data_count == 1
# assert buff.dtype == np.int64
# assert buff[0] == 4

# buff, data_count, data_type = my_comm.prep_data([3.14])
# assert len(buff) >=1 and data_count == 1
# assert buff.dtype == np.float64
# assert buff[0] == 3.14

# my_comm.delete_buffers()

# if my_comm.rank == 0:
#     value = 3
#     my_comm.send(value, 1)
# if my_comm.rank == 1:
#     value = 0
#     value = my_comm.recv(value, 0)
#     assert value == 3

# my_comm.delete_buffers()
# data = range(10)
# for i in range(10):
#     dataFormat = [0] * 10
#     if i % 2 == 0:
#         if my_comm.rank == 0:
#             my_comm.send(data[:i+1], 1)
#         if my_comm.rank == 1:
#             recv_data = my_comm.recv(dataFormat[:i+1], 0)
#             assert all([i==x for i, x in enumerate(recv_data)])
#     else:
#         if my_comm.rank == 0:
#             recv_data = my_comm.recv(dataFormat[:i+1], 1)
#             assert all([i==x for i, x in enumerate(recv_data)])
#         if my_comm.rank == 1:
#             my_comm.send(data[:i+1], 0)

# my_comm.delete_buffers()
# buff = my_comm.buffer(np.float64, 5, exact=True)
# data = [0.0, 1.0, 2.0, 3.0, 4.0]
# for i in range(5):
#     dataFormat = [0.0] * 5
#     if i % 2 == 0:
#         if my_comm.rank == 0:
#             my_comm.send(data[:i+1], 1)
#         if my_comm.rank == 1:
#             recv_data = my_comm.recv(data[:i+1], 0)

#             assert all([float(i)==x for i, x in enumerate(recv_data)])
#     else:
#         if my_comm.rank == 0:
#             recv_data = my_comm.recv(data[:i+1], 1)
#             assert all([float(i)==x for i, x in enumerate(recv_data)])
#         if my_comm.rank == 1:
#             my_comm.send(data[:i+1], 0)

# data = my_comm.bcast(3, 0)
# assert data == 3

# my_comm.delete_buffers()
# data = my_comm.bcast([0, 1, 2, 3], 0)
# assert len(data) == 4
# assert all([i==x for i, x in enumerate(data)])

# data = my_comm.bcast([0.0, 1.0, 2, 3], 1)
# assert len(data) == 4
# assert all([float(i)==x for i, x in enumerate(data)])

# data = my_comm.bcast([0.0, 1.0, (1, 2), 3], 1)
# print(my_comm.rank, data)
# if my_comm.rank == 0:
#     assert data == [0.0, 1.0, [1.0, 2.0], 3.0]

# if my_comm.size > 4:
#     agent_comm, env_comm = my_comm.split(2)
#     print("agent", agent_comm.global_rank, agent_comm.rank)
#     print("env", env_comm.global_rank, env_comm.rank)

# if exarl.ExaComm.agent_comm:
#     print("Env", exarl.ExaComm.global_comm.rank, exarl.ExaComm.agent_comm.rank)
# print("Agent", exarl.ExaComm.global_comm.rank, exarl.ExaComm.env_comm.rank)

print("Done.")