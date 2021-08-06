from mpi4py import MPI
import numpy as np
import struct
import sys
#################################################################
# Global constants
#################################################################
comm = MPI.COMM_WORLD
batch_size = 64
model_size = 208
#################################################################
# Functions
#################################################################
def get_time_original(epsilon_win, indices_win, loss_win, model_win, iter=100000, out_prefix = ""):
    # Start the microbenchmark
    if comm.rank==1:
        buff = bytearray(model_size)
        epsilon = np.ones(1, dtype=np.float64)
        indices = np.ones(batch_size, dtype=np.intc)
        loss = np.ones(batch_size, dtype=np.float64)

        start = MPI.Wtime()
        for _ in range(iter):
            model_win.Lock(0)
            model_win.Rget(buff, target_rank=0)
            model_win.Unlock(0)

            epsilon_win.Lock(0)
            epsilon_win.Rget(epsilon, target_rank=0)
            epsilon_win.Unlock(0)

            indices_win.Lock(0)
            indices_win.Rget(indices, target_rank=0)
            indices_win.Unlock(0)

            loss_win.Lock(0)
            loss_win.Rget(loss, target_rank=0)
            loss_win.Unlock(0)
        end = MPI.Wtime()

        total_exec_time = end - start
        us_exec_time = total_exec_time * 1e6
        print("---------------")
        #print(out_prefix+"Multi Windows Total exec time  : {} s".format(total_exec_time))
        print(out_prefix+"Multi Windows Sample exec time : {} us".format(us_exec_time/iter))

    comm.Barrier()

def single_win_pickle(iter=100000):
        # -- Single window test
        # epsilon - indices - loss - model
        # 3       - 2222222 - 1111 - 8888
        if comm.rank == 0:
            epsilon = 3*np.ones(1, dtype=np.float64)
            indices = 2*np.ones(batch_size, dtype=np.intc)
            loss = 1*np.ones(batch_size, dtype=np.float64)
            model = bytearray(model_size*[8])

        else:
            epsilon = np.zeros(1, dtype=np.float64)
            indices = np.zeros(batch_size, dtype=np.intc)
            loss = np.zeros(batch_size, dtype=np.float64)
            model = bytearray(model_size)

        serial_data = (MPI.pickle.dumps([epsilon,indices,loss,model]))
        serial_size = len(serial_data)
        win_size = serial_size if comm.rank == 0 else 0
        single_win = MPI.Win.Allocate(win_size, MPI.BYTE.Get_size(), comm=comm)

        if comm.rank == 0:
            buf =  (MPI.pickle.dumps([epsilon,indices,loss,model]))
            single_win.Lock(0)
            single_win.Put(buf, target_rank=0)
            single_win.Unlock(0)
            comm.Barrier()
        else:
            comm.Barrier()

            recv_buf = bytearray(serial_size)
            start = MPI.Wtime()
            for _ in range(iter):
                single_win.Lock(0)
                single_win.Get(recv_buf, target_rank=0)
                single_win.Unlock(0)
                #print(buf)
                l = MPI.pickle.loads(recv_buf)
                #print(l)
                #print(model)
            end = MPI.Wtime()

            total_exec_time = end - start
            us_exec_time = total_exec_time * 1e6
            print("---------------")
            #print("Single Window pickle Total exec time  : {} s".format(total_exec_time))
            print("Single Window pickle Sample exec time : {} us".format(us_exec_time/iter))
        comm.Barrier()
        single_win.Free()

def single_win_pack(iter=100000):
        # -- Single window test
        # epsilon - indices - loss - model
        # 3       - 2222222 - 1111 - 8888
        if comm.rank == 0:
            epsilon = 3*np.ones(1, dtype=np.float64)
            indices = 2*np.ones(batch_size, dtype=np.intc)
            loss = 1*np.ones(batch_size, dtype=np.float64)
            model = bytearray(model_size*[8])

        else:
            epsilon = np.zeros(1, dtype=np.float64)
            indices = np.zeros(batch_size, dtype=np.intc)
            loss = np.zeros(batch_size, dtype=np.float64)
            model = bytearray(model_size)

        serial_data = (MPI.pickle.dumps([epsilon,indices,loss,model]))
        serial_size = len(serial_data)
        win_size = serial_size if comm.rank == 0 else 0
        single_win = MPI.Win.Allocate(win_size, MPI.BYTE.Get_size(), comm=comm)
        buf_format= "@{}d{}i{}d".format(1,batch_size,batch_size)

        if comm.rank == 0:
            buf = struct.pack(buf_format,*epsilon,*indices,*loss)
            buf += model
            single_win.Lock(0)
            single_win.Put(buf, target_rank=0)
            single_win.Unlock(0)
            comm.Barrier()
        else:
            comm.Barrier()
            win_size = MPI.DOUBLE.Get_size() + MPI.INT.Get_size()*batch_size + MPI.DOUBLE.Get_size()*batch_size + model_size
            buf = bytearray(win_size)
            start = MPI.Wtime()
            for _ in range(iter):
                single_win.Lock(0)
                single_win.Get(buf, target_rank=0)
                single_win.Unlock(0)
                #print(buf)
                model = buf[-model_size:]
                l = list(struct.unpack(buf_format,buf[:len(buf)-model_size]))
                #epsilon = l[0]
                #indices = l[1:51]
                #loss = l[52:102]
                #print(l,model)
                #print(model)
            end = MPI.Wtime()

            total_exec_time = end - start
            us_exec_time = total_exec_time * 1e6
            print("---------------")
            #print("Single Window pack Total exec time  : {} s".format(total_exec_time))
            print("Single Window pack Sample exec time : {} us".format(us_exec_time/iter))
        comm.Barrier()
        single_win.Free()
#################################################################
# Main
#################################################################
if __name__=="__main__":

    if comm.size != 2 :
        print("Error : comm.size != 2")
        sys.exit(1)
    # -- Original test
    # init windows
    if comm.rank == 0:
        epsilon = np.zeros(1, dtype=np.float64)
        indices = np.zeros(batch_size, dtype=np.intc)
        loss = np.zeros(batch_size, dtype=np.float64)
        model = np.zeros(model_size, dtype=np.float64)
    else:
        epsilon = None
        indices = None
        loss = None
        model = None

    disp = MPI.DOUBLE.Get_size()
    epsilon_win = MPI.Win.Create(epsilon, disp, comm=comm)

    disp = MPI.INT.Get_size()
    indices_win = MPI.Win.Create(indices, disp, comm=comm)

    disp = MPI.DOUBLE.Get_size()
    loss_win = MPI.Win.Create(loss, disp, comm=comm)

    disp = MPI.DOUBLE.Get_size()
    model_win = MPI.Win.Create(model, disp, comm=comm)

    # time mesurements
    get_time_original(epsilon_win, indices_win, loss_win, model_win)

    model_win.Free()
    comm.Barrier()

    # -- Original test with alloc
    # init windows

    # epsilon - indices - loss - model
    # 3       - 2222222 - 1111 - 8888
    if comm.rank == 0:
        epsilon = 3*np.ones(1, dtype=np.float64)
        indices = 2*np.ones(batch_size, dtype=np.intc)
        loss = 1*np.ones(batch_size, dtype=np.float64)
        model = bytearray(model_size*[8])

    else:
        epsilon = np.zeros(1, dtype=np.float64)
        indices = np.zeros(batch_size, dtype=np.intc)
        loss = np.zeros(batch_size, dtype=np.float64)
        model = bytearray(model_size)


    epsilon_win = MPI.Win.Allocate(1*MPI.FLOAT.Get_size(), comm=comm)
    indices_win = MPI.Win.Allocate(batch_size*MPI.INT.Get_size(), comm=comm)
    loss_win = MPI.Win.Allocate(batch_size*MPI.FLOAT.Get_size(), comm=comm)
    model_win = MPI.Win.Allocate(model_size, comm=comm)

    if comm.rank == 0:
        model_win.Lock(0)
        model_win.Put(model,target_rank=0)
        model_win.Unlock(0)

        epsilon_win.Lock(0)
        epsilon_win.Put(epsilon, target_rank=0)
        epsilon_win.Unlock(0)

        indices_win.Lock(0)
        indices_win.Put(indices, target_rank=0)
        indices_win.Unlock(0)

        loss_win.Lock(0)
        loss_win.Put(loss, target_rank=0)
        loss_win.Unlock(0)

    # time mesurements
    get_time_original(epsilon_win, indices_win, loss_win, model_win,out_prefix="Allocated ")

    epsilon_win.Free()
    indices_win.Free()
    loss_win.Free()
    model_win.Free()

    comm.Barrier()

    epsilon_win = MPI.Win.Allocate_shared(1*MPI.FLOAT.Get_size(), comm=comm)
    indices_win = MPI.Win.Allocate_shared(batch_size*MPI.INT.Get_size(), comm=comm)
    loss_win = MPI.Win.Allocate_shared(batch_size*MPI.FLOAT.Get_size(), comm=comm)
    model_win = MPI.Win.Allocate_shared(model_size, comm=comm)

    if comm.rank == 0:
        model_win.Lock(0)
        model_win.Put(model,target_rank=0)
        model_win.Unlock(0)

        epsilon_win.Lock(0)
        epsilon_win.Put(epsilon, target_rank=0)
        epsilon_win.Unlock(0)

        indices_win.Lock(0)
        indices_win.Put(indices, target_rank=0)
        indices_win.Unlock(0)

        loss_win.Lock(0)
        loss_win.Put(loss, target_rank=0)
        loss_win.Unlock(0)

    # time mesurements
    get_time_original(epsilon_win, indices_win, loss_win, model_win,out_prefix="Allocated Shared")
    comm.Barrier()

    epsilon_win.Free()
    indices_win.Free()
    loss_win.Free()
    model_win.Free()


    # other benchs
    single_win_pack()
    single_win_pickle()
    print("--")
