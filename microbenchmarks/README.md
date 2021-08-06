# Microbenchmarks
This folder contains RMA microbenchmarks used for evaluate the performance of different approaches.

## rma_get.py
Based on the RMA workflow, the actor need to get data from four windows before each environment step. "RMA get" benchmark allows to compare 4 different approaches of getting data ("MPI.GET()") from RMA windows:
- Get data sequentially from 4 windows initialized with "MPI.Win.Create()"
- Get data sequentially from 4 windows initialized with "MPI.Win.Allocate()" (memory allocate with MPI_Mem_Alloc())
- Get data from a single buffer window containing the data of all four windows (use pickle for serializing data)
- Get data from a single buffer window containing the data of all four windows (use unpack)

## rma_put.py 
Identical with "rma_get.py" but measuring the "MPI.PUT()" operation.

## run_lock_test.py 
Allows to compate the execution time of concurent "MPI.GET()" from a window with :
- EXCLUSIVE LOCKS 
- SHARED LOCKS
