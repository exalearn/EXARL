Running EXARL using MPI
***********************

Existing environment can be paired with an available agent.

The following script is provided for convenience: ``ExaRL/driver/driver.py``

.. code-block:: python

    from mpi4py import MPI
    import utils.analyze_reward as ar
    import time
    import exarl as erl
    import mpi4py.rc
    mpi4py.rc.threads = False
    mpi4py.rc.recv_mprobe = False

    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get run parameters using CANDLE
    # run_params = initialize_parameters()

    # Create learner object and run
    exa_learner = erl.ExaLearner(comm)

    # Run the learner, measure time
    start = time.time()
    exa_learner.run()
    elapse = time.time() - start

    # Compute and print average time
    max_elapse = comm.reduce(elapse, op=MPI.MAX, root=0)
    elapse = comm.reduce(elapse, op=MPI.SUM, root=0)

    if rank == 0:
        print("Average elapsed time = ", elapse / size)
        print("Maximum elapsed time = ", max_elapse)
        # Save rewards vs. episodes plot
        ar.save_reward_plot()

Write your own script or modify the above as needed.

Run the following command:

.. code-block:: bash

    mpiexec -np <num_parent_processes> python driver/driver.py --<run_params>=<param_value>

If running a multi-process environment or agent, the communicators are available in ``exarl/mpi_settings.py``. 

E.g.:-

.. code-block:: python

    import exarl.mpi_settings as mpi_settings
    self.env_comm = mpi_settings.env_comm
    self.agent_comm = mpi_settings.agent_comm
