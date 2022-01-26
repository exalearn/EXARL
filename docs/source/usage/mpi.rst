MPI Structure in EXARL
======================


Backend
-------
**mpi4py** ``https://github.com/mpi4py/mpi4py`` is used to connect to an underlying MPI implementation.
mpi4py provides a python interface to MPI libraries like MPICH or OpenMPI.


Relevant classes
----------------
**ExaComm** ``exarl/base/comm_base.py`` is the abstract base class for communication. ExaComm defines an interface with send, receive, barrier, broadcast and other standard collective communication functions.
It also has global_comm, agent_comm, env_comm, and learner_comm and num_learners member variables

**ExaSimple** ``exarl/network/simple_comm.py`` inherits from ExaComm. It implements the ExaComm interface by calling the corresponding mpi4py functions  like send, recv, bcast, barrier, reduce, allreduce, gather, etc.

Communicator structure
----------------------
There are three types of communicators in EXARL. They are named agent, environment, and learner.

The way that MPI ranks are divided into communicators depends on the type of workflow and the specific settings for the workflow. In the case of the async learner, only one MPI rank will be a learner, and all ranks join environment and agent communicators.

Currently the async learner does not support using process_per_env > 1. In general, though, EXARL requires ``number of ranks - 1`` to be a multiple of process_per_env.

If process_per_env > 1 then  non-learner tasks are put into groups of size process_per_env and form their own environment communicator. This is achieved using MPI Split.

Initialization
--------------
ExaSimple is constructed when we initialize ExaLearner  ``exarl/base/learner_base.py``. This causes the assignment of the ranks to the various communicators as described above.

The ExaSimple constructor takes three arguments: A global MPI communicator (default is MPI.COMM_WORLD), process_per_env and learner_procs.  The values of learners_procs and process_per_env come from come from the configuration file ``exarl/config/workflow_cfg/async.json`` or they can be passed on the command line.
