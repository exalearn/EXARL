import os
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals

class Affinity:
    """
    This class is used to help place learners, actors, and envs on
    ranks across nodes.  It is currently built to support slurm
    schedulers by reading the environment variable SLURMD_NODENAME
    to determine what rank is on which node.  For other schedulers
    the get_node method should be changed or SLURMD_NODENAME should
    be set to a node name by a launcher script.

    Also this class will call an allreduce to get the node map in
    the constructure.  Be careful that all ranks in the global
    comm call the constructor or the program will hang.

    Attributes
    ----------
    comm : MPI Comm
        Should be MPI.COMM_WORLD
    size : int
        Size of MPI.COMM_WORLD
    procs_per_env : int
        Number of processes per environment (sub-comm)
    num_learners : int
        Number of learners (multi-learner)

    """

    def __init__(self, comm, procs_per_env, num_learners):
        """
        Parameters
        ----------
        comm : MPI Comm
            Should be MPI.COMM_WORLD
        procs_per_env : int
            Number of processes per environment (sub-comm)
        num_learners : int
            Number of learners (multi-learner)
        """
        self.procs_per_env = procs_per_env
        self.num_learners = num_learners
        self.comm = comm
        self.size = comm.Get_size()
        self._nodes_map = self._get_nodes_map()
        self._color_map = None

    def _get_node(self):
        """
        This is slurm specific.  This must be overridden for other
        resource managers.

        Returns
        -------
        int
            Current node id
        """
        return os.getenv('SLURMD_NODENAME', "node_0")

    def _get_nodes_map(self):
        """
        This function gets the node name of each
        node across all ranks.
        """
        my_node = self._get_node()
        return self.comm.allgather(my_node)

    def make_map(self, rank_and_color):
        """
        This turns list of tuples into a rank map.
        Each tuple is of form (rank, color).
        We return -1 for MPI.UNDEFINE.
        
        Parameters
        ----------
        rank_and_color : list
            List of tuples containing (rank, color)
        
        Returns
        -------
        list
            Integers for the colors of a communicator split
        """
        color_map = [-1] * self.size
        for rank, color in rank_and_color:
            color_map[rank] = color
        return color_map

    def _one_learner_per_node(self):
        """
        This function creates an affinity that puts a single learner on each node.
        This is to utilize systems with limited number of GPUs.  The remaining
        ranks are round robin as envs.  The non-learning agents are the min ranks of
        the env comms (i.e. rank 0 after the comm is created).  This method does not
        account for more than 1 learner per node).  If we are not able to determine
        the number of nodes using get_node method, it will look like there is only
        one node.  In this case, either manually set SLURMD_NODENAME for each rank in
        a launcher script, or modify the get_node method.

        Returns
        -------
        Tuple
            A list for learner, agent, and env colors
        """
        my_node = self._get_node()
        self._nodes_map = self.comm.allgather(my_node)
        nodes_set = set(self._nodes_map)

        error = """Using the affinity class assumes there is only one GPU per node.
            The number of learners is set to more than the number of nodes.
            If you are sure you want this behaivor, please ensure your agent/model
            can support this behavior and run without affinity.  Note: the affinity
            class must be able to determine the number of nodes.  For slurm, this is
            determined by the environment variable SLURMD_NODENAME.
            """
        assert len(nodes_set) <= self.num_learners, error
        
        # JS: Get a learner for each node
        learners = []
        for n in nodes_set:
            # JS: Give me all the ranks on node n
            temp = [rank for rank, node in enumerate(self._nodes_map) if node == n]
            # JS: Take only the first one
            learners.append((min(temp), 0))
        # JS: we only need num_learners
        learners = learners[:self.num_learners]
        
        # JS: Round robing the rest as envs
        remainder = set(range(self.size)) ^ set([rank for rank, _ in learners])
        envs = [(rank, int(i/self.procs_per_env)) for i, rank in enumerate(remainder)]
        envs.extend(learners)
        
        # JS: Agents are learners plus first of the envs
        agents = learners[:]
        env_set = {color for _, color in envs}
        for color in env_set:
            temp = [rank for rank, c in envs if c == color]
            agents.append((min(temp), 0))
        return learners, agents, envs

    def _default(self):
        """
        This method is the default layout.  All learners are placed in contiguous
        ranks starting at 0.  The remaining ranks are set up as envs.  The
        non-learner ranks are the first rank of the env comm.
        """
        learners = [(x, 0) for x in range(self.size) if x < self.num_learners]
        agents = [(x, 0) for x in range(self.size) if x < self.num_learners 
                  or ((x - self.num_learners) % self.procs_per_env == 0)]
        envs = []
        print("Default:", self.size)
        for x in range(self.size):
            print(x, "vs", self.num_learners)
            if x >= self.num_learners:
                envs.append((x, (int((x - self.num_learners) / self.procs_per_env)) + 1))
                print(envs[-1])
        return learners, agents, envs

    def _single_env(self):
        """
        This layout is for the sync workflow where we have only one learner/agent and the remaining
        ranks are the environment.
        """
        assert self.num_learners == 1
        learners = [(0, 0)]
        agents = [(0, 0)]
        envs = [(x, 0) for x in range(self.size)]
        return learners, agents, envs

    def get_map(self):
        """
        This is a wrapper around the different layouts.
        It will return maps that the comm split can use.
        """
        if self._color_map is None:
            arg = True if ExaGlobals.lookup_params('affinity') in ["true", "True", 1] else False
            if self.size == self.procs_per_env:
                ret = self._single_env()
            elif arg and self.num_learners > 1:
                ret = self._one_learner_per_node()
            else:
                ret = self._default()
            self._color_map = [self.make_map(x) for x in ret]
        return self._color_map
    
    def learners_on_node(self):
        """
        This function return the number of learners on ranks node.
        """
        my_node = self._get_node()
        learners = self.get_map()[0]
        nodes = [self._nodes_map[x] for x in learners]
        return len([x for x in nodes if x == my_node])
