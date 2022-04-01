# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import time
import csv
import exarl as erl
from exarl.utils.introspect import ib
from exarl.utils.profile import *
from exarl.utils import log
import exarl.utils.candleDriver as cd
from exarl.base.comm_base import ExaComm
from exarl.network.typing import TypeUtils
from exarl.workflows.workflow_vault.simple_learner import SIMPLE

logger = log.setup_logger(__name__, cd.lookup_params('log_level', [3, 3]))

class SIMPLE_ASYNC(SIMPLE):
    """
    This class builds ontop of the simple learner to support processing
    environments in parallel.  This is achieved by having separate leaners
    and actors.  The communication is performed by MPI sends/recvs.

    We are currently supporting single learner thus we set block_size = 2
    for off-policy learning.

    This class assumes a single learner. 
    """

    def __init__(self):
        super(SIMPLE_ASYNC, self).__init__()
        print('Creating SIMPLE ASYNC learner!', flush=True)

        if self.block_size == 1:
            self.block_size = 2
        # print("priority_scale", cd.lookup_params('priority_scale'), "batch_frequency", cd.lookup_params('batch_frequency'), "block_size", self.block_size, flush=True)

    def send_model(self, workflow, episode, train_ret, dst):
        """
        This function sends the model from the learner to
        other agents using MPI_Send.

        Parameters
        ----------
        workflow : ExaWorkflow
            This contains the agent and env
        
        episode : int
            The current episode curresponding to the model generation

        train_return : list
            This is what comes out of the learner calling train to be sent back
            to the actor (i.e. indices and losses).

        dst : int
            This is the destination rank given by the agent communicator
        """
        data = [episode, workflow.agent.epsilon, workflow.agent.get_weights()]
        if train_ret is not None:
            data.append(train_ret)
        ExaComm.agent_comm.send(data, dst)

    def recv_model(self):
        """
        This function receives the model from the learner
        using MPI_Recv.

        Returns
        ----------
        list :
            This list should contain the episode, epsilon, model weights,
            and the train return (indices and losses if turned on)
        """
        ret = ExaComm.agent_comm.recv(None, source=0)
        return ret

    def send_batch(self, batch_data, policy_type, done):
        """
        This function is used to send batches of data from the actor to the
        learner using MPI_Send.

        Parameters
        ----------
        batch_data : list
            This is a list of experiences generate by the actor to send to
            the learner.

        policy_type : int
            This is the policy given by the actor performing inference to get an action
        """
        ExaComm.agent_comm.send([ExaComm.agent_comm.rank, batch_data, policy_type, done], 0)

    def recv_batch(self):
        """
        This function receives batches of experiences sent from an actor
        using MPI_Recv.  

        Returns
        -------
        list :
            This list should contain the rank, batched data, policy type, and done flag.
            The done flag indicates if the episode the actor was working on finished.
        """
        return ExaComm.agent_comm.recv(None)
    
    def init_learner(self, workflow):
        """
        This function is used to initialize the model on every agent.
        We are assuming a single learner starting the range from 1.

        Parameters
        ----------
        workflow : ExaWorkflow
            This contains the agent and env
        """
        for dst in range(1, ExaComm.agent_comm.size):
            self.send_model(workflow, self.next_episode, None, dst)
            self.episode_per_rank[dst] = self.next_episode
            self.next_episode += 1

    @PROFILE
    def run(self, workflow):
        """
        This function is responsible for calling the appropriate initialization
        and looping over the actor/learner functions.

        Parameters
        ----------
        workflow : ExaWorkflow
            This contains the agent and env
        """
        nepisodes = self.episode_round(workflow)                
        
        # These are the loops used to keep everyone running
        if ExaComm.is_learner():
            self.init_learner(workflow)
            while self.done_episode < nepisodes:
                self.learner(workflow, nepisodes, 1)
        else:
            keep_running = True
            while keep_running:
                keep_running = self.actor(workflow, nepisodes)

        