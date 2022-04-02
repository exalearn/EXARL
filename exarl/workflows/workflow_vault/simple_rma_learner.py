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
from exarl.network.data_structures import *

logger = log.setup_logger(__name__, cd.lookup_params('log_level', [3, 3]))

class SIMPLE_RMA(SIMPLE):
    """
    This class builds ontop of the simple learner to support processing
    environments in parallel.  This is achieved by having separate leaners
    and actors.  The communication is performed by MPI sends/recvs.

    We are currently supporting single learner thus we set block_size = 2
    for off-policy learning.

    This class assumes a single learner. 
    """

    def __init__(self):
        super(SIMPLE_RMA, self).__init__()
        print('Creating SIMPLE RMA learner!', flush=True)

        if self.block_size == 1:
            self.block_size = 2

        data_exchange_constructors = {
            "buff_unchecked": ExaMPIBuffUnchecked,
            "buff_checked": ExaMPIBuffChecked,
            "queue_distribute": ExaMPIDistributedQueue,
            "stack_distribute": ExaMPIDistributedStack
            # "queue_central": ExaMPICentralizedQueue,
            # "stack_central": ExaMPICentralizedStack
        }

        # target weights - This should be an unchecked buffer that will always succeed a pop since weights need to be shared with everyone
        self.target_weight_data_structure = data_exchange_constructors["buff_unchecked"]
        
        self.batch_data_structure = data_exchange_constructors[cd.lookup_params('data_structure', default='queue_distribute')]
        self.de_length = cd.lookup_params('data_structure_length', default=32)
        self.de_lag = None  # cd.lookup_params('max_model_lag')

        self.de = cd.lookup_params('loss_data_structure', default='buff_unchecked')
        self.ind_loss_data_structure = data_exchange_constructors[cd.lookup_params('loss_data_structure', default='queue_distribute')]

        if ExaComm.is_actor():
            self.model_count = -1

    def block_constant(self, old, fn):
        ret = old
        while ret == old:
            ret = fn()
        return ret

    def block_model_pop(self):
        old_model_count = self.model_count
        while old_model_count == self.model_count:
            epsilon, weights, self.model_count = self.model_buff.pop(0)
        return epsilon, weights, self.model_count


    def block_pop(self, fn):
        ret = None
        while ret is None:
            ret = fn()
        return ret

    def block_push(self, fn):
        lost = 1
        while lost > 0:
            capacity, lost = fn()

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
        if dst > 0:
            self.episode_buff.put(episode, dst)
        self.model_buff.push((workflow.agent.epsilon, workflow.agent.get_weights(), self.model_count), 0)

        # if train_ret is not None:
        #     self.block_push(lambda : self.ind_loss_buff.push(train_ret, dst))

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
        
        # episode = self.episode_buff.get(ExaComm.agent_comm.rank)
        # epsilon, weights, self.model_count = self.model_buff.pop(0)
        # loss = self.ind_loss_buff.pop(ExaComm.agent_comm.rank)

        episode = self.episode_buff.get(ExaComm.agent_comm.rank)
        epsilon, weights, self.model_count = self.block_model_pop()
        
        loss = None
        # if self.model_count > 0:
        #     loss = self.block_pop(lambda : self.ind_loss_buff.pop(ExaComm.agent_comm.rank))

        if loss:
            print("LOSS:", loss)
            return episode, epsilon, weights, loss
        return episode, epsilon, weights

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
        data = ([ExaComm.agent_comm.rank, batch_data, policy_type, done], self.model_count)
        lost = 1
        capacity, lost = self.batch_buff.push(data, ExaComm.agent_comm.rank)

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
        # rank, batch_data, policy_type, done = 
        data = None
        while data is None:
            data, _, _ = self.batch_buff.get_data(self.model_count, 1, ExaComm.agent_comm.size)
            # print("Data", data)
        return data
    
    def init_learner(self, workflow):
        """
        This function is used to initialize the model on every agent.
        We are assuming a single learner starting the range from 1.

        Parameters
        ----------
        workflow : ExaWorkflow
            This contains the agent and env
        """
        data = (workflow.agent.epsilon, workflow.agent.get_weights(), 0)
        self.model_buff.push(data, 0)

        for dst in range(1, ExaComm.agent_comm.size):
            self.episode_buff.put(self.next_episode, dst)
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

        if ExaComm.is_agent():
            self.episode_buff = ExaMPIConstant(ExaComm.agent_comm,
                                                ExaComm.is_agent(),
                                                np.int64, 
                                                name="Episode_Const")

            target_weights = ([np.float64(workflow.agent.epsilon), workflow.agent.get_weights(), np.int64(0)])
            self.model_buff = self.target_weight_data_structure(ExaComm.agent_comm, 
                                                                target_weights,
                                                                rank_mask=ExaComm.is_agent(),  
                                                                length=self.de_length, 
                                                                max_model_lag=self.de_lag, 
                                                                failPush=False, 
                                                                name="Model_Buffer")

            agent_batch = ([np.int64(ExaComm.agent_comm.rank), next(workflow.agent.generate_data()), np.int64(0), np.bool(False)], np.int64(0))
            self.batch_buff = self.batch_data_structure(ExaComm.agent_comm, 
                                                            agent_batch, 
                                                            rank_mask=ExaComm.is_agent(),
                                                            length=self.de_length, 
                                                            max_model_lag=self.de_lag,
                                                            failPush=True, 
                                                            name="Data_Exchange")

            indices_for_size = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
            loss_for_size = np.zeros(workflow.agent.batch_size, dtype=np.float64)
            indices_and_loss_for_size = (indices_for_size, loss_for_size)
            self.ind_loss_buff = self.ind_loss_data_structure(ExaComm.agent_comm, 
                                                            indices_and_loss_for_size, 
                                                            rank_mask=ExaComm.is_agent(),
                                                            length=self.de_length, 
                                                            max_model_lag=self.de_lag,
                                                            failPush=True,
                                                            name="Loss_Buffer")

        nepisodes = self.episode_round(workflow)
        # These are the loops used to keep everyone running
        if ExaComm.is_learner():
            self.init_learner(workflow)
            while self.done_episode < nepisodes:
                self.learner(workflow, nepisodes, 1)
                print(self.done_episode, nepisodes, flush=True)
            self.send_model(workflow, self.done_episode, None, 1)
        else:
            keep_running = True
            while keep_running:
                keep_running = self.actor(workflow, nepisodes)
        print("DONE", ExaComm.global_comm.rank, flush=True)

        