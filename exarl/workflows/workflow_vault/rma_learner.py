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
from exarl.base.comm_base import ExaComm
from exarl.workflows.workflow_vault.async_learner import ASYNC
from exarl.network.data_structures import *
from exarl.utils.profile import PROFILE
from exarl.utils.globals import ExaGlobals

class RMA(ASYNC):
    """
    TODO: Write description / notes
    """

    def __init__(self, agent=None, env=None):
        super(RMA, self).__init__()
        self.debug('Creating RMA learner!')

        # self.model_size = 1024
        # self.train_ret_size = 1024
        # self.exp_data_size = 1024
        # ExaGlobals.lookup_params('poop')
        assert hasattr(agent, "rma_model"), "Agent must have rma_model to use rma"
        assert hasattr(agent, "rma_train_ret"), "Agent must have rma_train_ret to use rma"
        assert hasattr(agent, "rma_exp_data"), "Agent must have rma_exp_data to use rma"
        
        # JS: Model
        self.model_buff = ExaMPIBuffUnchecked(ExaComm.agent_comm, agent.rma_model, rank_mask=True, name="model")
        self.episode_const = ExaMPIConstant(ExaComm.agent_comm, True, int, name="episode")
        self.train_ret_buff = ExaMPIDistributedQueue(ExaComm.agent_comm, (0, agent.rma_train_ret), length=1, rank_mask=True, name="episode")

        # JS: Exps
        self.rma_exp_data = [ExaComm.agent_comm.rank, agent.rma_exp_data, 0, False, 0.0]
        self.data_queue = ExaMPIDistributedQueue(ExaComm.agent_comm, self.rma_exp_data, rank_mask=True, name="experience")
        
        # JS: Next episode
        if ExaComm.is_learner():
            self.next_episode_const = ExaMPIConstant(ExaComm.learner_comm, True, int, inc=self.batch_episode_frequency, name="next episode")

        # JS: This is to reduce times we update
        self.last_model_sent = None
        self.last_model_recv = None
        self.last_episode_per_rank = None
        if ExaComm.is_learner():
            self.last_episode_per_rank = [None] * ExaComm.agent_comm.size

    def send_model(self, workflow, episode, train_ret, dst):
        """
        This function sends the model from the learner to
        other agents using MPI_Send.

        Parameters
        ----------
        workflow : ExaWorkflow
            This contains the agent and env

        episode : int
            The current episode corresponding to the model generation

        train_return : list
            This is what comes out of the learner calling train to be sent back
            to the actor (i.e. indices and losses).

        dst : int
            This is the destination rank given by the agent communicator
        """
        model = workflow.agent.get_weights()
        # JS: The order here matters
        # We are sending the a modified train ret last
        # so we can spin on it when we really want a new model

        # JS: Send the new episode count to agent
        if self.last_episode_per_rank[dst] != episode:
            self.episode_const.put(episode, rank=dst)
            self.last_episode_per_rank[dst] = episode
        # JS: Push data to our local buffer let agents pull from us
        if self.last_model_sent != self.model_count:
            self.model_buff.push(model)
            self.last_model_sent = self.model_count
        # JS: We will use train_ret for blocking...
        lost = 1
        while lost:
                capacity, lost = self.train_ret_buff.push((self.model_count, train_ret), dst)

    def recv_model(self):
        """
        This function receives the model from the learner
        using MPI_Recv.

        Returns
        ----------
        list :
            This list should contain the episode, model weights,
            and the train return (indices and losses if turned on)
        """
        train_ret = None
        while train_ret is None:
            train_ret = self.train_ret_buff.pop(ExaComm.agent_comm.rank)
        self.last_model_recv, train_ret = train_ret
        
        episode = self.episode_const.get()
        model = self.model_buff.pop(0)

        ret = [episode, model, []]
        if train_ret is not None:
            ret[-1].append(train_ret)     
        return ret

    def send_batch(self, batch_data, policy_type, done, episode_reward):
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

        done : bool
            Indicates if the episode is competed

        episode_reward : float
            The total reward from the last episode.  If the episode in not done, it
            will be the current total reward.
        """
        self.data_queue.push([ExaComm.agent_comm.rank, batch_data, policy_type, done, episode_reward], 0)

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
        ret = None
        while ret is None:
            ret = self.data_queue.pop(0)
        return ret

    def init_learner(self, workflow):
        """
        This function is used to initialize the model on every agent.
        We are assuming a single learner starting the range from 1.

        Parameters
        ----------
        workflow : ExaWorkflow
            This contains the agent and env
        """
        if ExaComm.is_learner() and ExaComm.learner_comm.rank == 0:
            self.model_buff.push(workflow.agent.get_weights())
            for dst in range(1, ExaComm.agent_comm.size):
                self.episode_const.put(self.next_episode, rank=dst)
                self.train_ret_buff.push((self.model_count, None), dst)
                self.episode_per_rank[dst] = self.next_episode
                self.next_episode += self.batch_episode_frequency
                self.alive += 1
            self.next_episode_const.put(self.next_episode, rank=0)
        
        # JS: Try to barrier to ensure weights are out
        ExaComm.agent_comm.barrier()

    def inc_episode(self):
        """
        We abstract this increment for future workflows (RMA) which
        need to synchronize this value.
        """
        self.next_episode = self.next_episode_const.inc(rank=0)
        return self.next_episode

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
        convergence = -1
        nepisodes = self.episode_round(workflow)
        self.init_learner(workflow)
        last_print = 0
        # These are the loops used to keep everyone running
        if ExaComm.is_learner():
            while self.alive > 0 and self.done_episode < nepisodes:
                do_convergence_check = self.learner(workflow, nepisodes, 1)
                if do_convergence_check:
                    convergence = self.check_convergence()
                if self.done_episode % 10 == 0 and self.done_episode != last_print:
                    print("Learner:", self.done_episode, nepisodes, do_convergence_check, convergence, flush=True)
                    last_print = self.done_episode
        else:
            keep_running = True
            while keep_running:
                keep_running = self.actor(workflow, nepisodes)
                # print("Actor:", keep_running, flush=True)