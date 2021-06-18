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
import numpy as np
import exarl as erl
from exarl.utils.introspect import *
from exarl.utils.profile import *
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
from exarl.base.comm_base import ExaComm
from exarl.network.data_structures import *
from exarl.network.simple_comm import ExaSimple
MPI = ExaSimple.MPI

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA(erl.ExaWorkflow):
    def __init__(self):
        print("Creating ML_RMA workflow", flush=True)
        data_exchange_constructors = {
            "queue_distribute": ExaMPIDistributedQueue,
            "stack_distribute": ExaMPIDistributedStack,
            "queue_central": ExaMPICentralizedQueue,
            "stack_central": ExaMPICentralizedStack
        }

        self.de = cd.lookup_params('data_structure', default='queue_distribute')
        self.de_constr = data_exchange_constructors[self.de]
        self.de_length = cd.lookup_params('data_structure_length', default=32)
        self.de_lag = cd.lookup_params('max_model_lag')
        logger.info('Creating RMA workflow with ', self.de, "length", self.de_length, "lag", self.de_lag)

        self.de = cd.lookup_params('loss_data_structure', default='queue_distribute')
        self.de_constr_loss = data_exchange_constructors[self.de]

        priority_scale = cd.lookup_params('priority_scale')
        self.use_priority_replay = (priority_scale is not None and priority_scale > 0)

    @PROFILE
    def run(self, workflow):
        num_learners = ExaComm.num_learners
        self.de_attempts = 100 if num_learners > 1 else None

        # MPI communicators
        agent_comm = ExaComm.agent_comm.raw()
        env_comm = ExaComm.env_comm.raw()
        if ExaComm.is_learner():
            learner_comm = ExaComm.learner_comm.raw()

        # Allocate RMA windows
        if ExaComm.is_agent():
            # Get size of episode counter
            disp = MPI.DOUBLE.Get_size()
            episode_data = None
            if ExaComm.is_learner() and learner_comm.rank == 0:
                episode_data = np.zeros(1, dtype=np.float64)
            # Create episode window (attach instead of allocate for zero initialization)
            episode_win = MPI.Win.Create(episode_data, disp, comm=agent_comm)

            # Get size of epsilon
            disp = MPI.DOUBLE.Get_size()
            epsilon = None
            if ExaComm.is_learner() and learner_comm.rank == 0:
                epsilon = np.zeros(1, dtype=np.float64)
            # Create epsilon window
            epsilon_win = MPI.Win.Create(epsilon, disp, comm=agent_comm)

            if self.use_priority_replay:
                # Create windows for priority replay (loss and indicies)
                indices_for_size = -1 * np.ones(workflow.agent.batch_size, dtype=np.intc)
                loss_for_size = np.zeros(workflow.agent.batch_size, dtype=np.float64)
                indicies_and_loss_for_size = (indices_for_size, loss_for_size)
                data_exchange_loss = self.de_constr_loss(ExaComm.agent_comm, rank=ExaComm.learner_rank(),
                                                         data=indicies_and_loss_for_size, length=num_learners, max_model_lag=None)

            # Get serialized target weights size
            target_weights = (workflow.agent.get_weights(), np.int64(0))
            serial_target_weights = MPI.pickle.dumps(target_weights)
            serial_target_weights_size = len(serial_target_weights)
            target_weights_size = 0
            if ExaComm.is_learner() and learner_comm.rank == 0:
                target_weights_size = serial_target_weights_size
            # Allocate model window
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)

            # Get serialized batch data size
            learner_counter = np.int64(0)
            agent_batch = (next(workflow.agent.generate_data()), learner_counter)
            data_exchange = self.de_constr(ExaComm.agent_comm, rank=ExaComm.learner_rank(),
                                           data=agent_batch, length=self.de_length, max_model_lag=self.de_lag)
            # This is a data/flag that lets us know we have data
            agent_data = None

        if ExaComm.is_learner() and learner_comm.rank == 0:
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if ExaComm.is_learner():
            # Initialize batch data buffer
            episode_count_learner = np.zeros(1, dtype=np.float64)
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)
            # Initialize epsilon
            if learner_comm.rank == 0:
                epsilon_win.Lock(0)
                epsilon_win.Put(epsilon, target_rank=0)
                epsilon_win.Flush(0)
                epsilon_win.Unlock(0)

            while True:
                if learner_comm.rank == 0:
                    # Check episode counter
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to fetch episode count
                    episode_win.Get_accumulate(np.ones(1, dtype=np.float64), episode_count_learner, target_rank=0, op=MPI.NO_OP)
                    episode_win.Flush(0)
                    episode_win.Unlock(0)

                if num_learners > 1:
                    episode_count_learner = learner_comm.bcast(episode_count_learner, root=0)

                if episode_count_learner >= workflow.nepisodes:
                    break

                if agent_data is None:
                    ib.startTrace("RMA_Data_Exchange_Pop", 0)
                    agent_data, actor_idx, actor_counter = data_exchange.get_data(
                        learner_counter, learner_comm.size, agent_comm.size, attempts=self.de_attempts)
                    ib.stopTrace()
                    ib.simpleTrace("RMA_Learner_Get_Data", actor_idx, actor_counter, learner_counter - actor_counter, 0)

                # Do an allreduce to check if all learners have data
                if num_learners > 1:
                    process_has_data = 0 if agent_data is None else 1
                    sum_process_has_data = learner_comm.allreduce(process_has_data, op=MPI.SUM)
                    if sum_process_has_data < learner_comm.size:
                        continue

                # Train & Target train
                train_return = workflow.agent.train(agent_data)
                ib.update("RMA_Learner_Train", 1)

                if self.use_priority_replay and train_return is not None:
                    if train_return[0][0] != -1:
                        indices, loss = train_return
                        indices = np.array(indices, dtype=np.intc)
                        loss = np.array(loss, dtype=np.float64)
                        data_exchange_loss.push((indices, loss), rank=actor_idx)

                # Update flag/counters after train
                learner_counter += 1
                agent_data = None

                # Share new model weights
                if ExaComm.is_learner() and learner_comm.rank == 0:
                    workflow.agent.target_train()
                    ib.update("RMA_Learner_Target_Train", 1)

                    target_weights = (workflow.agent.get_weights(), learner_counter)
                    serial_target_weights = MPI.pickle.dumps(target_weights)
                    model_win.Lock(0)
                    model_win.Put(serial_target_weights, target_rank=0)
                    model_win.Unlock(0)

            logger.info('Learner exit on rank_episode: {}_{}'.format(agent_comm.rank, episode_data))

        # Actors
        else:
            local_actor_episode_counter = 0
            if ExaComm.env_comm.rank == 0:
                # Logging files
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' + filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            episode_count_actor = np.zeros(1, dtype=np.float64)
            one = np.ones(1, dtype=np.float64)
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)

            while True:
                if ExaComm.env_comm.rank == 0:
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_count_actor, target_rank=0)
                    episode_win.Flush(0)
                    episode_win.Unlock(0)

                # Broadcast episode count to all procs in env_comm
                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)

                # Check exit condition
                if episode_count_actor >= workflow.nepisodes:
                    break
                logger.info('Rank[{}] - working on episode: {}'.format(agent_comm.rank, episode_count_actor))

                # Episode initialization
                workflow.env.seed(0)
                current_state = workflow.env.reset()
                total_rewards = 0
                steps = 0
                action = 0
                done = False
                local_actor_episode_counter += 1

                while done != True:
                    # Update model weight
                    if ExaComm.env_comm.rank == 0:
                        buff = bytearray(serial_target_weights_size)
                        model_win.Lock(0)
                        model_win.Get(buff, target=0, target_rank=0)
                        model_win.Flush(0)
                        model_win.Unlock(0)
                        target_weights, learner_counter = MPI.pickle.loads(buff)
                        workflow.agent.set_weights(target_weights)
                        ib.simpleTrace("RMA_Actor_Get_Model", local_actor_episode_counter, learner_counter, 0, 0)

                        # Get epsilon
                        local_epsilon = np.array(workflow.agent.epsilon)
                        epsilon_win.Lock(0)
                        epsilon_win.Get_accumulate(local_epsilon, epsilon, target_rank=0, op=MPI.MIN)
                        epsilon_win.Flush(0)
                        epsilon_win.Unlock(0)
                        # Update the agent epsilon
                        workflow.agent.epsilon = min(epsilon, local_epsilon)

                        # Get the loss and indicies
                        if self.use_priority_replay:
                            loss_data = data_exchange_loss.pop(ExaComm.agent_comm.rank)
                            if loss_data is not None:
                                loss, indices = loss_data
                                workflow.agent.set_priorities(indices, loss)

                        # Inference action
                        action, policy_type = workflow.agent.action(current_state)

                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        ib.update("RMA_Env_Inference", 1)

                    # Broadcast episode count to all procs in env_comm
                    action = env_comm.bcast(action, root=0)

                    # Environment step
                    ib.startTrace("step", 0)
                    next_state, reward, done, _ = workflow.env.step(action)
                    ib.stopTrace()
                    ib.update("RMA_Env_Step", 1)
                    ib.simpleTrace("RMA_Reward", steps, 1 if done else 0, local_actor_episode_counter, reward)

                    steps += 1
                    # Broadcast done. Somewhat redundant.  Note rank 0 must be the one to signal done.
                    done = done or (steps >= workflow.nsteps)
                    done = env_comm.bcast(done, root=0)

                    if ExaComm.env_comm.rank == 0:
                        # Save memory
                        total_rewards += reward
                        memory = (current_state, action, reward, next_state, done, total_rewards)
                        workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                        batch_data = (next(workflow.agent.generate_data()), learner_counter)
                        ib.update("RMA_Env_Generate_Data", 1)

                        # Only push data if data is valid
                        # batch_data = [[batch_data], learner_counter]
                        # batch_data[0][-1] = data_valid flag
                        if batch_data[0][-1] == True:
                            ib.startTrace("RMA_Data_Exchange_Push", 0)
                            # Write to data window
                            capacity, lost = data_exchange.push(batch_data)
                            ib.stopTrace()
                            ib.simpleTrace("RMA_Actor_Put_Data", capacity, lost, 0, 0)

                        # Log state, action, reward, ...
                        ib.simpleTrace("RMA_Total_Reward", steps, 1 if done else 0, local_actor_episode_counter, total_rewards)
                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_rewards,
                                               done, local_actor_episode_counter, steps, policy_type, workflow.agent.epsilon])
                        train_file.flush()

        if ExaComm.is_agent():
            model_win.Free()
