import time
import csv
import numpy as np
import exarl as erl
from utils.introspect import ib
from utils.profile import *
import utils.log as log
import utils.candleDriver as cd
from exarl.comm_base import ExaComm
from mpi4py import MPI
from utils.trace_win import Trace_Win
from exarl.dataset_base import BufferDataset

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA_ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating RMA async workflow...')

    def run(self, workflow):
        # MPI communicators
        agent_comm = ExaComm.agent_comm.raw()
        env_comm = ExaComm.env_comm.raw()
        epTrace = Trace_Win(name="episodes_tr", comm=ExaComm.agent_comm, arrayType=np.int64)
        reTrace = Trace_Win(name="reward_tr", comm=ExaComm.agent_comm, arrayType=np.float64)
        moTrace = Trace_Win(name="model_tr", comm=ExaComm.agent_comm, arrayType=np.int64)

        if ExaComm.is_learner():
            workflow.agent.set_learner()

        # Allocate RMA windows
        if ExaComm.is_agent():
            # Get size of episode counter
            disp = MPI.DOUBLE.Get_size()
            episode_data = None
            if ExaComm.is_learner():
                episode_data = np.zeros(1, dtype=np.float64)
            # Create episode window (attach instead of allocate for zero initialization)
            episode_win = MPI.Win.Create(episode_data, disp, comm=agent_comm)

            # Get size of epsilon
            disp = MPI.DOUBLE.Get_size()
            epsilon = None
            if ExaComm.is_learner():
                epsilon = np.zeros(1, dtype=np.float64)
            # Create epsilon window
            epsilon_win = MPI.Win.Create(epsilon, disp, comm=agent_comm)

            # Get serialized target weights size
            target_weights = workflow.agent.get_weights()
            serial_target_weights = MPI.pickle.dumps(target_weights)
            serial_target_weights_size = len(serial_target_weights)
            target_weights_size = 0
            if ExaComm.is_learner():
                target_weights_size = serial_target_weights_size
            # Allocate model window
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)

            # Get serialized batch data size
            batch_states = np.zeros((workflow.agent.batch_size, 1, workflow.env.observation_space.shape[0])).astype("float64")
            batch_actions = np.zeros((workflow.agent.batch_size, workflow.env.action_space.n)).astype("float64")
            batch_reward = np.zeros(workflow.agent.batch_size, dtype=np.float64)
            batch_next_states = np.zeros((workflow.agent.batch_size, 1, workflow.env.observation_space.shape[0])).astype("float64")
            batch_done = np.zeros(workflow.agent.batch_size, dtype=np.bool)
            batch_total_reward = np.zeros(workflow.agent.batch_size, dtype=np.float64)

            agent_batch = [batch_states, batch_actions, batch_reward, batch_next_states, batch_done, batch_total_reward]
            # agent_batch = next(workflow.agent.generate_data())
            serial_agent_batch = (MPI.pickle.dumps(agent_batch))
            serial_agent_batch_size = len(serial_agent_batch)
            nserial_agent_batch = 0
            if ExaComm.is_actor():
                nserial_agent_batch = serial_agent_batch_size
            # Allocate data window
            data_win = MPI.Win.Allocate(nserial_agent_batch, 1, comm=agent_comm)

        if ExaComm.is_learner():
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if ExaComm.is_learner():
            # Initialize batch data buffer
            data_buffer = bytearray(serial_agent_batch_size)
            episode_count_learner = np.zeros(1, dtype=np.float64)
            epsilon = np.array(workflow.agent.epsilon, dtype=np.float64)
            learner_counter = 0
            flag = np.zeros(agent_comm.size - 1)
            # Initialize epsilon
            epsilon_win.Lock(0)
            epsilon_win.Put(epsilon, target_rank=0)
            epsilon_win.Flush(0)
            epsilon_win.Unlock(0)

            data_error_counter = 0
            data_counter = 0

            while episode_count_learner < workflow.nepisodes:
                # print("EPISODE COUNT: ", episode_count, flush=True)
                episode_win.Lock(0)
                # Atomic Get_accumulate to fetch episode count
                episode_win.Get_accumulate(np.ones(1, dtype=np.float64), episode_count_learner, target_rank=0, op=MPI.NO_OP)
                episode_win.Flush(0)
                episode_win.Unlock(0)

                # Go over all actors (actor processes start from rank 1)
                # s = (learner_counter % (agent_comm.size - 1)) + 1
                # s = np.random.randint(low=1, high=agent_comm.size, size=1)
                # # Get data
                # data_win.Lock(s)
                # data_win.Get(data_buffer, target_rank=s, target=None)
                # data_win.Unlock(s)
                # TODO: call BufferDataset
                buff = bytearray(serial_agent_batch_size)
                batch_gen = BufferDataset(workflow.agent, buff, data_win).prefetch(-1).cache().batch(256).map(workflow.agent.bellman_equation)
                # # Continue to the next actor if data_buffer is empty
                # try:
                #     agent_data = MPI.pickle.loads(data_buffer)
                # except Exception as e:
                #     continue

                reTrace.snapshot()
                epTrace.snapshot()
                # print('***************************')
                # Train & Target train
                workflow.agent.train(batch_gen)
                ib.update("Async_Learner_Train", 1)
                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                ib.update("Async_Learner_Target_Train", 1)
                # Share new model weights
                target_weights = workflow.agent.get_weights()
                serial_target_weights = MPI.pickle.dumps(target_weights)
                model_win.Lock(0)
                model_win.Put(serial_target_weights, target_rank=0)
                model_win.Unlock(0)
                moTrace.snapshot()
                learner_counter += 1
                # ib.update("Async_Learner_Episode", 1)

            logger.info('Learner exit on rank_episode: {}_{}'.format(agent_comm.rank, episode_data))

        # Actors
        else:
            local_actor_episode_counter = 0
            if ExaComm.is_actor():
                # Logging files
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' + filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

                episode_count_actor = np.zeros(1, dtype=np.float64)
                one = np.ones(1, dtype=np.float64)
                epsilon_update = np.zeros(1, dtype=np.float64)
                epsilon = np.zeros(1, dtype=np.float64)
                count = 0

                # Get initial value of episode counter
                episode_win.Lock(0)
                # Atomic Get using Get_accumulate
                episode_win.Get_accumulate(one, episode_count_actor, target_rank=0, op=MPI.NO_OP)
                episode_win.Flush(0)
                episode_win.Unlock(0)

            while episode_count_actor < workflow.nepisodes:
                if ExaComm.is_actor():
                    episode_win.Lock(0)
                    # Atomic Get_accumulate to increment the episode counter
                    episode_win.Get_accumulate(one, episode_count_actor, target_rank=0)
                    episode_win.Flush(0)
                    episode_win.Unlock(0)

                episode_count_actor = env_comm.bcast(episode_count_actor, root=0)

                # Include another check to avoid each actor running extra
                # set of steps while terminating
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
                    if ExaComm.is_actor():
                        # Update model weight
                        # TODO: weights are updated each step -- REVIEW --
                        buff = bytearray(serial_target_weights_size)
                        model_win.Lock(0)
                        model_win.Get(buff, target=0, target_rank=0)
                        model_win.Flush(0)
                        model_win.Unlock(0)
                        moTrace.update()
                        target_weights = MPI.pickle.loads(buff)
                        workflow.agent.set_weights(target_weights)

                        # Atomic Get_accumulate to get epsilon
                        epsilon_win.Lock(0)
                        epsilon_win.Get(epsilon, target_rank=0)
                        epsilon_win.Flush(0)
                        epsilon_win.Unlock(0)

                        workflow.agent.epsilon = epsilon

                        # Inference action
                        action, policy_type = workflow.agent.action(current_state)
                        ib.update("Async_Env_Inference", 1)
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11

                        epsilon = np.array(workflow.agent.epsilon)
                        # Atomic Get_accumulate to update epsilon
                        epsilon_win.Lock(0)
                        epsilon_win.Put(epsilon, target_rank=0)
                        epsilon_win.Flush(0)
                        epsilon_win.Unlock(0)

                    # Environment step
                    ib.startTrace("step", 0)
                    next_state, reward, done, _ = workflow.env.step(action)
                    ib.stopTrace()
                    ib.update("Async_Env_Step", 1)

                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True
                    # Broadcast done
                    done = env_comm.bcast(done, root=0)
                    if ExaComm.is_actor():
                        # Save memory
                        total_rewards += reward
                        memory = (current_state, action, reward, next_state, done, total_rewards)
                        workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                        # batch_data = BufferDataset(workflow.agent, buff, data_win).prefetch(-1).cache().batch(256).map(workflow.agent.bellman_equation)
                        # batch_data=next(workflow.agent.generate_data())
                        ib.update("Async_Env_Generate_Data", 1)

                        # Write to data window
                        serial_agent_batch = MPI.pickle.dumps(memory)
                        data_win.Lock(agent_comm.rank)
                        data_win.Put(serial_agent_batch, target_rank=agent_comm.rank)
                        data_win.Unlock(agent_comm.rank)
                        reTrace.update(value=total_rewards)
                        epTrace.update()

                        # Log state, action, reward, ...
                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_rewards,
                                               done, local_actor_episode_counter, steps, policy_type, workflow.agent.epsilon])
                        train_file.flush()
                    ib.update("Async_Env_Episode", 1)

        if ExaComm.is_agent():
            model_win.Free()
            data_win.Free()

        Trace_Win.write(workflow.results_dir)
        Trace_Win.plot(workflow.results_dir)
