import exarl.mpi_settings as mpi_settings
# import time
# import csv
from mpi4py import MPI
import numpy as np
import exarl as erl
from utils.profile import *
import utils.log as log
import utils.candleDriver as cd

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class RMA_ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print('Creating RMA async workflow...')

    def run(self, workflow):
        # MPI communicators
        agent_comm = mpi_settings.agent_comm
        env_comm = mpi_settings.env_comm

        if mpi_settings.is_learner():
            workflow.agent.set_learner()

        # Allocate RMA windows
        if mpi_settings.is_agent():
            # Get size of episode counter
            disp = MPI.INT64_T.Get_size()
            episode_count = None
            if mpi_settings.is_learner():
                episode_count = np.zeros(1, dtype=np.int64)
            # Allocate episode window
            episode_win = MPI.Win.Create(episode_count, disp, comm=agent_comm)

            # Get serialized target weights size
            target_weights = workflow.agent.get_weights()
            serial_target_weights = MPI.pickle.dumps(target_weights)
            serial_target_weights_size = len(serial_target_weights)            
            target_weights_size = 0
            if mpi_settings.is_learner():
                target_weights_size = serial_target_weights_size
            # Allocate model window
            model_win = MPI.Win.Allocate(target_weights_size, 1, comm=agent_comm)

            # Get serialized batch data size
            nserial_agent_batch = 0
            if mpi_settings.is_actor():
                agent_batch = next(workflow.agent.generate_data())
                serial_agent_batch = (MPI.pickle.dumps(agent_batch))
                nserial_agent_batch = len(serial_agent_batch)
            # Allocate data window
            data_win = MPI.Win.Allocate(nserial_agent_batch, 1, comm=agent_comm)
            
        if mpi_settings.is_learner():
            # Write target weight to model window of learner
            model_win.Lock(0)
            model_win.Put(serial_target_weights, target_rank=0)
            model_win.Unlock(0)

        # Synchronize
        agent_comm.Barrier()

        # Learner
        if mpi_settings.is_learner():
            # Initialize batch data buffer
            buff_data = bytearray(nserial_agent_batch)

            while 1:
                # Check episode counter
                episode_win.Lock(0)
                episode_win.Get(episode_count, target_rank=0, target=None)
                # print('Learner [{}] - total done episode: {}'.format(agent_comm.rank, episode_count))
                if episode_count >= workflow.nepisodes:
                    # print('Learner [{}] exit on episode: {}'.format(agent_comm.rank, episode_count))
                    break
                episode_win.Unlock(0)

                # Loop over all actor data, train, and update model
                for s in range(1, agent_comm.size):
                    # 1) Get data
                    data_win.Lock(s)
                    data_win.Get(buff_data, target_rank=s, target=None)
                    data_win.Unlock(s)
                    agent_data = MPI.pickle.loads(serial_agent_batch)
                    # 2) Train & Target train
                    workflow.agent.train(agent_data)
                    # TODO: Double check if this is already in the DQN code
                    workflow.agent.target_train()
                    # 3) Share new model weights
                    target_weights = workflow.agent.get_weights()
                    serial_target_weights = MPI.pickle.dumps(target_weights)
                    # TODO: loop over all actors (?)
                    model_win.Lock(0)
                    model_win.Put(serial_target_weights, target_rank=0)
                    model_win.Unlock(0)

            print('Learner exit on rank_episode: {}_{}'.format(agent_comm.rank, episode_count))

        else:
            episode_count_actor = np.zeros(1, dtype=np.int64)
            episode_win.Lock(0)
            episode_win.Get(episode_count_actor, target_rank=0, target=None)
            episode_win.Unlock(0)

            while episode_count_actor < workflow.nepisodes:
                workflow.env.seed(0)
                current_state = workflow.env.reset()
                total_rewards = 0
                steps = 0
                action = 0

                while steps < workflow.nsteps:
                    # Check if the next 5 lines are necessary
                    episode_win.Lock(0)
                    episode_win.Get(episode_count_actor, target_rank=0, target=None)
                    if episode_count_actor >= workflow.nepisodes:
                        break
                    episode_win.Unlock(0)

                    # Update model weight
                    # TODO: weights are updated each step -- REVIEW --
                    buff = bytearray(serial_target_weights_size)
                    model_win.Lock(0)
                    model_win.Get(buff, target=0, target_rank=0)
                    model_win.Unlock(0)
                    target_weights = MPI.pickle.loads(buff)
                    workflow.agent.set_weights(target_weights)

                    # Inference action
                    if mpi_settings.is_actor():
                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(current_state)

                    # Environment step
                    next_state, reward, done, _ = workflow.env.step(action)

                    steps += 1
                    if steps >= workflow.nsteps:
                        done = True

                    # Save memory
                    total_rewards += reward
                    memory = (current_state, action, reward, next_state, done, total_rewards)
                    workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                    batch_data = next(workflow.agent.generate_data())
                    # print('Rank [{}] - actor data shape: {}/{}'.format(agent_comm.rank,
                    #                                              batch_data[0].shape, batch_data[1].shape))

                    # Write to data window
                    serial_agent_batch = (MPI.pickle.dumps(batch_data))
                    data_win.Lock(agent_comm.rank)
                    data_win.Put(serial_agent_batch, target_rank=agent_comm.rank)
                    data_win.Unlock(agent_comm.rank)

                    # If done then update the episode counter and exit boolean
                    if done:
                        episode_win.Lock(0)
                        episode_win.Get(episode_count_actor, target_rank=0, target=None)
                        episode_count_actor += 1
                        # print('Rank[{}] - working on episode: {}'.format(agent_comm.rank, episode_count))
                        episode_win.Put(episode_count_actor, target_rank=0)
                        episode_win.Unlock(0)