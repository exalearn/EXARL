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
from exarl.utils.introspect import ib
from exarl.utils.profile import *
from exarl.utils import log
import exarl.utils.candleDriver as cd
from exarl.base.comm_base import ExaComm

logger = log.setup_logger(__name__, cd.run_params['log_level'])

class SIMPLE(erl.ExaWorkflow):
    def __init__(self):
        print('Creating SIMPLE learner workflow...')

        # Do we wait for an episode from each actor
        self.block = cd.lookup_params('episode_block', default=True)

        # Save weights after each episode
        self.save_weights_per_episode = cd.lookup_params('save_weights_per_episode', default=False)

    def save_weights(self, workflow, episode):
        if self.save_weights_per_episode and episode != self.nepisodes:
           workflow.agent.save(workflow.results_dir + '/' + self.filename_prefix + '_' + str(episode) + '.h5')
        else:
            workflow.agent.save(workflow.results_dir + '/' + self.filename_prefix + '.h5')

    def write_log(self, current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type, epsilon):
        if ExaComm.env_comm.rank == 0:
            self.train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type, epsilon])
            self.train_file.flush()

    def send_model(self, workflow, episode, dst):
        ExaComm.agent_comm.send([episode, workflow.agent.epsilon, workflow.agent.get_weights()], dst)

    def recv_model(self):
        return ExaComm.agent_comm.recv(None, source=0)

    def send_batch(self, batch_data, policy_type, done):
        ExaComm.agent_comm.send([ExaComm.agent_comm.rank, batch_data, policy_type, done], 0)

    def recv_batch(self):
        return ExaComm.agent_comm.recv(None)

    def learner(self, workflow, nepisodes, block):
        if block:
            block_size = ExaComm.agent_comm.size
        else:
            block_size = 1
        
        next_episode = 0
        done_episode = 0
        episode_per_rank = [0] * ExaComm.agent_comm.size
        
        for dst in range(1, ExaComm.agent_comm.size):
            self.send_model(workflow, next_episode, dst)
            episode_per_rank[dst] = next_episode
            next_episode += 1

        while done_episode < nepisodes:
            for dst in range(1, block_size):
                src, batch, policy_type, done = self.recv_batch()
                train_return = workflow.agent.train(batch)
                workflow.agent.target_train()

                if policy_type == 0:
                    workflow.agent.epsilon_adj()

                if done:
                    done_episode += 1
                    episode_per_rank[src] = next_episode
                    next_episode += 1

            for dst in range(1, block_size):
                self.send_model(workflow, episode_per_rank[dst], dst)

            self.save_weights(workflow, done_episode)

    def actor(self, workflow, nepisodes):
        while True:
            episode, epsilon, weights = self.recv_model()
            episode = ExaComm.env_comm.bcast(episode, 0)
            if episode >= nepisodes:
                break

            if ExaComm.env_comm.rank == 0:
                workflow.agent.epsilon = epsilon
                workflow.agent.set_weights(weights)
            
            total_reward = 0
            steps = 0
            done = False
            current_state = workflow.env.reset()
            
            while not done:
                action, policy_type = workflow.agent.action(current_state)
                if workflow.action_type == "fixed":
                    action, policy_type = 0, -11
                action = ExaComm.env_comm.bcast(action, root=0)
                next_state, reward, done, _ = workflow.env.step(action)
                
                if ExaComm.env_comm.rank == 0:
                    workflow.agent.remember(current_state, action, reward, next_state, done)
                    total_reward += reward
                
                self.write_log(current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type, workflow.agent.epsilon)
                
                current_state = next_state
                steps += 1

                if steps == workflow.nsteps:
                    done = True 
                done = ExaComm.env_comm.bcast(done, 0)

            batch_data = next(workflow.agent.generate_data())
            self.send_batch(batch_data, policy_type, done)
            
    @PROFILE
    def run(self, workflow):
        # Round to an even number for blocking purposes
        nactors = ExaComm.global_comm.size - ExaComm.num_learners
        if workflow.nepisodes % nactors == 0:
            nepisodes = workflow.nepisodes
        else:
            nepisodes = int(workflow.nepisodes / nactors) * nactors

        self.filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' % (str(workflow.nepisodes), str(workflow.nsteps), str(ExaComm.agent_comm.rank))
        if ExaComm.env_comm.rank == 0:
            self.train_file = open(workflow.results_dir + '/' + self.filename_prefix + ".log", 'w')
            self.train_writer = csv.writer(self.train_file, delimiter=" ")

        if ExaComm.is_learner():
            self.learner(workflow, nepisodes, self.block)
        else:
            self.actor(workflow, nepisodes)