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

    def learner(self, workflow, block_size):
        agent_comm = ExaComm.agent_comm
        episode = 0
        done_episodes = 0
        for dst in range(1, agent_comm.size):
            agent_comm.send([episode, workflow.agent.epsilon, workflow.agent.get_weights()], dst)
            episode += 1

        while done_episodes < workflow.nepisodes:
            dsts = []
            for dst in range(1, block_size):
                recv_data = agent_comm.recv(None)
                src, step, batch, policy_type, done  = recv_data
                dsts.append(src)

                train_return = workflow.agent.train(batch)
                workflow.agent.target_train()

                if policy_type == 0:
                    workflow.agent.epsilon_adj()

            for dst in dsts:
                agent_comm.send([episode, workflow.agent.epsilon, workflow.agent.get_weights()], dst)
            
            if done:
                episode += 1
                done_episodes += 1

        filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
        workflow.agent.save(workflow.results_dir + '/' + filename_prefix + '.h5')

    def worker(self, workflow):
        agent_comm = ExaComm.agent_comm
        env_comm = ExaComm.env_comm

        if env_comm.rank == 0:
            filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
            train_file = open(workflow.results_dir + '/' + filename_prefix + ".log", 'w')
            train_writer = csv.writer(train_file, delimiter=" ")

        while True:
            episode, epsilon, weights = agent_comm.recv(None, source=0)
            episode = env_comm.bcast(episode, 0)
            if episode >= workflow.nepisodes:
                break

            if env_comm.rank == 0:
                workflow.agent.epsilon = epsilon
                workflow.agent.set_weights(weights)
            
            steps = 0
            total_reward = 0
            current_state = workflow.env.reset()
            while steps < workflow.nsteps:
                done = False
                while done == False:
                    action, policy_type = workflow.agent.action(current_state)
                    if workflow.action_type == "fixed":
                        action, policy_type = 0, -11
                    action = env_comm.bcast(action, root=0)
                    next_state, reward, done, _ = workflow.env.step(action)
                    
                    if env_comm.rank == 0:
                        workflow.agent.remember(current_state, action, reward, next_state, done)
                        total_reward += reward
                    
                    train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward, done, episode, steps, policy_type, workflow.agent.epsilon])
                    train_file.flush()

                    current_state = next_state
                    steps += 1

                    if steps == workflow.nsteps:
                        done = True 
                    done = env_comm.bcast(done, 0)

            batch_data = next(workflow.agent.generate_data())
            agent_comm.send([agent_comm.rank, steps, batch_data, policy_type, done], 0)

    @PROFILE
    def run(self, workflow):
        if ExaComm.is_learner():
            self.learner(workflow, ExaComm.agent_comm.size)
        else:
            self.worker(workflow)
        