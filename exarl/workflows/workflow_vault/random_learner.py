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
import csv
import time
import pandas as pd
from os.path import join
import exarl
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals

class RANDOM(exarl.ExaWorkflow):
    """
    Random workflow class: inherits from Exaworkflow base class.
    Used for testing inference against random actions.
    """

    def __init__(self):
        """
        Random workflow class constructor. The weight file gets loaded for
        inference.
        """
        print('Class Random learner')
        data_dir = ExaGlobals.lookup_params("output_dir")
        data_file = ExaGlobals.lookup_params("random_results_file")
        self.load_data = ExaGlobals.lookup_params("weight_file")
        if self.load_data == "None":
            self.load_data = None
        self.out_file = join(data_dir, data_file)

    def run(self, exalearner):
        """
        This function implements the random workflow in EXARL.
        Args:
            exalearner (ExaLearner type object): The ExaLearner object is used to access
            different members of the base class.
        """
        agent_comm = ExaComm.agent_comm
        env_comm = ExaComm.env_comm

        episodesPerActor = int(exalearner.nepisodes / (agent_comm.size - 1))
        if exalearner.nepisodes % (agent_comm.size - 1):
            episodesPerActor += 1

        df = pd.DataFrame(columns=['rank', 'episode', 'step', 'reward', 'totalReward', 'done'])

        if self.load_data is not None:
            if ExaComm.is_learner():
                exalearner.agent.load(self.load_data)

            target_weights = exalearner.agent.get_weights()
            target_weights = agent_comm.bcast(target_weights, 0)

            if not ExaComm.is_learner():
                exalearner.agent.set_weights(target_weights)

        if not ExaComm.is_learner():
            if ExaComm.env_comm.rank == 0:
                # Setup logger
                filename_prefix = 'ExaLearner_Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(exalearner.nepisodes), str(exalearner.nsteps), str(agent_comm.rank))
                train_file = open(exalearner.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            for episode in range(episodesPerActor):
                total_reward = 0
                current_state = exalearner.env.reset()

                for step in range(exalearner.nsteps):
                    if ExaComm.env_comm.rank == 0:
                        if self.load_data is None:
                            action = exalearner.env.action_space.sample()
                        else:
                            action, _ = exalearner.agent.action(current_state)
                    action = env_comm.bcast(action, 0)
                    next_state, reward, done, _ = exalearner.env.step(action)
                    current_state = next_state

                    if step + 1 == exalearner.nsteps:
                        done = True

                    done = env_comm.bcast(done, 0)
                    if ExaComm.env_comm.rank == 0:
                        total_reward += reward

                    train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                           done, episode, step, 1, exalearner.agent.epsilon])
                    train_file.flush()

                    df = df.append({'rank': agent_comm.rank, 'episode': episode, 'step': step, 'reward': reward,
                                    'totalReward': total_reward, 'done': done}, ignore_index=True)
                    if done:
                        break
            agent_comm.send(df, 0)
            train_file.close()

        else:
            recv_data = None
            for i in range(1, agent_comm.size):
                recv_data = agent_comm.recv(recv_data, source=i)
                df = df.append(recv_data)

            print("Writing to", self.out_file)
            df.to_csv(path_or_buf=self.out_file)
            print("Done.")
