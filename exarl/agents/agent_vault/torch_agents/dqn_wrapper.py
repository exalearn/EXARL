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
import copy
import random
import functools
from deeprlalgo.agents.DQN_agents.DQN import DQN
from deeprlalgo.agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from deeprlalgo.agents.DQN_agents.DDQN import DDQN
from deeprlalgo.agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from deeprlalgo.agents.DQN_agents.Dueling_DDQN import Dueling_DDQN

from deeprlalgo.utilities.data_structures.Config import Config
from deeprlalgo.agents.Trainer import Trainer

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.utils.introspect import introspectTrace
logger = ExaGlobals.setup_logger(__name__)

class torch_dqn(exarl.ExaAgent):

    dqn_agents = {
        "DQN" : DQN,
        "DQN_With_Fixed_Q_Targets" : DQN_With_Fixed_Q_Targets,
        "DDQN" : DDQN,
        "DDQN_With_Prioritised_Experience_Replay" : DDQN_With_Prioritised_Experience_Replay,
        "Dueling_DDQN" : Dueling_DDQN
    }

    def __init__(self, env, is_learner):
        # These are exarl requirements. We probably won't use them...
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 0
        self.epsilon = 0

        self.nsteps = ExaGlobals.lookup_params('n_steps')

        # How often to train
        self.train_frequency = ExaGlobals.lookup_params('train_frequency')
        self.train_count = 0
        
        # Store data locally and pass to learner
        self.local = []
        self.local_reward = 0

        # These are the (hyper)parameters.  These need to connect to agent config
        agent_name = ExaGlobals.lookup_params('which')
        which = torch_dqn.dqn_agents[agent_name]
        self.config = Config()
        self.config.environment = self.env
        self.config.seed = ExaGlobals.lookup_params('seed')
        self.config.use_GPU = ExaGlobals.lookup_params('use_GPU')
        self.config.standard_deviation_results = ExaGlobals.lookup_params('standard_deviation_results')
        self.config.randomise_random_seed = ExaGlobals.lookup_params('randomise_random_seed')
        self.config.hyperparameters = {
            "learning_rate": ExaGlobals.lookup_params('learning_rate'),
            "batch_size": ExaGlobals.lookup_params('batch_size'),
            "buffer_size": ExaGlobals.lookup_params('buffer_size'),
            "epsilon": ExaGlobals.lookup_params('epsilon'),
            "epsilon_decay_rate_denominator": ExaGlobals.lookup_params('epsilon_decay_rate_denominator'),
            "discount_rate": ExaGlobals.lookup_params('discount_rate'),
            "tau": ExaGlobals.lookup_params('tau'),
            "alpha_prioritised_replay": ExaGlobals.lookup_params('alpha_prioritised_replay'),
            "beta_prioritised_replay": ExaGlobals.lookup_params('beta_prioritised_replay'),
            "incremental_td_error": ExaGlobals.lookup_params('incremental_td_error'),
            # "update_every_n_steps": ExaGlobals.lookup_params('update_target_frequency'),
            "linear_hidden_units": ExaGlobals.lookup_params('linear_hidden_units'),
            "final_layer_activation": ExaGlobals.lookup_params('final_layer_activation'),
            "batch_norm": ExaGlobals.lookup_params('batch_norm'),
            "gradient_clipping_norm": ExaGlobals.lookup_params('gradient_clipping_norm'),
            "learning_iterations": ExaGlobals.lookup_params('learning_iterations'),
            "clip_rewards": ExaGlobals.lookup_params('clip_rewards') is not None
        }
        
        if self.config.randomise_random_seed: 
            self.config.seed = random.randint(0, 2**32 - 2)

        # This is good enough for now
        if not hasattr(self.env.spec, "trials"):
            self.env.spec.trials = ExaGlobals.lookup_params('rolling_reward_length')

        # Create agent
        print("Creating", agent_name, flush=True)
        self.agent = which(self.config)

    def get_weights(self):
        # This needs to get the weights that the actors will use for inference
        return self.agent.q_network_local.state_dict()

    def set_weights(self, weights):
        # Stores the weights used by the actor for inference
        self.agent.q_network_local.load_state_dict(weights)
            
    def action(self, state):
        # The episode is used for epsilon in the greedy approach which is why we set it
        self.agent.episode_number = self.env.workflow_episode
        return self.agent.pick_action(state), 1

    def remember(self, state, action, reward, next_state, done):
        # We are just going to store the data locally, and then send it in generate data
        self.local.append((state, action, reward, next_state, done))
        self.local_reward += reward

    def has_data(self):
        # Do we have any local data
        return len(self.local) > 0

    def generate_data(self):
        # Here we send our batch of data to the learner
        # We use the flags to keep track of total reward within the algorithm
        if self.env.workflow_step == self.nsteps or self.local[-1][-1]:
            flags = (self.local[-1][-1], self.local_reward)
        else:
            flags = (False, 0)
        # Be sure to clear it out...
        ret = self.local
        self.local = []
        yield ret, flags

    def train(self, batch):
        # This will receive the batch from generate_data
        exps, flags = batch
        last_done = flags[0]
        total_reward = flags[1]

        # Store the experiences in the learner's memory
        [self.agent.save_experience(experience=exp) for exp in exps]
        if self.agent.enough_experiences_to_learn_from() and self.train_count % self.train_frequency == 0:
            self.agent.learn()
        
        # We do this because it is adjusting the learning rate
        if last_done:
            self.agent.total_episode_score_so_far = total_reward
            # self.agent.save_and_print_result()
            self.agent.save_result()
        self.train_count += 1

    # Ignore
    def update_target(self):
        pass

    # This is for if you need to get some return from training back to an actor... Ignore for now
    def set_priorities(self, indices, loss):
        pass
