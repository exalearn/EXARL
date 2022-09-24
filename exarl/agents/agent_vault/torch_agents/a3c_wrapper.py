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
import numpy as np
from deeprlalgo.agents.Base_Agent import Base_Agent
from deeprlalgo.agents.actor_critic_agents.A3C import Actor_Critic_Worker
from torch.optim import Adam


from deeprlalgo.utilities.data_structures.Config import Config
from deeprlalgo.agents.Trainer import Trainer

import exarl
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals
from exarl.utils.introspect import introspectTrace
logger = ExaGlobals.setup_logger(__name__)

class Actor_Critic:
    def __init__(self, worker_num, environment, shared_model, counter, optimizer_lock, shared_optimizer,
                 config, episodes_to_run, epsilon_decay_denominator, action_size, action_types, results_queue,
                 local_model, gradient_updates_queue):
        # JS: This copies all the methods from Actor_Critic_Worker
        # We are doing this to strip the Pytorch.multiprocessing
        [setattr(Actor_Critic, attribute, getattr(Actor_Critic_Worker, attribute)) for attribute in dir(Actor_Critic_Worker) if callable(getattr(Actor_Critic_Worker, attribute)) and attribute.startswith('__') is False]

        
        self.environment = environment
        self.config = config
        self.worker_num = worker_num

        self.gradient_clipping_norm = self.config.hyperparameters["gradient_clipping_norm"]
        self.discount_rate = self.config.hyperparameters["discount_rate"]
        self.normalise_rewards = self.config.hyperparameters["normalise_rewards"]

        self.action_size = action_size
        self.set_seeds(self.worker_num)
        # self.shared_model = shared_model
        self.local_model = local_model
        self.local_optimizer = Adam(self.local_model.parameters(), lr=0.0, eps=1e-4)
        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer
        self.episodes_to_run = episodes_to_run
        self.epsilon_decay_denominator = epsilon_decay_denominator
        self.exploration_worker_difference = self.config.hyperparameters["exploration_worker_difference"]
        self.action_types = action_types
        self.results_queue = results_queue

        self.gradient_updates_queue = gradient_updates_queue

        self.epsilon_exploration = self.calculate_new_exploration()
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_action_probabilities = []
        self.critic_outputs = []

    def action(self, state):
        # JS: This is stripped from the run method
        action, action_log_prob, critic_outputs = self.pick_action_and_get_critic_values(self.local_model, state, self.epsilon_exploration)
        self.episode_log_action_probabilities.append(action_log_prob)
        self.critic_outputs.append(critic_outputs)
        return action

    def remember(self, state, action, reward, done):
        # JS: Also stripped from the run method
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        if done:
            total_loss = self.calculate_total_loss()
            self.put_gradients_in_queue(total_loss)
            self.results_queue.put(np.sum(self.episode_rewards))

    def reset_game_for_worker(self):
        if self.action_types == "CONTINUOUS": 
            self.noise.reset()

class A3C(Base_Agent):
    agent_name = "A3C"

    class Queue:
        def __init__(self):
            self.data = []
        def put(self, item):
            self.data.append(item)
        def get(self):
            return self.data.pop(0)
        def empty(self):
            return len(self.data) == 0

    class dummy_lock:
        def __init__(self):
            pass
        def __enter__(self):
            pass
        def __exit__(self, type, value, traceback):
            pass 

    class counter:
        def __init__(self, env):
            self.env = env
            self.value = -1
            # JS: This is to use counter with "with"
            this = self.env
            class episode_lock:
                def __init__(self):
                    pass
                def __enter__(self):
                    this.value = this.env.workflow_episode
                def __exit__(self, type, value, traceback):
                    pass
            self.get_lock = episode_lock

    def __init__(self, config, env):
        super(A3C, self).__init__(config)
        self.actor_critic = self.create_NN(input_dim=self.state_size, output_dim=[self.action_size, 1])
        self.actor_critic_optimizer = Adam(self.actor_critic.parameters(), lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)

        self.results_queue = A3C.Queue()
        self.gradient_updates_queue = A3C.Queue()
        process_num = ExaComm.agent_comm.rank - ExaComm.num_learners
        # self.worker = Actor_Critic(process_num, env, self.actor_critic, A3C.counter(env), A3C.dummy_lock(),
        self.worker = Actor_Critic(process_num, env, None, A3C.counter(env), A3C.dummy_lock(),
                                    self.actor_critic_optimizer, self.config, -1,
                                    self.hyperparameters["epsilon_decay_rate_denominator"],
                                    self.action_size, self.action_types,
                                    self.results_queue, self.actor_critic, self.gradient_updates_queue)
                                    # self.results_queue, copy.deepcopy(self.actor_critic), self.gradient_updates_queue)

    def update_shared_model(self):
        """Worker that updates the shared model with gradients as they get put into the queue"""
        gradients = self.gradient_updates_queue.get()
        self.actor_critic_optimizer.zero_grad()
        for grads, params in zip(gradients, self.actor_critic.parameters()):
            params._grad = grads
        self.actor_critic_optimizer.step()

class torch_a3c(exarl.ExaAgent):
    def __init__(self, env, is_learner):
        # These are exarl requirements. We probably won't use them...
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 0
        self.epsilon = 0
        self.step_number = 0
        self.nsteps = ExaGlobals.lookup_params('n_steps')

        # These are the (hyper)parameters.  These need to connect to agent config
        # agent_name = ExaGlobals.lookup_params('which')
        # which = torch_dqn.dqn_agents[agent_name]
        self.config = Config()
        self.config.environment = self.env
        self.config.seed = ExaGlobals.lookup_params('seed')
        self.config.use_GPU = ExaGlobals.lookup_params('use_GPU')
        # self.config.standard_deviation_results = ExaGlobals.lookup_params('standard_deviation_results')
        self.config.randomise_random_seed = ExaGlobals.lookup_params('randomise_random_seed')
        self.config.hyperparameters = {
            # "learning_rate": ExaGlobals.lookup_params('learning_rate'),
            "learning_rate": 0.005,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": ["SOFTMAX", None],
            "gradient_clipping_norm": 5.0,
            "discount_rate": 0.99,
            "epsilon_decay_rate_denominator": 1.0,
            "normalise_rewards": True,
            "exploration_worker_difference": 2.0,
            "clip_rewards": False,

            "Actor": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [64, 64],
                "final_layer_activation": "Softmax",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [64, 64],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "min_steps_before_learning": 400,
            "batch_size": 256,
            "discount_rate": 0.99,
            "mu": 0.0, #for O-H noise
            "theta": 0.15, #for O-H noise
            "sigma": 0.25, #for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 1,
            "learning_updates_per_learning_session": 1,
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": None,
            "add_extra_noise": False,
            "do_evaluation_iterations": True,
            "epsilon_decay_rate_denominator": 10
        }

        if not hasattr(self.env.spec, "trials"):
            self.env.spec.trials = ExaGlobals.lookup_params('rolling_reward_length')

        self.agent = A3C(self.config, self.env)

    def get_weights(self):
        # This needs to get the weights that the actors will use for inference
        return self.agent.actor_critic.state_dict()

    def set_weights(self, weights):
        # Stores the weights used by the actor for inference
        self.agent.actor_critic.load_state_dict(weights)
            
    def action(self, state):
        return self.agent.worker.action(state), 1

    def remember(self, state, action, reward, next_state, done):
        self.step_number += 1
        done = done or self.nsteps == self.step_number
        self.agent.worker.remember(state, action, reward, done)
        
    def has_data(self):
        return not self.agent.gradient_updates_queue.empty()

    def generate_data(self):
        yield self.agent.gradient_updates_queue.get(), self.agent.results_queue.get()

    def train(self, batch):
        gradients, total_reward = batch
        
        # From update_shared_model
        self.agent.actor_critic_optimizer.zero_grad()
        for grads, params in zip(gradients, self.actor_critic.parameters()):
            params._grad = grads
        self.agent.actor_critic_optimizer.step()
        
        # From print_results
        self.agent.total_episode_score_so_far = total_reward
        # self.agent.save_and_print_result()
        self.agent.save_result()
    
    # Ignore
    def update_target(self):
        pass

    # This is for if you need to get some return from training back to an actor... Ignore for now
    def set_priorities(self, indices, loss):
        pass