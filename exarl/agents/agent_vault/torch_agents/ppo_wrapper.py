import random
from deeprlalgo.utilities.data_structures.Config import Config
from deeprlalgo.agents.policy_gradient_agents.PPO import PPO

import exarl
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals
from exarl.utils.introspect import introspectTrace


class torch_ppo(exarl.ExaAgent):
    def __init__(self, env, is_learner):
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 0
        self.epsilon = 0
        self.step_number = 0
        self.episode_number = 0

        self.nsteps = ExaGlobals.lookup_params('n_steps')
        batch_step_frequency = ExaGlobals.lookup_params('batch_step_frequency')
        self.episodes_per_learning_round = ExaGlobals.lookup_params("episodes_per_learning_round")
        batch_episode_frequency = ExaGlobals.lookup_params("batch_episode_frequency")
        num_actors = ExaComm.agent_comm.size - ExaComm.num_learners
        assert batch_step_frequency == -1 or batch_step_frequency == self.nsteps, "Batch step frequency is not support by this agent" 
        if ExaGlobals.lookup_params('workflow') != "sync":
            assert ExaGlobals.lookup_params('episode_block') == True, "Episode block must be set to true"
            assert num_actors * batch_episode_frequency == self.episodes_per_learning_round
        else:
            assert batch_episode_frequency == self.episodes_per_learning_round   
        
        self.config = Config()
        self.config.environment = self.env
        self.config.environment.spec.trials = 100
        self.config.seed = ExaGlobals.lookup_params('seed')
        self.config.use_GPU = ExaGlobals.lookup_params('use_GPU')
        self.config.randomise_random_seed = ExaGlobals.lookup_params('randomise_random_seed')
        self.config.hyperparameters = {
            "learning_rate": ExaGlobals.lookup_params("learning_rate"),
            "linear_hidden_units": ExaGlobals.lookup_params("linear_hidden_units"),
            "final_layer_activation": ExaGlobals.lookup_params("final_layer_activation"),
            "learning_iterations_per_round": ExaGlobals.lookup_params("learning_iterations_per_round"),
            "discount_rate": ExaGlobals.lookup_params("discount_rate"),
            "batch_norm": ExaGlobals.lookup_params("batch_norm"),
            "clip_epsilon": ExaGlobals.lookup_params("clip_epsilon"),
            "episodes_per_learning_round": self.episodes_per_learning_round,
            "normalise_rewards": ExaGlobals.lookup_params("normalise_rewards"),
            "gradient_clipping_norm": ExaGlobals.lookup_params("gradient_clipping_norm"),
            "mu": ExaGlobals.lookup_params("mu"),
            "theta": ExaGlobals.lookup_params("theta"),
            "sigma": ExaGlobals.lookup_params("sigma"),
            "epsilon_decay_rate_denominator": ExaGlobals.lookup_params("epsilon_decay_rate_denominator"),
            "clip_rewards": ExaGlobals.lookup_params("clip_rewards")
        }
        self.agent = PPO(self.config)  
        
        # For some envs this doesn't get set so we give an options via config
        average_score_required_to_win = ExaGlobals.lookup_params("average_score_required_to_win")
        if average_score_required_to_win != "None":
            self.agent.average_score_required_to_win = average_score_required_to_win

        self.local_states = [[]]
        self.local_actions = [[]]
        self.local_rewards = [[]]

        self.exploration_epsilon = None
        self.exploration = None
    
    def get_weights(self):
        exploration_epsilon = self.agent.exploration_strategy.get_updated_epsilon_exploration({"episode_number": self.episode_number})
        return self.agent.policy_old.state_dict(), self.agent.policy_new.state_dict(), exploration_epsilon
    
    def set_weights(self, weights):
        self.agent.policy_old.load_state_dict(weights[0])
        self.agent.policy_new.load_state_dict(weights[1])
        self.exploration_epsilon = weights[2]
        # Set this when we start a new batch of episodes
        self.exploration = None
    
    def action(self, state):
        if self.exploration is None:
            self.exploration = max(0.0, random.uniform(self.exploration_epsilon / 3.0, self.exploration_epsilon * 3.0))
        return self.agent.experience_generator.pick_action(self.agent.policy_new, state, self.exploration), 1

    def remember(self, state, action, reward, next_state, done):
        self.local_states[-1].append(state)
        self.local_actions[-1].append(action)
        self.local_rewards[-1].append(reward)
        self.step_number += 1

        if done or self.nsteps == self.step_number:
            self.local_states.append([])
            self.local_actions.append([])
            self.local_rewards.append([])
            self.step_number = 0
            self.exploration = None

    def has_data(self):
        return len(self.local_rewards[-1]) > 0
    
    def generate_data(self):
        self.local_states.pop()
        self.local_actions.pop()
        self.local_rewards.pop()
        ret = (self.local_states, self.local_actions, self.local_rewards)
        self.local_states = [[]]
        self.local_actions = [[]]
        self.local_rewards = [[]]
        yield ret
   
    def train(self, batch):
        self.agent.many_episode_states.extend(batch[0])
        self.agent.many_episode_actions.extend(batch[1])
        self.agent.many_episode_rewards.extend(batch[2])
        self.episode_number += len(batch[0])

        if len(self.agent.many_episode_states) == self.episodes_per_learning_round:
            self.agent.policy_learn()
            self.agent.update_learning_rate(self.agent.hyperparameters["learning_rate"], self.agent.policy_new_optimizer)
            self.agent.equalise_policies()
            
            # self.agent.save_and_print_result()
            self.agent.save_result()
            
            self.agent.many_episode_states = []
            self.agent.many_episode_actions = []
            self.agent.many_episode_rewards = []

    # Ignore
    def update_target(self):
        pass

    # This is for if you need to get some return from training back to an actor... Ignore for now
    def set_priorities(self, indices, loss):
        pass

