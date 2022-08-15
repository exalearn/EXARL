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
import functools

import tensorflow as tf
from exarl.agents.agent_vault.bsuite_baseline.utils.sequence import Trajectory
import sonnet as snt
import dm_env
import tree

from bsuite.utils.gym_wrapper import space2spec
from bsuite.baselines.utils import sequence
from bsuite.baselines.tf import dqn
from bsuite.baselines.tf import actor_critic
from bsuite.baselines.tf import actor_critic_rnn
from bsuite.baselines.tf.actor_critic.agent import PolicyValueNet
from bsuite.baselines.tf.actor_critic.agent import ActorCritic
from bsuite.baselines.tf.actor_critic_rnn.agent import PolicyValueRNN
from bsuite.baselines.tf.actor_critic_rnn.agent import ActorCriticRNN
# from exarl.agents.agent_vault.bsuite_baseline.tf import dqn

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.utils.introspect import introspectTrace

logger = ExaGlobals.setup_logger(__name__)

class bsuite_agent(exarl.ExaAgent):

    def __init__(self, env, is_learner):
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 0

        self.numEntries = 0
        self.last_timestep = None
        self.last_gym_state = None
        self.nsteps = ExaGlobals.lookup_params('n_steps')

        self.observation_spec = space2spec(env.observation_space)
        self.action_spec = space2spec(env.action_space)
        
        self.which_agent = ExaGlobals.lookup_params('which_agent')
        bsuite_default = ExaGlobals.lookup_params('bsuite_default')
        if bsuite_default:
            if self.which_agent == "dqn":
                self.agent = dqn.default_agent(self.observation_spec, self.action_spec)
                self.epsilon = self.agent._epsilon
                self.batch_size = self.agent._batch_size
                self.buffer_capacity = self.agent._replay._capacity
            elif self.which_agent == "actor-critic":
                self.agent = actor_critic.default_agent(self.observation_spec, self.action_spec)
                self.epsilon = 0
                self.batch_size = self.agent._buffer._max_sequence_length
                self.buffer_capacity = self.agent._buffer._max_sequence_length
            elif self.which_agent == "actor-critic-rnn":
                self.agent = actor_critic_rnn.default_agent(self.observation_spec, self.action_spec)
                self.epsilon = 0
                self.batch_size = self.agent._buffer._max_sequence_length
                self.buffer_capacity = self.agent._buffer._max_sequence_length
        else:
            if self.which_agent == "dqn":
                self.epsilon = ExaGlobals.lookup_params('epsilon')
                self.batch_size = ExaGlobals.lookup_params('batch_size')
                self.buffer_capacity = ExaGlobals.lookup_params('replay_capacity')
                network = snt.Sequential([snt.Flatten(), snt.nets.MLP([50, 50, self.action_spec.num_values]),])
                optimizer = snt.optimizers.Adam(learning_rate=ExaGlobals.lookup_params('learning_rate'))
                self.agent = dqn.DQN(action_spec=self.action_spec,
                                    network=network,
                                    optimizer=optimizer,
                                    epsilon=self.epsilon,
                                    batch_size=self.batch_size,
                                    replay_capacity=self.buffer_capacity,
                                    discount=ExaGlobals.lookup_params('discount'),
                                    sgd_period=ExaGlobals.lookup_params('sgd_period'),
                                    min_replay_size=ExaGlobals.lookup_params('min_replay_size'),
                                    target_update_period=ExaGlobals.lookup_params('update_target_frequency'),
                                    seed=ExaGlobals.lookup_params('seed'))
            elif self.which_agent == "actor-critic":
                self.epsilon = 0
                self.batch_size = ExaGlobals.lookup_params('max_sequence_length')
                self.buffer_capacity = self.batch_size
                optimizer = snt.optimizers.Adam(learning_rate=ExaGlobals.lookup_params('learning_rate'))
                network = PolicyValueNet(hidden_sizes=[64, 64], action_spec=self.action_spec,)
                self.agent = ActorCritic(obs_spec=self.observation_spec,
                                         action_spec=self.action_spec,
                                         network=network,
                                         optimizer=optimizer,
                                         max_sequence_length=self.batch_size,
                                         td_lambda=ExaGlobals.lookup_params('td_lambda'),
                                         discount=ExaGlobals.lookup_params('discount'),
                                         seed=ExaGlobals.lookup_params('seed'))
            elif self.which_agent == "actor-critic-rnn":
                self.epsilon = 0
                self.batch_size = ExaGlobals.lookup_params('max_sequence_length')
                self.buffer_capacity = self.batch_size
                optimizer = snt.optimizers.Adam(learning_rate=ExaGlobals.lookup_params('learning_rate'))
                network = PolicyValueRNN(hidden_sizes=[64, 64], num_actions=self.action_spec.num_values,)
                self.agent = ActorCriticRNN(obs_spec=self.observation_spec,
                                            action_spec=self.action_spec,
                                            network=network,
                                            optimizer=optimizer,
                                            max_sequence_length=self.batch_size,
                                            td_lambda=ExaGlobals.lookup_params('td_lambda'),
                                            discount=ExaGlobals.lookup_params('discount'),
                                            seed=ExaGlobals.lookup_params('seed'))
                    
    def get_weights(self):
        try:
            if self.which_agent == "dqn":
                return [copy.deepcopy(self.agent._target_network.trainable_variables), 
                        copy.deepcopy(self.agent._online_network.trainable_variables)]
            
            elif "actor-critic" in self.which_agent:
                return copy.deepcopy(self.agent._network.trainable_variables)
        except:
            return None

    def set_weights(self, weights):
        if weights:
            if self.which_agent == "dqn":
                if weights is not None:
                    target_weights, online_weights = weights
                    for target, param in zip(self.agent._target_network.trainable_variables, target_weights):
                        target.assign(param)
                    
                    for online, param in zip(self.agent._online_network.trainable_variables, target_weights):
                        online.assign(param)
            
            elif "actor-critic" in self.which_agent:
                for network, param in zip(self.agent._network.trainable_variables, weights):
                        network.assign(param)
            

    def action(self, state):
        if self.last_timestep is None:
            self.last_timestep = dm_env.restart(state.astype("float32"))
        else:
            assert state is self.last_gym_state, str(state) + " " + str(self.last_gym_state)
        return self.agent.select_action(self.last_timestep), 1

    @introspectTrace()
    def inner_remember(self, action, new_timestep):
        assert self.last_timestep is not None
        self.agent.update(self.last_timestep, action, new_timestep)

    def remember(self, state, action, reward, next_state, done):
        if self.last_timestep is None:
            self.last_timestep = dm_env.restart(state.astype("float32"))
        
        if self.env.workflow_step == self.nsteps:
            new_timestep = dm_env.truncation(reward, next_state.astype("float32"))
            local_done = True
        elif done:
            new_timestep = dm_env.termination(reward, next_state.astype("float32"))
            local_done = True
        else:
            new_timestep = dm_env.transition(reward, next_state.astype("float32"))
            local_done = False

        self.inner_remember(action, new_timestep)

        if local_done:
            self.last_timestep = None
            self.last_gym_state = None
        else:
            self.last_timestep = new_timestep
            self.last_gym_state = next_state

        self.numEntries+=1

    def has_data(self):
        return self.numEntries > 0

    def generate_data(self):
        yield [], []

    def train(self, batch):
        pass

    def update_target(self):
        pass

    def set_priorities(self, indices, loss):
        pass

def singleton(my_cls):
    _instance = None
    _init = False
    original_new = my_cls.__new__
    original_init = my_cls.__init__

    @functools.wraps(my_cls.__new__)
    def __new__(cls, *args, **kwargs):
        nonlocal _instance
        if _instance is None:
            _instance = original_new(cls)
        return _instance

    @functools.wraps(my_cls.__init__)
    def __init__(self, *args, **kwargs):
        nonlocal _init
        if not _init:
            _init = True
            original_init(self, *args, **kwargs)

    my_cls.__new__ = __new__
    my_cls.__init__ = __init__
    return my_cls

@singleton
class serial_bsuite_agent(bsuite_agent):
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass

class parallel_bsuite_agent(bsuite_agent):
    def __init__(self, env, is_learner):
        super(parallel_bsuite_agent, self).__init__(env, is_learner)

        self.train_frequency = ExaGlobals.lookup_params('train_frequency')
        self.train_count = 0
        
        if self.which_agent == "dqn":
            # JS: Do a prediction to initialize networks
            sample = [env.observation_space.sample() for i in range(self.batch_size)]
            sample = tf.convert_to_tensor(sample)
            self.agent._online_network(sample)
            self.agent._target_network(sample)
            self.local = []
        elif "actor-critic" in self.which_agent:
            # self.agent._network(sample)
            self.local = sequence.Buffer(self.observation_spec, self.action_spec, self.batch_size)

        

    def inner_remember(self, action, new_timestep):
        # JS: Store data locally
        if self.which_agent == "dqn":
            self.local.append((self.last_timestep, action, new_timestep))
        elif "actor-critic" in self.which_agent:
            self.local.append(self.last_timestep, action, new_timestep)

    def has_data(self):
        return len(self.local) > 0

    def generate_data(self):
        if self.which_agent == "dqn":
            ret = self.local
            self.local = []
            yield ret, []
        elif "actor-critic" in self.which_agent:
            trajectory = []
            if not self.local.empty():
                if self.local.full() or self.last_timestep is None or self.last_timestep.last():
                    trajectory = self.local.drain()
                    trajectory = [tree.map_structure(tf.convert_to_tensor, trajectory)]
            yield trajectory, []    
    
    @introspectTrace()
    def inner_train(self, exps):
        if self.which_agent == "dqn":
            self.agent.update(*exps[-1])
        elif "actor-critic" in self.which_agent and len(exps):
            self.agent._step(exps[0])
    
    def train(self, batch):
        exps, _ = batch
        if self.which_agent == "dqn":
            [self.agent._replay.add([timestep.observation, action, new_timestep.reward, new_timestep.discount, new_timestep.observation,]) for timestep, action, new_timestep in exps[:-1]]
            if self.train_count % self.train_frequency:
                timestep, action, new_timestep = exps[-1]
                self.agent._replay.add([timestep.observation, action, new_timestep.reward, new_timestep.discount, new_timestep.observation,])
                self.agent._total_steps.assign_add(len(exps))
            else:
                self.agent._total_steps.assign_add(len(exps)-1)
                self.inner_train(exps)
        elif "actor-critic" in self.which_agent:
            self.inner_train(exps)