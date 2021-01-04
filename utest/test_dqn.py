import tensorflow as tf
import sys

import exarl as erl
import pytest
import exarl.mpi_settings as mpi_settings

from tensorflow.keras import optimizers, activations, losses
from agents.agent_vault.dqn import DQN
from utils.candleDriver import initialize_parameters



class TestClass:

    #initialize a test_agent
    def __init_test_agent(self):
        global test_agent
        try:
            test_agent = DQN(erl.ExaLearner(run_params).env)
        except TypeError:
            pytest.fail('Abstract class methods not handled correctly',pytrace=True)
        except:
            pytest.fail('Bad Agent Implementation',pytrace=True)
            
        return test_agent

    #1: test initialize_parameters
    def test_initialize_parameters(self):
        global run_params
        try:
            run_params = initialize_parameters()
            assert type(run_params) is dict
        except:
            pytest.fail('Bad initialize_parameters()',pytrace=True)

    #2: test agent __init__ for DQN agent
    def test_init(self):

        try:
            test_agent = self.__init_test_agent()


            assert test_agent.results_dir == run_params['output_dir']
            assert test_agent.gamma == run_params['gamma'] and \
                0 < test_agent.gamma < 1
            assert test_agent.epsilon == run_params['epsilon'] and \
                test_agent.epsilon > test_agent.epsilon_min
            assert test_agent.epsilon_min == run_params['epsilon_min'] and \
                test_agent.epsilon_min > 0
            assert test_agent.epsilon_decay == run_params['epsilon_decay'] and \
                0 < test_agent.epsilon_decay < 1
            assert test_agent.learning_rate == run_params['learning_rate'] and \
                test_agent.learning_rate > 0
            assert test_agent.batch_size == run_params['batch_size'] and \
                test_agent.batch_size > 0
            assert test_agent.tau == run_params['tau'] and \
                0 < test_agent.tau < 1
            assert test_agent.model_type == run_params['model_type'] and \
                test_agent.model_type.upper() in ("LSTM", "MLP")

            # for mlp
            if test_agent.model_type.upper() == "MLP":
                assert test_agent.dense == run_params['dense'] and \
                    type(test_agent.dense) is list and \
                    len(test_agent.dense) > 0 and \
                    all([l > 0 for l in test_agent.dense])

            # for lstm
            if test_agent.model_type.upper() == "LSTM":
                assert test_agent.lstm_layers == run_params['lstm_layers'] and \
                    type(test_agent.lstm_layers) is list and \
                    len(test_agent.lstm_layers) > 0 and \
                    all([ l > 0 for l in test_agent.lstm_layers])
                assert test_agent.gauss_noise == run_params['gauss_noise'] and \
                    type(test_agent.gauss_noise) is list and \
                    len(test_agent.gauss_noise) == len(test_agent.lstm_layers) and \
                    len(test_agent.gauss_noise) > 0 and \
                    all([l > 0 for l in test_agent.gauss_noise])
                assert test_agent.regularizer == run_params['regularizer'] and \
                    type(test_agent.regularizer) is list and \
                    len(test_agent.regularizer) > 0 and \
                    all([l > 0 for l in test_agent.regularizer])

            # for both
            assert test_agent.activation == run_params['activation'] and \
                type(test_agent.activation) is str
            try:
                activations.get(test_agent.activation)
            except ValueError:
                pytest.fail('Bad activation function for TensorFlow Keras', pytrace=True)

            assert test_agent.out_activation == run_params['out_activation'] and \
                type(test_agent.out_activation) is str
            try:
                activations.get(test_agent.out_activation)
            except ValueError:
                pytest.fail('Bad activation function for TensorFlow Keras', pytrace=True)

            assert test_agent.optimizer == run_params['optimizer'] and \
                type(test_agent.optimizer) is str
            try:
                optimizers.get(test_agent.optimizer)
            except ValueError:
                pytest.fail('Bad optimizer for TensorFlow Keras', pytrace=True)

            assert test_agent.loss == run_params['loss'] and \
                type(test_agent.loss) is str
            try:
                losses.get(test_agent.loss)
            except ValueError:
                pytest.fail('Bad loss function for TensorFlow Keras', pytrace=True)

            assert test_agent.clipnorm == run_params['clipnorm']
            assert test_agent.clipvalue == run_params['clipvalue']

            assert test_agent.memory.maxlen == 1000

        except ValueError:
            pytest.fail('Invalid Arguments in model.compile() for optimizer, loss, or metrics', pytrace=True)
        except:
            pytest.fail("Bad DQN()", pytrace=True)

    #3: test set_learner() for agent
    def test_set_learner(self):

        try:
            test_agent.set_learner()
            assert test_agent.is_learner == True
        except ValueError:
            pytest.fail('Invalid argumensts for optimizer, loss, or metrics in compile()', pytrace=True)

    #4: test remember() for agent
    def test_remember(self):

        current_state = test_agent.env.reset()
        total_reward = 0
        next_state = 0
        action = 0
        done = 0
        reward = 0

        memory = (current_state, action, reward, next_state, done, total_reward)
        try:
            test_agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
            assert test_agent.memory[-1][1] == action
            assert test_agent.memory[-1][2] == reward
            assert test_agent.memory[-1][3] == next_state
            assert test_agent.memory[-1][4] == done
            assert all([a == b for a,b in zip(test_agent.memory[-1][0],current_state)])
        except:
            pytest.fail("Bad remember()", pytrace=True)

    #5: test get_weights() for agent
    def test_get_weights(self):
        assert test_agent.get_weights() is not None

    #6: test set_weight() for agent
    def test_set_weights(self):

        test_agent_comm = mpi_settings.agent_comm
        test_target_weights = test_agent.get_weights()
        test_current_weights = test_agent_comm.bcast(test_target_weights, root=0)

        try:
            test_agent.set_weights(test_current_weights)
        except:
            pytest.fail("Bad set_weights()", pytrace=True)

    #7: test action() for agent
    def test_action(self):

        try:
            action, policy = test_agent.action(test_agent.env.reset())
            assert action >= 0
            assert policy in [0,1]
        except:
            pytest.fail("Bad action()", pytrace=True)

    #8: test generate_data() for agent
    def test_generate_data(self):

        global test_batch_state, test_batch_target
        try:
            test_batch_state, test_batch_target = next(test_agent.generate_data())
            #assert len(test_batch_state) == 0 #test_agent.batch_size
            #assert len(test_batch_target) == test_agent.batch_size
        except:
            pytest.fail("Bad generate_data()", pytrace=True)

    #9: test train() for agent
    def test_train(self):

        try:
            test_agent.train([test_batch_state, test_batch_target])
            assert test_agent.epsilon > test_agent.epsilon_min
        except RuntimeError:
            pytest.fail('Model fit() failed. Model never compiled, or model.fit is wrapped in tf.function',pytrace=True)
        except ValueError:
            pytest.fail('Mismatch between input data and expected data',pytrace=True)

    #10: test target_train() for agent
    def test_target_train(self):

        try:
            test_agent.target_train()
        except:
            pytest.fail('Incorrect target weights update')

    #11: test load() for agents
    def test_load(self):

        #checking if abstractmethod load() is in agent (DQN) class
        try:
            method = getattr(test_agent, 'load')
            assert callable(method)
        except AttributeError:
            pytest.fail('Must implement abstractmethod load()',pytrace=True)


    #12: test save() for agents
    def test_save(self):
        #checking if abstractmethod save() is in agent (DQN) class
        try:
            method = getattr(test_agent, 'save')
            assert callable(method)
        except AttributeError:
            pytest.fail('Must implement abstractmethod save()',pytrace=True)

    #13 test update() for agents
    def test_update(self):
        #checking if abstractmethod update() is in agent (DQN) class
        try:
            method = getattr(test_agent, 'update')
            assert callable(method)
        except AttributeError:
            pytest.fail('Must implement abstractmethod update()',pytrace=True)

    #14 test monitor() for agents
    def test_monitor(self):
        #checking if abstractmethod monitor() is in agent (DQN) class
        try:
            method = getattr(test_agent, 'monitor')
            assert callable(method)
        except AttributeError:
            pytest.fail('Must implement abstractmethod monitor()',pytrace=True)
