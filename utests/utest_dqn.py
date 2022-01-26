import tensorflow as tf
import sys
import numpy as np
import exarl as erl
import pytest
import exarl.utils.candleDriver as cd
from exarl.base.comm_base import ExaComm

from tensorflow.keras.layers import Dense, GaussianNoise, BatchNormalization, LSTM
from tensorflow.python.client import device_lib
from tensorflow.keras import optimizers, activations, losses
from exarl.agents.agent_vault.dqn import DQN
from exarl.utils.candleDriver import initialize_parameters
from exarl.network.simple_comm import ExaSimple
MPI = ExaSimple.MPI


class TestClass:

    # initialize a test_agent
    def __init_test_agent(self):
        global test_agent
        global test_learner
        try:
            test_learner = erl.ExaLearner(comm)
            test_agent = DQN(test_learner.env, True)  # run_params).env)
        except TypeError:
            pytest.fail('Abstract class methods not handled correctly', pytrace=True)
        except:
            pytest.fail('Bad Agent Implementation', pytrace=True)

        return test_agent

    # look into model layers Attribute
    # TODO: add functionality to check for any type of layers attribute
    def _peek_layers_attribute(self):
        types = (LSTM, GaussianNoise, BatchNormalization, Dense)

        if self.device == 'CPU':

            lstms = len(test_agent.lstm_layers)
            gauss_noise = len(test_agent.gauss_noise)

            for layer in test_agent.target_model.layers:
                if isinstance(layer, types) is False:
                    return False

                if isinstance(layer, types[0]):
                    lstms -= 1
                elif isinstance(layer, types[1]):
                    gauss_noise -= 1

            return lstms == 0 and gauss_noise == 0

        elif self.device == 'GPU':

            dense = len(test_agent.dense)

            for layer in test_agent.model.layers:
                if isinstance(layer, types) is False:
                    return False
                if isinstance(layer, types[3]):
                    dense -= 1

            return dense == 0

        else:
            return False

    # 1: test MPI init
    def test_initialize_parameters(self):
        global comm  # run_params
        try:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            # run_params = initialize_parameters()
            # assert type(run_params) is dict
        except:
            pytest.fail('Bad MPI comm', pytrace=True)
            # pytest.fail('Bad initialize_parameters()',pytrace=True)

    # 2: test agent __init__ for DQN agent
    def test_init(self):

        try:
            self.__init_test_agent()

            assert test_agent.results_dir == cd.run_params['output_dir']
            assert test_agent.gamma == cd.run_params['gamma'] and \
                0 < test_agent.gamma < 1 and \
                isinstance(test_agent.gamma, float) is True
            assert test_agent.epsilon == cd.run_params['epsilon'] and \
                0 < test_agent.epsilon > test_agent.epsilon_min and \
                isinstance(test_agent.epsilon, float) is True
            assert test_agent.epsilon_min == cd.run_params['epsilon_min'] and \
                test_agent.epsilon_min > 0 and \
                isinstance(test_agent.epsilon_min, float) is True
            assert test_agent.epsilon_decay == cd.run_params['epsilon_decay'] and \
                0 < test_agent.epsilon_decay < 1 and \
                isinstance(test_agent.epsilon_decay, float) is True
            assert test_agent.learning_rate == cd.run_params['learning_rate'] and \
                test_agent.learning_rate > 0 and \
                isinstance(test_agent.learning_rate, float) is True
            assert test_agent.batch_size == cd.run_params['batch_size'] and \
                test_agent.batch_size > 0 and \
                test_agent.maxlen % test_agent.batch_size == 0 and \
                isinstance(test_agent.batch_size, int) is True
            assert test_agent.tau == cd.run_params['tau'] and \
                0 < test_agent.tau < 1 and \
                isinstance(test_agent.tau, float) is True
            assert test_agent.model_type == cd.run_params['model_type'] and \
                test_agent.model_type.upper() in ("LSTM", "MLP")

            # for mlp
            if test_agent.model_type.upper() == "MLP":
                assert test_agent.dense == cd.run_params['dense'] and \
                    isinstance(test_agent.dense, list) is True and \
                    len(test_agent.dense) > 0 and \
                    all([(l > 0 and isinstance(l, int)) for l in test_agent.dense])

            # for lstm
            if test_agent.model_type.upper() == "LSTM":
                assert test_agent.lstm_layers == cd.run_params['lstm_layers'] and \
                    isinstance(test_agent.lstm_layers, list) is True and \
                    len(test_agent.lstm_layers) > 0 and \
                    all([(l > 0 and isinstance(l, int)) for l in test_agent.lstm_layers])

                assert test_agent.gauss_noise == cd.run_params['gauss_noise'] and \
                    isinstance(test_agent.gauss_noise, list) is True and \
                    len(test_agent.gauss_noise) == len(test_agent.lstm_layers) and \
                    len(test_agent.gauss_noise) > 0 and \
                    all([(l > 0 and isinstance(l, float)) for l in test_agent.gauss_noise])

                assert test_agent.regularizer == cd.run_params['regularizer'] and \
                    isinstance(test_agent.regularizer, list) is True and \
                    len(test_agent.regularizer) > 0 and \
                    all([(0 < l < 1 and isinstance(l, float)) for l in test_agent.regularizer])

            # for both
            assert test_agent.activation == cd.run_params['activation'] and \
                isinstance(test_agent.activation, str) is True
            try:
                # check if it is a valid activation
                activations.get(test_agent.activation)
            except ValueError:
                pytest.fail('Bad activation function for TensorFlow Keras', pytrace=True)

            assert test_agent.out_activation == cd.run_params['out_activation'] and \
                isinstance(test_agent.out_activation, str) is True
            try:
                # check if it is a valid activation
                activations.get(test_agent.out_activation)
            except ValueError:
                pytest.fail('Bad activation function for TensorFlow Keras', pytrace=True)

            assert test_agent.optimizer == cd.run_params['optimizer'] and \
                isinstance(test_agent.optimizer, str) is True
            try:
                # check if it is a valid optimizer
                optimizers.get(test_agent.optimizer)
            except ValueError:
                pytest.fail('Bad optimizer for TensorFlow Keras', pytrace=True)

            assert test_agent.loss == cd.run_params['loss'] and \
                isinstance(test_agent.loss, str) is True
            try:
                # check if it is a valid loss
                losses.get(test_agent.loss)
            except ValueError:
                pytest.fail('Bad loss function for TensorFlow Keras', pytrace=True)

            # Currently unused
            # assert test_agent.clipnorm == cd.run_params['clipnorm'] and \
            #     isinstance(test_agent.clipnorm, float) is True
            # assert test_agent.clipvalue == cd.run_params['clipvalue'] and \
            #     isinstance(test_agent.clipvalue, float) is True

            assert test_agent.maxlen == cd.run_params['mem_length']

            # test model.compile()
            # gpu_names = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
            # if len(gpu_names) > 0:
            #     self.device = 'GPU'
            #     assert self._peek_layers_attribute() is True

            # on device /CPU:0
            # self.device = 'CPU'
            # assert isinstance(test_agent.target_model.layers, list) is True and \
            #     self._peek_layers_attribute() is True

        except ValueError:
            pytest.fail('Invalid Arguments in model.compile() for optimizer, loss, or metrics', pytrace=True)
        except:
            pytest.fail("Bad DQN()", pytrace=True)
            sys.exit()

    # 3: test set_learner() for agent
    def test_set_learner(self):

        try:
            test_agent.set_learner()
            assert test_agent.is_learner is True

        except ValueError:
            pytest.fail('Invalid argumensts for optimizer, loss, or metrics in compile()', pytrace=True)

    # 4: test remember() for agent
    def test_remember(self):

        current_state = test_agent.env.reset()
        total_reward = 0
        next_state = test_agent.env.reset()
        action = 0
        done = 0
        reward = 0

        memory = (current_state, action, reward, next_state, done, total_reward)
        try:
            test_agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
            minibatch, _, _ = test_agent.replay_buffer.sample(test_agent.batch_size, test_agent.priority_scale)
            assert minibatch[-1][1] == action
            assert minibatch[-1][2] == reward
            assert minibatch[-1][4] == done
            assert all([a == b for a, b in zip(minibatch[-1][0], current_state)])
            assert all([a == b for a, b in zip(minibatch[-1][3], next_state)])
        except:
            pytest.fail("Bad remember()", pytrace=True)

    # 5: test get_weights() for agent
    def test_get_weights(self):
        assert test_agent.get_weights() is not None

    # 6: test set_weight() for agent
    def test_set_weights(self):

        test_agent_comm = ExaComm.agent_comm
        test_target_weights = test_agent.get_weights()
        test_current_weights = test_agent_comm.bcast(test_target_weights, root=0)

        try:
            test_agent.set_weights(test_current_weights)
            assert np.array_equal(test_current_weights[0], test_agent.get_weights()[0]) is True
        except:
            pytest.fail("Bad set_weights()", pytrace=True)

    # 7: test action() for agent
    # TODO: Add testcase to verify difference between before and after an action
    def test_action(self):

        try:
            action, policy = test_agent.action(test_agent.env.reset())
            assert action >= 0
            assert policy in [0, 1]
        except:
            pytest.fail("Bad action()", pytrace=True)

    # 8: test generate_data() for agent
    # TODO: Need to generalize for a custom agent?
    def test_generate_data(self):

        # global batch  # test_batch_state, test_batch_target
        try:
            [test_agent.remember(test_agent.env.reset(), 0, 0, test_agent.env.reset(), 0) for _ in range(test_agent.maxlen)]
            batch1 = next(test_agent.generate_data())
            assert isinstance(batch1, tuple) is True
            batch2 = next(test_agent.generate_data())
            assert isinstance(batch2, tuple) is True
            if type(batch1[0]).__module__ == np.__name__ and type(batch2[0]).__module__ == np.__name__:
                assert np.array_equal(batch1[0], batch2[0]) is False
        except:
            pytest.fail("Bad generate_data()", pytrace=True)

    # 9: test model.fit() in train() for agent
    def test_train(self):

        try:

            with tf.device(test_agent.device):
                test_agent.train(next(test_agent.generate_data()))
                epsilon1 = test_agent.epsilon
                test_agent.train(next(test_agent.generate_data()))
                epsilon2 = test_agent.epsilon

            assert epsilon1 > test_agent.epsilon_min and \
                epsilon2 > test_agent.epsilon_min and \
                epsilon2 <= epsilon1

            # for h1, h2 in zip(history1.history.values(), history2.history.values()):
            #     if isinstance(h1, list) and isinstance(h2, list):
            #         assert all([a != b for a, b in zip(h1, h2)])
            #     else:
            #         assert h1 != h2

        except RuntimeError:
            pytest.fail('Model fit() failed. Model never compiled, or model.fit is wrapped in tf.function', pytrace=True)
        except ValueError:
            pytest.fail('Mismatch between input data and expected data', pytrace=True)
        except:
            pytest.fail('Bad train()', pytrace=True)

    # 10: test target_train() for agent
    def test_target_train(self):

        try:
            initial_weights = test_agent.get_weights()
            test_agent.target_train()
            changed_weights = test_agent.get_weights()
            for w1, w2 in zip(initial_weights, changed_weights):
                if type(w1).__module__ == np.__name__ and type(w2).__module__ == np.__name__:
                    assert np.array_equal(w1, w2) is False
                elif isinstance(w1, list) and isinstance(w2, list):
                    assert any(a != b for a, b in zip(w1, w2))
                else:
                    assert w1 != w2
        except:
            pytest.fail('Incorrect target weights update', pytrace=True)

    # 11: test load() for agents
    def test_load(self):

        # checking if abstractmethod load() is in agent (DQN) class
        try:
            method = getattr(test_agent, 'load')
            assert callable(method)
        except AttributeError:
            pytest.fail('Must implement abstractmethod load()', pytrace=True)

    # 12: test save() for agents
    def test_save(self):

        # checking if abstractmethod save() is in agent (DQN) class
        try:
            method = getattr(test_agent, 'save')
            assert callable(method)
        except AttributeError:
            pytest.fail('Must implement abstractmethod save()', pytrace=True)
