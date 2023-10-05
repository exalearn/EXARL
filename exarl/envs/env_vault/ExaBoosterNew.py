
import os
import sys
import errno
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
import tensorflow as tf
from tensorflow import keras

from exarl.utils.globals import ExaGlobals

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)
np.seterr(divide='ignore', invalid='ignore')

def load_reformated_cvs(filename, nrows=100000):
    df = pd.read_csv(filename, nrows=nrows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)
    return df

def create_dataset(dataset, look_back=10 * 15, look_forward=1):
    X, Y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), 0]
        yy = dataset[(i + look_back):(i + offset), 0]
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)


def get_dataset(df, variable='B:VIMIN'):
    dataset = df[variable].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    train_size = int(len(dataset) * 0.70)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    X_train, Y_train = create_dataset(train, look_back=15)  # 3/25 needed to change to replicate results of the paper #15)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))
    return X_train, Y_train  # scaler, X_test, Y_test


def all_inplace_scale(df):
    scale_dict = {}

    for var in ['B:VIMIN', 'B:IMINER', 'B_VIMIN', 'B:LINFRQ', 'I:IB', 'I:MDAT40']:  # TODO: make dynamic
        our_data2 = df
        trace = our_data2[var].astype('float32')
        data = np.array(trace)
        # print(data)
        median = np.median(data)
        upper_quartile = np.percentile(data, 75)
        lower_quartile = np.percentile(data, 25)
        # print(median, upper_quartile, lower_quartile)
        iqr = upper_quartile - lower_quartile
        lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()
        upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
        ranged = upper_whisker - lower_whisker
        # (value − median) / (upper - lower)
        our_data2[var] = 1 / ranged * (data - median)

        scale_dict[str(var)] = {"median": median, "range": ranged}

    return scale_dict

def unscale(var_name, tseries, scale_dict):
    # equivalent to inverse transform
    from_model = np.asarray(tseries)
    update = from_model * scale_dict[str(var_name)]["range"] + scale_dict[str(var_name)]["median"]

    return(update)

def rescale(var_name, tseries, scale_dict):
    # equivalent to transform
    data = np.asarray(tseries)
    update = 1 / scale_dict[str(var_name)]["range"] * (data - scale_dict[str(var_name)]["median"])
    return(update)

def create_dropout_predict_model(model, dropout):

    # Load the config of the original model
    conf = model.get_config()

    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"] == "Dropout":
            layer["config"]["rate"] = dropout

    # Create a new model with specified dropout
    model_dropout = keras.Model.from_config(conf)
    model_dropout.set_weights(model.get_weights())
    return model_dropout

def regulation(alpha, gamma, error, min_set, beta):
    # calculate the prediction with current regulation rules
    ER = error  # error
    _MIN = min_set  # setting
    for i in range(len(_MIN)):
        if i > 0:
            beta_t = beta[-1] + gamma * ER[i]
            beta.append(beta_t)  # hopefully this will update self.rachael_beta in place
    MIN_pred = _MIN - alpha * ER - np.asarray(beta[-15:]).reshape(15, 1)  # predict the next, shiftting happens in the plotting #check here
    # used to be 15, now 150
    return MIN_pred

class ExaBooster_v2(gym.Env):
    def __init__(self):

        self.save_dir = os.getcwd()  # './'
        self.episodes = 0
        self.steps = 0
        self.max_steps = 100
        self.total_reward = 0
        self.data_total_reward = 0
        self.diff = 0

        self.rachael_reward = 0
        self.rachael_beta = [0]  # unclear if needed... depends on whether the regulation should be allowed to build continuously

        try:
            booster_data_dir = ExaGlobals.lookup_params('booster_data_dir')
        except:
            sys.exit("Must set booster_data_dir")
        booster_dir = ExaGlobals.lookup_params('model_dir')
        if booster_dir == 'None':
            self.file_dir = os.path.dirname(__file__)
            booster_dir = os.path.join(self.file_dir, 'env_data/booster_data')
        logger.info('booster related directory: '.format(booster_dir))
        try:
            os.mkdir(booster_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                logger.error("Creation of the directory %s failed" % booster_dir)
        else:
            logger.error("Successfully created the directory %s " % booster_dir)
        booster_model_file = ExaGlobals.lookup_params('model_file')
        booster_model_pfn = os.path.join(booster_dir, booster_model_file)
        print("booster model file=", booster_model_pfn, flush=True)
        with tf.device('/cpu:0'):
            self.booster_model = keras.models.load_model(booster_model_pfn)

        # Check if data is available
        # booster_data_file = 'BOOSTR.csv'
        booster_data_file = ExaGlobals.lookup_params('data_file')
        booster_file_pfn = os.path.join(booster_data_dir, booster_data_file)
        logger.info('Booster data file pfn:{}'.format(booster_file_pfn))
        if not os.path.exists(booster_file_pfn):
            logger.info('No cached file. Downloading...')
            try:
                # url = 'https://zenodo.org/record/4088982/files/data%20release.csv?download=1'
                url = ExaGlobals.lookup_params('url')
                r = requests.get(url, allow_redirects=True)
                open(booster_file_pfn, 'wb').write(r.content)
            except:
                logger.error("Problem downloading file")
        else:
            logger.info('Using exiting cached file')

        self.booster_model = create_dropout_predict_model(self.booster_model, 0)  # calibrated on 3/02/2021: .2

        # Load scalers
        # Load data to initialize the env
        # filename = 'data_release.csv'  # 'decomposed_all.csv' #no longer want decomposed data
        # data = dp.load_reformated_cvs('../data/' + filename, nrows=250000)
        data = load_reformated_cvs(booster_file_pfn, nrows=250000)
        scale_dict = all_inplace_scale(data)
        data['B:VIMIN'] = data['B:VIMIN'].shift(-1)
        data = data.set_index(pd.to_datetime(data.time))
        data = data.dropna()
        data = data.drop_duplicates()
        self.variables = ['B:VIMIN', 'B:IMINER', 'B_VIMIN', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
        self.nvariables = len(self.variables)
        logger.info('Number of variables:{}'.format(self.nvariables))
        self.scale_dict = scale_dict
        # self.scalers = []
        data_list = []
        x_train = []

        # get_dataset also normalizes the data
        for v in range(len(self.variables)):
            data_list.append(get_dataset(data, variable=self.variables[v]))
            # self.scalers.append(data_list[v][2]) # comment out for new scaling
            x_train.append(data_list[v][0])

        # Axis
        self.concate_axis = 1
        self.X_train = np.concatenate(x_train, axis=self.concate_axis)

        self.nbatches = self.X_train.shape[0]
        self.nsamples = self.X_train.shape[2]
        self.batch_id = self.episodes + 4200
        self.data_state = None

        print('Data shape:{}'.format(self.X_train.shape))
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.nvariables,),
            dtype=np.float64
        )

        # DYNAMIC ACTION SPACE SIZING
        data['B:VIMIN_DIFF'] = data['B:VIMIN'] - data['B:VIMIN'].shift(-1, fill_value=0)
        self.nactions = 15  # set here
        # Discrete
        # self.action_space = spaces.Discrete(self.nactions)
        # Continuous
        self.action_space = spaces.Box(low=np.percentile(data['B:VIMIN_DIFF'], 25),
                                       high=np.percentile(data['B:VIMIN_DIFF'], 75),
                                       shape=(1,),
                                       dtype=np.float32)
        self.actionMap_VIMIN = []
        for i in range(1, self.nactions + 1):
            self.actionMap_VIMIN.append(data['B:VIMIN_DIFF'].quantile(i / (self.nactions + 1)))

        self.VIMIN = 0
        self.state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        # self.B_VIMIN_state = np.zeros(shape= (1, 1, self.nsamples)) ### shouldn't normally need this
        self.predicted_state = np.zeros(shape=(1, self.nvariables, 1))

        self.rachael_state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        self.rachael_predicted_state = np.zeros(shape=(1, self.nvariables, 1))

        logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        logger.debug('Episode/State: {}/{}'.format(self.episodes, self.steps))
        done = False

        # Steps:
        # 1) Update VIMIN based on action
        # 2) Predict booster variables
        # 3) Predict next step for injector
        # 4) Shift state with new values

        # Step 1: Calculate the new B:VIMIN based on policy action
        logger.debug('Step() before action VIMIN:{}'.format(self.VIMIN))
        # Dicrete
        # delta_VIMIN = self.actionMap_VIMIN[int(action)]
        # Continuous
        delta_VIMIN = action
        DENORN_BVIMIN = unscale(self.variables[0], np.array([self.VIMIN]).reshape(1, -1), self.scale_dict)
        DENORN_BVIMIN += delta_VIMIN
        logger.debug('Step() descaled VIMIN:{}'.format(DENORN_BVIMIN))
        logger.debug('Action:{}'.format(delta_VIMIN))

        # Rachael's Eq as an action
        alpha = 10e-2
        gamma = 7.535e-5

        B_VIMIN_trace = unscale(self.variables[2], self.state[0, 2, :].reshape(-1, 1), self.scale_dict)
        BIMINER_trace = unscale(self.variables[1], self.state[0, 1, :].reshape(-1, 1), self.scale_dict)

        self.rachael_state[0][0][self.nsamples - 1] = rescale(self.variables[0],
                                                              regulation(alpha,
                                                                         gamma,
                                                                         error=BIMINER_trace,
                                                                         min_set=B_VIMIN_trace,
                                                                         beta=self.rachael_beta)[-1].reshape(-1, 1),
                                                              self.scale_dict)
        self.VIMIN = rescale(self.variables[0], DENORN_BVIMIN, self.scale_dict)

        logger.debug('Step() updated VIMIN:{}'.format(self.VIMIN))
        self.state[0][0][self.nsamples - 1] = self.VIMIN

        # Step 2: Predict using booster model
        self.predicted_state = self.booster_model.predict(self.state)
        self.predicted_state = self.predicted_state.reshape(1, 2, 1)
        # used to be 3 in the center #TODO: make dynamic, changing back. now should be 2

        # Rachael's equation
        self.rachael_predicted_state = self.booster_model.predict(self.rachael_state)
        self.rachael_predicted_state = self.rachael_predicted_state.reshape(1, 2, 1)

        # Step 4: Shift state by one step
        self.state[0, :, 0:-1] = self.state[0, :, 1:]  # shift forward
        self.rachael_state[0, :, 0:-1] = self.rachael_state[0, :, 1:]

        # Update IMINER
        self.state[0][1][self.nsamples - 1] = self.predicted_state[0, 1:2]
        self.rachael_state[0][1][self.nsamples - 1] = self.rachael_predicted_state[0, 1:2]

        # Update data state for rendering
        self.data_state = np.copy(self.X_train[self.batch_id + self.steps].reshape(1, self.nvariables, self.nsamples))
        data_iminer = unscale(self.variables[1], self.data_state[0][1][self.nsamples - 1].reshape(1, -1), self.scale_dict)

        # where's data_vimin
        data_reward = -abs(data_iminer)

        # Use data for everything but the B:IMINER prediction
        self.state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]
        self.rachael_state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]

        iminer = self.predicted_state[0, 1]
        logger.debug('norm iminer:{}'.format(iminer))
        iminer = unscale(self.variables[1], np.array([iminer]), self.scale_dict).reshape(1, -1)

        logger.debug('iminer:{}'.format(iminer))

        # Reward
        reward = -abs(iminer)

        # update rachael state for rendering
        rach_reward = -abs(unscale(self.variables[1], np.array([self.rachael_predicted_state[0, 1]]).reshape(1, -1), self.scale_dict))

        if self.steps >= int(self.max_steps):
            done = True

        self.diff += np.asscalar(abs(data_iminer - iminer))
        self.data_total_reward += np.asscalar(data_reward)
        self.total_reward += np.asscalar(reward)
        self.rachael_reward += np.asscalar(rach_reward)

        if self.episodes % 100 == 0:  # so over this rendering...
            self.render()
        reward_list = [np.asscalar(reward), np.asscalar(rach_reward), np.asscalar(data_reward)]
        return self.state[0, :, -1:].flatten(), reward_list[0], done, False, {}

    def reset(self):
        self.episodes += 1
        self.steps = 0
        self.data_total_reward = 0
        self.total_reward = 0
        self.diff = 0
        self.rachael_reward = 0
        self.rachael_beta = [0]

        # Prepare the random sample ##
        # self.batch_id = 10
        self.batch_id = self.episodes + 4200
        logger.info('Resetting env')
        # self.state = np.zeros(shape=(1,5,150))
        logger.debug('self.state:{}'.format(self.state))
        self.state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))

        self.data_state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))

        self.rachael_state = None
        self.rachael_state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))

        logger.debug('self.state:{}'.format(self.state))
        logger.debug('reset_data.shape:{}'.format(self.state.shape))
        self.VIMIN = self.state[0, 0, -1:]
        logger.debug('Normed VIMIN:{}'.format(self.VIMIN))
        logger.debug('B:VIMIN:{}'.format(unscale(self.variables[0], np.array([self.VIMIN]), self.scale_dict).reshape(1, -1)))

        return self.state[0, :, -1:].flatten(), {}

    def render(self):
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.labelweight'] = 'regular'
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 16

        logger.debug('render()')

        import seaborn as sns
        sns.set_style("ticks")
        nvars = 2  # len(self.variables)> we just want B:VIMIN and B:IMINER
        fig, axs = plt.subplots(nvars, figsize=(12, 8))
        logger.debug('self.state:{}'.format(self.state))

        # Rachael's Eq
        alpha = 10e-2
        gamma = 7.535e-5
        # try dstate
        BVIMIN_trace = unscale(self.variables[0], self.state[0, 0, 1:-1].reshape(-1, 1), self.scale_dict)
        BIMINER_trace = unscale(self.variables[1], self.state[0, 1, :].reshape(-1, 1), self.scale_dict)

        B_VIMIN_trace = unscale(self.variables[2], self.state[0, 2, :].reshape(-1, 1), self.scale_dict)

        # something is weird with this change... it definitely is predicting 180 value which isn't right
        BVIMIN_pred = unscale(self.variables[0], self.rachael_state[0, 0, :].reshape(-1, 1), self.scale_dict)
        rachael_IMINER = unscale(self.variables[1], self.rachael_state[0, 1, :].reshape(-1, 1), self.scale_dict)

        for v in range(0, nvars):
            utrace = self.state[0, v, :]
            trace = unscale(self.variables[v], utrace.reshape(-1, 1), self.scale_dict)
            if v == 0:
                # soemthing seems weird... might need to actually track it above
                axs[v].set_title('Raw data reward: {:.2f} - RL agent reward: {:.2f} - PID Eq reward {:.2f}'.format(self.data_total_reward,
                                                                                                                   self.total_reward, self.rachael_reward))

            axs[v].plot(trace, label='RL Policy', color='black')

            # if v==1:
            data_utrace = self.data_state[0, v, :]
            data_trace = unscale(self.variables[v], data_utrace.reshape(-1, 1), self.scale_dict)

            if v == 1:
                x = np.linspace(0, 14, 15)  # np.linspace(0, 14, 15) #np.linspace(0, 149, 150) #TODO: change this so that it is dynamic for lookback
                axs[v].fill_between(x, -data_trace.flatten(), +data_trace.flatten(), alpha=0.2, color='red')

            axs[v].plot(data_trace, 'r--', label='Data')
            # axs[v].plot()
            axs[v].set_xlabel('time')
            axs[v].set_ylabel('{}'.format(self.variables[v]))
            # axs[v].legend(loc='upper left')

        # replaced np.linspace(0,14,15)
        axs[0].plot(np.linspace(0, 14, 15), BVIMIN_pred, label="PID Eq", color='blue', linestyle='dotted')
        axs[0].legend(loc='upper left')
        axs[1].plot(np.linspace(0, 14, 15), rachael_IMINER, label="PID Eq", color='blue', linestyle='dotted')
        axs[1].legend(loc='upper left')

        plt.savefig(ExaGlobals.lookup_params('output_dir') + 'episode{}_step{}_v1.png'.format(self.episodes, self.steps))
        plt.clf()

        fig, axs = plt.subplots(1, figsize=(12, 12))

        Y_agent_bvimin = unscale(self.variables[0], self.state[0][0].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        Y_agent_biminer = unscale(self.variables[1], self.state[0][1].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        Y_data_bvimin = unscale(self.variables[0], self.data_state[0][0].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        Y_data_biminer = unscale(self.variables[1], self.data_state[0][1].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        Y_rachael_bvimin = unscale(self.variables[0], self.rachael_state[0][0].reshape(-1, 1), self.scale_dict).reshape(-1, 1)
        Y_rachael_iminer = unscale(self.variables[1], self.rachael_state[0][1].reshape(-1, 1), self.scale_dict).reshape(-1, 1)

        np_predict = np.concatenate((Y_data_bvimin, Y_data_biminer, Y_agent_bvimin, Y_agent_biminer,
                                     Y_rachael_bvimin, Y_rachael_iminer), axis=self.concate_axis)
        df_cool = pd.DataFrame(np_predict, columns=['bvimin_data', 'biminer_data', 'bvimin_agent', 'biminer_agent', 'bvimin_rachael', 'biminer_rachael'])

        plt.scatter(Y_data_bvimin, Y_data_biminer, color='red', alpha=0.5, label='Data')
        plt.scatter(Y_agent_bvimin, Y_agent_biminer, color='black', alpha=0.5, label='RL Policy')
        plt.scatter(Y_rachael_bvimin, Y_rachael_iminer, color='blue', alpha=0.5, label='PID Eq')
        plt.xlabel('B:VIMIN')
        plt.ylabel('B:IMINER')
        plt.legend()
        plt.savefig(ExaGlobals.lookup_params('output_dir') + '/corr_episode{}_step{}.png'.format(self.episodes, self.steps))
        plt.close('all')
