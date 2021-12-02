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
import gym
import sys
import os
import errno
import math
from gym import spaces
from gym.utils import seeding
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf

# import logging

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('RL-Logger')
# logger.setLevel(logging.INFO)
from exarl.utils import log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.lookup_params('log_level', [3, 3]))

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.70)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    X_train, Y_train = create_dataset(train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

    return scaler, X_train, Y_train


class ExaBooster_v1(gym.Env):
    def __init__(self):

        # Environment is based on the "BOOSTR: A Dataset for Accelerator Reinforcement Learning Control"
        # https://zenodo.org/record/4088982#.X4836kJKhTY

        try:
            booster_data_dir = cd.run_params['booster_data_dir']
        except:
            sys.exit("Must set booster_data_dir")
        booster_dir = cd.run_params['model_dir']
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

        # Load surrogate models
        self.cpus = tf.config.list_physical_devices('CPU')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        # booster_model_file = 'fullbooster_noshift_e250_bs99_nsteps250k_invar5_outvar3_axis1_mmscaler_t0_D10122020-T175237_kfold2__e16_vl0.00038.h5'
        booster_model_file = cd.run_params['model_file']
        booster_model_pfn = os.path.join(booster_dir, booster_model_file)
        print("booster model file=", booster_model_pfn, flush=True)
        with tf.device('/cpu:0'):
            self.booster_model = keras.models.load_model(booster_model_pfn)

        # Check if data is available
        # booster_data_file = 'BOOSTR.csv'
        booster_data_file = cd.run_params['data_file']
        booster_file_pfn = os.path.join(booster_data_dir, booster_data_file)
        logger.info('Booster data file pfn:{}'.format(booster_file_pfn))
        if not os.path.exists(booster_file_pfn):
            logger.info('No cached file. Downloading...')
            try:
                # url = 'https://zenodo.org/record/4088982/files/data%20release.csv?download=1'
                url = cd.run_params['url']
                r = requests.get(url, allow_redirects=True)
                open(booster_file_pfn, 'wb').write(r.content)
            except:
                logger.error("Problem downloading file")
        else:
            logger.info('Using exiting cached file')

        # Load data to initialize the env
        data = load_reformated_cvs(booster_file_pfn, nrows=250000)
        data['B:VIMIN'] = data['B:VIMIN'].shift(-1)
        data = data.set_index(pd.to_datetime(data.time))
        data = data.dropna()
        data = data.drop_duplicates()

        self.save_dir = cd.run_params['output_dir']
        self.episodes = 0
        self.steps = 0
        self.max_steps = cd.run_params['max_steps']
        self.total_reward = 0
        self.data_total_reward = 0
        self.total_iminer = 0
        self.data_total_iminer = 0
        self.diff = 0

        # Define boundary
        self.min_BIMIN = cd.run_params['min_BIMIN']
        self.max_BIMIN = cd.run_params['max_BIMIN']
        self.variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
        self.nvariables = len(self.variables)
        logger.info('Number of variables:{}'.format(self.nvariables))

        self.scalers = []
        data_list = []
        x_train = []
        # get_dataset also normalizes the data
        for v in range(len(self.variables)):
            data_list.append(get_dataset(data, variable=self.variables[v]))
            self.scalers.append(data_list[v][0])
            x_train.append(data_list[v][1])
        # Axis
        concate_axis = 1
        self.X_train = np.concatenate(x_train, axis=concate_axis)

        self.nbatches = self.X_train.shape[0]
        self.nsamples = self.X_train.shape[2]
        self.batch_id = 10  # np.random.randint(0, high=self.nbatches)
        self.data_state = None

        # print('Data shape:{}'.format(self.X_train.shape))
        self.observation_space = spaces.Box(
            low=0,
            high=+1,
            shape=(self.nvariables,),
            dtype=np.float64
        )

        # Dynamically allocate
        data['B:VIMIN_DIFF'] = data['B:VIMIN'] - data['B:VIMIN'].shift(-1)
        self.nactions = cd.run_params['nactions']
        self.action_space = spaces.Discrete(self.nactions)
        self.actionMap_VIMIN = []
        for i in range(1, self.nactions + 1):
            self.actionMap_VIMIN.append(data['B:VIMIN_DIFF'].quantile(i / (self.nactions + 1)))

        self.VIMIN = 0
        self.state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        self.predicted_state = np.zeros(shape=(1, self.nvariables, 1))
        logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))
        self.do_render = False  # True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        done = False

        # Steps:
        # 1) Update VIMIN based on action
        # 2) Predict booster dynamic variables
        # 3) Shift state with new values
        # 4) Update B:IMINER

        # Step 1: Calculate the new B:VINMIN based on policy action
        logger.info('Step() before action VIMIN:{}'.format(self.VIMIN))
        delta_VIMIN = self.actionMap_VIMIN[action]
        DENORN_BVIMIN = self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))
        DENORN_BVIMIN += delta_VIMIN
        logger.debug('Step() descaled VIMIN:{}'.format(DENORN_BVIMIN))
        if DENORN_BVIMIN < self.min_BIMIN or DENORN_BVIMIN > self.max_BIMIN:
            logger.info('Step() descaled VIMIN:{} is out of bounds.'.format(DENORN_BVIMIN))
            done = True

        self.VIMIN = self.scalers[0].transform(DENORN_BVIMIN)
        logger.debug('Step() updated VIMIN:{}'.format(self.VIMIN))
        self.state[0][0][self.nsamples - 1] = self.VIMIN

        # Step 2: Predict using booster model
        with tf.device('/cpu:0'):
            self.predicted_state = self.booster_model.predict(self.state)
        self.predicted_state = self.predicted_state.reshape(1, 3, 1)

        # Step 3: Shift state by one step
        self.state[0, :, 0:-1] = self.state[0, :, 1:]

        # Step 4: Update IMINER
        self.state[0][1][self.nsamples - 1] = self.predicted_state[0, 1:2]

        # Update data state for rendering
        self.data_state = np.copy(self.X_train[self.batch_id + self.steps].reshape(1, self.nvariables, self.nsamples))
        data_iminer = self.scalers[1].inverse_transform(self.data_state[0][1][self.nsamples - 1].reshape(1, -1))
        data_reward = -abs(data_iminer)
        # data_reward = np.array(1. * math.exp(-5 * abs(np.asscalar(data_iminer))))

        # Use data for everything but the B:IMINER prediction
        self.state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]

        iminer = self.predicted_state[0, 1]
        logger.debug('norm iminer:{}'.format(iminer))
        iminer = self.scalers[1].inverse_transform(np.array([iminer]).reshape(1, -1))
        logger.debug('iminer:{}'.format(iminer))

        # Reward
        reward = -abs(iminer)
        # reward = np.array(-1 + 1. * math.exp(-5 * abs(np.asscalar(iminer))))
        # reward = np.array(1. * math.exp(-5 * abs(np.asscalar(iminer))))

        if abs(iminer) >= 2:
            logger.info('iminer:{} is out of bounds'.format(iminer))
            done = True
            penalty = 5 * (self.max_steps - self.steps)
            reward -= penalty

        # if done:
        #     penalty = 5 * (self.max_steps - self.steps)
        #     logger.info('penalty:{} is out of bounds'.format(penalty))
        #     reward -= penalty

        if self.steps >= int(self.max_steps):
            done = True

        self.diff += np.asscalar(abs(data_iminer - iminer))
        self.data_total_reward += np.asscalar(data_reward)
        self.total_reward += np.asscalar(reward)
        self.total_iminer  += np.asscalar(abs(iminer))
        self.data_total_iminer += np.asscalar(abs(data_iminer))

        if self.do_render:
            self.render()

        return self.state[0, :, -1:].flatten(), np.asscalar(reward), done, {}

    def reset(self):
        self.episodes += 1
        self.steps = 0
        self.data_total_reward = 0
        self.total_reward = 0
        self.total_iminer = 0
        self.data_total_iminer = 0
        self.diff = 0
        self.data_state = None

        # Prepare the random sample ##
        self.batch_id = 10
        # self.batch_id = np.random.randint(0, high=self.nbatches)
        logger.info('Resetting env')
        # self.state = np.zeros(shape=(1,5,150))
        logger.debug('self.state:{}'.format(self.state))
        self.state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))
        logger.debug('self.state:{}'.format(self.state))
        logger.debug('reset_data.shape:{}'.format(self.state.shape))
        self.min_BIMIN = self.scalers[0].inverse_transform(self.state[:, 0, :]).min()
        self.max_BIMIN = self.scalers[0].inverse_transform(self.state[:, 0, :]).max()
        logger.info('Lower and upper B:VIMIN: [{},{}]'.format(self.min_BIMIN, self.max_BIMIN))
        # self.min_BIMIN = self.min_BIMIN * 0.9999
        # self.max_BIMIN = self.max_BIMIN * 1.0001
        self.VIMIN = self.state[0, 0, -1:]
        logger.debug('Normed VIMIN:{}'.format(self.VIMIN))
        logger.debug('B:VIMIN:{}'.format(self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))))
        return self.state[0, :, -1:].flatten()

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
        logger.debug('Save path:{}'.format(self.save_dir))
        render_dir = os.path.join(self.save_dir, 'render')
        logger.debug('Render path:{}'.format(render_dir))
        if not os.path.exists(render_dir):
            os.mkdir(render_dir)
        import seaborn as sns
        sns.set_style("ticks")
        nvars = 2  # len(self.variables)
        fig, axs = plt.subplots(nvars, figsize=(12, 8))
        logger.debug('self.state:{}'.format(self.state))
        for v in range(0, nvars):
            utrace = self.state[0, v, :]
            trace = self.scalers[v].inverse_transform(utrace.reshape(-1, 1))
            if v == 0:
                iminer_imp = 0
                if self.total_iminer > 0:
                    iminer_imp = self.data_total_iminer / self.total_iminer
                axs[v].set_title('Raw data reward: {:.2f} - RL agent reward: {:.2f} - Improvement: {:.2f} '.format(self.data_total_reward,
                                                                                                                   self.total_reward, iminer_imp))
            axs[v].plot(trace, label='Digital twin', color='black')

            # if v==1:
            data_utrace = self.data_state[0, v, :]
            data_trace = self.scalers[v].inverse_transform(data_utrace.reshape(-1, 1))
            if v == 1:
                x = np.linspace(0, 149, 150)
                axs[v].fill_between(x, -data_trace.flatten(), +data_trace.flatten(), alpha=0.2, color='red')
            axs[v].plot(data_trace, 'r--', label='Data')
            axs[v].set_xlabel('time')
            axs[v].set_ylabel('{}'.format(self.variables[v]))
            axs[v].legend(loc='upper left')

        plt.savefig(render_dir + '/episode{}_step{}_v1.png'.format(self.episodes, self.steps))
        plt.close('all')
