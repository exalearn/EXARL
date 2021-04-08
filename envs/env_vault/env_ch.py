import datetime as dt
import os
import sys
import time
import math
import numpy as np
from collections import namedtuple

import gym
from gym import spaces

from exarl.comm_base import ExaComm

import image_structure

import ch2d.cahnhilliard as ch
import ch2d.aligned_vector as av

import utils.candleDriver as cd

sys.path.append('envs/env_vault/CahnHilliard2D/cpp/python')
sys.path.append('envs/env_vault/ImageStructure')

gym.logger.set_level(40)


def print_status(
        msg,
        *args,
        comm_rank=None,
        showtime=True,
        barrier=True,
        allranks=False,
        flush=True):

    if comm_rank is None:
        comm_rank = 0

    if showtime:
        s = dt.datetime.now().strftime('[%H:%M:%S.%f] ')
    else:
        s = ''
    if allranks:
        s += '{' + str(comm_rank) + '} '
    else:
        if comm_rank > 0:
            return

    s += msg.format(*args)
    print(s, flush=flush)


# Environment class

class CahnHilliardEnv(gym.Env):

    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # Declare hyper-parameters, initialized for determining datatype
        super().__init__()

        self.debug = cd.run_params['debug']              # 0
        self.change_T = cd.run_params['changeT']         # 0.1
        self.initT = cd.run_params['initT']              # 0.5
        self.targetT = cd.run_params['targetT']          # 0.5
        self.notTrain = cd.run_params['notTrain']        # False
        self.output_dir = cd.run_params['output_dir']
        self.target_dir = cd.run_params['target_dir']    # './data/ch/'
        self.target_file = cd.run_params['target_file']  # 'target.out'
        self.notPlotRL = cd.run_params['notPlotRL']      # False
        self.length = cd.run_params['length']            # 100
        self.genTarget = cd.run_params['genTarget']      # True
        self.randInitial = cd.run_params['randInitial']  # False
        self.steps = cd.run_params['n_steps']
        # self.episodes        = 0

        # self.args = args
        self.comm = ExaComm.global_comm      # mpi_settings.env_comm
        # self.comm.Get_rank() if self.comm else 0
        self.comm_rank = ExaComm.global_comm.size

        # These are problem dependent and must be available during environment
        # object creation time: cannot be set by CANDLE
        self.size_struct_vec = 200
        self.num_control_params = 1

        # spaces from gym
        self.action_space = spaces.Discrete(self.getActionSize())

        # TODO: fix the high values later
        #       since I do not know the maximum values
        self.observation_space = spaces.Box(
            low=np.append(
                np.zeros(
                    self.getStateSize() - 1),
                [0.000]),
            high=np.append(
                np.ones(
                    self.getStateSize() - 1),
                [1000]),
            dtype=np.float32)

        # stores current structure vector
        self.currStructVec = np.zeros(self.size_struct_vec)
        # target  structure vector (loaded or generated at setTargetState()
        # once)
        self.targetStructVec = np.zeros(self.size_struct_vec)
        self.targetStructVecNorm = 1.
        self.reference_norm_function = lambda x: np.linalg.norm(x, ord=np.inf)

        self.vecWeight = np.zeros(self.size_struct_vec)
        self.rangeStructVec = np.zeros(self.size_struct_vec)

        self.maxStructVec = [-math.inf for _ in range(self.size_struct_vec)]
        self.minStructVec = [math.inf for _ in range(self.size_struct_vec)]

        self.hasBaseScore = False

        # timers
        self.time_getNextState = 0
        self.time_getReward = 0
        self.time_isTerminalState = 0

        self.startAnalyze = False
        self.episode = -1
        self.time_step = -1
        self.isTest = True if self.notTrain else False
        self.isTarget = False

        self.setTargetState()

    '''
    def set_env(self):
        # Obtain hyperparameteres
        env_data = super().get_config()

        self.setTargetState()
        self.setInitSimParams()
    '''

    # get state space (mean, sd)
    # state size is # of structured vector components and the current
    # temperature
    def getStateSize(self):
        return self.size_struct_vec + self.num_control_params

    # get action space space (3^N)

    def getActionSize(self):
        return 3  # This is fixed for now

    # set the initial temperature randomly
    def setRandInitT(self):
        return np.random.uniform(self.chparams.T_min, self.chparams.T_max)

    # reset to the initial state
    def reset(self):

        self.episode += 1
        self.time_step = -1

        # self.setTargetState()   TODO: this is not efficient
        self.setInitSimParams()
        # TODO: I do not have to initialze all parameter at each episode

        if self.randInitial:
            self.T = self.setRandInitT()
        else:
            self.T = self.initT
        # initial value, always start from the same initial value for now.

        if self.debug >= 10:
            print_status(
                'Initial T: {}'.format(
                    self.T),
                comm_rank=self.comm_rank)

        self.setControlParams(T=self.T)

        # TODO: Do I need to call this function here
        #       considering not calculating a structure vector
        # t0 = time.time()
        # ch.run_ch_solver(self.chparams, self.info)
        # print("Time ch_solver: ",time.time()-t0)

        # No need to calculate initial structure vector
        # self.currStructVec = self.getImgStruct("C_1.out", 1)  get the current structure vector
        # if self.debug>=-1: print("Init struct vec: ", np.around(self.currStructVec, 2))

        state = []
        state = np.append([self.T], self.currStructVec)
        return state

    # set the target state
    def setTargetState(self):

        # Disabled this feature due to the error related to setTargetState
        # if (self.genTarget):   generate target
        #     self.generateTargetState()

        # print(self.target_dir + self.target_file)

        # given target
        target_sv_file = os.path.join(self.target_dir, self.target_file)
        self.targetStructVec = np.genfromtxt(target_sv_file)[1]

        if self.debug >= 10:
            print("Target Structured Vector: ",
                  np.around(self.targetStructVec, 2))

    # generate target state
    def generateTargetState(self):

        self.isTarget = True

        self.setInitSimParams()
        self.setControlParams(T=self.targetT)

        self.chparams.T_const = av.aligned_double_vector(
            self.targetT * np.ones(int(self.info.nx)**2))

        # create a target directory
        # out_dir = os.path.join(self.target_dir)
        # if not os.path.exists(out_dir):
        #     os.mkdir(out_dir)

        for i in range(self.steps):
            self.info.t0 = self.t[i]
            self.info.tf = self.t[i + 1]
            if self.debug >= 10:
                print('t0 = ', self.t[i] / self.lin_dt, ' dt_lin , tf = ',
                      self.t[i + 1] / self.lin_dt, ' dt_lin')

            if self.debug >= 1:
                t0 = time.time()
            residual = ch.run_ch_solver(self.chparams, self.info)
            residual /= np.linalg.norm(np.array(self.info.x), ord=2)
            if self.debug >= 1:
                print_status(
                    "Time ch_solver: {}, solver relative residual: {:.6f}".format(
                        time.time() - t0, residual), comm_rank=self.comm_rank)

            target_data = np.array(self.info.x)
            self.targetStructVec = self.getImgStruct(target_data, i)
            self.targetStructVecNorm = self.reference_norm_function(
                self.targetStructVec)

        if self.debug >= 10:
            print_status(
                "Target Structured Vector: ",
                np.around(
                    self.targetStructVec,
                    2),
                comm_rank=self.comm_rank,
                allranks=True)

        # self.targetStructVec = self.getImgStruct(target_data, self.steps)

        self.isTarget = False

    # get the next setp info
    # return the next state, reward, whether or not at the terimnal state

    def step(self, action_idx):

        self.time_step = self.time_step + 1
        # self.time_step = self.time_step % self.steps

        # get the next state
        if self.debug >= 1:
            time_tmp = time.time()
        self.currStructVec = self.getNextState(action_idx, self.time_step)
        state = []
        state = np.append([self.T], self.currStructVec)
        if self.debug >= 1:
            self.time_getNextState += time.time() - time_tmp

        # get reward
        if self.debug >= 1:
            time_tmp = time.time()
        reward = self.getReward(self.time_step)
        if self.debug >= 1:
            self.time_getReward += time.time() - time_tmp

        # check if done
        if self.debug >= 1:
            time_tmp = time.time()
        # deviation = self.distanceFromTerminalState()
        if self.debug >= 1:
            self.time_isTerminalState += time.time() - time_tmp

        items = dict()

        # print("Reward: ", reward)

        return state, reward, self.isTerminalState(), items

    # whether or not this state is the terminal state
    # if the difference of each component is less than 0.0001 of the range of each compoenent,
    # we consder that we reached the target
    def distanceFromTerminalState(self):
        return self.reference_norm_function(
            self.currStructVec - self.targetStructVec)

    # TODO: I need to give larger reward when it in the terminal state
    def isTerminalState(self, normdiff=None):
        # use defined norm. Don't recompute if previously computed
        if normdiff is None:
            normdiff = self.distanceFromTerminalState()
        if normdiff < 0.0001 * self.targetStructVecNorm:
            print_status(
                "!!!!!!!!!!!!!!!!!!!!!!!!! Terminal State Reached !!!!!!!!!!!!!!!!!!!!!!!!!",
                comm_rank=self.comm_rank,
                allranks=True)
            return True
        else:
            return False

    # reward function
    # TODO: modify this reward function
    def getReward(self, t):

        reward = 0.0

        for i in range(self.size_struct_vec):
            reward -= 1.0 / self.size_struct_vec * (self.currStructVec[i] - self.targetStructVec[i])**2

        return reward

    # get the next action

    def setNextAction(self, action_idx):

        if self.debug >= 30:
            print_status("action idex: {}".format(action_idx),
                         comm_rank=self.comm_rank, allranks=True)

        if self.T <= self.chparams.T_min + .05 and action_idx == 0:
            self.T = self.T
        elif self.T >= self.chparams.T_max - .05 and action_idx == 2:
            self.T = self.T
        else:
            # action_idx maps to action
            self.T += (action_idx - 1) * self.change_T

        if self.debug >= 30:
            print_status(
                "getNextAction T param: {}".format(
                    np.around(
                        self.T,
                        2)),
                comm_rank=self.comm_rank,
                allranks=True)
        # return self.T

    # return the next state after taking the action at the current state
    def getNextState(self, action_idx, i):

        t0 = 0  # initialize timer

        # self.T is updated to the next action
        self.setNextAction(action_idx)
        # set control parameter for the simulation code
        self.setControlParams(T=self.T)
        # print('i: ', i)
        self.info.t0 = self.t[i]
        self.info.tf = self.t[i + 1]

        if self.debug >= 10:
            print('t0 = ', self.t[i] / self.lin_dt, ' dt_lin , tf = ',
                  self.t[i + 1] / self.lin_dt, ' dt_lin')

        self.chparams.T_const = av.aligned_double_vector(
            self.T * np.ones(int(self.info.nx)**2))

        if self.debug >= 1:
            t0 = time.time()

        residual = ch.run_ch_solver(self.chparams, self.info)
        residual /= np.linalg.norm(np.array(self.info.x), ord=2)

        if self.debug >= 1:
            print_status(
                "Time ch_solver: {}, solver relative residual: {:.6f}".format(
                    time.time() - t0,
                    residual),
                comm_rank=self.comm_rank)

        if self.debug >= 10:
            print_status(
                "exit ch solver",
                comm_rank=self.comm_rank,
                allranks=True)

        img_data = np.array(self.info.x)

        # get image structure
        if self.debug >= 1:
            t0 = time.time()
        img_struct = self.getImgStruct(img_data, i + 2)
        if self.debug >= 0:
            print_status("Time getImgStruct: {}".format(time.time() - t0),
                         comm_rank=self.comm_rank)

        return img_struct

    # initialize parameters for the simulation
    # TODO: this function should be called once, not each episode

    def setInitSimParams(self):

        # ********* POLYMER PARAMETERS *********
        Xmin = 0.055
        Xmax = 0.5
        N = np.mean([200, 2000])
        L_repeat = (10**-9) * np.mean([20, 80])  # meters
        n_repeat = 15
        L_omega = n_repeat * L_repeat
        L_kuhn = (10**-9) * np.mean([0.5, 3.0])  # meters
        # Tmin = 0.1
        # Tmax = 1
        # T = 1.0
        # **************************************

        # *********** INPUTS ***********
        self.info = ch.SimInfo()

        if self.isTest:
            self.info.outdir = os.path.join(
                self.output_dir, "rank_" + str(self.comm_rank), 'test')
        else:
            self.info.outdir = os.path.join(
                self.output_dir, "rank_" + str(self.comm_rank), 'episode_' + str(self.episode))

        if self.isTarget:
            self.info.outdir = self.target_dir

        self.info.t0 = 0.0
        self.info.nx = 128
        self.info.ny = 128
        self.info.dx = 1. / self.info.nx
        self.info.dy = 1. / self.info.ny
        self.info.bc = 'neumann'
        self.info.rhs_type = 'ch_thermal_no_diffusion'

        # Set up grid for spatial-field quantities
        nx = int(self.info.nx)
        xx, yy = np.meshgrid(np.arange(0, 1, 1 / self.info.nx),
                             np.arange(0, 1, 1 / self.info.nx))

        self.chparams = ch.CHparamsVector(self.info.nx, self.info.ny)

        self.chparams.b = av.aligned_double_vector(1.0 * np.ones(nx**2))
        self.chparams.u = av.aligned_double_vector(1.0 * np.ones(nx**2))
        self.chparams.m = av.aligned_double_vector(0.15 * np.ones(nx**2))
        self.chparams.sigma_noise = 0.0
        self.chparams.eps2_min = 0.0
        self.chparams.eps2_max = 1.0
        self.chparams.sigma_min = 0.0
        self.chparams.sigma_max = 1.0e10
        self.chparams.T_min = 0.1
        self.chparams.T_max = 1.0
        self.chparams.T_const = av.aligned_double_vector(
            0.5 * (self.chparams.T_max + self.chparams.T_min) * np.ones(nx**2))
        self.chparams.L_kuhn = L_kuhn
        self.chparams.N = N
        self.chparams.L_omega = L_omega
        self.chparams.X_min = Xmin
        self.chparams.X_max = Xmax

        self.chparams.compute_and_set_eps2_and_sigma_from_polymer_params(
            0.5 * (self.chparams.T_max + self.chparams.T_min), self.info)
        # ******************************

        # Define timescales
        self.biharm_dt = (self.info.dx**4) / np.max(self.chparams.eps_2)
        self.diff_dt = (self.info.dx**2) / np.max([np.max(self.chparams.u),
                                                   np.max(self.chparams.b)])
        self.lin_dt = 1.0 / np.max(self.chparams.sigma)

        # Setup checkpointing in time
        n_dt = self.length  # 2000
        # TODO: THIS DOES NOT WORK ANYMORE if the setTargetState is called in
        # the constructor!
        n_tsteps = self.steps  # self._max_episode_steps
        self.info.t0 = 0
        self.info.iter = 0
        stiff_dt = np.min([self.biharm_dt, self.diff_dt, self.lin_dt])
        self.t = np.linspace(
            self.info.t0,
            self.info.t0 + n_dt * stiff_dt,
            n_tsteps + 1)
        # print('self.t:', self.t)
        dt_check = self.t[1] - self.t[0]

        # Run solver
        if self.debug >= 1:
            print('Biharmonic timescale dt_biharm = ', self.biharm_dt)
            print('Diffusion timescale dt_diff = ', self.diff_dt, ' = ',
                  self.diff_dt / self.biharm_dt, ' dt_biharm')
            print('Linear timescale dt_lin = ', self.lin_dt, ' = ',
                  self.lin_dt / self.biharm_dt, ' dt_biharm')
            print('Sampling interval = ', dt_check / stiff_dt, ' dt_stiff')

    # set the controlling parameters
    def setControlParams(self, T):
        T = self.chparams.T_min if T < self.chparams.T_min else T
        T = self.chparams.T_max if T > self.chparams.T_max else T
        self.chparams.T_const = av.aligned_double_vector(
            T * np.ones(int(self.info.nx)**2))

    # select one structured vector generator
    def getImgStruct(self, data, t):
        if self.size_struct_vec == 6:
            return self.getImgStruct6(data, t)
        elif self.size_struct_vec == 2:
            return self.getImgStruct2(data, t)
        elif self.size_struct_vec == 200:
            return self.getFullCircAvgFFT(
                data, t, interpolation_abscissa=np.linspace(
                    0, 2, self.size_struct_vec))

    # get Anthony's structured vector
    def getImgStruct2(self, data, t):

        dimensions = 2
        structure_function = 'fourier'

        # Named-tuple for handling input options
        Inputs = namedtuple(
            'Inputs',
            'data dimensions structure_function')  # output_file
        # datafiletype = datafile.split('.')[-1]
        # datafile     = os.path.join(self.output_dir , datafile)
        inputs = Inputs(data, dimensions, structure_function)  # , outfile

        # Compute structure function
        structure_analysis = ImageStructure(inputs)
        structure_metrics = structure_analysis.compute_structure(
            plot_metrics=True if not self.notPlotRL else False)
        # , outdir=outdir, str_figure=""

        # Get structured vector
        try:
            results = np.array([structure_metrics[0], structure_metrics[1]])
        except BaseException:
            results = structure_metrics
        # if self.debug>=30: print("Struct Vec: ", results)

        return results

    # get Kevin's structured vector
    def getImgStruct6(self, data, t):

        # datafile = os.path.join(self.output_dir , datafile)

        # Define expectations for simulations results
        w, h = self.info.nx, self.info.nx
        x_scale, y_scale = 1, 1  # Conversion of simulation units into realspace units

        # Load simulation output
        # frame   = load_result_file(datafile, w, h)
        frame = np.reshape(data, [w, h])
        results = structure_vector(frame, scale=[x_scale, y_scale],
                                   plot=True if not self.notPlotRL else False,
                                   output_condition=str(t))

        return results

    # get Kevin's full circ-avg'd fft
    def getFullCircAvgFFT(self, data, t, interpolation_abscissa=None):

        N = self.info.nx
        M = self.info.ny

        # Named-tuple for handling input options
        Inputs = namedtuple(
            'Inputs',
            'data dimensions structure_function output_file nx ny')

        # datafile = os.path.join(self.output_dir , datafile)
        # w        = np.genfromtxt( datafile )
        # Ci       = w.reshape([N,M],order='C');

        inputs = Inputs(data, 2, 'fourier_yager_full', os.path.join(
            self.output_dir, 'structure_metrics_2d.dat'), N, M)

        if self.debug >= 50:
            print_status(data, comm_rank=self.comm_rank, allranks=True)
        structure_analysis = image_structure.src.ImageStructure.ImageStructure(
            inputs)

        out_dir = os.path.join(self.output_dir)
        if self.isTarget:
            out_dir = os.path.join(self.target_dir)

        x_ftt, y_fft, lm_result = structure_analysis.compute_structure(
            plot_metrics=False,
            outdir=out_dir,
            str_figure='circfft_' + str(t) + '_',
            interpolation_abscissa=interpolation_abscissa)
        # plot_metrics= True if not self.notPlotRL else False,

        return y_fft

    # plot target image
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None

    def set_results_dir(self, results_dir):
        return results_dir
