import gym
import subprocess
import os
import logging
import json
import math
import sys
import lmfit
import random
import numpy as np
import pandas as pd
# import plotly.graph_objects as go
# from mpi4py import MPI
from gym import spaces
from shutil import copyfile, rmtree
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from datetime import datetime
import ast
# from plotly.subplots import make_subplots
# from utils.utils_tdlg.sv import *
import exarl as erl
# from utils.tdlg_plot import *
from importlib import reload
from mpi4py import MPI

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BlockCoPolymerTDLG-Logger')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.ERROR)


class BlockCoPolymerTDLGv3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()
        """
        Description:
           Environment used to run the TDLG 3D CH model

        Model source code:
           https://github.com/exalearn/TDLG.git

        Model execusion procedure:
           The initial settings for the model are defined in the param.in provide in the cfg/tdgl.json file
           The initialization and reset will be run using the JustWise=1 to account for any new geometry and compute changes
           All other steps will be performed using JustWise=0

        Observation states map directly to the model input parameters that are being changed by the actions
           - Current kappa
           - Normalized 1D FFT histogram

        Action space is discrete:
           - Increase/decrease kappa (2)
           - No change (1)
        """

        # Application setup
        self.app_dir  = kwargs['app_dir']
        sys.path.append(self.app_dir)
        import TDLG as TDLG
        self.app = TDLG

        self.param_dir  = './envs/env_vault/env_cfg/'
        self.param_name = 'tdlg_param.in'
        self.param_file = os.path.join(self.param_dir, self.param_name)
        self.model_parameter_init  = self._updateParamDict(self.param_file)
        self.model_parameter = self.model_parameter_init.copy()

        self.target_structure_name = 'target_field.out'
        self.target_structure_file = os.path.join(self.param_dir, self.target_structure_name)
        self.target_structure = self.setTargetStructure(self.target_structure_file)
        self.structure_len = len(self.target_structure)
        self.observation_space = spaces.Box(
            low=np.append(
                np.zeros(
                    self.structure_len),
                [0.004]),
            high=np.append(
                np.ones(
                    self.structure_len) *
                350,
                [0.012]),
            dtype=np.float32)
        self.app_threads  = 4

        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.param_file = os.path.join(self.param_dir, self.param_name)
        self.model_parameter_init  = self._updateParamDict(self.param_file)
        self.model_parameter = self.model_parameter_init.copy()

        self.target_structure_file = os.path.join(self.param_dir, self.target_structure_name)
        self.target_structure = self.setTargetStructure(self.target_structure_file)

        self.structure_len = len(self.target_structure)
        self.observation_space = spaces.Box(
            low=np.append(
                np.zeros(
                    self.structure_len),
                [0.004]),
            high=np.append(
                np.ones(
                    self.structure_len) *
                350,
                [0.012]),
            dtype=np.float32)

        # start with a random kappa
        # range_top = int(round((self.observation_space.high[-1]-self.observation_space.low[-1])/self.kappa_step_size + 1))
        # random.seed(datetime.now())
        # r = random.randint(0,range_top-1)
        # self.model_parameter['kappa'] = self.observation_space.low[-1]+self.kappa_step_size*r
        # self.model_parameter['kappa'] = self.model_parameter['kappa']

        # start with a static kappa
        if self.f_fixInit:
            self.model_parameter['kappa'] = self.fixInitValue

        print("Init. kappa: ", self.model_parameter['kappa'])
        # print("Target precision value: ", (-1+self.smape_shift)+self.target_precision)

        # Clear state and environment
        self.Phi = self.app.InitRandomField(0.5, self._getParametersStruct(self.model_parameter))
        field = self._getFieldFromPhi(self.Phi)
        self.current_structure, self.current_vol = self._get1DFFT(field)

        self.state = np.zeros(self.observation_space.shape)
        self.state = self._getState(self.current_structure)

        self.st = 0

        return self.state

    def _getParametersStruct(self, paramDict):
        paramStruct = self.app.ParametersStruct(
            time=paramDict['time'],
            dx=paramDict['dx'],
            dy=paramDict['dy'],
            dz=paramDict['dz'],
            NX=paramDict['NX'],
            NY=paramDict['NY'],
            NZ=paramDict['NZ'],
            Ntotal=paramDict['NX'] * paramDict['NY'] * paramDict['NZ'],
            dt=paramDict['dt'],
            seed=paramDict['seed'],
            threads=self.app_threads,
            a=paramDict['a'],
            b=paramDict['b'],
            c=paramDict['c'],
            kappa=paramDict['kappa'],
            fracA=paramDict['fracA'],
            noise=paramDict['noise'])
        return paramStruct

    def set_results_dir(self, results_dir):
        self.worker_dir = results_dir + 'multiworker/worker' + str(MPI.COMM_WORLD.rank)
        if os.path.exists(self.worker_dir):
            rmtree(self.worker_dir)
        os.makedirs(self.worker_dir)
        os.makedirs(self.worker_dir + '/archive/')

    def _inMemory(self, state, action):
        inMemDict = {'inMem': False}
        filename = 'tmp-bcp.log'
        if os.path.exists(filename):
            if os.stat(filename).st_size > 0:
                data = pd.read_csv(filename, header=None, delim_whitespace=True)
                data.reset_index(inplace=True, drop=True)
                tdata = data[data[1] == action]
                if len(tdata) > 0:
                    tdata = tdata[tdata[0] == state]
                    tdata.reset_index(inplace=True, drop=True)
                    if len(tdata[3]) > 0:
                        next_state = (tdata[3][0])
                        reward = float(tdata[2][0])
                        done = bool(tdata[5][0])
                        next_state = ast.literal_eval(next_state)
                        inMemDict = {'inMem': True, 'next_state': next_state, 'reward': reward, 'done': done}
        return inMemDict

    def _getState(self, s):
        # print (s)
        k = self.model_parameter['kappa']
        state = np.append(s, k)
        return state

    def step(self, action):
        # Default returns
        done   = False
        f_outOfBound = False

        # Check if the state-action has already been calculated
        # inMem = self._inMemory(str(self._getState(self.current_structure)),action)

        # Apply discrete actions
        if action == 2:
            self.model_parameter['kappa'] += self.kappa_step_size
            if self.model_parameter['kappa'] > self.observation_space.high[-1]:
                self.model_parameter['kappa'] -= self.kappa_step_size
                f_outOfBound = True

        elif action == 0:
            self.model_parameter['kappa'] -= self.kappa_step_size
            if self.model_parameter['kappa'] < self.observation_space.low[-1]:
                self.model_parameter['kappa'] += self.kappa_step_size
                f_outOfBound = True

        # Run model
        self._run_TDLG()

        # Get FFT from model output
        field = self._getFieldFromPhi(self.Phi)
        self.current_structure, self.current_vol = self._get1DFFT(field)

        # Calculate reward
        reward = self._calculateReward()

        self.st += 1

        if abs(round(reward, 6)) == 0:
            done = True
        elif f_outOfBound:
            done = True
            reward -= self._max_episode_steps
        if self._max_episode_steps == self.st:
            done = True
            reward -= 0

        # Return output
        return self._getState(self.current_structure), reward, done, {}

    def setTargetStructure(self, file_name):
        """
        Description: Read in a 3D volume file based on the TDLG output structure
        """
        field = self._readFieldFromFile(file_name)
        structure, self.target_vol = self._get1DFFT(field)
        print("Target Strcuture: ", structure)
        return structure

    def _readFieldFromFile(self, input_file_name):
        # Read in model output file ##
        with open(input_file_name, 'rb') as file:
            # read a list of lines into data
            field = file.readlines()
        return field

    def _getFieldFromPhi(self, Phi):
        Parameters = self._getParametersStruct(self.model_parameter)
        field = self._getField(Phi, Parameters)
        return field

    def _get1DFFT(self, fieldData):
        """
        Description: Process 3D data from model output using Kevin's code
        """
        data = self._getPaddedVol(fieldData)

        # Define expectations for simulations results
        scale = 1  # Conversion of simulation units into realspace units
        d0_expected = 7.5
        q0_expected = 2 * np.pi / d0_expected

        # Compute the structure vector
        vector = self.structure_vector(data, q0_expected, scale=scale)
        return vector[-1], data

    def _getPaddedVol(self, filedata):
        """
        Description: Read the model output and convert to readable format for Kevin's software
        """

        # TODO: Add try-exception if these parameters are not defined in the parameter input file
        # nshapshots = int(float(self.model_parameter['time'])/float(self.model_parameter['samp_freq']))
        nshapshots = 1
        nx = int(self.model_parameter['NX'])
        ny = int(self.model_parameter['NY'])
        nz = int(self.model_parameter['NZ'])

        # Zero padd to satisfy Kevin's code
        nbox = nx if nx > ny else ny
        nbox = nbox if nbox > nz else nz
        padded_vol = np.zeros([nbox, nbox, nbox])

        # Check if file is empty
        if len(filedata) == 0:
            logger.warning('### Empty filedata ###')
            return padded_vol

        # Set non to 0
        filedata = [0 if math.isnan(float(v)) else v for v in filedata]

        # TODO: Make sure the file is not empyty or has nan values
        # Parse volumes
        vols = []
        last_vol = np.zeros([nx, ny, nz])
        for ss in range(nshapshots):
            vol = np.zeros([nx, ny, nz])
            i = 0
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        vol[x][y][z] = float(filedata[i])
                        i += 1
            vols.append(vol)
            last_vol = vol

        np.copyto(padded_vol[:nx, :ny, :nz], vols[-1])
        return padded_vol

    def _calculateReward(self):
        """
        Description: Calculate the reduced chi^2 between the target structure and the current structure in 1D FFTW space
        """
        # print ("current structure: ",  self.current_structure)
        # print ("target structure: ",  self.target_structure)

        chi2 = 0
        smape = 0
        nvalues = len(self.target_structure)
        if (nvalues == len(self.current_structure)):
            for i in range(nvalues):
                chi2 += abs(self.target_structure[i] - self.current_structure[i])**2
                abs_diff = abs(self.target_structure[i] - self.current_structure[i])
                sum = self.target_structure[i] + self.current_structure[i]
                smape += abs_diff / sum
        chi2 = chi2 / nvalues if nvalues > 0 else -1
        smape = smape / nvalues if nvalues > 0 else -1

        # reward = -smape*100.0 ## yeilds a value between [-100,0]
        reward = -smape  # +self.smape_shift  ##smape yiels [-1, 0] reward, use the shift to adjust the range
        # print('reward: {}'.format(reward))
        # reward = math.sqrt(chi2)
        # reward = chi2
        # reward = - similaritymeasures.frechet_dist(self.target_structure, self.current_structure)

        target_peaks, tpp  = find_peaks(self.target_structure, height=50)
        current_peaks, cpp = find_peaks(self.current_structure, height=50)
        # print ("target_peaks:",target_peaks,tpp['peak_heights'])
        # print ("current_peaks:",current_peaks,cpp['peak_heights'])

        target_peaks_width  = peak_widths(self.target_structure, target_peaks, rel_height=0.5)
        current_peaks_width = peak_widths(self.current_structure, current_peaks, rel_height=0.5)

        return reward

    def _updateParamDict(self, param_file):
        """
        Description:
           - Reads parameter file and make dict
           - Save the dict in self.model_parameter
        """
        model_parameter = defaultdict(float)
        # Read default parameter file
        filedata = []
        with open(param_file, 'r') as file:
            # read a list of lines into data
            filedata = file.readlines()
        # Make dict
        for param in filedata:
            key = param.split()[1]
            value = param.split()[0]
            if key == 'time' or key == 'NX' or key == 'NY' or key == 'NZ' or key == 'seed' or key == 'samp_freq' or key == 'RandomInit':
                value = int(value)
            elif key == 'fracA' or key == 'kappa':
                value = float(value)  # round(float(value),3)
            else:
                value = float(value)
            model_parameter[key] = value
        return model_parameter

    def _run_TDLG(self):
        """
        Description:
           - Run TDLG model
        """
        Parameters = self._getParametersStruct(self.model_parameter)

        # field=self._getField(self.Phi, Parameters)
        # print ("pre_run_field: ", field[:5])

        if self.app_core == 'cpu':
            self.app.TDLG_CPU(Parameters, self.Phi)
        elif self.app_core == 'gpu':
            self.app.TDLG_GPU(Parameters, self.Phi)

        # field=self._getField(self.Phi, Parameters)
        # print ("post_run_field: ", field[:5])

    def _getField(self, Phi, Parameters):
        NX = Parameters.NX
        NY = Parameters.NY
        NZ = Parameters.NZ
        field = []
        for z in range(0, NZ):
            for y in range(0, NY):
                for x in range(0, NX):
                    i = z + NZ * (y + NY * x)
                    field.append(Phi[i])
        return field

    def close(self):
        return 0

    def structure_vector(self, result, expected, range_rel=1.0, scale=1, adjust=None, plot=False, output_dir='./', output_name='result', output_condition=''):

        # Compute FFT
        result_fft = np.fft.fftn(result)

        # Recenter FFT (by default the origin is in the 'corners' but we want the origin in the center of the matrix)
        hq, wq, dq = result_fft.shape
        result_fft = np.concatenate((result_fft[int(hq / 2):, :, :], result_fft[0:int(hq / 2), :, :]), axis=0)
        result_fft = np.concatenate((result_fft[:, int(wq / 2):, :], result_fft[:, 0:int(wq / 2), :]), axis=1)
        result_fft = np.concatenate((result_fft[:, :, int(dq / 2):], result_fft[:, :, 0:int(dq / 2)]), axis=2)
        origin = [int(wq / 2), int(hq / 2), int(dq / 2)]
        qy_scale, qx_scale, qz_scale = 2 * np.pi / (scale * hq), 2 * np.pi / (scale * wq), 2 * np.pi / (scale * dq)

        data = np.absolute(result_fft)

        # Compute 1D curve by doing a circular average (about the origin)
        qs, data1D = self.circular_average(data, scale=[qy_scale, qz_scale, qz_scale], origin=origin)

        # Eliminate the first point (which is absurdly high due to FFT artifacts)
        qs = qs[1:]
        data1D = data1D[1:]

        # Optionally adjust the curve to improve data extraction
        if adjust is not None:
            data1D *= np.power(qs, adjust)

        # Modify the code to use the max as the peak estimation
        idx = np.where(data1D == np.max(data1D))[0][0]

        expected = qs[idx]

        # Fit the 1D curve to a Gaussian
        # lm_result, fit_line, fit_line_extended = self.peak_fit(qs, data1D, x_expected=expected, range_rel=range_rel)
        p = 0  # lm_result.params['prefactor'].value # Peak height (prefactor)
        q = 0  # lm_result.params['x_center'].value # Peak position (center) ==> use this
        sigma = 0  # lm_result.params['sigma'].value # Peak width (stdev) ==> use this
        I = 0  # p*sigma*np.sqrt(2*np.pi) # Integrated peak area
        m = 0  # lm_result.params['m'].value # Baseline slope
        b = 0  # lm_result.params['b'].value # Baseline intercept
        return p, q, sigma, I, m, b, qs, data1D

    def circular_average(self, data, scale=[1, 1, 1], origin=None, bins_relative=3.0):

        h, w, d = data.shape
        y_scale, x_scale, z_scale = scale
        if origin is None:
            x0, y0, z0 = int(w / 2), int(h / 2), int(d / 2)
        else:
            x0, y0, z0 = origin

        # Compute map of distances to the origin
        x = (np.arange(w) - x0) * x_scale
        y = (np.arange(h) - y0) * y_scale
        z = (np.arange(d) - y0) * z_scale
        X, Y, Z = np.meshgrid(x, y, z)
        R = np.sqrt(X**2 + Y**2 + X**2)

        # Compute histogram
        data = data.ravel()
        R = R.ravel()

        scale = (x_scale + y_scale + z_scale) / 2.0
        r_range = [0, np.max(R)]
        num_bins = int(bins_relative * abs(r_range[1] - r_range[0]) / scale)
        num_per_bin, rbins = np.histogram(R, bins=num_bins, range=r_range)
        idx = np.where(num_per_bin != 0)  # Bins that actually have data

        r_vals, rbins = np.histogram(R, bins=num_bins, range=r_range, weights=R)
        r_vals = r_vals[idx] / num_per_bin[idx]
        I_vals, rbins = np.histogram(R, bins=num_bins, range=r_range, weights=data)
        I_vals = I_vals[idx] / num_per_bin[idx]

        return r_vals, I_vals
