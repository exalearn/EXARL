import gym, subprocess, os, logging, json, math, sys, lmfit,random
import numpy as np
import pandas as pd 
#import plotly.graph_objects as go
#from mpi4py import MPI
from gym import spaces
from shutil import copyfile,rmtree
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from datetime import datetime
#from plotly.subplots import make_subplots
#from utils.utils_tdlg.sv import *
import exarl as erl
#from utils.tdlg_plot import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BlockCoPolymerTDLG-Logger')
#logger.setLevel(logging.INFO)
logger.setLevel(logging.ERROR)

class BlockCoPolymerTDLGv3(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, cfg_file='envs/env_vault/env_cfg/ExaLearnBlockCoPolymerTDLG-v3.json'):
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

        ## Application setup
        self.app_dir  = './envs/env_vault/LibTDLG/'
        sys.path.append(self.app_dir)
        import TDLG as TDLG
        self.app = TDLG
        self.app_threads  = 0
        self.app_core     = ''
        
        ## Model parameters
        self.param_dir  = './envs/env_vault/env_cfg/'
        self.param_name = 'tdlg_param.in'
        self.param_file = os.path.join(self.param_dir,self.param_name)
        self.model_parameter_init  = self._updateParamDict(self.param_file)
        ## Fix the starting point
        self.model_parameter = self.model_parameter_init
        
        ## Step sizes for discrete environment
        self.kappa_step_size = 0.001

        ## Setup target structure
        self.target_structure_name = 'envs/env_vault/env_cfg/target_field.out'
        self.target_precision = 0.0
        self.target_structure = self.setTargetStructure(self.target_structure_name)
        self.structure_len = len(self.target_structure)

        self.rendering = False


        ## Define state and action spaces
        self.observation_space = spaces.Box(low=np.append(np.zeros(self.structure_len),[0.004]), high=np.append(np.ones(self.structure_len)*350,[0.012]),dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        ## reward
        #self.earlyTargetBonus  = 2
        #self.outOfBoundPenalty = -1
        #self.smape_shift       = 0.5
        #self.f_rewardScaling   = True
        self.f_fixInit = True
        self.fixInitValue = 0.01
        
        self._max_episode_steps=10
        #self._max_episode_steps=self.spec.max_episode_steps
        #print('max steps: '.format( self._max_episode_steps))
        
        ## Initialize current structure
        #self.reset()


        self.ep=0
        self.st=0

    def set_env(self):
        
        env_data = super().get_config()
       
        print('In TDLG env_data is: ')
        print(env_data)

        self.app_dir                = env_data['app_dir']
        #self.app                    = env_data['app']
        self.param_dir              = env_data['param_dir']
        self.param_name             = env_data['param_name']
        self.target_structure_name  = env_data['target_structure_name']
        self.target_precision       = env_data['target_precision']
        self.kappa_step_size        = env_data['kappa_step_size']

        # Use the learner_defined results directory. 


        sys.path.append(self.app_dir)
        # only TDLG is valid, app_name is never used
        import TDLG as TDLG
        self.app = TDLG

        ## Model parameters
        self.param_file = os.path.join(self.param_dir,self.param_name)
        self.param_file = os.path.join(self.param_dir,self.param_name)
        self.model_parameter_init  = self._updateParamDict(self.param_file)
        ## Fix the starting point
        self.model_parameter = self.model_parameter_init

        ## Setup target structure
        self.target_structure = self.setTargetStructure(self.target_structure_name)
        self.structure_len = len(self.target_structure)

        ## Define state and action spaces
        self.observation_space = spaces.Box(low=np.append(np.zeros(self.structure_len),[0.004]), high=np.append(np.ones(self.structure_len),[0.012]),dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset()
        
        
    def reset(self):
        ## Clear parameter dict
        self.model_parameter = self.model_parameter_init.copy()
        
        ## start with a random kappa
        #range_top = int(round((self.observation_space.high[-1]-self.observation_space.low[-1])/self.kappa_step_size + 1))
        #random.seed(datetime.now())
        #r = random.randint(0,range_top-1)
        #self.model_parameter['kappa'] = self.observation_space.low[-1]+self.kappa_step_size*r
        #self.model_parameter['kappa'] = self.model_parameter['kappa']
        
        ## start with a static kappa
        if self.f_fixInit:
            self.model_parameter['kappa']=self.fixInitValue
        
        print("Init. kappa: ", self.model_parameter['kappa'])
        #print("Target precision value: ", (-1+self.smape_shift)+self.target_precision)
        
        # Clear state and environment
        self.Phi = self.app.InitRandomField(0.5,self._getParametersStruct(self.model_parameter))
        field = self._getFieldFromPhi(self.Phi)
        self.current_structure, self.current_vol = self._get1DFFT(field)
        
        self.state = np.zeros(self.observation_space.shape)
        self.state = self._getState(self.current_structure)

        ##
        self.st=0
    
        return self.state
    
    def _getParametersStruct(self,paramDict):
        paramStruct = self.app.ParametersStruct(
            time   = paramDict['time'],
            dx     = paramDict['dx'],
            dy     = paramDict['dy'],
            dz     = paramDict['dz'],
            NX     = paramDict['NX'],
            NY     = paramDict['NY'],
            NZ     = paramDict['NZ'],
            Ntotal = paramDict['NX']*paramDict['NY']*paramDict['NZ'],
            dt     = paramDict['dt'],
            seed   = paramDict['seed'],
            threads= self.app_threads,
            a      = paramDict['a'],
            b      = paramDict['b'],
            c      = paramDict['c'],
            kappa  = paramDict['kappa'],
            fracA  = paramDict['fracA'],
            noise  = paramDict['noise'])
        return paramStruct

    def set_results_dir(self,results_dir):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.worker_dir=results_dir+'multiworker/worker'+str(rank)
        if os.path.exists(self.worker_dir): rmtree(self.worker_dir)
        os.makedirs(self.worker_dir)
        os.makedirs(self.worker_dir+'/archive/')
        
    def _inMemory(self,state,action):
        ##
        inMemDict = {'inMem':False}
        ##
        filename='tmp-bcp.log'
        if os.path.exists(filename):
            if os.stat(filename).st_size> 0:
                data = pd.read_csv(filename, header=None, delim_whitespace=True)
                data.reset_index(inplace=True, drop=True)
                tdata = data[data[1]==action]
                if len(tdata)>0:
                    tdata = tdata[tdata[0]==state]
                    tdata.reset_index(inplace=True, drop=True)
                    if len(tdata[3])>0:
                        next_state = (tdata[3][0])
                        reward = float(tdata[2][0])
                        done = bool(tdata[5][0])
                        next_state = ast.literal_eval(next_state)
                        inMemDict = {'inMem':True,'next_state':next_state,'reward':reward,'done':done}                
        return inMemDict
    
    def _getState(self, s):
        #print (s)
        k = self.model_parameter['kappa']
        #s_normal = list(map(lambda x, r=float(self.observation_space.high[0]-self.observation_space.low[0]): ((x - self.observation_space.low[0]) / r), s))
        #s_normal = list(map(lambda x, r=float(350-0): ((x - 0) / r), s))
        ##s_normal = [n / 350.0 for n in s]
        #print (s_normal)
        #k_normal = list(map(lambda x, r=float(self.observation_space.high[-1]-self.observation_space.low[-1]): ((x - self.observation_space.low[-1]) / r), [k]))[0]
        ##k_normal = (k-self.observation_space.low[-1])/(self.observation_space.high[-1]-self.observation_space.low[-1])
        ##state = np.append(s_normal,k_normal)
        
        #state = np.append(s-self.target_structure,k)
        state = np.append(s,k)
        return state
    
    def step(self, action):
        ## Default returns
        done   = False
        f_outOfBound = False
        
        ## Check if the state-action has already been calculated
        #inMem = self._inMemory(str(self._getState(self.current_structure)),action)

        ## Apply discrete actions
        if action==2:
            self.model_parameter['kappa']+=self.kappa_step_size
            if self.model_parameter['kappa']>self.observation_space.high[-1]:
                self.model_parameter['kappa']-=self.kappa_step_size
                f_outOfBound = True
                
        elif action==0:
            self.model_parameter['kappa']-=self.kappa_step_size
            if self.model_parameter['kappa']<self.observation_space.low[-1]:
                self.model_parameter['kappa']+=self.kappa_step_size
                f_outOfBound = True
                
        #print('New kappa: {}'.format(self.model_parameter['kappa']))
        ## Avoid running model
        #if inMem['inMem']==True:
        #    logger.info('### In memory ... skipping simulation ###')
        #    print ('### In memory ... skipping simulation ###')
        #    return inMem['next_state'],float(inMem['reward']),inMem['done'], {}
        
        ## Run model
        self._run_TDLG()

        ## Get FFT from model output 
        field = self._getFieldFromPhi(self.Phi)
        self.current_structure, self.current_vol = self._get1DFFT(field)
        
        ## Calculate reward
        reward = self._calculateReward()
        #if action == 0: reward += 10
        ## Define done based on how close it's
        #if self.f_earlyTargetAchieve or self.f_outOfBound:
        #    done = True
        
        ##=========print and save plot============
        #print ("action: ", action)
        #print ("current structure: ", self.current_structure[:5]) 
        #print ("target structure: ", self.target_structure[:5])
        #print ("reward: ", reward) 
        #print ("kappa: ", self.model_parameter['kappa'])

        ## TODO: Move to the render ##
        #plt.plot(self.target_structure,label='target')
        #plt.plot(self.current_structure,label='current')
        #plt.legend()
        #plt.title("episode_"+str(self.ep)+"_step_"+str(self.st)+"\nreward "+str(round(reward,2)))
        #plt.savefig(self.plot_path+"/episode_"+str(self.ep)+"_step_"+str(self.st)+".png")
        #plt.close()
        
        #np.savetxt(self.field_path+"/episode_"+str(self.ep)+"_step_"+str(self.st)+"_field"+str(self.st)+".out",self.current_structure)
        #==========================================
    
        
        #self.total_reward += reward
        #if self.f_earlyTargetAchieve and self.best_reward < reward - self.earlyTargetBonus:
        #    self.best_reward = reward - self.earlyTargetBonus
        #elif self.f_outOfBound and self.best_reward < reward - self.outOfBoundPenalty :
        #    self.best_reward = reward - self.outOfBoundPenalty
        #elif self.best_reward < reward:
        #    self.best_reward = reward

        self.st +=1

        if abs(round(reward,6))==0:
            done=True
        elif f_outOfBound:
            done=True
            reward-=self._max_episode_steps
        if self._max_episode_steps==self.st:
            done=True
            reward-=0

        #print('max {} vs. step {}'.format(self._max_episode_steps,self.st))
        ## Rendering
        #if self.rendering:
        #    logger.info("Plotting...")
        #    self._render()

        ## Return output
        return self._getState(self.current_structure),reward,done, {}

    def setTargetStructure(self,file_name):
        """
        Description: Read in a 3D volume file based on the TDLG output structure
        """
        field = self._readFieldFromFile(file_name) 
        structure, self.target_vol = self._get1DFFT(field)
        print ("Target Strcuture: ", structure)
        return structure
        
    def _readFieldFromFile(self, input_file_name):
        ## Read in model output file ##
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
        scale = 1 # Conversion of simulation units into realspace units
        d0_expected = 7.5
        q0_expected = 2*np.pi/d0_expected

        # Compute the structure vector
        vector = self.structure_vector(data, q0_expected, scale=scale)
        return vector[-1], data
        
    def _getPaddedVol(self, filedata):
        """
        Description: Read the model output and convert to readable format for Kevin's software
        """
        
        ## TODO: Add try-exception if these parameters are not defined in the parameter input file
        #nshapshots = int(float(self.model_parameter['time'])/float(self.model_parameter['samp_freq']))
        nshapshots = 1
        nx = int(self.model_parameter['NX'])
        ny = int(self.model_parameter['NY'])
        nz = int(self.model_parameter['NZ'])

        ## Zero padd to satisfy Kevin's code
        nbox = nx if nx > ny else ny
        nbox = nbox if nbox > nz else nz
        padded_vol = np.zeros([nbox,nbox,nbox])

        ## Check if file is empty
        if len(filedata)==0:
            logger.warning('### Empty filedata ###')
            return padded_vol
        
        ## Set non to 0 
        filedata = [0 if math.isnan(float(v)) else v for v in filedata]
        
        ## TODO: Make sure the file is not empyty or has nan values
        ## Parse volumes ##
        vols = []
        last_vol = np.zeros([nx,ny,nz])
        for ss in range(nshapshots):
            vol = np.zeros([nx,ny,nz])
            i=0
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        vol[x][y][z] = float(filedata[i])
                        i+=1
            vols.append(vol)
            last_vol = vol

        np.copyto(padded_vol[:nx,:ny,:nz],vols[-1])
        return padded_vol

    
    def _calculateReward(self):
        """
        Description: Calculate the reduced chi^2 between the target structure and the current structure in 1D FFTW space
        """
        #print ("current structure: ",  self.current_structure)
        #print ("target structure: ",  self.target_structure)

        chi2=0
        smape=0
        nvalues=len(self.target_structure)
        if (nvalues==len(self.current_structure)):
            for i in range(nvalues):
                chi2 += abs(self.target_structure[i]-self.current_structure[i])**2
                abs_diff = abs(self.target_structure[i]-self.current_structure[i])
                sum = self.target_structure[i]+self.current_structure[i]
                smape += abs_diff/sum
        chi2 = chi2/nvalues if nvalues > 0 else -1
        smape = smape/nvalues if nvalues > 0 else -1
        
        #reward = -smape*100.0 ## yeilds a value between [-100,0]
        reward = -smape#+self.smape_shift  ##smape yiels [-1, 0] reward, use the shift to adjust the range
        #print('reward: {}'.format(reward))
        #reward = math.sqrt(chi2)
        #reward = chi2
        #reward = - similaritymeasures.frechet_dist(self.target_structure, self.current_structure)
            
        target_peaks, tpp  = find_peaks(self.target_structure , height=50)
        current_peaks, cpp = find_peaks(self.current_structure, height=50)
        #print ("target_peaks:",target_peaks,tpp['peak_heights'])
        #print ("current_peaks:",current_peaks,cpp['peak_heights'])
        
        target_peaks_width  = peak_widths(self.target_structure ,target_peaks, rel_height=0.5)
        current_peaks_width = peak_widths(self.current_structure,current_peaks,rel_height=0.5)
        #print ("target_peaks_width:",target_peaks_width[0])
        #print ("current_peaks_width:",current_peaks_width[0])
        
        #print ('reward scale: ', self.reward_scale)
        #if self.f_rewardScaling:
        #    #print ('raw reward: ',reward)
        #    reward = (reward+(1-self.smape_shift))/self.reward_scale-(1-self.smape_shift)
        #    #print ('scaled reward: ',reward)   
        #    
        #if reward>=round((-1+self.smape_shift)+self.target_precision,3):
        #    reward = self.earlyTargetBonus
        #    #reward += self.earlyTargetBonus
        #    print ("reach target early")
        #    self.f_earlyTargetAchieve = True
        #elif self.f_outOfBound:
        #    reward = self.outOfBoundPenalty
        #    #reward += self.outOfBoundPenalty  #still reward the closeness to the target from smape above, but panelize the out of boun#d
        #    print ("out of bound")
            
        #self.last_reward = reward
            
        return reward

    def _updateParamDict(self,param_file):
        """
        Description: 
           - Reads parameter file and make dict
           - Save the dict in self.model_parameter
        """
        model_parameter = defaultdict(float)
        ## Read default parameter file
        filedata = []
        with open(param_file, 'r') as file:
            # read a list of lines into data
            filedata = file.readlines()
        ## Make dict
        for param in filedata:
            key=param.split()[1]
            value=param.split()[0]
            if key=='time' or key=='NX' or key=='NY' or key=='NZ' or key=='seed' or key=='samp_freq' or key=='RandomInit':
                value=int(value)
            elif key=='fracA' or key=='kappa':
                value = float(value)#round(float(value),3)
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
        
        #field=self._getField(self.Phi, Parameters)
        #print ("pre_run_field: ", field[:5])
        
        if self.app_core == 'cpu':
            self.app.TDLG_CPU(Parameters,self.Phi)
        elif self.app_core == 'gpu':
            self.app.TDLG_GPU(Parameters,self.Phi)
        
        #field=self._getField(self.Phi, Parameters)
        #print ("post_run_field: ", field[:5])

        
    def _getField(self, Phi, Parameters):
        NX = Parameters.NX
        NY = Parameters.NY
        NZ = Parameters.NZ
        field=[]
        for z in range(0,NZ):
            for y in range(0,NY):
                for x in range(0,NX):
                    i = z + NZ * (y + NY * x)
                    field.append(Phi[i])
        return field

    def close(self):
        return 0
        
    def structure_vector(self,result, expected, range_rel=1.0, scale=1, adjust=None, plot=False, output_dir='./', output_name='result', output_condition=''):
    
        # Compute FFT
        result_fft = np.fft.fftn(result)
    
        # Recenter FFT (by default the origin is in the 'corners' but we want the origin in the center of the matrix)
        hq, wq, dq = result_fft.shape
        result_fft = np.concatenate( (result_fft[int(hq/2):,:,:], result_fft[0:int(hq/2),:,:]), axis=0 )
        result_fft = np.concatenate( (result_fft[:,int(wq/2):,:], result_fft[:,0:int(wq/2),:]), axis=1 )
        result_fft = np.concatenate( (result_fft[:,:,int(dq/2):], result_fft[:,:,0:int(dq/2)]), axis=2 )
        origin = [int(wq/2), int(hq/2), int(dq/2)]
        qy_scale, qx_scale, qz_scale = 2*np.pi/(scale*hq), 2*np.pi/(scale*wq), 2*np.pi/(scale*dq)

        data = np.absolute(result_fft)
    
        # Compute 1D curve by doing a circular average (about the origin)
        qs, data1D = self.circular_average(data, scale=[qy_scale, qz_scale, qz_scale], origin=origin)
    
        # Eliminate the first point (which is absurdly high due to FFT artifacts)
        qs = qs[1:]
        data1D = data1D[1:]
    
        # Optionally adjust the curve to improve data extraction
        if adjust is not None:
            data1D *= np.power(qs, adjust)
    
        ## Modify the code to use the max as the peak estimation
        idx = np.where( data1D==np.max(data1D) )[0][0]

        expected = qs[idx] 
    
        # Fit the 1D curve to a Gaussian
        #lm_result, fit_line, fit_line_extended = self.peak_fit(qs, data1D, x_expected=expected, range_rel=range_rel)
        p = 0#lm_result.params['prefactor'].value # Peak height (prefactor)
        q = 0#lm_result.params['x_center'].value # Peak position (center) ==> use this
        sigma = 0#lm_result.params['sigma'].value # Peak width (stdev) ==> use this
        I = 0#p*sigma*np.sqrt(2*np.pi) # Integrated peak area
        m = 0#lm_result.params['m'].value # Baseline slope
        b = 0#lm_result.params['b'].value # Baseline intercept
        return p, q, sigma, I, m, b, qs, data1D
        
    def circular_average(self,data, scale=[1,1,1], origin=None, bins_relative=3.0):
    
         h, w, d = data.shape
         y_scale, x_scale, z_scale = scale
         if origin is None:
             x0, y0, z0 = int(w/2), int(h/2), int(d/2)
         else:
             x0, y0, z0 = origin
        
         # Compute map of distances to the origin
         x = (np.arange(w) - x0)*x_scale
         y = (np.arange(h) - y0)*y_scale
         z = (np.arange(d) - y0)*z_scale
         X,Y,Z = np.meshgrid(x,y,z)
         R = np.sqrt(X**2 + Y**2 + X**2)

         # Compute histogram
         data = data.ravel()
         R = R.ravel()
    
         scale = (x_scale + y_scale + z_scale)/2.0
         r_range = [0, np.max(R)]
         num_bins = int( bins_relative * abs(r_range[1]-r_range[0])/scale )
         num_per_bin, rbins = np.histogram(R, bins=num_bins, range=r_range)
         idx = np.where(num_per_bin!=0) # Bins that actually have data
    
         r_vals, rbins = np.histogram( R, bins=num_bins, range=r_range, weights=R )
         r_vals = r_vals[idx]/num_per_bin[idx]
         I_vals, rbins = np.histogram( R, bins=num_bins, range=r_range, weights=data )
         I_vals = I_vals[idx]/num_per_bin[idx]

         return r_vals, I_vals
         

        

