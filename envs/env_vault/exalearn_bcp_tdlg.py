import gym, subprocess, os, logging, json, math
import numpy as np
import pandas as pd 
import plotly.graph_objects as go

from gym import spaces
from shutil import copyfile,rmtree
from collections import defaultdict
from plotly.subplots import make_subplots
from utils.sv import *
#from utils.tdlg_plot import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BlockCoPolymerTDLG-Logger')
#logger.setLevel(logging.INFO)
logger.setLevel(logging.CRITICAL)

class BlockCoPolymerTDLG(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,cfg_file='../cfg/tdlg_setup.json'):#,worker_index=1):
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
        
        data = []
        
        with open(cfg_file) as json_file:
            data = json.load(json_file)
        self.cfg_data = data

        ## Application setup
        self.app_dir  = data['app_dir']  if 'app_dir' in data.keys() else '/home/schr476/exalearn/tdlg_fy19/build_cpu/'
        self.app_name = data['app_name'] if 'app_name' in data.keys() else 'tdlg_run'
        self.app      = os.path.join(self.app_dir,self.app_name)

        ## Model parameters
        self.param_dir  = data['param_dir']  if 'param_dir' in data.keys()  else '/home/schr476/exalearn/rl_repos/gym-exalearn/cfg/'
        self.param_name = data['param_name'] if 'param_name' in data.keys() else 'tdlg_param.in'
        self.param_file = os.path.join(self.param_dir,self.param_name)
        self.model_parameter  = defaultdict(float)
        self._updateParamDict()

        ## for multi-worker training
        #rank_index=0
        self.worker_dir = './' #data['worker_index'] if 'worker_index' in data.keys() else rank_index
        #self.worker_dir = './multiworker/worker'+str(self.worker_index)
        #if os.path.exists(self.worker_dir): rmtree(self.worker_dir)
        #os.mkdir(self.worker_dir)
        #os.mkdir(self.worker_dir+'/archive/')
        #print ("worker directory: " + self.worker_dir)
        
        ## for plotting
        self.rendering = False

        ## Step sizes for discrete environment 
        #self.fracA_step_size = float(data['fracA_step_size']) if 'fracA_step_size' in data.keys() else 0.005
        self.kappa_step_size = float(data['kappa_step_size']) if 'kappa_step_size' in data.keys() else 0.001
        
        ## Setup target structure
        self.target_structure_name = data['target_structure_name'] if 'target_structure_name' in data.keys() else 'cfg/target_field.out'
        self.target_structure = self.setTargetStructure(self.target_structure_name)
        self.target_precision = -2 ## 98[%]
        self.structure_len = len(self.target_structure)
        
        self.current_structure = np.zeros(self.structure_len)

        ## Define state and action spaces
        self.observation_space = spaces.Box(low=np.append(np.zeros(self.structure_len),[0.004]), high=np.append(np.ones(self.structure_len),[0.012]),dtype=np.float32)
        self.action_space = spaces.Discrete(3) 

        ## Fix the starting point
        self.model_parameter_init = {}
        #self.model_parameter_init['fracA'] = self.observation_space.high[0] - self.fracA_step_size
        #self.model_parameter_init['kappa'] = self.observation_space.high[-1] - self.kappa_step_size
        self.model_parameter_init['kappa'] = self.model_parameter['kappa']

        self.current_reward = 0
        self.total_reward   = 0

        ##
        self.episode_num = 0
        self.step_num = 0
        
        ## Use reset function to do the rest
        self.reset()
        logger.info ("from __init__")
        logger.info (self.model_parameter)

    def set_results_dir(self,results_dir):
        self.worker_dir=results_dir
        if os.path.exists(self.worker_dir): rmtree(self.worker_dir)
        os.mkdir(self.worker_dir)
        os.mkdir(self.worker_dir+'/archive/')
        #print ("worker directory: " + self.worker_dir)
        
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
    
    def _getState(self):
        #return [round(self.model_parameter['fracA'],3),round(self.model_parameter['kappa'],3)]
        return np.append(self.current_structure,round(self.model_parameter['kappa'],3))
    
    def step(self, action):
        self.step_num+=1
                
        logger.info('Env::step()')

        ## Default returns
        done   = False
        reward = -111 ## Default penalty

        ## Check if the state-action has already been calculated
        inMem = self._inMemory(str(self._getState()),action)

        ## Apply discrete actions
        if action==1:
            self.model_parameter['kappa']+=self.kappa_step_size
            if self.model_parameter['kappa']>self.observation_space.high[1]:
                self.model_parameter['kappa']-=self.kappa_step_size
                done = True
                return self._getState(),reward,done, {}
                
        elif action==2:
            self.model_parameter['kappa']-=self.kappa_step_size
            if self.model_parameter['kappa']<self.observation_space.low[1]:
                self.model_parameter['kappa']+=self.kappa_step_size
                done = True
                return self._getState(),reward,done, {}

        ## Avoid running model
        if inMem['inMem']==True:
            logger.info('### In memory ... skipping simulation ###')
            return inMem['next_state'],float(inMem['reward']),inMem['done'], {}

        ## Run model
        if os.path.exists(self.worker_dir+'/field.in') : os.remove(self.worker_dir+'/field.in')
        if os.path.exists(self.worker_dir+'/field.out'): os.rename(self.worker_dir+'/field.out',self.worker_dir+'/field.in')
        
        self._run_TDLG()

        ## Get structure from model output 
        self.current_structure_name = self.worker_dir + '/field.out'
        self.current_structure, self.current_vol = self._get1DFFT(self.worker_dir + '/field.out')
        #copyfile(self.worker_dir + '/field.out', self.worker_dir + '/archive/'+str(self.episode_num)+'_field_'+str(self.step_num)+'.out')
        
        ## Calculate reward
        reward = self._calculateReward()
        self.total_reward += reward

        ## Define done based on how close it's
        if reward>self.target_precision:
            reward = 1000 
            done = True
            
        ## Rendering
        if self.rendering:
            logger.info("Plotting...")
            self._render()

        ## Return output
        return self._getState(),reward,done, {}

    def setTargetStructure(self,input_file_name):
        """
        Description: Read in a 3D volume file based on the TDLG output structure
        """
        structure,self.target_vol = self._get1DFFT(input_file_name)  
        return structure
        
        
    def _get1DFFT(self, input_file_name):
        """
        Description: Process 3D data from model output using Kevin's code
        """
        data = self._parseModelOutput(input_file_name)
        
        # Define expectations for simulations results    
        scale = 1 # Conversion of simulation units into realspace units
        d0_expected = 7.5
        q0_expected = 2*np.pi/d0_expected

        # Compute the structure vector
        vector = structure_vector(data, q0_expected, scale=scale)
        return vector[-1], data
        
    def _parseModelOutput(self, input_file_name):
        """
        Description: Read the model output and convert to readable format for Kevin's software
        """
        
        ## TODO: Add try-exception if these parameters are not defined in the parameter input file
        nshapshots = int(float(self.model_parameter['time'])/float(self.model_parameter['samp_freq']))
        nx = int(self.model_parameter['NX'])
        ny = int(self.model_parameter['NY'])
        nz = int(self.model_parameter['NZ'])

        ## Zero padd to satisfy Kevin's code
        nbox = nx if nx > ny else ny
        nbox = nbox if nbox > nz else nz
        padded_vol = np.zeros([nbox,nbox,nbox])
        
        ## Read in model output file ##
        with open(input_file_name, 'rb') as file:
            # read a list of lines into data
            filedata = file.readlines()

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
    
    def _plot_vol(self,vol): 
        X, Y, Z = np.mgrid[0:32:32j, 0:32:32j, 0:8:8j]
        data = go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=vol.flatten(),
            isomin=vol.min(),
            isomax=vol.max(),
            opacity=0.25,
            surface_count=25,
            colorscale='RdBu',
            showscale = False
        )
        return data
    
    def _calculateReward(self):
        """
        Description: Calculate the reduced chi^2 between the target structure and the current structure in 1D FFTW space
        """
        
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
        
        reward = -smape*100.0 ## yeilds a value between [0,100]
        
        return reward

    def _updateParamDict(self):
        """
        Description: 
           - Reads parameter file and make dict
           
        """
        ## Read default parameter file
        filedata = []
        with open(self.param_file, 'r') as file:
            # read a list of lines into data
            filedata = file.readlines()
        ## Make dict
        for param in filedata:
            key=param.split()[1]
            value=param.split()[0]
            if key=='time' or key=='NX' or key=='NY' or key=='NZ' or key=='seed' or key=='samp_freq' or key=='RandomInit':
                value=int(value)
            elif key=='fracA' or key=='kappa':
                value = round(float(value),3)
            else:
                value = float(value)
            self.model_parameter[key] = value
            
    def _run_TDLG(self):
        """
        Description: 
           - Creates a new param.in file based on the para dict
           - Run TDLG model
        """
        
        ## Prepare ##
        if os.path.exists(self.worker_dir+'/param.in'):
            os.remove(self.worker_dir+'/param.in')
        
        new_filedata = []
        for key, value in self.model_parameter.items():
            param = '%s  %s' % (str(value),str(key)) + '\n'    
            new_filedata.append(param)

        logger.info("## New parameter file ##")
        logger.info(new_filedata)
            
        with open(self.worker_dir+'/param.in', 'w') as file:
            file.writelines( new_filedata )
          
        ## Run model state ##
        env_out = subprocess.Popen([self.app], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.worker_dir)
        stdout,stderr = env_out.communicate()
        
        logger.info(stdout)
        logger.info(stderr)

    def reset(self):
        # Clear state and environment
        self.state=np.zeros(self.observation_space.shape)
        self.current_structure = np.zeros(len(self.target_structure))
        
        ## Clear parameter dict
        self.model_parameter = {}

        ## Re-read parameter file and setup dict
        self._updateParamDict()#os.path.join(self.param_dir,self.param_name))
        
        ## Return the state
        self.current_reward = 0
        self.total_reward   = 0

        #
        self.episode_num += 1
        self.step_num = 0
        
        logger.info("=====reset=====")
        if os.path.exists(self.worker_dir+'/field.in'):
            os.remove(self.worker_dir+'/field.in')
        if os.path.exists(self.worker_dir+'/field.out'):
            os.remove(self.worker_dir+'/field.out')
        if os.path.exists(self.worker_dir+'/E.out'):
            os.remove(self.worker_dir+'/E.out')
       
        return self._getState()

    ## set rendering parameters
    def render(self):
        self.rendering = True
        data = self.cfg_data
        
        self.plot_folder = self.worker_dir + '/plots/' #data['plot_folder'] if 'plot_folder' in data.keys() else 'plot_bcp_tdlg/worker'+str(self.worker_index)+'/'
        if os.path.exists(self.plot_folder): rmtree(self.plot_folder)
        os.makedirs(self.plot_folder)
        
        self.fig_target = self._plot_vol(self.target_vol)

    def _render(self):
        self.fig = make_subplots(rows=2, 
                                 cols=6,
                                 subplot_titles=("current structure","","","target structure", "","", 
                                                 "current FFT","","target FFT","","FFT difference",""),
                                 specs=[[{'type': 'volume','colspan':3},{},{}, {'type': 'volume','colspan':3},{},{}],
                                        [{'type': 'xy','colspan':2},{}, {'type': 'xy','colspan':2},{}, {'type': 'xy','colspan':2},{}]],
                                 vertical_spacing   = 0.1,
                                 horizontal_spacing = 0.1)
        figName = 'ep_'+str(self.episode_num)+'_step_'+str(self.step_num)
        
        self.fig.add_trace(go.Scatter(x=list(range(len(self.current_structure))), y=list(self.current_structure)),row=2, col=1)
        self.fig.update_yaxes(range=[-500, 1100], row=2, col=1)
        
        self.fig.add_trace(go.Scatter(x=list(range(len(self.target_structure))), y=list(self.target_structure)),row=2, col=3)
        self.fig.update_yaxes(range=[-500, 1100], row=2, col=3)
        
        self.fig.add_trace(go.Scatter(x=list(range(len(self.current_structure))), y=[c - t for c, t in zip(self.current_structure, self.target_structure)]),row=2, col=5)
        
        self.fig_current = self._plot_vol(self.current_vol)
        self.fig.add_trace(self.fig_current, row=1, col=1)
        self.fig.add_trace(self.fig_target , row=1, col=4)
                           
        self.fig.update_layout(height=800, width=1000, 
                               title_text = 'episode '+str(self.episode_num)+' step '+str(self.step_num),
                               showlegend=False,
                               scene_xaxis_showticklabels=False,
                               scene_yaxis_showticklabels=False,
                               scene_zaxis_showticklabels=False)
        
        filename = self.plot_folder + figName
        self.fig.write_image(filename + ".png")
        #self.fig.savefig(filename+".png",format='png')
        logger.info("Plot saved to: "+filename+".png")
        

    def close(self):
        return 0
