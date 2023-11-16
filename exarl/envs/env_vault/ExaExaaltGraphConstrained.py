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

import numpy  as np
import gym
import os
from mpi4py import MPI
from exarl.utils.globals import ExaGlobals
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.cm import YlGnBu

from matplotlib.offsetbox import (TextArea, AnnotationBbox)

YlGnBu.set_bad(color="whitesmoke")

try:
    graph_size = ExaGlobals.lookup_params('graph_size')
except:
    graph_size = 10


run_name = str(ExaGlobals.lookup_params('experiment_id'))
now = datetime.now()
NAME = now.strftime("%d_%m_%Y_%H-%M-%S_")

run_name = NAME + run_name

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def dirichlet_draw(alphas):
    sample = [np.random.gamma(a, 1) for a in alphas]
    sums   = sum(sample)
    sample = [x/sums for x in sample]
    return sample

def get_graph_dist(knownStates, state):
    all_keys   = [k for k,v in knownStates.items() if v != None]
    graph_dist = {}
    for loop_state in all_keys:
        graph_dist[loop_state] = 5
    graph_keys = [[state]]
    inc_keys   = [state]
    for ii in range(5):
        new_keys = []
        for key in graph_keys[ii]:
            graph_dist[key] = ii
            # tmp_keys = [x for x in knownStates[key].probs.keys() if x not in inc_keys]
            tmp_keys = [x for x in knownStates[key].counts.keys() if (x not in inc_keys) and (x in knownStates.keys()) and (knownStates[x] != None)]
            inc_keys += tmp_keys
            new_keys += tmp_keys
        graph_keys.append(new_keys)
            
    return graph_dist

def get_graph_adj(knownStates, state, database):
    adj_mat    = np.zeros([graph_size, graph_size*2])
    all_keys = sum(value != None for value in knownStates.values())
    graph_keys = [[state]]
    inc_keys   = [state]

    for ii in range(graph_size):
        # Checks if # of included keys is equal to or greater than graph_size
        if len(inc_keys) >= graph_size:
            inc_keys = inc_keys[:graph_size]
            break

        if len(inc_keys) == all_keys:
            # Check to make sure tmp key is not equal to None
            tmp_keys = [x for x in knownStates.keys() if x not in inc_keys and knownStates[x] != None]
            if len(tmp_keys) > 0:
                if len(tmp_keys) + len(inc_keys) <= graph_size:
                    inc_keys += tmp_keys
                else:
                    inc_keys += tmp_keys[-(graph_size - len(inc_keys)):]
            break

        new_keys = []
        for key in graph_keys[ii]:
            tmp_keys = [x for x in knownStates[key].counts.keys() if (x not in inc_keys) and (x in knownStates.keys()) and (knownStates[x] != None)]
            
            inc_keys += tmp_keys
            new_keys += tmp_keys
            
        graph_keys.append(new_keys)
    
    for ii, row_key in enumerate(inc_keys):
        for jj, col_key in enumerate(inc_keys):
            if col_key in knownStates[row_key].counts.keys():
                adj_mat[ii,jj] = 1 if knownStates[row_key].counts[col_key] > 0 else 0
                dat_pos = jj+graph_size
                adj_mat[ii,dat_pos] = 1 if database[row_key].count(col_key) > 0 else 0

    # return adj_mat.flatten(), inc_keys
    return np.ones_like(adj_mat).flatten(), inc_keys

def VE(traj, knownStates, database, nWorkers, d_prior):
    print('running VE... '+str(len(knownStates.keys()))+' states discovered')
    knownStates_keys = [k for k,v in knownStates.items() if v != None]
    builds={}
    for a in knownStates_keys:
        builds[a]=0

    taskList=[]
    for _ in range(nWorkers):
        virtuallyConsumed={}
        for i in knownStates_keys:
            virtuallyConsumed[i]=0

        state=traj[-1]
        while(len(database[state])+builds[state]>virtuallyConsumed[state]):
            virtuallyConsumed[state]+=1
            try:
                state=database[state][virtuallyConsumed[state]-1]
            except:
                offset      = np.array([0., 0., 0., 0., 0., 0.])
                prior_p     = d_prior[0] + offset + 1.e-6
                graph_dist  = get_graph_dist(knownStates, state)
                d_alpha     = np.array([ prior_p[graph_dist[key]] for key in knownStates_keys ])
                keylist = list(knownStates_keys)
                for ii in range(d_alpha.size):
                    try:
                        d_alpha[ii] = d_alpha[ii] + knownStates[state].counts[keylist[ii]]
                    except:
                        pass
                if len(d_alpha) == 1:
                    sample_p = [1.]
                else:
                    sample_p   = dirichlet_draw(d_alpha)
                state      = np.random.choice(list(knownStates_keys),p=sample_p)
        taskList.append(state)
        builds[state]+=1
    return taskList

class StateStatistics:
    def __init__(self, label, Map):
        self.Map    = Map
        self.counts = {}
        for neigh in self.Map[int(label)].keys():
            self.counts[neigh]=0
        self.nSegments=0
        self.nTransitions=0
        self.label = int(label)
        self.probs={}
        self.l=0
        self.lp=0
        self.initP=-1

    def clearProbs(self):
        self.probs={}

    def update(self, finalState):
        self.nSegments+=1
        try:
                self.counts[finalState]+=1
        except:
                self.counts[finalState]=1
        if(finalState!=self.label):
                self.nTransitions+=11

class ExaExaaltGraphConstrained(gym.Env):
    def __init__(self,**kwargs):
        super().__init__()
        """
        """
        stateDepth       = 50 #segments
        number_of_states = 10000

        self.n_states  = number_of_states
        self.graph_len = graph_size
        self.nWorkers  = 500
        self.reward    = 0
        self.WCT       = 0
        self.RUN_TIME  = int(ExaGlobals.lookup_params('n_steps'))
        
        self.database    = {}
        self.knownStates = {}
        self.actions_avail = np.arange(0, self.n_states, 1)
        self.traj        = []

        self.state_order = [ii for ii in range(self.n_states)]

        self.Map={}
        side= 100
        for i in range(number_of_states):
            self.Map[i]={}
            self.Map[i][i]=1-1.0/stateDepth
        
            R=(1-self.Map[i][i])/4
            if(i%side==0):
                #left side
                self.Map[i][(i-1+side)%number_of_states]=R
                self.Map[i][(i+1)%number_of_states]=R
            else:
                if(i%side==(side-1)):
                    #right side
                    self.Map[i][i-1]=R
                    self.Map[i][(i+1-side)%number_of_states]=R
        
                else:
                    #not an edge
                    self.Map[i][i-1]=R
                    self.Map[i][(i+1)%number_of_states]=R
        
            self.Map[i][(i-side)%number_of_states]=R
            self.Map[i][(i+side)%number_of_states]=R

        self.INITIAL_STATE              = int((side/2)*side+side/2) 

        self.database[self.INITIAL_STATE]    = []
        self.traj.append(self.INITIAL_STATE)    

        self.adj_states = np.zeros(self.graph_len, dtype=int)
        self.adj_states[0] = self.INITIAL_STATE                       

        self.action_space  = gym.spaces.Box(low=0.0, high=1.0,    shape=(self.graph_len,), dtype=np.float32)
        # self.action_space  = gym.spaces.Box(low=-10.0, high=10.0,    shape=(self.graph_len,), dtype=np.float32)
        self.adj_space     = gym.spaces.Box(low=0.0, high=np.inf, shape=(graph_size, graph_size*2))
        
        # Position of the final trajectory
        self.end_traj = gym.spaces.Discrete(self.nWorkers)
        
        # Creation of the mask for the invalid action masking
        # MAY NEED TO CHANGE IF VALUE GETS TOO HIGH!!!
        #self.mask = gym.spaces.Dict({})
        #for i in range(number_of_states):
        #    self.mask[i] = gym.spaces.Discrete(65535)
        #    self.knownStates[i] = None

            # self.mask.update({i,gym.spaces.Discrete(65535)})
        
        self.knownStates[self.INITIAL_STATE] = StateStatistics(self.INITIAL_STATE, self.Map) 
        # self.observation_space = gym.spaces.Tuple((self.adj_space, self.end_traj, self.mask))
        self.observation_space     = gym.spaces.Box(low=0.0, high=np.inf, shape=(graph_size*graph_size*2,))

    def step(self, action):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        done = False

        # action_p = softmax(action)
        action_p = action
        
        taskList = np.random.choice(self.adj_states, size=self.nWorkers, replace=True, p=action_p)

        for i in range(self.nWorkers):
            workerID   = i
            buildState = taskList[i]
            if buildState == -1:
                continue
            endState   = np.random.choice(list(self.Map[buildState].keys()),p=list(self.Map[buildState].values()))

            self.knownStates[buildState].update(endState)
            self.database[buildState].append(endState)
            if (self.knownStates[endState] == None):
                self.knownStates[endState]=StateStatistics(endState, self.Map)
                self.database[endState] = []
        added = 0
        self.WCT+=1
        action_str = [str(x) for x in action_p]
        action_str = " ".join(action_str)
        taskList = list(taskList)

        if not os.path.exists("./outputs/Exaalt/" + run_name + "/"):
            os.makedirs("./outputs/Exaalt/" + run_name + "/")

        starting_state = self.traj[-1]
        while(True):
            current_state=self.traj[-1]
            next_state = None
            try:
                next_state =self.database[current_state].pop(0)
                self.traj.append(next_state)
            except:
                with open("./outputs/Exaalt/" + run_name + "/" + run_name + "_" + str(rank), "a") as myfile:
                        myfile.write(
                        str(round(self.WCT,3))+' '+
                        str(len(self.traj))+' '+
                        str(self.WCT*self.nWorkers)+' '+
                        str((len(self.traj)-1)/float(self.WCT*self.nWorkers))+' '+
                        str(added)+' '+
                        action_str + ' ' +
                        '\n')
                break
            if (next_state in taskList):
                added += 1
                taskList.remove(next_state)
                
        # self.reward =  ((self.RUN_TIME-self.WCT)/self.RUN_TIME)*(added/self.nWorkers)
        # self.reward = added/self.nWorkers

        self.reward = 0
        if (self.WCT >= self.RUN_TIME):
            self.reward = (len(self.traj)-1)/float(self.WCT*self.nWorkers) 
            done = True

        """ Iterates the testing process forward one step """

        # self.reward = 0.5*(len(self.traj)-1)/float(self.WCT*self.nWorkers) + 0.5*(added/self.nWorkers)
        # self.reward = 10*action_p[0]
        # self.reward = 10*action[0]
        # self.reward = (len(self.traj)-1)/float(self.WCT*self.nWorkers)
        current_state = self.traj[-1]
        adj_mat, inc_keys = self.generate_data()
        np.put(self.adj_states, np.arange(len(inc_keys)), inc_keys)        
        next_state = (adj_mat, current_state, self.knownStates)

        n_s  = len(inc_keys)
        info = {"mask":np.concatenate([np.repeat(True,n_s), np.repeat(False, graph_size-n_s)])}
        
        print("Step: ", self.WCT, " Reward: ", self.reward, " ", done, " Added: ", added, " Action[0]: ", action[:3])
        # self.render(taskList,starting_state,current_state)
        return next_state[0], self.reward, done, False, info

    def reset(self):
        """ Start environment over """
        side = 100

        self.WCT                             = 0 
        self.INITIAL_STATE                   = int((side/2)*side+side/2)
        self.reward                          = 0 
        self.traj                            = []
        self.database                        = {}
        self.knownStates                     = {}
        self.database[self.INITIAL_STATE]    = []
        self.traj.append(self.INITIAL_STATE)
        self.adj_states = -1*np.ones(self.graph_len, dtype=int)
        self.adj_states[0] = self.INITIAL_STATE    

        for i in range(self.n_states):
            self.knownStates[i] = None
        
        self.knownStates[self.INITIAL_STATE] = StateStatistics(self.INITIAL_STATE, self.Map)
        adj_mat, inc_keys = self.generate_data()

        state_tuple = (adj_mat, self.traj[-1], self.knownStates)
        # return state_tuple[0].flatten(), {} # Return new state
        return np.ones_like(state_tuple[0]).flatten(), {"mask":np.concatenate([np.repeat(True, 1), np.repeat(False, graph_size-1)])} # Return new state

    def render(self, taskList, start_state, end_state):
        """ Not relevant here but left for template convenience """
        database_matrix = np.zeros([100,100])
        schedule_matrix = np.zeros([100,100])
        fig, ax = plt.subplots(1,2,figsize=(16,8), dpi=250)

        for ii in range(100):
            for jj in range(100):
                database_matrix[ii,jj] = len(self.database[ii*100+jj]) if ii*100+jj in self.database.keys() else (0 if ii*100+jj in self.knownStates.keys() else np.nan)
                schedule_matrix[ii,jj] = np.sum([ii*100+jj == x for x in taskList]) if ii*100+jj in self.knownStates.keys() else np.nan

        im1 = ax[0].imshow(database_matrix, cmap = YlGnBu)
        im2 = ax[1].imshow(schedule_matrix, cmap = YlGnBu)
        ax[0].set_title("Database")
        ax[1].set_title("Schedule")

        startbox = TextArea("Step Starting State", minimumdescent=False)
        endbox   = TextArea("Step Ending State", minimumdescent=False)

        eb = AnnotationBbox(endbox, (end_state % 100, int(end_state/100)),
                            xybox=(0.5, -0.15),
                            xycoords='data',
                            boxcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->"))
        ax[0].add_artist(eb)

        sb = AnnotationBbox(startbox, (start_state % 100, int(start_state/100)),
                            xybox=(0.,-0.15),
                            xycoords='data',
                            boxcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->"))
        ax[0].add_artist(sb)

        plt.colorbar(im1, ax=ax[0])
        plt.colorbar(im2, ax=ax[1])
        
        fig.savefig("armen_step_"+str(self.WCT)+".png")
        plt.close('all')

        return 0
    
    def generate_data(self):
        return get_graph_adj(self.knownStates, self.traj[-1], self.database)

