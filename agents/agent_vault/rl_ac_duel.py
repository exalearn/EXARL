import os, sys, time, math, configparser
import numpy as np
from random import random, randint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from agents.rl_base import *


############################# Actor Critic Neural Net class #############################
# TODO: This neural network architecture can be modified later
# by using hyperparameters optimization
class Net(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=256):
        super(Net, self).__init__()
        self.fc1          = nn.Linear(state_size, hidden_size)
        self.fc2          = nn.Linear(hidden_size, hidden_size)
        self.value_layer  = nn.Linear(hidden_size, 1)
        self.action_layer = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value  = self.value_layer(x)                      # state_value
        action_probs = F.softmax(self.action_layer(x), dim=-1)  # action probabilities
        return state_value, action_probs

############################# Actor-Critic class #############################
class AC(RL):

    def __init__(self):

        RL.__init__(self)

        self.vtrace            = self.config.getboolean('DEFAULT', 'vtrace')
        print("vtrace ", self.vtrace)

        self.behavi_net = Net(self.state_size, self.action_size).to(self.device)
        if self.vtrace:
            self.target_net = Net(self.state_size, self.action_size).to(self.device)
            self.target_net.load_state_dict(self.behavi_net.state_dict())
            self.target_net.eval()

        # raise value error if distributed training mode is synchronous
        if self.distributed_mode == "sync":
            raise ValueError("Error, for A3C, distributed training mode cannot be async.")

        # prepare buffer for async training
        self.prepareAsyncExchangeBuffer()
        # make sure the model buffer contains something sane
        self.copyModelToExchangeBuffer()

        # adam settings
        self.betas          = (0.9, 0.999)
        self.optimizer      = torch.optim.Adam(self.getTrainableModelParameters(), \
                                               lr=self.alpha, betas=self.betas)

        # containers to save rewards, state_value, log_probs
        self.vecRewards        = [ 0.0 for _ in range(self.steps) ]
        self.vecStateValues    = [] # [ 0.0 for _ in range(TIMESTEPS)]
        self.vecBehaviLogProbs = [] # [ 0.0 for _ in range(TIMESTEPS)]
        if self.vtrace:
            self.vecTargetLogProbs = [] # [ 0.0 for _ in range(TIMESTEPS)]

    ############################### select an action ###############################
    def selectAction(self, state):

        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)  # put state in a tensor
        state_value, behavi_action_probs = self.behavi_net.forward(state)     # get state_velue and action_probs from behaviour network

        if self.vtrace:
            _,           target_action_probs = self.target_net.forward(state)     # get state_velue and action_probs from target network
            target_action_distribution = Categorical(target_action_probs)

        behavi_action_distribution = Categorical(behavi_action_probs)
        action                     = behavi_action_distribution.sample()

        # TODO: push back method should be avoided!
        self.vecStateValues.append(state_value)
        self.vecBehaviLogProbs.append(behavi_action_distribution.log_prob(action))

        if self.vtrace:
            self.vecTargetLogProbs.append(target_action_distribution.log_prob(action))

        if self.debug>=30:
            print_status("state {}".format(state), comm_rank=self.comm_rank, allranks=True)

        if self.debug>=10:
            print_status("behavi_action_probs {}".format(behavi_action_probs), \
                         comm_rank=self.comm_rank, allranks=True)
            print_status("target_action_probs {}".format(target_action_probs), \
                         comm_rank=self.comm_rank, allranks=True)

        if self.debug>=10:
            print_status("action {}".format(action.item()), \
                         comm_rank=self.comm_rank, allranks=True)

        return action.item()

    ############################### select a greedy action ###############################
    def selectGreedyAction(self, state):

        s = state
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        self.state_value, action_probs = self.behavi_net.forward(state)
        action_probs                   = self.restrictPolicy(s, action_probs)
        action                         = torch.argmax(action_probs)

        if self.debug>=30:
            print_status("state {}".format(state), \
                         comm_rank=self.comm_rank, allranks=True)

        if self.debug>=10:
            print_status("action_probs {}".format(action_probs), \
                         comm_rank=self.comm_rank, allranks=True)

        if self.debug>=10:
            print_status("action {}".format(action.item()), \
                         comm_rank=self.comm_rank, allranks=True)

        return action.item()

    ############################ restricting some actions ############################
    def restrictPolicy(self, state, P1):

        listRestricted = -1
        P = [ 0.0 for i in range(self.action_size)]

        for i in range(self.action_size): P[i] += P1[0,i].item()

        if  self.env.T >= self.env.chparams.T_max:
            P[2] = 0.0; listRestricted = 2
        elif self.env.T <= self.env.chparams.T_min:
            P[0] = 0.0; listRestricted = 0

        if self.debug>=20:
            print_status('T, P: {}, {}'.format(np.around(self.env.T, 2), \
                    np.around(P,2)), comm_rank=self.comm_rank, allranks=True)

        # adjust probabilities, they have to sum up to 1
        if listRestricted != -1 :
            total = 0.0
            for i in range(self.action_size): total += P[i]
                #print("total", P[0,i].item())
            if total==0.0:
                for i in range(self.action_size):
                    if i not in listRestricted:
                        P[i] = 1.0 / (self.action_size-1)
            else:
                for i in range(self.action_size): P[i] = P[i]/total

        return torch.tensor([P]) #.to(self.device)

    ############################ calculate loss function ############################
    def calculateLoss(self):

        # calculating discounted rewards:
        G = 0
        self.vecDiscRewards = [ 0.0 for _ in range(self.step)]
        for t in reversed(range(self.step)):
            G = self.vecRewards[t] + self.gamma * G
            self.vecDiscRewards[t] = G

        # convert to a tensor
        self.vecDiscRewards = torch.tensor(self.vecDiscRewards).to(self.device)

        self.vecDiscRewards = (self.vecDiscRewards - self.vecDiscRewards.mean()) \
                              / (self.vecDiscRewards.std()) # normalizing the discounted rewards

        if self.vtrace:
            self.target_net.load_state_dict(self.behavi_net.state_dict())  # copy the parameters
            self.setVtrace()

        total_loss = 0.0
        for logProb, state_val, reward in zip(self.vecBehaviLogProbs, \
                                      self.vecStateValues, self.vecDiscRewards):
            action_loss = -logProb * (reward  - state_val) # advantage = reward - state_val
            value_loss  = F.smooth_l1_loss(state_val, torch.tensor([[reward]]))
            total_loss += (action_loss + value_loss)

        return total_loss

    ############## return model parameters for use in superclass ###############
    def getTrainableModelParameters(self):
        return self.behavi_net.parameters()

    ############################### clear memory ###############################
    def clearMemory(self):
        self.vecRewards     = [ 0.0 for _ in range(self.steps)]
        del self.vecStateValues[:]        #= [] #= [ 0.0 for _ in range(TIMESTEPS)]
        del self.vecBehaviLogProbs[:]    #= [] #= [ 0.0 for _ in range(TIMESTEPS)]
        if self.vtrace: del self.vecTargetLogProbs[:]


    ############### skip analysis for the first several steps ####################
    def skipAnalysis(self, state):

        if self.debug>=1:
            print_status("skip t {}".format(self.step+1), comm_rank=self.comm_rank)

        action_idx = self.selectAction(state)  # TODO: modify this part
        self.env.currStructVec = self.env.getNextState(action_idx, self.step)
        self.endTimeStep = time.time()

        self.setControlParams(self.step)

        if self.debug>=0: self.printTimeStepInfo(reward=0)

    ################################### vtrace ########################################
    def setVtrace(self):
        C   = 1
        v_s = self.vecStateValues[0]
        self.setRho()
        self.setC()

        for t in range(self.steps):
            C *= self.vecC[t]
            if t==self.steps-1:
                v_s +=  C * self.vecRho[t] * ( self.vecRewards[t] - self.vecStateValues[t] )
            else:
                v_s +=  C * self.vecRho[t] * ( self.vecRewards[t] + pow(self.gamma, t-1) * self.vecStateValues[t+1] - self.vecStateValues[t] )
            self.vecStateValues[t] = v_s

    # create "rho" vector
    def setRho(self, rho_bar=1.0):
        self.vecRho = np.empty(self.steps)
        for t in range(self.steps):
            #print("TargetLogProbs, BehaviourLogProbs: ", self.vecTargetLogProbs[t].detach().numpy(), self.vecBehaviLogProbs[t].detach().numpy())
            self.vecRho[t] = np.exp( self.vecTargetLogProbs[t].item() - self.vecBehaviLogProbs[t].item() )
            #print("rho_bar, rho: ", rho_bar, self.vecRho[t])
            self.vecRho[t] = min(rho_bar, self.vecRho[t])

    # create "C" vector
    def setC(self, c_bar=1.0):
        self.vecC = np.empty(self.steps)
        for t in range(self.steps):
            self.vecC[t] = np.exp( self.vecTargetLogProbs[t].item() - self.vecBehaviLogProbs[t].item() )
            #print("C_bar, C: ", c_bar, self.vecC[t])
            self.vecC[t] = min(c_bar, self.vecC[t])

    ############################### train A2C ###############################
    def train(self):

        self.isTest = False

        for e in range(self.episodes):  # for each episode

            self.episode = e
            self.createDirectory(isTest=False)
            self.startTimeEpisode = time.time()

            print_status('########################## Episode: {} ##############################'.format(e), comm_rank=self.comm_rank)

            # for param in self.behavi_net.parameters(): print(param.data)

            state = self.env.reset()
            score = 0

            self.reward_accumulator.reset()
            self.distance_accumulator.reset()
            self.setControlParams(-1) # set structure vector and control parameter

            # init timers
            time_select_action = 0.0
            time_env_step      = 0.0

            for t in range(self.steps):  # for each time step

                self.step = t
                self.startTimeStep = time.time()

                if not (self.rewardOption==1 or self.rewardOption==3):
                    if self.step < self.perSkipSteps * self.steps:
                        self.skipAnalysis(state)  # Do not analyze first several steps
                        if self.debug>=1: print("skip")
                        continue

                if self.debug>=1: t0 = time.time()
                action               = self.selectAction(state)   # select action
                if self.debug>=1: time_select_action    += time.time() - t0
                #state, reward, distance, _ = self.env.step(action, self.step) # take action
                #done = self.env.isTerminalState(distance)
                state, reward, done, _ = self.env.step(action)
                distance               = self.env.getDistance()
                if self.debug>=1: time_env_step         += time.time() - t0

                if done or self.step == self.steps-1:
                    self.vecScores[self.episode][1] = reward

                reward = self.modifyReward(done, reward)

                # accumulate the scores and the deviation from terminal state
                self.reward_accumulator.update(torch.from_numpy(np.array(reward, dtype=np.float32)))
                self.distance_accumulator.update(torch.from_numpy(np.array(distance, dtype=np.float32)))

                #score += reward  # accumulate the score
                self.vecRewards[t] = reward
                self.setControlParams(t) # self.vecControlParams[t+1] = self.env.T

                if self.debug>=0: self.printTimeStepInfo(reward)

                if done or self.step>=self.steps-1: # in the last step of each episode

                    self.vecScores[self.episode][0] = score

                    if self.saveTrain():
                        self.saveTrainingNN(self.behavi_net)
                        self.saveScoreEpisode()

                    if self.debug>=-1: self.printEpisodeInfo()

                    break

            if self.debug>=1:
                print_status("time_select_action: {}".format(time_select_action), comm_rank=self.comm_rank)
                print_status("time_env_step: {}".format(time_env_step), comm_rank=self.comm_rank)
                print_status("env.time_getNextState: {}".format(self.env.time_getNextState), comm_rank=self.comm_rank)
                print_status("env.time_getReward: {}".format(self.env.time_getReward), comm_rank=self.comm_rank)
                print_status("env.time_isTerminalState: {}".format(self.env.time_isTerminalState), comm_rank=self.comm_rank)

            self.learnAsync()  # backpropagation
            self.clearMemory() # clear memory

            self.saveControlTime()

        ##################### After training save data and figures #####################
        if not self.saveTrain():  # if not yet saved, make sure we save the info
            self.saveTrainingNN(self.behavi_net)
            self.saveScoreEpisode()

    ############################### run a trained model ###############################
    def test(self):

        self.isTest = True
        self.createDirectory(isTest=True)

        print_status('########################## Test RL ##############################', \
                     comm_rank=self.comm_rank)

        score = 0
        state = self.env.reset()
        self.reward_accumulator.reset()
        self.distance_accumulator.reset()
        self.setControlParams(-1) # set structure vector and control parameter

        # if -notTrain was selected, load a saved neural network
        if self.runTrain and self.comm_rank==0:
            self.behavi_net = torch.load(os.path.join(self.global_output_dir, \
                          '/saved_nn_' + str(self.size_struct_vec)))
                       # + str(self.method)

        # Horovod: broadcast parameters & optimizer state.
        if self.use_mpi:
            self.broadcastModel()

        self.behavi_net.eval()  # set the behaviour network to be the evaluation mode

        for t in range(self.steps):  # for each time step

            self.step = t

            # Select and perform an action
            action                     = self.selectGreedyAction(state)
            if self.debug>=10: print("action: ", action)

            state, reward, done, _ = self.env.step(action)
            distance               = self.env.getDistance()

            reward                 = self.modifyReward(done, reward)

            if self.debug>=0: self.printTimeStepInfo(reward)

            # save score after the first 20% of the time steps
            self.distance_accumulator.update(torch.from_numpy(np.array(distance, dtype=np.float32)))
            if self.step >= self.perSkipSteps * self.steps:
                self.reward_accumulator.update(torch.from_numpy(np.array(reward, dtype=np.float32)))

            self.setControlParams(t) # self.vecControlParams[t+1] = self.env.T

            if done or self.step == self.steps-1:
                print_status("Test Accumulated Reward: {:.2f} \tAverage Distance: {:.6f} \tTime-steps: {}".format(self.reward_accumulator.sum.item(), \
                             self.distance_accumulator.avg.item(), self.step), \
                             comm_rank=self.comm_rank)
                break

        self.saveControlTime()
