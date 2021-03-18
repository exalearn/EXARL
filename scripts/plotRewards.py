# **********************************************************************************
# Change the directory

# base_dir = '/projects/users/vinayr/ExaLearn/ExaRL/summit_results_exabooster/'
# nodes_dir = ['ExaBooster_RMA_learner_1_nodes', 'ExaBooster_RMA_learner_4_nodes']

base_dir           = '/gpfs/alpine/scratch/aik07/ast153/results_dir_a1_mp16_e10_t10'
nodes_dir          = ['']
episodes_per_nodes = []

# **********************************************************************************

import os, sys
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.rcParams['font.family'] = [u'serif']
plt.rcParams['font.size'] = 14

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# read data
def read_data(filename, rank):

    frame = pd.read_csv(filename, sep=' ',header=None,
                        names=['time', 'current_state', 'action', 'reward', 'next_state', 'total_reward', 'done',
                               'episode', 'step', 'policy_type', 'epsilon'])
    # print(frame)

    del frame['current_state']
    del frame['next_state']

    frame['time'] = pd.to_datetime(frame['time'], unit='ns')
    frame         = frame[frame.done == True]
    frame         = frame.reset_index()
    frame['rank'] = int(rank)
    
    # print(frame['time'])
    # print(frame['reward'])

    return frame


# Set reward plot info
def set_reward_plot_info(results_dir):

    df_ranks = []
    rank=0

    # Candle directory stucture
    results_dir = results_dir

    for filename in os.listdir(results_dir):
        if filename.endswith(".log"):
            rank+=1
            print('rank {}: filename:{}'.format(rank,filename))
            df = read_data(results_dir + filename,rank)
            df_ranks.append(df)
            
    df_merged = pd.concat(df_ranks)
    df_merged = df_merged.dropna()
    time_min  = df_merged.time.min()
    time_max  = df_merged.time.max()

    print('time_min',time_min)
    print('time_diff',time_max-time_min)
    
    df_merged['rel_time'] = [idx - time_min for idx in df_merged.time]
    #df_merged['rel_time'] = df_merged['rel_time'].total_seconds()
    #df_merged['rel_time'] = df_merged.rel_time.total_seconds()
    #df_merged.set_index('rel_time')
    #df_merged = df_merged.reset_index()

    df_merged.sort_values(by=['rel_time'], inplace=True)
    return df_merged


# **************************************************************************
# a script to to show the reward plot

rolling_setting=25
fig,ax = plt.subplots(1, 1,figsize=(10,8))
plt.title('Rewards over time')


for node_dir in nodes_dir:

    current_dir = base_dir + node_dir + '/EXP000/RUN000/'
    print('Node path:{}'.format(current_dir))

    merged_df                      = set_reward_plot_info(current_dir)
    merged_df['total_reward_roll'] = merged_df['total_reward'].rolling(rolling_setting).mean()

    print((merged_df.shape))

    # print(merged_df['total_reward_roll'] )
    # print(merged_df['rel_time'])

    # merged_df = merged_df.cummax

    plt.plot(merged_df['time'], merged_df['reward'])
    
    # plt.plot(merged_df['rel_time'], merged_df['total_reward_roll'], label='{}'.format(node_dir))
    
    ###episodes_per_nodes.append(len(merged_df))


#print(merged_df.head(5))
plt.xlabel('Relative Time')
plt.ylabel('Rolling Total Reward ({})'.format(rolling_setting))

#plt.xlim(00,2500)
#plt.ylim(-20,0)
plt.legend(loc="lower right")

plt.show()
