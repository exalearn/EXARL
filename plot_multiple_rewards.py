import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt

def read_data(filename, rank):
    frame = pd.read_csv(filename, sep=' ', header=None,
                        names=['time', 'current_state', 'action', 'reward', 'next_state', 'total_reward', 'done',
                               'episode', 'step', 'policy_type', 'epsilon'])
    del frame['current_state']
    del frame['next_state']
    frame['time'] = pd.to_datetime(frame['time'], unit='ns')
    frame = frame[frame.done == True]
    frame = frame.reset_index()
    frame['rank'] = int(rank)
    return frame

def get_merged_df(results_dir):
    df_ranks = []
    rank = 0

    for filename in os.listdir(results_dir):
        if filename.endswith(".log"):
            rank += 1
            df = read_data(results_dir + filename, rank)
            df_ranks.append(df)

    df_merged = pd.concat(df_ranks)
    df_merged = df_merged.dropna()
    time_min = df_merged.time.min()
    time_max = df_merged.time.max()
    df_merged['rel_time'] = [idx - time_min for idx in df_merged.time]
    df_merged.sort_values(by=['rel_time'], inplace=True)
    return df_merged

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.title('Cartpole Convergence Comparisons')

    x_tick_type = 'episode'     # 'episode' or 'time'
    rolling_setting = 25

    base_dir = '/home/kcosburn/EXARL/results_dir/'
    # data_dirs = ['temp_a2c_sync_cartpole_3', 'temp_a2c_vtrace_sync_cartpole_1']       # A2C vs. A2C w/ vtrace
    # data_dirs = ['temp_a2c_async_cartpole_2', 'temp_a2c_vtrace_async_cartpole_2']     # A3C vs. A3C w/ vtrace
    # data_dirs = ['temp_dqn_sync_cartpole_2', 'temp_a2c_vtrace_sync_cartpole_1']       # A2C vs. DQN sync
    # data_dirs = ['temp_dqn_async_cartpole_1', 'temp_a2c_vtrace_async_cartpole_2']     # A3C vs. DQN async
    data_dirs = ['temp_a2c_vtrace_sync_cartpole_1', 'temp_a2c_vtrace_async_cartpole_2', 'temp_dqn_sync_cartpole_2', 'temp_dqn_async_cartpole_1']

    for dir in data_dirs:
        current_dir = base_dir + dir + '/EXP000/RUN000/'
        print('Data path:{}'.format(current_dir))
        merged_df = get_merged_df(current_dir)
        merged_df['total_reward_roll'] = merged_df['total_reward'].rolling(rolling_setting).mean()

        if x_tick_type == "time":
            plt.plot(merged_df['rel_time'], merged_df['total_reward_roll'], label='{}'.format(dir[5:-2]))
            plt.xlabel('Relative Time')
        else:
            plt.plot(np.linspace(0, len(merged_df), len(merged_df)), merged_df['total_reward_roll'], label='{}'.format(dir[5:-2]))
            plt.xlabel('Episode Count')

    plt.ylabel('Rolling Total Reward ({})'.format(rolling_setting))
    plt.legend(loc="lower right")
    fig.savefig(base_dir + 'MultiReward_plot_all.png')
