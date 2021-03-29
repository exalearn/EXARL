import pandas as pd
import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


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

def save_reward_plot():
    df_ranks = []
    rank = 0
    # Candle directory stucture
    results_dir = cd.run_params['output_dir'] + '/'
    for filename in os.listdir(results_dir):
        if filename.endswith(".log"):
            rank += 1
            logger.info('rank {}: filename:{}'.format(rank, filename))
            df = read_data(results_dir + filename, rank)
            df_ranks.append(df)

    df_merged = pd.concat(df_ranks)
    df_merged = df_merged.dropna()
    time_min = df_merged.time.min()
    time_max = df_merged.time.max()
    logger.info('time_min', time_min)
    logger.info('time_diff', time_max - time_min)
    df_merged['rel_time'] = [idx - time_min for idx in df_merged.time]
    df_merged.sort_values(by=['rel_time'], inplace=True)

    rolling_setting = 25
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    episodes_per_nodes = []
    logger.info('Node path:{}'.format(results_dir))
    df_merged['total_reward_roll'] = df_merged['total_reward'].rolling(rolling_setting).mean()
    logger.info((df_merged.shape))
    plt.plot(df_merged['rel_time'], df_merged['total_reward_roll'])
    episodes_per_nodes.append(len(df_merged))
    plt.xlabel('Relative Time')
    plt.ylabel('Rolling Total Reward ({})'.format(rolling_setting))
    if not os.path.exists(results_dir + '/Plots'):
        os.makedirs(results_dir + '/Plots')
    fig.savefig(results_dir + '/Plots/Reward_plot.png')
