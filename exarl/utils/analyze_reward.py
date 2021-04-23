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
import pandas as pd
import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
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
    time_diff = time_max - time_min
    logger.info('time_min:{}'.format(time_min))
    logger.info('time_diff:{}'.format(time_diff))
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
