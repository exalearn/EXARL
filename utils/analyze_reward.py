import pandas as pd
import os
import matplotlib.pyplot as plt


def read_data(filename):
    frame = pd.read_csv(filename, sep=' ',
                        header=None,
                        names=['time', 'current_state', 'action', 'reward', 'next_state', 'total_reward', 'done',
                               'episode', 'step', 'policy_type', 'epsilon'])

    ## Make time relative to the start time ##
    frame['time'] = pd.to_datetime(frame['time'], unit='ns')
    frame = frame[frame.done == True]
    return frame


## Can all log files
def save_reward_plot(results_dir):
    df_ranks = []
    # Candle directory stucture
    results_dir = results_dir
    for filename in os.listdir(results_dir):
        if filename.endswith(".log"):
            print('filename:{}'.format(filename))
            df = read_data(results_dir + filename)
            df_ranks.append(df)
            print(df.head())

    # df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time'],how='outer'), df_ranks)
    df_merged = pd.concat(df_ranks)
    df_merged.set_index('episode', inplace=True)
    df_merged = df_merged.reset_index()
    fig,ax = plt.subplots(figsize=(14,12))
    ax.plot(df_merged['total_reward'], label='Total Reward', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    if not os.path.exists(results_dir+'/Plots'):
        os.makedirs(results_dir+'/Plots')
    fig.savefig(results_dir+'/Plots/Reward_plot.png') 
