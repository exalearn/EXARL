import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(filename):
    frame = pd.read_csv(filename, sep=' ',
                        header=None,
                        names=['time', 'current_state', 'action', 'reward', 'next_state', 'total_reward', 'done',
                               'episode', 'step', 'policy_type', 'epsilon'])

    # Make time relative to the start time
    frame['time'] = pd.to_datetime(frame['time'], unit='ns')
    # TODO the time alignment after the merge
    frame['rel_time'] = [idx - frame.time[0] for idx in frame.time]
    frame['rel_time'] = frame['rel_time'].values.astype(float)
    frame = frame[frame.done == True]
    return frame


# Can all log files
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

    df_merged = pd.concat(df_ranks)
    df_merged.set_index('episode', inplace=True)
    df_merged = df_merged.reset_index()
    fig, ax = plt.subplots(figsize=(14, 12))
    ax = sns.scatterplot(x="rel_time", y="total_reward", data=df_merged)
    plt.xlabel('Time')
    plt.ylabel('Total Reward')
    if not os.path.exists(results_dir + '/Plots'):
        os.makedirs(results_dir + '/Plots')
    fig.savefig(results_dir + '/Plots/Reward_plot.png')
