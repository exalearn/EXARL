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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exarl.utils.globals import ExaGlobals

def read_data(filename):
    """The function reads csv-based learning data from the given log file into a pandas frame for use in plotting and result analysis.

    Parameters
    ----------
    filename : string
        csv file of log data from EXARL

    Returns
    -------
    pandas.DataFrame
        contains data extracted from the csv-based EXARL learning log file,
        except for current_state and next_state fields, and with the addition of a rank field.
    """
    frame = pd.read_csv(filename, sep=' ', header=None,
                        names=['time', 'current_state', 'action', 'reward',
                               'next_state', 'total_reward', 'done', 'episode',
                               'step', 'policy_type'])

    del frame['current_state']
    del frame['next_state']

    parts = os.path.basename(filename).split("_")
    rank = [int(part[4:]) for part in parts if "Rank" in part]
    frame['rank'] = rank[0]

    frame['time'] = pd.to_datetime(frame['time'], unit='s')
    frame = frame[frame.done == True]
    frame = frame.reset_index()
    return frame

def save_reward_plot():
    """Creates and saves a Rolling Total Reward (y-axis) by Relative Time (x-axis) plot based on .log files written by EXARL for each rank.
        It saves the plot in the results directory named by the output_dir run parameter in a subdirectory /Plots/reward_plot.png.
        It then tries to print the plot to the terminal.
    """
    # Candle directory stucture
    results_dir = ExaGlobals.lookup_params('output_dir')
    plot_path = os.path.join(results_dir, 'Plots')
    os.makedirs(plot_path, exist_ok=True)

    files = [filename for filename in os.listdir(results_dir) if filename.endswith(".log")]
    df_ranks = [read_data(os.path.join(results_dir, filename)) for filename in files]

    df_merged = pd.concat(df_ranks)
    df_merged = df_merged.dropna()
    time_min = df_merged.time.min()
    df_merged['rel_time'] = [idx - time_min for idx in df_merged.time]
    df_merged.sort_values(by=['rel_time'], inplace=True)

    rolling_setting = ExaGlobals.lookup_params('rolling_reward_length')
    df_merged['total_reward_roll'] = df_merged['total_reward'].rolling(rolling_setting).mean()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.plot(df_merged['rel_time'], df_merged['total_reward_roll'])
    plt.xlabel('Relative Time')
    plt.ylabel('Rolling Total Reward ({})'.format(rolling_setting))
    fig.savefig(os.path.join(plot_path, 'Reward_plot.png'))
    plt.clf()

    ranks = df_merged['rank'].unique()
    ranks.sort()
    for rank in ranks:
        to_plot = df_merged[df_merged['rank'] == rank]
        to_plot = to_plot.sort_values(by=['rel_time'])
        plt.plot(to_plot['rel_time'], to_plot['total_reward_roll'], label=rank)
    plt.xlabel('Relative Time')
    plt.ylabel('Rolling Total Reward ({})'.format(rolling_setting))
    plt.legend()
    fig.savefig(os.path.join(plot_path, 'Rank_plot.png'))
    plt.clf()

    # Terminal plot
    try:
        import plotille
        figure = plotille.Figure()
        figure.width = 60
        figure.height = 30
        figure.y_label = 'Rolling reward'
        figure.x_label = 'Episodes'
        figure.color_mode = 'byte'

        to_plot = df_merged['total_reward_roll'].dropna()
        x = list(to_plot.index)
        y = list(to_plot.values)
        figure.set_x_limits(min_=x[0], max_=x[-1])
        figure.set_y_limits(min_=min(y), max_=max(y))
        figure.plot(x, y, lc=200, label='rolling reward')
        print(figure.show(legend=True))
    except ModuleNotFoundError:
        print("Plottile not installed.")
