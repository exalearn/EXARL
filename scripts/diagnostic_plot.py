import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import array

names_short = ['time', 'current_state', 'action', 'reward', 'next_state', 'total_reward', 'done','episode', 'step', 'policy_type', 'epsilon']
names_long  = ['time', 'current_state', 'action', 'reward', 'next_state', 'total_reward', 'done','episode', 'step', 'policy_type', 'epsilon', "critic_loss", "actor_loss"]

class log_plotter:
    def __init__(self, filename):
        filenames = glob.glob(filename)
        self.dfs  = []

        self.has_losses = False
        for fn in filenames:
            new_df = pd.read_csv(fn, sep=' ', header=None)
            if new_df.shape[1] == 11:
                new_df.columns = names_short
                self.dfs.append(new_df)
            elif new_df.shape[1] == 13:
                new_df.columns = names_long
                self.dfs.append(new_df)
                self.has_losses = True
            else:
                print("!!!!!!!!!!!The number of columns in the log file is inconsistent with either naming scheme!!!!!!!!!!!")
                self.dfs.append(new_df)

        tmp_lab = []
        for fn in filenames:
            tmp_lab = tmp_lab + fn.split("/")
        tmp_lab, tmp_c = np.unique(tmp_lab, return_counts=True)
        tmp_lab = tmp_lab[tmp_c == 1]
        self.labels = [[x for x in tmp_lab if x+"/" in y][0] for y in filenames] 
        self.n_cases = len(self.labels)

    def plot_rewards(self):
        episodes      = []
        total_rewards = []
        
        for case in self.dfs:
            max_ep = case["episode"].max()
            tr = []
            ep = []
            for jj in range(max_ep):
                tr.append(case["total_reward"][case["episode"] == jj + 1 ].values[-1])
                ep.append(jj)
            episodes.append(np.array(ep))
            total_rewards.append(np.array(tr))
        
        plt.figure(dpi=200)
        for ii in range(len(episodes)):
            plt.plot(episodes[ii], total_rewards[ii], linewidth=3, label=self.labels[ii], alpha=0.3)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend(loc="lower right")
        plt.savefig("rewards.png")

    def plot_reward_by_episode(self):
        for ii in range(self.n_cases):
            plt.figure(dpi=200)
            max_ep = self.dfs[ii]["episode"].max()
            for jj in range(max_ep):
                x = self.dfs[ii]["step"][self.dfs[ii]["episode"] == jj + 1 ]
                y = self.dfs[ii]["reward"][self.dfs[ii]["episode"] == jj + 1 ]
                plt.plot(x,y, alpha = 0.3, linewidth=3, c=(1 - jj/(max_ep-1), 0., jj/(max_ep-1)))
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.title( self.labels[ii] )
            plt.savefig("reward_by_episode_"+self.labels[ii]+".png")

    def plot_critic_loss(self):
        for ii in range(self.n_cases):
            plot_df = self.dfs[ii]
            plt.plot(plot_df["critic_loss"], label=str(self.labels[ii]), alpha=0.3)
            plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Critic Loss")
        plt.legend()
        plt.savefig("critic_loss.png")

    def plot_actor_loss(self):
        for ii in range(self.n_cases):
            plot_df = self.dfs[ii]
            plt.plot(plot_df["actor_loss"], label=str(self.labels[ii]), alpha=0.3)
            plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Actor Loss")
        plt.legend()
        plt.savefig("actor_loss.png")

    def plot_actions(self):
        for ii in range(self.n_cases):
            plt.figure(dpi=200)
            plot_df = self.dfs[ii]
            max_ep  = plot_df["episode"].max()
            for ep in range(max_ep):
                ep_frame = plot_df.loc[plot_df["episode"] == ep,: ]
                actions  = np.vstack([eval(x) for x in ep_frame["action"]])
                plt.plot(actions, c=(1 - ep/(max_ep), 0., ep/(max_ep)));
            plt.xlabel("Step")
            plt.ylabel("Action")
            plt.title( self.labels[ii] )
            plt.savefig("actions_"+self.labels[ii]+".png")


def main(argv):
    parser        = argparse.ArgumentParser(description="Script to parse ExaRL logs and generate diagnostic plots")
    # required_args = parser.add_argument_group("required named arguments")

    parser.add_argument("--filename", "-f", help="Filename for log to parse and plot")
    parser.add_argument("--rewards", "-r", help="Make plot comparing reward curves across files", action='store_true')
    parser.add_argument("--reward-by-episode", "-e", help="Make plot of rewards by episode for each file", action='store_true')
    parser.add_argument("--critic-loss", "-c", help="Make plot comparing critic loss curves across files", action='store_true')
    parser.add_argument("--actor-loss",  "-a", help="Make plot comparing actor loss curves across files", action='store_true')
    parser.add_argument("--actions", help="Make plot comparing action curves by episode for each file", action='store_true')

    args              = parser.parse_args()
    filename          = args.filename
    plot_rewards      = args.rewards
    plot_reward_by_ep = args.reward_by_episode
    plot_critic       = args.critic_loss
    plot_actor        = args.actor_loss
    plot_actions      = args.actions

    plotter = log_plotter(filename)
    
    if plot_rewards:
        plotter.plot_rewards()
    
    if plot_reward_by_ep:
        plotter.plot_reward_by_episode()

    if plot_critic:
        if not plotter.has_losses:
            print("No Critic Losses In Log File!")
        else:
            plotter.plot_critic_loss()

    if plot_actor:
        if not plotter.has_losses:
            print("No Actor Losses In Log File!")
        else:
            plotter.plot_actor_loss()

    if plot_actions:
        plotter.plot_actions()

if __name__ == '__main__':
    main(sys.argv[1:])
