from agents import evaluate
import gym
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pvz import config
import matplotlib.pyplot as plt


def train(env, agent, n_iter=100000, n_record=500, n_save=1000):
    sum_score = 0
    sum_iter = 0
    score_plt = []
    iter_plt = []
    eval_score_plt = []
    eval_iter_plt = []
    save = False
    best_score = None

    for episode_idx in range(n_iter):

        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])

        sum_score += summary['score']
        sum_iter += min(env.env._scene._chrono, env.max_frames)

        # Update agent
        agent.update(summary["observations"],summary["actions"],summary["rewards"])

        if (episode_idx%n_record == n_record-1):
            if save:
                if sum_score >= best_score:
                    agent.save(nn_name1, nn_name2)
                    best_score = sum_score
            print("---Episode {}, mean score {}".format(episode_idx,sum_score/n_record))
            print("---n_iter {}".format(sum_iter/n_record))
            score_plt.append(sum_score/n_record)
            iter_plt.append(sum_iter/n_record)
            sum_iter = 0
            sum_score = 0
            # input()
        if not save:
            if (episode_idx%n_save == n_save-1):
                s = input("Save? (y/n): ")
                if (s=='y'):
                    save = True
                    best_score = 0
                    nn_name1 = input("Save name for policy net: ")
                    nn_name2 = input("Save name for value net: ")

    plt.figure(200)
    plt.plot(range(n_record, n_iter+1, n_record), score_plt)
    plt.show()
    plt.figure(300)
    plt.plot(range(n_record, n_iter+1, n_record), iter_plt)
    plt.show()


# Import your agent
from agents import ACAgent3, TrainerAC3

if __name__ == "__main__":

    env = TrainerAC3(render=False,max_frames = 400)
    agent = ACAgent3(
        input_size = env.num_observations(),
        possible_actions=env.get_actions()
    )
    train(env, agent)




