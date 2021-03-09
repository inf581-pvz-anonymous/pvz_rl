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
# from game_render import render


def evaluate(env, agent, n_iter=1000, verbose = True):
    sum_score = 0
    sum_iter = 0
    score_hist = []
    iter_hist = []
    n_iter = n_iter
    actions = []

    for episode_idx in range(n_iter):
        if verbose:
            print("\r{}/{}".format(episode_idx, n_iter), end="")
        
        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])

        score_hist.append(summary['score'])
        iter_hist.append(min(env.env._scene._chrono, config.MAX_FRAMES))
        
        sum_score += summary['score']
        sum_iter += min(env.env._scene._chrono, config.MAX_FRAMES)
        
        # if env.env._scene._chrono >= 1000:
        #    render_info = env.env._scene._render_info
        #    render(render_info)
        #    input()
        actions.append(summary['actions'])

    actions = np.concatenate(actions)
    plant_action = np.mod(actions - 1, 4)
    if verbose:
        # Plot of the score
        plt.hist(score_hist)
        plt.title("Score per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the iterations
        plt.hist(iter_hist)
        plt.title("Survived frames per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the action
        plt.hist(np.concatenate(actions), np.arange(0, config.N_LANES * config.LANE_LENGTH * 4 + 2) -0.5, density=True)
        plt.title("Action usage density over {} plays".format(n_iter))
        plt.show()
        plt.hist(plant_action, np.arange(0,5) - 0.5, density=True)
        plt.title("Plant usage density over {} plays".format(n_iter))
        plt.show()

    return sum_score/n_iter, sum_iter/n_iter
