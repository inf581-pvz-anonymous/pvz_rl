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


# Import your agent

from agents import ReinforceAgentV2, PlayerV2


def train(env, agent, n_iter=100000, n_record=500, n_save=1000, n_evaluate=10000, n_iter_evaluation=1000):
    sum_score = 0
    sum_iter = 0
    score_plt = []
    iter_plt = []
    eval_score_plt = []
    eval_iter_plt = []
    # threshold = Threshold(seq_length = n_iter, start_epsilon=0.005, end_epsilon=0.005)
    save = False
    best_score = None

    for episode_idx in range(n_iter):
        
        # play episodes
        # epsilon = threshold.epsilon(episode_idx)
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])
        # print("Episode {}, mean score {}".format(episode_idx,summary['score']))
        # print("n_iter {}".format(summary['rewards'].shape[0]))

        sum_score += summary['score']
        sum_iter += min(env.env._scene._chrono, env.max_frames)

        # Update agent
        agent.update(summary["observations"],summary["actions"],summary["rewards"])
        # print(agent.policy(torch.from_numpy(np.random.random(env.num_observations())).type(torch.FloatTensor)))

        if (episode_idx%n_record == n_record-1):
            if save:
                if sum_score >= best_score:
                    agent.save(nn_name)
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
                    nn_name = input("Save name: ")

        # if (episode_idx%n_evaluate == n_evaluate-1):
        #     avg_score, avg_iter = evaluate(env, agent, n_iter_evaluation)
        #     print("\n----------->Episode {}, mean score {}".format(episode_idx,avg_score))
        #     print("----------->n_iter {}".format(avg_iter))
        #     eval_score_plt.append(avg_score)
        #     eval_iter_plt.append(avg_iter)
        #     # input()
    plt.plot(range(n_record, n_iter+1, n_record), score_plt)
    # plt.plot(range(n_evaluate, n_iter+1, n_evaluate), eval_score_plt, color='red')
    plt.show()
    plt.plot(range(n_record, n_iter+1, n_record), iter_plt)
    # plt.plot(range(n_evaluate, n_iter+1, n_evaluate), eval_iter_plt, color='red')
    plt.show()
    evaluate(env, agent)


if __name__ == "__main__":

    env = PlayerV2(render=False,max_frames = 400)
    agent = ReinforceAgentV2(
        input_size = env.num_observations(),
        possible_actions=env.get_actions()
    )
    # agent.policy = torch.load("saved/policy13_v2")
    
    train(env, agent)

    
        

    
