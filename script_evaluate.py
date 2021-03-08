
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

from agents import evaluate, PlayerV2, ReinforceAgentV2
from agents import QNetwork, PlayerQ
from agents import QNetwork_DQN

if __name__=="__main__":
#     player = TrainerV2(render=False, max_frames = 400)
#     agent = DiscreteAgentV2(
#              input_size=player.num_observations(),
#              possible_actions=player.get_actions()
#     )
#     agent.load("agents/benchmark/dfp5")
    player = PlayerQ(render=False)
    agent = torch.load("agents/agent_zoo/dfq5_epsexp")
    avg_score, avg_iter = evaluate(player, agent)
    print("\nMean score {}".format(avg_score))
    print("Mean iterations {}".format(avg_iter))
