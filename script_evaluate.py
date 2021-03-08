
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

from agents import evaluate, TrainerV2_1, DiscreteAgentV2_1
from agents import TrainerV2, DiscreteAgentV2
from agents.ddqn_agent import QNetwork, PlayerQ

if __name__=="__main__":
    player = TrainerV2(render=False, max_frames = 400)
    agent = DiscreteAgentV2(
             input_size=player.num_observations(),
             possible_actions=player.get_actions()
    )
    # agent = DiscreteAgentV2_1(
    #          n_plants=4,
    #          possible_actions=env.get_actions()
    # )
    # PolicyNet=PolicyNetV2
    agent.load("agents/benchmark/dfp5")
    # player = PlayerQ(render=False)
    # agent = torch.load("dfp5_masked")
    print(agent)
    # agent.coeff = torch.full((9,), 1.0)
    avg_score, avg_iter = evaluate(player, agent)
    print("\nMean score {}".format(avg_score))
    print("Mean iterations {}".format(avg_iter))