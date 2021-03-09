
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
from agents import ACAgent, TrainerAC

agent_type = "DDQN" # DDQN or Reinforce or Keyboard


if __name__ == "__main__":

    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames = 500 * config.FPS)
        agent = ReinforceAgentV2(
                input_size=env.num_observations(),
                possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/dfp5")
         
    if agent_type == "AC":
        env = TrainerAC(render=False, max_frames = 500*config.FPS)
        agent = ACAgent(
                input_size=env.num_observations(),
                possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/ac_policy_v1", "agents/agent_zoo/ac_value_v1")
        
    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        agent = torch.load("agents/agent_zoo/dfq5_epsexp")
    
    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames = 500*config.FPS)
        agent = KeyboardAgent()

        
    avg_score, avg_iter = evaluate(env, agent)
    print("\nMean score {}".format(avg_score))
    print("Mean iterations {}".format(avg_iter))
