import sys
import gc

import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm
import shap
import numpy as np

from agents import QNetwork, PlayerQ
from pvz import config

n_ep = 100
obs = []

DEVICE = "cpu"
agent = torch.load("agents/agent_zoo/dfq5_epsexp").to(DEVICE)
player = PlayerQ(render=False)

for episode_idx in range(n_ep):
    print("\r{}/{}".format(episode_idx, n_ep), end="")
    summary = player.play(agent)
    obs.append(summary["observations"])

_grid_size = config.N_LANES * config.LANE_LENGTH

obs = np.concatenate(obs)
obs = np.array([np.concatenate([state[:_grid_size], 
                       np.sum(state[_grid_size: 2 * _grid_size].reshape(-1, config.LANE_LENGTH), axis=1), 
                       state[2 * _grid_size:]]) for state in obs])

n_obs = len(obs)

e = shap.DeepExplainer(
        agent.network, 
        torch.from_numpy(
            obs[np.random.choice(np.arange(len(obs)), 100, replace=False)]
        ).type(torch.FloatTensor).to(DEVICE))

shap_values = e.shap_values(
    torch.from_numpy(obs[np.random.choice(np.arange(len(obs)), 30, replace=False)]).type(torch.FloatTensor).to(DEVICE)
)

s = np.stack([np.sum(s, axis=0) for s in shap_values])
print(np.sum(s, axis=0))
shap.summary_plot(shap_values)
