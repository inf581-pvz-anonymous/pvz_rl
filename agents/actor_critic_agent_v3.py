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
from torch.distributions import Categorical

HP_NORM = 100
SUN_NORM = 200

class PolicynetAC3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=80):
        super(PolicynetAC3, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob

class ValuenetAC3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=80):
        super(ValuenetAC3, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.affine1(x))
        state_value = self.value_head(x)
        return state_value


class ACAgent3():
    def __init__(self,input_size, possible_actions):
        self.possible_actions = possible_actions
        self.policy = PolicynetAC3(input_size, output_size=len(possible_actions))
        self.valuenet = ValuenetAC3(input_size, output_size=len(possible_actions))
        self.optimizer1 = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.optimizer2 = optim.Adam(self.valuenet.parameters(), lr=1e-4)
        self.saved_actions = []

    def decide_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy(state)
        state_value = self.valuenet(state)
        # Create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # Sample an action using the distribution
        action = m.sample()
        # Save to action buffer
        self.saved_actions.append((m.log_prob(action), state_value))
        # Return the action to take
        return action.item()

    def discount_rewards(self,r,gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r


    def update(self,observation, actions, rewards):
        # Discount rewards through the whole episode
        rewards = (torch.tensor(self.discount_rewards(rewards, gamma = 0.99))).float()
        saved_actions = self.saved_actions
        policy_losses = [] # List to save actor (policy) loss
        value_losses = [] # List to save critic (value) loss
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        # Store the losses
        for  (log_prob, value), R in zip(saved_actions, rewards):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # Compute both losses and backpropagate
        loss = torch.stack(policy_losses).sum()
        loss.backward(loss)
        self.optimizer1.step()
        loss = torch.stack(value_losses).sum()
        loss.backward(loss)
        self.optimizer2.step()
        self.saved_actions = []

    def save(self, nn_name_1, nn_name_2):
        torch.save(self.policy, nn_name_1)
        torch.save(self.valuenet, nn_name_2)


    def load(self, nn_name_1, nn_name_2):
        self.policy = torch.load(nn_name_1)
        self.valuenet = torch.load(nn_name_2)




class TrainerAC3():
    def __init__(self,render=True, max_frames = 1000, n_iter = 100000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH


    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(self.env.plant_deck) + 1

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        observation_zombie = self._grid_to_lane(observation[self._grid_size:2*self._grid_size])
        observation = np.concatenate([observation[:self._grid_size], observation_zombie,
        [observation[2 * self._grid_size]/SUN_NORM],
        observation[2 * self._grid_size+1:]])
        if self.render:
            print(observation)
        return observation

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM

    def play(self,agent):
        """ Play one episode and collect observations and rewards """

        summary = dict()
        summary['rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        observation = self._transform_observation(self.env.reset())

        t = 0

        while(self.env._scene._chrono<self.max_frames):
            if(self.render):
                self.env.render()

            action = agent.decide_action(observation)

            summary['observations'].append(observation)
            summary['actions'].append(action)
            observation, reward, done, info = self.env.step(action)
            observation = self._transform_observation(observation)
            summary['rewards'].append(reward)

            if done:
                break

        summary['observations'] = np.vstack(summary['observations'])
        summary['actions'] = np.vstack(summary['actions'])
        summary['rewards'] = np.vstack(summary['rewards'])
        return summary

    def get_render_info(self):
        return self.env._scene._render_info