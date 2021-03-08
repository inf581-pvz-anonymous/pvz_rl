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
# from .evaluate_agent import evaluate
# from .threshold import Threshold

HP_NORM = 100
SUN_NORM = 200

class PolicyNetR(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(PolicyNetR, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, verbose=False):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class DiscreteAgentR():
    def __init__(self,input_size, possible_actions):
        self.possible_actions = possible_actions
        self.policy = PolicyNetR(input_size, output_size=len(possible_actions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

    def decide_action(self, observation, mask):
        # predict probabilities for actions
        var_s = Variable(torch.from_numpy(observation.astype(np.float32)))

        action_prob = self.policy.forward(var_s.view(1,-1)).view(-1)[mask]
        sum_proba = torch.sum(action_prob)
        if (sum_proba == 0):
            action =  np.random.choice(np.array(self.possible_actions)[mask], 1)[0]
        else:
            if len(torch.nonzero(torch.isnan(action_prob))):
                for p in self.policy.parameters():
                    print(p)
            action_prob /= torch.sum(action_prob)
            if len(torch.nonzero(torch.isnan(action_prob))):
                print(action_prob)
            # select random action weighted by probabilities
            action =  np.random.choice(np.array(self.possible_actions)[mask], 1, p=action_prob.data.numpy())[0]
        return action

    def discount_rewards(self,r,gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r

    def iterate_minibatches(self, observation, actions, rewards,  batchsize, shuffle=False):
        assert len(observation) == len(actions)
        assert len(observation) == len(rewards)

        indices = np.arange(len(observation))
        if shuffle:
            np.random.shuffle(indices)
        #import pdb; pdb.set_trace()
        for start_idx in range(0, len(observation), batchsize):
            if shuffle:
                excerpt = indices[start_idx:min(start_idx + batchsize, len(indices))]
            yield observation[excerpt], actions[excerpt], rewards[excerpt]

    def update(self,observation, actions, rewards):
        # discounted reward
        rewards = self.discount_rewards(rewards, gamma = 0.99)
        self.optimizer.zero_grad()
        # L = log π(a | s ; θ)*A
        loss = 0
        for observation_batch, action_batch, reward_batch in self.iterate_minibatches(observation, actions, rewards, batchsize = 100, shuffle=True):
            #import pdb; pdb.set_trace()
            s_var =  Variable(torch.from_numpy(observation_batch.astype(np.float32)))
            a_var = Variable(torch.from_numpy(action_batch).view(-1).type(torch.LongTensor))
            A_var = Variable(torch.from_numpy(reward_batch.astype(np.float32)))
            pred = torch.log(self.policy.forward(s_var))
            loss += F.nll_loss(pred * A_var,a_var)

        loss.backward(loss)
        self.optimizer.step()

    def save(self, nn_name):
        torch.save(self.policy, nn_name)

    def load(self, nn_name):
        self.policy = torch.load(nn_name)




class TrainerR():
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
        observation = observation.astype(np.float64)
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

    def play(self,agent, epsilon=0):
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
            if np.random.random()<epsilon:
                # print("exploration")
                action = np.random.choice(self.get_actions(), 1)[0]
            else:
                mask = self.env.mask_available_actions()
                action = agent.decide_action(observation, mask)

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

