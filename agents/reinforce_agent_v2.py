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

class PolicyNetV2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(PolicyNetV2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class ReinforceAgentV2():
    def __init__(self,input_size, possible_actions):
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.possible_actions = possible_actions
        self.policy = PolicyNetV2(input_size, output_size=len(possible_actions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3,)
        self.n_plants = 4

    def decide_action(self, observation):
        mask = self._get_mask(observation)
        # predict probabilities for actions
        var_s = Variable(torch.from_numpy(observation.astype(np.float32)))
        action_prob = torch.exp(self.policy.forward(var_s))
        action_prob[np.logical_not(mask)] = 0
        action_prob /= torch.sum(action_prob[mask])
        # select random action weighted by probabilities
        action =  np.random.choice(self.possible_actions, 1, p=action_prob.data.numpy())[0]
        return action

    def discount_rewards(self,r,gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r

    def iterate_minibatches(self, observation, actions, rewards, batchsize, shuffle=False):
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
            mask_batch = torch.Tensor([self._get_mask(s) for s in observation_batch]).type(torch.BoolTensor).detach()
            
            s_var =  Variable(torch.from_numpy(observation_batch.astype(np.float32)))
            a_var = Variable(torch.from_numpy(action_batch).view(-1).type(torch.LongTensor))
            A_var = Variable(torch.from_numpy(reward_batch.astype(np.float32)))
            
            pred = self.policy.forward(s_var)
            pred = pred / torch.Tensor([torch.sum(pred[i,:][mask_batch[i,:]]) for i in range(len(pred))]).view(-1,1)
            
            loss += F.nll_loss(pred * A_var,a_var)

        loss.backward(loss)
        self.optimizer.step()

    def save(self, nn_name):
        torch.save(self.policy, nn_name)

    def load(self, nn_name):
        self.policy = torch.load(nn_name)

    def _get_mask(self, observation):
        empty_cells = np.nonzero((observation[:self._grid_size]==0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(len(self.possible_actions), dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * self.n_plants

        available_plants = observation[-self.n_plants:]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask




class PlayerV2():
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


if __name__ == "__main__":

    env = PlayerV2(render=False,max_frames = 1000)
    agent = DiscreteAgentV2(
        input_size=env.num_observations(),
        possible_actions=env.get_actions()
    )
    # agent.policy = torch.load("saved/policy13_v2")
    sum_score = 0
    sum_iter = 0
    score_plt = []
    iter_plt = []
    eval_score_plt = []
    eval_iter_plt = []
    n_iter = 100000
    n_record = 500
    n_save = 1000
    n_evaluate = 10000
    n_iter_evaluation = 1000
    save = False
    best_score = None
    #threshold = Threshold(seq_length = n_iter, start_epsilon=0.1,
    #                      end_epsilon=0.0,interpolation='sinusoidal',
    #                      periods=np.floor(n_iter/(8*n_record)))
    # threshold = Threshold(seq_length = n_iter, start_epsilon=0.005, end_epsilon=0.005)

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
                    torch.save(agent.policy, nn_name)
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

        if (episode_idx%n_evaluate == n_evaluate-1):
            avg_score, avg_iter = evaluate(env, agent, n_iter_evaluation)
            print("\n----------->Episode {}, mean score {}".format(episode_idx,avg_score))
            print("----------->n_iter {}".format(avg_iter))
            eval_score_plt.append(avg_score)
            eval_iter_plt.append(avg_iter)
            # input()
        

    plt.plot(range(n_record, n_iter+1, n_record), score_plt)
    plt.plot(range(n_evaluate, n_iter+1, n_evaluate), eval_score_plt, color='red')
    plt.show()
    plt.plot(range(n_record, n_iter+1, n_record), iter_plt)
    plt.plot(range(n_evaluate, n_iter+1, n_evaluate), eval_iter_plt, color='red')
    plt.show()
