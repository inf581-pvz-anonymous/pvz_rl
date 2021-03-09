import torch.nn as nn
import torch
import torch.nn.functional as F
import gym
from pvz import config
from copy import deepcopy
from collections import namedtuple, deque
import numpy as np
from .threshold import Threshold
from . import evaluate

HP_NORM = 1
SUN_NORM = 200

def sum_onehot(grid):
    return torch.cat([torch.sum(grid==(i+1), axis=-1).unsqueeze(-1) for i in range(4)], axis=-1)


class QNetwork_DQN(nn.Module):
    
    def __init__(self, env, epsilon=0.05, learning_rate=1e-3, device='cpu', use_zombienet=True, use_gridnet=True):
        super(QNetwork_DQN, self).__init__()
        self.device = device

        self.n_inputs = config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(env.plant_deck) + 1
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self._grid_size = config.N_LANES * config.LANE_LENGTH

        # TODO
        self.use_zombienet = use_zombienet
        if use_zombienet:
            self.zombienet_output_size = 1
            self.zombienet = ZombieNet(output_size=self.zombienet_output_size)
            self.n_inputs += (self.zombienet_output_size - 1) * config.N_LANES

        self.use_gridnet = use_gridnet
        if use_gridnet:
            self.gridnet_output_size=4
            self.gridnet = nn.Linear(self._grid_size, self.gridnet_output_size)
            self.n_inputs += self.gridnet_output_size - self._grid_size
            # self.gridnet = sum_onehot

        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 50, bias=True),
            nn.LeakyReLU(),
            nn.Linear(50, self.n_outputs, bias=True))

        # Set to GPU if cuda is specified
        if self.device == 'cuda':
            self.network.cuda()
            
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.learning_rate)
        
    def decide_action(self, state, mask, epsilon):
        # mask = self.env.mask_available_actions()
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions[mask])
        else:
            action = self.get_greedy_action(state, mask)
        return action
    
    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        qvals[np.logical_not(mask)] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state, use_zombienet=True):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[:, self._grid_size:(2 * self._grid_size)].reshape(-1, config.LANE_LENGTH)
            plant_grid = state_t[:, :self._grid_size]
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1, self.zombienet_output_size * config.N_LANES)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1).view(-1, config.N_LANES)
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            state_t = torch.cat([plant_grid, zombie_grid, state_t[:,2 * self._grid_size:]], axis=1)
        else:
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[self._grid_size:(2 * self._grid_size)].reshape(-1, config.LANE_LENGTH)
            plant_grid = state_t[:self._grid_size]
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1)
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            state_t = torch.cat([plant_grid, zombie_grid, state_t[2 * self._grid_size:]])
        return self.network(state_t)


class ZombieNet(nn.Module):
    def __init__(self, output_size=1, hidden_size=5):
        super(ZombieNet, self).__init__()
        self.fc1 = nn.Linear(config.LANE_LENGTH, output_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = F.leaky_relu(self.fc1(x))
        # x = self.fc2(x)
        return self.fc1(x)

class DQNAgent:
    
    def __init__(self, env, network, buffer, n_iter = 100000, batch_size=32):
        
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        # self.pre_buffer = []
        # self.pre_buffer_rewards = []
        # self.threshold = Threshold(seq_length = 100000, start_epsilon=1.0,
        #                   end_epsilon=0.2,interpolation='sinusoidal',
        #                   periods=np.floor(n_iter/100))
        self.threshold = Threshold(seq_length = 100000, start_epsilon=1.0, interpolation="exponential",
                           end_epsilon=0.05)
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 30000
        self.initialize()
        self.player = PlayerQ_DQN(env = env, render=False)
        

    def take_step(self, mode='train'):
        # mask = np.full(self.env.mask_available_actions().size(), True)
        mask = np.array(self.env.mask_available_actions())
        if mode == 'explore':
            if np.random.random()<0.5:
                action=0 # Do nothing
            else:
                action = np.random.choice(np.arange(self.env.action_space.n)[mask])
        else:
            action = self.network.decide_action(self.s_0, mask, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        s_1 = self._transform_observation(s_1)
        self.rewards += r
        # self.pre_buffer.append((self.s_0, action, done, s_1))
        # self.pre_buffer_rewards.append(r)
        self.buffer.append(self.s_0, action, r, done, s_1)
        self.s_0 = s_1.copy()
        if done:
            if mode != "explore": # We document the end of the play
                self.training_iterations.append(min(config.MAX_FRAMES, self.env._scene._chrono))
            self.s_0 = self._transform_observation(self.env.reset())
        return done
    
    # def add_play_to_buffer(self):
    #     rewards = self.discount_rewards(np.array(self.pre_buffer_rewards))
    #     for i in range(len(rewards)):
    #         s_0, action, done, s_1 = self.pre_buffer[i]
    #         r = rewards[i]
    #         self.buffer.append(s_0, action, r, done, s_1)
    #     self.pre_buffer_rewards = []
    #     self.pre_buffer = []
        
    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=100000,
              network_update_frequency=32,
              network_sync_frequency=2000,
              evaluate_frequency=500,
              evaluate_n_iter=1000):

        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            # if done:
            #     self.add_play_to_buffer()
        ep = 0
        training = True
        self.s_0 = self._transform_observation(self.env.reset())


        while training:
            self.rewards = 0
            done = False
            while done == False:
                self.epsilon = self.threshold.epsilon(ep)
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    self.sync_eps.append(ep)
                    
                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)

                    mean_iteration = np.mean(
                        self.training_iterations[-self.window:])
                    self.mean_training_iterations.append(mean_iteration)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                        ep, mean_rewards,mean_iteration), end="")
                    
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break
                    if (ep%evaluate_frequency) == evaluate_frequency - 1:
                        avg_score, avg_iter = evaluate(self.player, self.network, n_iter = evaluate_n_iter, verbose=False)
                        self.real_iterations.append(avg_iter)
                        self.real_rewards.append(avg_score)


                    

    def calculate_loss(self, batch):
        full_mask = np.full(self.env.action_space.n, True)

        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1,1)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1,1).to(
            device=self.network.device)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device)

        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t) # The selected action already respects the mask
        
        #################################################################
        # DDQN Update
        next_masks = np.array([self._get_mask(s) for s in next_states])
        qvals_next_pred = self.network.get_qvals(next_states)
        qvals_next_pred[np.logical_not(next_masks)] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1,1).to(
            device=self.network.device)
        target_qvals = self.network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        #################################################################
        qvals_next[dones_t] = 0 # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def _transform_observation(self, observation):
        observation = observation.astype(np.float64)
        observation = np.concatenate([observation[:self._grid_size],
        observation[self._grid_size:(2*self._grid_size)]/HP_NORM,
        [observation[2 * self._grid_size]/SUN_NORM], 
        observation[2 * self._grid_size+1:]])
        return observation

    def _get_mask(self, observation):
        empty_cells = np.nonzero((observation[:self._grid_size]==0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(self.env.action_space.n, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * len(self.env.plant_deck)

        available_plants = observation[-len(self.env.plant_deck):]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM
        
    def _save_training_data(self, nn_name):
        np.save(nn_name+"_rewards", self.training_rewards)
        np.save(nn_name+"_iterations", self.training_iterations)
        np.save(nn_name+"_real_rewards", self.real_rewards)
        np.save(nn_name+"_real_iterations", self.real_iterations)
        torch.save(self.training_loss, nn_name+"_loss")
        
    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self._transform_observation(self.env.reset())

class experienceReplayBuffer_DQN:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque 
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class PlayerQ_DQN():
    def __init__(self, env = None, render=True):
        if env==None:
            self.env = gym.make('gym_pvz:pvz-env-v2')
        else:
            self.env = env
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
        observation = np.concatenate([observation[:self._grid_size],
        observation[self._grid_size:(2*self._grid_size)]/HP_NORM,
        [observation[2 * self._grid_size]/SUN_NORM], 
        observation[2 * self._grid_size+1:]])
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

        while(True):
            if(self.render):
                self.env.render()
            # if np.random.random()<epsilon:
            #     # print("exploration")
            #     action = np.random.choice(self.get_actions(), 1)[0]
            # else:
            # action = agent.decide_action(observation, np.full(self.num_actions(), True), epsilon)
            action = agent.decide_action(observation, self.env.mask_available_actions(), epsilon)
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
    
