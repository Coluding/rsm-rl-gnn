import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import numpy as np
import random
from collections import deque
from torch_geometric.data import Data
import gym
from dataclasses import dataclass

from src.models.model import BaseSwapModel
from src.algorithm.replay_buffer import ReplayBuffer

@dataclass()
class DQNAgentConfig:
    policy_net: BaseSwapModel
    target_net: BaseSwapModel
    env: gym.Env
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 10000
    target_update: int = 10
    priority: bool = False
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.1


class DQNAgent:
    def __init__(self, config: DQNAgentConfig):
        self.policy_net = config.policy_net
        self.target_net = config.target_net
        self.env = config.env
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.target_update = config.target_update
        self.priority = config.priority
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.priority)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = state.to(self.device)
            add_logits, remove_logits = self.policy_net(state)
            return torch.argmax(add_logits).item(), torch.argmax(remove_logits).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        add_q_values, remove_q_values = self.policy_net(states)
        add_q_values = add_q_values.gather(1, actions[:, 0].unsqueeze(1))
        remove_q_values = remove_q_values.gather(1, actions[:, 1].unsqueeze(1))

        with torch.no_grad():
            next_add_q_values, next_remove_q_values = self.target_net(next_states)
            next_add_q_values = next_add_q_values.max(1)[0].detach().unsqueeze(1)
            next_remove_q_values = next_remove_q_values.max(1)[0].detach().unsqueeze(1)

        target_add_q_values = rewards + (1 - dones) * self.gamma * next_add_q_values
        target_remove_q_values = rewards + (1 - dones) * self.gamma * next_remove_q_values

        loss_add = nn.functional.mse_loss(add_q_values, target_add_q_values)
        loss_remove = nn.functional.mse_loss(remove_q_values, target_remove_q_values)
        loss = loss_add + loss_remove

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self, episode):
        if episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_episodes=500):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.to(self.device)

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.train_step()

            self.update_target(episode)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Episode {episode}: Total Reward = {total_reward}")