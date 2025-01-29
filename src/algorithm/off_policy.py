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
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

from src.models.model import BaseSwapModel
from src.algorithm.replay_buffer import OffPolicyReplayBuffer
from src.environment import initialize_logger

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
    reward_scaling: bool = False
    eval_every_episode: int = 10

logger = initialize_logger("dqn.log")

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

        self.reward_scaling = config.reward_scaling

        self.device = self.env.config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = OffPolicyReplayBuffer(self.buffer_size, self.priority)

        self.writer = SummaryWriter()
        self.steps = 0
        self.eval_every = config.eval_every_episode

    def select_action(self, state, deterministic=False):
        if random.random() < self.epsilon and not deterministic:
            return self.env.sample_action() # TODO change to action space
        with torch.no_grad():
            state = state.to(self.device)
            add_q_vals, remove_q_vals = self.policy_net(state)
            return torch.argmax(add_q_vals).item(), torch.argmax(remove_q_vals).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        batched_states = Batch.from_data_list(states).to(self.device)
        batched_next_states = Batch.from_data_list(next_states).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        if self.reward_scaling:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        add_q_values, remove_q_values = self.policy_net(batched_states)
        add_q_values = add_q_values.gather(1, actions[:, 0].unsqueeze(1))
        remove_q_values = remove_q_values.gather(1, actions[:, 1].unsqueeze(1))

        with torch.no_grad():
            next_add_q_values, next_remove_q_values = self.target_net(batched_next_states)
            next_add_q_values = next_add_q_values.max(1)[0].detach().unsqueeze(1)
            next_remove_q_values = next_remove_q_values.max(1)[0].detach().unsqueeze(1)

        target_add_q_values = rewards + (1 - dones) * self.gamma * next_add_q_values
        target_remove_q_values = rewards + (1 - dones) * self.gamma * next_remove_q_values

        loss_add = nn.functional.mse_loss(add_q_values, target_add_q_values)
        loss_remove = nn.functional.mse_loss(remove_q_values, target_remove_q_values)
        loss = loss_add + loss_remove

        logger.info(f"Loss: {loss.item()}")
        logger.info(f"Q values example: {add_q_values[0].tolist()} {remove_q_values[0].tolist()}")
        logger.info(f"Corresponding rewards {rewards.tolist()}")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.writer.add_scalar("Loss", loss.item())

    def update_target(self, episode):
        if episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_episodes=500):
        rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            steps = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.to(self.device)
                steps += 1
                self.steps += 1
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.train_step()
                rewards.append(reward)

            if episode % self.eval_every == 0:
                self.evaluate()

            self.update_target(episode)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            logger.info(f"Episode {episode}: Average Reward = {total_reward / steps: .4f}")
            logger.info(f"Running average reward: {sum(rewards[-100:]) / 100: .4f}")
            logger.info(f"Epsilon: {self.epsilon}")
            logger.info(f"Buffer Size: {len(self.replay_buffer)}")
            self.writer.add_scalar("Episode Reward", total_reward / steps)
            self.writer.add_scalar("Running Average Reward", sum(rewards[-100:]) / 100)


    def evaluate(self):
        state, _ = self.env.reset()
        state = state.to(self.device)
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = next_state.to(self.device)
            steps += 1
            state = next_state
            total_reward += reward

        logger.info(f"Average Reward = {total_reward / steps: .4f}")
        self.writer.add_scalar("Evaluation reward Reward", total_reward / steps)