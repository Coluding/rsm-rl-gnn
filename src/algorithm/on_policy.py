import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import numpy as np
import random
from collections import deque

import tqdm
from torch_geometric.data import Batch
import gym
from dataclasses import dataclass

from src.algorithm import OnPolicyReplayBuffer
from src.models.model import BaseSwapModel, BaseValueModel
from src.algorithm.replay_buffer import *
from src.environment import initialize_logger


@dataclass()
class PPOAgentConfig:
    policy_net: BaseSwapModel
    value_net: BaseValueModel
    env: gym.Env
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 32
    buffer_size: int = 10000
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    update_epochs: int = 10
    reward_scaling: bool = False
    train_every: int = 50  # Train every n steps
    temporal_size: int = 4


torch.autograd.set_detect_anomaly(True)

logger = initialize_logger("ppo.log")


class PPOAgent:
    def __init__(self, config: PPOAgentConfig):
        self.policy_net = config.policy_net
        self.value_net = config.value_net
        self.env = config.env
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.clip_epsilon = config.clip_epsilon
        self.entropy_coeff = config.entropy_coeff
        self.value_coeff = config.value_coeff
        self.update_epochs = config.update_epochs
        self.reward_scaling = config.reward_scaling
        self.train_every = config.train_every

        self.device = self.env.config.device
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.replay_buffer = OnPolicyReplayBuffer(self.buffer_size)

        self.step = 0
        self.temporal_size = config.temporal_size

    def select_action(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            add_logits, remove_logits = self.policy_net(state, self.temporal_size)
            add_dist = torch.distributions.Categorical(logits=add_logits)
            remove_dist = torch.distributions.Categorical(logits=remove_logits)
            add_action = add_dist.sample()
            remove_action = remove_dist.sample()
            add_log_prob = add_dist.log_prob(add_action)
            remove_log_prob = remove_dist.log_prob(remove_action)
        return (add_action.item(), remove_action.item()), (add_log_prob.item(), remove_log_prob.item())

    def compute_advantage(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards, dtype=torch.float)

        last_advantage = 0
        for t in reversed(range(len(rewards))):
            terminal = 1 - dones[t].int()
            next_value = values[t + 1] if t + 1 < len(rewards) else 0
            delta = terminal * (rewards[t] + self.gamma * next_value) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * terminal * last_advantage
            last_advantage = advantages[t]

        return advantages

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        iterator = tqdm.tqdm(range(self.update_epochs), desc="Running epoch training...", unit="epoch")
        for _ in iterator:
            # maybe sample with a seed inside the loop
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, old_log_probs, rewards, next_states, dones = zip(*batch)

            batched_states = Batch.from_data_list(states).to(self.device)
            batched_next_states = Batch.from_data_list(next_states).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            if self.reward_scaling:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            values = self.value_net(batched_states, self.temporal_size).squeeze()
            next_values = self.value_net(batched_next_states, self.temporal_size).squeeze()
            advantages = self.compute_advantage(rewards, values, dones)

            # Should not they be in order?
            add_logits, remove_logits = self.policy_net(batched_states, self.temporal_size)
            add_dist = torch.distributions.Categorical(logits=add_logits)
            remove_dist = torch.distributions.Categorical(logits=remove_logits)
            add_log_probs = add_dist.log_prob(actions[:, 0])
            remove_log_probs = remove_dist.log_prob(actions[:, 1])
            entropy = (add_dist.entropy() + remove_dist.entropy()).mean()

            ratio_add = torch.exp(add_log_probs - old_log_probs[:, 0])
            ratio_remove = torch.exp(remove_log_probs - old_log_probs[:, 1])
            clipped_ratio_add = torch.clamp(ratio_add, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            clipped_ratio_remove = torch.clamp(ratio_remove, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio_add * advantages, clipped_ratio_add * advantages).mean()
            policy_loss = policy_loss - torch.min(ratio_remove * advantages, clipped_ratio_remove * advantages).mean()
            policy_loss = policy_loss - self.entropy_coeff * entropy

            value_loss = nn.functional.mse_loss(values, (rewards + self.gamma * next_values * (1 - dones)))
            value_loss = value_loss * self.value_coeff

            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            policy_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

            self.optimizer.step()
            self.value_optimizer.step()

            iterator.set_postfix({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item()})
            if self.step % 100 == 0:
                logger.info(f"Policy Loss: {policy_loss.item()}")
                logger.info(f"Value Loss: {value_loss.item()}")
                logger.info(f"Example Probabilities: {add_dist.probs.tolist()[0]}, {remove_dist.probs.tolist()[0]}")

    def train(self, num_episodes=500):
        iterator = tqdm.tqdm(range(num_episodes), desc="Training PPO...", unit="episode")
        all_rewards = []
        for episode in iterator:
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            steps = 0
            episode_buffer = []
            rewards = []

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.to(self.device)
                steps += 1
                episode_buffer.append((state, action, log_prob, reward, next_state, done))
                state = next_state
                total_reward += reward
                rewards.append(reward)
                all_rewards.append(reward)
                self.step += 1

                if self.step % self.train_every == 0:
                    self.train_step()

            for transition in episode_buffer:
                self.replay_buffer.push(*transition)

            self.train_step() # We need to train one more time after the episode ends
            logger.info(f"Episode {episode}: Total Reward = {total_reward:.4f}")
            logger.info(f"Episode {episode}: Steps = {steps}")
            logger.info("Average Reward: {:.4f}".format(np.mean(rewards)))
            logger.info("Rolling Average Reward: {:.4f}".format(np.mean(all_rewards[-100:])))
