import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import gym
from dataclasses import dataclass
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from src.models.model import BaseSwapModel, BaseValueModel, SwapActionMapper, CrossProductSwapActionMapper
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
    cross_product_action_space: CrossProductSwapActionMapper = None
    use_timestep_context: bool = False
    update_epochs: int = 10
    train_every_steps: int = 100
    temporal_size: int = 4


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
        self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = OffPolicyReplayBuffer(self.buffer_size, self.priority)

        self.writer = SummaryWriter(comment="DQN")
        self.steps = 0
        self.eval_every = config.eval_every_episode

        self.cross_product_action_space = config.cross_product_action_space
        self.use_timestep_context = config.use_timestep_context
        self.update_epochs = config.update_epochs
        self.train_every_steps = config.train_every_steps
        self.temporal_size = config.temporal_size

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.to(self.device)
        logger.info(f"Model loaded from {path}")

    def select_action(self, state, timestep = None, deterministic=False):
        if random.random() < self.epsilon and not deterministic:
            return self.env.sample_action(self.cross_product_action_space is not None) # TODO change to action space

        with torch.no_grad():
            if isinstance(self.policy_net.action_mapper, SwapActionMapper):
                state = state.to(self.device)
                add_q_vals, remove_q_vals = self.policy_net(state)
                return torch.argmax(add_q_vals).item(), torch.argmax(remove_q_vals).item()
            elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
                state = state.to(self.device)
                timestep = torch.tensor(timestep, dtype=torch.long, device=self.device).unsqueeze(0)
                q_vals = self.policy_net(state, timestep)
                mask = self._process_mask(state)
                q_vals = q_vals + mask

                return torch.argmax(q_vals).item()

    def _process_mask(self, batched_states, B=1):
        location_indices = torch.where(batched_states.label != 0)[0]
        loc_ids = batched_states.name[location_indices].view(B, self.temporal_size, len(self.env.loc_mapping))[:, -1, :]
        mask = batched_states.add_mask[location_indices].view(B, self.temporal_size, len(self.env.loc_mapping))[:, -1,:]

        zero_mask_index = mask == 0
        removable_locations = loc_ids[zero_mask_index].view(B, -1).tolist()
        addable_locations = loc_ids[~zero_mask_index].view(-1, len(self.env.active_locations)).tolist()
        final_add_mask = [self.cross_product_action_space.build_add_action_mask(x) for x in addable_locations]
        final_remove_mask = [self.cross_product_action_space.build_remove_action_mask(x) for x in removable_locations]

        final_mask = torch.tensor(final_add_mask) + torch.tensor(final_remove_mask)

        return final_mask.to(self.device)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.full_training_runs += 1
        epoch_losses = []
        iterator = tqdm(range(self.update_epochs), desc="Training DQN...")
        for _ in iterator:
            batch = self.replay_buffer.sample(self.batch_size)

            loss = self._compute_q_vals_and_loss(batch) if self.cross_product_action_space is None else self._compute_q_vals_and_loss_cross_product(batch)

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), "best_q_model.pt")

            self.optimizer.zero_grad()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            loss.backward()
            self.optimizer.step()
            self.training_step += 1
            self.writer.add_scalar("Loss", loss.item(), self.training_step)
            epoch_losses.append(loss.item())

            iterator.set_postfix({"Loss": loss.item()})

        self.writer.add_scalar("Epoch Loss", np.mean(epoch_losses), self.full_training_runs)

    def _compute_q_vals_and_loss(self, batch):
        states, actions, rewards, next_states, dones, timesteps = zip(*batch)

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

        if self.training_step % 100 == 0:
            logger.info(f"Loss: {loss.item()}")
            logger.info(f"Q values example: {add_q_values[0].tolist()} {remove_q_values[0].tolist()}")
            logger.info(f"Corresponding rewards {rewards.tolist()}")

        return loss

    def _compute_q_vals_and_loss_cross_product(self, batch):
        states, actions, rewards, next_states, dones, timesteps = zip(*batch)

        batched_states = Batch.from_data_list(states).to(self.device)
        batched_next_states = Batch.from_data_list(next_states).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        timesteps = torch.tensor(timesteps, dtype=torch.long, device=self.device)

        if self.reward_scaling:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.policy_net(batched_states, timesteps)

        with torch.no_grad():
            next_q_vals = self.target_net(batched_next_states, timesteps)
            next_q_vals = next_q_vals.max(1)[0].detach().unsqueeze(1)

        target_q_vals = rewards + (1 - dones) * self.gamma * next_q_vals
        loss = nn.functional.mse_loss(q_vals, target_q_vals)

        if self.training_step % 100 == 0:
            logger.info(f"Loss: {loss.item()}")
            logger.info(f"Q values example: {q_vals.tolist()}")
            logger.info(f"Corresponding rewards {rewards.tolist()}")

        return loss

    def update_target(self, episode):
        if episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.to(self.device)

    def train(self, num_episodes=500):
        self.training_step = 0
        self.best_loss = float("inf")
        self.full_training_runs = 0
        self.step = 0
        self.evaluation_step = 0
        action_history = []
        all_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            counter = 0
            rewards = []
            episode_actions = []

            while not done:
                action = self.select_action(state, counter)

                if not isinstance(action, tuple):
                    env_action = self.cross_product_action_space[action]

                else:
                    env_action = action

                if env_action[1] == (-1, -1):
                    pass
                next_state, reward, done, _, _ = self.env.step(env_action)
                next_state = next_state.to(self.device)
                self.step += 1
                self.replay_buffer.push(state, action, reward, next_state, done, counter)
                state = next_state
                total_reward += reward

                rewards.append(reward)
                all_rewards.append(reward)
                episode_actions.append(action)

                counter += 1

                if self.step % self.train_every_steps == 0:
                    self.train_step()

                if self.step % self.target_update == 0:
                    self.update_target(episode)


            action_history.extend(episode_actions)

            if episode % self.eval_every == 0:
                self.evaluate()

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            logger.info(f"Episode {episode}: Average Reward = {total_reward / counter: .4f}")
            logger.info(f"Running average reward: {sum(rewards[-100:]) / 100: .4f}")
            logger.info(f"Epsilon: {self.epsilon}")
            logger.info(f"Buffer Size: {len(self.replay_buffer)}")
            self.writer.add_scalar("Episode Average Reward", total_reward / counter, episode)
            self.writer.add_scalar("Episode Reward Standard Deviation", np.std(rewards), episode)
            self.writer.add_scalar("Running Average Reward", sum(rewards[-100:]) / 100, episode)

            if episode % 10 == 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(rewards)
                ax.set_xlabel("Step")
                ax.set_ylabel("Reward")
                ax.set_title("Episode Reward")
                self.writer.add_figure(f"Episode {episode} Reward", fig, episode)
                all_rewards.clear()

                if not isinstance(episode_actions[0], tuple):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(episode_actions, bins=np.arange(min(action_history) - 0.5, max(action_history) + 1.5, 1),
                            edgecolor="black")
                    ax.set_xlabel("Action")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Action Selection Histogram (Last 10 Episodes)")
                    self.writer.add_figure(f"Episode {episode} Action Histogram", fig, episode)
                    action_history.clear()


    def evaluate(self):
        state, _ = self.env.reset()
        self.evaluation_step += 1
        state = state.to(self.device)
        total_reward = 0
        done = False
        steps = 0
        action_history = []
        reward_history = []
        while not done:
            action = self.select_action(state, timestep=steps, deterministic=True)

            if not isinstance(action, tuple):
                env_action = self.cross_product_action_space[action]

            action_history.append(action)
            next_state, reward, done, _, _ = self.env.step(env_action)
            next_state = next_state.to(self.device)
            steps += 1
            state = next_state
            total_reward += reward
            reward_history.append(reward)

        logger.info(f"Average Reward = {total_reward / steps: .4f}")
        self.writer.add_scalar("Evaluation reward Reward", total_reward / steps)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(reward_history)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title(f"Reward development of Evaluation {self.evaluation_step}")
        self.writer.add_figure(f"Evaluation {self.evaluation_step} Reward", fig, self.evaluation_step)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(action_history, bins=np.arange(min(action_history) - 0.5, max(action_history) + 1.5, 1),
                edgecolor="black")
        ax.set_xlabel("Action")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Action Selection Histogram of Evaluation {self.evaluation_step}")
        self.writer.add_figure(f"Evaluation {self.evaluation_step} Action Histogram", fig, self.evaluation_step)
        action_history.clear()