import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.data
import torch_geometric.nn as gnn
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from dataclasses import dataclass

from src.algorithm import OnPolicyReplayBuffer
from src.models.model import BaseSwapModel, BaseValueModel, SwapActionMapper, CrossProductSwapActionMapper
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
    cross_product_action_space: CrossProductSwapActionMapper = None
    use_timestep_context: bool = False



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
        self.cross_product_action_space = config.cross_product_action_space
        self.temporal_size = config.temporal_size
        self.use_timestep_context = config.use_timestep_context

        assert not (self.cross_product_action_space is None and isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper)), "Cross Product Action Space is required for CrossProductSwapActionMapper"

        self.device = self.env.config.device
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.replay_buffer = OnPolicyReplayBuffer(self.buffer_size)

        self.step = 0

        self.writer = SummaryWriter(comment="ppo")

    def select_action(self, state, timestep=None):
        state = state.to(self.device)
        with torch.no_grad():
            if isinstance(self.policy_net.action_mapper, SwapActionMapper):
                add_logits, remove_logits = self.policy_net(state, self.temporal_size)
                add_dist = torch.distributions.Categorical(logits=add_logits)
                remove_dist = torch.distributions.Categorical(logits=remove_logits)
                add_action = add_dist.sample()
                remove_action = remove_dist.sample()
                add_log_prob = add_dist.log_prob(add_action)
                remove_log_prob = remove_dist.log_prob(remove_action)

                return (add_action.item(), remove_action.item()), (add_log_prob.item(), remove_log_prob.item())

            elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
                logits = self.policy_net(state, timestep, self.temporal_size)
                mask = self._process_mask(state)
                logits = logits + mask
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                return action.item(), log_prob.item()

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

    def compute_advantage(self, rewards, values, dones):
        """
        Computes the generalized advantage estimate
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float)

        last_advantage = 0
        for t in reversed(range(len(rewards))):
            terminal = 1 - dones[t].int()
            next_value = values[t + 1] if (t + 1 < len(rewards) and not dones[t]) else 0
            delta = terminal * (rewards[t] + self.gamma * next_value - values[t])
            advantages[t] = delta + self.gamma * self.gae_lambda * terminal * last_advantage
            last_advantage = advantages[t]

        return advantages

    def save_model(self, directory: str):
        torch.save(self.policy_net.state_dict(), directory + "policy.pth")
        torch.save(self.value_net.state_dict(), directory + "value.pth")

    def load_model(self, directory: str):
        self.policy_net.load_state_dict(torch.load(directory + "policy.pth"))
        self.value_net.load_state_dict(torch.load(directory + "value.pth"))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.replay_buffer.normalize_rewards()
        iterator = tqdm.tqdm(range(self.update_epochs), desc="Running epoch training...", unit="epoch")
        for _ in iterator:
            for batch in self.replay_buffer(self.batch_size, shuffle=False):
                states, actions, old_log_probs, rewards, next_states, dones, timesteps = zip(*batch)

                batched_states = Batch.from_data_list(states).to(self.device)
                batched_next_states = Batch.from_data_list(next_states).to(self.device)
                actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
                timesteps = torch.tensor(timesteps, dtype=torch.long, device=self.device) if self.use_timestep_context else None

                #if self.reward_scaling:
                #    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

                value_loss, advantages = self._value_loss(batched_states, rewards, dones, batched_next_states, timesteps)
                policy_loss = self._ppo_loss_logic(batched_states, actions, old_log_probs, advantages, timesteps)

                total_loss = policy_loss + value_loss

                self.writer.add_scalar("Policy Loss", policy_loss.item(), self.training_step)
                self.writer.add_scalar("Value Loss", value_loss.item(), self.training_step)

                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                total_loss.backward()

                # log grad norm
                self.writer.add_scalar("Policy Grad Norm", nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0), self.training_step)
                self.writer.add_scalar("Value Grad Norm", nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0), self.training_step)

                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

                self.optimizer.step()
                self.value_optimizer.step()

                self.training_step += 1

                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    torch.save(self.policy_net.state_dict(), "best_policy_40.pth")
                    torch.save(self.value_net.state_dict(), "best_value_40.pth")

                iterator.set_postfix({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item()})
                if self.step % 100 == 0:
                    logger.info(f"Policy Loss: {policy_loss.item()}")
                    logger.info(f"Value Loss: {value_loss.item()}")

    def _value_loss(self, batched_states: torch_geometric.data.Data, rewards: torch.Tensor,
                    dones: torch.Tensor, batched_next_states: torch_geometric.data.Data,
                    timesteps: torch.Tensor = None
                    ):
        values = self.value_net(batched_states, timesteps, self.temporal_size).squeeze()
        next_values = self.value_net(batched_next_states, timesteps ,self.temporal_size).squeeze()
        advantages = self.compute_advantage(rewards, values, dones)
        value_loss = nn.functional.mse_loss(values, (rewards + self.gamma * next_values * (1 - dones)))
        value_loss = value_loss * self.value_coeff

        return value_loss, advantages

    def _ppo_loss_logic(self, batched_states: torch_geometric.data.Data, actions: torch.Tensor,
                         old_log_probs: torch.Tensor, advantages: torch.Tensor, timesteps : torch.Tensor = None):

        if isinstance(self.policy_net.action_mapper, SwapActionMapper):
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

        elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
            logits = self.policy_net(batched_states, timesteps, self.temporal_size)
            mask = self._process_mask(batched_states, actions.shape[0])
            logits = logits + mask
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            policy_loss = policy_loss - self.entropy_coeff * entropy

        else:
            raise NotImplementedError(f"Action mapper of type {type(self.policy_net.action_mapper)} not supported")

        if self.step % 100 == 0:
            if isinstance(self.policy_net.action_mapper, SwapActionMapper):
                logger.info(f"Example Probabilities: {add_dist.probs.tolist()[0]}, {remove_dist.probs.tolist()[0]}")
            elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
                logger.info(f"Example Probabilities: {dist.probs.tolist()[0]}")

        return policy_loss

    def train(self, num_episodes=500):
        iterator = tqdm.tqdm(range(num_episodes), desc="Training PPO...", unit="episode")
        all_rewards = []
        self.training_step = 0
        self.best_loss = float("inf")
        for episode in iterator:
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            steps = 0
            episode_buffer = []
            rewards = []

            while not done:
                action, log_prob = self.select_action(state, torch.tensor(steps, dtype=torch.long, device=self.device).unsqueeze(0) if self.use_timestep_context else None)

                if not isinstance(action, tuple):
                    env_action = self.cross_product_action_space[action]

                else:
                    env_action = action

                next_state, reward, done, _, _ = self.env.step(env_action)
                next_state = next_state.to(self.device)
                steps += 1
                #episode_buffer.append((state, action, log_prob, reward, next_state, done))
                self.replay_buffer.push(state, action, log_prob, reward, next_state, done, steps)
                state = next_state
                total_reward += reward
                rewards.append(reward)
                all_rewards.append(reward)
                self.step += 1

                if self.step % self.train_every == 0:
                    self.train_step()
                    self.replay_buffer.clear()


            logger.info(f"Episode {episode}: Total Reward = {total_reward:.4f}")
            logger.info(f"Episode {episode}: Steps = {steps}")
            logger.info("Average Reward: {:.4f}".format(np.mean(rewards)))
            logger.info("Rolling Average Reward: {:.4f}".format(np.mean(all_rewards[-100:])))

            if episode % 10 == 0:
            # Save a plot of the episode reward to tensorboard
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(all_rewards)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.set_title("Episode Reward")
                self.writer.add_figure(f"Episode {episode} Reward", fig, episode)

            self.writer.add_scalar("Episode Reward", total_reward, episode)
            self.writer.add_scalar("Average Reward", np.mean(rewards), episode)
            self.writer.add_scalar("Rolling Average Reward", np.mean(all_rewards[-20:]), episode)
