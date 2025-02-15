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
from typing import Literal

from src.algorithm import OnPolicyReplayBuffer
from src.models.model import BaseSwapModel, BaseValueModel, SwapActionMapper, CrossProductSwapActionMapper
from src.algorithm.replay_buffer import *
from src.environment import initialize_logger


@dataclass()
class OnPolicyAgentConfig:
    algorithm: Literal["PPO", "REINFORCE"]
    policy_net: BaseSwapModel
    env: gym.Env
    value_net: BaseValueModel = None
    lr: float = 3e-4
    lr_value_fn: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 32
    buffer_size: int = 10000
    clip_epsilon: float = 0.3
    entropy_coeff: float = 0.01
    value_coeff: float = 2
    update_epochs: int = 10
    reward_scaling: bool = False
    train_every: int = 50  # Train every n steps
    temporal_size: int = 4
    cross_product_action_space: CrossProductSwapActionMapper = None
    use_timestep_context: bool = False
    use_gae: bool = True



torch.autograd.set_detect_anomaly(True)

logger = initialize_logger("ppo.log")



class OnPolicyAgent:
    def __init__(self, config: OnPolicyAgentConfig):
        self.algorithm = config.algorithm
        self.policy_net = config.policy_net
        self.value_net = config.value_net
        self.env = config.env
        self.lr = config.lr
        self.value_lr = config.lr_value_fn
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
        self.use_gae = config.use_gae

        assert not (self.cross_product_action_space is None and isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper)), "Cross Product Action Space is required for CrossProductSwapActionMapper"

        self.device = self.env.config.device
        self.policy_net.to(self.device)
        self.value_net.to(self.device) if self.value_net is not None else None

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr) if self.value_net is not None else None
        self.replay_buffer = OnPolicyReplayBuffer(self.buffer_size)

        self.step = 0

        self.writer = SummaryWriter(comment="ppo" if self.algorithm == "PPO" else "reinforce")

    def load_model(self, directory: str):
        self.policy_net.load_state_dict(torch.load(directory + "policy.pth"))
        self.value_net.load_state_dict(torch.load(directory + "value.pth"))

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

    def compute_gae(self, rewards, values, dones):
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

    def compute_reward_to_go(rewards, gamma=0.99):
        """
        Computes reward-to-go for each timestep t:
        G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^(T-t) r_T
        """
        G = torch.zeros_like(rewards, dtype=torch.float)
        future_return = 0
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + gamma * future_return
            G[t] = future_return
        return G

    def compute_reward_to_go_with_baseline(self, rewards):
        """
        Computes reward-to-go for each timestep t and subtracts baseline (mean return).
        G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^(T-t) r_T
        Baseline: mean of all G_t values in the episode.
        """
        G = torch.zeros_like(rewards, dtype=torch.float)
        future_return = 0
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            G[t] = future_return

        baseline = G.mean()

        return G - baseline

    def compute_advantage_normal(self, rewards, values, dones):
        """
        Computes the normal one-step advantage: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float)

        for t in range(len(rewards)):
            next_value = values[t + 1] if (t + 1 < len(rewards) and not dones[t]) else 0
            advantages[t] = rewards[t] + self.gamma * next_value - values[t]

        return advantages

    def compute_clipped_advantage(self, rewards, values, dones):
        """
        Computes the clipped advamtage to ensure smooth training
        """

    def train_step_reinforce(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        iterator = tqdm.tqdm(range(self.update_epochs), desc="Running epoch training...", unit="epoch")
        for _ in iterator:
            epoch_loss = 0
            for batch in self.replay_buffer(self.batch_size, shuffle=False):
                states, actions, _, rewards, _, dones, timesteps = zip(*batch)

                batched_states = Batch.from_data_list(states).to(self.device)
                actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                timesteps = torch.tensor(timesteps, dtype=torch.long, device=self.device) if self.use_timestep_context else None


                # Compute baseline-subtracted reward-to-go
                reward_to_go = self.compute_reward_to_go_with_baseline(rewards)

                # Compute policy loss
                logits = self.policy_net(batched_states, timesteps)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                policy_loss = -torch.mean(log_probs * reward_to_go)  # Policy gradient update

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                epoch_loss += policy_loss.item()
                self.writer.add_scalar("Policy Loss", policy_loss.item(), self.training_step)
                iterator.set_postfix({"Policy Loss": policy_loss.item()})

                self.training_step += 1

                if policy_loss < self.best_loss:
                    self.best_loss = policy_loss
                    torch.save(self.policy_net.state_dict(), "best_policy_re.pth")

            self.writer.add_scalar("Epoch Loss", epoch_loss)

        self.replay_buffer.clear()

    def save_model(self, directory: str):
        torch.save(self.policy_net.state_dict(), directory + "policy.pth")
        torch.save(self.value_net.state_dict(), directory + "value.pth")

    def load_model(self, directory: str):
        self.policy_net.load_state_dict(torch.load(directory + "policy.pth"))
        self.value_net.load_state_dict(torch.load(directory + "value.pth"))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        #self.replay_buffer.normalize_rewards() if self.reward_scaling else None
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
                policy_grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                value_grad_norm = nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

                self.writer.add_scalar("Policy Grad Norm", policy_grad_norm, self.training_step)
                self.writer.add_scalar("Value Grad Norm", value_grad_norm, self.training_step)

                self.optimizer.step()
                self.value_optimizer.step()

                self.training_step += 1

                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    torch.save(self.policy_net.state_dict(), "best_policy.pth")
                    torch.save(self.value_net.state_dict(), "best_value.pth")

                iterator.set_postfix({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item()})

        self.replay_buffer.clear()

    def _value_loss(self, batched_states: torch_geometric.data.Data, rewards: torch.Tensor,
                    dones: torch.Tensor, batched_next_states: torch_geometric.data.Data,
                    timesteps: torch.Tensor = None
                    ):
        values = self.value_net(batched_states, timesteps, self.temporal_size).squeeze()
        next_values = self.value_net(batched_next_states, timesteps ,self.temporal_size).squeeze()
        advantages = self.compute_gae(rewards, values, dones) if self.use_gae else self.compute_advantage_normal(rewards, values, dones)
        value_loss = nn.functional.mse_loss(values, (rewards + self.gamma * next_values * (1 - dones)))
        value_loss = value_loss * self.value_coeff

        # log example values
        if self.training_step % 200 == 0:
            logger.info(f"Example Values: {str(values.tolist())}")
            logger.info("Corresponding rewards: ", str(rewards.tolist()))
            logger.info("Next Values: ", str(next_values.tolist()))

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

            self.writer.add_scalar("Entropy", entropy, self.training_step)

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

            self.writer.add_scalar("Entropy", entropy, self.training_step)
            self.writer.add_scalar("Ratio", ratio.mean(), self.training_step)

        else:
            raise NotImplementedError(f"Action mapper of type {type(self.policy_net.action_mapper)} not supported")

        if self.step % 1000 == 0:
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

        action_history = []  # Store actions per episode

        for episode in iterator:
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            steps = 0
            episode_buffer = []
            rewards = []
            episode_actions = []
            raw_actions = []

            starting_locs = self.env.active_locations

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
                self.replay_buffer.push(state, action, log_prob, -reward, next_state, done, steps)
                state = next_state
                total_reward += reward
                rewards.append(reward)
                all_rewards.append(reward)
                self.step += 1

                episode_actions.append(action)
                raw_actions.append(env_action)

                if self.step % self.train_every == 0:
                    match self.algorithm:
                        case "PPO":
                            self.train_step()
                        case "REINFORCE":
                            self.train_step_reinforce()

            logger.info("Episode [{}]Average Reward: {:.4f}".format(episode, np.mean(rewards)))

            action_history.extend(episode_actions)

            if episode % 50 == 0:

                baseline_rewards = self.run_static_baseline()

            # Save a plot of the episode reward to tensorboard
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(rewards, label="Policy Reward")
                ax.plot(baseline_rewards, label="Static Baseline Reward")
                ax.set_xlabel("Step")
                ax.set_ylabel("Reward")
                ax.set_title("Episode Reward")
                ax.legend()
                self.writer.add_figure(f"Episode {episode} Reward", fig, episode)

                if not isinstance(action, tuple) :
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(episode_actions, bins=np.arange(min(episode_actions) - 0.5, max(episode_actions) + 1.5, 1),
                            edgecolor="black")
                    ax.set_xlabel("Action")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Action Selection Histogram (Last 10 Episodes)")
                    self.writer.add_figure(f"Episode {episode} Action Histogram", fig, episode)
                    action_history.clear()

                self.writer.add_text(f"Starting Locations episode {episode}", str(starting_locs), episode)

            self.writer.add_text("Episode Actions", str(raw_actions), episode)
            self.writer.add_scalar("Episode Reward", total_reward, episode)
            self.writer.add_scalar("Average Reward", np.mean(rewards), episode)
            self.writer.add_scalar("Rolling Average Reward", np.mean(all_rewards[-20:]), episode)

    def run_static_baseline(self):

        state, _ = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        all_rewards = []

        while not done:
            next_state, reward, done, _, _ = self.env.step((-1,-1))
            steps += 1
            total_reward += reward
            all_rewards.append(reward)
            self.step += 1

        return all_rewards