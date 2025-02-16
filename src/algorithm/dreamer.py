import copy
import os
import random

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal

from src.algorithm import OffPolicyReplayBuffer
from src.models.rssm import RSSM, GraphEncoder, LatentDecoder, RewardModel, DynamicsModel
from src.algorithm.replay_buffer import *
from src.environment.utils import CrossProductActionSpace
from src.environment import initialize_logger


@dataclass()
class DreamerConfig:
    rssm: RSSM
    env: gym.Env
    trajectory_length: int
    lr: float = 3e-4
    lr_value_fn: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 32
    buffer_size: int = 10000
    predictor_update_freq: int = 1
    predictor_gamma: float = 0.8
    update_epochs: int = 10
    reward_scaling: bool = False
    train_every: int = 50  # Train every n steps
    temporal_size: int = 4
    cross_product_action_space: CrossProductActionSpace = None
    use_timestep_context: bool = False
    use_gae: bool = True



torch.autograd.set_detect_anomaly(True)

logger = initialize_logger("rssm.log")



class Dreamer:
    def __init__(self, config: DreamerConfig):
        self.rssm = config.rssm
        self.predictor = copy.deepcopy(self.rssm.encoder)
        self.env = config.env
        self.lr = config.lr
        self.trajectory_length = config.trajectory_length
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.update_epochs = config.update_epochs
        self.reward_scaling = config.reward_scaling
        self.train_every = config.train_every
        self.cross_product_action_space = config.cross_product_action_space
        self.temporal_size = config.temporal_size
        self.use_timestep_context = config.use_timestep_context

        self.predictor_update_freq = config.predictor_update_freq
        self.predictor_gamma = config.predictor_gamma

        self.device = self.env.config.device
        self.rssm.to(self.device)
        self.predictor.to(self.device)

        self.optimizer = optim.AdamW(self.rssm.parameters(), lr=self.lr)
        self.replay_buffer = TrajectoryReplayBuffer(self.buffer_size)
        self.step = 0

        self.writer = SummaryWriter(comment="dreamer")

    def load_model(self, directory: str):
        self.rssm.load(directory + "rssm.pth")

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.rssm.save(directory + "rssm.pth")

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

        #self.replay_buffer.normalize_rewards() if self.reward_scaling else None
        iterator = tqdm.tqdm(range(self.update_epochs), desc="Running dreamer epoch training...", unit="epoch")
        for _ in iterator:
            trajectories = self.replay_buffer.sample(self.batch_size)
            states = Batch.from_data_list([Batch.from_data_list(trajectory["states"]).to(self.device) for trajectory in trajectories])
            actions = torch.cat([torch.tensor(trajectory["actions"]) for trajectory in trajectories]).view(-1, self.trajectory_length, 1).to(self.device)
            rewards = torch.cat([torch.tensor(trajectory["rewards"]) for trajectory in trajectories]).view(-1, self.trajectory_length, 1).to(self.device)
            dones = torch.cat([torch.tensor(trajectory["dones"]).float() for trajectory in trajectories]).view(-1, self.trajectory_length, 1).to(self.device)

            encoded_states = self.rssm.encoder(states).view(actions.shape[0], actions.shape[1], -1)

            with torch.no_grad():
                # the target network is an exponential moving average of the encoder
                # This is similar to BYOL and avoids collapse of the representation
                target_encoded_states = self.predictor(states).view(actions.shape[0], actions.shape[1], -1)

            actions_ohe = F.one_hot(actions, num_classes=len(self.cross_product_action_space.action_space)).float().squeeze()
            hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = (
                self.rssm.generate_rollout(actions=actions_ohe, obs=encoded_states, dones=dones))

            # We remove the first elements, as they were zero initialized to enable proper RNN processing
            # We got from T+1 to T to math the target states length
            relevant_hiddens = hiddens[:, 1:, :]
            relevant_posteriors = posterior_states[:, 1:, :]

            reward_pred = self.rssm.reward_model(relevant_hiddens, relevant_posteriors)
            encoded_states_pred = self.rssm.decoder(relevant_hiddens, relevant_posteriors)

            kl_loss = self._compute_kl_loss(
                prior_means[:, 1:, :], prior_logvars[:, 1:, :],
                posterior_means[:, 1:, :], posterior_logvars[:, 1:, :]
            )
            # ELBO loss: MSE(o_t, o_t_pred)  --> We reconstruct the observation in the latent space, hence o_t = enc(g_t) where g_t is the graph observation
            elbo_loss = self._compute_elbo_loss(encoded_states_pred, target_encoded_states)
            reward_loss = self._compute_reward_loss(reward_pred, rewards)

            loss = kl_loss + elbo_loss + reward_loss
            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.rssm.parameters(), 10.0)

            self.optimizer.step()

            self.writer.add_scalar("Loss/kl_loss", kl_loss, self.training_step)
            self.writer.add_scalar("Loss/elbo_loss", elbo_loss, self.training_step)
            self.writer.add_scalar("Loss/reward_loss", reward_loss, self.training_step)

            if loss < self.best_loss:
                self.best_loss = loss
                self.save_model("best/")

            iterator.set_postfix({"kl_loss": kl_loss.item(), "elbo_loss": elbo_loss.item(), "reward_loss": reward_loss.item()})
            self.training_step += 1

            if self.training_step % self.predictor_update_freq == 0:
                self.update_predictor()


    def _compute_kl_loss(self, prior_means, prior_logvars, posterior_means, posterior_logvars):
        posterior_std = torch.exp(0.5 * posterior_logvars)
        prior_std = torch.exp(0.5 * prior_logvars)

        prior_dist = torch.distributions.Normal(prior_means, prior_std)
        posterior_dist = torch.distributions.Normal(posterior_means, posterior_std)

        kl_div = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist).mean()

        return kl_div

    def _compute_elbo_loss(self, reconstruced_obs, target_obs):
        return F.mse_loss(reconstruced_obs, target_obs) # assuming gaussian observation model

    def _compute_reward_loss(self, reward_pred, target_rewards):
        return F.mse_loss(reward_pred, target_rewards.float())

    def update_predictor(self):
        #TODO Implement this: IT is the exponential moving average of the target network which is the encoder

        encoder_params = dict(self.rssm.encoder.named_parameters())
        predictor_params = dict(self.predictor.named_parameters())

        for name, param in encoder_params.items():
            predictor_params[name].data.copy_(self.predictor_gamma * predictor_params[name].data +
                                              (1 - self.predictor_gamma) * param.data)


    def select_action(self, state: torch.Tensor) -> int:
        if random.random() < 0.5:
            # Also sample invalid actions that are connected to high reward (bad, as we want to minimize reward)
            return random.choice(self.cross_product_action_space.action_space)
        else:
            # Sample from the environment where we only can sample valid actions
            return self.cross_product_action_space[self.env.sample_action()]

    def train(self, num_episodes=500):
        iterator = tqdm.tqdm(range(num_episodes), desc="Training Dreamer...", unit="episode")
        all_rewards = []
        self.training_step = 0
        self.best_loss = float("inf")

        action_history = []  # Store actions per episode

        for episode in iterator:
            state, _ = self.env.reset()
            state = state.to(self.device)
            done = False
            steps = 0
            rewards = []
            episode_actions = []
            raw_actions = []

            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": []
            }

            starting_locs = self.env.active_locations

            while not done:
                action = self.select_action(state)

                if not isinstance(action, tuple):
                    env_action = self.cross_product_action_space[action]

                else:
                    env_action = action

                next_state, reward, done, _, _ = self.env.step(env_action)

                trajectory["states"].append(state.to(self.device))
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)

                steps += 1
                state = next_state
                rewards.append(reward)
                all_rewards.append(reward)
                self.step += 1

                if self.step % self.trajectory_length == 0:
                    self.replay_buffer.push(trajectory)
                    trajectory = {
                        "states": [],
                        "actions": [],
                        "rewards": [],
                        "dones": []
                    }

                episode_actions.append(action)
                raw_actions.append(env_action)

                if self.step % self.train_every == 0:
                    self.train_step()

    def evaluate(self):
        state, _ = self.env.reset()
        state = state.to(self.device)
        done = False
        steps = 0
        rewards = []
        episode_actions = []
        raw_actions = []

        while not done:
            action = self.select_action(state)

            if not isinstance(action, tuple):
                env_action = self.cross_product_action_space[action]

            else:
                env_action = action

            next_state, reward, done, _, _ = self.env.step(env_action)

            encode_state = self.rssm.encoder(state)
            action_ohe = F.one_hot(torch.tensor(action), num_classes=len(self.cross_product_action_space.action_space)).float().to(self.device)

            hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = (
                self.rssm.generate_rollout(actions=action_ohe, obs=encode_state, dones=torch.tensor(done).float().to(self.device))
            )


            steps += 1
            state = next_state
            rewards.append(reward)
            self.step += 1



            if self.step % self.trajectory_length == 0:
                self.replay_buffer.push(trajectory)
                trajectory = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                    "dones": []
                }

            episode_actions.append(action)

        def dream(self, state: torch.Tensor, initial_action: torch.Tensor, horizon: int):
            """
            Generates an imagined rollout for `horizon` steps.

            Args:
                state (torch.Tensor): The starting state (B, state_dim).
                initial_action (torch.Tensor): The first action (B, action_dim).
                horizon (int): Number of steps to simulate.

            Returns:
                dreamed_trajectory (dict): A dictionary containing:
                    - "states": Tensor of imagined states (B, horizon, state_dim)
                    - "actions": Tensor of actions taken (B, horizon, action_dim)
                    - "rewards": Tensor of predicted rewards (B, horizon, 1)
                    - "hiddens": Tensor of hidden states (B, horizon, hidden_dim)
            """

            batch_size = state.shape[0]
            dreamed_trajectory = {
                "encoded_states": [],
                "actions": [],
                "rewards": [],
                "hiddens": [],
            }

            # Initialize hidden state from the prior
            hidden = torch.zeros(batch_size, self.rssm.hidden_dim, device=state.device)

            # Assign initial action
            action = initial_action

            for t in range(horizon):
                # One-hot encode action if needed
                action_ohe = F.one_hot(action, num_classes=len(self.cross_product_action_space.action_space)).float()

                # Use prior dynamics (since we don't have real observations)
                hidden_action = torch.cat([hidden, action_ohe], dim=-1)
                hidden_action = self.rssm.dynamics.act_fn(self.rssm.dynamics.project_hidden_action(hidden_action))

                prior_params = self.rssm.dynamics.prior(hidden_action)
                prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

                prior_dist = torch.distributions.Normal(prior_mean, torch.exp(F.softplus(prior_logvar)))
                prior_state = prior_dist.rsample()  # Sample from prior

                # Predict reward using the RSSM reward model
                reward = self.rssm.reward_model(hidden, prior_state)

                # Store imagined rollout data
                dreamed_trajectory["encoded_states"].append(prior_state.unsqueeze(1))  # (B, 1, state_dim)
                dreamed_trajectory["actions"].append(action.unsqueeze(1))  # (B, 1, action_dim)
                dreamed_trajectory["rewards"].append(reward.unsqueeze(1))  # (B, 1, 1)
                dreamed_trajectory["hiddens"].append(hidden.unsqueeze(1))  # (B, 1, hidden_dim)

                # Get next action dynamically (this should come from your policy)
                action = self.select_action()

                # Update hidden state using the RNN
                hidden = self.rssm.dynamics.rnn[0](self.rssm.dynamics.act_fn(
                    self.rssm.dynamics.project_state_action(torch.cat([prior_state, action_ohe], dim=-1))), hidden)

            # Convert lists to tensors
            dreamed_trajectory = {key: torch.cat(value, dim=1) for key, value in
                                  dreamed_trajectory.items()}  # (B, horizon, dim)

            return dreamed_trajectory


if __name__ == "__main__":
    from src.environment.fluidity_environment import FluidityEnvironment, FluidityEnvironmentConfig, TorchGraphObservationWrapper

    config = FluidityEnvironmentConfig(
            jar_path="../../ressources/jars/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
            jvm_options=['-Djava.security.properties=../../ressources/run_configs/40_steps/simurun/server0/xmr/config/java.security'],
            configuration_directory_simulator="../../ressources/run_configs/40_steps/",
            node_identifier="server0",
            device="cpu",
            feature_dim_node=1
        )

    env = FluidityEnvironment(config)
    env = TorchGraphObservationWrapper(env, one_hot=False)

    action_mapper = CrossProductActionSpace.from_json("../data/action_space.json")

    D = 128
    OBS_D = 64
    encoder = GraphEncoder(8, 32, OBS_D)
    dynamics = DynamicsModel(D, 57, D, OBS_D * 2, 2)
    latent_decoder = LatentDecoder(D, D, OBS_D * 2)
    reward_model = RewardModel(D, D, )

    rssm = RSSM(encoder, latent_decoder, reward_model, dynamics, D, D, 57, 32)

    dreamer = Dreamer(DreamerConfig(rssm, env, 6, batch_size=4, update_epochs=500,
                                    cross_product_action_space=action_mapper,
                                    train_every=41))

    dreamer.train(1000)