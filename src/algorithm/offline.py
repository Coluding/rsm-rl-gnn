import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from src.models.model import RSMDecisionTransformer
from src.utils import initialize_logger
import os
from tqdm import tqdm


logger = initialize_logger("decision_transformer.log")


class DecisionTransformerGraphDataset(Dataset):
    def __init__(self, max_size=10000, max_len=6, scale=5000.0):
        """
        A dataset for storing and sampling training examples dynamically.

        :param max_size: Maximum number of trajectories to store.
        :param max_len: Maximum sequence length for each training sample.
        :param scale: Scaling factor for returns-to-go normalization.
        """
        self.max_size = max_size
        self.max_len = max_len
        self.scale = scale

        self.trajectories = []  # Stores full episode data
        self.state_mean = None
        self.state_std = None

    def add_trajectory(self, observations, actions, rewards, dones):
        """
        Adds a trajectory to the dataset.

        :param observations: List of `torch_geometric.data.Data` graphs.
        :param actions: List of (add, remove) actions as `torch.Tensor`.
        :param rewards: List of reward values.
        :param dones: List of done flags.
        """
        if len(self.trajectories) >= self.max_size:
            self.trajectories.pop(0)  # Remove oldest trajectory

        self.trajectories.append({
            "observations": observations,
            "actions": torch.stack(actions) if not isinstance(actions, torch.Tensor) else actions,
            "rewards": torch.tensor(rewards, dtype=torch.float32) if not isinstance(rewards, torch.Tensor) else rewards,
            "dones": torch.tensor(dones, dtype=torch.float32) if not isinstance(dones, torch.Tensor) else dones,
        })

        self._update_statistics()

    def _update_statistics(self):
        """Updates mean and std of node features for normalization."""
        all_node_features = [obs.request_quantity for traj in self.trajectories for obs in traj["observations"]]
        if len(all_node_features) > 0:
            all_node_features = torch.cat(all_node_features, dim=0)
            self.state_mean = all_node_features.mean(dim=0)
            self.state_std = all_node_features.std(dim=0) + 1e-6  # Avoid division by zero

    def _discount_cumsum(self, x, gamma=1.0):
        """Compute discounted cumulative sums of rewards."""
        discount_cumsum = torch.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def _create_empty_graph(self, device="cpu"):
        num_edges = 0
        num_nodes = 8

        return Data(
            edge_index=torch.empty((2, num_edges), dtype=torch.long).to(device),
            label=torch.ones(num_nodes).to(device),
            name=torch.zeros(num_nodes).to(device),
            request_quantity=torch.zeros(num_nodes).to(device),
            weight=torch.empty(num_edges, dtype=torch.float).to(device),
            num_nodes=num_nodes,
            add_mask=torch.zeros(num_nodes).to(device)
        )

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        """
        Retrieves a sampled subsequence from a random trajectory.

        :param index: Index of the sample (ignored, as we sample randomly).
        :return: Dictionary with batched training data.
        """
        traj = random.choice(self.trajectories)
        si = random.randint(0, len(traj["rewards"]) - 1)

        # Extract subsequence
        obs_graphs = traj["observations"][si:si + self.max_len]

        actions = traj["actions"][si: si + self.max_len]
        rewards = traj["rewards"][si: si + self.max_len].unsqueeze(-1)  # Shape (seq_len, 1)
        dones = traj["dones"][si: si + self.max_len]
        timesteps = torch.arange(si, si + len(obs_graphs)).long()

        # Compute returns-to-go
        rtg = self._discount_cumsum(traj["rewards"][si:], gamma=1.0)[:len(obs_graphs)].squeeze()  #TODO not sure about this

        if len(rtg.shape) == 0:
            rtg = rtg.unsqueeze(0)

        # Padding and normalization
        tlen = len(obs_graphs)
        pad_len = self.max_len - tlen

        # Pad with empty graphs
        device = obs_graphs[0].request_quantity.device
        padded_graphs = [self._create_empty_graph(device) for _ in range(pad_len)]
        obs_graphs = padded_graphs + obs_graphs

        actions = F.pad(actions, (0, 0, pad_len, 0), value=-1)
        rewards = F.pad(rewards, (0, 0, pad_len, 0))
        dones = F.pad(dones, (pad_len, 0), value=2)
        rtg = F.pad(rtg, (pad_len, 0 )).unsqueeze(-1) / self.scale
        timesteps = F.pad(timesteps, (pad_len, 0))
        mask = F.pad(torch.ones(tlen), (pad_len, 0))

        return {
            "graphs": obs_graphs,  # List of graphs
            "actions": actions,  # Action sequence
            "rewards": rewards,  # Reward values
            "returns_to_go": rtg,  # Discounted returns
            "timesteps": timesteps,  # Timestep info
            "attention_mask": mask,  # Mask for padding
            "dones": dones,  # Done flags
        }

    def sample_batch(self, batch_size):
        """
        Samples a batch of training examples from stored trajectories.

        :param batch_size: Number of samples to draw.
        :return: Dictionary containing batched training data.
        """
        batch_graphs = []
        batch_actions = []
        batch_rewards = []
        batch_rtg = []
        batch_timesteps = []
        batch_attention_mask = []
        batch_dones = []

        for _ in range(batch_size):
            # Sample a random trajectory
            traj = random.choice(self.trajectories)
            si = random.randint(0, len(traj["rewards"]) - 1)

            # Extract subsequence
            obs_graphs = traj["observations"][si:si + self.max_len]

            actions = traj["actions"][si: si + self.max_len]
            rewards = traj["rewards"][si: si + self.max_len].unsqueeze(-1)  # Shape (seq_len, 1)
            dones = traj["dones"][si: si + self.max_len]
            timesteps = torch.arange(si, si + len(obs_graphs)).long()

            # Compute returns-to-go
            rtg = self._discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                  :len(obs_graphs)].squeeze()  # TODO not sure about this

            if len(rtg.shape) == 0:
                rtg = rtg.unsqueeze(0)

            # Padding and normalization
            tlen = len(obs_graphs)
            pad_len = self.max_len - tlen

            # Pad with empty graphs
            device = obs_graphs[0].request_quantity.device
            padded_graphs = [self._create_empty_graph(device) for _ in range(pad_len)]
            obs_graphs = padded_graphs + obs_graphs

            actions = F.pad(actions, (0, 0, pad_len, 0), value=-1)
            rewards = F.pad(rewards, (0, 0, pad_len, 0))
            dones = F.pad(dones, (pad_len, 0), value=2)
            rtg = F.pad(rtg, (pad_len, 0)).unsqueeze(-1) / self.scale
            timesteps = F.pad(timesteps, (pad_len, 0))
            mask = F.pad(torch.ones(tlen), (pad_len, 0))

            # Store in batch
            batch_graphs.extend(obs_graphs)  # Collect all graphs for batching
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_rtg.append(rtg)
            batch_timesteps.append(timesteps)
            batch_attention_mask.append(mask)
            batch_dones.append(dones)

        batched_graphs = Batch.from_data_list(batch_graphs)

        return {
            "graphs": batched_graphs,  # Batched graph object
            "actions": torch.stack(batch_actions),  # (batch_size, max_len, action_dim)
            "returns_to_go": torch.stack(batch_rtg),  # (batch_size, max_len, 1)
            "timesteps": torch.stack(batch_timesteps, dim=0),  # (batch_size, max_len)
            "attention_mask": torch.stack(batch_attention_mask),  # (batch_size, max_len)
            "dones": torch.stack(batch_dones),  # (batch_size, max_len)
        }



# add and remove embeddings auch bei kombinatorischer swap nachbarschaft


class DecisionTransformerAgent:
    def __init__(self, env, model, dataset, batch_size=32, lr=1e-4, device="cuda"):
        self.env = env
        self.device = device
        self.dataset = dataset
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def collect_data(self, num_episodes=100):
        """Collects experience trajectories from the environment."""
        iterator = tqdm(range(num_episodes), desc="Collecting data...", unit="episode")
        for i in iterator:
            obs, _ = self.env.reset()
            done = False
            observations, actions, rewards, dones = [], [], [], []

            while not done:
                action = self.env.sample_action()
                next_obs, reward, done, _, _ = self.env.step(action)

                observations.append(obs)
                actions.append(torch.tensor(action, dtype=torch.float32))
                rewards.append(reward)
                dones.append(done)

                obs = next_obs

            self.dataset.add_trajectory(observations, actions, rewards, dones)
        logger.info("Data collection complete.")

    def save_dataset(self, file_path="dataset.pt"):
        """Saves the dataset to a file."""
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.dataset, file_path)
        logger.info(f"Dataset saved to {file_path}")

    def load_dataset(self, file_path="dataset.pt"):
        """Loads the dataset from a file."""
        self.dataset = torch.load(file_path)
        logger.info(f"Dataset loaded from {file_path}")

    def train(self, epochs=100):
        """Trains the Decision Transformer on stored trajectories."""
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            batch = self.dataset.sample_batch(self.batch_size)
            self.optimizer.zero_grad()

            # Extract batch elements
            graphs = batch["graphs"].to(self.device)
            actions = batch["actions"].to(self.device)
            returns_to_go = batch["returns_to_go"].to(self.device)
            timesteps = batch["timesteps"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            pred_actions = self.model(
                graphs, actions, returns_to_go, timesteps, attention_mask
            )

            loss1 = self._compute_masked_ce_loss(pred_actions[0], actions[:, :, 0], attention_mask)
            loss2 = self._compute_masked_ce_loss(pred_actions[1], actions[:, :, 1], attention_mask)
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            logger.info(f"Epoch {epoch + 1} Loss: {epoch_loss:.2f}")

    def _compute_masked_ce_loss(self, pred_action: torch.Tensor, targets: torch.Tensor,
                                padding_mask: torch.Tensor, reduction: str ="mean"):
        logits = pred_action.view(-1, self.model.num_locations)

        # Targets: (batch_size, seq_len) → (batch_size * seq_len)
        targets = targets.reshape(-1)

        # Mask: (batch_size, seq_len) → (batch_size * seq_len)
        mask = padding_mask.reshape(-1)

        # Compute Cross Entropy Loss for all elements
        loss = F.cross_entropy(logits, targets.long())  # Shape: (batch_size * seq_len)

        # Apply mask: Only compute loss where mask == 1
        masked_loss = loss * mask

        if reduction == "mean":
            return masked_loss.sum() / mask.sum()

        elif reduction == "sum":
            return masked_loss.sum()

        else:
            return masked_loss

    def evaluate(self, num_episodes=1):
        """Evaluates the trained model in the environment."""
        self.model.eval()
        total_rewards = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                with torch.no_grad():
                    graphs = Batch.from_data_list([obs]).to(self.device)
                    actions = torch.zeros((1, 2), device=self.device)  # Dummy initial action
                    returns_to_go = torch.zeros((1, 1), device=self.device)
                    timesteps = torch.zeros((1,), device=self.device, dtype=torch.long)
                    attention_mask = torch.ones((1,), device=self.device)

                    pred_action = self.model(
                        graphs, actions, returns_to_go, timesteps, attention_mask
                    )

                add_action = torch.argmax(pred_action[0], dim=-1).item()
                remove_action = torch.argmax(pred_action[1], dim=-1).item()
                obs, reward, done, _, _ = self.env.step(tuple(add_action, remove_action))
                total_reward += reward

            total_rewards.append(total_reward)

        avg_reward = sum(total_rewards) / num_episodes
        logger.info(f"Average Reward: {avg_reward:.2f}")
        return avg_reward


if __name__ == "__main__":
    from src.environment import FluidityEnvironment, FluidityEnvironmentConfig, TorchGraphObservationWrapper

    config = FluidityEnvironmentConfig(
        jar_path="/home/lukas/Projects/emusphere/simulator-xmr/target/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        jvm_options=['-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        configuration_directory_simulator="/home/lukas/flusim/simurun/",
        node_identifier="server0",
        device="cuda",
        feature_dim_node=1
    )

    env = FluidityEnvironment(config)
    env = TorchGraphObservationWrapper(env, one_hot=False)
    model = RSMDecisionTransformer(5, 32, 128, 2, 8)

    dataset = DecisionTransformerGraphDataset(max_size=10000, max_len=40)
    agent = DecisionTransformerAgent(env, model, dataset, batch_size=8)
    agent.collect_data(num_episodes=100)
    agent.save_dataset(".data/data.pt")
    #agent.load_dataset(".data/data.pt")
    #agent.train()
