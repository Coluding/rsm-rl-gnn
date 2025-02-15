import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from src.models.model import RSMDecisionTransformer, CustomCrossProductDecisionTransformer
from src.utils import initialize_logger
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from src.environment import CrossProductActionSpace



logger = initialize_logger("decision_transformer.log")


@dataclass
class TrainingParams:
    learning_rate: float
    batch_size: int
    scheduler: str
    clip_grad_norm: float = 0
    scheduler_step_size: int = 0
    scheduler_gamma: float = 0.1
    num_epochs: int = 100
    device: str = "cuda"

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

    def save(self, path: str):
        data = {
            "trajectories": self.trajectories,
            "state_mean": self.state_mean,
            "state_std": self.state_std
        }

        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path)
        self.trajectories = data["trajectories"]
        self.state_mean = data["state_mean"]
        self.state_std = data["state_std"]

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
    def __init__(self, env = None, model = None, dataset = None, device="cuda",
                 cross_product_action_space : CrossProductActionSpace=None):
        self.env = env
        self.device = device
        self.dataset = dataset
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.cross_product_action_space = cross_product_action_space


    def collect_data(self, num_episodes=100):
        """Collects experience trajectories from the environment."""
        assert self.env is not None, "Environment not set."
        assert self.dataset is not None, "Dataset not set."

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

            if i % 100 == 0:
                self.save_dataset(f".data/data_backup{i}.pt")

            logger.info(f"Finished episode [{i}].")
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

    def save_model(self, file_path="model/dt.pt"):
        """Saves the model to a file."""
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.model.state_dict(), file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path="model/dt.pt"):
        """Loads the model from a file."""
        self.model.load_state_dict(torch.load(file_path))
        logger.info(f"Model loaded from {file_path}")

    def train(self, train_params: TrainingParams = None):
        """Trains the Decision Transformer on stored trajectories."""
        assert self.dataset is not None, "Dataset not set."
        assert self.model is not None, "Model not set."

        self.model.train()
        summary_writer = SummaryWriter(comment="decision_transformer")

        if train_params is None:
            train_params = TrainingParams(learning_rate=3e-5, batch_size=32, scheduler="none")

        optimizer = optim.AdamW(self.model.parameters(), lr=train_params.learning_rate)

        if train_params.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_params.scheduler_step_size,
                                                  gamma=train_params.scheduler_gamma)
        elif train_params.scheduler == "none":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {train_params.scheduler}")

        step = 0
        iterator = tqdm(range(train_params.num_epochs), desc="Training model...", unit="epoch")
        best_loss = float("inf")
        for epoch in iterator:
            epoch_loss = 0
            for _ in range(len(self.dataset)):
                batch = self.dataset.sample_batch(train_params.batch_size)
                optimizer.zero_grad()

                # Extract batch elements
                graphs = batch["graphs"].to(self.device)

                if self.cross_product_action_space is not None:
                    batch["actions"] = torch.stack([torch.tensor([self.cross_product_action_space[action[0].long().item(), action[1].long().item()] for action in actions]) for actions in batch["actions"]])

                actions = batch["actions"].to(self.device)
                returns_to_go = batch["returns_to_go"].to(self.device)
                timesteps = batch["timesteps"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                pred_actions = self.model(
                    graphs, actions, returns_to_go, timesteps, attention_mask
                )

                if self.cross_product_action_space is not None:
                    loss = self._compute_masked_ce_loss(pred_actions, actions, attention_mask)
                else:
                    loss1 = self._compute_masked_ce_loss(pred_actions[0], actions[:, :, 0], attention_mask)
                    loss2 = self._compute_masked_ce_loss(pred_actions[1], actions[:, :, 1], attention_mask)
                    loss = loss1 + loss2

                loss.backward()

                if train_params.clip_grad_norm > 0:
                    summary_writer.add_scalar("Gradient Norm", nn.utils.clip_grad_norm_(self.model.parameters(), train_params.clip_grad_norm), epoch)
                    nn.utils.clip_grad_norm_(self.model.parameters(), train_params.clip_grad_norm)

                optimizer.step()
                epoch_loss += loss.item()
                summary_writer.add_scalar("Step Loss", loss.item(), step)
                step += 1

                if scheduler is not None:
                    scheduler.step()
                    summary_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save_model()

            logger.info(f"Epoch {epoch + 1} Loss: {epoch_loss / len(self.dataset):.2f}")
            summary_writer.add_scalar("Epoch Loss", epoch_loss / len(self.dataset), epoch)

    def _compute_masked_ce_loss(self, pred_action: torch.Tensor, targets: torch.Tensor,
                                padding_mask: torch.Tensor, reduction: str ="mean"):
        logits = pred_action.view(-1, self.model.num_locations ** 2 + 1 if self.cross_product_action_space is not None else self.model.num_locations)

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
        assert self.env is not None, "Environment not set."
        assert self.model is not None, "Model not set."

        self.model.eval()
        total_rewards = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            # Initialize sequence storage
            all_obs = [obs]
            all_actions = [torch.zeros((1, 2), device=self.device)]  # Dummy initial action
            all_returns = []  # Arbitrary high return
            all_timesteps = [torch.zeros((1, 1), device=self.device, dtype=torch.long)]
            all_attention_mask = [torch.ones((1, 1), device=self.device)]

            while not done:
                counter = 0
                with (torch.no_grad()):
                    graphs = Batch.from_data_list(all_obs).to(self.device)

                    # Convert lists to tensors
                    actions = torch.cat(all_actions, dim=0).unsqueeze(0)  # Shape: (1, seq_len, action_dim)
                    if len(all_returns) == 0:
                        returns_to_go = torch.tensor([[100.]], device=self.device).unsqueeze(0)
                    else:
                        returns_to_go = torch.cat((all_returns, torch.tensor(100).unsqueeze(0)), dim=-1).unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_len, 1)
                    #all_returns.append(torch.tensor([[100.]], device=self.device))
                    #returns_to_go = torch.cat(returns_to_go, dim=0).unsqueeze(0)  # Shape: (1, seq_len, 1)
                    timesteps = torch.cat(all_timesteps, dim=-1)  # Shape: (1, seq_len)
                    attention_mask = torch.cat(all_attention_mask, dim=-1)  # Shape: (1, seq_len)

                    # Forward pass
                    pred_action = self.model(graphs, actions, returns_to_go, timesteps, attention_mask)

                # Select actions from prediction
                add_action = torch.argmax(pred_action[0][:,-1], dim=-1).item()
                remove_action = torch.argmax(pred_action[1][:,-1], dim=-1).item()

                # Step environment
                obs, reward, done, _, _ = self.env.step((add_action, remove_action))
                total_reward += reward

                # Append new observations to sequences
                all_obs.append(obs)
                total_rewards.append(reward)
                # cumsum the rewards
                all_returns = torch.cumsum(torch.tensor(total_rewards), 0)
                # normalize the returns
                all_returns = (all_returns - self.dataset.state_mean) / self.dataset.state_std
                all_actions.append(torch.tensor([[add_action, remove_action]], device=self.device))
                #all_returns.append(torch.tensor([[max(0, all_returns[-1].item() - reward)]], device=self.device))
                #all_returns.append(torch.tensor([[-10000 + all_returns[-1].item()]], device=self.device))
                all_timesteps.append(torch.tensor([[len(all_timesteps)]], device=self.device, dtype=torch.long))
                all_attention_mask.append(torch.ones((1, 1), device=self.device))  # Ensure attention mask stays valid

                if counter % 10 == 0:
                    logger.info(f"Actions: {add_action}, {remove_action}")
                counter += 1


            total_rewards.append(total_reward)

        avg_reward = sum(total_rewards) / num_episodes
        logger.info(f"Average Reward: {avg_reward:.2f}")
        return avg_reward


if __name__ == "__main__":
    from src.environment import FluidityEnvironment, FluidityEnvironmentConfig, TorchGraphObservationWrapper

    config = FluidityEnvironmentConfig(
        jar_path="../../ressources/jars/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        jvm_options=['-Djava.security.properties=../../ressources/simurun/server0/xmr/config/java.security'],
        configuration_directory_simulator="../../ressources/simurun_400/",
        node_identifier="server0",
        device="mps",
        feature_dim_node=1
    )

    env = FluidityEnvironment(config)
    env = TorchGraphObservationWrapper(env, one_hot=False)
    model = RSMDecisionTransformer(True, 5, 32, 128, 2, 8,
                                   max_ep_len=400, max_position_embedding=120)

    train_params = TrainingParams(learning_rate=3e-5, batch_size=4, scheduler="none", num_epochs=1000,
                                  clip_grad_norm=1.0)

    dataset = DecisionTransformerGraphDataset(max_size=1_000_000, max_len=40)
    ca = CrossProductActionSpace.from_json("../data/action_space.json")
    agent = DecisionTransformerAgent(env=env, model=model, dataset=dataset, device="cpu", cross_product_action_space=ca)
    agent.load_dataset(".data/data_backup500.pt")
    #agent.collect_data(num_episodes=500)
    #agent.load_dataset(".data/data.pt")
    #agent.load_dataset(".data/data.pt")
    #agent.dataset.max_len = 40
    #agent.load_model("model/dt.pt")
    #agent.evaluate(num_episodes=1)
    #agent.save_dataset(".data/data.pt")
    agent.train(train_params)
