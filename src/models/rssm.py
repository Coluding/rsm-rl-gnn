import torch.nn as nn
import torch
import torch_geometric
import torch.nn.functional as F

from src.models import GATEncoder, AttentionAggregator


class DynamicsModel(nn.Module):
    """
    This is the dynamics model of the RSSM.
    """
    def __init__(self, hidden_dim: int, action_dim: int, state_dim: int, obs_dim: int, rnn_layer: int = 1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Can be any recurrent network
        self.rnn = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layer)])

        # Projection layer to make efficient use of concatenated inputs
        self.project_state_action = nn.Linear(action_dim + state_dim, hidden_dim)

        # Return mean and log-variance of the normal distribution
        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(hidden_dim + action_dim, hidden_dim)

        # Return mean and log-variance of the normal distribution
        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_obs = nn.Linear(hidden_dim + obs_dim, hidden_dim)

        self.state_dim = state_dim
        self.act_fn = nn.functional.gelu

    def forward(self, prev_hidden: torch.Tensor, prev_state: torch.Tensor, actions: torch.Tensor,
                obs: torch.Tensor = None, dones: torch.Tensor = None):
        """
        Forward pass of the dynamics model for one time step.
        :param prev_hidden: Previous hidden state of the RNN: (batch_size, hidden_dim)
        :param prev_state: Previous stochastic state: (batch_size, state_dim)
        :param action: One hot encoded actions: (sequence_length, batch_size, action_dim)
        :param obs: This is the encoded observation from the encoder, not the raw observation!: (sequence_length, batch_size, embedding_dim)
        :param dones: Terminal states of the environment
        :return:
        """

        B, T, _ = actions.size()

        hiddens_list = []
        posterior_means_list = []
        posterior_logvars_list = []
        prior_means_list = []
        prior_logvars_list = []
        prior_states_list = []
        posterior_states_list = []

        # (B, 1, hidden_dim)
        hiddens_list.append(prev_hidden.unsqueeze(1))
        prior_states_list.append(prev_state.unsqueeze(1))
        posterior_states_list.append(prev_state.unsqueeze(1))

        for t in range(T):
            action_t = actions[:, t, :]
            obs_t = obs[:, t, :] if obs is not None else torch.zeros(B, self.embedding_dim, device=actions.device)
            state_t = posterior_states_list[-1][:, 0, :] if obs is not None else prior_states_list[-1][:, 0, :]
            state_t = state_t if dones is None else state_t * (1 - dones[:, t, :])
            hidden_t = hiddens_list[-1][:, 0, :]

            state_action = torch.cat([state_t, action_t], dim=-1)
            state_action = self.act_fn(self.project_state_action(state_action))

            ### Update the deterministic hidden state ###
            for rnn in self.rnn:
                # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
                hidden_t = rnn(state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            # p(s_t | s_{t-1}, a_{t-1}) -- prior
            hidden_action = self.act_fn(self.project_hidden_action(hidden_action))
            prior_params = self.prior(hidden_action)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            ### Sample from the prior distribution ###
            prior_dist = torch.distributions.Normal(prior_mean, torch.exp(F.softplus(prior_logvar)))
            prior_state_t = prior_dist.rsample()

            ### Determine the posterior distribution ###
            # If observations are not available, we just use the prior
            if obs is None:
                posterior_mean = prior_mean
                posterior_logvar = prior_logvar
            else:
                # p(s_t | s_{t-1}, a_{t-1}, o_t) -- posterior --> Only if observations are available
                # If not we can sample form the prior
                hidden_obs = torch.cat([hidden_t, obs_t], dim=-1)
                hidden_obs = self.act_fn(self.project_hidden_obs(hidden_obs))
                posterior_params = self.posterior(hidden_obs)
                posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            ### Sample from the posterior distribution ###
            posterior_dist = torch.distributions.Normal(posterior_mean, torch.exp(F.softplus(posterior_logvar)))

            # Make sure to use rsample to enable the gradient flow
            # Otherwise we could also use code the reparameterization trick by hand
            posterior_state_t = posterior_dist.rsample()

            ### Store results in lists (instead of in-place modification) ###
            posterior_means_list.append(posterior_mean.unsqueeze(1))
            posterior_logvars_list.append(posterior_logvar.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_logvars_list.append(prior_logvar.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

            # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_logvars = torch.cat(prior_logvars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_logvars = torch.cat(posterior_logvars_list, dim=1)

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars


class LatentDecoder(nn.Module):
    """
    This is a simple decoder that maps the recurrent hidden state to the latent space of the graph.
    """

    def __init__(self, hidden_dim: int, state_dim: int, embedding_dim: int):
        super(LatentDecoder, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x




class GraphEncoder(nn.Module):
    """
    This is a simple encoder that maps the graph to a fixed size embedding.
    """
    def __init__(self, num_locations: int,  input_dim: int, embedding_dim: int):
        super().__init__()
        self.num_locations = num_locations
        self.spatial_encoder = GATEncoder(num_layers=4, input_dim=input_dim, embedding_dim=embedding_dim,
                                          hidden_dim=embedding_dim, output_dim=embedding_dim, heads=2, )


        self.aggregator = AttentionAggregator(2 * embedding_dim, embedding_dim)

    def forward(self, x: torch_geometric.data.Data):
        all_embeddings = self.spatial_encoder(x)
        B = len(x.batch.unique())
        # Select location embeddings only
        location_indices = torch.where(x.label != 0)[0]
        location_embeddings = all_embeddings[location_indices]
        location_embeddings = location_embeddings.view(B, -1, location_embeddings.shape[-1])
        graph_embedding, _ = self.aggregator(location_embeddings)
        return graph_embedding


class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class RSSM:
    def __init__(self,
                 encoder: GraphEncoder,
                 decoder: LatentDecoder,
                 reward_model: RewardModel,
                 dynamics_model: nn.Module,
                 hidden_dim: int,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 device: str = "cpu"):
        """
        Recurrent State-Space Model (RSSM) for learning dynamics models.

        Args:
            encoder: Encoder network for deterministic state
            prior_model: Prior network for stochastic state
            decoder: Decoder network for reconstructing observation
            sequence_model: Recurrent model for deterministic state
            hidden_dim: Hidden dimension of the RNN
            latent_dim: Latent dimension of the stochastic state
            action_dim: Dimension of the action space
            obs_dim: Dimension of the encoded observation space


        """
        super(RSSM, self).__init__()

        self.dynamics = dynamics_model
        self.encoder = encoder
        self.decoder = decoder
        self.reward_model = reward_model

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        #shift to device
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reward_model.to(device)

    def generate_rollout(self, actions: torch.Tensor, hiddens: torch.Tensor = None, states: torch.Tensor = None,
                         obs: torch.Tensor = None, dones: torch.Tensor = None):

        if hiddens is None:
            hiddens = torch.zeros(actions.size(0), self.hidden_dim).to(actions.device)

        if states is None:
            states = torch.zeros(actions.size(0), self.state_dim).to(actions.device)

        dynamics_result = self.dynamics(hiddens, states, actions, obs, dones)
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics_result

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars

    def train(self):
        self.dynamics.train()
        self.encoder.train()
        self.decoder.train()
        self.reward_model.train()

    def eval(self):
        self.dynamics.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.reward_model.eval()

    def encode(self, obs: torch.Tensor):
        return self.encoder(obs)

    def decode(self, state: torch.Tensor):
        return self.decoder(state)

    def predict_reward(self, h: torch.Tensor, s: torch.Tensor):
        return self.reward_model(h, s)

    def parameters(self):
        return list(self.dynamics.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.reward_model.parameters())

    def save(self, path: str):
        torch.save({
            "dynamics": self.dynamics.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_model": self.reward_model.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])


    def to(self, device):
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reward_model.to(device)

        return self