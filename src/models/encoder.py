import torch_geometric.nn as gnn
import torch.nn as nn
import torch
from matplotlib import interactive
from torch_geometric.data import Data, Batch
from abc import ABC, abstractmethod

class BaseSpatialEncoder(ABC):
    def __init__(self,):
        self.assert_called = False

    @abstractmethod
    def forward(self, x):
        """
        The encoder expects a torch_geometric Data object as input with the following attributes:
        - label: The node labels of shape (N)
        - edge_index: The edge indices of shape (2, E)
        - weight: The edge weights of shape (E)
        Each graph should have num_location + num_client nodes, where the first num_location nodes are locations and the
        remaining nodes are clients. The encoder should output embeddings for each node in the graph
        Args:
            x: torch.Tensor: Torch_geometric Data object with the attributes described above
        Returns:
            torch.Tensor: The output tensor of shape (B, N, D)
        """
        assert hasattr(x, "label"), "Input data must have a 'label' attribute"
        assert hasattr(x, "edge_index"), "Input data must have a 'edge_index' attribute"
        assert hasattr(x, "weight"), "Input data must have a 'weight' attribute"
        self.assert_called = True


class GNNEncoder(BaseSpatialEncoder, nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        BaseSpatialEncoder.__init__(self,)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.conv1 = gnn.GCNConv(embedding_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, output_dim)


    def forward(self, data):
        BaseSpatialEncoder.forward(self, data)
        x, edge_index, edge_weight = data.label, data.edge_index, data.weight
        if x.dtype == torch.float32 or x.dtype == torch.float64:
            x = x.long().to(x.device)

        edge_weight = edge_weight.float().to(edge_weight.device)

        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class BaseTemporalEncoder(ABC):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.assert_called = False

    @abstractmethod
    def forward(self, x):
        """
        The temporal encoder takes in a sequence of location embeddings (not client embeddings!) and outputs a
        a temporal embedding for each location in the sequence. The input is expected to be of shape (B, T, N, D)
        Args:
            x: torch.Tensor: The input tensor of shape (B, T, N, D)
        Returns:
            torch.Tensor: The output tensor of shape (B, N, D)
        """
        assert len(x.size()) == 4, f"Expected input of shape (B, T, N, D), got {x.size()}"
        self.assert_called = True

class LSTMEncoder(BaseTemporalEncoder, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers=1):
        BaseTemporalEncoder.__init__(self, input_dim, hidden_dim, output_dim)
        nn.Module.__init__(self)
        lstm_layers = [nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False) for _ in range(num_lstm_layers)]
        self.lstm = nn.Sequential(*lstm_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        BaseTemporalEncoder.forward(self, x)
        B, T, N, D = x.size()
        x = x.view(B * N, T, D)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.view(B, N, -1)

class Attention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, type: str = "scaled_dot", num_heads: int = 4):

        nn.Module.__init__(self)
        self.type = type
        self.num_heads = num_heads

        # Projection layers
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)

        # Additive attention parameters
        if type == "additive":
            self.additive_w = nn.Linear(2 * hidden_dim, hidden_dim)
            self.additive_v = nn.Linear(hidden_dim, 1)

        # Multi-head Attention
        if type == "multihead":
            self.q_heads = nn.Linear(input_dim, hidden_dim * num_heads)
            self.k_heads = nn.Linear(input_dim, hidden_dim * num_heads)
            self.v_heads = nn.Linear(input_dim, hidden_dim * num_heads)
            self.output_projection = nn.Linear(hidden_dim * num_heads, hidden_dim)

        if type == "gated":
            self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        super().forward(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        if self.type == "scaled_dot":
            return F.scaled_dot_product_attention(q, k, v)

        elif self.type == "additive":
            N = x.shape[1]
            q_exp = q.unsqueeze(2).expand(-1, -1, N, -1)
            k_exp = k.unsqueeze(1).expand(-1, N, -1, -1)
            energy = F.tanh(self.additive_w(torch.cat([q_exp, k_exp], dim=-1)))
            attention_weights = F.softmax(self.additive_v(energy), dim=2)
            return (attention_weights * v.unsqueeze(1)).sum(dim=2)

        elif self.type == "multihead":
            B, N, _ = x.shape
            q_heads = self.q_heads(x).view(B, N, self.num_heads, -1)
            k_heads = self.k_heads(x).view(B, N, self.num_heads, -1)
            v_heads = self.v_heads(x).view(B, N, self.num_heads, -1)

            attn_scores = torch.einsum("bnhd,bmhd->bhnm", q_heads, k_heads) / (q_heads.shape[-1] ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.einsum("bhnm,bmhd->bnhd", attn_weights, v_heads)

            attn_output = attn_output.contiguous().view(B, N, -1)
            return self.output_projection(attn_output)

        elif self.type == "softmax_weighted":
            scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            return torch.bmm(attn_weights, v)

        elif self.type == "gated":
            gate_values = torch.sigmoid(self.gate(q))
            return gate_values * v

        else:
            raise ValueError(f"Unknown attention type: {self.type}")

if __name__ == "__main__":
    from aggregator import *
    from model import StandardModel
    model = StandardModel(5, 8, 128, 1, 8)
    obs1 = torch.load("../environment/test_data.pt")
    obs2 = torch.load("../environment/test_data2.pt")
    data = [obs1, obs2]
    flat_graphs = [graph for batch_graphs in data for graph in batch_graphs]
    batched_data = Batch.from_data_list(flat_graphs)
    out = model(batched_data)
    gnn = GNNEncoder(5, 12, 16, 16)
    out = gnn(batched_data)
    location_indices = torch.where(batched_data.label != 0)[0]
    location_embeddings = out[location_indices]
    temporal_location_embeddings = location_embeddings.view(len(data), len(obs1), -1, 16)
    lstm = LSTMEncoder(16, 32, 32)
    out = lstm(temporal_location_embeddings)
    agg = AttentionAggregator(input_dim=32, hidden_dim=32)
    node_action_mapper = SwapBasedAttentionActionMapper(2, 32, 8)
    shared_action_mapper = SharedActionMapper(1, 32, 8)
    interactive_action_mapper = InteractiveActionMapper(1, 32, 8, use_attn=True)
    out, weights = agg(out)
    actions = interactive_action_mapper(out)
    actions = node_action_mapper(agg)
    print(out)