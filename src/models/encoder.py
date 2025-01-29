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
        data.request_quantity = (data.request_quantity - data.request_quantity.mean()) / (
                    data.request_quantity.std() + 1e-6)
        x = torch.cat([x, data.request_quantity.unsqueeze(1)], dim=-1)
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GATEncoder(BaseSpatialEncoder, nn.Module):
    def __init__(self, num_layers: int, input_dim, embedding_dim, hidden_dim, output_dim, heads: int = 4,
                 act_fn: str = "gelu"):
        BaseSpatialEncoder.__init__(self,)
        assert num_layers > 1, "GAT must have at least 2 layers"
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.request_embedding = nn.Linear(1, embedding_dim)

        layers = nn.ModuleList()
        for i in range(num_layers - 1):
            layers.append(gnn.GATConv(embedding_dim * 2 if i == 0 else 0, hidden_dim, heads))
            layers.append(self._get_act_fn(act_fn))
            embedding_dim = hidden_dim * heads

        layers.append(gnn.GATConv(embedding_dim, output_dim, heads))
        layers.append(self._get_act_fn(act_fn))

        self.layers = nn.ModuleList(layers)

    def _get_act_fn(self, act_fn: str):
            match act_fn:
                case "tanh":
                    return nn.Tanh()
                case "relu":
                    return nn.ReLU()
                case "gelu":
                    return nn.GELU()
                case "leaky_relu":
                    return nn.LeakyReLU()
                case _:
                    raise ValueError(f"Unknown activation function: {act_fn}")

    def forward(self, data):
        BaseSpatialEncoder.forward(self, data)
        x, edge_index, edge_weight = data.label, data.edge_index, data.weight
        if x.dtype == torch.float32 or x.dtype == torch.float64:
            x = x.long().to(x.device)
        edge_weight = edge_weight.float().to(edge_weight.device)

        x = self.embedding(x)

        #Normalize  the request quantity
        data.request_quantity = (data.request_quantity - data.request_quantity.mean()) / (data.request_quantity.std() + 1e-6)

        request_embedding = self.request_embedding(data.request_quantity.unsqueeze(1))

        x = torch.cat([x, request_embedding], dim=-1)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, gnn.GATConv):
                x = layer(x, edge_index, edge_weight)  # Pass edge index and weights to GATConv
            else:
                x = layer(x)

        return x


class BaseTemporalEncoder(ABC):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.assert_called = False

    @abstractmethod
    def forward(self, x):
        """
        The temporal encoder takes in a sequence of location embeddings (not client embeddings!) and outputs a
        a temporal embedding for each location in the sequence. The input is expected to be of shape (B, T, N, D) or
        of shape(B, T, D) when we allready aggregated over the locations. The output should be of shape (B, N, D)
        Args:
            x: torch.Tensor: The input tensor of shape (B, T, N, D) or (B, T, D)
        Returns:
            torch.Tensor: The output tensor of shape (B, N, D)
        """
        assert len(x.size()) == 4 or len(x.size()) == 3, f"Expected input of shape (B, T, N, D) or (B, T, D) got {x.size()}"
        self.assert_called = True

class LSTMEncoder(BaseTemporalEncoder, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers=1, layer_norm: bool = True):
        BaseTemporalEncoder.__init__(self, input_dim, hidden_dim, output_dim)
        nn.Module.__init__(self)
        lstm_layers = []
        for i in range(num_lstm_layers):
            lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False))

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.lstm = nn.Sequential(*lstm_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        BaseTemporalEncoder.forward(self, x)
        if len(x.shape) == 3:
            B, T, D = x.size()
            x, _ = self.lstm(x)
            x = self.fc(x[:, -1, :])
            return x

        B, T, N, D = x.size()
        x = x.view(B * N, T, D)
        x, _ = self.lstm(x)
        x = self.layer_norm(x) if hasattr(self, "layer_norm") else x
        x = self.fc(x[:, -1, :])
        return x.view(B, N, -1)

class GRUEncoder(BaseTemporalEncoder, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gru_layers=1, layer_norm: bool = True):
        BaseTemporalEncoder.__init__(self, input_dim, hidden_dim, output_dim)
        nn.Module.__init__(self)
        gru_layers = []
        for i in range(num_gru_layers):
            gru_layers.append(nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False))

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.Sequential(*gru_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        BaseTemporalEncoder.forward(self, x)
        B, T, N, D = x.size()
        x = x.view(B * N, T, D)
        x, _ = self.gru(x)
        x = self.layer_norm(x) if hasattr(self, "layer_norm") else x
        x = self.fc(x[:, -1, :])
        return x.view(B, N, -1)


if __name__ == "__main__":
    from aggregator import *
    from model import StandardModel, RSMDecisionTransformer, RecurrentDecisionModel
    from src.algorithm.offline import DecisionTransformerGraphDataset
    from torch.utils.data import DataLoader
    #model = RSMDecisionTransformer(5, 8, 128, 1, 8)
    model = RSMDecisionTransformer(5, 8, 128, 1, 8)
    dataset = torch.load("../algorithm/.data/data.pt")
    r0 = dataset[0]
    for _ in range(5):
        r = dataset.sample_batch(12)
        #g = Batch.from_data_list(r["graphs"])
        out = model(r["graphs"], r["actions"], r["returns_to_go"],
                    r["timesteps"], r["attention_mask"], )

    #enc = GATEncoder(4, 5, 8, 16, 32, 2)
    #out = enc(batched_data)


