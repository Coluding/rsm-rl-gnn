import torch

from src.models.encoder import *
from src.models.aggregator import *


class BaseSwapModel(ABC):
    def __init__(self):
        self.assert_called = False
        self.spatial_encoder: BaseSpatialEncoder = None
        self.temporal_encoder: BaseTemporalEncoder = None
        self.aggregator: BaseAggregator = None
        self.action_mapper: SwapActionMapper | SwapActionNodeMapper = None

    @abstractmethod
    def forward(self, x):
        """
        The model takes in a torch_geometric Data object and outputs logits for each add and remove actions. The input
        data object should have the following attributes:
        - label: The node labels of shape (N)
        - edge_index: The edge indices of shape (2, E)
        - weight: The edge weights of shape (E)
        Args:
            x: torch.Tensor: Torch_geometric Data object with the attributes described above
        Returns:
            torch.Tensor: The output tensor of shape (B, 2)
        """
        assert hasattr(x, "label"), "Input data must have a 'label' attribute"
        assert hasattr(x, "edge_index"), "Input data must have a 'edge_index' attribute"
        assert hasattr(x, "weight"), "Input data must have a 'weight' attribute"
        self.assert_called = True


class StandardModel(BaseSwapModel, nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int,
                 action_layer: int = 1, num_locations: int = 8, num_heads = 2):
        BaseSwapModel.__init__(self)
        nn.Module.__init__(self)
        self.num_locations = num_locations
        self.heads = num_heads
        self.spatial_encoder = GATEncoder(num_layers=4, input_dim=input_dim, embedding_dim=embedding_dim,
                                          hidden_dim=hidden_dim, output_dim=hidden_dim, heads=2)
        self.temporal_encoder = LSTMEncoder(hidden_dim * num_heads, hidden_dim, hidden_dim)
        self.aggregator = AttentionAggregator(hidden_dim, hidden_dim)
        self.action_mapper = InteractiveActionMapper(action_layer, hidden_dim, num_locations, use_attn=True)

    def forward(self, x, temporal_size: int = 3, return_attn=False):
        BaseSwapModel.forward(self, x)
        all_embeddings = self.spatial_encoder(x)
        B = len(x.batch.unique()) // temporal_size
        # We discard the client node embeddings as the decision is made only for the locations and the GNN encoder
        # should compress the client embeddings into the location embeddings with its message passing mechanism
        # Why? This is much more efficient, as we are now only working with the locations and not the clients in the
        # temporal dimension, drastically reducing the number of nodes we need to process as num_clients > num_locations
        location_indices = torch.where(x.label != 0)[0]
        #L: Number of locations
        #Shape: B*L*T,D
        location_embeddings = all_embeddings[location_indices]
        # Shape B,T,L,D: Batch, Temporal, Locations, Embedding
        temporal_location_embeddings = location_embeddings.view(B, temporal_size, -1, location_embeddings.size(-1))

        # For DQN not needed
        #add_mask = x.add_mask[location_indices].view(B, temporal_size, self.num_locations)[:, -1, :]

        #Shape: B, L, D as we are aggregating over the temporal dimension using LSTM
        temporal_output = self.temporal_encoder(temporal_location_embeddings)

        #Shape: B, D as we are aggregating over the location dimension using attention
        aggregated_temporal_node_embeddings, attn_weights = self.aggregator(temporal_output)

        #Shape: B, L, 2 as we are mapping the aggregated embeddings to actions
        add_q_vals, remove_q_vals = self.action_mapper(aggregated_temporal_node_embeddings,) #add_mask)

        if return_attn:
            return (add_q_vals, remove_q_vals), attn_weights

        return add_q_vals, remove_q_vals