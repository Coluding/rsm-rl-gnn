import torch
from torchrl.modules import DecisionTransformer
from transformers import (DecisionTransformerConfig, DecisionTransformerModel, DecisionTransformerPreTrainedModel,
                          DecisionTransformerGPT2Model)
import torch_geometric
from typing import Optional, Tuple

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
        The model takes in a torch_geometric Data object and outputs logits/Q-values for each add and remove actions.
        The input data object should have the following attributes:
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

class BaseCrossProductActionSpaceModel(ABC):
    def __init__(self):
        self.assert_called = False
        self.spatial_encoder: BaseSpatialEncoder = None
        self.temporal_encoder: BaseTemporalEncoder = None
        self.aggregator: BaseAggregator = None
        self.action_mapper: SwapActionMapper | SwapActionNodeMapper = None

    @abstractmethod
    def forward(self, x):
        """
        The model takes in a torch_geometric Data object and outputs logits/Q-values for the cross product between
        add and remove action, i.e. num_locations**2 actions.
        The input data object should have the following attributes:
        - label: The node labels of shape (N)
        - edge_index: The edge indices of shape (2, E)
        - weight: The edge weights of shape (E)
        Args:
            x: torch.Tensor: Torch_geometric Data object with the attributes described above
        Returns:
            torch.Tensor: The output tensor of shape (B, 1)
        """
        assert hasattr(x, "label"), "Input data must have a 'label' attribute"
        assert hasattr(x, "edge_index"), "Input data must have a 'edge_index' attribute"
        assert hasattr(x, "weight"), "Input data must have a 'weight' attribute"
        self.assert_called = True


class BaseValueModel(ABC):
    def __init__(self):
        self.assert_called = False
        self.spatial_encoder: BaseSpatialEncoder = None
        self.temporal_encoder: BaseTemporalEncoder = None
        self.aggregator: BaseAggregator = None
        self.action_mapper: SwapActionMapper | SwapActionNodeMapper = None

    @abstractmethod
    def forward(self, x):
        """
        The model takes in a torch_geometric Data object and outputs the state value for each add and remove actions.
        The input data object should have the following attributes:
        - label: The node labels of shape (N)
        - edge_index: The edge indices of shape (2, E)
        - weight: The edge weights of shape (E)
        Args:
            x: torch.Tensor: Torch_geometric Data object with the attributes described above
        Returns:
            torch.Tensor: The output tensor of shape (B, 1)
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
        self.spatial_encoder = GATEncoder(num_layers=3, input_dim=input_dim, embedding_dim=embedding_dim,
                                          hidden_dim=hidden_dim, output_dim=hidden_dim, heads=2)
        self.temporal_encoder = GRUEncoder(hidden_dim * num_heads, hidden_dim, hidden_dim)
        self.aggregator = AttentionAggregator(hidden_dim, hidden_dim)
        self.action_mapper = InteractiveActionMapper(action_layer, hidden_dim, num_locations, use_attn=True)

    def forward(self, x, temporal_size: int = 4, return_attn=False):
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

        #Shape: B, L, D as we are aggregating over the temporal dimension using LSTM
        temporal_output = self.temporal_encoder(temporal_location_embeddings)

        #Shape: B, D as we are aggregating over the location dimension using attention
        aggregated_temporal_node_embeddings, attn_weights = self.aggregator(temporal_output)

        #Shape: B, L, 2 as we are mapping the aggregated embeddings to actions
        add_vals, remove_vals = self.action_mapper(aggregated_temporal_node_embeddings,) #add_mask)

        if return_attn:
            return (add_vals, remove_vals), attn_weights

        return add_vals, remove_vals


class StandardCrossProductModel(BaseCrossProductActionSpaceModel, nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int,
                 action_layer: int = 1, num_locations: int = 8, num_heads = 2,
                 use_client_embeddings: bool = True, max_timesteps : int = None):
        BaseCrossProductActionSpaceModel.__init__(self)
        nn.Module.__init__(self)
        self.num_locations = num_locations
        self.heads = num_heads
        self.use_client_embeddings = use_client_embeddings
        self.spatial_encoder = GATEncoder(num_layers=3, input_dim=input_dim, embedding_dim=embedding_dim,
                                          hidden_dim=hidden_dim, output_dim=hidden_dim, heads=2)
        self.temporal_encoder = GRUEncoder(hidden_dim * num_heads, hidden_dim, hidden_dim)
        self.aggregator = AttentionAggregator(hidden_dim, hidden_dim)
        self.action_mapper = StandardCrossProductActionMapper(action_layer, hidden_dim, num_locations,
                                                              max_timestep=max_timesteps)

    def forward(self, x, timestep: torch.Tensor = None, temporal_size: int = 4, return_attn=False):
        BaseCrossProductActionSpaceModel.forward(self, x)
        all_embeddings = self.spatial_encoder(x)
        B = len(x.batch.unique()) // temporal_size

        if not self.use_client_embeddings:
            # Select location embeddings only
            location_indices = torch.where(x.label != 0)[0]
            location_embeddings = all_embeddings[location_indices]
            temporal_location_embeddings = location_embeddings.view(B, temporal_size, -1, location_embeddings.size(-1))
            node_embeddings = temporal_location_embeddings
        else:
            node_embeddings = all_embeddings.view(B, temporal_size, -1, all_embeddings.size(-1))


        #Shape: B, N, D as we are aggregating over the temporal dimension using LSTM
        temporal_output = self.temporal_encoder(node_embeddings)

        #Shape: B, D as we are aggregating over the node dimension using attention
        aggregated_temporal_node_embeddings, attn_weights = self.aggregator(temporal_output)

        #Shape: B, L, 2 as we are mapping the aggregated embeddings to actions
        action_values = self.action_mapper(aggregated_temporal_node_embeddings, timestep) #add_mask)

        return action_values


class StandardValueModel(BaseValueModel, nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int,
                 num_locations: int = 8, num_heads: int = 2, max_timestep: int = None,
                 use_client_embeddings=False
                 ):
        nn.Module.__init__(self)
        BaseValueModel.__init__(self)
        self.num_locations = num_locations
        self.spatial_encoder = GATEncoder(
            num_layers=3, input_dim=input_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, output_dim=hidden_dim, heads=num_heads
        )

        self.use_client_embeddings = use_client_embeddings

        if max_timestep is not None:
            self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)

        self.temporal_encoder = GRUEncoder(hidden_dim * num_heads, hidden_dim, hidden_dim)
        self.aggregator = AttentionAggregator(hidden_dim, hidden_dim)

        # Single output head for value estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 if max_timestep is not None else hidden_dim * 1 , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single scalar value V(s)
        )

    def forward(self, x, timestep = None, temporal_size: int = 4):
        all_embeddings = self.spatial_encoder(x)
        B = len(x.batch.unique()) // temporal_size

        if not self.use_client_embeddings:
            # Select location embeddings only
            location_indices = torch.where(x.label != 0)[0]
            location_embeddings = all_embeddings[location_indices]
            temporal_location_embeddings = location_embeddings.view(B, temporal_size, -1, location_embeddings.size(-1))
            node_embeddings = temporal_location_embeddings
        else:
            node_embeddings = all_embeddings.view(B, temporal_size, -1, all_embeddings.size(-1))

        # Temporal encoding
        temporal_output = self.temporal_encoder(node_embeddings)

        # Aggregate over locations
        aggregated_temporal_node_embeddings, _ = self.aggregator(temporal_output)

        if timestep is not None:
            timestep_embedding = self.timestep_embedding(timestep)
            aggregated_temporal_node_embeddings = torch.cat([aggregated_temporal_node_embeddings, timestep_embedding], dim=-1)

        # Predict state value V(s)
        # Shape: B, 1
        state_value = self.value_head(aggregated_temporal_node_embeddings)

        return state_value.squeeze(-1)


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int,
                 num_locations: int = 8, num_heads: int = 2):
        nn.Module.__init__(self)
        self.num_locations = num_locations
        self.spatial_encoder = GATEncoder(
            num_layers=3, input_dim=input_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, output_dim=hidden_dim, heads=num_heads
        )
        self.temporal_encoder = GRUEncoder(hidden_dim * num_heads, hidden_dim, hidden_dim)
        self.aggregator = MeanAggregator()

    def forward(self, x, temporal_size: int = 4):
        all_embeddings = self.spatial_encoder(x)
        B = len(x.batch.unique()) // temporal_size

        # Select location embeddings only
        location_indices = torch.where(x.label != 0)[0]
        location_embeddings = all_embeddings[location_indices]
        temporal_location_embeddings = location_embeddings.view(B, temporal_size, -1, location_embeddings.size(-1))

        # Temporal encoding
        temporal_output = self.temporal_encoder(temporal_location_embeddings)

        # Aggregate over locations
        aggregated_temporal_node_embeddings = self.aggregator(temporal_output)

        return aggregated_temporal_node_embeddings


class CustomDecisionTransformerModel(DecisionTransformerPreTrainedModel):
    def __init__(self, config, num_actions_1: int , num_actions_2: int):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = nn.Linear(1, config.hidden_size)
        self.embed_state = nn.Linear(config.state_dim, config.hidden_size,)

        # Separate embedding layers for two actions
        self.embed_action1 = nn.Linear(num_actions_1, config.hidden_size // 2)
        self.embed_action2 = nn.Linear(num_actions_2, config.hidden_size // 2)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # Separate prediction heads for two actions
        self.predict_action1 = nn.Linear(config.hidden_size, num_actions_1)
        self.predict_action2 = nn.Linear(config.hidden_size, num_actions_2)

        self.predict_state = nn.Linear(config.hidden_size, config.state_dim)
        self.predict_return = nn.Linear(config.hidden_size, 1)

        self.post_init()

    def forward(
            self,
            states: Optional[torch.FloatTensor] = None,
            actions: Optional[torch.FloatTensor] = None,
            returns_to_go: Optional[torch.FloatTensor] = None,
            timesteps: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embed states, returns, timesteps
        state_embeddings = self.embed_state(states)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Embed both actions separately
        action1_embeddings = self.embed_action1(actions[:, :, 0])  # First action
        action2_embeddings = self.embed_action2(actions[:, :, 1])  # Second action

        # Concatenate action embeddings
        action_embeddings = torch.cat([action1_embeddings, action2_embeddings], dim=-1)

        # Add time embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # Stack input sequence (return, state, action)
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Adjust attention mask
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long),
        )

        x = encoder_outputs[0].reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Predict actions separately
        action_preds1 = self.predict_action1(x[:, 1])  # Predict first action
        action_preds2 = self.predict_action2(x[:, 1])  # Predict second action

        return action_preds1, action_preds2

class CustomCrossProductDecisionTransformer(DecisionTransformerPreTrainedModel):
    def __init__(self, config, num_actions: int):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = nn.Linear(1, config.hidden_size)
        self.embed_state = nn.Linear(config.state_dim, config.hidden_size,)

        # Separate embedding layers for two actions
        self.embed_action = nn.Linear(num_actions, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # Separate prediction heads for two actions
        self.predict_action = nn.Linear(config.hidden_size, num_actions)
        self.predict_state = nn.Linear(config.hidden_size, config.state_dim)
        self.predict_return = nn.Linear(config.hidden_size, 1)

        self.post_init()

    def forward(
            self,
            states: Optional[torch.FloatTensor] = None,
            actions: Optional[torch.FloatTensor] = None,
            returns_to_go: Optional[torch.FloatTensor] = None,
            timesteps: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embed states, returns, timesteps
        state_embeddings = self.embed_state(states)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Embed both actions separately
        action_embeddings = self.embed_action(actions)  # First action

        # Add time embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # Stack input sequence (return, state, action)
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Adjust attention mask
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long),
        )

        x = encoder_outputs[0].reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Predict actions separately
        action_preds = self.predict_action(x[:, 1]) # Why this slicing? Because we need of every embedding in shape B,N,D the embedding that corresponds to the state and reward before the action

        return action_preds

class RSMDecisionTransformer(nn.Module):
    def __init__(self, cross_product: bool, input_dim: int, embedding_dim: int, hidden_dim: int,
                 dt_heads: int = 4, num_locations: int = 8, max_ep_len: int = 40,
                 max_position_embedding: int = 400):

        super().__init__()
        self.num_locations = num_locations
        self.spatial_encoder = GATEncoder(num_layers=4, input_dim=input_dim, embedding_dim=embedding_dim,
                                          hidden_dim=hidden_dim, output_dim=hidden_dim, heads=2,)

        self.dt_config = DecisionTransformerConfig(state_dim=hidden_dim * 2, n_head=dt_heads, num_layers=2,
                                                   act_dim=num_locations,
                                                   max_ep_len=max_ep_len, action_tanh=False,
                                                   vocab_size=num_locations,
                                                   max_position_embeddings=max_position_embedding,
                                                   )

        self.aggregator = AttentionAggregator(2 * hidden_dim, hidden_dim)

        if cross_product:
            self.dt = CustomCrossProductDecisionTransformer(self.dt_config, num_actions=num_locations)
        else:
            self.dt = CustomDecisionTransformerModel(self.dt_config, num_actions_1=num_locations, num_actions_2=num_locations)


    def forward(self,
                x: torch_geometric.data.Data,
                action: torch.Tensor,
                return_to_go: torch.Tensor,
                timesteps: torch.Tensor,
                attn_mask: torch.Tensor = None,
                ):

        all_embeddings = self.spatial_encoder(x)
        T = action.shape[1]
        B = action.shape[0]
        # Select location embeddings only
        location_indices = torch.where(x.label != 0)[0]
        location_embeddings = all_embeddings[location_indices]
        location_embeddings = location_embeddings.view(B, T, -1, location_embeddings.shape[-1])

        graph_embedding, _ = self.aggregator(location_embeddings)

        #ohe action
        action[action == -1] = 0
        action_ohe = F.one_hot(action.long(), num_classes=self.num_locations).float()

        # We do not pass an attention mask yet because it is only for apssing not causality. That is handled by the GPT internally
        out = self.dt.forward(graph_embedding,
                              action_ohe,
                              returns_to_go=return_to_go,
                              timesteps=timesteps,
                              attention_mask=attn_mask)

        return out


class RecurrentDecisionModel(nn.Module):
    def __init__(self, input_dim: int, node_type_embedding_dim: int, hidden_dim: int,
                 num_lstm_layers: int = 3, num_locations: int = 8, num_gat_heads: int = 2,
                 num_actions : int = 2):
        super().__init__()
        self.num_locations = num_locations
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Spatial Encoder: GAT
        self.spatial_encoder = GATEncoder(num_layers=4, input_dim=input_dim, embedding_dim=node_type_embedding_dim,
                                          hidden_dim=hidden_dim, output_dim=hidden_dim, heads=num_gat_heads)

        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(hidden_dim * num_gat_heads, hidden_dim)

        # Separate embedding layers for two actions
        self.embed_action1 = nn.Linear(num_locations, hidden_dim // 2)
        self.embed_action2 = nn.Linear(num_locations, hidden_dim // 2)

        self.embed_ln = nn.LayerNorm(hidden_dim)

        self.aggregator = MeanAggregator()

        # Temporal Encoder: GRU
        self.temporal_encoder = LSTMEncoder(hidden_dim, hidden_dim, hidden_dim, num_lstm_layers)

        if num_actions == 2:
            self.action_mapper = InteractiveActionMapper(1, hidden_dim, num_locations, use_attn=True)
        else:
            self.action_mapper = nn.Linear(hidden_dim, num_locations)

    def forward(self,
                x: torch_geometric.data.Data,
                actions: torch.Tensor,
                returns_to_go: torch.Tensor,
                timesteps: torch.Tensor, attention_mask: torch.Tensor):
        B, T = actions.shape[:2]

        all_embeddings = self.spatial_encoder(x)

        # Select location embeddings only
        location_indices = torch.where(x.label != 0)[0]
        location_embeddings = all_embeddings[location_indices]
        location_embeddings = location_embeddings.view(B, T, -1, location_embeddings.size(-1))

        # Apply attention mask to ignore padded inputs
        location_embeddings = location_embeddings

        graph_embedding = self.aggregator(location_embeddings)

        # ohe action
        actions[actions == -1] = 0
        action_ohe = F.one_hot(actions.long(), num_classes=self.num_locations).float()

        state_embeddings = self.embed_state(graph_embedding)
        returns_embeddings = self.embed_return(returns_to_go)
        action1_embeddings = self.embed_action1(action_ohe[:, :, 0].float())  # First action
        action2_embeddings = self.embed_action2(action_ohe[:, :, 1].float())  # Second action

        # Concatenate action embeddings
        action_embeddings = torch.cat([action1_embeddings, action2_embeddings], dim=-1)

        # Stack input sequence (return, state, action)
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(B, 3 * T, self.hidden_dim)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Temporal Encoding with GRU
        temporal_output = self.temporal_encoder(stacked_inputs)

        # Predict actions
        action_logits = self.action_mapper(temporal_output)

        # Reshape output into two action heads (add and remove)
        add_vals, remove_vals = action_logits.view(B, self.num_locations, 2).chunk(2, dim=-1)

        return add_vals.squeeze(-1), remove_vals.squeeze(-1)