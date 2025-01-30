import torch
from torch import nn
from abc import ABC, abstractmethod
import torch.nn.functional as F

from src.models.encoder import BaseSpatialEncoder, BaseTemporalEncoder



class BaseAggregator(ABC):
    def __init__(self):
        self.assert_called = False

    @abstractmethod
    def forward(self, x):
        """
        The aggregator expects a torch.Tensor as input with the following shape:
        - (B, N, D) where B is the batch size, N is the number of nodes, and D is the dimension of the node embeddings
        Args:
            x: torch.Tensor: The input tensor of shape (B, N, D)
        Returns:
            torch.Tensor: The output tensor of shape (B, D)
        """
        assert len(x.shape) == 3 or len(x.shape) == 4, "Input tensor must have shape (B, N, D) or (B, T, N, D)"
        self.assert_called = True

class SwapActionMapper(ABC):
    def __init__(self, num_locations: int):
        self.assert_called = False
        self.num_locations = num_locations

    @abstractmethod
    def forward(self, x):
        """
        The action mapper expects a torch.Tensor as input with the following shape:
       - (B,D) where B is the batch size, and D is the dimension of the aggregated temporal node embeddings

        Args:
            x: torch.Tensor: The input tensor of shape (B, D)
        Returns:
            torch.Tensor: The output tensor of shape (B, 2)
        """
        assert len(x.shape) == 2, "Input tensor must have shape (B, D)"
        self.assert_called = True

class CrossProductSwapActionMapper(ABC):
    def __init__(self, num_locations: int):
        self.assert_called = False
        self.num_locations = num_locations

    @abstractmethod
    def forward(self, x):
        """
        The action mapper expects a torch.Tensor as input with the following shape:
       - (B,D) where B is the batch size, and D is the dimension of the aggregated temporal node embeddings

        Args:
            x: torch.Tensor: The input tensor of shape (B, D)
        Returns:
            torch.Tensor: The output tensor of shape (B, 1)
        """
        assert len(x.shape) == 2, "Input tensor must have shape (B, D)"
        self.assert_called = True

class SwapActionNodeMapper(ABC):
    def __init__(self, num_locations: int):
        self.assert_called = False
        self.num_locations = num_locations

    @abstractmethod
    def forward(self, x):
        """
        The action mapper expects a torch.Tensor as input with the following shape:
       - (B,N,D) where B is the batch size, N is the number of nodes, and D is the dimension of the aggregated temporal node embeddings
       Args:
            x: torch.Tensor: The input tensor of shape (B, N, D)
        Returns:
            torch.Tensor: The output tensor of shape (B, N, 2)
        """
        assert len(x.shape) == 3, "Input tensor must have shape (B, N, D)"
        self.assert_called = True


class MeanAggregator(BaseAggregator, nn.Module):
    def __init__(self):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)

    def forward(self, x):
        BaseAggregator.forward(self, x)
        return torch.mean(x, dim=-2)

class MaxAggregator(BaseAggregator, nn.Module):
    def __init__(self):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)

    def forward(self, x):
        BaseAggregator.forward(self, x)
        return torch.max(x, dim=1).values


class SumAggregator(BaseAggregator, nn.Module):
    def __init__(self):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)

    def forward(self, x):
        BaseAggregator.forward(self, x)
        return torch.sum(x, dim=1)


class AdditiveAttention(BaseAggregator, nn.Module):
    """Bahdanau-style additive attention."""
    def __init__(self, input_dim, hidden_dim):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)
        self.W = nn.Linear(input_dim, hidden_dim)  # Transform input
        self.v = nn.Linear(hidden_dim, 1)  # Score projection

    def forward(self, x):
        #B, N, D = x.shape
        energy = torch.tanh(self.W(x))  # (B, N, hidden_dim)
        attn_scores = self.v(energy).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize

        if len(x.shape) == 4:
            aggregated = torch.einsum("abc,abcd->abd", attn_weights, x)
        else:
            aggregated = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return aggregated, attn_weights


class ScaledDotProductAttention(BaseAggregator, nn.Module):
    """Transformer-style scaled dot-product attention."""
    def __init__(self, input_dim):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)
        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k = self.q(x), self.k(x)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)  # (B, N, N)
        attn_weights = F.softmax(attn_scores.mean(dim=-1), dim=1)  # (B, N)
        aggregated = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return aggregated, attn_weights

class MultiHeadAttention(BaseAggregator, nn.Module):
    """Multi-head self-attention with learnable heads."""
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)
        self.q = nn.Linear(input_dim, hidden_dim * num_heads)
        self.k = nn.Linear(input_dim, hidden_dim * num_heads)
        self.v = nn.Linear(input_dim, hidden_dim * num_heads)
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = q.view(B, N, self.num_heads, -1), k.view(B, N, self.num_heads, -1), v.view(B, N, self.num_heads, -1)

        attn_scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / (D ** 0.5)  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores.mean(dim=1), dim=1)  # Aggregate heads, (B, N)
        attn_output = torch.bmm(attn_weights.unsqueeze(1), v.mean(dim=2)).squeeze(1)  # (B, D)

        return self.out_proj(attn_output), attn_weights


class SoftmaxWeightedAttention(BaseAggregator, nn.Module):
    """Uses a learnable vector to compute importance."""
    def __init__(self, input_dim):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)
        self.attn_vector = nn.Parameter(torch.randn(1, input_dim))

    def forward(self, x):
        attn_scores = torch.matmul(x, self.attn_vector.T).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N)
        aggregated = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return aggregated, attn_weights


class GatedAttention(BaseAggregator, nn.Module):
    """Learnable gating mechanism for attention."""
    def __init__(self, input_dim):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)
        self.gate = nn.Linear(input_dim, 1)

    def forward(self, x):
        gate_values = torch.sigmoid(self.gate(x)).squeeze(-1)  # (B, N)
        attn_weights = gate_values / gate_values.sum(dim=1, keepdim=True)  # Normalize
        aggregated = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return aggregated, attn_weights


class AttentionAggregator(nn.Module):
    """Wrapper to select different attention mechanisms."""
    def __init__(self, input_dim, hidden_dim, type="additive", num_heads=4):
        super().__init__()
        if type == "additive":
            self.attn = AdditiveAttention(input_dim, hidden_dim)
        elif type == "scaled_dot":
            self.attn = ScaledDotProductAttention(input_dim)
        elif type == "multihead":
            self.attn = MultiHeadAttention(input_dim, hidden_dim, num_heads)
        elif type == "softmax_weighted":
            self.attn = SoftmaxWeightedAttention(input_dim)
        elif type == "gated":
            self.attn = GatedAttention(input_dim)
        else:
            raise ValueError(f"Unknown attention type: {type}")

    def forward(self, x):
        return self.attn(x)



class IndependentActionMapper(SwapActionMapper, nn.Module):
    def __init__(self,  num_layer: int, hidden_dim: int, num_locations: int, act_fn: str = "gelu"):
        SwapActionMapper.__init__(self)
        nn.Module.__init__(self)

        layers_add = []
        layers_remove = []

        match act_fn:
            case "tanh":
                self.act_fn = nn.Tanh()
            case "relu":
                self.act_fn = nn.ReLU()
            case "gelu":
                self.act_fn = nn.GELU()
            case "leaky_relu":
                self.act_fn = nn.LeakyReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        layers_add.append(nn.Linear(hidden_dim, hidden_dim if num_layer > 1 else num_locations))
        layers_remove.append(nn.Linear(hidden_dim, hidden_dim if num_layer > 1 else num_locations))

        for i in range(num_layer - 1):
            layers_add.append(self.act_fn)
            layers_remove.append(self.act_fn)
            layers_add.append(nn.Linear(hidden_dim, hidden_dim))
            layers_remove.append(nn.Linear(hidden_dim, hidden_dim))

        if num_layer > 1:
            layers_add.append(self.act_fn)
            layers_remove.append(self.act_fn)
            layers_remove.append(nn.Linear(hidden_dim, num_locations))
            layers_add.append(nn.Linear(hidden_dim, num_locations))

        self.add = nn.Sequential(*layers_add)
        self.remove = nn.Sequential(*layers_remove)

    def forward(self, x):
        SwapActionMapper.forward(self, x)
        add_scores = self.add(x).squeeze(-1)
        remove_scores = self.remove(x).squeeze(-1)
        return add_scores, remove_scores

class SharedActionMapper(SwapActionMapper, nn.Module):
    def __init__(self, num_layer: int, hidden_dim: int, num_locations: int, act_fn: str = "gelu"):
        SwapActionMapper.__init__(self, num_locations)
        nn.Module.__init__(self)

        layers = []
        match act_fn:
            case "tanh":
                self.act_fn = nn.Tanh()
            case "relu":
                self.act_fn = nn.ReLU()
            case "gelu":
                self.act_fn = nn.GELU()
            case "leaky_relu":
                self.act_fn = nn.LeakyReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        layers.append(nn.Linear(hidden_dim, hidden_dim if num_layer > 1 else num_locations))
        for i in range(num_layer - 1):
            layers.append(self._get_activation_fn(act_fn))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        if num_layer > 1:
            layers.append(self._get_activation_fn(act_fn))
            layers.append(nn.Linear(hidden_dim, num_locations))

        self.shared = nn.Sequential(*layers)

    def forward(self, x):
        SwapActionMapper.forward(self, x)
        shared_output = self.shared(x)
        add_scores, remove_scores = torch.chunk(self.decision(shared_output), 2, dim=-1)
        return add_scores, remove_scores

    def _get_activation_fn(self, act_fn: str):
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

class InteractiveActionMapper(SwapActionMapper, nn.Module):
    def __init__(self, num_layer: int, hidden_dim: int, num_locations: int, act_fn: str = "gelu",
                 use_attn: bool = False, max_timesteps: int = None):
        SwapActionMapper.__init__(self, num_locations)
        nn.Module.__init__(self)
        #TODO: How do we model a no-op: add and remove the same node? But with masking we can't do that in the current setting
        layers = []

        for i in range(num_layer):
            layers.append(nn.Linear(hidden_dim * 2 if max_timesteps is not None and i == 0 else 1, hidden_dim))
            layers.append(self._get_act_fn(act_fn))

        self.timestep_embedding = nn.Embedding(max_timesteps, hidden_dim) if max_timesteps is not None else None

        self.shared = nn.Sequential(*layers)
        self.remove = nn.Linear(hidden_dim, num_locations)
        self.action_embedding = nn.Embedding(num_locations, hidden_dim)
        self.add = nn.Linear(hidden_dim * 2, num_locations)

        self.use_attn = use_attn
        self.attention_weight_layer = nn.Linear(hidden_dim, hidden_dim)

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

    def forward(self, x, timesteps: torch.Tensor = None, add_mask=None):
        SwapActionMapper.forward(self, x)

        if timesteps is not None:
            timestep_embedding = self.timestep_embedding(timesteps)
            x = torch.cat([x, timestep_embedding], dim=-1)

        shared_output = self.shared(x)
        remove_scores = self.remove(shared_output)

        remove_action = torch.argmax(remove_scores, dim=-1)

        # Use learned embedding instead of one-hot encoding
        remove_action_embedding = self.action_embedding(remove_action)

        if self.use_attn:
            attention_weights = torch.sigmoid(self.attention_weight_layer(shared_output)) #TODO: maybe this should be based on the remove action as well
            scaled_action_embedding = remove_action_embedding * attention_weights
            add_scores = self.add(torch.cat([shared_output, scaled_action_embedding], dim=-1))

        else:
            add_scores = self.add(torch.cat([shared_output, remove_action_embedding], dim=-1))

        if add_mask is not None:
            add_scores = add_scores + add_mask

            mask = add_mask != -float('inf')
            remove_mask = torch.full_like(add_mask, -float('inf'))
            remove_mask[~mask] = 0

            remove_scores = remove_scores + remove_mask

        return add_scores, remove_scores


class SwapBasedAttentionActionMapper(SwapActionNodeMapper, nn.Module):
    def __init__(self, num_hidden_layer: int, node_dim: int, num_locations: int, act_fn: str = "gelu"):
        """
        Based on https://arxiv.org/pdf/2312.15658
        Args:
            num_hidden_layer:
            node_dim:
            num_locations:
            act_fn:
        """
        SwapActionNodeMapper.__init__(self, num_locations)
        nn.Module.__init__(self)

        self.add_mlp = nn.Linear(node_dim, node_dim)
        self.remove_mlp = nn.Linear(node_dim, node_dim)

        if act_fn == "tanh":
            self.act_fn = nn.Tanh()
        elif act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        elif act_fn == "leaky_relu":
            self.act_fn = nn.LeakyReLU()

        linear_layer = []
        for i in range(num_hidden_layer):
            linear_layer.append(nn.Linear(node_dim, node_dim))
            linear_layer.append(self.act_fn)
        self.project = nn.Sequential(*linear_layer)


    def forward(self, x):
        SwapActionNodeMapper.forward(self, x)
        g_l2 = self.project(x)
        remove_scores = self.remove_mlp(g_l2)
        f = x.T @ F.tanh(self.remove_mlp(x))
        add_scores = f

        return add_scores, remove_scores


class StandardCrossProductActionMapper(CrossProductSwapActionMapper, nn.Module):
    def __init__(self, num_layer: int, hidden_dim: int, num_locations: int, act_fn: str = "gelu",
                 use_attn: bool = False, max_timestep: int = None):
        CrossProductSwapActionMapper.__init__(self, num_locations)
        nn.Module.__init__(self)

        self.use_attn = use_attn

        layers = []
        for i in range(num_layer):
            layers.append(nn.Linear(hidden_dim * 2 if max_timestep is not None and i == 0 else hidden_dim * 1 , hidden_dim))
            layers.append(self._get_act_fn(act_fn))

        if max_timestep is not None:
            self.timestep_context_embedding = nn.Embedding(max_timestep, hidden_dim)

        self.shared = nn.Sequential(*layers)
        self.add_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self._get_act_fn(act_fn))
        self.remove_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self._get_act_fn(act_fn))
        self.cross_product_action_mapping = nn.Linear(hidden_dim * 2, num_locations ** 2)

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

    def forward(self, x, timestep = None):
        CrossProductSwapActionMapper.forward(self, x)
        if timestep is not None:
            timestep_embedding = self.timestep_context_embedding(timestep)
            x = torch.cat([x, timestep_embedding], dim=-1)
        shared_output = self.shared(x)
        add_embedding = self.add_head(shared_output)
        remove_embedding = self.remove_head(shared_output)

        actions = self.cross_product_action_mapping(torch.cat([add_embedding, remove_embedding], dim=-1))

        return actions





