import torch
from torch import nn
from abc import ABC, abstractmethod

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
        assert len(x.shape) == 3, "Input tensor must have shape (B, N, D)"
        self.assert_called = True



class MeanAggregator(BaseAggregator, nn.Module):
    def __init__(self):
        BaseAggregator.__init__(self)
        nn.Module.__init__(self)

    def forward(self, x):
        BaseAggregator.forward(self, x)
        return torch.mean(x, dim=1)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.assert_called = False

    def forward(self, x):
        assert len(x.shape) == 3, f"Expected input of shape (B, N, D), got {x.shape}"
        self.assert_called = True


class AttentionAggregator(BaseAggregator):
    def __init__(self, input_dim: int, hidden_dim: int, type: str = "scaled_dot", num_heads: int = 4):
        super().__init__()
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
        super().forward(x)  # Ensure base class validation
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
            # Split into multiple heads
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
