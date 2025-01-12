import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from typing import Tuple
Tensor = torch.Tensor


class SelfOtherAwareAttention(Module):
    """
    Base class for AgentAwareAttention and MapAgentAwareAttention
    """

    def __init__(
            self, qk_dim: int, v_dim: int, num_heads: int, dropout: float = 0.1,
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True
    ):
        super().__init__()
        self.qk_dim = qk_dim                    # T
        self.v_dim = v_dim                      # V
        self.num_heads = num_heads              # H

        self.bias_self = bias_self
        self.bias_other = bias_other
        self.bias_out = bias_out

        self.qk_head_dim = qk_dim // num_heads          # t
        assert self.qk_head_dim * self.num_heads == self.qk_dim, "traj_dim must be divisible by num_heads"

        self.v_head_dim = v_dim // num_heads            # v
        assert self.v_head_dim * self.num_heads == self.v_dim, "vdim must be divisible by num_heads"

        self.qk_scaling = float(self.qk_head_dim ** -0.5)

        # MLP's for mapping trajectory sequences to keys, queries and values
        self.w_q_self = torch.nn.Linear(self.qk_dim, self.qk_dim, bias=self.bias_self)
        self.w_q_other = torch.nn.Linear(self.qk_dim, self.qk_dim, bias=self.bias_other)
        self.w_k_self = torch.nn.Linear(self.qk_dim, self.qk_dim, bias=self.bias_self)
        self.w_k_other = torch.nn.Linear(self.qk_dim, self.qk_dim, bias=self.bias_other)
        self.w_v = torch.nn.Linear(self.qk_dim, self.v_dim, bias=self.bias_other)

        # output MLP
        self.fc = torch.nn.Linear(self.v_dim, self.v_dim, bias=self.bias_out)

        # dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # we might need to initialize according to the following:
        # https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head
        torch.nn.init.xavier_uniform_(self.w_q_self.weight)
        torch.nn.init.xavier_uniform_(self.w_k_self.weight)
        torch.nn.init.xavier_uniform_(self.w_q_other.weight)
        torch.nn.init.xavier_uniform_(self.w_k_other.weight)
        torch.nn.init.xavier_uniform_(self.w_v.weight)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.bias_self:
            torch.nn.init.zeros_(self.w_q_self.bias)
            torch.nn.init.zeros_(self.w_k_self.bias)
        if self.bias_other:
            torch.nn.init.zeros_(self.w_q_other.bias)
            torch.nn.init.zeros_(self.w_k_other.bias)
        if self.bias_out:
            torch.nn.init.zeros_(self.fc.bias)

    def self_other_scaled_dot_product(
            self,
            q: Tensor,                  # [B, L, T]
            k: Tensor,                  # [B, S, T]
            self_other_mask: Tensor,    # [B, L, S]
            mask: Tensor                # [B, L, S]
    ) -> Tensor:
        B, L, _ = q.size()
        _, S, _ = k.size()

        q_self = self.w_q_self(q) * self.qk_scaling             # [B, L, T]
        q_other = self.w_q_other(q) * self.qk_scaling           # [B, L, T]
        k_self = self.w_k_self(k)                               # [B, S, T]
        k_other = self.w_k_other(k)                             # [B, S, T]

        q_self = q_self.view(B, L, self.num_heads, self.qk_head_dim).transpose(1, 2)          # [B, H, L, t]
        q_other = q_other.view(B, L, self.num_heads, self.qk_head_dim).transpose(1, 2)        # [B, H, L, t]
        k_self = k_self.view(B, S, self.num_heads, self.qk_head_dim).permute(0, 2, 3, 1)      # [B, H, t, S]
        k_other = k_other.view(B, S, self.num_heads, self.qk_head_dim).permute(0, 2, 3, 1)    # [B, H, t, S]

        attention_self = q_self @ k_self            # [B, H, L, t] @ [B, H, t, S] = [B, H, L, S]
        attention_other = q_other @ k_other         # [B, H, L, t] @ [B, H, t, S] = [B, H, L, S]

        self_other_mask = self_other_mask.unsqueeze(1)     # [B, 1, L, S]

        attention = attention_other * (~self_other_mask) + attention_self * self_other_mask        # [B, H, L, S]

        attention += mask.unsqueeze(1)                                                               # [B, H, L, S]

        return attention


class AgentAwareAttention(SelfOtherAwareAttention):
    def __init__(
            self, traj_dim: int, v_dim: int, num_heads: int, dropout: float = 0.1,
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True
    ):
        super().__init__(
            qk_dim=traj_dim, v_dim=v_dim, num_heads=num_heads, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
        )

    def forward(
            self,
            q: Tensor,                  # [B, L, T]
            k: Tensor,                  # [B, S, T]
            v: Tensor,                  # [B, S, T]
            self_other_mask: Tensor,    # [B, L, S]
            mask: Tensor                # [B, L, S]
    ) -> Tuple[Tensor, Tensor]:

        B, L, _ = q.size()
        _, S, _ = k.size()

        # mapping inputs to keys, queries and values
        v = self.w_v(v)                                                         # [B, S, V]
        v = v.view(B, S, self.num_heads, self.v_head_dim).transpose(1, 2)    # [B, H, S, v]

        attention = self.self_other_scaled_dot_product(
            q=q, k=k, self_other_mask=self_other_mask, mask=mask
        )       # [B, H, L, S]

        attention = F.softmax(attention, dim=-1)        # [B, H, L, S]
        attention = self.dropout(attention)             # [B, H, L, S]

        attention_output = attention @ v                # [B, H, L, S] @ [B, H, S, v] = [B, H, L, v]

        attention_output = attention_output.transpose(1, 2).reshape(B, L, self.v_dim)       # [B, L, V]

        attention_output = self.fc(attention_output)                                        # [B, L, V]

        return attention_output, attention.sum(dim=1) / self.num_heads      # [B, L, V], [B, L, S]


class MapAgentAwareAttention(SelfOtherAwareAttention):

    def __init__(
            self, traj_dim: int, map_dim: int, v_dim: int, num_heads: int, dropout: float = 0.1,
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True,
            bias_map: bool = False
    ):
        super().__init__(
            qk_dim=traj_dim, v_dim=v_dim, num_heads=num_heads, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
        )
        self.qk_map_dim = map_dim                   # M

        self.bias_map = bias_map

        self.qk_map_head_dim = map_dim // num_heads        # m
        assert self.qk_map_head_dim * self.num_heads == self.qk_map_dim, "map_dim must be divisible by num_heads"

        self.qk_map_scaling = float(self.qk_map_head_dim ** -0.5)

        # MLP's for map keys to attend to trajectory sequence queries
        self.w_q_traj_map = torch.nn.Linear(self.qk_dim, self.qk_dim, bias=self.bias_map)
        self.w_k_map_agents = torch.nn.Linear(self.qk_map_dim, self.qk_dim, bias=self.bias_map)

        # MLP for map values
        self.w_v_map = torch.nn.Linear(self.qk_map_dim, self.v_dim, bias=self.bias_map)

        # output MLP for the map
        self.fc_map = torch.nn.Linear(self.v_dim, self.v_dim, bias=self.bias_out)

        self._reset_map_aware_parameters()

    def _reset_map_aware_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_q_traj_map.weight)
        torch.nn.init.xavier_uniform_(self.w_k_map_agents.weight)
        torch.nn.init.xavier_uniform_(self.w_v_map.weight)
        # torch.nn.init.xavier_uniform_(self.fc_map.weight)

        if self.bias_map:
            torch.nn.init.zeros_(self.w_q_traj_map.bias)
            torch.nn.init.zeros_(self.w_k_map_agents.bias)
        if self.bias_out:
            torch.nn.init.zeros_(self.fc_map.bias)

    def forward(
            self,
            q: Tensor,                  # [B, L, T]
            k: Tensor,                  # [B, S, T]
            v: Tensor,                  # [B, S, T]
            self_other_mask: Tensor,    # [B, L, S]
            mask: Tensor,               # [B, L, S]
            k_map: Tensor,              # [B, M]
            v_map: Tensor               # [B, M]
    ) -> Tuple[Tensor, Tensor]:         # [B, L, V], [B, L, S+1]
        B, L, _ = q.size()
        _, S, _ = k.size()

        # map and trajectory values
        v_map_ = self.w_v_map(v_map)    # [B, V]
        v_traj = self.w_v(v)            # [B, S, V]
        v_map_ = v_map_.view(B, self.num_heads, 1, self.v_head_dim)                     # [B, H, 1, v]
        v_traj = v_traj.view(B, S, self.num_heads, self.v_head_dim).transpose(1, 2)     # [B, H, S, v]

        # cross agent attention
        cross_agent_attention = self.self_other_scaled_dot_product(
            q=q, k=k, self_other_mask=self_other_mask, mask=mask
        )       # [B, H, L, S]

        # trajectory queries, map keys and values
        q_traj_map = self.w_q_traj_map(q) * self.qk_map_scaling     # [B, L, T]
        k_map_agents = self.w_k_map_agents(k_map)                   # [B, T]
        q_traj_map = q_traj_map.view(B, L, self.num_heads, self.qk_map_head_dim).transpose(1, 2)    # [B, H, L, t]
        k_map_agents = k_map_agents.view(B, self.num_heads, self.qk_head_dim, 1)                    # [B, H, t, 1]

        # agent map attention, agents query the map
        agent_map_attention = q_traj_map @ k_map_agents     # [B, H, L, t] @ [B, H, t, 1] = [B, H, L, 1]

        # Combine attention scores
        attention = torch.cat([agent_map_attention, cross_agent_attention], dim=-1)     # [B, H, L, S+1]

        # softmax
        attention = F.softmax(attention, dim=-1)                                        # [B, H, L, S+1]

        # dropout
        attention = self.dropout(attention)                                             # [B, H, L, S+1]

        # score multiply values
        combined_v = torch.cat([v_map_, v_traj], dim=-2)                                # [B, H, S+1, v]

        attention_output = attention @ combined_v      # [B, H, L, S+1] @ [B, H, S+1, v] = [B, H, L, v]

        # return output
        attention_output = attention_output.transpose(1, 2).reshape(B, L, self.v_dim)       # [B, L, V]

        attention_output = self.fc(attention_output)                                        # [B, L, V]

        return attention_output, attention.sum(dim=1) / self.num_heads
