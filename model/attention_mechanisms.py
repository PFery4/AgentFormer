import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from typing import Tuple
Tensor = torch.Tensor


class AgentAwareAttentionV2(Module):

    def __init__(self, traj_dim: int, vdim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.traj_dim = traj_dim            # T
        self.vdim = vdim                    # V
        self.num_heads = num_heads          # H

        self.traj_head_dim = traj_dim // num_heads      # t
        assert self.traj_head_dim * self.num_heads == self.traj_dim, "traj_dim must be divisible by num_heads"

        self.v_head_dim = vdim // num_heads             # v
        assert self.v_head_dim * self.num_heads == self.vdim, "vdim must be divisible by num_heads"

        self.traj_scaling = float(self.traj_head_dim ** -0.5)

        # MLP's for mapping trajectory sequences to keys, queries and values
        self.w_q_traj_self = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_q_traj_other = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_k_traj_self = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_k_traj_other = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_v_traj = torch.nn.Linear(self.traj_dim, self.vdim, bias=False)

        # output MLP
        self.fc_traj = torch.nn.Linear(self.vdim, self.vdim)

        # dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        self._reset_parameters()

        # print(f"Hey, here are my params:\n")
        # for k, v in self.__dict__.items():
        #     prnt_str = f"\n{k}: {v}"
        #     try:
        #         prnt_str += f"\t\t(shape: {v.shape})"
        #     except:
        #         prnt_str += ""
        #     print(prnt_str)
        # print(f"\n\n\n\n")

    def _reset_parameters(self):
        # https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head
        torch.nn.init.xavier_normal_(self.w_q_traj_self.weight)
        torch.nn.init.xavier_normal_(self.w_k_traj_self.weight)
        torch.nn.init.xavier_normal_(self.w_q_traj_other.weight)
        torch.nn.init.xavier_normal_(self.w_k_traj_other.weight)
        torch.nn.init.xavier_normal_(self.w_v_traj.weight)
        torch.nn.init.xavier_normal_(self.fc_traj.weight)
        torch.nn.init.zeros_(self.fc_traj.bias)

    @staticmethod
    def agent_aware_mask(q_identities: Tensor, k_identities: Tensor):
        # q_identities: [L]
        # k_identities: [S]
        return torch.stack([k_identities == q_id for q_id in q_identities])     # [L, S]

    def agent_scaled_dot_product(
            self,
            q: Tensor, k: Tensor,
            q_identities: Tensor, k_identities: Tensor,
            mask: Tensor
    ) -> Tensor:
        # q: [B, L, T]
        # k: [B, S, T]
        # v: [B, S, T]
        # q_identities: [B, L]
        # k_identities: [B, S]
        # mask: [B, L, S]

        B, L, _ = q.size()
        _, S, _ = k.size()

        q_self = self.w_q_traj_self(q) * self.traj_scaling          # [B, L, T]
        q_other = self.w_q_traj_other(q) * self.traj_scaling        # [B, L, T]
        k_self = self.w_k_traj_self(k)                              # [B, S, T]
        k_other = self.w_k_traj_other(k)                            # [B, S, T]

        # print(f"2. {q_self.shape, q_other.shape, k_self.shape, k_other.shape=}")

        q_self = q_self.reshape(B, L, self.num_heads, self.traj_head_dim).transpose(1, 2)          # [B, H, L, t]
        q_other = q_other.reshape(B, L, self.num_heads, self.traj_head_dim).transpose(1, 2)        # [B, H, L, t]
        k_self = k_self.reshape(B, S, self.num_heads, self.traj_head_dim).permute(0, 2, 3, 1)      # [B, H, t, S]
        k_other = k_other.reshape(B, S, self.num_heads, self.traj_head_dim).permute(0, 2, 3, 1)    # [B, H, t, S]

        # print(f"3. {q_self.shape, q_other.shape, k_self.shape, k_other.shape=}")

        attention_self = q_self @ k_self            # [B, H, L, t] @ [B, H, t, S] = [B, H, L, S]
        attention_other = q_other @ k_other         # [B, H, L, t] @ [B, H, t, S] = [B, H, L, S]

        # print(f"4. {attention_self.shape, attention_other.shape=}")

        agent_aware_mask = torch.stack(
            [self.agent_aware_mask(q_id, k_id) for q_id, k_id in zip(q_identities, k_identities)]
        ).unsqueeze(1)    # [B, 1, L, S]
        # print(f"5. {agent_aware_mask.shape=}")

        attention = attention_other * (~agent_aware_mask) + attention_self * agent_aware_mask        # [B, H, L, S]
        # print(f"6. {attention.shape=}")

        attention += mask.unsqueeze(1)                                                               # [B, H, L, S]
        # print(f"7. {attention.shape=}")

        return attention

    def forward(
            self,
            q: Tensor, k: Tensor, v: Tensor,
            q_identities: Tensor, k_identities: Tensor,
            mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # q: [B, L, T]
        # k: [B, S, T]
        # v: [B, S, T]
        # q_identities: [B, L]
        # k_identities: [B, S]
        # mask: [B, L]

        # print(f"\n\n{q.shape, k.shape, v.shape=}")
        # print(f"{q_identities.shape, k_identities.shape=}")
        # print(f"{mask, mask.shape=}")

        B, L, _ = q.size()
        _, S, _ = k.size()

        # mapping inputs to keys, queries and values
        v = self.w_v_traj(v)                                                    # [B, S, V]
        v = v.reshape(B, S, self.num_heads, self.v_head_dim).transpose(1, 2)    # [B, H, S, v]

        # print(f"1. {v.shape=}")

        attention = self.agent_scaled_dot_product(
            q=q, k=k, q_identities=q_identities, k_identities=k_identities, mask=mask
        )       # [B, H, L, S]

        # print(f"8. {attention.shape=}")

        attention = F.softmax(attention, dim=-1)        # [B, H, L, S]
        # print(f"9. {attention.shape=}")
        attention = self.dropout(attention)             # [B, H, L, S]
        # print(f"10. {attention.shape=}")

        attention_output = attention @ v                # [B, H, L, S] @ [B, H, S, v] = [B, H, L, v]
        # print(f"11. {attention_output.shape=}")

        attention_output = attention_output.transpose(1, 2).reshape(B, L, self.vdim)        # [B, L, V]
        # print(f"12. {attention_output.shape=}")

        attention_output = self.fc_traj(attention_output)                                       # [B, L, V]
        # print(f"13. {attention_output.shape=}")

        # print(f"14. {attention_output.shape, (attention.sum(dim=1) / self.num_heads).shape=}")
        return attention_output, attention.sum(dim=1) / self.num_heads      # [B, L, V], [B, L, S]


class MapAgentAwareAttention(AgentAwareAttentionV2):

    def __init__(self, traj_dim: int, map_dim: int, vdim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(traj_dim=traj_dim, vdim=vdim, num_heads=num_heads, dropout=dropout)
        self.map_dim = map_dim              # M

        self.map_head_dim = map_dim // num_heads        # m
        assert self.map_head_dim * self.num_heads == self.map_dim, "map_dim must be divisible by num_heads"

        self.map_scaling = float(self.map_head_dim ** -0.5)

        # MLP's for mapping trajectory sequences to keys and queries for attending to the map features
        self.w_q_traj_map = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_k_traj_map = torch.nn.Linear(self.traj_dim, self.map_dim, bias=False)

        # MLP's for mapping map feature data to keys, queries and values
        self.w_q_map_self = torch.nn.Linear(self.map_dim, self.map_dim, bias=False)
        self.w_q_map_agents = torch.nn.Linear(self.map_dim, self.map_dim, bias=False)
        self.w_k_map_self = torch.nn.Linear(self.map_dim, self.map_dim, bias=False)
        self.w_k_map_agents = torch.nn.Linear(self.map_dim, self.traj_dim, bias=False)
        self.w_v_map = torch.nn.Linear(self.map_dim, self.vdim, bias=False)

        # output MLP for the map
        self.fc_map = torch.nn.Linear(self.vdim, self.vdim)

        self._reset_parameters()
        self._reset_map_aware_parameters()

    def _reset_map_aware_parameters(self):
        torch.nn.init.xavier_normal_(self.w_q_traj_map.weight)
        torch.nn.init.xavier_normal_(self.w_k_traj_map.weight)
        torch.nn.init.xavier_normal_(self.w_q_map_self.weight)
        torch.nn.init.xavier_normal_(self.w_k_map_self.weight)
        torch.nn.init.xavier_normal_(self.w_q_map_agents.weight)
        torch.nn.init.xavier_normal_(self.w_k_map_agents.weight)
        torch.nn.init.xavier_normal_(self.w_v_map.weight)
        torch.nn.init.xavier_normal_(self.fc_map.weight)
        torch.nn.init.zeros_(self.fc_map.bias)

    def forward(
            self,
            q: Tensor, k: Tensor, v: Tensor, q_map: Tensor, k_map: Tensor, v_map: Tensor,
            q_identities: Tensor, k_identities: Tensor,
            mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # q: [B, L, T]
        # k: [B, S, T]
        # v: [B, S, T]
        # q_map: [B, M]
        # k_map: [B, M]
        # v_map: [B, M]
        # q_identities: [B, L]
        # k_identities: [B, S]
        # mask: [B, L, S]

        # print(f"IN ATTENTION MECHANISM")
        # print(f"{q.shape, k.shape, v.shape, q_map.shape, k_map.shape, v_map.shape=}")
        # print(f"{q_identities.shape, k_identities.shape=}")
        # print(f"{mask, mask.shape=}")

        B, L, _ = q.size()
        _, S, _ = k.size()

        # trajectory related keys, queries and values
        q_traj_map = self.w_q_traj_map(q) * self.map_scaling        # [B, L, T]
        k_traj_map = self.w_k_traj_map(k)                           # [B, S, M]
        v_traj = self.w_v_traj(v)                                   # [B, S, V]

        # map related keys, queries and values
        q_map_self = self.w_q_map_self(q_map) * self.map_scaling            # [B, M]
        q_map_agents = self.w_q_map_agents(q_map) * self.traj_scaling       # [B, M]
        k_map_self = self.w_k_map_self(k_map)                               # [B, M]
        k_map_agents = self.w_k_map_agents(k_map)                           # [B, T]
        v_map_ = self.w_v_map(v_map)                                        # [B, V]

        # print(f"1. {q_traj_map.shape, k_traj_map.shape, v_traj.shape=}")
        # print(f"{q_map_self.shape, q_map_agents.shape, k_map_self.shape, k_map_agents.shape, v_map_.shape=}")

        # Tensor reshaping
        q_traj_map = q_traj_map.reshape(B, L, self.num_heads, self.map_head_dim).transpose(1, 2)        # [B, H, L, t]
        k_traj_map = k_traj_map.reshape(B, S, self.num_heads, self.map_head_dim).permute(0, 2, 3, 1)    # [B, H, m, S]
        v_traj = v_traj.reshape(B, S, self.num_heads, self.v_head_dim).transpose(1, 2)                  # [B, H, S, v]

        q_map_self = q_map_self.reshape(B, self.num_heads, 1, self.map_head_dim)                        # [B, H, 1, m]
        q_map_agents = q_map_agents.reshape(B, self.num_heads, 1, self.map_head_dim)                    # [B, H, 1, m]
        k_map_self = k_map_self.reshape(B, self.num_heads, self.map_head_dim, 1)                        # [B, H, m, 1]
        k_map_agents = k_map_agents.reshape(B, self.num_heads, self.traj_head_dim, 1)                   # [B, H, t, 1]
        v_map_ = v_map_.reshape(B, self.num_heads, 1, self.v_head_dim)                                  # [B, H, 1, v]

        # print(f"\n{q_traj_map.shape, k_traj_map.shape, v_traj.shape=}")
        # print(f"{q_map_self.shape, q_map_agents.shape, k_map_self.shape, k_map_agents.shape, v_map_.shape=}\n")

        # cross agent attention
        cross_agent_attention = self.agent_scaled_dot_product(
            q=q, k=k, q_identities=q_identities, k_identities=k_identities, mask=mask
        )       # [B, H, L, S]
        # print(f"8. {cross_agent_attention.shape=}")

        # cross map attention
        map_map_attention = q_map_self @ k_map_self         # [B, H, 1, m] @ [B, H, m, 1] = [B, H, 1, 1]
        # print(f"9. {map_map_attention.shape=}")

        # agent map attention, agents query the map
        agent_map_attention = q_traj_map @ k_map_agents     # [B, H, L, t] @ [B, H, t, 1] = [B, H, L, 1]
        # print(f"10. {agent_map_attention.shape=}")

        # map agent attention, the map queries the agents
        map_agent_attention = q_map_agents @ k_traj_map     # [B, H, 1, m] @ [B, H, m, S] = [B, H, 1, S]
        # print(f"11. {map_agent_attention.shape=}")

        # Combine attention scores
        traj_attention = torch.cat([agent_map_attention, cross_agent_attention], dim=-1)    # [B, H, L, S+1]
        map_attention = torch.cat([map_map_attention, map_agent_attention], dim=-1)         # [B, H, 1, S+1]
        combined_attention = torch.cat([map_attention, traj_attention], dim=-2)             # [B, H, L+1, S+1]
        # print(f"12. {traj_attention.shape, map_attention.shape, combined_attention.shape=}")

        # softmax
        combined_attention = F.softmax(combined_attention, dim=-1)
        # print(f"13. {combined_attention.shape=}")

        # dropout       # TODO: CAREFUL: ARE WE IN DANGER IF WE DROPOUT THE MAP ENTIRELY? --> Should we dropout only the trajectory data?
        combined_attention = self.dropout(combined_attention)
        # print(f"14. {combined_attention.shape=}")

        # score multiply values
        combined_v = torch.cat([v_map_, v_traj], dim=-2)             # [B, H, S+1, v]
        # print(f"15. {combined_v.shape=}")

        attention_output = combined_attention @ combined_v          # [B, H, L+1, S+1] @ [B, H, S+1, v] = [B, H, L+1, v]
        # print(f"16. {attention_output.shape=}")

        # return output
        attention_output = attention_output.transpose(1, 2).reshape(B, L+1, self.vdim)      # [B, L+1, V]
        traj_output = attention_output[:, 1:, :]                     # [B, L, V]
        map_output = attention_output[:, 0, :]                       # [B, V]
        # print(f"17. {attention_output.shape, traj_output.shape, map_output.shape=}")

        # separate MLP for map and traj
        traj_output = self.fc_traj(traj_output)                     # [B, L, V]
        map_output = self.fc_map(map_output)                        # [B, V]
        # print(f"18. {traj_output.shape, map_output.shape, (combined_attention.sum(dim=1) / self.num_heads).shape=}")

        # [B, L, V], [B, V], [B, L+1, S+1]
        return traj_output, map_output, combined_attention.sum(dim=1) / self.num_heads


if __name__ == '__main__':
    print("Hello!")
