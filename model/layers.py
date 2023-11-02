import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from model.attention_mechanisms import AgentAwareAttentionV2, MapAgentAwareAttention

from typing import Optional, Tuple
Tensor = torch.Tensor


LAYER_ACTIVATION_FUNCTIONS = {
    'relu': F.relu,
    'gelu': F.gelu
}

# ENCODER LAYERS ######################################################################################################


class AgentFormerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self, d_model: int, nhead: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__()
        # print(f"ENCODER_LAYER")
        self.self_attn = AgentAwareAttentionV2(
            traj_dim=d_model,
            vdim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = LAYER_ACTIVATION_FUNCTIONS[activation]

    # def __setstate__(self, state):
    #     if 'activation' not in state:
    #         state['activation'] = F.relu
    #     super().__setstate__(state)

    def forward(
            self, src: Tensor, src_identities: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, _ = self.self_attn(
            q=src, k=src, v=src,
            q_identities=src_identities, k_identities=src_identities, mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MapAwareAgentFormerEncoderLayer(AgentFormerEncoderLayer):

    def __init__(
            self, d_model: int, nhead: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.self_attn = MapAgentAwareAttention(
            traj_dim=d_model,
            map_dim=d_model,
            vdim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        self.map_linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.map_dropout = torch.nn.Dropout(dropout)
        self.map_linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.map_norm1 = torch.nn.LayerNorm(d_model)
        self.map_norm2 = torch.nn.LayerNorm(d_model)
        self.map_dropout1 = torch.nn.Dropout(dropout)
        self.map_dropout2 = torch.nn.Dropout(dropout)

        self.map_activation = LAYER_ACTIVATION_FUNCTIONS[activation]

    def forward(
            self, src: Tensor, src_identities: Tensor, map_feature: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        src2, map_feature2, _ = self.self_attn(
            q=src, k=src, v=src,
            q_map=map_feature, k_map=map_feature, v_map=map_feature,
            q_identities=src_identities, k_identities=src_identities, mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        map_feature = map_feature + self.map_dropout1(map_feature2)
        map_feature = self.map_norm1(map_feature)
        map_feature2 = self.map_linear2(self.map_dropout(self.map_activation(self.map_linear1(map_feature))))
        map_feature = map_feature + self.map_dropout2(map_feature2)
        map_feature = self.map_norm2(map_feature)

        return src, map_feature


class WrappedAgentFormerEncoderLayerForMapInput(AgentFormerEncoderLayer):
    def __init__(
            self, d_model: int, nhead: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )

    def forward(
            self, src: Tensor, src_identities: Tensor, map_feature: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, None]:
        src = super().forward(
            src=src, src_identities=src_identities, src_mask=src_mask
        )
        return src, None

# DECODER LAYERS ######################################################################################################


class AgentFormerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
            self, d_model: int, nhead: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__()
        # print(f"DECODER_LAYER")
        self.self_attn = AgentAwareAttentionV2(
            traj_dim=d_model,
            vdim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        self.cross_attn = AgentAwareAttentionV2(
            traj_dim=d_model,
            vdim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = LAYER_ACTIVATION_FUNCTIONS[activation]

    # def __setstate__(self, state):
    #     if 'activation' not in state:
    #         state['activation'] = F.relu
    #     super().__setstate__(state)

    def forward(
            self, tgt: Tensor, memory: Tensor, tgt_identities: Tensor, mem_identities: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        tgt2, self_attn_weights = self.self_attn(
            q=tgt, k=tgt, v=tgt, q_identities=tgt_identities, k_identities=tgt_identities, mask=tgt_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.cross_attn(
            q=tgt, k=memory, v=memory, q_identities=tgt_identities, k_identities=mem_identities, mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, cross_attn_weights


class MapAwareAgentFormerDecoderLayer(AgentFormerDecoderLayer):
    def __init__(
            self, d_model: int, nhead: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu'
    ):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.self_attn = MapAgentAwareAttention(
            traj_dim=d_model,
            map_dim=d_model,
            vdim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        self.cross_attn = MapAgentAwareAttention(
            traj_dim=d_model,
            map_dim=d_model,
            vdim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        self.map_linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.map_dropout = torch.nn.Dropout(dropout)
        self.map_linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.map_norm1 = torch.nn.LayerNorm(d_model)
        self.map_norm2 = torch.nn.LayerNorm(d_model)
        self.map_norm3 = torch.nn.LayerNorm(d_model)
        self.map_dropout1 = torch.nn.Dropout(dropout)
        self.map_dropout2 = torch.nn.Dropout(dropout)
        self.map_dropout3 = torch.nn.Dropout(dropout)

        self.map_activation = LAYER_ACTIVATION_FUNCTIONS[activation]

    def forward(
            self, tgt: Tensor, memory: Tensor,
            tgt_identities: Tensor, mem_identities: Tensor,
            tgt_map: Tensor, mem_map: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tgt2, tgt_map2, self_attn_weights = self.self_attn(
            q=tgt, k=tgt, v=tgt,
            q_map=tgt_map, k_map=tgt_map, v_map=tgt_map,
            q_identities=tgt_identities, k_identities=tgt_identities, mask=tgt_mask
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt_map = tgt_map + self.map_dropout1(tgt_map2)
        tgt_map = self.map_norm1(tgt_map)

        tgt2, tgt_map2, cross_attn_weights = self.cross_attn(
            q=tgt, k=memory, v=memory,
            q_map=tgt_map, k_map=mem_map, v_map=mem_map,
            q_identities=tgt_identities, k_identities=mem_identities, mask=memory_mask
        )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        tgt_map = tgt_map + self.map_dropout2(tgt_map2)
        tgt_map = self.map_norm2(tgt_map)
        tgt_map2 = self.map_linear2(self.map_dropout(self.map_activation(self.map_linear1(tgt_map))))
        tgt_map = tgt_map + self.map_dropout3(tgt_map2)
        tgt_map = self.map_norm3(tgt_map)

        return tgt, tgt_map, self_attn_weights, cross_attn_weights


class WrappedAgentFormerDecoderLayerForMapInput(AgentFormerDecoderLayer):
    def __init__(
            self, d_model: int, nhead: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )

    def forward(
            self, tgt: Tensor, memory: Tensor,
            tgt_identities: Tensor, mem_identities: Tensor,
            tgt_map: Tensor, mem_map: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, None, Tensor, Tensor]:
        tgt, self_attn_weights, cross_attn_weights = super().forward(
            tgt=tgt, memory=memory,
            tgt_identities=tgt_identities, mem_identities=mem_identities,
            tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return tgt, None, self_attn_weights, cross_attn_weights
