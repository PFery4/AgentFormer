import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from model.attention_mechanisms import AgentAwareAttention, MapAgentAwareAttention

from typing import Optional, Tuple
Tensor = torch.Tensor


LAYER_ACTIVATION_FUNCTIONS = {
    'relu': F.relu,
    'gelu': F.gelu
}

# ENCODER LAYERS ######################################################################################################


class BaseAttentionEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        n_head: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self, d_model: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu'
    ):
        super().__init__()

        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = LAYER_ACTIVATION_FUNCTIONS[activation]


class AgentAwareAttentionEncoderLayer(BaseAttentionEncoderLayer):
    def __init__(
            self, d_model: int, n_head: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu',
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True
    ):
        super().__init__(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.self_attn = AgentAwareAttention(
            traj_dim=d_model, v_dim=d_model, num_heads=n_head, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
        )

    def forward(
            self, src: Tensor, src_self_other_mask: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_self_other_mask: the self/other attention mask that corresponds to the src sequence (required).
            src_mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, _ = self.self_attn(
            q=src, k=src, v=src,
            self_other_mask=src_self_other_mask, mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MapAgentAwareAttentionEncoderLayer(BaseAttentionEncoderLayer):

    def __init__(
            self, d_model: int, n_head: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu',
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True
    ):
        super().__init__(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.self_attn = MapAgentAwareAttention(
            traj_dim=d_model, map_dim=d_model, v_dim=d_model, num_heads=n_head, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
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
            self, src: Tensor, src_self_other_mask: Tensor, map_feature: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        src2, map_feature2, _ = self.self_attn(
            q=src, k=src, v=src,
            q_map=map_feature, k_map=map_feature, v_map=map_feature,
            self_other_mask=src_self_other_mask, mask=src_mask
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


# DECODER LAYERS ######################################################################################################


class BaseAttentionDecoderLayer(Module):
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
            self, d_model: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu'
    ):
        super().__init__()

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


class AgentAwareAttentionDecoderLayer(BaseAttentionDecoderLayer):
    def __init__(
            self, d_model: int, n_head: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu',
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True
    ):
        super().__init__(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )

        self.self_attn = AgentAwareAttention(
            traj_dim=d_model, v_dim=d_model, num_heads=n_head, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
        )
        self.cross_attn = AgentAwareAttention(
            traj_dim=d_model, v_dim=d_model, num_heads=n_head, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
        )

    def forward(
            self, tgt: Tensor, memory: Tensor, tgt_tgt_self_other_mask: Tensor, tgt_mem_self_other_mask: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_tgt_self_other_mask: the self/other attention mask that corresponds to the tgt sequence (required).
            tgt_mem_self_other_mask: the self/other attention mask that corresponds to the memory sequence (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        tgt2, self_attn_weights = self.self_attn(
            q=tgt, k=tgt, v=tgt, self_other_mask=tgt_tgt_self_other_mask, mask=tgt_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.cross_attn(
            q=tgt, k=memory, v=memory, self_other_mask=tgt_mem_self_other_mask, mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, cross_attn_weights


class MapAgentAwareAttentionDecoderLayer(BaseAttentionDecoderLayer):
    def __init__(
            self, d_model: int, n_head: int,
            dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu',
            bias_self: bool = False, bias_other: bool = False, bias_out: bool = True
    ):
        super().__init__(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.self_attn = MapAgentAwareAttention(
            traj_dim=d_model, map_dim=d_model, v_dim=d_model, num_heads=n_head, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
        )
        self.cross_attn = MapAgentAwareAttention(
            traj_dim=d_model, map_dim=d_model, v_dim=d_model, num_heads=n_head, dropout=dropout,
            bias_self=bias_self, bias_other=bias_other, bias_out=bias_out
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
            tgt_tgt_self_other_mask: Tensor, tgt_mem_self_other_mask: Tensor,
            tgt_map: Tensor, mem_map: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tgt2, tgt_map2, self_attn_weights = self.self_attn(
            q=tgt, k=tgt, v=tgt,
            q_map=tgt_map, k_map=tgt_map, v_map=tgt_map,
            self_other_mask=tgt_tgt_self_other_mask, mask=tgt_mask
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt_map = tgt_map + self.map_dropout1(tgt_map2)
        tgt_map = self.map_norm1(tgt_map)

        tgt2, tgt_map2, cross_attn_weights = self.cross_attn(
            q=tgt, k=memory, v=memory,
            q_map=tgt_map, k_map=mem_map, v_map=mem_map,
            self_other_mask=tgt_mem_self_other_mask, mask=memory_mask
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

