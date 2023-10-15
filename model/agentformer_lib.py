"""
Modified version of PyTorch Transformer module for the implementation of Agent-Aware Attention (L290-L308)
"""


import warnings
import copy

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.functional import *
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import _LinearWithBias
import torch.nn.init
from torch.nn.parameter import Parameter

from model.layers import \
    WrappedAgentFormerEncoderLayerForMapInput,\
    WrappedAgentFormerDecoderLayerForMapInput,\
    MapAwareAgentFormerEncoderLayer,\
    MapAwareAgentFormerDecoderLayer

from typing import Optional, Tuple, Dict
Tensor = torch.Tensor       # type alias for torch tensor


def agent_aware_mask(q_identities: Tensor, k_identities: Tensor) -> Tensor:
    return torch.cat([(k_identities == q_val).unsqueeze(0) for q_val in q_identities], dim=0).to(q_identities.dtype)


def visualize_mask(q_identities: Tensor, k_identities: Tensor) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()

    uniques = torch.unique(torch.cat((q_identities, k_identities), dim=0)).cpu().numpy()

    ax_k = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_q = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    agent_mask = agent_aware_mask(q_identities, k_identities)
    print(f"{q_identities=}")
    print(f"{k_identities=}")
    print(f"{agent_mask=}")
    img = agent_mask.cpu().numpy()
    print(f"{img=}")
    ax.imshow(img)

    k_band = torch.cat([k_identities.unsqueeze(0)]*10, dim=0).cpu().numpy()
    k_band = k_band * (255. - 0.) / (np.max(uniques) - np.min(uniques))
    ax_k.imshow(k_band)
    q_band = torch.cat([q_identities.unsqueeze(1)]*10, dim=1).cpu().numpy()
    q_band = q_band * (255. - 0.) / (np.max(uniques) - np.min(uniques))
    ax_q.imshow(q_band)
    plt.show()


def agent_aware_attention(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          embed_dim_to_check: int,
                          num_heads: int,
                          in_proj_weight: Tensor,
                          in_proj_bias: Tensor,
                          bias_k: Optional[Tensor],
                          bias_v: Optional[Tensor],
                          add_zero_attn: bool,
                          dropout_p: float,
                          out_proj_weight: Tensor,
                          out_proj_bias: Tensor,
                          q_identities: Tensor,
                          k_identities: Tensor,
                          training: bool = True,
                          key_padding_mask: Optional[Tensor] = None,
                          need_weights: bool = True,
                          attn_mask: Optional[Tensor] = None,
                          use_separate_proj_weight: bool = False,
                          q_proj_weight: Optional[Tensor] = None,
                          k_proj_weight: Optional[Tensor] = None,
                          v_proj_weight: Optional[Tensor] = None,
                          static_k: Optional[Tensor] = None,
                          static_v: Optional[Tensor] = None,
                          gaussian_kernel: bool = True,
                          in_proj_weight_self: Optional[Tensor] = None,
                          in_proj_bias_self: Optional[Tensor] = None
                          ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # print(f"{query.shape=}")
    # print(f"{key.shape=}")
    # print(f"{value.shape=}")
    # print(f"{q_identities.shape, k_identities.shape=}")
    # print(f"{in_proj_bias[0]=}")
    tgt_len, batch_size, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
    scaling = float(head_dim) ** -0.5

    # print(f"{head_dim, embed_dim, num_heads=}")

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention. each tensor is of shape [tgt_len, embed_dim]
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
            # WIP CODE
            # print(f"{query.shape, key.shape, value.shape, in_proj_weight.shape, in_proj_bias.shape=}")
            # print(f"{q.shape, k.shape, v.shape=}")
            # WIP CODE
            if in_proj_weight_self is not None:
                q_self, k_self = linear(query, in_proj_weight_self, in_proj_bias_self).chunk(2, dim=-1)
            # print(f"1: {q.shape, k.shape, v.shape, q_self.shape, k_self.shape=}")

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)
            if in_proj_weight_self is not None:
                _w = in_proj_weight_self[:embed_dim, :]
                _b = in_proj_bias_self[:embed_dim]
                q_self = linear(query, _w, _b)
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _w = in_proj_weight_self[embed_dim:, :]
                _b = in_proj_bias_self[embed_dim:]
                k_self = linear(key, _w, _b)
            # print(f"2: {q.shape, k.shape, v.shape, q_self.shape, k_self.shape=}")
        else:
            raise NotImplementedError
    # k, v and k_self are of shape [src_len, 1, embed_dim]
    # q and q_self are of shape [tgt_len, 1, embed_dim]

    else:
        raise NotImplementedError

    # print(f"{gaussian_kernel=}")
    if not gaussian_kernel:
        q = q * scaling       # remove scaling
        if in_proj_weight_self is not None:
            q_self = q_self * scaling       # remove scaling

    # print(f"3. {attn_mask, attn_mask.shape=}")
    # attn_mask is shape: [src_len, src_len]
    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
               attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now. [1, src_len, src_len]
    # print(f"4. {attn_mask, attn_mask.shape=}")

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        raise NotImplementedError
    else:
        assert bias_k is None
        assert bias_v is None

    # print(f"{q[..., 0], q.shape=}")

    q = q.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    if in_proj_weight_self is not None:
        q_self = q_self.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
        k_self = k_self.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    # k, v and k_self are of shape [batch_size * num_heads, src_len, head_dim]
    # q, q_self are of shape [batch_size * num_heads, tgt_len, head_dim]
    # print(f"5. {q.shape=}")

    # print(f"{static_k=}")
    if static_k is not None:
        raise NotImplementedError

    # print(f"{static_v=}")
    if static_v is not None:
        raise NotImplementedError

    src_len = k.size(1)

    # print(f"{key_padding_mask=}")
    if key_padding_mask is not None:
        raise NotImplementedError

    # print(f"{add_zero_attn=}")
    if add_zero_attn:
        raise NotImplementedError

    # print(f"{gaussian_kernel=}")
    if gaussian_kernel:
        raise NotImplementedError
    else:
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))       # [batch_size * num_heads, tgt_len, src_len]
    # print(f"{attn_output_weights.shape=}")

    assert list(attn_output_weights.size()) == [batch_size * num_heads, tgt_len, src_len]

    if in_proj_weight_self is not None:
        """
        ==================================
            Agent-Aware Attention
        ==================================
        """
        attn_output_weights_inter = attn_output_weights             # [batch_size * num_heads, tgt_len, src_len]
        print(f"7. {q_identities.shape, k_identities.shape=}")
        attn_weight_self_mask = agent_aware_mask(q_identities, k_identities)        # [tgt_len, src_len]

        # print(f"{attn_weight_self_mask.shape=}")

        attn_output_weights_self = torch.bmm(q_self, k_self.transpose(1, 2))    # [batch_size * num_heads, tgt_len, src_len]

        assert attn_weight_self_mask.shape == attn_output_weights.shape[-2:] == attn_output_weights_self.shape[-2:]

        attn_output_weights = attn_output_weights_inter * (1 - attn_weight_self_mask) + attn_output_weights_self * attn_weight_self_mask

        # print(f"{attn_output_weights.shape=}")

        # print(f"{attn_mask=}")
        if attn_mask is not None:
            assert attn_mask.shape[-2:] == attn_output_weights.shape[-2:]
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        # NO div by sqrt(d)         ????
        attn_output_weights = softmax(attn_output_weights, dim=-1)      # [batch_size * num_heads, tgt_len, src_len]

    else:
        raise NotImplementedError

    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)         # [batch_size * num_heads, tgt_len, head_dim]

    assert list(attn_output.size()) == [batch_size * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)     # [tgt_len, batch_size, embed_dim]

    # print(f"8. {out_proj_weight.shape, out_proj_bias.shape=}")
    # print(f"{out_proj_bias[0]=}")
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)                               # [tgt_len, batch_size, embed_dim]
    # print(f"{attn_output.shape, attn_output_weights.shape=}")

    # print(f"{need_weights=}")
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
        # print(f"Need_Weights: {attn_output_weights.shape, (attn_output_weights.sum(dim=1) / num_heads).shape=}")
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class AgentAwareAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, cfg, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__()
        self.cfg = cfg
        self.gaussian_kernel = self.cfg.get('gaussian_kernel', False)
        self.sep_attn = self.cfg.get('sep_attn', True)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        if self.sep_attn:
            self.in_proj_weight_self = Parameter(torch.empty(2 * embed_dim, embed_dim))
            self.in_proj_bias_self = Parameter(torch.empty(2 * embed_dim))
        else:
            self.in_proj_weight_self = self.in_proj_bias_self = None

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
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.q_proj_weight)
            torch.nn.init.xavier_uniform_(self.k_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

        if self.sep_attn:
            torch.nn.init.xavier_uniform_(self.in_proj_weight_self)
            torch.nn.init.constant_(self.in_proj_bias_self, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(self, query, key, value, query_identities, key_identities, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return agent_aware_attention(
                query=query, key=key, value=value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias,
                bias_k=self.bias_k, bias_v=self.bias_v, add_zero_attn=self.add_zero_attn,
                dropout_p=self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
                q_identities=query_identities, k_identities=key_identities,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, gaussian_kernel=self.gaussian_kernel,
                in_proj_weight_self=self.in_proj_weight_self,
                in_proj_bias_self=self.in_proj_bias_self
                )
        else:
            return agent_aware_attention(
                query=query, key=key, value=value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias,
                bias_k=self.bias_k, bias_v=self.bias_v, add_zero_attn=self.add_zero_attn,
                dropout_p=self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
                q_identities=query_identities, k_identities=key_identities,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, gaussian_kernel=self.gaussian_kernel,
                in_proj_weight_self=self.in_proj_weight_self,
                in_proj_bias_self=self.in_proj_bias_self
                )


# class AgentFormerEncoder(Module):
#     r"""TransformerEncoder is a stack of N encoder layers
#
#     Args:
#         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#         num_layers: the number of sub-encoder-layers in the encoder (required).
#         norm: the layer normalization component (optional).
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = transformer_encoder(src)
#     """
#
#     layer_types = {
#         'agent': WrappedAgentFormerEncoderLayerForMapInput,
#         'map_agent': MapAwareAgentFormerEncoderLayer
#     }
#
#     def __init__(self, encoder_config, num_layers):
#         super().__init__()
#         self.encoder_config = encoder_config
#         self.num_layers = num_layers
#         layer = self.layer_types[encoder_config['layer_type']](**encoder_config['layer_params'])
#         self.layers = _get_clones(layer, num_layers)
#
#     def forward(
#             self, src: Tensor, src_identities: Tensor, map_feature: Tensor,
#             src_mask: Optional[Tensor] = None,
#     ) -> Tuple[Tensor, Tensor]:
#         r"""Pass the input through the encoder layers in turn.
#
#         Args:
#             src: the sequence to the encoder (required).
#             src_mask: the mask for the src sequence (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#         output = src
#         map_output = map_feature
#
#         for layer in self.layers:
#             output, map_output = layer(
#                 src=output, src_identities=src_identities, map_feature=map_output, src_mask=src_mask,
#             )
#
#         return output, map_output
#
#
# class AgentFormerDecoder(Module):
#     r"""TransformerDecoder is a stack of N decoder layers
#
#     Args:
#         decoder_layer: an instance of the TransformerDecoderLayer() class (required).
#         num_layers: the number of sub-decoder-layers in the decoder (required).
#         norm: the layer normalization component (optional).
#
#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = transformer_decoder(tgt, memory)
#     """
#     layer_types = {
#         'agent': WrappedAgentFormerDecoderLayerForMapInput,
#         'map_agent': MapAwareAgentFormerDecoderLayer
#     }
#
#     def __init__(self, decoder_config, num_layers):
#         super().__init__()
#         self.decoder_config = decoder_config
#         self.num_layers = num_layers
#         layer = self.layer_types[decoder_config['layer_type']](**decoder_config['layer_params'])
#         self.layers = _get_clones(layer, num_layers)
#
#     def forward(
#             self, tgt: Tensor, memory: Tensor,
#             tgt_identities: Tensor, mem_identities: Tensor,
#             tgt_map: Tensor, mem_map: Tensor,
#             tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
#     ) -> Tuple[Tensor, Tensor, Dict]:
#         r"""Pass the inputs (and mask) through the decoder layer in turn.
#
#         Args:
#             tgt: the sequence to the decoder (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#         output = tgt
#         map_output = tgt_map
#
#         self_attn_weights = [None] * len(self.layers)
#         cross_attn_weights = [None] * len(self.layers)
#         for i, mod in enumerate(self.layers):
#             output, map_output, self_attn_weights[i], cross_attn_weights[i] = mod(
#                 tgt=output, memory=memory,
#                 tgt_identities=tgt_identities, mem_identities=mem_identities,
#                 tgt_map=map_output, mem_map=mem_map,
#                 tgt_mask=tgt_mask, memory_mask=memory_mask,
#             )
#
#         # if need_weights:
#         #     self_attn_weights = torch.stack(self_attn_weights).cpu().numpy()
#         #     cross_attn_weights = torch.stack(cross_attn_weights).cpu().numpy()
#
#         return output, map_output, {'self_attn_weights': self_attn_weights, 'cross_attn_weights': cross_attn_weights}


# def _get_clones(module, N):
#     return ModuleList([copy.deepcopy(module) for i in range(N)])


# def _get_activation_fn(activation):
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu
#
#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


if __name__ == '__main__':

    q_identities = torch.tensor([[3, 3, 3, 4, 4, 3, 4, 3, 4, 3], [5, 5, 4, 5, 4, 5, 4, 5, 5, 5]]).T
    k_identities = torch.tensor([[3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3], [4, 5, 4, 5, 4, 5, 4, 4, 6, 4, 5]]).T
    print(f"{k_identities.T, k_identities.shape=}")
    print(f"{q_identities, q_identities.shape=}")

    mask = torch.empty([*reversed(q_identities.shape), k_identities.shape[0]]).to(q_identities.dtype)
    # print(f"{mask, mask.shape=}")

    # for k_row, q_row in zip(k_identities.T, q_identities.T):
    #     print(f"{k_row=}")
    #     print(f"{q_row=}")
    # [mask[:, idx, :] = (k_identities == q_id).T for idx, q_id in enumerate(q_identities)]

    for idx, q_id in enumerate(q_identities):
        mask[:, idx, :] = (k_identities == q_id).T

    print(f"{mask, mask.shape=}")

    # for row_idx in range(mask.shape[1]):
    #
    #     print(mask[:, row_idx, :].shape)
    #     print(k_identities[row_idx, :])
    #     # mask[:, row_idx, :]
    #

    test_torch = torch.randn([8, 2, 12, 10])
    print(f"{test_torch.shape=}")
    print(f"{(test_torch.sum(dim=0) / 8).shape=}")

