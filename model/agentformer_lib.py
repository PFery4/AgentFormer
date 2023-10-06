"""
Modified version of PyTorch Transformer module for the implementation of Agent-Aware Attention (L290-L308)
"""


import warnings
import math
import copy

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.functional import *
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear, _LinearWithBias
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.overrides import has_torch_function, handle_torch_function

from typing import Optional, Tuple
Tensor = torch.Tensor       # type alias for torch tensor class


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
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

        if self.sep_attn:
            xavier_uniform_(self.in_proj_weight_self)
            constant_(self.in_proj_bias_self, 0.)

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


class AgentAwareAttentionV2(Module):

    def __init__(self, cfg, traj_dim: int, vdim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.traj_dim = traj_dim            # T
        self.vdim = vdim                    # V
        self.num_heads = num_heads          # H

        self.dropout = torch.nn.Dropout(dropout)

        self.traj_head_dim = traj_dim // num_heads      # t
        assert self.traj_head_dim * self.num_heads == self.traj_dim, "traj_dim must be divisible by num_heads"

        self.v_head_dim = vdim // num_heads             # v
        assert self.v_head_dim * self.num_heads == self.vdim, "vdim must be divisible by num_heads"

        self.traj_scaling = float(self.traj_head_dim) ** -0.5

        # inprojweight and bias map from embed_dim to embed_dim
        self.w_q_traj_self = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_q_traj_other = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_k_traj_self = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_k_traj_other = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_v_traj = torch.nn.Linear(self.traj_dim, self.vdim, bias=False)

        self.fc = torch.nn.Linear(self.vdim, self.vdim, bias=False)

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
        xavier_uniform_(self.w_q_traj_self.weight)
        xavier_uniform_(self.w_k_traj_self.weight)
        xavier_uniform_(self.w_q_traj_other.weight)
        xavier_uniform_(self.w_k_traj_other.weight)
        xavier_uniform_(self.w_v_traj.weight)
        constant_(self.w_q_traj_self.bias, 0.)
        constant_(self.w_k_traj_self.bias, 0.)
        constant_(self.w_q_traj_other.bias, 0.)
        constant_(self.w_k_traj_other.bias, 0.)
        constant_(self.w_v_traj.bias, 0.)
        constant_(self.fc.bias, 0.)

    @staticmethod
    def agent_aware_mask(q_identities: Tensor, k_identities: Tensor):
        # q_identities: [L]
        # k_identities: [S]
        return torch.cat([(k_identities == q_val).unsqueeze(0) for q_val in q_identities], dim=0).to(q_identities.dtype)

    def agent_scaled_dot_product(self, q: Tensor, k: Tensor, q_identities: Tensor, k_identities: Tensor, mask: Tensor):
        # q: [L, N, T]
        # k: [S, N, T]
        # v: [S, N, T]
        # q_identities: [L]
        # k_identities: [S]
        L, N, _ = q.size()
        S, _, _ = k.size()

        # NOTE: No residual connections used in AgentAwareAttention
        q_self = self.w_q_traj_self(q) * self.traj_scaling          # [L, N, T]
        q_other = self.w_q_traj_other(q) * self.traj_scaling        # [L, N, T]
        k_self = self.w_k_traj_self(k)          # [S, N, T]
        k_other = self.w_k_traj_other(k)        # [S, N, T]

        # print(f"2. {q_self.shape, q_other.shape, k_self.shape, k_other.shape=}")

        q_self = q_self.view(L, N, self.num_heads, self.traj_head_dim).transpose(0, 2)          # [H, N, L, t]
        q_other = q_other.view(L, N, self.num_heads, self.traj_head_dim).transpose(0, 2)        # [H, N, L, t]
        k_self = k_self.view(S, N, self.num_heads, self.traj_head_dim).transpose(0, 2)          # [H, N, S, t]
        k_other = k_other.view(S, N, self.num_heads, self.traj_head_dim).transpose(0, 2)        # [H, N, S, t]

        # print(f"3. {q_self.shape, q_other.shape, k_self.shape, k_other.shape=}")

        attention_self = q_self @ k_self.transpose(2, 3)            # [H, N, L, S]
        attention_other = q_other @ k_other.transpose(2, 3)         # [H, N, L, S]

        # print(f"4. {attention_self.shape, attention_other.shape=}")

        agent_aware_mask = self.agent_aware_mask(q_identities, k_identities)    # [L, S]
        # print(f"5. {agent_aware_mask.shape=}")

        attention = attention_other * (1 - agent_aware_mask) + attention_self * agent_aware_mask        # [H, N, L, S]

        # print(f"6. {attention.shape=}")
        attention += mask                                                                               # [H, N, L, S]
        # print(f"7. {attention.shape=}")

        return attention

    def forward(
            self,
            q: Tensor, k: Tensor, v: Tensor,
            q_identities: Tensor, k_identities: Tensor,
            mask: Tensor, need_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # q: [L, N, T]
        # k: [S, N, T]
        # v: [S, N, T]
        # q_identities: [L]
        # k_identities: [S]

        # print(f"{q.shape, k.shape, v.shape=}")
        # print(f"{q_identities.shape, k_identities.shape=}")
        # print(f"{mask, mask.shape=}")

        L, N, _ = q.size()
        S, _, _ = k.size()

        # NOTE: No residual connections used in AgentAwareAttention

        # mapping inputs to keys, queries and values
        v = self.w_v_traj(v)                                 # [S, N, V]
        v = v.view(S, N, self.num_heads, self.v_head_dim).transpose(0, 2)                       # [H, N, S, v]

        # print(f"1. {v.shape=}")

        attention = self.agent_scaled_dot_product(
            q=q, k=k, q_identities=q_identities, k_identities=k_identities, mask=mask
        )       # [N, H, L, S]

        # print(f"8. {attention.shape=}")

        attention = F.softmax(attention, dim=-1)                                                        # [H, N, L, S]
        # print(f"9. {attention.shape=}")
        attention = self.dropout(attention)                                                             # [H, N, L, S]
        # print(f"10. {attention.shape=}")

        attention_output = attention @ v                                                                # [H, N, L, v]
        # print(f"11. {attention_output.shape=}")

        attention_output = attention_output.permute(2, 0, 1, 3).contiguous().view(L, N, self.vdim)      # [L, N, V]
        # print(f"12. {attention_output.shape=}")

        attention_output = self.fc(attention_output)                                                    # [L, N, V]
        # print(f"13. {attention_output.shape=}")

        if need_weights:
            return attention_output, attention.sum(dim=1) / self.num_heads
        else:
            return attention_output, None


class MapAgentAwareAttention(AgentAwareAttentionV2):

    def __init__(self, cfg, traj_dim: int, map_dim: int, vdim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(cfg=cfg, traj_dim=traj_dim, vdim=vdim, num_heads=num_heads, dropout=dropout)
        self.map_dim = map_dim              # M

        self.map_head_dim = map_dim // num_heads        # m
        assert self.map_head_dim * self.num_heads == self.map_input_dim, "map_dim must be divisible by num_heads"

        self.map_scaling = float(self.map_head_dim) ** -0.5

        self.w_q_traj_map = torch.nn.Linear(self.traj_dim, self.traj_dim, bias=False)
        self.w_k_traj_map = torch.nn.Linear(self.traj_dim, self.map_dim, bias=False)

        self.w_q_map_self = torch.nn.Linear(self.map_dim, self.map_dim, bias=False)
        self.w_q_map_agents = torch.nn.Linear(self.map_dim, self.map_dim, bias=False)
        self.w_k_map_self = torch.nn.Linear(self.map_dim, self.map_dim, bias=False)
        self.w_k_map_agents = torch.nn.Linear(self.map_dim, self.traj_dim, bias=False)
        self.w_v_map = torch.nn.Linear(self.map_dim, self.vdim, bias=False)

    def forward(
            self,
            q: Tensor, k: Tensor, v: Tensor, map_feature: Tensor,
            q_identities: Tensor, k_identities: Tensor,
            mask: Tensor, need_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # q: [L, N, T]
        # k: [S, N, T]
        # v: [S, N, T]
        # map_feature: [M]
        # q_identities: [L]
        # k_identities: [S]

        # print(f"{q.shape, k.shape, v.shape, map_feature.shape=}")
        # print(f"{q_identities.shape, k_identities.shape=}")
        # print(f"{mask, mask.shape=}")

        L, N, _ = q.size()
        S, _, _ = k.size()

        # NOTE: No residual connections used in AgentAwareAttention

        # trajectory related keys, queries and values
        q_traj_map = self.w_q_traj_map(q) * self.map_scaling        # [L, N, T]
        k_traj_map = self.w_k_traj_map(k)                           # [S, N, M]
        v_traj = self.w_v_traj(v)                                   # [S, N, V]

        # map related keys, queries and values
        q_map_self = self.w_q_map_self(map_feature) * self.map_scaling          # [M]
        q_map_agents = self.w_q_map_agents(map_feature) * self.traj_scaling     # [M]
        k_map_self = self.w_k_map_self(map_feature)         # [M]
        k_map_agents = self.w_k_map_agents(map_feature)     # [T]
        v_map = self.w_v_map(map_feature)                   # [V]

        # Tensor reshaping
        q_traj_map = q_traj_map.view(L, N, self.num_heads, self.map_head_dim).transpose(0, 2)       # [H, N, L, t]
        k_traj_map = k_traj_map.view(L, N, self.num_heads, self.map_head_dim).transpose(0, 2)       # [H, N, L, m]
        v_traj = v_traj.view(S, N, self.num_heads, self.v_head_dim).transpose(0, 2)                 # [H, N, S, v]

        q_map_self = q_map_self.view(self.num_heads, self.map_head_dim)     # [H, m]
        q_map_agents = q_map_agents.view(self.num_heads, self.traj_head_dim).unsqueeze(1).unsqueeze(1)  # [H, 1, 1, m]
        k_map_self = k_map_self.view(self.num_heads, self.map_head_dim)     # [H, m]
        k_map_agents = k_map_agents.view(self.num_heads, self.traj_head_dim).unsqueeze(1).unsqueeze(1)  # [H, 1, 1, t]
        v_map = v_map.view(self.num_heads, self.v_head_dim)     # [H, v]

        # cross agent attention
        cross_agent_attention = self.agent_scaled_dot_product(
            q=q, k=k, q_identities=q_identities, k_identities=k_identities, mask=mask
        )       # [N, H, L, S]

        # cross map attention
        map_map_attention = (q_map_self * k_map_self).sum(dim=-1)       # [H]

        # agent map attention, agents query the map
        agent_map_attention = q_traj_map @ k_map_agents.transpose(2, 3)         # [H, N, L, 1]

        # map agent attention, the map queries the agents
        map_agent_attention = q_map_agents @ k_traj_map.transpose(2, 3)         # [H, N, 1, L]

        # TODO: Combine attention scores
        # TODO: dropout
        # TODO: score multiply values
        # TODO: return output

        # TODO: CHECK VALID TENSOR SIZES


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

    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.cfg = cfg
        # print(f"ENCODER_LAYER")
        # self.self_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        self.self_attn = AgentAwareAttentionV2(cfg=cfg, traj_dim=d_model, vdim=d_model, num_heads=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
            self, src: Tensor, src_identities: Tensor,
            src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # print(f"{src, src.shape=}")
        # src2 = self.self_attn(query=src, key=src, value=src,
        #                       query_identities=src_identities, key_identities=src_identities,
        #                       attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(q=src, k=src, v=src, q_identities=src_identities, k_identities=src_identities, mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


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

    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.cfg = cfg
        # print(f"DECODER_LAYER")
        # self.self_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        self.self_attn = AgentAwareAttentionV2(cfg=cfg, traj_dim=d_model, vdim=d_model, num_heads=nhead, dropout=dropout)
        # self.multihead_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        self.multihead_attn = AgentAwareAttentionV2(cfg=cfg, traj_dim=d_model, vdim=d_model, num_heads=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
            self, tgt: Tensor, memory: Tensor, tgt_identities: Tensor, mem_identities: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            need_weights = False
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # tgt2, self_attn_weights = self.self_attn(
        #     query=tgt, key=tgt, value=tgt, query_identities=tgt_identities, key_identities=tgt_identities,
        #     attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=need_weights
        # )
        tgt2, self_attn_weights = self.self_attn(
            q=tgt, k=tgt, v=tgt, q_identities=tgt_identities, k_identities=tgt_identities, mask=tgt_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2, cross_attn_weights = self.multihead_attn(
        #     query=tgt, key=memory, value=memory, query_identities=tgt_identities, key_identities=mem_identities,
        #     attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=need_weights
        # )
        tgt2, cross_attn_weights = self.multihead_attn(
            q=tgt, k=memory, v=memory, q_identities=tgt_identities, k_identities=mem_identities, mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, cross_attn_weights


class AgentFormerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            src_identities: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(
                src=output, src_identities=src_identities, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class AgentFormerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            tgt: Tensor, memory: Tensor,
            tgt_identities: Tensor, mem_identities: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            need_weights = False
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        self_attn_weights = [None] * len(self.layers)
        cross_attn_weights = [None] * len(self.layers)
        for i, mod in enumerate(self.layers):
            output, self_attn_weights[i], cross_attn_weights[i] = mod(
                tgt=output, memory=memory,
                tgt_identities=tgt_identities, mem_identities=mem_identities,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                need_weights=need_weights
            )

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            self_attn_weights = torch.stack(self_attn_weights).cpu().numpy()
            cross_attn_weights = torch.stack(cross_attn_weights).cpu().numpy()

        return output, {'self_attn_weights': self_attn_weights, 'cross_attn_weights': cross_attn_weights}


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
