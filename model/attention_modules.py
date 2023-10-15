import torch
import copy
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from model.layers import \
    WrappedAgentFormerEncoderLayerForMapInput,\
    WrappedAgentFormerDecoderLayerForMapInput,\
    MapAwareAgentFormerEncoderLayer,\
    MapAwareAgentFormerDecoderLayer

from typing import Dict, Optional, Tuple
Tensor = torch.Tensor


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


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

    layer_types = {
        'agent': WrappedAgentFormerEncoderLayerForMapInput,
        'map_agent': MapAwareAgentFormerEncoderLayer
    }

    def __init__(self, encoder_config, num_layers):
        super().__init__()
        self.encoder_config = encoder_config
        self.num_layers = num_layers
        layer = self.layer_types[encoder_config['layer_type']](**encoder_config['layer_params'])
        self.layers = _get_clones(layer, num_layers)

    def forward(
            self, src: Tensor, src_identities: Tensor, map_feature: Tensor,
            src_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        map_output = map_feature

        for layer in self.layers:
            output, map_output = layer(
                src=output, src_identities=src_identities, map_feature=map_output, src_mask=src_mask,
            )

        return output, map_output


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
    layer_types = {
        'agent': WrappedAgentFormerDecoderLayerForMapInput,
        'map_agent': MapAwareAgentFormerDecoderLayer
    }

    def __init__(self, decoder_config, num_layers):
        super().__init__()
        self.decoder_config = decoder_config
        self.num_layers = num_layers
        layer = self.layer_types[decoder_config['layer_type']](**decoder_config['layer_params'])
        self.layers = _get_clones(layer, num_layers)

    def forward(
            self, tgt: Tensor, memory: Tensor,
            tgt_identities: Tensor, mem_identities: Tensor,
            tgt_map: Tensor, mem_map: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Dict]:
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
        map_output = tgt_map

        self_attn_weights = [None] * len(self.layers)
        cross_attn_weights = [None] * len(self.layers)
        for i, mod in enumerate(self.layers):
            output, map_output, self_attn_weights[i], cross_attn_weights[i] = mod(
                tgt=output, memory=memory,
                tgt_identities=tgt_identities, mem_identities=mem_identities,
                tgt_map=map_output, mem_map=mem_map,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
            )

        # if need_weights:
        #     self_attn_weights = torch.stack(self_attn_weights).cpu().numpy()
        #     cross_attn_weights = torch.stack(cross_attn_weights).cpu().numpy()

        return output, map_output, {'self_attn_weights': self_attn_weights, 'cross_attn_weights': cross_attn_weights}
