import torch
import copy
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from model.attention_layers import \
    AgentAwareAttentionEncoderLayer, AgentAwareAttentionDecoderLayer,\
    MapAgentAwareAttentionEncoderLayer, MapAgentAwareAttentionDecoderLayer

from typing import Dict, Optional, Tuple
Tensor = torch.Tensor


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


# ENCODERS ############################################################################################################


class AgentFormerEncoder(Module):

    def __init__(self, layer_params: Dict, num_layers: int):
        super().__init__()
        self.layer_params = layer_params
        self.num_layers = num_layers
        layer = AgentAwareAttentionEncoderLayer(**layer_params)
        self.layers = _get_clones(layer, num_layers)

    def forward(
            self, src: Tensor, src_identities: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tensor:

        output = src

        for layer in self.layers:
            output = layer(
                src=output, src_identities=src_identities, src_mask=src_mask,
            )

        return output


class OcclusionFormerEncoder(Module):

    def __init__(self, layer_params: Dict, num_layers: int):
        super().__init__()
        self.layer_params = layer_params
        self.num_layers = num_layers
        layer = MapAgentAwareAttentionEncoderLayer(**layer_params)
        self.layers = _get_clones(layer, num_layers)

    def forward(
            self, src: Tensor, src_identities: Tensor, map_feature: Tensor,
            src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        output = src
        map_output = map_feature

        for layer in self.layers:
            output, map_output = layer(
                src=output, src_identities=src_identities, map_feature=map_output, src_mask=src_mask,
            )

        return output, map_output


# DECODERS ############################################################################################################


class AgentFormerDecoder(Module):

    def __init__(self, layer_params: Dict, num_layers: int):
        super().__init__()
        self.layer_params = layer_params
        self.num_layers = num_layers
        layer = AgentAwareAttentionDecoderLayer(**layer_params)
        self.layers = _get_clones(layer, num_layers)

    def forward(
            self, tgt: Tensor, memory: Tensor,
            tgt_identities: Tensor, mem_identities: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict]:
        output = tgt

        self_attn_weights = [None] * len(self.layers)
        cross_attn_weights = [None] * len(self.layers)

        for i, mod in enumerate(self.layers):
            output, self_attn_weights[i], cross_attn_weights[i] = mod(
                tgt=output, memory=memory,
                tgt_identities=tgt_identities, mem_identities=mem_identities,
                tgt_mask=tgt_mask, memory_mask=memory_mask
            )

        return output, {'self_attn_weights': self_attn_weights, 'cross_attn_weights': cross_attn_weights}


class OcclusionFormerDecoder(Module):

    def __init__(self, layer_params, num_layers):
        super().__init__()
        self.layer_params = layer_params
        self.num_layers = num_layers
        layer = MapAgentAwareAttentionDecoderLayer(**layer_params)
        self.layers = _get_clones(layer, num_layers)

    def forward(
            self, tgt: Tensor, memory: Tensor,
            tgt_identities: Tensor, mem_identities: Tensor,
            tgt_map: Tensor, mem_map: Tensor,
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Dict]:
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
