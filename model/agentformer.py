import matplotlib.axes
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple
import model.decoder_out_submodels as decoder_out_submodels
from model.common.mlp import MLP
from model.agentformer_loss import loss_func
from model.common.dist import Normal, Categorical
from model.attention_modules import \
    AgentFormerEncoder, AgentFormerDecoder, OcclusionFormerEncoder, OcclusionFormerDecoder
from model.map_encoder import MapEncoder
from utils.torch_ops import ExpParamAnnealer
from utils.utils import initialize_weights

from typing import Dict
Tensor = torch.Tensor


def self_other_aware_mask(
        q_identities: Tensor,   # [L]
        k_identities: Tensor    # [S]
) -> Tensor:                    # [L, S]
    return q_identities.unsqueeze(1) == k_identities.unsqueeze(0)


def causal_attention_mask(
        timestep_sequence: torch.Tensor,        # [T]
        batch_size: int = 1
) -> torch.Tensor:
    return torch.where(
        timestep_sequence.unsqueeze(1) < timestep_sequence.unsqueeze(0), float('-inf'), 0.
    ).unsqueeze(0).repeat(batch_size, 1, 1)


def zeros_mask(tgt_sz: int, src_sz: int, batch_size: int = 1) -> torch.Tensor:
    """
    This mask generation process is responsible for the functionality discussed in the paragraph
    "Encoding Agent Connectivity" in the original AgentFormer paper. The function presented here is modified such
    that all agents are connected to one another (or, in other words, the distance threshold value eta is infinite).
    The resulting mask is a tensor full of zero's,
    shaped like the attention matrix QK^T performed in the agent_aware_attention function
    If you need to apply some distance thresholding for your own experiments, you will need to change
    the implementation of this function accordingly.
    """
    return torch.zeros(batch_size, tgt_sz, src_sz)


def non_causal_attention_mask(
        timestep_sequence: torch.Tensor,    # [T]
        batch_size: int = 1
) -> torch.Tensor:                          # [batch_size, T, T]
    return torch.zeros(batch_size, timestep_sequence.shape[0], timestep_sequence.shape[0])


def single_mean_pooling(
        feature_sequence: Tensor,       # [L, *]
        identity_sequence: Tensor       # [L], with N unique values
) -> Tensor:
    agent_masks = identity_sequence.unsqueeze(0) == identity_sequence.unique().unsqueeze(1)     # [N, L]
    sequence_copies = feature_sequence.unsqueeze(0).repeat([agent_masks.shape[0], 1, 1])
    return torch.sum(
        sequence_copies.where(agent_masks.unsqueeze(-1), torch.tensor(0., device=feature_sequence.device)), dim=-2
    ) / torch.sum(agent_masks, dim=-1).unsqueeze(-1)


def mean_pooling(
        sequences: Tensor,      # [B, N, *]
        identities: Tensor      # [B, N]
) -> Tensor:
    # Note: does not work on batched data (ie only works on batch size = 1)
    return single_mean_pooling(sequences[0, ...], identities[0, ...]).unsqueeze(0)


def plot_tensor(ax: matplotlib.axes.Axes, tensor: torch.Tensor, cmap: str = 'Blues'):
    assert tensor.dim() == 2
    img = tensor.detach().cpu().numpy()

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0]+1, 1)
    pos = ax.imshow(img, cmap=cmap, extent=(0, img.shape[1], img.shape[0] + 1, 1))
    plt.colorbar(pos)


POOLING_FUNCTIONS = {
    'mean': mean_pooling,
    'max': None
}


class PositionalEncoding(nn.Module):

    """ Positional Encoding """
    def __init__(
            self, d_model: int,
            dropout: float = 0.1, timestep_window: Tuple[int, int] = (0, 20),
            concat: bool = False, t_zero_index: int = 7
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.concat = concat
        self.t_zero_index = t_zero_index
        timestep_window = torch.arange(*timestep_window, dtype=torch.int)       # [T_total]
        self.register_buffer('timestep_window', timestep_window)
        assert t_zero_index < self.timestep_window.shape[0]
        self.t_index_shift = self.t_zero_index - self.timestep_window[0]

        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_enc_table()
        self.register_buffer('pe', pe)

    def build_enc_table(self) -> torch.Tensor:
        # shape [t_range, d_model]
        pe = torch.zeros(self.timestep_window.shape[0], self.d_model)

        # shape [d_model//2]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(self.timestep_window.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(self.timestep_window.unsqueeze(1) * div_term)
        return pe       # [t_range, d_model]

    def forward(
            self,
            x: Tensor,              # [B, T, model_dim]
            time_tensor: Tensor     # [B, T]
    ) -> Tensor:                    # [B, T, model_dim]
        pos_enc = self.pe[time_tensor + self.t_index_shift]

        if self.concat:
            x = torch.cat([x, pos_enc], dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc

        return self.dropout(x)

    @staticmethod
    def plot_positional_window(
            ax: matplotlib.axes.Axes, tensor: torch.Tensor, offset: int = 0, cmap: str = "Blues"
    ) -> None:
        """
        # self.plot_positional_window(ax, tensor=self.pe, offset=int(self.timestep_window[0, 0]))
        # self.plot_positional_window(ax, tensor=pos_enc)
        """
        img = tensor.T.cpu().numpy()

        ax.set_xlim(offset, img.shape[1] + offset)
        ax.set_ylim(img.shape[0] + 1, 1)
        pos = ax.imshow(img, cmap=cmap,
                        extent=(offset, img.shape[1] + offset, img.shape[0] + 1, 1))
        plt.colorbar(pos)


class ContextEncoder(nn.Module):
    """ Context (Past) Encoder """
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.motion_dim = ctx['motion_dim']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.n_head = ctx['tf_n_head']
        self.dropout = ctx['tf_dropout']
        self.bias_self = ctx.get('bias_self', False)
        self.bias_other = ctx.get('bias_other', False)
        self.bias_out = ctx.get('bias_out', True)
        self.n_layer = ctx['context_encoder'].get('n_layer', 6)
        self.input_type = ctx['input_type']
        self.input_impute_markers = ctx['input_impute_markers']
        self.pooling = ctx['context_encoder'].get('pooling', 'mean')
        self.global_map_attention = ctx['global_map_attention']
        self.causal_attention = ctx['causal_attention']

        in_dim = self.motion_dim * len(self.input_type) + int(self.input_impute_markers)
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        layer_params = {
            'd_model': self.model_dim,
            'n_head': self.n_head,
            'dim_feedforward': self.ff_dim,
            'dropout': self.dropout,
            'bias_self': self.bias_self,
            'bias_other': self.bias_other,
            'bias_out': self.bias_out
        }

        if self.global_map_attention:
            self.bias_map = ctx.get('bias_map', False)
            layer_params['bias_map'] = self.bias_map

            self.tf_encoder = OcclusionFormerEncoder(layer_params=layer_params, num_layers=self.n_layer)
            self.tf_encoder_call = self.map_agent_encoder_call
        else:
            self.tf_encoder = AgentFormerEncoder(layer_params=layer_params, num_layers=self.n_layer)
            self.tf_encoder_call = self.agent_encoder_call

        self.pos_encoder = PositionalEncoding(
            self.model_dim, dropout=self.dropout,
            concat=ctx['pos_concat'], t_zero_index=ctx['t_zero_index']
        )

        if self.causal_attention:
            self.attention_mask = causal_attention_mask
        else:
            self.attention_mask = non_causal_attention_mask

        self.pool = POOLING_FUNCTIONS[self.pooling]

    def agent_encoder_call(
            self,
            data: Dict,
            tf_in_pos: Tensor,              # [B, O, model_dim]
            src_self_other_mask: Tensor,    # [B, O, O]
            src_mask: Tensor                # [B, O, O]
    ):
        data['context_enc'] = self.tf_encoder(
            src=tf_in_pos,
            src_self_other_mask=src_self_other_mask,
            src_mask=src_mask
        )                                   # [B, O, model_dim], [B, model_dim]

    def map_agent_encoder_call(
            self,
            data: Dict,
            tf_in_pos: Tensor,              # [B, O, model_dim]
            src_self_other_mask: Tensor,    # [B, O, O]
            src_mask: Tensor                # [B, O, O]
    ):
        data['context_enc'], data['context_map'] = self.tf_encoder(
            src=tf_in_pos,
            src_self_other_mask=src_self_other_mask,
            map_feature=data['global_map_encoding'],    # [B, model_dim]
            src_mask=src_mask
        )                                   # [B, O, model_dim], [B, model_dim]

    def forward(self, data: Dict):
        # NOTE: This function does not work with batch sizes != 1

        seq_in = [data[f'obs_{key}_sequence'] for key in self.input_type]
        if self.input_impute_markers:
            seq_in.append(data['obs_imputation_sequence'])
        seq_in = torch.cat(seq_in, dim=-1)      # [B, O, Features]
        tf_seq_in = self.input_fc(seq_in)       # [B, O, model_dim]

        tf_in_pos = self.pos_encoder(
            x=tf_seq_in,
            time_tensor=data['obs_timestep_sequence']   # [B, O]
        )                                               # [B, O, model_dim]

        self_other_mask = self_other_aware_mask(
            q_identities=data['obs_identity_sequence'][0], k_identities=data['obs_identity_sequence'][0]
        ).unsqueeze(0)          # [B, O, O]

        src_mask = self.attention_mask(
            timestep_sequence=data['obs_timestep_sequence'][0, ...],     # [O]
            batch_size=1
        ).to(tf_seq_in.device)       # [B, O, O]

        self.tf_encoder_call(data=data, tf_in_pos=tf_in_pos, src_self_other_mask=self_other_mask, src_mask=src_mask)

        # compute per agent context
        assert torch.all(data['obs_identity_sequence'][0, ...].unique() == data['valid_id'][0, ...]),\
            f"{data['obs_identity_sequence'][0, ...].unique()}, {data['valid_id'][0, ...]}"
        data['agent_context'] = self.pool(
            sequences=data['context_enc'], identities=data['obs_identity_sequence']
        )           # [B, N, model_dim]


class FutureEncoder(nn.Module):
    """ Future Encoder """
    def __init__(self, ctx):
        super().__init__()
        self.forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.n_head = ctx['tf_n_head']
        self.dropout = ctx['tf_dropout']
        self.bias_self = ctx.get('bias_self', False)
        self.bias_other = ctx.get('bias_other', False)
        self.bias_out = ctx.get('bias_out', True)
        self.n_layer = ctx['future_encoder'].get('n_layer', 6)
        self.out_mlp_dim = ctx['future_encoder'].get('out_mlp_dim', None)
        self.input_type = ctx['fut_input_type']
        self.pooling = ctx['future_encoder'].get('pooling', 'mean')
        self.global_map_attention = ctx['global_map_attention']
        self.causal_attention = ctx['causal_attention']

        # networks
        in_dim = self.forecast_dim * len(self.input_type)
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        layer_params = {
            'd_model': self.model_dim,
            'n_head': self.n_head,
            'dim_feedforward': self.ff_dim,
            'dropout': self.dropout,
            'bias_self': self.bias_self,
            'bias_other': self.bias_other,
            'bias_out': self.bias_out
        }

        if self.global_map_attention:
            self.bias_map = ctx.get('bias_map', False)
            layer_params['bias_map'] = self.bias_map

            self.tf_decoder = OcclusionFormerDecoder(layer_params=layer_params, num_layers=self.n_layer)
            self.tf_decoder_call = self.map_agent_decoder_call
        else:
            self.tf_decoder = AgentFormerDecoder(layer_params=layer_params, num_layers=self.n_layer)
            self.tf_decoder_call = self.agent_decoder_call

        self.pos_encoder = PositionalEncoding(
            self.model_dim, dropout=self.dropout,
            concat=ctx['pos_concat'], t_zero_index=ctx['t_zero_index']
        )

        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

        if self.causal_attention:
            self.attention_mask = causal_attention_mask
        else:
            self.attention_mask = non_causal_attention_mask

        self.pool = POOLING_FUNCTIONS[self.pooling]

    def agent_decoder_call(
            self,
            data: Dict,
            tf_in_pos: Tensor,                  # [B, P, model_dim]
            tgt_tgt_self_other_mask: Tensor,    # [B, P]
            tgt_mem_self_other_mask: Tensor,    # [B, O]
            tgt_mask: Tensor,                   # [B, P, P]
            mem_mask: Tensor                    # [B, P, O]
    ) -> Tensor:                                # [B, P, model_dim]
        tf_out, _ = self.tf_decoder(
            tgt=tf_in_pos,
            memory=data['context_enc'],         # [B, O, model_dim]
            tgt_tgt_self_other_mask=tgt_tgt_self_other_mask,
            tgt_mem_self_other_mask=tgt_mem_self_other_mask,
            tgt_mask=tgt_mask,
            memory_mask=mem_mask,
        )                                       # [B, P, model_dim], [B, model_dim]
        return tf_out

    def map_agent_decoder_call(
            self,
            data: Dict,
            tf_in_pos: Tensor,                  # [B, P, model_dim]
            tgt_tgt_self_other_mask: Tensor,    # [B, P]
            tgt_mem_self_other_mask: Tensor,    # [B, O]
            tgt_mask: Tensor,                   # [B, P, P]
            mem_mask: Tensor                    # [B, P, O]
    ) -> Tensor:                                # [B, P, model_dim]
        tf_out, _, _ = self.tf_decoder(
            tgt=tf_in_pos,
            memory=data['context_enc'],             # [B, O, model_dim]
            tgt_tgt_self_other_mask=tgt_tgt_self_other_mask,
            tgt_mem_self_other_mask=tgt_mem_self_other_mask,
            tgt_map=data['global_map_encoding'],    # [B, model_dim]
            mem_map=data['context_map'],            # [B, model_dim]
            tgt_mask=tgt_mask,
            memory_mask=mem_mask,
        )                                           # [B, P, model_dim], [B, model_dim]
        return tf_out

    def forward(self, data):
        # NOTE: This function does not work with batch sizes != 1

        seq_in = [data[f'pred_{key}_sequence'] for key in self.input_type]
        seq_in = torch.cat(seq_in, dim=-1)      # [B, P, Features]
        tf_seq_in = self.input_fc(seq_in)       # [B, P, model_dim]

        tf_in_pos = self.pos_encoder(
            x=tf_seq_in,
            time_tensor=data['pred_timestep_sequence']      # [B, P]
        )                                                   # [B, P, model_dim]

        tgt_self_other_mask = self_other_aware_mask(
            q_identities=data['pred_identity_sequence'][0], k_identities=data['pred_identity_sequence'][0]
        ).unsqueeze(0)      # [B, P, P]
        mem_self_other_mask = self_other_aware_mask(
            q_identities=data['pred_identity_sequence'][0], k_identities=data['obs_identity_sequence'][0]
        ).unsqueeze(0)      # [B, P, O]

        mem_mask = zeros_mask(
            tgt_sz=data['pred_timestep_sequence'].shape[1],
            src_sz=data['obs_timestep_sequence'].shape[1],
            batch_size=1
        ).to(tf_seq_in.device)          # [B, P, O]

        tgt_mask = self.attention_mask(
            timestep_sequence=data['pred_timestep_sequence'][0, ...],       # [P]
            batch_size=1
        ).to(tf_seq_in.device)          # [B, P, P]

        tf_out = self.tf_decoder_call(
            data=data, tf_in_pos=tf_in_pos,
            tgt_tgt_self_other_mask=tgt_self_other_mask, tgt_mem_self_other_mask=mem_self_other_mask,
            tgt_mask=tgt_mask, mem_mask=mem_mask
        )

        h = self.pool(
            sequences=tf_out, identities=data['pred_identity_sequence']
        )       # [B, N, model_dim]

        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)        # [B, N, nz (*2 if self.z_type == gaussian)]

        if self.z_type == 'gaussian':
            data['q_z_dist'] = Normal(params=q_z_params)
        else:
            data['q_z_dist'] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data['q_z_samp'] = data['q_z_dist'].rsample()       # [B, N, nz]


class FutureDecoder(nn.Module):
    """ Future Decoder """
    def __init__(self, ctx):
        super().__init__()
        self.ar_detach = ctx['ar_detach']
        self.forecast_dim = ctx['forecast_dim']
        self.pred_type = ctx['pred_type']
        self.pred_mode = ctx['future_decoder'].get('mode', 'point')
        self.sn_out_type = ctx['sn_out_type']
        self.input_type = ctx['dec_input_type']
        self.future_frames = ctx['future_frames']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.n_head = ctx['tf_n_head']
        self.dropout = ctx['tf_dropout']
        self.bias_self = ctx.get('bias_self', False)
        self.bias_other = ctx.get('bias_other', False)
        self.bias_out = ctx.get('bias_out', True)
        self.n_layer = ctx['future_decoder'].get('n_layer', 6)
        self.out_mlp_dim = ctx['future_decoder'].get('out_mlp_dim', None)
        self.learn_prior = ctx['learn_prior']
        self.global_map_attention = ctx['global_map_attention']

        assert self.pred_mode in ["point"]

        # networks
        in_dim = self.forecast_dim + len(self.input_type) * self.forecast_dim + self.nz
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        layer_params = {
            'd_model': self.model_dim,
            'n_head': self.n_head,
            'dim_feedforward': self.ff_dim,
            'dropout': self.dropout,
            'bias_self': self.bias_self,
            'bias_other': self.bias_other,
            'bias_out': self.bias_out
        }

        if self.global_map_attention:
            self.bias_map = ctx.get('bias_map', False)
            layer_params['bias_map'] = self.bias_map

            self.tf_decoder = OcclusionFormerDecoder(layer_params=layer_params, num_layers=self.n_layer)
            self.tf_decoder_call = self.map_agent_decoder_call
        else:
            self.tf_decoder = AgentFormerDecoder(layer_params=layer_params, num_layers=self.n_layer)
            self.tf_decoder_call = self.agent_decoder_call

        self.pos_encoder = PositionalEncoding(
            self.model_dim, dropout=self.dropout,
            concat=ctx['pos_concat'], t_zero_index=ctx['t_zero_index']
        )

        out_module_kwargs = {"hidden_dims": self.out_mlp_dim}
        self.out_module = getattr(decoder_out_submodels, f"{self.pred_mode}_out_module")(
            model_dim=self.model_dim, forecast_dim=self.forecast_dim, **out_module_kwargs
        )
        initialize_weights(self.out_module.modules())

        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def agent_decoder_call(
            self,
            data: Dict,
            tf_in_pos: Tensor,                  # [B * K, M, model_dim]
            context: Tensor,                    # [B * K, O, model_dim]
            tgt_tgt_self_other_mask: Tensor,    # [B, M, M]
            tgt_mem_self_other_mask: Tensor,    # [B, M, O]
            tgt_mask: Tensor,                   # [B * K, M, M]
            mem_mask: Tensor,                   # [B * K, M, O]
            sample_num: int                     # K
    ) -> Tuple[Tensor, Dict]:                   # [B * K, M, model_dim], Dict
        tf_out, attn_weights = self.tf_decoder(
            tgt=tf_in_pos,
            memory=context,
            tgt_tgt_self_other_mask=tgt_tgt_self_other_mask.repeat(sample_num, 1, 1),  # [B * K, M, M]
            tgt_mem_self_other_mask=tgt_mem_self_other_mask.repeat(sample_num, 1, 1),  # [B * K, M, O]
            tgt_mask=tgt_mask,
            memory_mask=mem_mask
        )
        return tf_out, attn_weights

    def map_agent_decoder_call(
            self,
            data: Dict,
            tf_in_pos: Tensor,                  # [B * K, M, model_dim]
            context: Tensor,                    # [B * K, O, model_dim]
            tgt_tgt_self_other_mask: Tensor,    # [B, M, M]
            tgt_mem_self_other_mask: Tensor,    # [B, M, O]
            tgt_mask: Tensor,                   # [B * K, M, M]
            mem_mask: Tensor,                   # [B * K, M, O]
            sample_num: int                     # K
    ) -> Tuple[Tensor, Dict]:                   # [B * K, M, model_dim], Dict
        tf_out, map_out, attn_weights = self.tf_decoder(
            tgt=tf_in_pos,
            memory=context,
            tgt_tgt_self_other_mask=tgt_tgt_self_other_mask.repeat(sample_num, 1, 1),   # [B * K, M, M]
            tgt_mem_self_other_mask=tgt_mem_self_other_mask.repeat(sample_num, 1, 1),   # [B * K, M, O]
            tgt_map=data['global_map_encoding'].repeat(sample_num, 1),                  # [B * K, model_dim]
            mem_map=data['context_map'].repeat(sample_num, 1),                          # [B * K, model_dim]
            tgt_mask=tgt_mask,
            memory_mask=mem_mask
        )       # [B * K, M, model_dim], [B * K, model_dim], Dict
        return tf_out, attn_weights

    def decode_next_timestep(
            self,
            dec_in_orig: torch.Tensor,              # [B * K, N, 2]
            z_in_orig: torch.Tensor,                # [B * K, N, nz]
            dec_input_sequence: torch.Tensor,       # [B * K, M, nz + 2]
            timestep_sequence: torch.Tensor,        # [M]
            agent_sequence: torch.Tensor,           # [B, M]
            data: dict,
            context: torch.Tensor,                  # [B * K, O, model_dim]
            sample_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:

        # Embed input sequence in high-dim space
        tf_in = self.input_fc(dec_input_sequence)   # [B * K, M, model_dim]

        # Temporal encoding
        tf_in_pos = self.pos_encoder(
            x=tf_in,
            time_tensor=timestep_sequence.unsqueeze(0).repeat(tf_in.shape[0], 1)       # [B * K, M]
        )           # [B * K, M, model_dim]

        tgt_self_other_mask = self_other_aware_mask(
            q_identities=agent_sequence[0], k_identities=agent_sequence[0]
        ).unsqueeze(0)      # [B, M, M]
        mem_self_other_mask = self_other_aware_mask(
            q_identities=agent_sequence[0], k_identities=data['obs_identity_sequence'][0]
        ).unsqueeze(0)      # [B, P, O]

        # Generate attention masks (tgt_mask ensures proper autoregressive attention, such that predictions which
        # were originally made at loop iteration nr t cannot attend from sequence elements which have been added
        # at loop iterations >t)
        tgt_mask = causal_attention_mask(
            timestep_sequence=timestep_sequence,
            batch_size=tf_in.shape[0]
        ).to(tf_in.device)      # [B * K, M, M]
        mem_mask = zeros_mask(
            tgt_sz=timestep_sequence.shape[0],
            src_sz=context.shape[1],
            batch_size=tf_in.shape[0]
        ).to(tf_in.device)      # [B * K, M, O]

        # Go through the attention mechanism
        tf_out, attn_weights = self.tf_decoder_call(
            data=data, tf_in_pos=tf_in_pos, context=context,
            tgt_tgt_self_other_mask=tgt_self_other_mask, tgt_mem_self_other_mask=mem_self_other_mask,
            tgt_mask=tgt_mask, mem_mask=mem_mask, sample_num=sample_num
        )

        # Map back to physical space
        seq_out = self.out_module(tf_out)  # [B * K, M, 2]

        # self.sn_out_type='norm' is used to have the model predict offsets from the last observed position of agents,
        # instead of absolute coordinates in space
        if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
            # norm_motion = seq_out  # [B * K, M, 2]

            if self.sn_out_type == 'vel':
                raise NotImplementedError("self.sn_out_type == 'vel'")

            # defining origins for each element in the sequence, using agent_sequence, dec_in and data['valid_id']
            # NOTE: current implementation cannot handle batched data
            seq_origins = dec_in_orig[
                :, (agent_sequence.unsqueeze(2) == data['valid_id'].unsqueeze(1)).nonzero()[..., -1], :
            ]       # [B * K, M, 2]

            seq_out = seq_out + seq_origins  # [B * K, M, 2]

        # create out_in -> Partially from prediction, partially from dec_in (due to occlusion asynchronicity)
        from_pred_indices = (timestep_sequence == torch.max(timestep_sequence))  # [M]
        agents_from_pred = agent_sequence[0, from_pred_indices]         # [⊆M] <==> [m]

        from_dec_in_indices = (data['last_obs_timesteps'][0] == torch.max(timestep_sequence) + 1)  # [N]
        agents_from_dec_in = data['valid_id'][0, from_dec_in_indices]      # [⊆N] <==> [n]

        if self.ar_detach:
            out_in_from_pred = seq_out[:, from_pred_indices, :].clone().detach()          # [B * K, m, 2]
            out_in_from_dec_in = dec_in_orig[:, from_dec_in_indices, :].clone().detach()  # [B * K, n, 2]
        else:
            out_in_from_pred = seq_out[from_pred_indices, ...]              # [B * K, m, 2]
            out_in_from_dec_in = dec_in_orig[from_dec_in_indices, ...]      # [B * K, n, 2]

        # concatenate with latent z codes
        z_in_from_pred = z_in_orig[
            :, (agents_from_pred.unsqueeze(0).unsqueeze(2) == data['valid_id'].unsqueeze(1)).nonzero()[..., -1], :
        ]  # [B * K, m, nz]
        out_in_z_from_pred = torch.cat(
            [out_in_from_pred, z_in_from_pred], dim=-1
        )  # [B * K, m, nz + 2]

        z_in_from_dec_in = z_in_orig[:, from_dec_in_indices, :]  # [B * sample_num, n, nz]
        out_in_z_from_dec_in = torch.cat(
            [out_in_from_dec_in, z_in_from_dec_in], dim=-1
        )  # [B * K, n, nz + 2]

        # generate timestep tensor to extend timestep_sequence for next loop iteration
        next_timesteps = torch.full(
            [agents_from_pred.shape[0] + agents_from_dec_in.shape[0]], torch.max(timestep_sequence) + 1,
            device=tf_in.device
        )       # [m + n]

        # update trajectory sequence
        dec_input_sequence = torch.cat(
            [dec_input_sequence, out_in_z_from_pred, out_in_z_from_dec_in], dim=1
        )  # [B * K, M + m + n, nz + 2]     -> next loop: B == B + *~B + *~N

        # update agent_sequence, timestep_sequence
        agent_sequence = torch.cat(
            [agent_sequence, agents_from_pred.unsqueeze(0), agents_from_dec_in.unsqueeze(0)], dim=1
        )  # [B, M + m + n]

        timestep_sequence = torch.cat(
            [timestep_sequence, next_timesteps], dim=0
        )  # [M + m + n]

        return seq_out, dec_input_sequence, agent_sequence, timestep_sequence, attn_weights

    def decode_traj_ar(self, data, mode, context, z, sample_num, need_weights=False):
        # retrieving the most recent observation for each agent
        dec_in = data['last_obs_positions'].repeat(sample_num, 1, 1)     # [B * K, N, 2]

        # z: [B * sample_num, N, nz]
        in_arr = [dec_in, z]
        dec_in_z = torch.cat(in_arr, dim=-1)        # [B * K, N, nz + 2]

        catch_up_timestep_sequence = data['last_obs_timesteps'][0, ...].detach().clone().to(dec_in.device)      # [N]
        starting_seq_indices = (catch_up_timestep_sequence == torch.min(catch_up_timestep_sequence, dim=0)[0])  # [N]

        timestep_sequence = catch_up_timestep_sequence[starting_seq_indices]                            # [⊆N] == [M]
        agent_sequence = data['valid_id'][:, starting_seq_indices].detach().clone()                     # [B, M]
        dec_input_sequence = dec_in_z[:, starting_seq_indices].detach().clone()                         # [B * K, M, nz + 2]

        # catch up to t_0
        while not torch.all(catch_up_timestep_sequence == torch.zeros_like(catch_up_timestep_sequence)):
            seq_out, dec_input_sequence, agent_sequence, timestep_sequence, attn_weights = self.decode_next_timestep(
                dec_in_orig=dec_in,                         # [B * K, N, 2]
                z_in_orig=z,                                # [B * K, N, nz]
                dec_input_sequence=dec_input_sequence,      # [B * K, M, nz + 2]
                timestep_sequence=timestep_sequence,        # [M]
                agent_sequence=agent_sequence,              # [B, M]
                data=data,
                context=context,                            # [B * K, O, model_dim]
                sample_num=sample_num
            )       # [B * K, M, 2], [B * K, M + m + n, nz + 2], [B, M + m + n], [M + m + n], Dict

            catch_up_timestep_sequence[catch_up_timestep_sequence == torch.min(catch_up_timestep_sequence)] += 1

        # predict the future
        for i in range(self.future_frames):
            seq_out, dec_input_sequence, agent_sequence, timestep_sequence, attn_weights = self.decode_next_timestep(
                dec_in_orig=dec_in,
                z_in_orig=z,
                dec_input_sequence=dec_input_sequence,
                timestep_sequence=timestep_sequence,
                agent_sequence=agent_sequence,
                data=data,
                context=context,
                sample_num=sample_num
            )

        # timestep_sequence is defined as the timesteps corresponding to the observations / predictions in the *input*
        # sequence that is being fed to the model. The timesteps corresponding to the *predicted* sequence are shifted
        # by one from this original timestep_sequence. Additionally, the model did not actually consume the given
        # timestep_sequence at the last loop iteration. The timestep_sequence (and its corresponding trajectory and
        # agent sequences) was extended at the last loop iteration without being fed to the model. In order to obtain
        # the actual sequences of predicted timesteps and agents, we need to remove the indices corresponding to the
        # last timestep value present in the *input* timestep sequence.
        keep_indices = (timestep_sequence < torch.max(timestep_sequence))
        pred_timestep_sequence = (timestep_sequence[keep_indices] + 1).detach().clone()     # [P]
        pred_agent_sequence = (agent_sequence[:, keep_indices]).detach().clone()            # [B, P]

        past_indices = (pred_timestep_sequence <= 0)

        if self.pred_type == 'scene_norm':
            scene_origs = data['scene_orig'].repeat(sample_num, 1).unsqueeze(1)         # [B * K, 1, 2]
            seq_out += scene_origs                                                      # [B * K, P, 2]
        else:
            raise NotImplementedError

        data[f'{mode}_dec_motion'] = seq_out                                    # [B * K, P, 2]
        data[f'{mode}_dec_agents'] = pred_agent_sequence.repeat(sample_num, 1)  # [B * K, P]
        data[f'{mode}_dec_past_mask'] = past_indices                            # [P]
        data[f'{mode}_dec_timesteps'] = pred_timestep_sequence                  # [P]
        if need_weights:
            data['attn_weights'] = attn_weights

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data['context_enc'].repeat(sample_num, 1, 1)       # [B * K, O, model_dim], with sample_num <==> K

        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        if self.learn_prior:
            h = data['agent_context'].repeat(sample_num, 1, 1)      # [B * K, N, model_dim]
            p_z_params = self.p_z_net(h)                            # [B * K, N, nz (*2 if self.z_type == gaussian)]
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(
                    mu=torch.zeros(context.shape[0], data['agent_num'], self.nz).to(data['context_enc'].device),
                    logvar=torch.zeros(context.shape[0], data['agent_num'], self.nz).to(data['context_enc'].device)
                )
            else:
                data[prior_key] = Categorical(
                    logits=torch.zeros(context.shape[0], data['agent_num'], self.nz).to(data['context_enc'].device)
                )

        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()    # [B, N, nz]
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()         # [B * K, N, nz]
            else:
                raise ValueError('Unknown Mode!')

        if autoregress:
            self.decode_traj_ar(
                data=data,
                mode=mode,
                context=context,
                z=z,
                sample_num=sample_num,
                need_weights=need_weights
            )


class AgentFormer(nn.Module):
    """ AgentFormer """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')

        input_type = cfg.get('input_type', 'pos')
        pred_type = cfg.get('pred_type', input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get('fut_input_type', input_type)
        dec_input_type = cfg.get('dec_input_type', [])
        ctx = {
            'nz': cfg.nz,
            'z_type': cfg.get('z_type', 'gaussian'),
            'future_frames': cfg.future_frames,
            'motion_dim': cfg.motion_dim,
            'forecast_dim': cfg.forecast_dim,
            'input_type': input_type,
            'fut_input_type': fut_input_type,
            'dec_input_type': dec_input_type,
            'pred_type': pred_type,
            'tf_n_head': cfg.tf_n_head,
            'tf_model_dim': cfg.tf_model_dim,
            'tf_ff_dim': cfg.tf_ff_dim,
            'tf_dropout': cfg.tf_dropout,
            'bias_self': cfg.get('bias_self', False),
            'bias_other': cfg.get('bias_other', False),
            'bias_out': cfg.get('bias_out', True),
            'bias_map': cfg.get('bias_map', False),
            'pos_concat': cfg.get('pos_concat', False),
            't_zero_index': int(cfg.get('t_zero_index', 0)),
            'ar_detach': cfg.get('ar_detach', True),
            'sn_out_type': cfg.get('sn_out_type', 'scene_norm'),
            'learn_prior': cfg.get('learn_prior', False),
            'global_map_attention': cfg.get('global_map_attention', False),
            'causal_attention': cfg.get('causal_attention', False),
            'context_encoder': cfg.context_encoder,
            'future_encoder': cfg.future_encoder,
            'future_decoder': cfg.future_decoder
        }
        self.global_map_attention = cfg.get('global_map_attention', False)
        self.input_impute_markers = cfg.get('input_impute_markers', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        self.loss_cfg = cfg.get('loss_cfg')
        for map_key in ('occl_map', 'infer_occl_map'):
            if map_key in self.loss_cfg.keys():
                if self.loss_cfg[map_key].get('kernel', None) == 'squared':
                    self.loss_cfg[map_key]['kernel'] = lambda x: torch.pow(x, 2)
                elif self.loss_cfg[map_key].get('kernel', None) is not None:
                    raise NotImplementedError

        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        self.z_type = cfg.get('z_type', 'gaussian')
        if self.z_type == 'discrete':
            ctx['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None

        if self.global_map_attention:
            map_enc_cfg = cfg.global_map_encoder
            map_enc_cfg['map_resolution'] = cfg.global_map_resolution
            self.global_map_encoder = MapEncoder(map_enc_cfg)
            ctx['global_map_enc_dim'] = self.global_map_encoder.out_dim
            if map_enc_cfg.use_scene_map and map_enc_cfg.use_occlusion_map:
                self.set_map_data = self.set_map_data_combined
            elif map_enc_cfg.use_scene_map and not map_enc_cfg.use_occlusion_map:
                self.set_map_data = self.set_map_data_scene
            elif not map_enc_cfg.use_scene_map and map_enc_cfg.use_occlusion_map:
                self.set_map_data = self.set_map_data_occlusion
            else:
                raise NotImplementedError

            self.occl_loss_map_key = cfg.get('loss_map', 'clipped_dist_transformed_occlusion_map')

        ctx['input_impute_markers'] = self.input_impute_markers

        # models
        self.context_encoder = ContextEncoder(ctx)
        self.future_encoder = FutureEncoder(ctx)
        self.future_decoder = FutureDecoder(ctx)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_map_data_combined(self, data: Dict) -> None:
        self.data['scene_map'] = data['scene_map'].detach().clone().to(self.device)  # [B, C, H, W]
        self.data['occlusion_map'] = data['dist_transformed_occlusion_map'] \
            .detach().clone().to(self.device).unsqueeze(1)  # [B, 1, H, W]
        self.data['combined_map'] = torch.cat(
            (self.data['scene_map'], self.data['occlusion_map']), dim=1
        )  # [B, (C) + (1), H, W]
        self.data['input_global_map'] = self.data['combined_map']

    def set_map_data_scene(self, data: Dict) -> None:
        self.data['scene_map'] = data['scene_map'].detach().clone().to(self.device)  # [B, C, H, W]
        self.data['input_global_map'] = self.data['scene_map']

    def set_map_data_occlusion(self, data: Dict) -> None:
        self.data['occlusion_map'] = data['dist_transformed_occlusion_map'] \
            .detach().clone().to(self.device).unsqueeze(1)  # [B, 1, H, W]
        self.data['input_global_map'] = self.data['occlusion_map']

    def set_data(self, data: Dict) -> None:
        # NOTE: in our case, batch size B is always 1
        self.data = defaultdict(lambda: None)

        self.data['valid_id'] = data['identities'].detach().clone().to(self.device)     # [B, N]
        self.data['T_total'] = data['timesteps'].shape[-1]
        self.data['agent_num'] = self.data['valid_id'].shape[-1]
        self.data['timesteps'] = data['timesteps'].detach().clone().to(self.device)     # [B, T]
        self.data['scene_orig'] = data['scene_orig'].detach().clone().to(self.device)   # [B, 2]

        self.data['obs_position_sequence'] = data['obs_position_sequence'].detach().clone().to(self.device)     # [B, O, 2]
        self.data['obs_velocity_sequence'] = data['obs_velocity_sequence'].detach().clone().to(self.device)     # [B, O, 2]
        self.data['obs_timestep_sequence'] = data['obs_timestep_sequence'].detach().clone().to(self.device)     # [B, O]
        self.data['obs_identity_sequence'] = data['obs_identity_sequence'].detach().clone().to(self.device)     # [B, O]
        self.data['last_obs_positions'] = data['last_obs_positions'].detach().clone().to(self.device)               # [B, N, 2]
        self.data['last_obs_timesteps'] = data['last_obs_timesteps'].detach().clone().to(self.device)               # [B, N]
        self.data['agent_mask'] = torch.zeros([1, self.data['agent_num'], self.data['agent_num']]).to(self.device)  # [B, N, N]

        self.data['pred_position_sequence'] = data['pred_position_sequence'].detach().clone().to(self.device)       # [B, P, 2]
        self.data['pred_velocity_sequence'] = data['pred_velocity_sequence'].detach().clone().to(self.device)       # [B, P, 2]
        self.data['pred_timestep_sequence'] = data['pred_timestep_sequence'].detach().clone().to(self.device)       # [B, P]
        self.data['pred_identity_sequence'] = data['pred_identity_sequence'].detach().clone().to(self.device)       # [B, P]

        if self.global_map_attention:
            self.set_map_data(data=data)

            self.data['occlusion_loss_map'] = data[self.occl_loss_map_key] \
                .detach().clone().to(self.device)  # [B, H, W]
            self.data['map_homography'] = data['map_homography'].detach().clone().to(self.device)  # [B, 3, 3]

        if self.input_impute_markers:
            self.data['obs_imputation_sequence'] = data['imputation_mask'] \
                .detach().clone().unsqueeze(-1).to(torch.float32).to(self.device)      # [B, O, 1]

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self):
        if self.global_map_attention:
            self.data['global_map_encoding'] = self.global_map_encoder(self.data['input_global_map'])

        self.context_encoder(self.data)
        self.future_encoder(self.data)
        self.future_decoder(self.data, mode='train', autoregress=self.ar_train)

        if self.compute_sample:
            self.inference(sample_num=self.loss_cfg['sample']['k'])

        return self.data

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        if self.global_map_attention:
            self.data['global_map_encoding'] = self.global_map_encoder(self.data['input_global_map'])
        if self.data['context_enc'] is None:
            self.context_encoder(self.data)
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(self.data)
        self.future_decoder(self.data, mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data       # [B * sample_num, P, 2], Dict

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}

        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()

        return total_loss, loss_dict, loss_unweighted_dict
