import matplotlib.axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from model.attention_modules import AgentFormerEncoder, AgentFormerDecoder
from model.map_encoder import MapEncoder
from utils.torch import ExpParamAnnealer
from utils.utils import initialize_weights


def generate_ar_mask(sz: int, agent_num: int, agent_mask: torch.Tensor) -> torch.Tensor:
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    raise NotImplementedError
    return mask


def generate_ar_mask_with_variable_agents_per_timestep(
        timestep_sequence: torch.Tensor,
        batch_size: int = 1
) -> torch.Tensor:
    # timestep_sequence [T]
    stop_at = torch.argmax(timestep_sequence)
    mask = torch.zeros(timestep_sequence.shape[0], timestep_sequence.shape[0])
    for idx in range(stop_at):
        mask_seq = (timestep_sequence > timestep_sequence[idx])
        mask[idx, mask_seq] = float('-inf')
    return mask.unsqueeze(0).repeat(batch_size, 1, 1)


def generate_mask(tgt_sz: int, src_sz: int, batch_size: int = 1) -> torch.Tensor:
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


class PositionalEncoding(nn.Module):

    """ Positional Encoding """
    def __init__(
            self, d_model: int,
            dropout: float = 0.1, timestep_window: Tuple[int, int] = (-20, 30), concat: bool = False
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.concat = concat
        timestep_window = torch.arange(*timestep_window, dtype=torch.float).unsqueeze(1)       # [t_range, 1]
        self.register_buffer('timestep_window', timestep_window)
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_enc_table()
        self.register_buffer('pe', pe)

    def build_enc_table(self) -> torch.Tensor:
        # shape [t_range, d_model]
        pe = torch.zeros(len(self.timestep_window), self.d_model)

        # shape [d_model//2]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(self.timestep_window * div_term)
        pe[:, 1::2] = torch.cos(self.timestep_window * div_term)
        return pe       # [t_range, d_model]

    def time_encode(self, sequence_timesteps: torch.Tensor) -> torch.Tensor:
        # sequence_timesteps: [T_total]
        # out: [T_total, self.d_model]
        return torch.cat([self.pe[(self.timestep_window == t).squeeze(), ...] for t in sequence_timesteps], dim=0)

    def forward(self, x: torch.Tensor, time_tensor: torch.Tensor):
        # x: [B, T, model_dim]
        # time_tensor: [B, T]
        pos_enc = torch.stack(
            [self.time_encode(time_sequence) for time_sequence in time_tensor], dim=0
        )    # [B, T, model_dim]

        if self.concat:
            x = torch.cat([x, pos_enc], dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc

        return self.dropout(x)      # [B, T, model_dim]

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
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = ctx['context_encoder'].get('nlayer', 6)
        self.input_type = ctx['input_type']
        self.pooling = ctx['context_encoder'].get('pooling', 'mean')
        self.vel_heading = ctx['vel_heading']
        self.global_map_attention = ctx['global_map_attention']

        ctx['context_dim'] = self.model_dim
        in_dim = self.motion_dim * len(self.input_type)
        # if 'map' in self.input_type:
        #     in_dim += ctx['map_enc_dim'] - self.motion_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        encoder_config = {
            'layer_type': 'map_agent' if self.global_map_attention else 'agent',
            'layer_params': {
                'd_model': self.model_dim,
                'nhead': self.nhead,
                'dim_feedforward': self.ff_dim,
                'dropout': self.dropout,
            }
        }
        self.tf_encoder = AgentFormerEncoder(encoder_config=encoder_config, num_layers=self.nlayer)

        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout, concat=ctx['pos_concat'])

    def forward(self, data):
        # NOTE: THIS FUNCTION IS NOT CAPABLE OF OPERATING ON BATCH SIZES != 1

        seq_in = [data[f'obs_{key}_sequence'] for key in self.input_type]
        seq_in = torch.cat(seq_in, dim=-1)      # [B, O, Features]
        tf_seq_in = self.input_fc(seq_in)       # [B, O, model_dim]

        print(f"{tf_seq_in.shape=}")

        tf_in_pos = self.pos_encoder(
            x=tf_seq_in,
            time_tensor=data['obs_timestep_sequence']   # [B, O]
        )                                               # [B, O, model_dim]

        src_mask = generate_mask(
            data['obs_timestep_sequence'].shape[1],
            data['obs_timestep_sequence'].shape[1]
        ).to(tf_seq_in.device)        # [B, O, O]

        data['context_enc'], data['context_map'] = self.tf_encoder(
            src=tf_in_pos,                                          # [B, O, model_dim]
            src_identities=data['obs_identity_sequence'],           # [B, O]
            map_feature=data['global_map_encoding'],                # [B, model_dim]
            src_mask=src_mask                                       # [B, O, O]
        )                                                           # [B, O, model_dim], [B, model_dim]

        print(f"{data['context_enc'].shape, data['context_map'].shape=}")

        # compute per agent context
        # print(f"{self.pooling=}")
        if self.pooling == 'mean':

            agent_contexts = []
            for context_seq, identities in zip(data['context_enc'], data['obs_identity_sequence']):
                print(f"{torch.unique(identities), torch.unique(identities).shape}")
                agent_contexts.append(
                    torch.stack(
                        [torch.mean(context_seq[identities == ag_id, ...], dim=0)
                         for ag_id in torch.unique(identities)], dim=0
                    )
                )
            data['agent_context'] = torch.stack(agent_contexts)     # [B, N, model_dim]
            print(f"{data['agent_context'].shape=}")

        else:

            agent_contexts = []
            for context_seq, identities in zip(data['context_seq'], data['obs_identity_sequence']):
                agent_contexts.append(
                    torch.stack(
                        [torch.max(context_seq[identities == ag_id, ...])
                         for ag_id in torch.unique(identities)], dim=0
                    )
                )
            data['agent_context'] = torch.stack(agent_contexts)     # [B, N, model_dim]
            print(f"{data['agent_context'].shape=}")


class FutureEncoder(nn.Module):
    """ Future Encoder """
    def __init__(self, ctx):
        super().__init__()
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = ctx['future_encoder'].get('nlayer', 6)
        self.out_mlp_dim = ctx['future_encoder'].get('out_mlp_dim', None)
        self.input_type = ctx['fut_input_type']
        self.pooling = ctx['future_encoder'].get('pooling', 'mean')
        self.vel_heading = ctx['vel_heading']
        self.global_map_attention = ctx['global_map_attention']

        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_config = {
            'layer_type': 'map_agent' if self.global_map_attention else 'agent',
            'layer_params': {
                'd_model': self.model_dim,
                'nhead': self.nhead,
                'dim_feedforward': self.ff_dim,
                'dropout': self.dropout
            }
        }
        self.tf_decoder = AgentFormerDecoder(decoder_config=decoder_config, num_layers=self.nlayer)
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout, concat=ctx['pos_concat'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        # NOTE: THIS FUNCTION IS NOT CAPABLE OF OPERATING ON BATCH SIZES != 1

        seq_in = [data[f'pred_{key}_sequence'] for key in self.input_type]
        seq_in = torch.cat(seq_in, dim=-1)      # [B, P, Features]
        tf_seq_in = self.input_fc(seq_in)       # [B, P, model_dim]
        print(f"{tf_seq_in.shape=}")

        tf_in_pos = self.pos_encoder(
            x=tf_seq_in,
            time_tensor=data['pred_timestep_sequence']      # [B, P]
        )                                                   # [B, P, model_dim]
        print(f"{tf_in_pos.shape=}")

        mem_mask = generate_mask(
            data['pred_timestep_sequence'].shape[1],
            data['obs_timestep_sequence'].shape[1]
        ).to(tf_seq_in.device)          # [B, P, O]
        tgt_mask = generate_mask(
            data['pred_timestep_sequence'].shape[1],
            data['pred_timestep_sequence'].shape[1]
        ).to(tf_seq_in.device)          # [B, P, P]

        print(f"{mem_mask.shape, tgt_mask.shape=}")
        print(f"{data['global_map_encoding'].shape=}")
        print(f"{data['context_map'].shape=}")

        tf_out, map_out, _ = self.tf_decoder(
            tgt=tf_in_pos,                                          # [B, P, model_dim]
            memory=data['context_enc'],                             # [B, O, model_dim]
            tgt_identities=data['pred_identity_sequence'],          # [B, P]
            mem_identities=data['obs_identity_sequence'],           # [B, O]
            tgt_map=data['global_map_encoding'],                    # [B, model_dim]
            mem_map=data['context_map'],                            # [B, model_dim]
            tgt_mask=tgt_mask,                                      # [B, P, P]
            memory_mask=mem_mask,                                   # [B, P, O]
        )                                                           # [B, P, model_dim], [B, model_dim]

        print(f"{tf_out.shape, map_out.shape=}")
        print(f"{self.pooling=}")
        if self.pooling == 'mean':

            h = []
            for feature_seq, identities in zip(tf_out, data['pred_identity_sequence']):
                print(f"{torch.unique(identities), torch.unique(identities).shape=}")
                h.append(
                    torch.stack(
                        [torch.mean(feature_seq[identities == ag_id, ...], dim=0)
                         for ag_id in torch.unique(identities)], dim=0
                    )
                )
            h = torch.stack(h)      # [B, N, model_dim]

            print(f"{h.shape=}")

            # h = torch.cat(
            #     [torch.mean(tf_out[data['fut_agents'] == ag_id, :], dim=0)
            #      for ag_id in torch.unique(data['fut_agents'])], dim=0
            # )       # [N, model_dim]
        else:

            h = []
            for feature_seq, identities in zip(tf_out, data['pred_identity_sequence']):
                print(f"{torch.unique(identities), torch.unique(identities).shape=}")
                h.append(
                    torch.stack(
                        [torch.mean(feature_seq[identities == ag_id, ...])
                         for ag_id in torch.unique(identities)], dim=0
                    )
                )
            h = torch.stack(h)      # [B, N, model_dim]

            print(f"{h.shape=}")
            # h = torch.cat(
            #     [torch.max(tf_out[data['fut_agents'] == ag_id, :], dim=0)[0]
            #      for ag_id in torch.unique(data['fut_agents'])], dim=0
            # )       # [N, model_dim]
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)        # [B, N, nz (*2 if self.z_type == gaussian)]

        print(f"{q_z_params.shape=}")
        print(f"{self.z_type=}")
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
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.pred_scale = ctx['future_decoder'].get('pred_scale', 1.0)
        self.pred_type = ctx['pred_type']
        self.pred_mode = ctx['future_decoder'].get('mode', 'point')
        self.sn_out_type = ctx['sn_out_type']
        self.sn_out_heading = ctx['sn_out_heading']
        self.input_type = ctx['dec_input_type']
        self.future_frames = ctx['future_frames']
        self.past_frames = ctx['past_frames']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = ctx['future_decoder'].get('nlayer', 6)
        self.out_mlp_dim = ctx['future_decoder'].get('out_mlp_dim', None)
        self.pos_offset = ctx['future_decoder'].get('pos_offset', False)
        self.learn_prior = ctx['learn_prior']
        self.global_map_attention = ctx['global_map_attention']

        # sanity check
        assert self.pred_mode in ["point", "gauss"]

        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        if self.pred_mode == "gauss":            # TODO: Maybe this can be integrated in a better way
            in_dim += 3     # adding three extra input dimensions: for the variance terms and correlation term of the distribution
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_config = {
            'layer_type': 'map_agent' if self.global_map_attention else 'agent',
            'layer_params': {
                'd_model': self.model_dim,
                'nhead': self.nhead,
                'dim_feedforward': self.ff_dim,
                'dropout': self.dropout
            }
        }
        self.tf_decoder = AgentFormerDecoder(decoder_config=decoder_config, num_layers=self.nlayer)

        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout, concat=ctx['pos_concat'])

        out_module_kwargs = {"hidden_dims": self.out_mlp_dim}
        self.out_module = getattr(decoder_out_submodels, f"{self.pred_mode}_out_module")(
            model_dim=self.model_dim, forecast_dim=self.forecast_dim, **out_module_kwargs
        )
        initialize_weights(self.out_module.modules())

        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def decode_next_timestep(
            self,
            dec_in_orig: torch.Tensor,              # [B * sample_num, N, 2]
            z_in_orig: torch.Tensor,                # [B * sample_num, N, nz]
            dec_input_sequence: torch.Tensor,       # [B * sample_num, K, nz + 2]
            timestep_sequence: torch.Tensor,        # [K]
            agent_sequence: torch.Tensor,           # [B, K]
            data: dict,
            context: torch.Tensor,                  # [B * sample_num, O, model_dim]
            sample_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # old shapes
        # dec_in_orig: torch.Tensor,  # [N, n_sample, 2]
        # z_in_orig: torch.Tensor,  # [N, n_sample, nz]
        # dec_input_sequence: torch.Tensor,  # [B, n_sample, nz + 2]
        # timestep_sequence: torch.Tensor,  # [B]
        # agent_sequence: torch.Tensor,  # [B]
        # data: dict,
        # context: torch.Tensor,  # [O, n_sample, model_dim]
        # sample_num: int  # n_sample

        # Embed input sequence in high-dim space
        tf_in = self.input_fc(dec_input_sequence)   # [B * sample_num, K, model_dim]
        print(f"{tf_in.shape=}")
        print(f"{timestep_sequence.shape=}")

        # Temporal encoding
        tf_in_pos = self.pos_encoder(
            x=tf_in,
            time_tensor=timestep_sequence.unsqueeze(0).repeat(tf_in.shape[0], 1)       # [B * sample_num, K]
        )           # [B * sample_num, K, model_dim]
        print(f"{tf_in_pos.shape=}")

        # Generate attention masks (tgt_mask ensures proper autoregressive attention, such that predictions which
        # were originally made at loop iteration nr t cannot attend from sequence elements which have been added
        # at loop iterations >t)
        tgt_mask = generate_ar_mask_with_variable_agents_per_timestep(
            timestep_sequence=timestep_sequence,
            batch_size=tf_in.shape[0]
        ).to(tf_in.device)      # [B * sample_num, K, K]
        mem_mask = generate_mask(
            timestep_sequence.shape[0],
            context.shape[1],
            batch_size=tf_in.shape[0]
        ).to(tf_in.device)      # [B * sample_num, K, O]
        print(f"{tgt_mask, tgt_mask.shape=}")
        print(f"{mem_mask, mem_mask.shape=}")

        # Go through the attention mechanism
        # print(f"{data['global_map_encoding'].shape=}")
        # print(f"{data['context_map'].shape=}")
        tf_out, map_out, attn_weights = self.tf_decoder(
            tgt=tf_in_pos,                                                          # [B * sample_num, K, model_dim]
            memory=context,                                                         # [B * sample_num, O, model_dim]
            tgt_identities=agent_sequence.repeat(sample_num, 1),                    # [B * sample_num, K]
            mem_identities=data['obs_identity_sequence'].repeat(sample_num, 1),     # [B * sample_num, O]
            tgt_map=data['global_map_encoding'].repeat(sample_num, 1),              # [B * sample_num, model_dim]
            mem_map=data['context_map'].repeat(sample_num, 1),                      # [B * sample_num, model_dim]
            tgt_mask=tgt_mask,                                                      # [B * sample_num, K, K]
            memory_mask=mem_mask                                                    # [B * sample_num, K, O]
        )       # [B * sample_num, K, model_dim], [B * sample_num, model_dim], Dict

        print(f"{tf_out.shape, map_out.shape=}")

        # Map back to physical space
        seq_out = self.out_module(tf_out)  # [B * sample_num, K, 2]
        print(f"{seq_out.shape=}")

        # self.sn_out_type='norm' is used to have the model predict offsets from the last observed position of agents,
        # instead of absolute coordinates in space
        print(f"{self.sn_out_type=}")
        if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
            # norm_motion = seq_out  # [B * n_sample, K, 2]
            # # print(f"{norm_motion.shape=}")

            if self.sn_out_type == 'vel':
                raise NotImplementedError("self.sn_out_type == 'vel'")

            print(f"{agent_sequence, agent_sequence.shape=}")       # [B, K]
            print(f"{data['valid_id'], data['valid_id'].shape=}")     # [B, N]
            print(f"{dec_in_orig, dec_in_orig.shape=}")          # [B * sample_num, N, 2]

            # defining origins for each element in the sequence, using agent_sequence, dec_in and data['valid_id']
            # NOTE: current implementation cannot handle batched data
            # TODO: VERIFY THIS IS CORRECT
            seq_origins = torch.cat(
                [dec_in_orig[:, data['valid_id'][0] == ag_id, :] for ag_id in agent_sequence[0]], dim=1
            )       # [B * sample_num, K, 2]
            print(f"{seq_origins, seq_origins.shape=}")

            seq_out = seq_out + seq_origins  # [B * sample_num, K, 2]

        # create out_in -> Partially from prediction, partially from dec_in (due to occlusion asynchronicity)
        from_pred_indices = (timestep_sequence == torch.max(timestep_sequence))  # [K]
        agents_from_pred = agent_sequence[0, from_pred_indices]         # [⊆K] <==> [k]

        print(f"{from_pred_indices=}")
        print(f"{agents_from_pred=}")

        from_dec_in_indices = (data['last_obs_timesteps'][0] == torch.max(timestep_sequence) + 1)  # [N]
        agents_from_dec_in = data['valid_id'][0, from_dec_in_indices]      # [⊆N] <==> [n]

        print(f"{from_dec_in_indices=}")
        print(f"{agents_from_dec_in=}")

        # print(f"{self.ar_detach=}")
        if self.ar_detach:
            out_in_from_pred = seq_out[:, from_pred_indices, :].clone().detach()          # [B * sample_num, k, 2]
            out_in_from_dec_in = dec_in_orig[:, from_dec_in_indices, :].clone().detach()  # [B * sample_num, n, 2]
        else:
            out_in_from_pred = seq_out[from_pred_indices, ...]              # [B * sample_num, k, 2]
            out_in_from_dec_in = dec_in_orig[from_dec_in_indices, ...]      # [B * sample_num, n, 2]

        # concatenate with latent z codes
        z_in_from_pred = torch.cat(
            [z_in_orig[:, (data['valid_id'][0] == agent_idx), :]
             for agent_idx in agents_from_pred], dim=1
        )  # [B * sample_num, k, nz]
        out_in_z_from_pred = torch.cat(
            [out_in_from_pred, z_in_from_pred], dim=-1
        )  # [B * sample_num, k, nz + 2]
        print(f"{out_in_from_pred.shape, z_in_from_pred.shape, out_in_z_from_pred.shape=}")

        z_in_from_dec_in = z_in_orig[:, from_dec_in_indices, :]  # [B * sample_num, n, nz]
        out_in_z_from_dec_in = torch.cat(
            [out_in_from_dec_in, z_in_from_dec_in], dim=-1
        )  # [B * sample_num, n, nz + 2]
        print(f"{out_in_from_dec_in.shape, z_in_from_dec_in.shape, out_in_z_from_dec_in.shape=}")

        # generate timestep tensor to extend timestep_sequence for next loop iteration
        next_timesteps = torch.full(
            [agents_from_pred.shape[0] + agents_from_dec_in.shape[0]], torch.max(timestep_sequence) + 1
        ).to(tf_in.device)  # [k + n]
        print(f"{next_timesteps=}")

        # update trajectory sequence
        dec_input_sequence = torch.cat(
            [dec_input_sequence, out_in_z_from_pred, out_in_z_from_dec_in], dim=1
        )  # [B * sample_num, K + k + n, nz + 2]     -> next loop: B == B + *~B + *~N
        print(f"{dec_input_sequence.shape=}")

        # update agent_sequence, timestep_sequence
        agent_sequence = torch.cat(
            [agent_sequence, agents_from_pred.unsqueeze(0), agents_from_dec_in.unsqueeze(0)], dim=1
        )  # [B, K + k + n]
        print(f"{agent_sequence=}")

        timestep_sequence = torch.cat(
            [timestep_sequence, next_timesteps], dim=0
        )  # [K + k + n]
        print(f"{timestep_sequence=}")

        return seq_out, dec_input_sequence, agent_sequence, timestep_sequence, attn_weights

    def decode_traj_ar(self, data, mode, context, z, sample_num, need_weights=False):
        # retrieving the most recent observation for each agent
        dec_in = data['last_obs_positions'].repeat(sample_num, 1, 1)     # [B * sample_num, N, 2]
        print(f"{dec_in.shape=}")

        print(f"{self.pred_mode=}")
        if self.pred_mode == "gauss":
            dist_params = torch.zeros([*dec_in.shape[:-1], self.out_module[0].N_params]).to(dec_in.device)
            dist_params[..., :self.forecast_dim] = dec_in
            dec_in = dist_params

        # print(f"AFTER GAUSS{dec_in, dec_in.shape=}")
        # print(f"{data['pre_timesteps']=}")
        # print(f"{data['last_observed_timesteps']=}")

        # z: [B * sample_num, N, nz]
        in_arr = [dec_in, z]
        # print(f"{self.input_type=}")
        # for key in self.input_type:
        #     if key == 'heading':
        #         # heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
        #         # in_arr.append(heading)
        #         raise NotImplementedError("if key == 'heading'")
        #     elif key == 'map':
        #         # map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
        #         # in_arr.append(map_enc)
        #         raise NotImplementedError("if key == 'map'")
        #     else:
        #         raise ValueError('wrong decode input type!')
        dec_in_z = torch.cat(in_arr, dim=-1)        # [B * sample_num, N, nz + 2]
        print(f"{dec_in_z.shape=}")


        print(f"{data['last_obs_timesteps'].shape=}")
        # TODO: CAREFULLY CHECK HOW THIS WORKS ON MULTIPLE AGENTS
        catch_up_timestep_sequence = data['last_obs_timesteps'][0, ...].detach().clone()                        # [N]
        starting_seq_indices = (catch_up_timestep_sequence == torch.min(catch_up_timestep_sequence, dim=0)[0])  # [N]

        print(f"{starting_seq_indices, starting_seq_indices.shape=}")

        timestep_sequence = catch_up_timestep_sequence[starting_seq_indices].to(dec_in.device)          # [⊆N] == [K]
        agent_sequence = data['valid_id'][:, starting_seq_indices].detach().clone()                     # [B, K]
        dec_input_sequence = dec_in_z[:, starting_seq_indices].detach().clone()                         # [B * sample_num, K, nz + 2]

        print(f"{timestep_sequence, timestep_sequence.shape=}")
        print(f"{agent_sequence, agent_sequence.shape=}")
        print(f"{dec_input_sequence.shape=}")

        # CATCH UP TO t_0
        # print("ENTERING BELIEF GENERATION SEQUENCE:")
        while not torch.all(catch_up_timestep_sequence == torch.zeros_like(catch_up_timestep_sequence)):
            print(f"\nBEGINNING LOOP: {catch_up_timestep_sequence=}")

            seq_out, dec_input_sequence, agent_sequence, timestep_sequence, attn_weights = self.decode_next_timestep(
                dec_in_orig=dec_in,                         # [B * sample_num, N, 2]
                z_in_orig=z,                                # [B * sample_num, N, nz]
                dec_input_sequence=dec_input_sequence,      # [B * sample_num, K, nz + 2]
                timestep_sequence=timestep_sequence,        # [K]
                agent_sequence=agent_sequence,              # [B, K]
                data=data,
                context=context,                            # [B * sample_num, O, model_dim]
                sample_num=sample_num
            )       # [B * sample_num, K, 2], [B * sample_num, K + k + n, nz + 2], [B, K + k + n], [K + k + n], Dict

            catch_up_timestep_sequence[catch_up_timestep_sequence == torch.min(catch_up_timestep_sequence)] += 1

            # print(f"{seq_out.shape=}")
            # print(f"{dec_input_sequence.shape=}")
            # print(f"{agent_sequence=}")
            # print(f"{timestep_sequence=}")
            # print(f"{catch_up_timestep_sequence=}")
            # print(f"{attn_weights.shape=}")
        # print("DONE WITH CATCHING UP")
        # PREDICT THE FUTURE

        for i in range(self.future_frames):
            # print(f"\nBEGINNING FUTURE LOOP: {i}")

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

        # print(f"DONE PREDICTING")
        # print(f"{seq_out.shape=}")
        # print(f"{agent_sequence, agent_sequence.shape=}")
        # print(f"{timestep_sequence, timestep_sequence.shape=}")

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
        print(f"{pred_timestep_sequence=}")
        print(f"{past_indices=}")
        print(f"{seq_out.shape=}")

        print(f"{pred_agent_sequence, pred_timestep_sequence.shape=}")
        print(f"{pred_timestep_sequence, pred_timestep_sequence.shape=}")
        print(f"{data['pred_identity_sequence'], data['pred_identity_sequence'].shape=}")
        print(f"{data['pred_timestep_sequence'], data['pred_timestep_sequence'].shape=}")

        if self.pred_type == 'scene_norm':
            scene_origs = data['scene_orig'].repeat(sample_num, 1).unsqueeze(1)         # [B * sample_num, 1, 2]
            seq_out += scene_origs                                                      # [B * sample_num, P, 2]
        else:
            raise NotImplementedError

        data[f'{mode}_dec_motion'] = seq_out                                    # [B * sample_num, P, 2]
        data[f'{mode}_dec_agents'] = pred_agent_sequence.repeat(sample_num, 1)  # [B * sample_num, P]
        data[f'{mode}_dec_past_mask'] = past_indices                            # [P]
        data[f'{mode}_dec_timesteps'] = pred_timestep_sequence                  # [P]
        if self.pred_mode == "gauss":
            raise NotImplementedError
            # data[f'{mode}_dec_mu'] = dec_motion[..., 0:self.forecast_dim]
            # data[f'{mode}_dec_sig'] = dec_motion[..., self.forecast_dim:2*self.forecast_dim]
            # data[f'{mode}_dec_rho'] = dec_motion[..., 2*self.forecast_dim:]
            # data[f'{mode}_dec_Sig'] = self.out_module[0].covariance_matrix(sig=data[f'{mode}_dec_sig'], rho=data[f'{mode}_dec_rho'])
        if need_weights:
            data['attn_weights'] = attn_weights

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data['context_enc'].repeat(sample_num, 1, 1)       # [B * sample_num, O, model_dim]

        print(f"{context.shape=}")

        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        print(f"{self.learn_prior, self.z_type=}")
        if self.learn_prior:
            h = data['agent_context'].repeat(sample_num, 1, 1)      # [B * sample_num, N, model_dim]
            p_z_params = self.p_z_net(h)                            # [B * sample_num, N, nz (*2 if self.z_type == gaussian)]
            print(f"{p_z_params.shape=}")
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

        print(f"{z is None=}")
        print(f"{mode=}")
        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()    # [B, N, nz]
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()         # [B * sample_num, N, nz]
            else:
                raise ValueError('Unknown Mode!')

        print(f"{z.shape=}")

        # print(f"{z.shape=}")
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
            'tf_cfg': cfg.get('tf_cfg', {}),
            'nz': cfg.nz,
            'z_type': cfg.get('z_type', 'gaussian'),
            'future_frames': cfg.future_frames,
            'past_frames': cfg.past_frames,
            'motion_dim': cfg.motion_dim,
            'forecast_dim': cfg.forecast_dim,
            'input_type': input_type,
            'fut_input_type': fut_input_type,
            'dec_input_type': dec_input_type,
            'pred_type': pred_type,
            'tf_nhead': cfg.tf_nhead,
            'tf_model_dim': cfg.tf_model_dim,
            'tf_ff_dim': cfg.tf_ff_dim,
            'tf_dropout': cfg.tf_dropout,
            'pos_concat': cfg.get('pos_concat', False),
            'ar_detach': cfg.get('ar_detach', True),
            'max_agent_len': cfg.get('max_agent_len', 128),
            'use_agent_enc': cfg.get('use_agent_enc', False),
            'agent_enc_learn': cfg.get('agent_enc_learn', False),
            'sn_out_type': cfg.get('sn_out_type', 'scene_norm'),
            'sn_out_heading': cfg.get('sn_out_heading', False),
            'vel_heading': cfg.get('vel_heading', False),
            'learn_prior': cfg.get('learn_prior', False),
            'use_map': cfg.get('use_map', False),
            'global_map_attention': cfg.get('global_map_attention', False),
            'context_encoder': cfg.context_encoder,
            'future_encoder': cfg.future_encoder,
            'future_decoder': cfg.future_decoder
        }
        self.scene_orig_all_past = cfg.get('scene_orig_all_past', False)
        self.use_map = cfg.get('use_map', False)
        self.global_map_attention = cfg.get('global_map_attention', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        # self.max_train_agent = cfg.get('max_train_agent', 100)        # this has been moved to preprocessor
        self.loss_cfg = cfg.get('loss_cfg')
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        self.z_type = cfg.get('z_type', 'gaussian')
        if self.z_type == 'discrete':
            ctx['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None

        # map encoder
        if self.use_map:
            self.map_encoder = MapEncoder(cfg.map_encoder)
            ctx['map_enc_dim'] = self.map_encoder.out_dim

        if self.global_map_attention:
            self.global_map_encoder = MapEncoder(cfg.global_map_encoder)
            ctx['global_map_enc_dim'] = self.global_map_encoder.out_dim

        # models
        self.context_encoder = ContextEncoder(ctx)
        self.future_encoder = FutureEncoder(ctx)
        self.future_decoder = FutureDecoder(ctx)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_data(self, data: dict) -> None:
        # NOTE: in our case, batch size B is always 1

        self.data = defaultdict(lambda: None)

        self.data['valid_id'] = data['identities'].detach().clone().to(self.device)     # [B, N]
        self.data['T_total'] = data['timesteps'].shape[0]
        self.data['agent_num'] = self.data['valid_id'].shape[0]
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

        self.data['scene_map'] = data['scene_map'].detach().clone().to(self.device)             # [B, C, H, W]
        self.data['occlusion_map'] = data['dist_transformed_occlusion_map'].detach().clone().to(self.device)    # [B, H, W]

        self.data['nlog_probability_occlusion_map'] = data['nlog_probability_occlusion_map'].detach().clone().to(self.device)   # [B, H, W]
        self.data['combined_map'] = torch.cat((self.data['scene_map'], self.data['occlusion_map'].unsqueeze(1)), dim=1)     # [B, C + 1, H, W]
        self.data['map_homography'] = data['map_homography'].detach().clone().to(self.device)        # [B, 3, 3]

        # print(f"{self.data['valid_id'], self.data['valid_id'].shape=}")
        # print(f"{self.data['timesteps'], self.data['timesteps'].shape=}")
        # print(f"{self.data['scene_orig'], self.data['scene_orig'].shape=}")
        # print(f"{self.data['obs_position_sequence'].shape, self.data['obs_position_sequence'].dtype=}")
        # print(f"{self.data['obs_velocity_sequence'].shape, self.data['obs_velocity_sequence'].dtype=}")
        # print(f"{self.data['obs_timestep_sequence'].shape, self.data['obs_timestep_sequence'].dtype=}")
        # print(f"{self.data['obs_identity_sequence'].shape, self.data['obs_identity_sequence'].dtype=}")
        # print(f"{self.data['last_obs_positions'].shape, self.data['last_obs_positions'].dtype=}")
        # print(f"{self.data['agent_mask'].shape, self.data['agent_mask'].dtype=}")
        # print(f"{self.data['pred_position_sequence'].shape, self.data['pred_position_sequence'].dtype=}")
        # print(f"{self.data['pred_velocity_sequence'].shape, self.data['pred_velocity_sequence'].dtype=}")
        # print(f"{self.data['pred_timestep_sequence'].shape, self.data['pred_timestep_sequence'].dtype=}")
        # print(f"{self.data['pred_identity_sequence'].shape, self.data['pred_identity_sequence'].dtype=}")
        # print(f"{self.data['scene_map'].shape, self.data['scene_map'].dtype=}")
        # print(f"{self.data['occlusion_map'].shape, self.data['occlusion_map'].dtype=}")
        # print(f"{self.data['nlog_probability_occlusion_map'].shape, self.data['nlog_probability_occlusion_map'].dtype=}")
        # print(f"{self.data['combined_map'].shape, self.data['combined_map'].dtype, self.data['combined_map'].device=}")
        # print(f"{self.data['map_homography'].shape, self.data['map_homography'].dtype=}")

        # REWORK LINE #####################################################
        # self.data['valid_id'] = data['valid_id'].detach().clone().to(self.device).to(int)       # [N]
        # self.data['T_total'] = len(data['timesteps'])                                           # int: T_total
        # self.data['batch_size'] = len(self.data['valid_id'])                                    # int: N
        # self.data['agent_num'] = len(self.data['valid_id'])                                     # int: N
        # self.data['timesteps'] = data['timesteps'].detach().clone().to(self.device)             # [T_total]
        # full_motion = data['trajectories'].\
        #     to(self.device).to(dtype=torch.float32).transpose(0, 1).contiguous()                # [T_total, N, 2]
        #
        # obs_mask = data['observation_mask'].\
        #     to(self.device).to(dtype=torch.bool).transpose(0, 1).contiguous()                   # [T_total, N]
        #
        # print(f"{full_motion.shape=}")
        # print(f"{obs_mask.shape=}")
        #
        # last_observed_timestep_indices = torch.stack(
        #     [mask.nonzero().flatten()[-1] for mask in data['observation_mask']]
        # ).to(self.device)                                                                       # [N]
        # last_observed_pos = full_motion[last_observed_timestep_indices, torch.arange(full_motion.size(1))]  # [N, 2]
        # timesteps_to_predict = torch.stack(
        #     [torch.cat(
        #         (torch.full([int(last_obs + 1)], False),
        #          torch.full([int(self.data['T_total'] - (last_obs + 1))], True))
        #     ) for last_obs in last_observed_timestep_indices], dim=0
        # ).transpose(0, 1)                                                           # [T_total, N]
        #
        # full_agent_mask = self.data['valid_id'].repeat(self.data['T_total'], 1)                         # [T_total, N]
        # full_timestep_mask = self.data['timesteps'].view(-1, 1).repeat(1, self.data['agent_num'])       # [T_total, N]
        #
        # self.data['last_observed_timesteps'] = torch.stack(
        #     [self.data['timesteps'][last_obs] for last_obs in last_observed_timestep_indices], dim=0
        # ).to(self.device)        # [N]
        #
        # print(f"{timesteps_to_predict=}")
        # print(f"{full_agent_mask=}")
        # print(f"{full_timestep_mask=}")
        #
        # # define the scene origin
        # if self.scene_orig_all_past:
        #     scene_orig = full_motion[obs_mask].mean(dim=0).contiguous()           # [2]
        # else:
        #     scene_orig = last_observed_pos.mean(dim=0).contiguous()               # [2]
        #
        # self.data['scene_orig'] = scene_orig.to(self.device)
        #
        # full_motion_scene_norm = full_motion - scene_orig
        #
        # # print(f"{torch.max(torch.linalg.norm(full_motion_scene_norm, dim=-1))=}")
        #
        # # create past and future tensors
        # self.data['pre_sequence'] = full_motion[obs_mask, ...].clone().detach()          # [O, 2], where O is equal to sum(obs_mask)
        # self.data['pre_sequence_scene_norm'] = full_motion_scene_norm[obs_mask, ...].clone().detach()        # [O, 2]
        # self.data['pre_agents'] = full_agent_mask[obs_mask].clone().detach()                                 # [O]
        # self.data['pre_timesteps'] = full_timestep_mask[obs_mask].clone().detach()                           # [O]
        #
        # self.data['fut_sequence'] = full_motion[timesteps_to_predict, ...].clone().detach()          # [P, 2], where P equals sum(timesteps_to_predict)
        # self.data['fut_sequence_scene_norm'] = full_motion_scene_norm[timesteps_to_predict, ...].clone().detach()        # [P, 2]
        # self.data['fut_agents'] = full_agent_mask[timesteps_to_predict].clone().detach()                                 # [P]
        # self.data['fut_timesteps'] = full_timestep_mask[timesteps_to_predict].clone().detach()                           # [P]
        #
        # pre_vel_seq = torch.full_like(full_motion, float('nan')).to(self.device)                    # [T_total, N, 2]
        # for agent_i in range(self.data['agent_num']):
        #     obs_indices = torch.nonzero(obs_mask[:, agent_i]).squeeze()
        #
        #     # no information about velocity for the first observation: assume zero velocity
        #     pre_vel_seq[obs_indices[0], agent_i, :] = 0
        #
        #     # impute velocities for subsequent timesteps by position differentiation
        #     # (normalizing by timestep gap for cases of occlusion)
        #     motion_diff = full_motion[obs_indices[1:], agent_i, :] - full_motion[obs_indices[:-1], agent_i, :]
        #     pre_vel_seq[obs_indices[1:], agent_i, :] = motion_diff / (obs_indices[1:] - obs_indices[:-1]).unsqueeze(1)
        #
        # self.data['pre_vel_seq'] = pre_vel_seq[obs_mask, ...]                       # [O, 2]
        # assert not torch.any(torch.isnan(self.data['pre_vel_seq']))
        #
        # full_vel = torch.full_like(full_motion, float('nan')).to(self.device)                        # [T_total, N, 2]
        # full_vel[1:, ...] = full_motion[1:, ...] - full_motion[:-1, ...]
        # self.data['fut_vel_seq'] = full_vel[timesteps_to_predict, ...]              # [P, 2]
        # assert not torch.any(torch.isnan(self.data['fut_vel_seq']))
        #
        # # create tensors for the last observed position of each agent
        # cur_motion = full_motion[last_observed_timestep_indices, torch.arange(self.data['agent_num'])].unsqueeze(0)      # [1, N, 2]
        # self.data['cur_motion'] = cur_motion.to(self.device)
        # cur_motion_scene_norm = full_motion_scene_norm[last_observed_timestep_indices, torch.arange(self.data['agent_num'])].unsqueeze(0)
        # self.data['cur_motion_scene_norm'] = cur_motion_scene_norm.to(self.device)                   # [1, N, 2]
        #
        # print(f"{self.data['cur_motion'], self.data['cur_motion_scene_norm']=}")
        # raise NotImplementedError

        # # NOTE: heading does not follow the occlusion pattern
        # if in_data['heading'] is not None:
        #     self.data['heading'] = torch.tensor(in_data['heading']).float().to(self.device)      # [N]
        # # NOTE: heading does not follow the occlusion pattern
        # if in_data['heading'] is not None:
        #     self.data['heading_vec'] = torch.stack([torch.cos(self.data['heading']), torch.sin(self.data['heading'])], dim=-1)      # [N, 2]

        # # NOTE: agent maps should not be used in the occlusion case (at least not without important modification wrt
        # # effective data observedness)
        # # agent maps
        # if self.use_map:
        #     scene_map = data['scene_map']
        #     scene_points = np.stack(in_data['pre_motion_3D'])[:, -1] * data['traj_scale']
        #     if self.map_global_rot:
        #         patch_size = [50, 50, 50, 50]
        #         rot = theta.repeat(self.data['agent_num']).cpu().numpy() * (180 / np.pi)
        #     else:
        #         patch_size = [50, 10, 50, 90]
        #         rot = -np.array(in_data['heading']) * (180 / np.pi)
        #     self.data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot).to(self.device)      # [N, 3, 100, 100]

        # # global scene map
        # # self.data['global_map'] = torch.from_numpy(data['scene_map'].data).to(self.device)
        # self.data['scene_map'] = data['scene_map']
        # self.data['global_map'] = torch.from_numpy(data['scene_map'].data.transpose(0, 2, 1)).to(self.device)
        # # occlusion map
        # self.data['occlusion_map'] = data['occlusion_map'].detach().clone().to(self.device)
        # self.data['min_log_p_occl_map'] = data['min_log_p_occl_map'].detach().clone().to(self.device)
        #
        # self.data['combined_map'] = torch.cat((self.data['global_map'], self.data['occlusion_map'].unsqueeze(0))).to(torch.float32)
        # print(f"{self.data['combined_map'], self.data['combined_map'].shape=}")
        #
        # self.data['dt_occlusion_map'] = data['dt_occlusion_map'].detach().clone().to(self.device)
        # self.data['p_occl_map'] = data['p_occl_map'].detach().clone().to(self.device)
        #
        # mask = torch.zeros([self.data['agent_num'], self.data['agent_num']]).to(self.device)
        # self.data['agent_mask'] = mask          # [N, N]
        #
        # self.visualize_data_dict()

    # def visualize_data_dict(self, show: bool = True):
    #     # TODO: there are a few things to fix in this data visualization function
    #
    #     [print(f"{k}: {type(v)}") for k, v in self.data.items()]
    #     print()
    #
    #     # High level metadata
    #     print(f"{self.data['T_total']=}")
    #     print(f"{self.data['batch_size']=}")
    #     print(f"{self.data['agent_num']=}")
    #     print()
    #
    #     # Multi-Agent sequence relevant data
    #     print(f"{self.data['timesteps']=}")
    #     print(f"{self.data['valid_id']=}")
    #     print()
    #
    #     # Geometric data
    #     print(f"{self.data['scene_orig']=}")
    #     print(f"{self.data['scene_map']=}")
    #
    #     # Trajectory data
    #     # print(f"{self.data['pre_sequence']=}")
    #     # print(f"{self.data['pre_sequence_scene_norm']=}")
    #     # print(f"{self.data['pre_agents']=}")
    #     # print(f"{self.data['pre_timesteps']=}")
    #     # print(f"{self.data['pre_vel_seq']=}")
    #     # print()
    #     # print(f"{self.data['fut_sequence']=}")
    #     # print(f"{self.data['fut_sequence_scene_norm']=}")
    #     # print(f"{self.data['fut_agents']=}")
    #     # print(f"{self.data['fut_timesteps']=}")
    #     # print(f"{self.data['fut_vel_seq']=}")
    #     # print()
    #
    #     # extra Traj relevant data
    #     print(f"{self.data['last_observed_timesteps']=}")
    #     print(f"{self.data['cur_motion']=}")
    #     print(f"{self.data['cur_motion_scene_norm']=}")
    #     print(f"{self.data['agent_mask']=}")
    #
    #     fig = plt.figure()
    #     ax0 = fig.add_subplot(131, projection='3d')
    #     ax1 = fig.add_subplot(132, projection='3d')
    #     ax2 = fig.add_subplot(133)
    #
    #     scene_map = self.data['scene_map']
    #
    #     if scene_map is not None:
    #         ax0.set_xlim(0., scene_map.get_map_dimensions()[0])
    #         ax0.set_ylim(scene_map.get_map_dimensions()[1], 0.)
    #     ax0.view_init(90, -90)
    #
    #     scene_orig = scene_map.to_map_points(self.data['scene_orig'].detach().cpu().numpy())
    #     ax0.scatter(scene_orig[0], scene_orig[1], 0.0, marker='D', s=30, c='red', label='scene_orig')
    #
    #     if scene_map is not None:
    #         ax1.set_xlim(0. - scene_orig[0], scene_map.get_map_dimensions()[0] - scene_orig[0])
    #         ax1.set_ylim(scene_map.get_map_dimensions()[1] - scene_orig[1], 0. - scene_orig[1])
    #     ax1.view_init(90, -90)
    #
    #     valid_ids = self.data['valid_id'].detach().cpu().numpy()
    #
    #     pre_timesteps = self.data['pre_timesteps'].detach().cpu().numpy()
    #     pre_agents = self.data['pre_agents'].detach().cpu().numpy()
    #     pre_seq = scene_map.to_map_points(self.data['pre_sequence'].detach().cpu().numpy())
    #     pre_seq_scene_norm = scene_map.to_map_points(self.data['pre_sequence_scene_norm'].detach().cpu().numpy())
    #
    #     fut_timesteps = self.data['fut_timesteps'].detach().cpu().numpy()
    #     fut_agents = self.data['fut_agents'].detach().cpu().numpy()
    #     fut_seq = scene_map.to_map_points(self.data['fut_sequence'].detach().cpu().numpy())
    #     fut_seq_scene_norm = scene_map.to_map_points(self.data['fut_sequence_scene_norm'].detach().cpu().numpy())
    #
    #     cmap = plt.cm.get_cmap('hsv', len(valid_ids))
    #
    #     for i, agent in enumerate(valid_ids):
    #         pre_mask = (pre_agents == agent)
    #         ag_pre_seq = pre_seq[pre_mask]
    #         ag_pre_seq_scene_norm = pre_seq_scene_norm[pre_mask]
    #         ag_pre_timesteps = pre_timesteps[pre_mask]
    #         fut_mask = (fut_agents == agent)
    #         ag_fut_seq = fut_seq[fut_mask]
    #         ag_fut_seq_scene_norm = fut_seq_scene_norm[fut_mask]
    #         ag_fut_timesteps = fut_timesteps[fut_mask]
    #
    #         marker_line, stem_lines, base_line = ax0.stem(
    #             ag_pre_seq[..., 0], ag_pre_seq[..., 1], ag_pre_timesteps,
    #             linefmt='grey'
    #         )
    #         marker_line.set(markeredgecolor=cmap(i), markerfacecolor=cmap(i), alpha=0.7, markersize=5, marker='X')
    #         stem_lines.set(alpha=0.0)
    #         base_line.set(alpha=0.6, c=cmap(i))
    #
    #         marker_line, stem_lines, base_line = ax0.stem(
    #             ag_fut_seq[..., 0], ag_fut_seq[..., 1], ag_fut_timesteps,
    #             linefmt='grey'
    #         )
    #         marker_line.set(markeredgecolor=cmap(i), markerfacecolor=cmap(i), alpha=0.7, markersize=5)
    #         stem_lines.set(alpha=0.0)
    #         base_line.set(alpha=0.5, c=cmap(i))
    #
    #         marker_line, stem_lines, base_line = ax1.stem(
    #             ag_pre_seq_scene_norm[..., 0], ag_pre_seq_scene_norm[..., 1], ag_pre_timesteps,
    #             linefmt='grey'
    #         )
    #         marker_line.set(markeredgecolor=cmap(i), markerfacecolor=cmap(i), alpha=0.7, markersize=5, marker='X')
    #         stem_lines.set(alpha=0.0)
    #         base_line.set(alpha=0.6, c=cmap(i))
    #
    #         marker_line, stem_lines, base_line = ax1.stem(
    #             ag_fut_seq_scene_norm[..., 0], ag_fut_seq_scene_norm[..., 1], ag_fut_timesteps,
    #             linefmt='grey'
    #         )
    #         marker_line.set(markeredgecolor=cmap(i), markerfacecolor=cmap(i), alpha=0.7, markersize=5)
    #         stem_lines.set(alpha=0.0)
    #         base_line.set(alpha=0.5, c=cmap(i))
    #
    #     if scene_map is not None:
    #         ax2.set_xlim(0., scene_map.get_map_dimensions()[0])
    #         ax2.set_ylim(scene_map.get_map_dimensions()[1], 0.)
    #         ax2.imshow(scene_map.as_image())
    #
    #     fig_2, axes_2 = plt.subplots(1, 6)
    #     divider_1 = make_axes_locatable(axes_2[4])
    #     divider_2 = make_axes_locatable(axes_2[5])
    #     cax_1 = divider_1.append_axes('right', size='5%', pad=0.05)
    #     cax_2 = divider_2.append_axes('right', size='5%', pad=0.05)
    #
    #     for dim in range(self.data['combined_map'].shape[0]):
    #         img = self.data['combined_map'][dim, ...].cpu().numpy()
    #         axes_2[dim].imshow(img)
    #
    #     dist_t_occl_map = self.data['dt_occlusion_map'].cpu().numpy()
    #     im = axes_2[4].imshow(dist_t_occl_map)
    #     fig_2.colorbar(im, cax=cax_1, orientation='vertical')
    #
    #     p_occl_map = self.data['p_occl_map'].cpu().numpy()
    #     im2 = axes_2[5].imshow(p_occl_map, cmap='Greys')
    #     fig_2.colorbar(im2, cax=cax_2, orientation='vertical')
    #
    #     if show:
    #         plt.show()

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self):
        if self.global_map_attention:
            self.data['global_map_encoding'] = self.global_map_encoder(self.data['combined_map'])
            # print(f"{self.data['global_map_encoding'].shape=}")

        if self.use_map:
            # self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
            # print(f"{self.data['combined_map'], self.data['combined_map'].shape=}")
            # self.data['map_encoding'] = self.map_encoder(self.data['combined_map'].unsqueeze(0)).squeeze()
            # print(f"{self.data['map_encoding'], self.data['map_encoding'].shape=}")
            raise NotImplementedError
        print(f"\nCALLING:  CONTEXT ENCODER\n")
        self.context_encoder(self.data)
        print(f"\nCALLING:  FUTURE ENCODER\n")
        self.future_encoder(self.data)
        print(f"\nCALLING:  FUTURE DECODER\n")
        self.future_decoder(self.data, mode='train', autoregress=self.ar_train)
        if self.compute_sample:
            # print(f"\nCALLING:  INFERENCE\n")
            self.inference(sample_num=self.loss_cfg['sample']['k'])
        return self.data

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        if self.use_map and self.data['map_enc'] is None:
            # self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
            raise NotImplementedError("self.use_map and self.data['map_enc'] is None")
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
