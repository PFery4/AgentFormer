import matplotlib.axes
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Union, Tuple, Optional
import model.decoder_out_submodels as decoder_out_submodels
from model.common.mlp import MLP
from model.agentformer_loss import loss_func
from model.common.dist import Normal, Categorical
from model.agentformer_lib import AgentFormerEncoderLayer, AgentFormerDecoderLayer, AgentFormerDecoder, AgentFormerEncoder
from model.map_encoder import MapEncoder
from utils.torch import rotation_2d_torch, ExpParamAnnealer
from utils.utils import initialize_weights
from typing import List


def generate_ar_mask(sz: int, agent_num: int, agent_mask: torch.Tensor):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask


def generate_ar_mask_with_variable_agents_per_timestep(timestep_sequence: torch.Tensor) -> torch.Tensor:
    stop_at = torch.argmax(timestep_sequence)
    mask = torch.zeros(timestep_sequence.shape[0], timestep_sequence.shape[0])
    for idx in range(stop_at):
        mask_seq = (timestep_sequence > timestep_sequence[idx])
        mask[idx, mask_seq] = float('-inf')
    return mask


def generate_mask(tgt_sz: int, src_sz: int) -> torch.Tensor:
    """
    This mask generation process is responsible for the functionality discussed in the paragraph
    "Encoding Agent Connectivity" in the original AgentFormer paper. The function presented here is modified such
    that all agents are connected to one another (or, in other words, the distance threshold value eta is infinite).
    The resulting mask is a tensor full of zero's,
    shaped like the attention matrix QK^T performed in the agent_aware_attention function
    If you need to apply some distance thresholding for your own experiments, you will need to change
    the implementation of this function accordingly.
    """
    return torch.zeros(tgt_sz, src_sz)


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
        timestep_window = torch.arange(*timestep_window, dtype=torch.float).unsqueeze(1)       # (t_range, 1)
        self.register_buffer('timestep_window', timestep_window)
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_enc_table()
        self.register_buffer('pe', pe)

    def build_enc_table(self) -> torch.Tensor:
        # shape (t_range, d_model)
        pe = torch.zeros(len(self.timestep_window), self.d_model)

        # shape (d_model//2)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(self.timestep_window * div_term)
        pe[:, 1::2] = torch.cos(self.timestep_window * div_term)
        return pe       # (t_range, d_model)

    def time_encode(self, sequence_timesteps: torch.Tensor) -> torch.Tensor:
        # sequence_timesteps: (T_total)
        # out: (T_total, self.d_model)
        return torch.cat([self.pe[(self.timestep_window == t).squeeze(), ...] for t in sequence_timesteps], dim=0)

    def forward(self, x: torch.Tensor, time_tensor: torch.Tensor):
        # x: (T, batch_size, model_dim)
        # time_tensor: (T)
        pos_enc = self.time_encode(time_tensor).unsqueeze(1)     # (T, 1, self.d_model)
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
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.motion_dim = ctx['motion_dim']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.input_type = ctx['input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.vel_heading = ctx['vel_heading']
        ctx['context_dim'] = self.model_dim
        in_dim = self.motion_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - self.motion_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        encoder_layers = AgentFormerEncoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout, concat=ctx['pos_concat'])

    def forward(self, data):
        # traj_in = []
        seq_in = []
        for key in self.input_type:
            if key == 'pos':
                # traj_in.append(data['pre_motion'])
                seq_in.append(data['pre_sequence'])
            elif key == 'vel':
                # vel = data['pre_vel']
                vel_seq = data['pre_vel_seq']
                if len(self.input_type) > 1:
                    # vel = torch.cat([vel[[0]], vel], dim=0)          # Imputation
                    vel_seq = torch.cat([                            # Imputation
                        vel_seq[:data['agent_num']],
                        vel_seq
                    ], dim=0)

                if self.vel_heading:
                    # vel = rotation_2d_torch(vel, -data['heading'])[0]
                    raise NotImplementedError("if self.vel_heading")
                # traj_in.append(vel)
                seq_in.append(vel_seq)
            elif key == 'norm':
                # traj_in.append(data['pre_motion_norm'])
                raise NotImplementedError("if key == 'norm'")
            elif key == 'scene_norm':
                # traj_in.append(data['pre_motion_scene_norm'])
                seq_in.append(data['pre_sequence_scene_norm'])
            elif key == 'heading':
                # hv = data['heading_vec'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                # traj_in.append(hv)
                raise NotImplementedError("if key == 'heading'")
            elif key == 'map':
                # map_enc = data['map_enc'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                # traj_in.append(map_enc)
                raise NotImplementedError("if key == 'map'")
            else:
                raise ValueError('unknown input_type!')
        # traj_in = torch.cat(traj_in, dim=-1)                    # (T, N, Features)
        seq_in = torch.cat(seq_in, dim=-1)                        # (O, Features)

        # tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)      # (T * N, 1, model_dim)
        tf_seq_in = self.input_fc(seq_in).view(-1, 1, self.model_dim)               # (O, 1, model_dim)

        tf_in_pos = self.pos_encoder(
            x=tf_seq_in,                            # (O, 1, model_dim)
            time_tensor=data['pre_timesteps']       # (O)
        )                                           # (O, 1, model_dim)

        # src_agent_mask = data['agent_mask'].clone()         # (N, N)

        src_mask = generate_mask(data['pre_timesteps'].shape[0], data['pre_timesteps'].shape[0]).to(tf_seq_in.device)        # (O, O)

        # print(f"{tf_in_pos.shape=}")
        data['context_enc'] = self.tf_encoder(
            src=tf_in_pos,                          # (O, 1, model_dim)
            src_identities=data['pre_agents'],      # (O)
            mask=src_mask                           # (O, O)
        )                                           # (O, 1, model_dim)
        # print(f"{data['context_enc'].shape=}")
        print(f"ENCODER ATTENTION: DONE!")

        # compute per agent context
        # print(f"{self.pooling=}")
        if self.pooling == 'mean':
            data['agent_context'] = torch.cat(
                [torch.mean(data['context_enc'][data['pre_agents'] == ag_id, ...], dim=0)
                 for ag_id in torch.unique(data['pre_agents'])], dim=0
            )       # (N, model_dim)
        else:
            data['agent_context'] = torch.cat(
                [torch.max(data['context_enc'][data['pre_agents'] == ag_id, :], dim=0)[0]
                 for ag_id in torch.unique(data['pre_agents'])], dim=0
            )       # (N, model_dim)


class FutureEncoder(nn.Module):
    """ Future Encoder """
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.input_type = ctx['fut_input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.vel_heading = ctx['vel_heading']
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout, concat=ctx['pos_concat'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data, reparam=True):
        # traj_in = []
        seq_in = []
        for key in self.input_type:
            if key == 'pos':
                # traj_in.append(data['fut_motion'])
                print(f"{data['fut_sequence'].shape=}")
                seq_in.append(data['fut_sequence'])
            elif key == 'vel':
                # vel = data['fut_vel']
                vel_seq = data['fut_vel_seq']
                print(f"{data['fut_vel_seq'].shape=}")
                if self.vel_heading:
                    # vel = rotation_2d_torch(vel, -data['heading'])[0]
                    raise NotImplementedError("if self.vel_heading")
                # traj_in.append(vel)
                seq_in.append(vel_seq)
            elif key == 'norm':
                # traj_in.append(data['fut_motion_norm'])
                raise NotImplementedError("if key == 'norm'")
            elif key == 'scene_norm':
                # traj_in.append(data['fut_motion_scene_norm'])
                print(f"{data['fut_sequence_scene_norm'].shape=}")
                seq_in.append(data['fut_sequence_scene_norm'])
            elif key == 'heading':
                # hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                # traj_in.append(hv)
                raise NotImplementedError("if key == 'heading'")
            elif key == 'map':
                # map_enc = data['map_enc'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                # traj_in.append(map_enc)
                raise NotImplementedError("if key == 'map'")
            else:
                raise ValueError('unknown input_type!')
        # traj_in = torch.cat(traj_in, dim=-1)
        seq_in = torch.cat(seq_in, dim=-1)      # (P, Features)

        print(f"{seq_in.shape=}")

        # tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        tf_seq_in = self.input_fc(seq_in).view(-1, 1, self.model_dim)       # (P, 1, model_dim)

        print(f"{tf_seq_in.shape=}")
        print(f"{data['pre_timesteps']=}")
        print(f"{data['fut_timesteps'], data['fut_timesteps'].shape=}")

        tf_in_pos = self.pos_encoder(
            x=tf_seq_in,                            # (P, 1, model_dim)
            time_tensor=data['fut_timesteps']       # (P)
        )                                           # (P, 1, model_dim)
        # # WIP CODE
        # fig, ax = plt.subplots()
        # print(f"FUTURE ENCODER")
        # self.pos_encoder.plot_positional_window(ax=ax, num_t=tf_in.shape[0]//data['agent_num'])
        # plt.show()
        # # WIP CODE

        # mem_agent_mask = data['agent_mask'].clone()
        # tgt_agent_mask = data['agent_mask'].clone()
        mem_mask = generate_mask(data['fut_timesteps'].shape[0], data['pre_timesteps'].shape[0]).to(tf_seq_in.device)
        tgt_mask = generate_mask(data['fut_timesteps'].shape[0], data['fut_timesteps'].shape[0]).to(tf_seq_in.device)

        tf_out, _ = self.tf_decoder(
            tgt=tf_in_pos,                          # (P, 1, model_dim)
            memory=data['context_enc'],             # (O, 1, model_dim)
            tgt_identities=data['fut_agents'],      # (P)
            mem_identities=data['pre_agents'],      # (O)
            tgt_mask=tgt_mask,                      # (P, P)
            memory_mask=mem_mask,                   # (P, O)
        )                                           # (P, 1, model_dim)
        print(f"DECODER ATTENTION: DONE!")

        if self.pooling == 'mean':
            h = torch.cat(
                [torch.mean(tf_out[data['fut_agents'] == ag_id, :], dim=0)
                 for ag_id in torch.unique(data['fut_agents'])], dim=0
            )       # (N, model_dim)
        else:
            h = torch.cat(
                [torch.max(tf_out[data['fut_agents'] == ag_id, :], dim=0)[0]
                 for ag_id in torch.unique(data['fut_agents'])], dim=0
            )       # (N, model_dim)
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)
        if self.z_type == 'gaussian':
            data['q_z_dist'] = Normal(params=q_z_params)
        else:
            data['q_z_dist'] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data['q_z_samp'] = data['q_z_dist'].rsample()


class FutureDecoder(nn.Module):
    """ Future Decoder """
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ar_detach = ctx['ar_detach']
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.pred_scale = cfg.get('pred_scale', 1.0)
        self.pred_type = ctx['pred_type']
        self.pred_mode = cfg.get('mode', 'point')
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
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.pos_offset = cfg.get('pos_offset', False)
        self.learn_prior = ctx['learn_prior']

        # sanity check
        assert self.pred_mode in ["point", "gauss"]

        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        if self.pred_mode == "gauss":            # TODO: Maybe this can be integrated in a better way
            in_dim += 3     # adding three extra input dimensions: for the variance terms and correlation term of the distribution
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

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
            dec_in_orig: torch.Tensor,
            z_in_orig: torch.Tensor,
            dec_input_sequence: torch.Tensor,
            timestep_sequence: torch.Tensor,
            agent_sequence: torch.Tensor,
            data: dict,
            context: torch.Tensor,
            sample_num: int
    ):

        # Embed input sequence in high-dim space
        tf_in = self.input_fc(
            dec_input_sequence.view(-1, dec_input_sequence.shape[-1])  # (B * n_sample, nz + 2)
        ).view(dec_input_sequence.shape[0], -1, self.model_dim)  # (B, n_sample, model_dim)
        print(f"{tf_in.shape=}")

        # Temporal encoding
        tf_in_pos = self.pos_encoder(
            x=tf_in,  # (B, n_sample, model_dim)
            time_tensor=timestep_sequence  # (B)
        ).view(timestep_sequence.shape[0], sample_num, self.model_dim)  # (B, n_sample, model_dim)

        # Generate attention masks (tgt_mask ensures proper autoregressive attention, such that predictions which
        # were originally made at loop iteration nr k cannot attend from sequence elements which have been added
        # at loop iterations >k)
        tgt_mask = generate_ar_mask_with_variable_agents_per_timestep(timestep_sequence=timestep_sequence).to(
            tf_in.device)
        mem_mask = generate_mask(timestep_sequence.shape[0], context.shape[0]).to(tf_in.device)

        # Go through the attention mechanism
        tf_out, attn_weights = self.tf_decoder(
            tgt=tf_in_pos,  # (B, n_sample, model_dim)
            memory=context,  # (O, n_sample, model_dim)
            tgt_identities=agent_sequence,  # (B)
            mem_identities=data['pre_agents'],  # (O)
            tgt_mask=tgt_mask,  # (B, B)
            memory_mask=mem_mask  # (B, O)
        )  # (B, n_sample, model_dim)
        print(f"{tf_out.shape=}")

        # Map back to physical space
        seq_out = self.out_module(
            tf_out.view(-1, tf_out.shape[-1])  # (B * n_sample, model_dim)
        ).view(*tf_out.shape[:2], -1)  # (B, n_sample, 2)
        print(f"{seq_out.shape=}")

        # this mode is used to have the model predict offsets from the last observed position of agents, instead of
        # absolute coordinates in space
        # print(f"{self.sn_out_type=}")
        if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
            norm_motion = seq_out  # (B, n_sample, 2)
            print(f"{norm_motion.shape=}")

            if self.sn_out_type == 'vel':
                raise NotImplementedError("self.sn_out_type == 'vel'")

            print(f"{agent_sequence=}")
            print(f"{data['valid_id']=}")
            print(f"{dec_in_orig.shape=}")

            # defining origins for each element in the sequence, using agent_sequence, dec_in and data['valid_id']
            seq_origins = torch.cat(
                [dec_in_orig[(data['valid_id'] == ag_id), ...] for ag_id in agent_sequence], dim=0
            )  # (B, n_sample, 2)
            print(f"{seq_origins.shape=}")
            seq_out = norm_motion + seq_origins  # (B, n_sample, 2)
        print(f"{seq_out.shape=}")

        # create out_in -> Partially from prediction, partially from dec_in (due to occlusion asynchronicity)
        from_pred_indices = (timestep_sequence == torch.max(timestep_sequence))  # (B)
        agents_from_pred = agent_sequence[from_pred_indices]  # (*~B)     (subset of B)

        print(f"{from_pred_indices=}")
        print(f"{agents_from_pred=}")

        from_dec_in_indices = (data['last_observed_timesteps'] == torch.max(timestep_sequence) + 1)  # (N)
        agents_from_dec_in = data['valid_id'][from_dec_in_indices]  # (*~N)

        print(f"{from_dec_in_indices=}")
        print(f"{agents_from_dec_in=}")

        # print(f"{self.ar_detach=}")
        if self.ar_detach:
            out_in_from_pred = seq_out[from_pred_indices, ...].clone().detach()  # (*~B, n_sample, 2)
            out_in_from_dec_in = dec_in_orig[from_dec_in_indices, ...].clone().detach()  # (*~N, n_sample, 2)
        else:
            out_in_from_pred = seq_out[from_pred_indices, ...]  # (*~B, n_sample, 2)
            out_in_from_dec_in = dec_in_orig[from_dec_in_indices, ...]  # (*~N, n_sample, 2)

        # concatenate with latent z codes
        z_in_from_pred = torch.cat(
            [z_in_orig[(data['valid_id'] == agent_idx), ...]
             for agent_idx in agents_from_pred], dim=0
        )  # (*~B, n_sample, nz)
        out_in_z_from_pred = torch.cat(
            [out_in_from_pred, z_in_from_pred], dim=-1
        )  # (*~B, n_sample, nz + 2)
        print(f"{out_in_from_pred.shape, z_in_from_pred.shape, out_in_z_from_pred.shape=}")

        z_in_from_dec_in = z_in_orig[from_dec_in_indices, ...]  # (*~N, n_sample, nz)
        out_in_z_from_dec_in = torch.cat(
            [out_in_from_dec_in, z_in_from_dec_in], dim=-1
        )  # (*~N, n_sample, nz + 2)
        print(f"{out_in_from_dec_in.shape, z_in_from_dec_in.shape, out_in_z_from_dec_in.shape=}")

        # generate timestep tensor to extend timestep_sequence for next loop iteration
        next_timesteps = torch.full(
            [agents_from_pred.shape[0] + agents_from_dec_in.shape[0]], torch.max(timestep_sequence) + 1
        ).to(tf_in.device)  # (*~B + *~N)
        print(f"{next_timesteps=}")

        # update trajectory sequence
        dec_input_sequence = torch.cat(
            [dec_input_sequence, out_in_z_from_pred, out_in_z_from_dec_in], dim=0
        )  # (B + *~B + *~N, n_sample, nz + 2)     -> next loop: B == B + *~B + *~N
        print(f"{dec_input_sequence.shape=}")

        # update agent_sequence, timestep_sequence
        agent_sequence = torch.cat(
            [agent_sequence, agents_from_pred, agents_from_dec_in], dim=0
        )  # (B + *~B + *~N)
        print(f"{agent_sequence=}")

        timestep_sequence = torch.cat(
            [timestep_sequence, next_timesteps], dim=0
        )  # (B + *~B + *~N)
        print(f"{timestep_sequence=}")

        return dec_input_sequence, agent_sequence, timestep_sequence

    def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=False):
        agent_num = data['agent_num']
        # retrieving the most recent observation for each agent
        # print(f"{self.pred_type=}")
        if self.pred_type == 'vel':
            # dec_in = pre_vel[[-1]]
            # print(f"{pre_vel[[-1]].shape=}")
            raise NotImplementedError
        elif self.pred_type == 'pos':
            # dec_in = pre_motion[[-1]]
            # print(f"{pre_motion[[-1]].shape=}")
            raise NotImplementedError
        elif self.pred_type == 'scene_norm':
            # dec_in = pre_motion_scene_norm[[-1]]  # (1, sample_num * N, 2)
            # print(f"{pre_motion_scene_norm.shape=}")
            # print(f"{pre_motion_scene_norm[[0]], pre_motion_scene_norm[[0]].shape=}")
            # print(f"{pre_motion_scene_norm[[-1]], pre_motion_scene_norm[[-1]].shape=}")
            # print(f"{data['cur_motion_scene_norm'], data['cur_motion_scene_norm'].shape=}")
            dec_in = data['cur_motion_scene_norm'].repeat_interleave(sample_num, dim=1)     # (1, sample_num * N, 2)
        else:
            # dec_in = torch.zeros_like(pre_motion[[-1]])
            raise NotImplementedError
        # print(f"{dec_in, dec_in.shape=}")
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])      # (N, sample_num, 2)
        # print(f"{dec_in, dec_in.shape=}")
        # print(f"{dec_in[:, 0, :]=}")
        # print(f"{self.pred_mode=}")
        if self.pred_mode == "gauss":
            dist_params = torch.zeros([*dec_in.shape[:-1], self.out_module[0].N_params]).to(dec_in.device)
            dist_params[..., :self.forecast_dim] = dec_in
            dec_in = dist_params

        # print(f"AFTER GAUSS{dec_in, dec_in.shape=}")
        # print(f"{data['pre_timesteps']=}")
        # print(f"{data['last_observed_timesteps']=}")

        z_in = z.view(-1, sample_num, z.shape[-1])          # (N, sample_num, nz)
        in_arr = [dec_in, z_in]
        for key in self.input_type:
            if key == 'heading':
                # heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                # in_arr.append(heading)
                raise NotImplementedError("if key == 'heading'")
            elif key == 'map':
                # map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                # in_arr.append(map_enc)
                raise NotImplementedError("if key == 'map'")
            else:
                raise ValueError('wrong decode input type!')
        dec_in_z = torch.cat(in_arr, dim=-1)        # (N, sample_num, nz + 2)
        print(f"{dec_in_z.shape=}")

        catch_up_timestep_sequence = data['last_observed_timesteps']                                    # (N)
        starting_seq_indices = (catch_up_timestep_sequence == torch.min(catch_up_timestep_sequence))    # (*~N) (subset)

        timestep_sequence = catch_up_timestep_sequence[starting_seq_indices].to(dec_in.device)          # (*~N) == (B)
        agent_sequence = data['valid_id'][starting_seq_indices]                     # (B)
        dec_input_sequence = dec_in_z[starting_seq_indices, ...]                    # (B, sample_num, nz + 2)

        print("#" * 100)
        print(f"{catch_up_timestep_sequence=}")
        print(f"{starting_seq_indices=}")
        print(f"{timestep_sequence=}")
        print(f"{data['valid_id']=}")
        print(f"{agent_sequence=}")
        print(f"{dec_input_sequence.shape=}")
        # CATCH UP TO t_0
        print("ENTERING BELIEF GENERATION SEQUENCE:")
        while not torch.all(catch_up_timestep_sequence == torch.zeros_like(catch_up_timestep_sequence)):
            print(f"\nBEGINNING LOOP: {catch_up_timestep_sequence=}")

            dec_input_sequence, agent_sequence, timestep_sequence = self.decode_next_timestep(
                dec_in_orig=dec_in,
                z_in_orig=z_in,
                dec_input_sequence=dec_input_sequence,
                timestep_sequence=timestep_sequence,
                agent_sequence=agent_sequence,
                data=data,
                context=context,
                sample_num=sample_num
            )

            catch_up_timestep_sequence[catch_up_timestep_sequence == torch.min(catch_up_timestep_sequence)] += 1


        print("DONE WITH THE CATCHING UP")
        # PREDICT THE FUTURE

        for i in range(self.future_frames):
            print(f"\nBEGINNING FUTURE LOOP: {i}")

            dec_input_sequence, agent_sequence, timestep_sequence = self.decode_next_timestep(
                dec_in_orig=dec_in,
                z_in_orig=z_in,
                dec_input_sequence=dec_input_sequence,
                timestep_sequence=timestep_sequence,
                agent_sequence=agent_sequence,
                data=data,
                context=context,
                sample_num=sample_num
            )

        print(f"DONE PREDICTING")
        print(zblu)


        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        data[f'{mode}_seq_out'] = seq_out

        if self.pred_type == 'vel':
            # dec_motion = torch.cumsum(seq_out, dim=0)
            # dec_motion += pre_motion[[-1]]
            raise NotImplementedError
        elif self.pred_type == 'pos':
            # dec_motion = seq_out.clone()
            raise NotImplementedError
        elif self.pred_type == 'scene_norm':
            dec_motion = seq_out
            dec_motion[..., :self.forecast_dim] += data['scene_orig']
        else:
            # dec_motion = seq_out + pre_motion[[-1]]
            raise NotImplementedError

        dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
        if mode == 'infer':
            dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
        data[f'{mode}_dec_motion'] = dec_motion
        if self.pred_mode == "gauss":
            data[f'{mode}_dec_mu'] = dec_motion[..., 0:self.forecast_dim]
            data[f'{mode}_dec_sig'] = dec_motion[..., self.forecast_dim:2*self.forecast_dim]
            data[f'{mode}_dec_rho'] = dec_motion[..., 2*self.forecast_dim:]
            data[f'{mode}_dec_Sig'] = self.out_module[0].covariance_matrix(sig=data[f'{mode}_dec_sig'], rho=data[f'{mode}_dec_rho'])
        if need_weights:
            data['attn_weights'] = attn_weights

    def decode_traj_batch(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        raise NotImplementedError

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data['context_enc'].repeat_interleave(sample_num, dim=1)       # (O, sample_num * model_dim)
        print(f"{data['pre_motion'].shape=}")
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=1)             # (O, sample_num * model_dim, 2)
        print(f"{pre_motion.shape=}")
        print(f"{data['pre_vel'].shape=}")
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=1) if self.pred_type == 'vel' else None     # (O, sample_num * model_dim, 2)
        # print(f"{pre_vel.shape=}")
        print(f"{data['pre_motion_scene_norm'].shape=}")
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=1)
        print(f"{pre_motion_scene_norm.shape=}")

        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        if self.learn_prior:
            h = data['agent_context'].repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device), logvar=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))
            else:
                data[prior_key] = Categorical(logits=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))

        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()
            else:
                raise ValueError('Unknown Mode!')

        if autoregress:
            self.decode_traj_ar(
                data=data,
                mode=mode,
                context=context,
                pre_motion=pre_motion,
                pre_vel=pre_vel,
                pre_motion_scene_norm=pre_motion_scene_norm,
                z=z,
                sample_num=sample_num,
                need_weights=need_weights
            )
        else:
            self.decode_traj_batch(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)
        

class AgentFormer(nn.Module):
    """ AgentFormer """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg

        input_type = cfg.get('input_type', 'pos')
        pred_type = cfg.get('pred_type', input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get('fut_input_type', input_type)
        dec_input_type = cfg.get('dec_input_type', [])
        self.ctx = {
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
            'use_map': cfg.get('use_map', False)
        }
        self.use_map = self.ctx['use_map']
        self.rand_rot_scene = cfg.get('rand_rot_scene', False)
        self.discrete_rot = cfg.get('discrete_rot', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        self.max_train_agent = cfg.get('max_train_agent', 100)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        if self.ctx['z_type'] == 'discrete':
            self.ctx['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None

        # map encoder
        if self.use_map:
            self.map_encoder = MapEncoder(cfg.map_encoder)
            self.ctx['map_enc_dim'] = self.map_encoder.out_dim

        # models
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        self.future_encoder = FutureEncoder(cfg.future_encoder, self.ctx)
        self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_data(self, data: dict) -> None:
        device = self.device
        if self.training and len(data['full_motion_3D']) > self.max_train_agent:
            in_data = {}
            ind = np.random.choice(len(data['full_motion_3D']), self.max_train_agent, replace=False).tolist()
            for key in ['heading', 'full_motion_3D', 'obs_mask', 'fut_motion_mask', 'pre_motion_mask', 'timesteps', 'valid_id']:
                in_data[key] = [data[key][i] for i in ind if data[key] is not None]
        else:
            in_data = data

        self.data = defaultdict(lambda: None)
        self.data['valid_id'] = torch.tensor(in_data['valid_id']).to(device).to(int)
        self.data['T_total'] = len(in_data['timesteps'])                     # int: T_total
        self.data['batch_size'] = len(self.data['valid_id'])                 # int: N
        self.data['agent_num'] = len(self.data['valid_id'])                  # int: N
        self.data['timesteps'] = in_data['timesteps'].detach().clone().to(device)       # (T_total)
        full_motion = torch.stack(in_data['full_motion_3D'], dim=0).to(device).transpose(0, 1).contiguous()                     # (T_total, N, 2)
        obs_mask = torch.stack(in_data['obs_mask'], dim=0).to(device).to(dtype=torch.bool).transpose(0, 1).contiguous()         # (T_total, N)
        last_observed_timestep_indices = torch.stack([mask.nonzero().flatten()[-1] for mask in in_data['obs_mask']]).to(device)        # (N)
        last_observed_pos = full_motion[last_observed_timestep_indices, torch.arange(full_motion.size(1))]            # (N, 2)
        timesteps_to_predict = []
        for k, last_obs in enumerate(last_observed_timestep_indices):
            timesteps_to_predict.append(torch.cat((
                torch.full([int(last_obs + 1)], False),
                torch.full([int(full_motion.shape[0] - (last_obs + 1))], True)
            )))
        timesteps_to_predict = torch.stack(timesteps_to_predict, dim=0).transpose(0, 1)                                         # (T_total, N)
        fut_mask = torch.stack(in_data['fut_motion_mask'], dim=0)
        self.data['fut_mask'] = fut_mask.to(device)       # (1, T_pred)
        pre_mask = torch.stack(in_data['pre_motion_mask'], dim=0)
        self.data['pre_mask'] = pre_mask.to(device)       # (1, T_obs)
        full_agent_mask = self.data['valid_id'].repeat(self.data['T_total'], 1)                 # (T_total, N)
        full_timestep_mask = self.data['timesteps'].view(-1, 1).repeat(1, self.data['agent_num'])       # (T_total, N)
        self.data['last_observed_timesteps'] = torch.tensor(
            [self.data['timesteps'][last_obs] for last_obs in last_observed_timestep_indices]
        ).to(device)        # (N)

        # define the scene origin
        scene_orig_all_past = self.cfg.get('scene_orig_all_past', False)
        if scene_orig_all_past:
            scene_orig = full_motion[obs_mask].mean(dim=0).contiguous()           # (2)
        else:
            scene_orig = last_observed_pos.mean(dim=0).contiguous()               # (2)

        self.data['scene_orig'] = scene_orig.to(device)

        # perform random rotation
        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                raise NotImplementedError
            else:
                theta = torch.rand(1).to(device) * np.pi * 2
                full_motion, full_motion_scene_norm = rotation_2d_torch(full_motion, theta, scene_orig)
        else:
            theta = torch.zeros(1).to(device)
            full_motion_scene_norm = full_motion - scene_orig


        # create past and future tensors
        pre_motion = torch.full_like(full_motion, float('nan'))         # (T_total, N, 2)
        pre_motion[obs_mask, ...] = full_motion[obs_mask, ...]
        self.data['pre_motion'] = pre_motion.to(device)
        pre_motion_scene_norm = pre_motion - scene_orig                 # (T_total, N, 2)
        self.data['pre_motion_scene_norm'] = pre_motion_scene_norm.to(device)
        self.data['pre_sequence'] = full_motion[obs_mask, ...]          # (O, 2), where O is equal to sum(obs_mask)
        self.data['pre_sequence_scene_norm'] = full_motion_scene_norm[obs_mask, ...]        # (O, 2)
        self.data['pre_agents'] = full_agent_mask[obs_mask]                                 # (O)
        self.data['pre_timesteps'] = full_timestep_mask[obs_mask]                           # (O)

        # print(f"{self.data['pre_sequence'], self.data['pre_sequence'].shape=}")
        # print(f"{self.data['pre_agents'], self.data['pre_agents'].shape=}")
        # print(f"{self.data['pre_timesteps'], self.data['pre_timesteps'].shape=}")
        # print(f"{self.data['pre_motion'][:, 0, :]=}")
        # print(f"{self.data['pre_sequence'][self.data['pre_agents']==392]=}")
        # print(f"{self.data['pre_motion'][0, :, :]=}")
        # print(f"{self.data['pre_sequence'][self.data['pre_timesteps']==-7]=}")

        # print(f"{full_agent_mask, full_agent_mask.shape=}")
        # print(f"{full_timestep_mask, full_timestep_mask.shape=}")
        # print(f"{obs_mask, obs_mask.shape=}")
        # print(f"{full_motion[..., 0], full_motion.shape=}")

        fut_motion = torch.full_like(full_motion, float('nan'))         # (T_total, N, 2)
        fut_motion[timesteps_to_predict, ...] = full_motion[timesteps_to_predict, ...]
        self.data['fut_motion'] = fut_motion.to(device)
        fut_motion_scene_norm = fut_motion - scene_orig                 # (T_total, N, 2)
        self.data['fut_motion_scene_norm'] = fut_motion_scene_norm.to(device)
        self.data['fut_sequence'] = full_motion[timesteps_to_predict, ...]          # (P, 2), where P equals sum(timesteps_to_predict)
        self.data['fut_sequence_scene_norm'] = full_motion_scene_norm[timesteps_to_predict, ...]        # (P, 2)
        self.data['fut_agents'] = full_agent_mask[timesteps_to_predict]                                 # (P)
        self.data['fut_timesteps'] = full_timestep_mask[timesteps_to_predict]                           # (P)

        fut_motion_orig = fut_motion.detach().clone().transpose(0, 1)
        self.data['fut_motion_orig'] = fut_motion_orig.to(device)
        fut_motion_orig_scene_norm = fut_motion_orig - scene_orig       # (N, T_total, 2)
        self.data['fut_motion_orig_scene_norm'] = fut_motion_orig_scene_norm.to(device)

        full_vel = full_motion[1:] - full_motion[:-1]                   # (T_total - 1, N, 2)
        pre_vel = torch.full_like(full_vel, float('nan'))
        pre_vel[obs_mask[1:, ...], ...] = full_vel[obs_mask[1:, ...], ...]        # (T_total - 1, N, 2)
        self.data['pre_vel'] = pre_vel.to(device)
        self.data['pre_vel_seq'] = full_vel[obs_mask[1:, ...], ...]               # ()

        fut_vel = torch.full_like(full_vel, float('nan'))
        fut_vel[timesteps_to_predict[1:, ...], ...] = full_vel[timesteps_to_predict[1:, ...], ...]    # (T_total - 1, N, 2)
        self.data['fut_vel'] = fut_vel.to(device)
        self.data['fut_vel_seq'] = full_vel[timesteps_to_predict[1:, ...], ...]

        cur_motion = full_motion[last_observed_timestep_indices, torch.arange(full_motion.size(1))].unsqueeze(0)      # (1, N, 2)
        self.data['cur_motion'] = cur_motion.to(device)
        cur_motion_scene_norm = full_motion_scene_norm[last_observed_timestep_indices, torch.arange(full_motion_scene_norm.size(1))].unsqueeze(0)
        self.data['cur_motion_scene_norm'] = cur_motion_scene_norm.to(device)                   # (1, N, 2)

        # print(f"{self.data['pre_motion_scene_norm'], self.data['pre_motion_scene_norm'].shape=}")
        # print(f"{self.data['cur_motion_scene_norm'], self.data['cur_motion_scene_norm'].shape=}")
        # print(f"{self.data['pre_timesteps']=}")
        # print(f"{self.data['pre_agents']=}")
        # print(f"{last_observed_timestep_indices=}")

        pre_motion_norm = pre_motion - cur_motion           # (T_total, N, 2)
        self.data['pre_motion_norm'] = pre_motion_norm.to(device)
        fut_motion_norm = fut_motion - cur_motion           # (T_total, N, 2)
        self.data['fut_motion_norm'] = fut_motion_norm.to(device)

        self.data['fut_mask'] = torch.stack(in_data['fut_motion_mask'], dim=0).to(device)       # (1, T_pred)
        self.data['pre_mask'] = torch.stack(in_data['pre_motion_mask'], dim=0).to(device)       # (1, T_obs)
        # NOTE: heading does not follow the occlusion pattern
        if in_data['heading'] is not None:
            self.data['heading'] = torch.tensor(in_data['heading']).float().to(device)      # (N)

        # NOTE: heading does not follow the occlusion pattern
        if in_data['heading'] is not None:
            self.data['heading_vec'] = torch.stack([torch.cos(self.data['heading']), torch.sin(self.data['heading'])], dim=-1)      # (N, 2)

        # NOTE: agent maps should not be used in the occlusion case (at least not without important modification wrt
        # effective data observedness)
        # agent maps
        if self.use_map:
            scene_map = data['scene_map']
            scene_points = np.stack(in_data['pre_motion_3D'])[:, -1] * data['traj_scale']
            if self.map_global_rot:
                patch_size = [50, 50, 50, 50]
                rot = theta.repeat(self.data['agent_num']).cpu().numpy() * (180 / np.pi)
            else:
                patch_size = [50, 10, 50, 90]
                rot = -np.array(in_data['heading']) * (180 / np.pi)
            self.data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot).to(device)      # (N, 3, 100, 100)

        conn_dist = self.cfg.get('conn_dist', 100000.0)
        cur_motion = self.data['cur_motion'][0]
        if conn_dist < 1000.0:
            # threshold = conn_dist / self.cfg.traj_scale
            # pdist = F.pdist(cur_motion)
            # D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
            # D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            # D += D.T
            # mask = torch.zeros_like(D)
            # mask[D > threshold] = float('-inf')
            raise NotImplementedError("if conn_dist < 1000.0")
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        self.data['agent_mask'] = mask          # (N, N)

        self.data['past_window'] = data.get('past_window', None)
        self.data['future_window'] = data.get('future_window', None)

        print("#" * 100)
        # print(f"{obs_mask, obs_mask.shape=}")
        [print(f"{k}:\t{type(v)}\t" + (str(v.shape) if isinstance(v, torch.Tensor) else str(v))) for k, v in self.data.items()]
        # print(f"{self.data['pre_motion']=}")
        # print(f"{self.data['fut_motion']=}")
        # print(f"{self.data['pre_mask']=}")
        # print(f"{self.data['fut_mask']=}")
        # print(f"{self.data['scene_orig']=}")
        # print(f"{self.data['pre_motion_scene_norm']=}")
        # print(f"{self.data['fut_motion_scene_norm']=}")
        # print(f"{self.data['pre_vel']=}")
        # print(f"{self.data['fut_vel']=}")
        # print(f"{self.data['cur_motion']=}")
        # print(f"{self.data['pre_motion_norm']=}")
        # print(f"{self.data['fut_motion_norm']=}")
        # print(f"{self.data['agent_mask']=}")

        # # # WIP CODE
        # for i in [0, 12]:
        #     print(f"checking agent {i} ########################################################")
        #     print(f"{theta=}")
        #     print(f"{full_motion[:, i]=}")
        #     print(f"{obs_mask[:, i]=}")
        #     print(f"{last_observed_timesteps[i]=}")
        #     print(f"{timesteps_to_predict[:, i]=}")
        #
        #     # print(f"{pre_motion[:, i, :]=}")
        #     print(f"{self.data['pre_motion'][:, i, :]=}")
        #     # print(f"{pre_motion_scene_norm[:, i, :]=}")
        #     print(f"{self.data['pre_motion_scene_norm'][:, i, :]=}")
        #     # print(f"{fut_motion[:, i, :]=}")
        #     print(f"{self.data['fut_motion'][:, i, :]=}")
        #     # print(f"{fut_motion_scene_norm[:, i, :]=}")
        #     print(f"{self.data['fut_motion_scene_norm'][:, i, :]=}")
        #     # print(f"{fut_motion_orig[i, :]=}")
        #     print(f"{self.data['fut_motion_orig'][i, :]=}")
        #     # print(f"{fut_motion_orig_scene_norm[i, :]=}")
        #     print(f"{self.data['fut_motion_orig_scene_norm'][i, :]=}")
        #     # print(f"{full_vel[:, i, :]=}")
        #     # print(f"{pre_vel[:, i, :]=}")
        #     print(f"{self.data['pre_vel'][:, i, :]=}")
        #     # print(f"{fut_vel[:, i, :]=}")
        #     print(f"{self.data['fut_vel'][:, i, :]=}")
        #     # print(f"{cur_motion[:, i, :]=}")
        #     print(f"{self.data['cur_motion'][:, i, :]=}")
        #     # print(f"{pre_motion_norm[:, i, :]=}")
        #     print(f"{self.data['pre_motion_norm'][:, i, :]=}")
        #     # print(f"{fut_motion_norm[:, i, :]=}")
        #     print(f"{self.data['fut_motion_norm'][:, i, :]=}")
        # #
        # # # WIP CODE

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self):
        if self.use_map:
            self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
        self.context_encoder(self.data)
        self.future_encoder(self.data)
        self.future_decoder(self.data, mode='train', autoregress=self.ar_train)
        if self.compute_sample:
            self.inference(sample_num=self.loss_cfg['sample']['k'])
        return self.data

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        if self.use_map and self.data['map_enc'] is None:
            self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
        if self.data['context_enc'] is None:
            self.context_encoder(self.data)
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(self.data)
        self.future_decoder(self.data, mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data

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
