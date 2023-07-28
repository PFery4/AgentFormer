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


def generate_ar_mask(sz: int, agent_num: int, agent_mask: torch.Tensor):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask


def generate_mask(tgt_sz: int, src_sz: int, agent_num: int, agent_mask: torch.Tensor) -> torch.Tensor:
    # TODO: WIP WIP WIP WIP WIP WIP
    # assert tgt_sz % agent_num == 0 and src_sz % agent_num == 0
    # mask = agent_mask.repeat(tgt_sz // agent_num, src_sz // agent_num)
    mask = torch.zeros(99, 99).to(agent_mask.device)
    return mask


class PositionalEncoding(nn.Module):

    """ Positional Encoding """
    def __init__(self, d_model: int, dropout: float = 0.1, timestep_window: Tuple[int, int] = (-20, 30), concat: bool = False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.concat = concat
        timestep_window = torch.arange(*timestep_window, dtype=torch.float).unsqueeze(1).to()       # (t_range, 1)
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
        return pe       # shape (t_range, d_model)

    def time_encode(self, sequence_timesteps: torch.Tensor) -> torch.Tensor:
        # sequence_timesteps: (T_total)
        # out: (T_total, self.d_model)
        return torch.cat([self.pe[(self.timestep_window == t).squeeze(), ...] for t in sequence_timesteps], dim=0)

    def forward(self, x: torch.Tensor, time_tensor: torch.Tensor):
        # x: (T, model_dim)
        # time_tensor: (T)
        pos_enc = self.time_encode(time_tensor)
        if self.concat:
            x = torch.cat([x, pos_enc], dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x)      # How does dropout behave on sequence tensor?

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
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
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
        traj_in = []
        seq_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['pre_motion'])
                seq_in.append(data['pre_sequence'])
            elif key == 'vel':
                vel = data['pre_vel']
                vel_seq = data['pre_vel_seq']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)          # Imputation
                    vel_seq = torch.cat([                            # Imputation
                        vel_seq[:data['agent_num']],
                        vel_seq
                    ], dim=0)

                if self.vel_heading:
                    # vel = rotation_2d_torch(vel, -data['heading'])[0]
                    raise NotImplementedError("hmmm")
                traj_in.append(vel)
                seq_in.append(vel_seq)
            elif key == 'norm':
                # traj_in.append(data['pre_motion_norm'])
                raise NotImplementedError("HMMMM")
            elif key == 'scene_norm':
                traj_in.append(data['pre_motion_scene_norm'])
                seq_in.append(data['pre_sequence_scene_norm'])
            elif key == 'heading':
                # hv = data['heading_vec'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                # traj_in.append(hv)
                raise NotImplementedError("Hmmmm")
            elif key == 'map':
                # map_enc = data['map_enc'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                # traj_in.append(map_enc)
                raise NotImplementedError("hmmm")
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)                    # (T, N, Features)
        seq_in = torch.cat(seq_in, dim=-1)

        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)      # (T * N, 1, model_dim)
        tf_seq_in = self.input_fc(seq_in)               # (O, model_dim)

        tf_in_pos = self.pos_encoder(x=tf_seq_in, time_tensor=data['pre_timesteps'])            # (O, model_dim)

        src_agent_mask = data['agent_mask'].clone()         # (N, N)

        src_mask = generate_mask(tf_seq_in.shape[0], tf_seq_in.shape[0], data['agent_num'], src_agent_mask).to(tf_in.device)        # (T * N, T * N)        # TODO: FIX THIS MASK GENERATION PROCESS
        # print(f"{src_mask.shape=}")

        data['context_enc'] = self.tf_encoder(tf_in_pos, data['pre_agents'], mask=src_mask, num_agent=data['agent_num'])        # (O, model_dim)

        # compute per agent context
        print(f"{self.pooling=}")
        if self.pooling == 'mean' and False:
            data['agent_context'] = torch.cat(
                [torch.mean(data['context_enc'][data['pre_agents'] == ag_id, :], dim=0).unsqueeze(0)
                 for ag_id in torch.unique(data['pre_agents'])], dim=0
            )       # (N, model_dim)
        else:
            data['agent_context'] = torch.cat(
                [torch.max(data['context_enc'][data['pre_agents'] == ag_id, :], dim=0)[0].unsqueeze(0)
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
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
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
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['fut_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)
        # # WIP CODE
        # fig, ax = plt.subplots()
        # print(f"FUTURE ENCODER")
        # self.pos_encoder.plot_positional_window(ax=ax, num_t=tf_in.shape[0]//data['agent_num'])
        # plt.show()
        # # WIP CODE

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()
        mem_mask = generate_mask(tf_in.shape[0], data['context_enc'].shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)

        tf_out, _ = self.tf_decoder(tf_in_pos, data['context_enc'], memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'])
        tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)

        if self.pooling == 'mean':
            h = torch.mean(tf_out, dim=0)
        else:
            h = torch.max(tf_out, dim=0)[0]
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
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
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

    def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=False):
        agent_num = data['agent_num']
        if self.pred_type == 'vel':
            dec_in = pre_vel[[-1]]
        elif self.pred_type == 'pos':
            dec_in = pre_motion[[-1]]
        elif self.pred_type == 'scene_norm':
            dec_in = pre_motion_scene_norm[[-1]]
        else:
            dec_in = torch.zeros_like(pre_motion[[-1]])
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        if self.pred_mode == "gauss":
            dist_params = torch.zeros([*dec_in.shape[:-1], self.out_module[0].N_params]).to(dec_in.device)
            dist_params[..., :self.forecast_dim] = dec_in
            dec_in = dist_params

        z_in = z.view(-1, sample_num, z.shape[-1])
        in_arr = [dec_in, z_in]
        for key in self.input_type:
            if key == 'heading':
                heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            else:
                raise ValueError('wrong decode input type!')
        dec_in_z = torch.cat(in_arr, dim=-1)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()

        for i in range(self.future_frames):
            tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
            agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
            toffset = -1
            tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle, t_offset=toffset)

            # # WIP CODE
            # fig, ax = plt.subplots()
            # print(f"FUTURE DECODER")
            # self.pos_encoder.plot_positional_window(ax=ax, num_t=tf_in.shape[0] // agent_num, t_offset=toffset)
            # plt.show()
            # # WIP CODE

            # tf_in_pos = tf_in
            mem_mask = generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
            tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

            tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=need_weights)

            out_tmp = tf_out.view(-1, tf_out.shape[-1])

            seq_out = self.out_module(out_tmp)
            seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])

            if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
                norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
                if self.sn_out_type == 'vel':
                    norm_motion = torch.cumsum(norm_motion, dim=0)
                    if self.pred_mode == "gauss":
                        raise NotImplementedError
                if self.sn_out_heading:
                    angles = data['heading'].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]
                    if self.pred_mode == "gauss":
                        raise NotImplementedError
                seq_out = norm_motion + dec_in.view(-1, agent_num * sample_num, dec_in.shape[-1])
                seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
            if self.ar_detach:
                out_in = seq_out[-agent_num:].clone().detach()
            else:
                out_in = seq_out[-agent_num:]
            # create dec_in_z
            in_arr = [out_in, z_in]
            for key in self.input_type:
                if key == 'heading':
                    in_arr.append(heading)
                elif key == 'map':
                    in_arr.append(map_enc)
                else:
                    raise ValueError('wrong decoder input type!')
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        data[f'{mode}_seq_out'] = seq_out

        if self.pred_type == 'vel':
            dec_motion = torch.cumsum(seq_out, dim=0)
            dec_motion += pre_motion[[-1]]
        elif self.pred_type == 'pos':
            dec_motion = seq_out.clone()
        elif self.pred_type == 'scene_norm':
            dec_motion = seq_out
            dec_motion[..., :self.forecast_dim] += data['scene_orig']
        else:
            dec_motion = seq_out + pre_motion[[-1]]

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
        context = data['context_enc'].repeat_interleave(sample_num, dim=1)       # 80 x 64
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=1)             # 10 x 80 x 2
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=1) if self.pred_type == 'vel' else None
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=1)

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
            self.decode_traj_ar(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=need_weights)
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
            'agent_enc_shuffle': cfg.get('agent_enc_shuffle', False),
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
        last_observed_timesteps = torch.stack([mask.nonzero().flatten()[-1] for mask in in_data['obs_mask']]).to(device)        # (N)
        last_observed_pos = full_motion[last_observed_timesteps, torch.arange(full_motion.size(1))]            # (N, 2)
        timesteps_to_predict = []
        for k, last_obs in enumerate(last_observed_timesteps):
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
        self.data['fut_timesteps'] = full_agent_mask[timesteps_to_predict]                              # (P)

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

        cur_motion = full_motion[last_observed_timesteps, torch.arange(full_motion.size(1))].unsqueeze(0)      # (1, N, 2)
        self.data['cur_motion'] = cur_motion.to(device)

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

        # agent shuffling
        if self.training and self.ctx['agent_enc_shuffle']:
            self.data['agent_enc_shuffle'] = torch.randperm(self.ctx['max_agent_len'])[:self.data['agent_num']].to(device)
        else:
            self.data['agent_enc_shuffle'] = None

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
            raise NotImplementedError
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        self.data['agent_mask'] = mask          # (N, N)

        self.data['past_window'] = data.get('past_window', None)
        self.data['future_window'] = data.get('future_window', None)

        print("#" * 100)
        print(f"{obs_mask, obs_mask.shape=}")
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
