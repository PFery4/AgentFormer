"""

This file implements a wrapper class for the original DLow model. The source code for the original model is
stored in a separate directory (OriginalAgentFormer)

The wrapper contains an instance of the original Dlow class, and ensures that its data
consumption and production follows the same structure/format as that of our implementation.

"""

import torch
from torch import nn

from utils.config import Config
from model.original_agentformer import OrigModelWrapper
from OriginalAgentFormer.model.dlow import loss_func
from OriginalAgentFormer.model.common.mlp import MLP
from OriginalAgentFormer.model.common.dist import Normal


class OrigDLowWrapper(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        assert cfg.model_id == 'orig_dlow'
        self.device = torch.device('cpu')
        self.cfg = cfg
        self.nk = nk = cfg.sample_k
        self.nz = nz = cfg.nz
        self.share_eps = cfg.get('share_eps', True)
        self.train_w_mean = cfg.get('train_w_mean', False)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())

        pred_cfg = Config(cfg.pred_cfg, tmp=False, create_dirs=False)
        pred_model = OrigModelWrapper(pred_cfg)
        self.pred_model_dim = pred_cfg.tf_model_dim
        assert cfg.pred_checkpoint_name is not None
        cp_path = pred_cfg.model_path % cfg.pred_checkpoint_name
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location='cpu')
        pred_model.load_state_dict(model_cp['model_dict'])
        pred_model.eval()
        self.pred_model = [pred_model]

        # Dlow's Q net
        self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, nk * nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, nk * nz)

        self.traj_scale = cfg.traj_scale
        self.T_obs = cfg.past_frames
        self.T_pred = cfg.future_frames
        self.future_timesteps = torch.arange(self.T_pred) + 1

    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data

    def main(self, mean=False, need_weights=False):
        orig_pred_model = self.pred_model[0].orig_model
        if hasattr(orig_pred_model, 'use_map') and orig_pred_model.use_map:
            self.data['map_enc'] = orig_pred_model.map_encoder(self.data['agent_maps'])
        orig_pred_model.context_encoder(self.data)

        if not mean:
            if self.share_eps:
                eps = torch.randn([1, self.nz]).to(self.device)
                eps = eps.repeat((self.data['agent_num'] * self.nk, 1))
            else:
                eps = torch.randn([self.data['agent_num'], self.nz]).to(self.device)
                eps = eps.repeat_interleave(self.nk, dim=0)

        qnet_h = self.q_mlp(self.data['agent_context'])
        A = self.q_A(qnet_h).view(-1, self.nz)
        b = self.q_b(qnet_h).view(-1, self.nz)

        z = b if mean else A*eps + b
        logvar = (A ** 2 + 1e-8).log()
        self.data['q_z_dist_dlow'] = Normal(mu=b, logvar=logvar)

        orig_pred_model.future_decoder(self.data, mode='infer', sample_num=self.nk, autoregress=True, z=z, need_weights=need_weights)
        return self.data

    def forward(self):
        return self.main(mean=self.train_w_mean)

    def inference(self, mode, sample_num, need_weights=False):
        self.main(mean=True, need_weights=need_weights)

        dec_timesteps = torch.vstack([self.future_timesteps] * self.data['valid_id'].shape[-1]).to(self.device)  # [N, T_pred]
        dec_agents = torch.hstack([self.data['valid_id'][0, ...].unsqueeze(1)] * self.T_pred)  # [N, T_pred]

        dec_motion = self.data[f'infer_dec_motion']
        dec_motion = dec_motion.transpose(0, 1)     # [K, N, T_pred, 2]

        # collapse N and T:
        dec_timesteps = dec_timesteps.reshape(-1)                           # [P]
        dec_agents = torch.vstack([dec_agents.reshape(-1)] * sample_num)    # [B * sample_num, P]
        dec_motion = dec_motion.reshape(sample_num, -1, 2)                  # [B * sample_num, P, 2]
        dec_past_mask = torch.full_like(dec_timesteps, False)               # [P]

        self.data[f'infer_dec_motion'] = dec_motion            # [B * sample_num, P, 2]
        self.data[f'infer_dec_agents'] = dec_agents            # [B * sample_num, P]
        self.data[f'infer_dec_past_mask'] = dec_past_mask      # [P]
        self.data[f'infer_dec_timesteps'] = dec_timesteps      # [P]

        res = self.data[f'infer_dec_motion']
        if mode == 'recon':
            res = res[:, 0]
        return res, self.data

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

    def step_annealer(self):
        pass

