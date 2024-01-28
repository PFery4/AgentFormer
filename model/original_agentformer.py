"""

This file implements a wrapper class for the original AgentFormer model. The source code for the original model is
stored in a separate directory (OriginalAgentFormer)

The wrapper contains an instance of the original AgentFormer class, and ensures that its data
consumption and production follows the same structure/format as that of our implementation.

"""

import torch
from torch import nn
from collections import defaultdict

from OriginalAgentFormer.model.agentformer import AgentFormer
# from utils.utils import memory_report


class OrigModelWrapper(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        assert cfg.model_id == 'orig_agentformer'
        self.model_id = cfg.model_id
        self.orig_model = AgentFormer(cfg=cfg)
        self.device = torch.device('cpu')

        self.traj_scale = cfg.traj_scale
        self.T_obs = cfg.past_frames
        self.T_pred = cfg.future_frames
        self.future_timesteps = torch.arange(self.T_pred) + 1

        self.data = defaultdict(lambda: None)

    def set_device(self, device):
        self.orig_model.set_device(device)
        self.device = self.orig_model.device

    def set_data(self, data):

        trajs = data['trajectories'].squeeze(0)     # [N, T_total, 2]

        in_data_dict = {
            'pre_motion_3D': torch.unbind(trajs[:, :self.T_obs, :], dim=0),
            'fut_motion_3D': torch.unbind(trajs[:, self.T_obs:, :], dim=0),
            'pre_motion_mask': [torch.ones(self.T_obs)] * trajs.shape[0],
            'fut_motion_mask': [torch.ones(self.T_pred)] * trajs.shape[0],
            'heading': None,
            'scene_map': None,
            'traj_scale': self.traj_scale,
        }

        self.orig_model.set_data(in_data_dict)

        self.data = self.orig_model.data

        self.data['valid_id'] = data['identities'].detach().clone().to(self.device)     # [B, N]

        self.data['last_obs_positions'] = data['last_obs_positions'].detach().clone().to(self.device)   # [B, N, 2]
        self.data['last_obs_timesteps'] = data['last_obs_timesteps'].detach().clone().to(self.device)   # [B, N]

        self.data['pred_position_sequence'] = data['pred_position_sequence'].detach().clone().to(self.device)   # [B, P, 2]
        self.data['pred_velocity_sequence'] = data['pred_velocity_sequence'].detach().clone().to(self.device)   # [B, P, 2]
        self.data['pred_timestep_sequence'] = data['pred_timestep_sequence'].detach().clone().to(self.device)   # [B, P]
        self.data['pred_identity_sequence'] = data['pred_identity_sequence'].detach().clone().to(self.device)   # [B, P]

    def step_annealer(self):
        self.orig_model.step_annealer()

    def forward(self):
        return self.orig_model.forward()

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        _, out_data_dict = self.orig_model.inference(mode=mode, sample_num=sample_num, need_weights=need_weights)

        dec_timesteps = torch.vstack([self.future_timesteps] * self.data['valid_id'].shape[-1])         # [N, T_pred]
        dec_agents = torch.hstack([self.data['valid_id'][0, ...].unsqueeze(1)] * self.T_pred)           # [N, T_pred]

        dec_motion = out_data_dict[f'{mode}_dec_motion']
        if mode != 'infer':
            dec_motion = dec_motion.unsqueeze(1)
        dec_motion = dec_motion.transpose(0, 1)     # [K, N, T_pred, 2]

        # collapse N and T:
        dec_timesteps = dec_timesteps.reshape(-1)                           # [P]
        dec_agents = torch.vstack([dec_agents.reshape(-1)] * sample_num)    # [B * sample_num, P]
        dec_motion = dec_motion.reshape(sample_num, -1, 2)                  # [B * sample_num, P, 2]
        dec_past_mask = torch.full_like(dec_timesteps, False)               # [P]

        self.data[f'{mode}_dec_motion'] = dec_motion            # [B * sample_num, P, 2]
        self.data[f'{mode}_dec_agents'] = dec_agents            # [B * sample_num, P]
        self.data[f'{mode}_dec_past_mask'] = dec_past_mask      # [P]
        self.data[f'{mode}_dec_timesteps'] = dec_timesteps      # [P]

        return self.data[f'{mode}_dec_motion'], self.data       # [B * sample_num, P, 2], Dict

    def compute_loss(self):
        return self.orig_model.compute_loss()


if __name__ == '__main__':
    print("Goodbye !")
