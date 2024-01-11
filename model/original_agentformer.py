"""

This file implements a wrapper class for the original AgentFormer model. The source code for the original model is
stored in a separate directory (OriginalAgentFormer)

The wrapper contains an instance of the original AgentFormer class, and ensures that its data
consumption and production follows the same structure/format as that of our implementation.

"""

import torch
from torch import nn
from collections import defaultdict

from OriginalAgentFormer.model.model_lib import model_dict
from utils.utils import memory_report


class OrigModelWrapper(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        assert '_' in cfg.model_id
        self.model_id = cfg.model_id
        self.orig_model = model_dict[self.model_id.split('_')[-1]](cfg=cfg)
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
        self.data = defaultdict(lambda: None)

        self.data['valid_id'] = data['identities'].detach().clone().to(self.device)     # [B, N]

        self.data['last_obs_positions'] = data['last_obs_positions'].detach().clone().to(self.device)   # [B, N, 2]
        self.data['last_obs_timesteps'] = data['last_obs_timesteps'].detach().clone().to(self.device)   # [B, N]

        self.data['pred_position_sequence'] = data['pred_position_sequence'].detach().clone().to(self.device)   # [B, P, 2]
        self.data['pred_velocity_sequence'] = data['pred_velocity_sequence'].detach().clone().to(self.device)   # [B, P, 2]
        self.data['pred_timestep_sequence'] = data['pred_timestep_sequence'].detach().clone().to(self.device)   # [B, P]
        self.data['pred_identity_sequence'] = data['pred_identity_sequence'].detach().clone().to(self.device)   # [B, P]

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

        # print(f"{data['trajectories'], data['trajectories'].shape=}")
        # print(f"{len(in_data_dict['pre_motion_3D']), in_data_dict['pre_motion_3D'][0], in_data_dict['pre_motion_3D'][0].shape=}")
        # print(f"{len(in_data_dict['fut_motion_3D']), in_data_dict['fut_motion_3D'][0], in_data_dict['fut_motion_3D'][0].shape=}")
        # print(f"{len(in_data_dict['pre_motion_mask']), in_data_dict['pre_motion_mask'][0], in_data_dict['pre_motion_mask'][0].shape=}")
        # print(f"{len(in_data_dict['fut_motion_mask']), in_data_dict['fut_motion_mask'][0], in_data_dict['fut_motion_mask'][0].shape=}")
        # print(f"{in_data_dict['traj_scale']=}")
        # print()
        self.orig_model.set_data(in_data_dict)

    def step_annealer(self):
        self.orig_model.step_annealer()

    def forward(self):
        return self.orig_model.forward()

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        _, out_data_dict = self.orig_model.inference(mode=mode, sample_num=sample_num, need_weights=need_weights)

        # todo: check that the manipulated output data dict contains the appropriate data for performance evaluation

        dec_timesteps = torch.vstack([self.future_timesteps] * self.data['valid_id'].shape[-1])         # [N, T_pred]
        dec_agents = torch.hstack([self.data['valid_id'][0, ...].unsqueeze(1)] * self.T_pred)           # [N, T_pred]

        print(f"{dec_timesteps.shape=}")
        print(f"{dec_agents.shape=}")

        dec_motion = out_data_dict[f'{mode}_dec_motion']
        if mode != 'infer':
            dec_motion = dec_motion.unsqueeze(1)
        dec_motion = dec_motion.transpose(0, 1)     # [K, N, T_pred, 2]
        print(f"{dec_motion.shape=}")

        # collapse N and T:
        dec_timesteps = dec_timesteps.reshape(-1)                           # [P]
        dec_agents = torch.vstack([dec_agents.reshape(-1)] * sample_num)    # [B * sample_num, P]
        dec_motion = dec_motion.reshape(sample_num, -1, 2)                  # [B * sample_num, P, 2]
        dec_past_mask = torch.full_like(dec_timesteps, False)               # [P]

        self.data[f'{mode}_dec_motion'] = dec_motion            # [B * sample_num, P, 2]
        self.data[f'{mode}_dec_agents'] = dec_agents            # [B * sample_num, P]
        self.data[f'{mode}_dec_past_mask'] = dec_past_mask      # [P]
        self.data[f'{mode}_dec_timesteps'] = dec_timesteps      # [P]

        # if key == 'valid_id' or 'last_obs_' in key or 'pred_' in key or '_dec_' in key:
        # valid_id: Done
        # last_obs_: Done
        # pred_: Done
        # _dec_: Done

        # OURS:
        # data[f'{mode}_dec_motion'] = seq_out                                    # [B * sample_num, P, 2]
        # data[f'{mode}_dec_agents'] = pred_agent_sequence.repeat(sample_num, 1)  # [B * sample_num, P]
        # data[f'{mode}_dec_past_mask'] = past_indices                            # [P]
        # data[f'{mode}_dec_timesteps'] = pred_timestep_sequence                  # [P]

        # THEIRS:
        # dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
        # if mode == 'infer':
        #     dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
        # data[f'{mode}_dec_motion'] = dec_motion

        # - Check correct shape of orig _dec_motion
        #   - train: [N, T_pred, 2]
        #   - infer: [N, K, T_pred, 2]
        # - translate to our shape format
        # - add that to self.data
        return self.data[f'{mode}_dec_motion'], self.data       # [B * sample_num, P, 2], Dict

    def compute_loss(self):
        # TODO: CHECK IF THE LOSS DICT KEYS DO NOT CREATE ERRORS WHILE TRAINING
        return self.orig_model.compute_loss()


# class OrigAgentFormerWrapper(OrigModelBaseWrapper):
#
#     def __init__(self, cfg):
#         super().__init__(cfg=cfg)
#         self.orig_model = AgentFormer(cfg=cfg)
#
#
# class OrigDLowWrapper(OrigModelBaseWrapper):
#
#     def __init__(self, cfg):
#         super().__init__(cfg=cfg)
#         self.orig_model = DLow(cfg=cfg)


# TODO: once verifications have been made for both AgentFormer and Dlow wrappers, delete this commented out code
# class OrigAgentFormerChild(AgentFormer):
#
#     def __init__(self, cfg):
#         super().__init__(cfg=cfg)
#
#         self.T_obs = self.ctx['past_frames']
#         self.T_pred = self.ctx['future_frames']
#         self.future_timesteps = torch.arange(self.T_pred) + 1
#
#     def set_data(self, data):
#         super().set_data(data={
#             'pre_motion_3D': torch.unbind(data['trajectories'].squeeze(0)[:, :self.T_obs, :], dim=0),
#             'fut_motion_3D': torch.unbind(data['trajectories'].squeeze(0)[:, self.T_obs:, :], dim=0),
#             'pre_motion_mask': [torch.ones(self.T_obs)] * data['trajectories'].squeeze(0).shape[0],
#             'fut_motion_mask': [torch.ones(self.T_pred)] * data['trajectories'].squeeze(0).shape[0],
#             'heading': None,
#             'scene_map': None,
#             'traj_scale': self.cfg.traj_scale,
#         })
#
#         self.data['valid_id'] = data['identities'].detach().clone().to(self.device)     # [B, N]
#
#         self.data['last_obs_positions'] = data['last_obs_positions'].detach().clone().to(self.device)   # [B, N, 2]
#         self.data['last_obs_timesteps'] = data['last_obs_timesteps'].detach().clone().to(self.device)   # [B, N]
#
#         self.data['pred_position_sequence'] = data['pred_position_sequence'].detach().clone().to(self.device)   # [B, P, 2]
#         self.data['pred_velocity_sequence'] = data['pred_velocity_sequence'].detach().clone().to(self.device)   # [B, P, 2]
#         self.data['pred_timestep_sequence'] = data['pred_timestep_sequence'].detach().clone().to(self.device)   # [B, P]
#         self.data['pred_identity_sequence'] = data['pred_identity_sequence'].detach().clone().to(self.device)   # [B, P]
#
#     def occlusionformer_output_data(self, mode='infer', sample_num=20, need_weights=False):
#         super().inference(mode=mode, sample_num=sample_num, need_weights=need_weights)
#
#         dec_timesteps = torch.vstack([self.future_timesteps] * self.data['agent_num'])          # [N, T_pred]
#         dec_agents = torch.hstack([self.data['valid_id'][0, ...].unsqueeze(1)] * self.T_pred)   # [N, T_pred]
#         dec_motion = self.data[f'{mode}_dec_motion']
#         if mode != 'infer':
#             dec_motion = dec_motion.unsqueeze(1)
#         dec_motion = dec_motion.transpose(0, 1)     # [K, N, T_pred, 2]
#
#         self.data[f'{mode}_dec_motion'] = dec_motion.reshape(sample_num, -1, 2)                 # [B * sample_num, P, 2]
#         self.data[f'{mode}_dec_agents'] = torch.vstack([dec_agents.reshape(-1)] * sample_num)   # [B * sample_num, P]
#         self.data[f'{mode}_dec_past_mask'] = torch.full([self.data['agent_num'] * self.T_pred], False)  # [P]
#         self.data[f'{mode}_dec_timesteps'] = dec_timesteps.reshape(-1)                                  # [P]
#         return self.data[f'{mode}_dec_motion'], self.data


if __name__ == '__main__':
    print("Goodbye !")
