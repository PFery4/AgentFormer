import torch
from torch import nn
from collections import defaultdict

from typing import Dict


class BasePredictorClass(nn.Module):

    def __init__(self):
        super().__init__()

        self.device = torch.device('cpu')

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_data(self, data: Dict):
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

    def inference(self, mode='infer', *args, **kwargs):
        raise NotImplementedError


class Oracle(BasePredictorClass):
    """
    Perfect Oracle:
    This model predicts the Ground Truth (yes, this is cheating).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def inference(self, mode='infer', *args, **kwargs):
        self.data[f'{mode}_dec_motion'] = self.data['pred_position_sequence']               # [B, P, 2]
        self.data[f'{mode}_dec_agents'] = self.data['pred_identity_sequence']               # [B, P]
        self.data[f'{mode}_dec_timesteps'] = self.data['pred_timestep_sequence'][0]         # [P]
        self.data[f'{mode}_dec_past_mask'] = (self.data['pred_timestep_sequence'][0] <= 0)  # [P]

        return self.data[f'{mode}_dec_motion'], self.data       # [B, P, 2], Dict


class ConstantVelocityPredictor(BasePredictorClass):
    """
    Constant Velocity model:
    This model predicts constant velocity motion starting from the last observed position.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def inference(self, mode='infer', *args, **kwargs):
        pred_position_sequence = []
        pred_agent_sequence = []
        pred_timestep_sequence = []

        for agent_id, timestep in zip(self.data['valid_id'][0], self.data['last_obs_timesteps'][0]):
            agent_obs_seq_mask = self.data['obs_identity_sequence'][0] == agent_id
            timestep_obs_seq_mask = self.data['obs_timestep_sequence'][0] == timestep
            obs_seq_index = torch.logical_and(agent_obs_seq_mask, timestep_obs_seq_mask)
            last_obs_pos = self.data['obs_position_sequence'][0, obs_seq_index, :]
            last_obs_vel = self.data['obs_velocity_sequence'][0, obs_seq_index, :]
            pred_length = self.data['timesteps'][0, -1] - timestep

            pred_timesteps = torch.arange(timestep, self.data['timesteps'][0, -1])
            pred_agent = torch.full([pred_length], agent_id)
            added_positions = (torch.arange(pred_length, device=self.device) + 1).unsqueeze(1) * last_obs_vel
            pred_positions = last_obs_pos + added_positions

            pred_position_sequence.append(pred_positions)
            pred_agent_sequence.append(pred_agent)
            pred_timestep_sequence.append(pred_timesteps)

        pred_position_sequence = torch.cat(pred_position_sequence).to(self.device)
        pred_agent_sequence = torch.cat(pred_agent_sequence).to(self.device)
        pred_timestep_sequence = torch.cat(pred_timestep_sequence).to(self.device) + 1

        self.data[f'{mode}_dec_motion'] = pred_position_sequence.unsqueeze(0)                   # [B, P, 2]
        self.data[f'{mode}_dec_agents'] = pred_agent_sequence.unsqueeze(0).to(torch.int64)      # [B, P]
        self.data[f'{mode}_dec_timesteps'] = pred_timestep_sequence.to(torch.int64)             # [P]
        self.data[f'{mode}_dec_past_mask'] = (pred_timestep_sequence <= 0)                      # [P]

        return self.data[f'{mode}_dec_motion'], self.data       # [B, P, 2], Dict
