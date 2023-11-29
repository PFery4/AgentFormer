import torch
from torch import nn
from collections import defaultdict

from typing import Dict


class Oracle(nn.Module):
    """
    Perfect Oracle:
    This model predicts the Ground Truth (yes, this is cheating).
    """

    def __init__(self, cfg):
        super.__init__()

        self.cfg = cfg

    def set_data(self, data: Dict):
        # NOTE: in our case, batch size B is always 1

        # memory_report('BEFORE PUPOLATING DATA DICT')
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

        # if self.global_map_attention:
        #     self.data['scene_map'] = data['scene_map'].detach().clone().to(self.device)             # [B, C, H, W]
        #     self.data['occlusion_map'] = data['dist_transformed_occlusion_map'].detach().clone().to(self.device)    # [B, H, W]
        #
        #     self.data['nlog_probability_occlusion_map'] = data['nlog_probability_occlusion_map'].detach().clone().to(self.device)   # [B, H, W]
        #     self.data['combined_map'] = torch.cat((self.data['scene_map'], self.data['occlusion_map'].unsqueeze(1)), dim=1)     # [B, C + 1, H, W]
        #     self.data['map_homography'] = data['map_homography'].detach().clone().to(self.device)        # [B, 3, 3]
        # memory_report('AFTER PUPOLATING DATA DICT')

    def inference(self, mode='infer', *args, **kwargs):
        self.data[f'{mode}_dec_motion'] = self.data['pred_position_sequence']               # [B, P, 2]
        self.data[f'{mode}_dec_agents'] = self.data['pred_identity_sequence']               # [B, P]
        self.data[f'{mode}_dec_timesteps'] = self.data['pred_timestep_sequence'][0]         # [P]

        return self.data[f'{mode}_dec_motion'], self.data       # [B, P, 2], Dict


class ConstantVelocityPredictor(nn.Module):
    """
    Constant Velocity model:
    This model predicts constant velocity motion starting from the last observed position.
    """

    def __init__(self):
        super().__init__()

    def inference(self, mode='infer', *args, **kwargs):
        # TODO
        pass

