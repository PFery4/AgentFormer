import glob
import pickle
import os.path
from io import TextIOWrapper

import cv2
import h5py
import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.colors as colors
import pandas as pd
from matplotlib.path import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys
import numpy as np
import skgeom as sg
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from collections import defaultdict

# from data.map import TorchGeometricMap
from data.map import HomographyMatrix, MapManager, MAP_DICT
from utils.config import Config, REPO_ROOT
from utils.utils import print_log, get_timestring

from typing import Dict, Optional
Tensor = torch.Tensor

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
import src.visualization.sdd_visualize as visualize
import src.occlusion_simulation.visibility as visibility
import src.occlusion_simulation.polygon_generation as poly_gen
import src.data.config as sdd_conf


# TODO: move these functions to a separate file.
def last_observed_indices(
        obs_mask: Tensor    # [N, T]
) -> Tensor:                # [N]
    return obs_mask.shape[1] - torch.argmax(torch.flip(obs_mask, dims=[1]), dim=1) - 1


def last_observed_positions(
        trajs: Tensor,              # [N, T, 2]
        last_obs_indices: Tensor    # [N]
) -> Tensor:                        # [N, 2]
    return trajs[torch.arange(trajs.shape[0]), last_obs_indices, :]


def true_velocity(
        trajs: Tensor   # [N, T, 2]
) -> Tensor:            # [N, T, 2]
    vel = torch.zeros_like(trajs)
    vel[:, 1:, :] = trajs[:, 1:, :] - trajs[:, :-1, :]
    return vel


def observed_velocity(
        trajs: Tensor,      # [N, T, 2]
        obs_mask: Tensor    # [N, T]
) -> Tensor:                # [N, T, 2]
    vel = torch.zeros_like(trajs)
    for traj, mask, v in zip(trajs, obs_mask, vel):
        obs_indices = torch.nonzero(mask)  # [Z, 1]
        motion_diff = traj[obs_indices[1:, 0], :] - traj[obs_indices[:-1, 0], :]  # [Z - 1, 2]
        v[obs_indices[1:].squeeze(), :] = motion_diff / (obs_indices[1:, :] - obs_indices[:-1, :])  # [Z - 1, 2]
    return vel  # [N, T, 2]


def cv_extrapolate(
        trajs: Tensor,              # [N, T, 2]
        obs_vel: Tensor,            # [N, T, 2]
        last_obs_indices: Tensor    # [N]
) -> Tensor:                        # [N, T, 2]
    xtrpl_trajs = trajs.detach().clone()
    for traj, vel, obs_idx in zip(xtrpl_trajs, obs_vel, last_obs_indices):
        last_pos = traj[obs_idx]
        last_vel = vel[obs_idx]
        extra_seq = last_pos + torch.arange(traj.shape[0] - obs_idx).unsqueeze(1) * last_vel
        traj[obs_idx:] = extra_seq
    return xtrpl_trajs


def impute_and_cv_predict(
        trajs: Tensor,      # [N, T, 2]
        obs_mask: Tensor,   # [N, T]
        timesteps: Tensor   # [T]
) -> Tensor:                # [N, T, 2]
    imputed_trajs = torch.zeros_like(trajs)
    for idx, (traj, mask) in enumerate(zip(trajs, obs_mask)):
        # if none of the values are observed, then skip this trajectory altogether
        if mask.sum() == 0:
            continue
        f = interp1d(timesteps[mask], traj[mask], axis=0, fill_value='extrapolate')
        interptraj = f(timesteps)
        imputed_trajs[idx, ...] = torch.from_numpy(interptraj)
    return imputed_trajs


def points_within_distance(
        target_point: Tensor,   # [2]
        points: Tensor,         # [N, 2]
        distance: Tensor        # [1]
) -> Tensor:                    # [N]
    distances = torch.linalg.norm(points - target_point, dim=1)
    return distances <= distance


class TorchDataGeneratorSDD(Dataset):
    def __init__(self, parser: Config, split: str = 'train'):
        self.split = split
        assert split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"Error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        self.sdd_config = sdd_conf.get_config(os.path.join(sdd_conf.REPO_ROOT, parser.sdd_config_file_name))
        dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config, split=self.split)

        # stealing StanfordDroneDatasetWithOcclusionSim relevant data
        self.coord_conv = dataset.coord_conv
        self.frames = dataset.frames
        self.lookuptable = dataset.lookuptable
        self.occlusion_table = dataset.occlusion_table
        self.image_path = os.path.join(dataset.SDD_root, 'annotations')
        assert os.path.exists(self.image_path)

        self.rand_rot_scene = parser.get('rand_rot_scene', False)
        self.max_train_agent = parser.get('max_train_agent', 100)
        self.traj_scale = parser.traj_scale
        self.occlusion_process = parser.get('occlusion_process', 'fully_observed')
        self.with_rgb_map = parser.get('with_rgb_map', True)      # whether to process the RGB scene map, or not.
        self.map_side = parser.get('scene_side_length', 80.0)  # [m]
        self.distance_threshold_occluded_target = self.map_side / 4  # [m]
        self.map_resolution = parser.get('global_map_resolution', 800)  # [px]
        self.map_crop_coords = self.get_map_crop_coordinates()
        self.map_homography = self.get_map_homography()

        self.to_torch_image = transforms.ToTensor()

        if self.occlusion_process == 'fully_observed':
            self.trajectory_processing_strategy = self.process_fully_observed_cases
        elif self.occlusion_process == 'occlusion_simulation':
            self.trajectory_processing_strategy = self.process_cases_with_simulated_occlusions
        else:
            raise NotImplementedError

        self.impute = parser.get('impute', False)
        if self.impute:
            assert self.occlusion_process != 'fully_observed'

        # we are preparing reflect padded versions of the dataset, so that it becomes quicker to process the dataset.
        # the reason why we need to produce reflect padded images is that we will need a full representation of the
        # scene image (as the model requires a global map). With the known SDD pixel to meter ratios and a desired
        # scene side length of <self.map_side> meters, the required padding is equal to the following value.
        # (note that if you need to change the desired side length to a different value, then you will need to
        # recompute the required padding)
        self.padding_px = 2075
        self.padded_images_path = os.path.join(REPO_ROOT, 'datasets', 'SDD', f'padded_images_{self.padding_px}')
        self.make_padded_scene_images()

        self.T_obs = dataset.T_obs
        self.T_pred = dataset.T_pred
        self.T_total = self.T_obs + self.T_pred
        self.timesteps = torch.arange(-dataset.T_obs, dataset.T_pred) + 1
        self.frame_skip = int(dataset.orig_fps // dataset.fps)
        self.lookup_time_window = np.arange(0, self.T_total) * self.frame_skip

    def get_map_crop_coordinates(self) -> Tensor:       # [2, 2]
        return Tensor(
            [[-self.map_side, -self.map_side],
             [self.map_side, self.map_side]]
        ) * self.traj_scale / 2

    def get_map_homography(self) -> Tensor:             # [3, 3]
        return Tensor(
            [[self.map_resolution / self.map_side, 0., self.map_resolution / 2],
             [0., self.map_resolution / self.map_side, self.map_resolution / 2],
             [0., 0., 1.]]
        )

    def make_padded_scene_images(self):
        os.makedirs(self.padded_images_path, exist_ok=True)
        for scene in os.scandir(self.image_path):
            for video in os.scandir(scene):
                save_padded_img_path = os.path.join(self.padded_images_path,
                                                    f"{scene.name}_{video.name}_padded_img.jpg")
                if os.path.exists(save_padded_img_path):
                    continue
                else:
                    image_path = os.path.join(video, 'reference.jpg')
                    assert os.path.exists(image_path)
                    img = cv2.imread(image_path)
                    padded_img = cv2.copyMakeBorder(
                        img, self.padding_px, self.padding_px, self.padding_px, self.padding_px, cv2.BORDER_REFLECT_101
                    )
                    cv2.imwrite(save_padded_img_path, padded_img)
                    print(f"Saving padded image of {scene.name} {video.name}, with padding {self.padding_px}, under:\n"
                          f"{save_padded_img_path}")

    def last_observed_timesteps(
            self,
            last_obs_indices: Tensor    # [N]
    ) -> Tensor:                        # [N]
        return self.timesteps[last_obs_indices]

    def predict_mask(
            self,
            last_obs_indices: Tensor    # [N]
    ) -> Tensor:                        # [N, T]
        predict_mask = torch.full([last_obs_indices.shape[0], self.T_total], False)
        pred_indices = last_obs_indices + 1  # [N]
        for i, pred_idx in enumerate(pred_indices):
            predict_mask[i, pred_idx:] = True
        return predict_mask

    def agent_grid(self, ids: Tensor    # [N]
                   ) -> Tensor:         # [N, T]
        return torch.hstack([ids.unsqueeze(1)] * self.T_total)

    def timestep_grid(self, ids: Tensor     # [N]
                      ) -> Tensor:          # [N, T]
        return torch.vstack([self.timesteps] * ids.shape[0])

    def __len__(self) -> int:
        return len(self.occlusion_table)

    def crop_scene_map(
            self,
            # scene_map: TorchGeometricMap
            scene_map_manager: MapManager
    ):
        # scene_map.crop(crop_coords=scene_map.to_map_points(self.map_crop_coords), resolution=self.map_resolution)
        # scene_map.set_homography(matrix=self.map_homography)
        cropping_coordinates = scene_map_manager.to_map_points(self.map_crop_coords)
        scene_map_manager.map_cropping(crop_coordinates=cropping_coordinates, resolution=self.map_resolution)
        scene_map_manager.set_homography(matrix=self.map_homography)

    def remove_agents_far_from(self, keep_mask: Tensor, target_point: Tensor, points: Tensor):
        # keep_mask [N]
        # target_point [2]
        # points [N, 2]
        distances = torch.linalg.norm(points - target_point, dim=1)

        idx_sort = torch.argsort(distances, dim=-1)
        sorted_mask = keep_mask[idx_sort]
        agent_mask = (torch.cumsum(sorted_mask, dim=-1) <= self.max_train_agent)
        final_mask = torch.logical_and(sorted_mask, agent_mask)

        keep_mask[:] = False
        keep_mask[idx_sort[final_mask]] = True
        return keep_mask

    def trajectory_processing_without_occlusion(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            # scene_map: TorchGeometricMap,
            scene_map_manager: MapManager,
            m_by_px: float
    ) -> defaultdict:
        obs_mask = torch.ones(trajs.shape[:-1])
        obs_mask[..., self.T_obs:] = False
        # scene_map.set_homography(torch.eye(3))
        scene_map_manager.set_homography(torch.eye(3))

        last_obs_positions = trajs[:, self.T_obs - 1, :]
        center_point = torch.mean(last_obs_positions, dim=0)

        keep_agent_mask = torch.full([trajs.shape[0]], True)
        if trajs.shape[0] > self.max_train_agent:
            # print(f"HEY, WE HAVE TOO MANY: {trajs.shape[0]} (full_obs)")
            keep_agent_mask = self.remove_agents_far_from(
                keep_mask=keep_agent_mask,
                target_point=center_point,
                points=last_obs_positions
            )
            # print(f"{keep_agent_mask=}")

            center_point = torch.mean(last_obs_positions[keep_agent_mask], dim=0)

        scaling = self.traj_scale * m_by_px
        trajs = (trajs - center_point) * scaling
        # scene_map.homography_translation(center_point)
        # scene_map.homography_scaling(1 / scaling)
        scene_map_manager.homography_translation(center_point)
        scene_map_manager.homography_scaling(1 / scaling)

        # cropping the scene map
        self.crop_scene_map(
            # scene_map=scene_map
            scene_map_manager=scene_map_manager
        )

        process_dict['trajs'] = trajs
        process_dict['obs_mask'] = obs_mask
        process_dict['keep_agent_mask'] = keep_agent_mask
        # process_dict['scene_map_image'] = scene_map.image
        process_dict['scene_map_image'] = scene_map_manager.get_map()
        process_dict['center_point'] = center_point

        return process_dict

    def wrapped_trajectory_processing_without_occlusion(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            # scene_map: TorchGeometricMap,
            scene_map_manager: MapManager,
            m_by_px: float
    ) -> defaultdict:

        process_dict = self.trajectory_processing_without_occlusion(
            process_dict=process_dict, trajs=trajs, scene_map_manager=scene_map_manager, m_by_px=m_by_px
        )

        last_obs_indices = torch.full([trajs.shape[0]], self.T_obs - 1)

        occlusion_map = torch.full([self.map_resolution, self.map_resolution], True)
        dist_transformed_occlusion_map = torch.zeros([self.map_resolution, self.map_resolution])
        probability_map = torch.zeros([self.map_resolution, self.map_resolution])
        nlog_probability_map = torch.zeros([self.map_resolution, self.map_resolution])

        process_dict['last_obs_indices'] = last_obs_indices
        process_dict['occlusion_map'] = occlusion_map
        process_dict['dist_transformed_occlusion_map'] = dist_transformed_occlusion_map
        process_dict['probability_map'] = probability_map
        process_dict['nlog_probability_map'] = nlog_probability_map

        return process_dict

    def trajectory_processing_with_occlusion(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            ego: Tensor,
            occluder: Tensor,
            tgt_idx: Tensor,
            # scene_map: TorchGeometricMap,
            scene_map_manager: MapManager,
            px_by_m: float,
            m_by_px: float
    ):
        # mapping trajectories to the scene map coordinate system
        # ego = scene_map.to_map_points(ego)
        # occluder = scene_map.to_map_points(occluder)
        # scene_map.set_homography(torch.eye(3))
        ego = scene_map_manager.to_map_points(ego)
        occluder = scene_map_manager.to_map_points(occluder)
        scene_map_manager.set_homography(torch.eye(3))

        # computing the ego visibility polygon
        # scene_boundary = poly_gen.default_rectangle(corner_coords=scene_map.get_map_dimensions())
        scene_boundary = poly_gen.default_rectangle(corner_coords=scene_map_manager.get_map_dimensions())
        ego_visipoly = visibility.torch_compute_visipoly(
            ego_point=ego.squeeze(),
            occluder=occluder,
            boundary=scene_boundary
        )

        # computing the observation mask
        obs_mask = visibility.torch_occlusion_mask(
            points=trajs.reshape(-1, trajs.shape[-1]),
            ego_visipoly=ego_visipoly
        ).reshape(trajs.shape[:-1])  # [N, T]
        obs_mask[..., self.T_obs:] = False

        if self.impute:
            # copying the unaltered trajectories before we perform the imputation
            true_trajs = trajs.detach().clone()
            true_obs_mask = obs_mask.detach().clone()

            # performing the imputation
            trajs = impute_and_cv_predict(trajs=trajs, obs_mask=obs_mask.to(torch.bool), timesteps=self.timesteps)
            obs_mask = torch.full_like(obs_mask, True)
            obs_mask[..., self.T_obs:] = False
            trajs[~obs_mask.to(torch.bool), :] = true_trajs[~obs_mask.to(torch.bool), :]

            # consider every agent as sufficiently observed
            sufficiently_observed_mask = (torch.sum(true_obs_mask, dim=1) >= 2)                          # [N]
        else:
            # checking for all agents that they have at least 2 observations available for the model to process
            sufficiently_observed_mask = (torch.sum(obs_mask, dim=1) >= 2)                          # [N]

        # computing agents' last observed positions
        last_obs_indices = last_observed_indices(obs_mask=obs_mask)
        last_obs_positions = last_observed_positions(trajs=trajs, last_obs_indices=last_obs_indices)  # [N, 2]

        # further removing agents if we have too many.
        close_keep_mask = torch.full_like(sufficiently_observed_mask, True)
        if torch.sum(sufficiently_observed_mask) > self.max_train_agent:
            # identifying the target agent's last observed position (the agent for whom an occlusion was simulated)
            tgt_last_obs_pos = last_obs_positions[tgt_idx]

            # print(f"HEY, WE HAVE TOO MANY: {torch.sum(sufficiently_observed_mask)} (occlusion)")
            close_keep_mask = self.remove_agents_far_from(
                keep_mask=sufficiently_observed_mask,
                target_point=tgt_last_obs_pos,
                points=last_obs_positions
            )
            # print(f"{close_keep_mask=}")

        keep_agent_mask = torch.logical_and(sufficiently_observed_mask, close_keep_mask)        # [N]

        center_point = torch.mean(last_obs_positions[keep_agent_mask], dim=0)

        # shifting and scaling (to metric) of the trajectories, visibility polygon and scene map coordinates
        scaling = self.traj_scale * m_by_px

        trajs = (trajs - center_point) * scaling
        ego = (ego - center_point) * scaling
        occluder = (occluder - center_point) * scaling

        if self.impute:
            true_trajs = (true_trajs - center_point) * scaling
        ego_visipoly = sg.Polygon((torch.from_numpy(ego_visipoly.coords) - center_point) * scaling)
        # scene_map.homography_translation(center_point)
        # scene_map.homography_scaling(1 / scaling)
        scene_map_manager.homography_translation(center_point)
        scene_map_manager.homography_scaling(1 / scaling)

        # cropping the scene map
        self.crop_scene_map(
            # scene_map=scene_map
            scene_map_manager=scene_map_manager
        )

        # computing the occlusion map and distance transformed occlusion map
        # map_dims = scene_map.get_map_dimensions()
        map_dims = scene_map_manager.get_map_dimensions()
        occ_y = torch.arange(map_dims[0])
        occ_x = torch.arange(map_dims[1])
        xy = torch.dstack((torch.meshgrid(occ_x, occ_y))).reshape((-1, 2))
        # mpath = Path(scene_map.to_map_points(torch.from_numpy(ego_visipoly.coords).to(torch.float32)))
        mpath = Path(scene_map_manager.to_map_points(torch.from_numpy(ego_visipoly.coords).to(torch.float32)))
        occlusion_map = torch.from_numpy(mpath.contains_points(xy).reshape(map_dims)).to(torch.bool).T

        invert_occlusion_map = ~occlusion_map
        dist_transformed_occlusion_map = (torch.where(
            invert_occlusion_map,
            torch.from_numpy(-distance_transform_edt(invert_occlusion_map)),
            torch.from_numpy(distance_transform_edt(occlusion_map))
        ) * scaling).to(torch.float32)

        clipped_map = -torch.clamp(dist_transformed_occlusion_map, min=0.)
        probability_map = torch.nn.functional.softmax(clipped_map.view(-1), dim=0).view(clipped_map.shape)
        nlog_probability_map = -torch.nn.functional.log_softmax(clipped_map.view(-1), dim=0).view(clipped_map.shape)

        process_dict['trajs'] = trajs
        process_dict['obs_mask'] = obs_mask
        process_dict['last_obs_indices'] = last_obs_indices
        process_dict['keep_agent_mask'] = keep_agent_mask
        process_dict['occlusion_map'] = occlusion_map
        process_dict['dist_transformed_occlusion_map'] = dist_transformed_occlusion_map
        process_dict['probability_map'] = probability_map
        process_dict['nlog_probability_map'] = nlog_probability_map
        # process_dict['scene_map_image'] = scene_map.image
        process_dict['scene_map_image'] = scene_map_manager.get_map()

        # TODO: figure out whether this needs to be provided by process_dict in other processing functions
        process_dict['center_point'] = center_point
        process_dict['ego'] = ego
        process_dict['occluder'] = occluder
        if self.impute:
            process_dict['true_trajs'] = true_trajs
            process_dict['true_obs_mask'] = true_obs_mask
        return process_dict

    def process_cases_with_simulated_occlusions(
            self,
            process_dict: defaultdict,
            occlusion_case: Dict,
            trajs: Tensor,
            # scene_map: TorchGeometricMap,
            scene_map_manager: MapManager,
            px_by_m: float,
            m_by_px: float
    ):
        if np.isnan(occlusion_case['ego_point']).any():
            return self.wrapped_trajectory_processing_without_occlusion(
                process_dict=process_dict, trajs=trajs, scene_map_manager=scene_map_manager, m_by_px=m_by_px
            )
        else:
            tgt_idx = torch.from_numpy(occlusion_case['target_agent_indices']).squeeze()
            return self.trajectory_processing_with_occlusion(
                process_dict=process_dict,
                trajs=trajs,
                ego=torch.from_numpy(occlusion_case['ego_point']).to(torch.float32).unsqueeze(0),
                occluder=torch.from_numpy(np.vstack(occlusion_case['occluders'][0])).to(torch.float32),
                tgt_idx=tgt_idx,
                scene_map_manager=scene_map_manager,
                px_by_m=px_by_m, m_by_px=m_by_px
            )

    def process_fully_observed_cases(
            self,
            process_dict: defaultdict,
            occlusion_case: Dict,
            trajs: Tensor,
            # scene_map: TorchGeometricMap,
            scene_map_manager: MapManager,
            px_by_m: float,
            m_by_px: float
    ):
        return self.wrapped_trajectory_processing_without_occlusion(
            process_dict=process_dict, trajs=trajs, scene_map_manager=scene_map_manager, m_by_px=m_by_px
        )

    def __getitem__(self, idx: int) -> Dict:
        # look up the row in the occlusion_table
        occlusion_case = self.occlusion_table.iloc[idx]
        sim_id, scene, video, timestep, trial = occlusion_case.name

        # find the corresponding index in self.lookuptable
        lookup_idx = occlusion_case['lookup_idx']
        lookup_row = self.lookuptable.iloc[lookup_idx]

        # extract the reference image
        image_path = os.path.join(self.padded_images_path, f'{scene}_{video}_padded_img.jpg')
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self.to_torch_image(image)
        # scene_map = TorchGeometricMap(
        #     map_image=image, homography=torch.eye(3)
        # )
        _scene_map = MAP_DICT[self.with_rgb_map](image_path=image_path)
        map_homography = HomographyMatrix(matrix=torch.eye(3))
        scene_map_mgr = MapManager(map_object=_scene_map, homography=map_homography)

        # scene_map.homography_translation(Tensor([self.padding_px, self.padding_px]))
        scene_map_mgr.homography_translation(Tensor([self.padding_px, self.padding_px]))

        # generate a time window to extract the relevant section of the scene
        lookup_time_window = self.lookup_time_window + timestep

        instance_df = self.frames.loc[(scene, video)]
        instance_df = instance_df[instance_df['frame'].isin(lookup_time_window)]

        # extract the trajectory data and corresponding agent identities
        trajs = torch.empty([len(lookup_row['targets']), self.T_total, 2])  # [N, T, 2]
        ids = torch.empty([len(lookup_row['targets'])], dtype=torch.int64)
        for i, agent in enumerate(lookup_row['targets']):
            agent_df = instance_df[instance_df['Id'] == agent].sort_values(by=['frame'])
            ids[i] = agent
            trajs[i, :, 0] = torch.from_numpy(agent_df['x'].values)
            trajs[i, :, 1] = torch.from_numpy(agent_df['y'].values)

        # extract the metric / pixel space coordinate conversion factors
        px_by_m = self.coord_conv.loc[scene, video]['px/m']
        m_by_px = self.coord_conv.loc[scene, video]['m/px']

        # prepare for random rotation by choosing a rotation angle and rotating the map
        theta_rot = np.random.rand() * 360 * self.rand_rot_scene
        # scene_map.rotate_around_center(theta=theta_rot)
        scene_map_mgr.rotate_around_center(theta=theta_rot)

        # mapping the trajectories to scene map coordinate system
        # trajs = scene_map.to_map_points(trajs)
        trajs = scene_map_mgr.to_map_points(trajs)

        process_dict = defaultdict(None)
        process_dict = self.trajectory_processing_strategy(
            process_dict=process_dict,
            occlusion_case=occlusion_case,
            trajs=trajs,
            # scene_map=scene_map,
            scene_map_manager=scene_map_mgr,
            px_by_m=px_by_m,
            m_by_px=m_by_px
        )

        # identifying agents who are outside the global scene map
        # inside_map_mask = ~torch.any(torch.any(torch.abs(trajs) >= 0.95 * 0.5 * self.map_side, dim=-1), dim=-1)
        # print(f"{inside_map_mask=}")

        # removing agent surplus
        ids = ids[process_dict['keep_agent_mask']]
        trajs = process_dict['trajs'][process_dict['keep_agent_mask']]
        obs_mask = process_dict['obs_mask'][process_dict['keep_agent_mask']]
        last_obs_indices = process_dict['last_obs_indices'][process_dict['keep_agent_mask']]

        if self.impute:
            if 'true_trajs' in process_dict.keys():
                true_trajs = process_dict['true_trajs'][process_dict['keep_agent_mask']]
                true_obs_mask = process_dict['true_obs_mask'][process_dict['keep_agent_mask']].to(torch.bool)
            else:
                true_trajs = process_dict['trajs'][process_dict['keep_agent_mask']]
                true_obs_mask = process_dict['obs_mask'][process_dict['keep_agent_mask']].to(torch.bool)
        else:
            true_trajs, true_obs_mask = None, None

        obs_mask = obs_mask.to(torch.bool)

        pred_mask = self.predict_mask(last_obs_indices=last_obs_indices)
        agent_grid = self.agent_grid(ids=ids)
        timestep_grid = self.timestep_grid(ids=ids)

        obs_trajs = trajs[obs_mask, ...]
        obs_vel = observed_velocity(trajs=trajs, obs_mask=obs_mask)
        obs_vel_seq = obs_vel[obs_mask, ...]
        obs_ids = agent_grid[obs_mask, ...]
        obs_timesteps = timestep_grid[obs_mask, ...]
        last_obs_positions = last_observed_positions(trajs=trajs, last_obs_indices=last_obs_indices)
        last_obs_timesteps = self.last_observed_timesteps(last_obs_indices=last_obs_indices)

        pred_trajs = trajs.transpose(0, 1)[pred_mask.T, ...]
        pred_vel = true_velocity(trajs=trajs)
        pred_vel_seq = pred_vel.transpose(0, 1)[pred_mask.T, ...]
        pred_ids = agent_grid.T[pred_mask.T, ...]
        pred_timesteps = timestep_grid.T[pred_mask.T, ...]

        data_dict = {
            'trajectories': trajs,
            'velocities': pred_vel,
            'observation_mask': obs_mask,
            'observed_velocities': obs_vel,
            # 'pred_velocities': pred_vel,
            # 'prediction_mask': pred_mask,

            'identities': ids,
            'timesteps': self.timesteps,

            'obs_identity_sequence': obs_ids,
            'obs_position_sequence': obs_trajs,
            'obs_velocity_sequence': obs_vel_seq,
            'obs_timestep_sequence': obs_timesteps,
            'last_obs_positions': last_obs_positions,
            'last_obs_timesteps': last_obs_timesteps,

            'pred_identity_sequence': pred_ids,
            'pred_position_sequence': pred_trajs,
            'pred_velocity_sequence': pred_vel_seq,
            'pred_timestep_sequence': pred_timesteps,

            'scene_orig': torch.zeros([2]),

            'occlusion_map': process_dict['occlusion_map'],
            'dist_transformed_occlusion_map': process_dict['dist_transformed_occlusion_map'],
            'probability_occlusion_map': process_dict['probability_map'],
            'nlog_probability_occlusion_map': process_dict['nlog_probability_map'],
            # 'scene_map': scene_map.image,
            'scene_map': scene_map_mgr.get_map(),
            'map_homography': self.map_homography,
            'theta': theta_rot,
            'center_point': process_dict['center_point'],
            'ego': process_dict.get('ego', torch.full([1, 2], float('nan'))),
            'occluder': process_dict.get('occluder', torch.full([2, 2], float('nan'))),

            'scene': scene,
            'video': video,
            'seq': f'{scene}_{video}',
            'frame': timestep,

            'true_trajectories': true_trajs if self.impute else None,
            'true_observation_mask': true_obs_mask if self.impute else None
        }

        # # visualization stuff
        # fig, ax = plt.subplots(1, 5)
        # self.visualize(
        #     data_dict=data_dict,
        #     draw_ax=ax[0],
        #     draw_ax_sequences=ax[1],
        #     draw_ax_dist_transformed_map=ax[2],
        #     draw_ax_probability_map=ax[3],
        #     draw_ax_nlog_probability_map=ax[4]
        # )
        # plt.show()

        return data_dict


class MomentaryTorchDataGeneratorSDD(TorchDataGeneratorSDD):

    def __init__(self, parser: Config, log: Optional[TextIOWrapper] = None, split: str = 'train',
                 momentary_t_obs: int = 2):
        super().__init__(parser=parser, log=log, split=split)

        assert 1 < momentary_t_obs < self.T_obs
        assert self.occlusion_process == 'fully_observed'
        assert not self.impute

        timestep_shift = self.T_obs - momentary_t_obs
        self.T_obs = momentary_t_obs
        self.T_total = self.T_obs + self.T_pred
        self.timesteps = torch.arange(-self.T_obs, self.T_pred) + 1
        self.lookup_time_window = (np.arange(0, self.T_total) + timestep_shift) * self.frame_skip


class PresavedDatasetSDD(Dataset):
    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    # This is a "quick fix".
    # We performed the wrong px/m coordinate conversion when computing the distance transformed map.
    # Ideally we should correct this in the TorchDatasetGenerator class.
    # the real fix is to have the distance transformed map scaled by the proper factor:
    # TorchDatasetGenerator.map_side / TorchDatasetGenerator.map_resolution
    coord_conv_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'coordinates_conversion.txt')
    coord_conv_table = pd.read_csv(coord_conv_dir, sep=';', index_col=('scene', 'video'))

    def __init__(self, parser: Config, log: Optional[TextIOWrapper] = None, split: str = 'train'):

        self.quick_fix = parser.get('quick_fix', False)   # again, the quick fix should be result upstream, in the TorchDatasetGenerator class.

        self.split = split

        assert split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"
        assert parser.occlusion_process in ['fully_observed', 'occlusion_simulation']

        prnt_str = "\n-------------------------- loading %s data --------------------------" % split
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

        # occlusion process specific parameters
        self.occlusion_process = parser.get('occlusion_process', 'fully_observed')
        self.impute = parser.get('impute', False)
        assert not self.impute or self.occlusion_process != 'fully_observed'
        self.momentary = parser.get('momentary', False)
        assert not (self.momentary and self.occlusion_process != 'fully_observed')

        # map specific parameters
        self.map_side = parser.get('scene_side_length', 80.0)               # [m]
        self.map_resolution = parser.get('global_map_resolution', 800)      # [px]
        self.map_homography = torch.Tensor(
            [[self.map_resolution / self.map_side, 0., self.map_resolution / 2],
             [0., self.map_resolution / self.map_side, self.map_resolution / 2],
             [0., 0., 1.]]
        )

        # timesteps specific parameters
        self.T_obs = parser.past_frames
        self.T_pred = parser.future_frames
        self.T_total = self.T_obs + self.T_pred
        self.timesteps = torch.arange(-self.T_obs, self.T_pred) + 1

        # dataset identification
        self.dataset_name = None
        self.dataset_dir = None
        self.set_dataset_name_and_dir(log=log)
        assert os.path.exists(self.dataset_dir)
        prnt_str = f"Dataset directory is:\n{self.dataset_dir}"
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

    def set_dataset_name_and_dir(self, log: Optional[TextIOWrapper] = None):
        try_name = self.occlusion_process
        if self.impute:
            try_name += '_imputed'
        if self.momentary:
            try_name += f'_momentary_{self.T_obs}'
        try_dir = os.path.join(self.presaved_datasets_dir, try_name, self.split)
        if os.path.exists(try_dir):
            self.dataset_name = try_name
            self.dataset_dir = try_dir
        else:
            prnt_str = "Couldn't find full dataset path, trying with tiny dataset instead..."
            print_log(prnt_str, log=log) if log is not None else print(prnt_str)
            try_name += '_tiny'
            self.dataset_name = try_name
            self.dataset_dir = os.path.join(self.presaved_datasets_dir, self.dataset_name, self.split)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError

    def apply_quick_fix(self, data_dict: Dict):
        scene, video = data_dict['seq'].split('_')
        px_by_m = self.coord_conv_table.loc[scene, video]['px/m']

        data_dict['dist_transformed_occlusion_map'] *= px_by_m * self.map_side / self.map_resolution

        # clipped_map = -torch.clamp(data_dict['dist_transformed_occlusion_map'], min=0.)
        # data_dict['probability_occlusion_map'] = torch.nn.functional.softmax(
        #     clipped_map.view(-1), dim=0
        # ).view(clipped_map.shape)
        # data_dict['nlog_probability_occlusion_map'] = -torch.nn.functional.log_softmax(
        #     clipped_map.view(-1), dim=0
        # ).view(clipped_map.shape)

        data_dict['clipped_dist_transformed_occlusion_map'] = torch.clamp(
            data_dict['dist_transformed_occlusion_map'], min=0.
        )

        del data_dict['probability_occlusion_map']
        del data_dict['nlog_probability_occlusion_map']

        return data_dict


class PickleDatasetSDD(PresavedDatasetSDD):

    def __init__(self, parser: Config, log: Optional[TextIOWrapper] = None, split: str = 'train'):
        super().__init__(parser=parser, log=log, split=split)

        self.pickle_files = sorted(glob.glob1(self.dataset_dir, "*.pickle"))

        if parser.get('difficult', False):
            print("KEEPING ONLY THE DIFFICULT CASES")

            # verify the dataset is the correct configuration
            assert self.occlusion_process == 'occlusion_simulation'
            assert not self.impute
            assert not self.momentary

            from utils.performance_analysis import get_perf_scores_df

            cv_perf_df = get_perf_scores_df(
                experiment_name='const_vel_occlusion_simulation',
                model_name='untrained',
                split=self.split,
                drop_idx=False
            )
            cv_perf_df = cv_perf_df[cv_perf_df['past_pred_length'] > 0]
            cv_perf_df = cv_perf_df[cv_perf_df['OAC_t0'] == 0.]

            difficult_instances = cv_perf_df.index.get_level_values('idx').unique().tolist()
            difficult_instances = [f'{instance:08}.pickle' for instance in difficult_instances]

            difficult_mask = np.in1d(self.pickle_files, difficult_instances)

            difficult_pickle_files = list(np.array(self.pickle_files)[difficult_mask])
            self.pickle_files = difficult_pickle_files

        elif self.split == 'val' and parser.get('validation_set_size', None) is not None:
            assert len(self.pickle_files) > parser.validation_set_size
            # Training set might be quite large. When that is the case, we prefer to validate after every
            # <parser.validation_freq> batches rather than every epoch (i.e., validating multiple times per epoch).
            # train / val split size ratios are typically ~80/20. Our desired validation set size should then be:
            #       <parser.validation_freq> * 20/80
            # We let the user choose the validation set size.
            # The dataset will then be effectively reduced to <parser.validation_set_size>
            required_val_set_size = parser.validation_set_size
            keep_instances = np.linspace(0, len(self.pickle_files)-1, num=required_val_set_size).round().astype(int)

            assert np.all(keep_instances[1:] != keep_instances[:-1])        # verifying no duplicates

            keep_pickle_files = [self.pickle_files[i] for i in keep_instances]

            prnt_str = f"Val set size too large! --> {len(self.pickle_files)} " \
                       f"(validating after every {parser.validation_freq} batch).\n" \
                       f"Reducing val set size to {len(keep_pickle_files)}."
            print_log(prnt_str, log=log) if log is not None else print(prnt_str)
            self.pickle_files = keep_pickle_files
            assert len(self.pickle_files) == required_val_set_size

        assert self.__len__() != 0
        prnt_str = f'total number of samples: {self.__len__()}'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)
        prnt_str = f'------------------------------ done --------------------------------\n'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx: int) -> Dict:
        filename = self.pickle_files[idx]

        with open(os.path.join(self.dataset_dir, filename), 'rb') as f:
            data_dict = pickle.load(f)

        data_dict['instance_name'] = filename
        data_dict['timesteps'] = self.timesteps
        data_dict['scene_orig'] = torch.zeros([2])

        if self.impute:
            if data_dict['true_trajectories'] is None:
                data_dict['true_trajectories'] = data_dict['trajectories']
            if data_dict['true_observation_mask'] is None:
                data_dict['true_observation_mask'] = data_dict['observation_mask']
            data_dict['imputation_mask'] = data_dict['true_observation_mask'][data_dict['observation_mask']]

        # Quick Fix
        if self.quick_fix:
            data_dict = self.apply_quick_fix(data_dict)

        return data_dict


class HDF5DatasetSDD(PresavedDatasetSDD):

    def __init__(self, parser: Config, log: Optional[TextIOWrapper] = None, split: str = 'train'):
        super().__init__(parser=parser, log=log, split=split)

        self.hdf5_file = os.path.join(self.dataset_dir, 'dataset.h5')
        assert os.path.exists(self.hdf5_file)

        # For integrating the hdf5 dataset into the Pytorch class,
        # we follow the principles recommended by Piotr Januszewski:
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.h5_dataset = None
        with h5py.File(self.hdf5_file, 'r') as h5_file:
            instances = sorted([int(key) for key in h5_file.keys() if key.isdecimal()])
            self.instance_names = [f'{instance:08}' for instance in instances]
            self.instance_nums = list(range(len(self.instance_names)))

            self.separate_dataset_keys = [key for key in h5_file.keys() if not key.isdecimal() and key not in ['seq', 'frame']]

        if parser.get('difficult', False):
            print("KEEPING ONLY THE DIFFICULT CASES")

            # verify the dataset is the correct configuration
            assert self.occlusion_process == 'occlusion_simulation'
            assert not self.impute
            assert not self.momentary

            from utils.performance_analysis import get_perf_scores_df

            cv_perf_df = get_perf_scores_df(
                experiment_name='const_vel_occlusion_simulation',
                model_name='untrained',
                split=self.split,
                drop_idx=False
            )
            cv_perf_df = cv_perf_df[cv_perf_df['past_pred_length'] > 0]
            cv_perf_df = cv_perf_df[cv_perf_df['OAC_t0'] == 0.]

            difficult_instances = cv_perf_df.index.get_level_values('idx').unique().tolist()
            difficult_instances = [f'{instance:08}' for instance in difficult_instances]
            difficult_mask = np.in1d(self.instance_names, difficult_instances)

            difficult_instance_names = list(np.array(self.instance_names)[difficult_mask])
            difficult_instance_nums = list(np.array(self.instance_nums)[difficult_mask])
            self.instance_names = difficult_instance_names
            self.instance_nums = difficult_instance_nums

        elif self.split == 'val' and parser.get('validation_set_size', None) is not None:
            assert len(self.instance_names) > parser.validation_set_size
            # Training set might be quite large. When that is the case, we prefer to validate after every
            # <parser.validation_freq> batches rather than every epoch (i.e., validating multiple times per epoch).
            # train / val split size ratios are typically ~80/20. Our desired validation set size should then be:
            #       <parser.validation_freq> * 20/80
            # We let the user choose the validation set size.
            # The dataset will then be effectively reduced to <parser.validation_set_size>
            required_val_set_size = parser.validation_set_size
            keep_instances = np.linspace(0, len(self.instance_names)-1, num=required_val_set_size).round().astype(int)

            assert np.all(keep_instances[1:] != keep_instances[:-1])        # verifying no duplicates

            keep_instance_names = [self.instance_names[i] for i in keep_instances]
            keep_instance_nums = [self.instance_nums[i] for i in keep_instances]

            prnt_str = f"Val set size too large! --> {len(self.instance_names)} " \
                       f"(validating after every {parser.validation_freq} batch).\n" \
                       f"Reducing val set size to {len(keep_instances)}."
            print_log(prnt_str, log=log) if log is not None else print(prnt_str)
            self.instance_names = keep_instance_names
            self.instance_nums = keep_instance_nums
            assert len(self.instance_names) == required_val_set_size

        assert len(self.instance_names) == len(self.instance_nums)

        assert self.__len__() != 0
        prnt_str = f'total number of samples: {self.__len__()}'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)
        prnt_str = f'------------------------------ done --------------------------------\n'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

    def __len__(self):
        return len(self.instance_names)

    def __getitem__(self, idx: int) -> Dict:
        instance_name = self.instance_names[idx]
        instance_num = self.instance_nums[idx]

        if self.h5_dataset is None:
            self.h5_dataset = h5py.File(self.hdf5_file, 'r')

        data_dict = dict()

        data_dict['seq'] = self.h5_dataset['seq'].asstr()[instance_num]
        data_dict['frame'] = self.h5_dataset['frame'][instance_num]

        # reading instance elements which do not change shapes from their respective datasets
        for key in self.separate_dataset_keys:
            data_dict[key] = torch.from_numpy(self.h5_dataset[key][instance_num])

        # reading remaining instance elements from the corresponding group
        for key in self.h5_dataset[instance_name].keys():
            data_dict[key] = torch.from_numpy(self.h5_dataset[instance_name][key][()])

        data_dict['instance_name'] = instance_name
        data_dict['timesteps'] = self.timesteps
        data_dict['scene_orig'] = torch.zeros([2])

        if self.impute:
            if 'true_trajectories' not in data_dict.keys():
                data_dict['true_trajectories'] = data_dict['trajectories']
            if 'true_observation_mask' not in data_dict.keys():
                data_dict['true_observation_mask'] = data_dict['observation_mask']
            data_dict['imputation_mask'] = data_dict['true_observation_mask'][data_dict['observation_mask']]

        # Quick Fix
        if self.quick_fix:
            data_dict = self.apply_quick_fix(data_dict)

        return data_dict

    def get_instance_idx(self, instance_name: Optional[str] = None, instance_num: Optional[int] = None):
        assert instance_name is not None or instance_num is not None

        if instance_name is not None and instance_name in self.instance_names:
            return self.instance_names.index(instance_name)

        if instance_num is not None and instance_num in self.instance_nums:
            return self.instance_nums.index(instance_num)

        return None


dataset_dict = {
        'hdf5': HDF5DatasetSDD,
        'pickle': PickleDatasetSDD,
        'torch_preprocess': TorchDataGeneratorSDD
    }


if __name__ == '__main__':
    import utils.sdd_visualize
    from utils.utils import prepare_seed

    # config_str = 'dataset_fully_observed_momentary_2_no_rand_rot'
    # config_str = 'dataset_fully_observed_no_rand_rot'
    # config_str = 'dataset_occlusion_simulation_no_rand_rot'
    # config_str = 'dataset_occlusion_simulation'
    # config_str = 'original_100_pre'
    # dataset_class = 'torch_preprocess'
    # dataset_class = 'hdf5'
    # dataset_class = 'pickle'
    # split = 'test'
    # split = 'train'

    # config_str, dataset_class, split = 'dataset_fully_observed', 'hdf5', 'train'
    config_str, dataset_class, split = 'dataset_fully_observed', 'torch_preprocess', 'train'
    config_str, dataset_class, split = 'dataset_fully_observed', 'pickle', 'train'
    # config_str, dataset_class, split = 'dataset_fully_observed', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'torch_preprocess', 'test'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'hdf5', 'test'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'pickle', 'train'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'pickle', 'val'

    cfg = Config(config_str)
    prepare_seed(42)
    torch_dataset = dataset_dict[dataset_class](parser=cfg, log=None, split=split)

    n_row, n_col = 4, 6
    fig, ax = plt.subplots(n_row, n_col)

    start_idx = 100

    for i in range(n_row * n_col):

        row_i, col_i = i // n_col, i % n_col

        out_dict = torch_dataset.__getitem__(i + start_idx)
        if 'map_homography' not in out_dict.keys():
            out_dict['map_homography'] = torch_dataset.map_homography

        # print(f"{out_dict=}")
        utils.sdd_visualize.visualize(
            data_dict=out_dict,
            draw_ax=ax[row_i, col_i],
            # draw_ax_sequences=ax[1],
            # draw_ax_dist_transformed_map=ax[2],
            draw_ax_dist_transformed_map=None,
            # draw_ax_probability_map=ax[3],
            draw_ax_probability_map=None,
            # draw_ax_nlog_probability_map=ax[4],
            draw_ax_nlog_probability_map=None
        )

    plt.show()
    ###################################################################################################################
    print("Goodbye!")
