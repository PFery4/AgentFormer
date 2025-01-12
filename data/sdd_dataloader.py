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
from scipy.ndimage import distance_transform_edt
import struct
from collections import defaultdict

# from data.map import TorchGeometricMap
from data.trajectory_operations import impute_and_cv_predict, \
    last_observed_indices, last_observed_positions, \
    observed_velocity, true_velocity
from data.map import \
    compute_occlusion_map, compute_distance_transformed_map, compute_probability_map, compute_nlog_probability_map, \
    HomographyMatrix, MapManager, MAP_DICT
from utils.config import Config, REPO_ROOT
from utils.performance_analysis import get_difficult_occlusion_indices

from typing import Dict, Optional
Tensor = torch.Tensor

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
import src.visualization.sdd_visualize as visualize
import src.occlusion_simulation.visibility as visibility
import src.occlusion_simulation.polygon_generation as poly_gen
import src.data.config as sdd_conf


class BaseDataset:

    # we are preparing reflect padded versions of the dataset, so that it becomes quicker to process the dataset.
    # the reason why we need to produce reflect padded images is that we will need a full representation of the
    # scene image (as the model requires a global map). With the known SDD pixel to meter ratios and a desired
    # scene side length of <self.map_side> meters, the required padding is equal to the following value.
    # (note that if you need to change the desired side length to a different value, then you will need to
    # recompute the required padding)
    padding_px = 2075
    padded_images_path = os.path.join(REPO_ROOT, 'datasets', 'SDD', f'padded_images_{padding_px}')

    def __init__(self, parser: Config, split: str = 'train'):
        self.split = split
        assert self.split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"Error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        self.occlusion_process = str(parser.occlusion_process)      # 'fully_observed' | 'occlusion_simulation'
        assert self.occlusion_process in ['fully_observed', 'occlusion_simulation']
        self.impute = bool(parser.impute)
        if self.impute:
            assert self.occlusion_process == 'occlusion_simulation'

        # timesteps specific parameters
        self.T_obs = int(parser.past_frames)
        self.T_pred = int(parser.future_frames)
        self.T_total = int(self.T_obs + self.T_pred)
        self.timesteps = torch.arange(-self.T_obs, self.T_pred) + 1

        # map specific parameters
        self.map_side = float(parser.scene_side_length)             # [m]
        self.map_resolution = int(parser.global_map_resolution)     # [px]
        self.traj_scale = float(parser.traj_scale)
        self.with_rgb_map = bool(parser.with_rgb_map)
        assert self.map_resolution % 8 == 0

        self.map_crop_coords = self.get_map_crop_coordinates()
        self.map_homography = self.get_map_homography()

        # dataset identification
        self.dataset_name = self.get_dataset_name()

    def get_dataset_name(self):
        dset_name = self.occlusion_process
        if self.impute:
            dset_name += '_imputed'
        return dset_name

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

    def agent_grid(self, ids: Tensor    # [N]
                   ) -> Tensor:         # [N, T]
        return torch.hstack([ids.unsqueeze(1)] * self.T_total)

    def timestep_grid(self, ids: Tensor     # [N]
                      ) -> Tensor:          # [N, T]
        return torch.vstack([self.timesteps] * ids.shape[0])

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

    def get_scene_map_manager(self, image_path: os.PathLike) -> MapManager:
        scene_map = MAP_DICT[self.with_rgb_map](image_path=image_path)
        homography = HomographyMatrix(matrix=torch.eye(3))
        return MapManager(map_object=scene_map, homography=homography)

    def crop_scene_map(self, scene_map_manager: MapManager):
        scene_map_manager.map_cropping(
            crop_coordinates=scene_map_manager.to_map_points(self.map_crop_coords),
            resolution=self.map_resolution
        )
        scene_map_manager.set_homography(matrix=self.map_homography)


class TorchDataGeneratorSDD(BaseDataset, Dataset):
    def __init__(self, parser: Config, split: str = 'train'):
        BaseDataset.__init__(self, parser=parser, split=split)

        self.sdd_config = sdd_conf.get_config(os.path.join(sdd_conf.REPO_ROOT, parser.sdd_config_file_name))
        dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config, split=self.split)

        # extract <StanfordDroneDatasetWithOcclusionSim> relevant data
        self.coord_conv = dataset.coord_conv
        self.frames = dataset.frames
        self.lookuptable = dataset.lookuptable
        self.occlusion_table = dataset.occlusion_table
        self.image_path = os.path.join(dataset.SDD_root, 'annotations')
        assert os.path.exists(self.image_path)

        self.rand_rot_scene = bool(parser.rand_rot_scene)
        self.max_train_agent = int(parser.max_train_agent)
        self.distance_threshold_occluded_target = self.map_side / 4     # [m]

        strategies = {
            'fully_observed': self.handle_fully_observed_cases,
            'occlusion_simulation': self.handle_cases_with_simulated_occlusions
        }
        self.trajectory_processing_strategy = strategies[self.occlusion_process]

        self.make_padded_scene_images()

        assert self.T_obs == dataset.T_obs
        assert self.T_pred == dataset.T_pred
        self.frame_skip = int(dataset.orig_fps // dataset.fps)
        self.lookup_time_window = np.arange(0, self.T_total) * self.frame_skip

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

    def __len__(self) -> int:
        return len(self.occlusion_table)

    def remove_agents_far_from(
            self,
            keep_mask: Tensor,      # [N]
            target_point: Tensor,   # [2]
            points: Tensor          # [N, 2]
    ) -> Tensor:                    # [N]
        distances = torch.linalg.norm(points - target_point, dim=1)

        idx_sort = torch.argsort(distances, dim=-1)
        sorted_mask = keep_mask[idx_sort]
        agent_mask = (torch.cumsum(sorted_mask, dim=-1) <= self.max_train_agent)
        final_mask = torch.logical_and(sorted_mask, agent_mask)

        keep_mask[:] = False
        keep_mask[idx_sort[final_mask]] = True
        return keep_mask

    def process_fully_observed_trajectories(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            scene_map_manager: MapManager,
            m_by_px: float
    ) -> defaultdict:

        obs_mask = torch.ones(trajs.shape[:-1])
        obs_mask[..., self.T_obs:] = False
        scene_map_manager.set_homography(torch.eye(3))

        last_obs_indices = torch.full([trajs.shape[0]], self.T_obs - 1)
        last_obs_positions = trajs[:, self.T_obs - 1, :]
        center_point = torch.mean(last_obs_positions, dim=0)

        keep_agent_mask = torch.full([trajs.shape[0]], True)
        if trajs.shape[0] > self.max_train_agent:
            keep_agent_mask = self.remove_agents_far_from(
                keep_mask=keep_agent_mask,
                target_point=center_point,
                points=last_obs_positions
            )
            center_point = torch.mean(last_obs_positions[keep_agent_mask], dim=0)

        scaling = self.traj_scale * m_by_px
        trajs = (trajs - center_point) * scaling
        scene_map_manager.homography_translation(center_point)
        scene_map_manager.homography_scaling(1 / scaling)

        self.crop_scene_map(scene_map_manager=scene_map_manager)

        occlusion_map = torch.full([self.map_resolution, self.map_resolution], True)
        dist_transformed_occlusion_map = torch.zeros([self.map_resolution, self.map_resolution])

        process_dict['trajs'] = trajs
        process_dict['obs_mask'] = obs_mask
        process_dict['last_obs_indices'] = last_obs_indices
        process_dict['keep_agent_mask'] = keep_agent_mask
        process_dict['occlusion_map'] = occlusion_map
        process_dict['dist_transformed_occlusion_map'] = dist_transformed_occlusion_map
        process_dict['scene_map_image'] = scene_map_manager.get_map()
        process_dict['center_point'] = center_point

        return process_dict

    def process_trajectories_with_occlusions(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            ego: Tensor,
            occluder: Tensor,
            tgt_idx: Tensor,
            scene_map_manager: MapManager,
            m_by_px: float
    ):
        # mapping trajectories to the scene map coordinate system
        ego = scene_map_manager.to_map_points(ego)
        occluder = scene_map_manager.to_map_points(occluder)
        scene_map_manager.set_homography(torch.eye(3))

        # computing the ego visibility polygon
        scene_boundary = poly_gen.default_rectangle(corner_coords=scene_map_manager.get_map_dimensions())
        ego_visipoly = visibility.torch_compute_visipoly(
            ego_point=ego.squeeze(), occluder=occluder, boundary=scene_boundary
        )

        # computing the observation mask
        obs_mask = visibility.torch_occlusion_mask(
            points=trajs.reshape(-1, trajs.shape[-1]),
            ego_visipoly=ego_visipoly
        ).reshape(trajs.shape[:-1])  # [N, T]
        obs_mask[..., self.T_obs:] = False

        true_trajs = None
        true_obs_mask = None
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

            close_keep_mask = self.remove_agents_far_from(
                keep_mask=sufficiently_observed_mask,
                target_point=tgt_last_obs_pos,
                points=last_obs_positions
            )

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
        scene_map_manager.homography_translation(center_point)
        scene_map_manager.homography_scaling(1 / scaling)

        # cropping the scene map
        self.crop_scene_map(scene_map_manager=scene_map_manager)

        # computing the occlusion map and distance transformed occlusion map
        occlusion_map = compute_occlusion_map(
            map_dimensions=scene_map_manager.get_map_dimensions(),
            visibility_polygon_coordinates=scene_map_manager.to_map_points(
                torch.from_numpy(ego_visipoly.coords).to(torch.float32)
            )
        )
        dist_transformed_occlusion_map = compute_distance_transformed_map(
            occlusion_map=occlusion_map,
            scaling=scaling
        )

        process_dict['trajs'] = trajs
        process_dict['obs_mask'] = obs_mask
        process_dict['last_obs_indices'] = last_obs_indices
        process_dict['keep_agent_mask'] = keep_agent_mask
        process_dict['occlusion_map'] = occlusion_map
        process_dict['dist_transformed_occlusion_map'] = dist_transformed_occlusion_map
        process_dict['scene_map_image'] = scene_map_manager.get_map()
        process_dict['center_point'] = center_point
        process_dict['ego'] = ego
        process_dict['occluder'] = occluder
        if self.impute:
            process_dict['true_trajs'] = true_trajs
            process_dict['true_obs_mask'] = true_obs_mask

        return process_dict

    def handle_cases_with_simulated_occlusions(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            scene_map_manager: MapManager,
            m_by_px: float,
            occlusion_case: Dict,
    ):
        if np.isnan(occlusion_case['ego_point']).any():
            return self.process_fully_observed_trajectories(
                process_dict=process_dict,
                trajs=trajs,
                scene_map_manager=scene_map_manager,
                m_by_px=m_by_px
            )
        else:
            tgt_idx = torch.from_numpy(occlusion_case['target_agent_indices']).squeeze()
            return self.process_trajectories_with_occlusions(
                process_dict=process_dict,
                trajs=trajs,
                ego=torch.from_numpy(occlusion_case['ego_point']).to(torch.float32).unsqueeze(0),
                occluder=torch.from_numpy(np.vstack(occlusion_case['occluders'][0])).to(torch.float32),
                tgt_idx=tgt_idx,
                scene_map_manager=scene_map_manager,
                m_by_px=m_by_px
            )

    def handle_fully_observed_cases(
            self,
            process_dict: defaultdict,
            trajs: Tensor,
            scene_map_manager: MapManager,
            m_by_px: float,
            **unused_kwargs
    ):
        return self.process_fully_observed_trajectories(
            process_dict=process_dict,
            trajs=trajs,
            scene_map_manager=scene_map_manager,
            m_by_px=m_by_px
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
        scene_map_mgr = self.get_scene_map_manager(image_path=image_path)
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
        m_by_px = self.coord_conv.loc[scene, video]['m/px']
        px_by_m = self.coord_conv.loc[scene, video]['px/m']

        # prepare for random rotation by choosing a rotation angle and rotating the map
        theta_rot = np.random.rand() * 360 * self.rand_rot_scene
        scene_map_mgr.rotate_around_center(theta=theta_rot)

        # mapping the trajectories to scene map coordinate system
        trajs = scene_map_mgr.to_map_points(trajs)

        process_dict = defaultdict(None)
        process_dict = self.trajectory_processing_strategy(
            process_dict=process_dict,
            trajs=trajs,
            scene_map_manager=scene_map_mgr,
            m_by_px=m_by_px,
            occlusion_case=occlusion_case
        )

        # We performed the wrong px/m coordinate conversion when computing the distance transformed map.
        # we apply a fix here, ensuring proper rescaling of the distance transformed map.
        process_dict['dist_transformed_occlusion_map'] *= px_by_m * self.map_side / self.map_resolution
        clipped_dist_transformed_occlusion_map = torch.clamp(process_dict['dist_transformed_occlusion_map'], min=0.)

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
            'clipped_dist_transformed_occlusion_map': clipped_dist_transformed_occlusion_map,
            # 'scene_map': scene_map_mgr.get_map(),
            'map_homography': self.map_homography,
            'theta': theta_rot,
            'center_point': process_dict['center_point'],
            'is_occluded': True if 'ego' in process_dict.keys() else False,
            'ego': process_dict.get('ego', torch.full([1, 2], float('nan'))),
            'occluder': process_dict.get('occluder', torch.full([2, 2], float('nan'))),

            'scene': scene,
            'video': video,
            'seq': f'{scene}_{video}',
            'frame': timestep,
            'instance_name': f'{idx:08}',

            # 'true_trajectories': true_trajs if self.impute else None,
            # 'true_observation_mask': true_obs_mask if self.impute else None
        }

        if self.with_rgb_map:
            data_dict.update(
                scene_map=scene_map_mgr.get_map()
            )

        if self.impute:
            data_dict.update(
                true_trajectories=true_trajs,
                true_observation_mask=true_obs_mask
            )

        return data_dict


class HDF5PresavedDatasetSDD(BaseDataset, Dataset):

    dataset_filenames = {False: 'dataset_v2.h5', True: 'legacy_dataset_v2.h5'}
    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    coord_conv_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'coordinates_conversion.txt')
    coord_conv_table = pd.read_csv(coord_conv_dir, sep=';', index_col=('scene', 'video'))

    def __init__(self, parser: Config, split: str = 'train', legacy_mode: bool = False):
        BaseDataset.__init__(self, parser=parser, split=split)
        print("\n-------------------------- loading %s data --------------------------" % split)

        # occlusion map extraction struct format
        self.struct_format = f'{int(self.map_resolution * self.map_resolution / 8)}B'

        # dataset identification
        self.dataset_filename = self.dataset_filenames[legacy_mode]
        dataset_dir_name = f'{self.occlusion_process}_imputed' if self.impute else self.occlusion_process
        self.dataset_dir = os.path.join(self.presaved_datasets_dir, dataset_dir_name, self.split)
        self.hdf5_file = os.path.join(self.dataset_dir, self.dataset_filename)
        assert os.path.exists(self.dataset_dir)
        assert os.path.exists(self.hdf5_file)
        print(f"Dataset directory is:\n{self.dataset_dir}")

        # For integrating the hdf5 dataset into the Pytorch class,
        # we follow the principles recommended by Piotr Januszewski:
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.h5_dataset = None
        self.lookup_indices = None
        self.lookup_datasets = []       # datasets which will have to be indexed using self.lookup_indices
        with h5py.File(self.hdf5_file, 'r') as h5_file:
            for dset_name, dset in h5_file.items():       # str, dataset
                if None in dset.maxshape:
                    self.lookup_datasets.append(dset_name)
            self.lookup_indices = torch.from_numpy(h5_file['lookup_indices'][()])
        self.indices = torch.arange(len(self.lookup_indices))

        # flags for __getitem__ behaviour
        self.with_map_transforms = True
        self.with_occlusion_state = True if self.occlusion_process == 'occlusion_simulation' else False
        self.with_occlusion_objects = True if self.occlusion_process == 'occlusion_simulation' else False
        self.with_occlusion_map_data = True if self.occlusion_process == 'occlusion_simulation' else False

        if legacy_mode:
            print(f"USING LEGACY DATASET")
            self.with_map_transforms = False
            self.with_occlusion_objects = False

        if parser.get('difficult', False):
            print("KEEPING ONLY THE DIFFICULT CASES")
            # verifying the dataset is the correct configuration
            assert self.occlusion_process == 'occlusion_simulation'
            assert not self.impute
            difficult_instances = get_difficult_occlusion_indices(
                split=self.split).get_level_values('idx').unique().tolist()
            self.indices = torch.tensor(difficult_instances, dtype=torch.int64)
            assert len(self.indices) == len(difficult_instances)
            assert torch.all(self.indices[1:] != self.indices[:-1])        # verifying no duplicates

        # dataset subsampling
        elif parser.get('custom_dataset_size', None) is not None:
            assert len(self.indices) > parser.custom_dataset_size
            print(f"Setting a custom dataset size: {parser.custom_dataset_size} (original size: {len(self.indices)})")
            self.indices = torch.from_numpy(np.linspace(
                0, len(self.indices)-1, num=parser.custom_dataset_size
            ).round().astype(int))
            assert len(self.indices) == parser.custom_dataset_size
            assert torch.all(self.indices[1:] != self.indices[:-1])        # verifying no duplicates

        assert self.__len__() != 0
        print(f'total number of samples: {self.__len__()}')
        print(f'------------------------------ done --------------------------------\n')

    def add_instance_identifiers(self, data_dict: Dict, idx: int):
        data_dict['frame'] = self.h5_dataset['frame'][idx].astype(np.int64)
        data_dict['scene'] = self.h5_dataset['scene'].asstr()[idx]
        data_dict['video'] = self.h5_dataset['video'].asstr()[idx]
        data_dict['seq'] = f"{data_dict['scene']}_{data_dict['video']}"
        data_dict['instance_name'] = f'{idx:08}'

    def add_occlusion_state(self, data_dict: Dict, idx: int):
        data_dict['is_occluded'] = bool(self.h5_dataset['is_occluded'][idx])

    def add_occlusion_objects(self, data_dict: Dict, idx: int):
        data_dict['ego'] = torch.from_numpy(self.h5_dataset['ego'][idx]).view(1, 2)
        data_dict['occluder'] = torch.from_numpy(self.h5_dataset['occluder'][idx])

    def add_scene_map_transform_parameters(self, data_dict: Dict, idx: int):
        data_dict['theta'] = float(self.h5_dataset['theta'][idx])
        data_dict['center_point'] = torch.from_numpy(self.h5_dataset['center_point'][idx])

    def add_trajectory_data(self, data_dict):
        agent_grid = self.agent_grid(ids=data_dict['identities'])
        timestep_grid = self.timestep_grid(ids=data_dict['identities'])
        last_obs_indices = last_observed_indices(obs_mask=data_dict['observation_mask'].to(torch.int16))
        pred_mask = self.predict_mask(last_obs_indices=last_obs_indices)

        data_dict['obs_identity_sequence'] = agent_grid[data_dict['observation_mask'], ...]
        data_dict['obs_timestep_sequence'] = timestep_grid[data_dict['observation_mask'], ...]
        data_dict['obs_position_sequence'] = data_dict['trajectories'][data_dict['observation_mask'], ...]
        data_dict['obs_velocity_sequence'] = data_dict['observed_velocities'][data_dict['observation_mask'], ...]

        data_dict['last_obs_positions'] = last_observed_positions(
            trajs=data_dict['trajectories'], last_obs_indices=last_obs_indices
        )
        data_dict['last_obs_timesteps'] = self.last_observed_timesteps(
            last_obs_indices=last_obs_indices
        )

        data_dict['pred_identity_sequence'] = agent_grid.T[pred_mask.T, ...]
        data_dict['pred_position_sequence'] = data_dict['trajectories'].transpose(0, 1)[pred_mask.T, ...]
        data_dict['pred_velocity_sequence'] = data_dict['velocities'].transpose(0, 1)[pred_mask.T, ...]
        data_dict['pred_timestep_sequence'] = timestep_grid.T[pred_mask.T, ...]

    def add_occlusion_map_data(self, data_dict: Dict, idx: int):
        retrieved_bytes = self.h5_dataset['occlusion_map'][idx]
        retrieved_bytes = struct.unpack(self.struct_format, retrieved_bytes)
        retrieved_bytes = [f'{num:08b}' for num in retrieved_bytes]
        retrieved_bytes = "".join(retrieved_bytes)
        processed_occl_map = torch.BoolTensor(
            [int(num) for num in retrieved_bytes]
        ).reshape(self.map_resolution, self.map_resolution)
        data_dict['occlusion_map'] = processed_occl_map

        px_by_m = self.coord_conv_table.loc[data_dict['scene'], data_dict['video']]['px/m']
        m_by_px = self.coord_conv_table.loc[data_dict['scene'], data_dict['video']]['m/px']
        scaling = self.traj_scale * m_by_px
        if not data_dict['is_occluded']:
            dist_transformed_occlusion_map = torch.zeros([self.map_resolution, self.map_resolution])

        else:
            dist_transformed_occlusion_map = compute_distance_transformed_map(
                occlusion_map=processed_occl_map,
                scaling=scaling
            )

        # We performed the wrong px/m coordinate conversion when computing the distance transformed map.
        # we apply a fix here, ensuring proper rescaling of the distance transformed map.
        dist_transformed_occlusion_map *= px_by_m * self.map_side / self.map_resolution
        data_dict['dist_transformed_occlusion_map'] = dist_transformed_occlusion_map
        data_dict['clipped_dist_transformed_occlusion_map'] = torch.clamp(
            dist_transformed_occlusion_map, min=0.
        )

    def add_scene_map_data(self, data_dict: Dict):
        # loading the scene map
        image_path = os.path.join(self.padded_images_path, f"{data_dict['seq']}_padded_img.jpg")
        scene_map_mgr = self.get_scene_map_manager(image_path=image_path)
        scene_map_mgr.homography_translation(Tensor([self.padding_px, self.padding_px]))
        scene_map_mgr.rotate_around_center(theta=data_dict['theta'])
        scene_map_mgr.set_homography(torch.eye(3))
        scene_map_mgr.homography_translation(data_dict['center_point'])
        scene_map_mgr.homography_scaling(
            1 / (self.traj_scale * self.coord_conv_table.loc[data_dict['scene'], data_dict['video']]['m/px'])
        )
        self.crop_scene_map(scene_map_manager=scene_map_mgr)

        data_dict['scene_map'] = scene_map_mgr.get_map()

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        instance_idx = self.indices[idx]
        lookup_idx_start, lookup_idx_end = self.lookup_indices[instance_idx]

        if self.h5_dataset is None:
            self.h5_dataset = h5py.File(self.hdf5_file, 'r')

        data_dict = dict()

        self.add_instance_identifiers(data_dict=data_dict, idx=instance_idx)
        if self.with_map_transforms:
            self.add_scene_map_transform_parameters(data_dict=data_dict, idx=instance_idx)
        if self.with_occlusion_state:
            self.add_occlusion_state(data_dict=data_dict, idx=instance_idx)
        if self.with_occlusion_objects:
            self.add_occlusion_objects(data_dict=data_dict, idx=instance_idx)

        for dset_name in self.lookup_datasets:
            data_dict[dset_name] = torch.from_numpy(self.h5_dataset[dset_name][lookup_idx_start:lookup_idx_end])
        data_dict['identities'] = data_dict['identities'].to(torch.int64)

        self.add_trajectory_data(data_dict=data_dict)
        if self.with_occlusion_map_data:
            self.add_occlusion_map_data(data_dict=data_dict, idx=instance_idx)
        # self.add_scene_map_data(data_dict=data_dict)

        data_dict['timesteps'] = self.timesteps
        data_dict['scene_orig'] = torch.zeros([2])
        data_dict['map_homography'] = self.map_homography

        if self.impute:
            assert 'true_trajectories' in data_dict.keys()
            data_dict['imputation_mask'] = data_dict['true_observation_mask'][data_dict['observation_mask']]

        return data_dict


# TODO: move this class to a DEPRECATED file (no longer used)
class PresavedDatasetSDD(Dataset):
    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    # This is a "quick fix".
    # We performed the wrong px/m coordinate conversion when computing the distance transformed map.
    # Ideally we should correct this in the TorchDatasetGenerator class.
    # the real fix is to have the distance transformed map scaled by the proper factor:
    # TorchDatasetGenerator.map_side / TorchDatasetGenerator.map_resolution
    coord_conv_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'coordinates_conversion.txt')
    coord_conv_table = pd.read_csv(coord_conv_dir, sep=';', index_col=('scene', 'video'))

    def __init__(self, parser: Config, split: str = 'train'):

        self.quick_fix = parser.get('quick_fix', False)   # again, the quick fix should be result upstream, in the TorchDatasetGenerator class.

        self.split = split

        assert split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"
        assert parser.occlusion_process in ['fully_observed', 'occlusion_simulation']

        print("\n-------------------------- loading %s data --------------------------" % split)

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
        self.set_dataset_name_and_dir()
        assert os.path.exists(self.dataset_dir)
        print(f"Dataset directory is:\n{self.dataset_dir}")

    def set_dataset_name_and_dir(self):
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
            print("Couldn't find full dataset path, trying with tiny dataset instead...")
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


# TODO: move this class to a DEPRECATED file (no longer used)
class PickleDatasetSDD(PresavedDatasetSDD):

    def __init__(self, parser: Config, split: str = 'train'):
        super().__init__(parser=parser, split=split)
        print("PICKLE PICKLE PICKLE DATASET DATASET DATASET")

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

            print(
                f"Val set size too large! --> {len(self.pickle_files)} "
                f"(validating after every {parser.validation_freq} batch).\n"
                f"Reducing val set size to {len(keep_pickle_files)}."
            )
            self.pickle_files = keep_pickle_files
            assert len(self.pickle_files) == required_val_set_size

            print(f"{self.pickle_files=}")

        assert self.__len__() != 0
        print(f'total number of samples: {self.__len__()}')
        print(f'------------------------------ done --------------------------------\n')

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
        if self.quick_fix and self.occlusion_process == 'occlusion_simulation':
            data_dict = self.apply_quick_fix(data_dict)

        return data_dict


# TODO: move this class to a DEPRECATED file (no longer used)
class HDF5DatasetSDD(PresavedDatasetSDD):

    def __init__(self, parser: Config, split: str = 'train'):
        super().__init__(parser=parser, split=split)

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

            print(
                f"Val set size too large! --> {len(self.instance_names)} "
                f"(validating after every {parser.validation_freq} batch).\n"
                f"Reducing val set size to {len(keep_instances)}."
            )
            self.instance_names = keep_instance_names
            self.instance_nums = keep_instance_nums
            assert len(self.instance_names) == required_val_set_size

        assert len(self.instance_names) == len(self.instance_nums)

        assert self.__len__() != 0
        print(f'total number of samples: {self.__len__()}')
        print(f'------------------------------ done --------------------------------\n')

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
        if self.quick_fix and self.occlusion_process == 'occlusion_simulation':
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
        # 'hdf5': HDF5DatasetSDD,           # TODO: remove once done
        # 'pickle': PickleDatasetSDD,       # TODO: remove once done
        'torch': TorchDataGeneratorSDD,
        'hdf5': HDF5PresavedDatasetSDD
    }
