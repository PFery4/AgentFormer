import os.path
from io import TextIOWrapper

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.colors as colors
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

from data.map import TorchGeometricMap
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


class TorchDataGeneratorSDD(Dataset):
    def __init__(self, parser: Config, log: Optional[TextIOWrapper] = None, split: str = 'train'):
        self.split = split
        assert split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        prnt_str = "\n-------------------------- loading %s data --------------------------" % split
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)
        self.sdd_config = sdd_conf.get_config(parser.sdd_config_file_name)
        dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config, split=self.split)
        prnt_str = f"instantiating dataloader from {dataset.__class__} class"
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

        # stealing StanfordDroneDatasetWithOcclusionSim relevant data
        self.coord_conv = dataset.coord_conv
        self.frames = dataset.frames
        self.lookuptable = dataset.lookuptable
        self.occlusion_table = dataset.occlusion_table
        self.image_path = os.path.join(dataset.root, 'annotations')
        assert os.path.exists(self.image_path)

        self.rand_rot_scene = parser.get('rand_rot_scene', False)
        self.max_train_agent = parser.get('max_train_agent', 100)
        self.traj_scale = parser.traj_scale
        self.occlusion_process = parser.get('occlusion_process', 'fully_observed')
        self.map_side = parser.get('scene_side_length', 80.0)  # [m]
        self.distance_threshold_occluded_target = self.map_side / 4  # [m]
        self.map_resolution = parser.get('global_map_resolution', 800)  # [px]
        self.map_crop_coords = torch.Tensor(
            [[-self.map_side, -self.map_side],
             [self.map_side, self.map_side]]
        ) * self.traj_scale / 2
        self.map_homography = torch.Tensor(
            [[self.map_resolution / self.map_side, 0., self.map_resolution / 2],
             [0., self.map_resolution / self.map_side, self.map_resolution / 2],
             [0., 0., 1.]]
        )

        self.to_torch_image = transforms.ToTensor()

        if self.occlusion_process == 'fully_observed':
            self.trajectory_processing_strategy = self.process_fully_observed_cases
        elif self.occlusion_process == 'occlusion_simulation':
            self.trajectory_processing_strategy = self.process_cases_with_simulated_occlusions
        else:
            raise NotImplementedError

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
        self.lookup_time_window = np.arange(0, self.T_total) * int(dataset.orig_fps // dataset.fps)

        prnt_str = f'total num samples: {len(dataset)}'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)
        prnt_str = "------------------------------ done --------------------------------\n"
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

    def make_padded_scene_images(self):
        os.makedirs(self.padded_images_path, exist_ok=True)
        orig_sdd_dataset_path = os.path.join(self.sdd_config['dataset']['path'], 'annotations')
        for scene in os.scandir(orig_sdd_dataset_path):
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

    @staticmethod
    def last_observed_indices(obs_mask: Tensor) -> Tensor:
        # obs_mask [N, T]
        return obs_mask.shape[1] - torch.argmax(torch.flip(obs_mask, dims=[1]), dim=1) - 1  # [N]

    def last_observed_timesteps(self, last_obs_indices: Tensor) -> Tensor:
        # last_obs_indices [N]
        return self.timesteps[last_obs_indices]  # [N]

    @staticmethod
    def last_observed_positions(trajs: Tensor, last_obs_indices: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # last_obs_indices [N]
        return trajs[torch.arange(trajs.shape[0]), last_obs_indices, :]  # [N, 2]

    def predict_mask(self, last_obs_indices: Tensor) -> Tensor:
        # last_obs_indices [N]
        predict_mask = torch.full([last_obs_indices.shape[0], self.T_total], False)  # [N, T]
        pred_indices = last_obs_indices + 1  # [N]
        for i, pred_idx in enumerate(pred_indices):
            predict_mask[i, pred_idx:] = True
        return predict_mask

    def agent_grid(self, ids: Tensor) -> Tensor:
        # ids [N]
        return torch.hstack([ids.unsqueeze(1)] * self.T_total)  # [N, T]

    def timestep_grid(self, ids: Tensor) -> Tensor:
        return torch.vstack([self.timesteps] * ids.shape[0])  # [N, T]

    @staticmethod
    def true_velocity(trajs: Tensor) -> Tensor:
        # trajs [N, T, 2]
        vel = torch.zeros_like(trajs)
        vel[:, 1:, :] = trajs[:, 1:, :] - trajs[:, :-1, :]
        return vel

    @staticmethod
    def observed_velocity(trajs: Tensor, obs_mask: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        vel = torch.zeros_like(trajs)  # [N, T, 2]
        for traj, mask, v in zip(trajs, obs_mask, vel):
            obs_indices = torch.nonzero(mask)  # [Z, 1]
            motion_diff = traj[obs_indices[1:, 0], :] - traj[obs_indices[:-1, 0], :]  # [Z - 1, 2]
            v[obs_indices[1:].squeeze(), :] = motion_diff / (obs_indices[1:, :] - obs_indices[:-1, :])  # [Z - 1, 2]
        return vel  # [N, T, 2]

    @staticmethod
    def cv_extrapolate(trajs: Tensor, obs_vel: Tensor, last_obs_indices: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # obs_vel [N, T, 2]
        # last_obs_indices [N]
        xtrpl_trajs = trajs.detach().clone()
        for traj, vel, obs_idx in zip(xtrpl_trajs, obs_vel, last_obs_indices):
            last_pos = traj[obs_idx]
            last_vel = vel[obs_idx]
            extra_seq = last_pos + torch.arange(traj.shape[0] - obs_idx).unsqueeze(1) * last_vel
            traj[obs_idx:] = extra_seq
        return xtrpl_trajs

    def impute_and_cv_predict(self, trajs: Tensor, obs_mask: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        imputed_trajs = torch.zeros_like(trajs)
        for i, (traj, mask) in enumerate(zip(trajs, obs_mask)):
            # print(f"{self.timesteps[mask]=}")
            # print(f"{traj[mask]=}")
            f = interp1d(self.timesteps[mask], traj[mask], axis=0, fill_value='extrapolate')
            interptraj = f(self.timesteps)
            # print(f"{interptraj=}")
            imputed_trajs[i, ...] = torch.from_numpy(interptraj)
        return imputed_trajs

    @staticmethod
    def random_index(bool_mask: Tensor) -> Tensor:
        # bool_mask [N]
        # we can only select one index from bool_mask which points to a True value
        candidates = torch.nonzero(bool_mask)
        candidate_idx = torch.randint(0, candidates.shape[0], (1,))
        return candidates[candidate_idx]

    @staticmethod
    def points_within_distance(target_point: Tensor, points: Tensor, distance: Tensor) -> Tensor:
        # target_point [2]
        # agent_points [N, 2]
        # distance [1]
        distances = torch.linalg.norm(points - target_point, dim=1)
        return distances <= distance

    def __len__(self) -> int:
        return len(self.occlusion_table)

    def crop_scene_map(self, scene_map: TorchGeometricMap):
        scene_map.crop(crop_coords=scene_map.to_map_points(self.map_crop_coords), resolution=self.map_resolution)
        scene_map.set_homography(matrix=self.map_homography)

    def random_agent_removal(self, keep_mask: Tensor, tgt_idx: int, center_agent_idx: int):
        keep_indices = torch.Tensor([tgt_idx, center_agent_idx]).to(torch.int64).unique()
        candidate_indices = torch.nonzero(keep_mask).squeeze()
        candidate_indices = candidate_indices[(candidate_indices[:, None] != keep_indices).all(dim=1)]

        kps = torch.randperm(candidate_indices.shape[0])[:self.max_train_agent - keep_indices.shape[0]]

        keep_indices = torch.cat((keep_indices, candidate_indices[kps]))
        keep_mask[:] = False
        keep_mask[keep_indices] = True
        return keep_mask

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

    def trajectory_processing_without_occlusion(self, trajs: Tensor, scene_map: TorchGeometricMap, m_by_px: float):
        obs_mask = torch.ones(trajs.shape[:-1])
        obs_mask[..., self.T_obs:] = False
        scene_map.set_homography(torch.eye(3))

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
        scene_map.homography_translation(center_point)
        scene_map.homography_scaling(1 / scaling)

        # cropping the scene map
        self.crop_scene_map(scene_map=scene_map)

        return trajs, obs_mask, keep_agent_mask, scene_map.image

    def wrapped_trajectory_processing_without_occlusion(
            self, trajs: Tensor,
            scene_map: TorchGeometricMap, m_by_px: float
    ):
        trajs, obs_mask, keep_mask, scene_map_image = self.trajectory_processing_without_occlusion(
            trajs=trajs, scene_map=scene_map, m_by_px=m_by_px
        )

        last_obs_indices = torch.full([trajs.shape[0]], self.T_obs - 1)

        occlusion_map = torch.full([self.map_resolution, self.map_resolution], True)
        dist_transformed_occlusion_map = torch.zeros([self.map_resolution, self.map_resolution])
        probability_map = torch.zeros([self.map_resolution, self.map_resolution])
        nlog_probability_map = torch.zeros([self.map_resolution, self.map_resolution])

        return trajs, obs_mask, last_obs_indices, keep_mask, \
               occlusion_map, dist_transformed_occlusion_map, probability_map, nlog_probability_map, scene_map.image

    def trajectory_processing_with_occlusion(
            self, trajs: Tensor,
            ego: Tensor, occluder: Tensor, tgt_idx: Tensor,
            scene_map: TorchGeometricMap, px_by_m: float, m_by_px: float
    ):
        # mapping trajectories to the scene map coordinate system
        ego = scene_map.to_map_points(ego)
        occluder = scene_map.to_map_points(occluder)
        scene_map.set_homography(torch.eye(3))

        # computing the ego visibility polygon
        scene_boundary = poly_gen.default_rectangle(corner_coords=scene_map.get_map_dimensions())
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

        # checking for all agents that they have at least 2 observations available for the model to process
        sufficiently_observed_mask = (torch.sum(obs_mask, dim=1) >= 2)

        # computing agents' last observed positions
        last_obs_indices = self.last_observed_indices(obs_mask=obs_mask)
        last_obs_positions = self.last_observed_positions(trajs=trajs, last_obs_indices=last_obs_indices)  # [N, 2]

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

        keep_agent_mask = torch.logical_and(sufficiently_observed_mask, close_keep_mask)

        center_point = torch.mean(last_obs_positions[keep_agent_mask], dim=0)

        # shifting and scaling (to metric) of the trajectories, visibility polygon and scene map coordinates
        scaling = self.traj_scale * m_by_px

        trajs = (trajs - center_point) * scaling
        ego_visipoly = sg.Polygon((torch.from_numpy(ego_visipoly.coords) - center_point) * scaling)
        scene_map.homography_translation(center_point)
        scene_map.homography_scaling(1 / scaling)

        # cropping the scene map
        self.crop_scene_map(scene_map=scene_map)

        # computing the occlusion map and distance transformed occlusion map
        map_dims = scene_map.get_map_dimensions()
        occ_y = torch.arange(map_dims[0])
        occ_x = torch.arange(map_dims[1])
        xy = torch.dstack((torch.meshgrid(occ_x, occ_y))).reshape((-1, 2))
        mpath = Path(scene_map.to_map_points(torch.from_numpy(ego_visipoly.coords).to(torch.float32)))
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

        return trajs, obs_mask, last_obs_indices, keep_agent_mask, \
               occlusion_map, dist_transformed_occlusion_map, probability_map, nlog_probability_map, scene_map.image

    def process_cases_with_simulated_occlusions(
            self, occlusion_case: Dict, trajs: Tensor, scene_map: TorchGeometricMap, px_by_m: float, m_by_px: float
    ):
        if np.isnan(occlusion_case['ego_point']).any():
            return self.wrapped_trajectory_processing_without_occlusion(
                trajs=trajs, scene_map=scene_map, m_by_px=m_by_px
            )
        else:
            tgt_idx = torch.from_numpy(occlusion_case['target_agent_indices']).squeeze()
            return self.trajectory_processing_with_occlusion(
                trajs=trajs,
                ego=torch.from_numpy(occlusion_case['ego_point']).to(torch.float32).unsqueeze(0),
                occluder=torch.from_numpy(np.vstack(occlusion_case['occluders'][0])).to(torch.float32),
                tgt_idx=tgt_idx,
                scene_map=scene_map,
                px_by_m=px_by_m, m_by_px=m_by_px
            )

    def process_fully_observed_cases(
            self, occlusion_case: Dict, trajs: Tensor, scene_map: TorchGeometricMap, px_by_m: float, m_by_px: float
    ):
        return self.wrapped_trajectory_processing_without_occlusion(
            trajs=trajs, scene_map=scene_map, m_by_px=m_by_px
        )

    def __getitem__(self, idx: int) -> Dict:
        # lookup the row in the occlusion_table
        occlusion_case = self.occlusion_table.iloc[idx]
        sim_id, scene, video, timestep, trial = occlusion_case.name

        # find the corresponding index in self.lookuptable
        lookup_idx = occlusion_case['lookup_idx']
        lookup_row = self.lookuptable.iloc[lookup_idx]

        # extract the reference image
        image_path = os.path.join(self.padded_images_path, f'{scene}_{video}_padded_img.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_torch_image(image)

        scene_map = TorchGeometricMap(
            map_image=image, homography=torch.eye(3)
        )
        scene_map.homography_translation(Tensor([self.padding_px, self.padding_px]))

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
        theta_rot = np.random.rand() * 360
        scene_map.rotate_around_center(theta=theta_rot)

        # mapping the trajectories to scene map coordinate system
        trajs = scene_map.to_map_points(trajs)

        trajs, obs_mask, last_obs_indices, keep_mask, \
        occlusion_map, dist_transformed_occlusion_map, \
        probability_map, nlog_probability_map, scene_map_image = self.trajectory_processing_strategy(
            occlusion_case=occlusion_case, trajs=trajs, scene_map=scene_map, px_by_m=px_by_m, m_by_px=m_by_px
        )

        # identifying agents who are outside the global scene map
        # inside_map_mask = ~torch.any(torch.any(torch.abs(trajs) >= 0.95 * 0.5 * self.map_side, dim=-1), dim=-1)
        # print(f"{inside_map_mask=}")

        # removing agent surplus
        ids = ids[keep_mask]
        trajs = trajs[keep_mask]
        obs_mask = obs_mask[keep_mask]
        last_obs_indices = last_obs_indices[keep_mask]

        obs_mask = obs_mask.to(torch.bool)
        pred_mask = self.predict_mask(last_obs_indices=last_obs_indices)
        agent_grid = self.agent_grid(ids=ids)
        timestep_grid = self.timestep_grid(ids=ids)

        obs_trajs = trajs[obs_mask, ...]
        obs_vel = self.observed_velocity(trajs=trajs, obs_mask=obs_mask)
        obs_vel_seq = obs_vel[obs_mask, ...]
        obs_ids = agent_grid[obs_mask, ...]
        obs_timesteps = timestep_grid[obs_mask, ...]
        last_obs_positions = self.last_observed_positions(trajs=trajs, last_obs_indices=last_obs_indices)
        last_obs_timesteps = self.last_observed_timesteps(last_obs_indices=last_obs_indices)

        pred_trajs = trajs.transpose(0, 1)[pred_mask.T, ...]
        pred_vel = self.true_velocity(trajs=trajs)
        pred_vel_seq = pred_vel.transpose(0, 1)[pred_mask.T, ...]
        pred_ids = agent_grid.T[pred_mask.T, ...]
        pred_timesteps = timestep_grid.T[pred_mask.T, ...]

        data_dict = {
            'trajectories': trajs,
            # 'obs_velocities': obs_vel,
            'observation_mask': obs_mask,
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

            'occlusion_map': occlusion_map,
            'dist_transformed_occlusion_map': dist_transformed_occlusion_map,
            'probability_occlusion_map': probability_map,
            'nlog_probability_occlusion_map': nlog_probability_map,
            'scene_map': scene_map.image,
            'map_homography': self.map_homography,

            'seq': f'{scene}_{video}',
            'frame': timestep
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

    @staticmethod
    def visualize_sequences(data_dict: Dict, draw_ax: matplotlib.axes.Axes):
        ids = data_dict['identities']

        obs_ids = data_dict['obs_identity_sequence']
        obs_trajs = data_dict['obs_position_sequence']
        obs_vel = data_dict['obs_velocity_sequence']
        obs_timesteps = data_dict['obs_timestep_sequence']
        last_obs_pos = data_dict['last_obs_positions']

        pred_ids = data_dict['pred_identity_sequence']
        pred_trajs = data_dict['pred_position_sequence']
        pred_vel = data_dict['pred_velocity_sequence']
        pred_timesteps = data_dict['pred_timestep_sequence']

        homography = data_dict['map_homography']

        vel_homography = homography.clone()
        vel_homography[:2, 2] = 0.

        color = plt.cm.rainbow(np.linspace(0, 1, ids.shape[0]))

        homogeneous_last_obs_pos = torch.cat((last_obs_pos, torch.ones([*last_obs_pos.shape[:-1], 1])),
                                             dim=-1).transpose(-1, -2)
        plot_last_obs_pos = (homography @ homogeneous_last_obs_pos).transpose(-1, -2)[..., :-1]

        for i, (ag_id, last_pos) in enumerate(zip(ids, plot_last_obs_pos)):
            c = color[i].reshape(1, -1)
            draw_ax.scatter(last_pos[0], last_pos[1], s=70, facecolors='none', edgecolors=c, alpha=0.3)

        homogeneous_obs_trajs = torch.cat((obs_trajs, torch.ones([*obs_trajs.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        plot_obs_trajs = (homography @ homogeneous_obs_trajs).transpose(-1, -2)[..., :-1]

        homogeneous_obs_vel = torch.cat((obs_vel, torch.ones([*obs_vel.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        plot_obs_vel = (vel_homography @ homogeneous_obs_vel).transpose(-1, -2)[..., :-1]

        for i, (ag_id, pos, vel, timestep) in enumerate(zip(obs_ids, plot_obs_trajs, plot_obs_vel, obs_timesteps)):
            c = color[np.nonzero(ids == ag_id).flatten()].reshape(1, -1)
            s = 5 * (timestep + 8)
            draw_ax.scatter(pos[0], pos[1], marker='x', s=5 + s, color=c)
            old_pos = pos - vel
            draw_ax.plot([old_pos[0], pos[0]], [old_pos[1], pos[1]], color=c, linestyle='--', alpha=0.8)

        homogeneous_pred_trajs = torch.cat((pred_trajs, torch.ones([*pred_trajs.shape[:-1], 1])), dim=-1).transpose(-1,
                                                                                                                    -2)
        plot_pred_trajs = (homography @ homogeneous_pred_trajs).transpose(-1, -2)[..., :-1]

        homogeneous_pred_vel = torch.cat((pred_vel, torch.ones([*pred_vel.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        plot_pred_vel = (vel_homography @ homogeneous_pred_vel).transpose(-1, -2)[..., :-1]

        for i, (ag_id, pos, vel, timestep) in enumerate(zip(pred_ids, plot_pred_trajs, plot_pred_vel, pred_timesteps)):
            c = color[np.nonzero(ids == ag_id).flatten()].reshape(1, -1)
            s = 5 * (timestep)
            draw_ax.scatter(pos[0], pos[1], marker='*', s=5 + s, color=c)
            old_pos = pos - vel
            draw_ax.plot([old_pos[0], pos[0]], [old_pos[1], pos[1]], color=c, linestyle=':', alpha=0.8)

    @staticmethod
    def visualize_trajectories(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
        ids = data_dict['identities']
        trajs = data_dict['trajectories']
        obs_mask = data_dict['observation_mask']
        homography = data_dict['map_homography']

        homogeneous_trajs = torch.cat((trajs, torch.ones([*trajs.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        plot_trajs = (homography @ homogeneous_trajs).transpose(-1, -2)[..., :-1]

        color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, ids.shape[0])))
        for traj, mask in zip(plot_trajs, obs_mask):
            c = next(color_iter).reshape(1, -1)
            draw_ax.scatter(traj[:, 0][mask], traj[:, 1][mask], marker='x', s=20, color=c)
            draw_ax.scatter(traj[:, 0][~mask], traj[:, 1][~mask], marker='*', s=20, color=c)

    @staticmethod
    def visualize_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
        scene_map_img = data_dict['scene_map']  # [C, H, W]
        occlusion_map_img = data_dict['occlusion_map']  # [H, W]

        draw_ax.set_xlim(0., scene_map_img.shape[2])
        draw_ax.set_ylim(scene_map_img.shape[1], 0.)
        draw_ax.imshow(scene_map_img.permute(1, 2, 0))

        occlusion_map_render = np.full(
            (*occlusion_map_img.shape, 4), (1., 0, 0, 0.3)
        ) * (~occlusion_map_img)[..., None].numpy()
        draw_ax.imshow(occlusion_map_render)

    @staticmethod
    def visualize_dist_transformed_occlusion_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
        dt_occlusion_map = data_dict['dist_transformed_occlusion_map']

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax)
        colors_visible = plt.cm.Purples(np.linspace(0.5, 1, 256))
        colors_occluded = plt.cm.Reds(np.linspace(1, 0.5, 256))
        all_colors = np.vstack((colors_occluded, colors_visible))
        color_map = colors.LinearSegmentedColormap.from_list('color_map', all_colors)
        divnorm = colors.TwoSlopeNorm(
            vmin=np.min([-1, torch.min(dt_occlusion_map)]),
            vcenter=0.0,
            vmax=np.max([1, torch.max(dt_occlusion_map)])
        )
        cax = divider.append_axes('right', size='5%', pad=0.05)
        img = draw_ax.imshow(dt_occlusion_map, norm=divnorm, cmap=color_map)
        draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')

    @staticmethod
    def visualize_probability_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
        probability_map = data_dict['probability_occlusion_map']

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        img = draw_ax.imshow(probability_map, cmap='Greys')
        draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')

    @staticmethod
    def visualize_nlog_probability_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
        nlog_probability_map = data_dict['nlog_probability_occlusion_map']

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        img = draw_ax.imshow(nlog_probability_map, cmap='Greys')
        draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')

    def visualize(
            self,
            data_dict: Dict,
            draw_ax: matplotlib.axes.Axes,
            draw_ax_sequences: Optional[matplotlib.axes.Axes] = None,
            draw_ax_dist_transformed_map: Optional[matplotlib.axes.Axes] = None,
            draw_ax_probability_map: Optional[matplotlib.axes.Axes] = None,
            draw_ax_nlog_probability_map: Optional[matplotlib.axes.Axes] = None
    ) -> None:
        self.visualize_map(data_dict=data_dict, draw_ax=draw_ax)
        self.visualize_trajectories(data_dict=data_dict, draw_ax=draw_ax)
        if draw_ax_sequences is not None:
            self.visualize_map(data_dict=data_dict, draw_ax=draw_ax_sequences)
            self.visualize_sequences(data_dict=data_dict, draw_ax=draw_ax_sequences)

        if draw_ax_dist_transformed_map is not None:
            self.visualize_dist_transformed_occlusion_map(data_dict=data_dict, draw_ax=draw_ax_dist_transformed_map)

        if draw_ax_probability_map is not None:
            self.visualize_probability_map(data_dict=data_dict, draw_ax=draw_ax_probability_map)

        if draw_ax_nlog_probability_map is not None:
            self.visualize_nlog_probability_map(data_dict=data_dict, draw_ax=draw_ax_nlog_probability_map)


def show_example_instances_dataloader():
    from utils.utils import prepare_seed, memory_report
    from tqdm import tqdm

    print(sdd_conf.REPO_ROOT)

    n_calls = 10000
    # config_str = 'sdd_agentformer_pre'
    # config_str = 'sdd_occlusion_agentformer_pre'
    config_str = 'sdd_baseline_occlusionformer_pre'

    config = Config(config_str)
    prepare_seed(config.seed)

    split = 'train'

    generator = TorchDataGeneratorSDD(parser=config, log=None, split=split)

    sdd_config = sdd_conf.get_config(config.sdd_config_file_name)
    compare_generator = StanfordDroneDatasetWithOcclusionSim(sdd_config, split=split)

    for idx in range(len(generator)):

        # memory_report('BEFORE')

        # 225, 615, 735
        # (run this on the v3 simulators, with max_train_agent 16 to
        # observe preprocessing of cases with too many agents)
        # if idx < 735:
        #     continue

        data_dict = generator.__getitem__(idx)
        print(f"{idx, len(data_dict['identities'])=}")

        if len(data_dict['identities']) == generator.max_train_agent:

            fig, ax = plt.subplots(1, 6)
            visualize.visualize_training_instance(
                draw_ax=ax[0], instance_dict=compare_generator.__getitem__(idx)
            )
            generator.visualize(
                data_dict=data_dict,
                draw_ax=ax[1],
                draw_ax_sequences=ax[2],
                draw_ax_dist_transformed_map=ax[3],
                draw_ax_probability_map=ax[4],
                draw_ax_nlog_probability_map=ax[5]
            )
            plt.show()

        # memory_report('AFTER')


if __name__ == '__main__':
    show_example_instances_dataloader()
    # save_preprocessed_dataset()
