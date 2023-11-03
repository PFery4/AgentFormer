import os.path
import random
from io import TextIOWrapper

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.colors as colors
import numpy
from matplotlib.path import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys
import numpy as np
import skgeom as sg
from scipy.ndimage import distance_transform_edt
from scipy.special import log_softmax, softmax

from data.map import GeometricMap, TorchGeometricMap
from utils.config import Config, REPO_ROOT
from utils.utils import print_log, get_timestring

from typing import Dict, Optional
from numpy.typing import NDArray
Tensor = torch.Tensor

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
import src.visualization.sdd_visualize as visualize
import src.occlusion_simulation.visibility as visibility
import src.occlusion_simulation.polygon_generation as poly_gen
import src.data.config as sdd_conf


class AgentFormerDataGeneratorForSDD:
    """
    This class wraps the dataset classes implemented in the occlusion-prediction repo in such a way that
    they are directly usable as 'generator' objects in the source code of AgentFormer.
    """

    def __init__(self, parser: Config, log: TextIOWrapper, split: str = 'train', phase: str = 'training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.rand_rot_scene = parser.get('rand_rot_scene', False)
        self.max_train_agent = parser.get('max_train_agent', 100)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        self.sdd_config = sdd_conf.get_config(parser.sdd_config_file_name)
        if not parser.get('sdd_occlusion_data', False):
            self.dataset = StanfordDroneDataset(self.sdd_config, split=self.split)
            self.traj_processing = self.traj_processing_without_occlusion
        else:
            self.dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config, split=self.split)
            self.traj_processing = self.traj_processing_with_occlusion
        print(f"instantiating dataloader from {self.dataset.__class__} class")

        self.num_total_samples = len(self.dataset)

        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

        self.sample_list = list(range(self.num_total_samples))
        self.index = 0

        self.traj_scale = parser.traj_scale
        self.map_side = parser.get('scene_side_length', 80.0)           # [m]
        self.dist_threshold_tgt_agent = self.map_side / 4               # [m]
        self.map_res = parser['global_map_encoder'].get('map_resolution', 800)      # [px]
        self.map_crop_coords = np.array(
            [[-self.map_side, -self.map_side], [self.map_side, self.map_side]]
        ) * self.traj_scale / 2

        # self.compute_center_point = self.mean_last_observations
        self.compute_center_point = self.random_agent_last_obs

        self.timesteps = np.arange(-self.dataset.T_obs, self.dataset.T_pred) + 1

    def shuffle(self) -> None:
        random.shuffle(self.sample_list)

    def is_epoch_end(self) -> bool:
        if self.index >= self.num_total_samples:
            self.index = 0
            return True
        else:
            return False

    def cropped_scene_map(self, scene_map: GeometricMap):
        # cropping the scene_map
        crop_coords = scene_map.to_map_points(self.map_crop_coords)
        scene_map.square_crop(
            crop_coords=crop_coords,
            side_length=self.map_side,
            resolution=self.map_res
        )
        return scene_map

    @staticmethod
    def last_observed_indices(obs_mask: NDArray) -> NDArray:
        # obs_mask [N, T]
        return obs_mask.shape[1] - np.argmax(obs_mask[:, ::-1], axis=1) - 1         # [N]

    def last_observed_timesteps(self, obs_mask: NDArray) -> NDArray:
        # obs_mask [N, T]
        return self.timesteps[self.last_observed_indices(obs_mask=obs_mask)]        # [N]

    def last_observed_positions(self, trajs: NDArray, obs_mask: NDArray) -> NDArray:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        last_obs_indices = self.last_observed_indices(obs_mask=obs_mask)            # [N]
        return trajs[np.arange(trajs.shape[0]), last_obs_indices, :]                # [N, 2]

    def predict_mask(self, obs_mask: NDArray) -> NDArray:
        # obs_mask [N, T]
        predict_mask = np.full_like(obs_mask, False)                                # [N, T]
        pred_indices = self.last_observed_indices(obs_mask=obs_mask) + 1            # [N]
        for i, pred_idx in enumerate(pred_indices):
            predict_mask[i, pred_idx:] = True
        return predict_mask                                                         # [N, T]

    def agent_mask(self, ids: NDArray) -> NDArray:
        # ids [N]
        return np.hstack([ids[:, np.newaxis]] * self.timesteps.shape[0])        # [N, T]

    def timestep_mask(self, ids: NDArray) -> NDArray:
        # ids [N]
        return np.vstack([self.timesteps] * ids.shape[0])          # [N, T]

    @staticmethod
    def true_velocity(trajs: NDArray) -> NDArray:
        # trajs [N, T, 2]
        vel = np.zeros_like(trajs)
        vel[:, 1:, :] = trajs[:, 1:, :] - trajs[:, :-1, :]
        return vel                  # [N, T, 2]

    @staticmethod
    def observed_velocity(trajs: NDArray, obs_mask: NDArray) -> NDArray:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        vel = np.zeros_like(trajs)
        for traj, mask, v in zip(trajs, obs_mask, vel):
            obs_indices = np.flatnonzero(mask)
            motion_diff = traj[obs_indices[1:], :] - traj[obs_indices[:-1], :]
            v[obs_indices[1:], :] = motion_diff / (obs_indices[1:] - obs_indices[:-1])[:, np.newaxis]
        return vel              # [N, T, 2]

    @staticmethod
    def cv_extrapolate(trajs: NDArray, obs_vel: NDArray, last_obs_indices: NDArray) -> NDArray:
        # trajs [N, T, 2]
        # obs_vel [N, T, 2]
        # last_obs_indices [N]
        xtrpl_trajs = trajs.copy()
        for traj, vel, obs_idx in zip(xtrpl_trajs, obs_vel, last_obs_indices):
            last_pos = traj[obs_idx]
            last_vel = vel[obs_idx]
            extra_seq = last_pos + np.arange(traj.shape[0] - obs_idx)[:, np.newaxis] * last_vel
            traj[obs_idx:] = extra_seq
        return xtrpl_trajs

    @staticmethod
    def impute_interpolate(trajs: NDArray, obs_mask: NDArray) -> NDArray:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        imputed_trajs = np.zeros_like(trajs)
        for imputed_traj, traj, mask in zip(imputed_trajs, trajs, obs_mask):
            obs_indices = np.flatnonzero(mask)
            trajpoints = traj[mask]
            imputed_traj[:, 0] = np.interp(np.arange(traj.shape[0]), obs_indices, trajpoints[:, 0])
            imputed_traj[:, 1] = np.interp(np.arange(traj.shape[0]), obs_indices, trajpoints[:, 1])
        return imputed_trajs

    def mean_position_last_observations(self, trajs: NDArray, obs_mask: NDArray) -> NDArray:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        return np.mean(self.last_observed_positions(trajs=trajs, obs_mask=obs_mask), axis=0).copy()         # [2]

    @staticmethod
    def random_agent_last_obs(trajs: NDArray, obs_mask: NDArray) -> NDArray:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        rnd_agent_idx = np.random.randint(trajs.shape[0])
        last_obs_index = trajs.shape[1] - np.argmax(obs_mask[rnd_agent_idx, ::-1], axis=0) - 1
        return trajs[rnd_agent_idx, last_obs_index, :].copy()          # [2]

    @staticmethod
    def agents_within_distance(target_point: NDArray, trajs: NDArray, obs_mask: NDArray, distance: float) -> NDArray:
        # target_point [2]
        # trajs [N, T, 2]
        # obs_mask [N, T]
        last_obs_indices = trajs.shape[1] - np.argmax(obs_mask[:, ::-1], axis=1) - 1
        last_obs_points = trajs[np.arange(trajs.shape[0]), last_obs_indices, :]     # [N, 2]
        agent_distances = np.linalg.norm(last_obs_points - target_point, axis=-1)
        return agent_distances <= distance      # [N]

    def traj_processing_without_occlusion(
            self, trajs: NDArray, **kwargs
    ):
        # trajs.shape [N, T, 2]
        scene_map = kwargs['scene_map']             # GeometricMap
        past_window = kwargs['past_window']         # NDArray   [T_obs]
        ids = kwargs['ids']                         # NDArray   [N]
        m_per_px = kwargs['m_per_px']               # float

        trajs = scene_map.to_map_points(trajs)
        scene_map.set_homography(np.eye(3))

        obs_mask = np.full(trajs.shape[:2], True)
        obs_mask[..., len(past_window):] = False

        # calculating the mean of all points at t_0
        center_point = self.compute_center_point(trajs=trajs, obs_mask=obs_mask)

        # normalize
        trajs -= center_point
        scene_map.translation(center_point)

        scaling = self.traj_scale * m_per_px
        trajs *= scaling
        scene_map.scale(scaling=1/scaling)
        scene_map = self.cropped_scene_map(scene_map)

        # removing agents who are lying outside the scene
        outside = np.any(np.abs(trajs) >= 0.9 * self.map_side, axis=(1, 2))
        trajs, ids, obs_mask = trajs[~outside], ids[~outside], obs_mask[~outside]

        # randomly removing additional agents if we have too many
        if ids.shape[0] > self.max_train_agent:
            keep_indices = np.sort(np.random.choice(ids.shape[0], self.max_train_agent, replace=False))
            ids, trajs, obs_mask = ids[keep_indices], trajs[keep_indices], obs_mask[keep_indices]

        out_dict = {
            "ids": ids,
            "trajs": trajs,
            "obs_mask": obs_mask,
            "scene_map": scene_map,
            "occlusion_map": np.ones(scene_map.get_map_dimensions()),
            "dt_occlusion_map": np.zeros(scene_map.get_map_dimensions()),
            "p_occl_map": np.zeros(scene_map.get_map_dimensions()),
            "min_log_p_occl_map": np.zeros(scene_map.get_map_dimensions()),
            "ego": None,
            "ego_visipoly": None,
            "occluders": None,
            "occlusion_case": False
        }
        return out_dict

    def traj_processing_with_occlusion(
            self, trajs: NDArray, **kwargs
    ):
        # trajs.shape [N, T, 2]
        if np.any(np.isnan(kwargs['ego'])):
            return self.traj_processing_without_occlusion(trajs=trajs, **kwargs)

        scene_map = kwargs['scene_map']             # GeometricMap
        orig_ego = kwargs['ego']                    # NDArray   [2]
        orig_occluders = kwargs['occluders']        # List[List[NDArray]]
        past_window = kwargs['past_window']         # NDArray   [T_obs]
        ids = kwargs['ids']                         # NDArray   [N]
        tgt_idx = kwargs['tgt_idx']                 # NDArray   [1]
        m_per_px = kwargs['m_per_px']               # float

        # compute trajs, ego and occluder positions (transforming to map coords)
        trajs = scene_map.to_map_points(trajs)
        ego = scene_map.to_map_points(scene_pts=orig_ego)
        occluders = []
        for occluder in orig_occluders:
            p1 = scene_map.to_map_points(scene_pts=occluder[0])
            p2 = scene_map.to_map_points(scene_pts=occluder[1])
            occluders.append((p1, p2))
        scene_map.set_homography(np.eye(3))

        # compute visibility polygon
        scene_boundary = poly_gen.default_rectangle(corner_coords=reversed(scene_map.get_map_dimensions()))
        ego_visipoly = visibility.compute_visibility_polygon(
            ego_point=ego,
            occluders=occluders,
            boundary=scene_boundary
        )

        # obtain observation_mask
        obs_mask = visibility.occlusion_mask(
            points=trajs.reshape(-1, trajs.shape[-1]),
            ego_visipoly=ego_visipoly
        ).reshape(trajs.shape[:-1])  # [N, T]
        obs_mask[..., len(past_window):] = False

        # check for all agents that they have at least 2 observations available for the model to process
        # other agents (those who are insufficiently observed) will be discarded
        sufficiently_observed = (np.sum(obs_mask, axis=1) >= 2)  # [N]

        # computing the list of potential agents to use as the center point of the instance. We want to ensure that
        # this agent is within some distance threshold of the agent we simulated an occlusion for, in order to ensure
        # the occlusion event is visible from the model.
        target_last_obs = trajs.shape[1] - np.argmax(obs_mask[tgt_idx[0], ::-1], axis=0) - 1
        dist_threshold_mask = self.agents_within_distance(
            target_point=trajs[tgt_idx[0], target_last_obs, ...],
            trajs=trajs, obs_mask=obs_mask, distance=self.dist_threshold_tgt_agent/m_per_px
        )

        # performing all shifting / normalization only based on points we have observed
        # (using all past points, even unobserved ones, would constitute data leakage)
        center_point = self.compute_center_point(
            trajs=trajs[np.logical_and(sufficiently_observed, dist_threshold_mask)],
            obs_mask=obs_mask[np.logical_and(sufficiently_observed, dist_threshold_mask)]
        )

        # centering the instance
        ego -= center_point
        trajs -= center_point
        ego_visipoly = sg.Polygon(ego_visipoly.coords - center_point)
        scene_map.translation(center_point)

        # converting to metric space coordinate
        scaling = self.traj_scale * m_per_px

        ego *= scaling
        trajs *= scaling
        ego_visipoly = sg.Polygon(ego_visipoly.coords * scaling)
        scene_map.scale(scaling=1/scaling)
        scene_map = self.cropped_scene_map(scene_map)

        # removing the unsufficiently observed agents
        trajs, ids, obs_mask = trajs[sufficiently_observed], ids[sufficiently_observed], obs_mask[sufficiently_observed]

        # removing agents who are lying outside the scene
        # 0.95 is a safety margin
        # 0.5 is half the map side, as the coordinate system is centered at the middle of the scene map
        outside = np.any(np.abs(trajs) >= 0.95 * 0.5 * self.map_side, axis=(1, 2))
        trajs, ids, obs_mask = trajs[~outside], ids[~outside], obs_mask[~outside]

        # randomly removing additional agents if we have too many, while making sure we do not remove the agent who
        # is the target of the occlusion simulation
        if ids.shape[0] > self.max_train_agent:
            # decreasing tgt_idx, as we have filtered out some trajectories previously
            tgt_idx -= (~sufficiently_observed[:tgt_idx[0]]).sum()
            tgt_idx -= (outside[:tgt_idx[0]]).sum()

            selection = np.setdiff1d(np.arange(ids.shape[0]), tgt_idx)
            keep_indices = np.sort(
                np.concatenate(
                    [tgt_idx, np.random.choice(selection, self.max_train_agent, replace=False)]
                )
            )
            ids, trajs, obs_mask = ids[keep_indices], trajs[keep_indices], obs_mask[keep_indices]

        # compute occlusion map and distance transformed occlusion map
        map_dims = scene_map.get_map_dimensions()
        occ_x = np.arange(map_dims[0])
        occ_y = np.arange(map_dims[1])
        xy = np.dstack((np.meshgrid(occ_x, occ_y))).reshape((-1, 2))
        mpath = Path(scene_map.to_map_points(ego_visipoly.coords))
        occlusion_map = mpath.contains_points(xy).reshape(*reversed(map_dims)).astype(np.int32)          # NDArray [map_res, map_res]

        invert_occl_map = 1 - occlusion_map
        dt_occl_map = np.where(
            invert_occl_map,
            -distance_transform_edt(invert_occl_map),
            distance_transform_edt(occlusion_map)
        ) * scaling

        p_occl_map = softmax(-np.clip(dt_occl_map, a_min=0, a_max=None))
        min_log_p_occl_map = -log_softmax(-np.clip(dt_occl_map, a_min=0, a_max=None))

        out_dict = {
            "ids": ids,
            "trajs": trajs,
            "obs_mask": obs_mask,
            "scene_map": scene_map,
            "occlusion_map": occlusion_map,
            "dt_occlusion_map": dt_occl_map,
            "p_occl_map": p_occl_map,
            "min_log_p_occl_map": min_log_p_occl_map,
            "ego": ego,
            "ego_visipoly": ego_visipoly,
            "occluders": occluders,
            "occlusion_case": True
        }

        return out_dict

    def convert_to_preprocessor_data(self, extracted_data: dict) -> dict:

        data = dict()

        heading = None
        # from the nuscenes implementation:
        # pred mask is a numpy array, of shape (n_agents,), with values either 1 or 0
        pred_mask = None

        # perform the loading of the map
        scene_map = GeometricMap(
            data=np.transpose(extracted_data['scene_image'], (2, 1, 0)),
            homography=np.eye(3)
        )
        # expand the map
        scene_map.mirror_expand(2.0)

        # rotate_the_map
        if self.rand_rot_scene:
            data['theta'] = float(np.random.rand() * 2 * np.pi)
        else:
            data['theta'] = 0.0
        scene_map.rotate_around_center(data['theta'])

        # load trajectories and agent identities
        trajs = np.stack(
            [agent.get_traj_section(extracted_data['full_window'])
             for agent in extracted_data['agents']]
        )       # [N, T, 2]
        ids = np.stack([agent.id for agent in extracted_data['agents']])        # [N]

        processed_data = self.traj_processing(
            trajs=trajs,
            scene_map=scene_map,
            ego=extracted_data['ego_point'],
            occluders=extracted_data['occluders'],
            past_window=extracted_data['past_window'],
            ids=ids,
            tgt_idx=extracted_data['target_agent_indices'],
            m_per_px=extracted_data['m/px'],
        )

        # TODO: Cleanup this bit of code
        # TODO: Remove preprocessing from AgentFormer.
        # try_time = np.arange(-7, 13)
        # try_traj = np.hstack([np.sin(try_time)[:, np.newaxis], np.cos(try_time)[:, np.newaxis]])[np.newaxis, :]
        # try_mask = np.full([1, 20], True)
        # try_mask[:, 8:] = False
        # try_mask[:, 2:5] = False
        #
        # print(f"{try_traj, try_traj.shape=}")
        # print(f"{try_mask=}")
        # try_vel = self.observed_velocity(try_traj, try_mask)
        # try_last = self.last_observed_indices(try_mask)
        # print(f"{try_vel=}")
        # print(f"{try_last=}")
        # print(f"{self.cv_extrapolate(try_traj, try_vel, try_last)=}")
        # print(zblu)
        # print(f"{self.cv_extrapolate(trajs=, obs_vel=, last_obs_indices=)=}")
        # print(f"{self.impute_interpolate(trajs=, obs_mask=)=}")

        data['full_motion_3D'] = torch.from_numpy(processed_data['trajs'])
        data['valid_id'] = torch.from_numpy(processed_data['ids'])
        data['obs_mask'] = torch.from_numpy(processed_data['obs_mask'])
        # data['ego'] = torch.from_numpy(processed_data['ego'])
        # data['occluders'] = torch.from_numpy(np.stack(processed_data['occluders']))         # [n_occluders, 2, 2]
        # data['ego_visipoly'] = processed_data['ego_visipoly']                               # sg.Polygon
        data['occlusion_map'] = torch.from_numpy(processed_data['occlusion_map'])               # [H, W]
        data['dt_occlusion_map'] = torch.from_numpy(processed_data['dt_occlusion_map'])         # [H, W]
        data['p_occl_map'] = torch.from_numpy(processed_data['p_occl_map'])                     # [H, W]
        data['min_log_p_occl_map'] = torch.from_numpy(processed_data['min_log_p_occl_map'])     # [H, W]
        data['timesteps'] = torch.from_numpy(self.timesteps)
        data['heading'] = heading
        data['traj_scale'] = self.traj_scale
        data['pred_mask'] = pred_mask
        data['scene_map'] = scene_map
        data['seq'] = extracted_data['scene'] + '_' + extracted_data['video']
        data['frame'] = extracted_data['timestep']
        return data

    def next_sample(self) -> dict:
        sample_index = self.sample_list[self.index]
        self.index += 1

        data = self.dataset.__getitem__(sample_index)
        return self.convert_to_preprocessor_data(data)

    def __call__(self) -> dict:
        return self.next_sample()

    def visualize(
            self,
            draw_ax: matplotlib.axes.Axes,
            data_dict: Dict,
            draw_ax_dt_map: Optional[matplotlib.axes.Axes] = None,
            draw_ax_p_occl_map: Optional[matplotlib.axes.Axes] = None,
            draw_ax_min_log_p_occl_map: Optional[matplotlib.axes.Axes] = None
    ) -> None:
        draw_ax.set_xlim(0., data_dict['scene_map'].get_map_dimensions()[0])
        draw_ax.set_ylim(data_dict['scene_map'].get_map_dimensions()[1], 0.)
        draw_ax.imshow(data_dict['scene_map'].as_image())

        occlusion_map = np.full((*data_dict['occlusion_map'].shape, 4), (255, 0, 0, 0.3)) * (1 - data_dict['occlusion_map'])[..., None].numpy()
        draw_ax.imshow(occlusion_map)

        plot_trajs = data_dict['scene_map'].to_map_points(data_dict['full_motion_3D'])

        color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, data_dict['valid_id'].shape[0])))
        for agent, obs_mask in zip(plot_trajs, data_dict['obs_mask']):
            c = next(color_iter).reshape(1, -1)
            draw_ax.scatter(agent[:, 0][obs_mask], agent[:, 1][obs_mask], marker='x', s=20, color=c)
            draw_ax.scatter(agent[:, 0][~obs_mask], agent[:, 1][~obs_mask], marker='*', s=20, color=c)

        if draw_ax_dt_map is not None:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax_dt_map)
            colors_visible = plt.cm.Purples(np.linspace(0.5, 1, 256))
            colors_occluded = plt.cm.Reds(np.linspace(1, 0.5, 256))
            all_colors = np.vstack((colors_occluded, colors_visible))
            color_map = colors.LinearSegmentedColormap.from_list('color_map', all_colors)
            divnorm = colors.TwoSlopeNorm(
                vmin=np.min([-1, torch.min(data_dict['dt_occlusion_map'])]),
                vcenter=0.0,
                vmax=np.max([1, torch.max(data_dict['dt_occlusion_map'])])
            )
            cax = divider.append_axes('right', size='5%', pad=0.05)
            img = draw_ax_dt_map.imshow(data_dict['dt_occlusion_map'], norm=divnorm, cmap=color_map)
            draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')

        if draw_ax_p_occl_map is not None:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax_p_occl_map)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            img = draw_ax_p_occl_map.imshow(data_dict['p_occl_map'], cmap='Greys')
            draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')

        if draw_ax_min_log_p_occl_map is not None:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax_min_log_p_occl_map)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            img = draw_ax_min_log_p_occl_map.imshow(data_dict['min_log_p_occl_map'], cmap='Greys')
            draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')


class TorchDataGeneratorSDD(Dataset):
    def __init__(self, parser: Config, log: TextIOWrapper, split='train'):
        self.split = split
        assert split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        self.sdd_config = sdd_conf.get_config(parser.sdd_config_file_name)
        dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config, split=self.split)
        print_log(f"instantiating dataloader from {dataset.__class__} class", log)

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
        self.map_side = parser.get('scene_side_length', 80.0)           # [m]
        self.distance_threshold_occluded_target = self.map_side / 4     # [m]
        self.map_resolution = parser.global_map_encoder.get('map_resolution', 800)       # [px]
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

        # TODO: remove once finished implementing this class
        self.temp_generator = dataset

        print_log(f'total num samples: {len(dataset)}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

        # TODO: REMOVE THIS ONCE READY
        torch.manual_seed(2737)

    def make_padded_scene_images(self):
        os.makedirs(self.padded_images_path, exist_ok=True)
        orig_sdd_dataset_path = os.path.join(self.sdd_config['dataset']['path'], 'annotations')
        for scene in os.scandir(orig_sdd_dataset_path):
            for video in os.scandir(scene):
                save_padded_img_path = os.path.join(self.padded_images_path, f"{scene.name}_{video.name}_padded_img.jpg")
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
        return obs_mask.shape[1] - torch.argmax(torch.flip(obs_mask, dims=[1]), dim=1) - 1       # [N]

    def last_observed_timesteps(self, last_obs_indices: Tensor) -> Tensor:
        # last_obs_indices [N]
        return self.timesteps[last_obs_indices]     # [N]

    @staticmethod
    def last_observed_positions(trajs: Tensor, last_obs_indices: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # last_obs_indices [N]
        return trajs[torch.arange(trajs.shape[0]), last_obs_indices, :]     # [N, 2]

    def predict_mask(self, last_obs_indices: Tensor) -> Tensor:
        # last_obs_indices [N]
        predict_mask = torch.full([last_obs_indices.shape[0], self.T_total], False)     # [N, T]
        pred_indices = last_obs_indices + 1                                             # [N]
        for i, pred_idx in enumerate(pred_indices):
            predict_mask[i, pred_idx:] = True
        return predict_mask

    def agent_grid(self, ids: Tensor) -> Tensor:
        # ids [N]
        return torch.hstack([ids.unsqueeze(1)] * self.T_total)      # [N, T]

    def timestep_grid(self, ids: Tensor) -> Tensor:
        return torch.vstack([self.timesteps] * ids.shape[0])        # [N, T]

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
        vel = torch.zeros_like(trajs)       # [N, T, 2]
        for traj, mask, v in zip(trajs, obs_mask, vel):
            obs_indices = torch.nonzero(mask)                                                   # [Z, 1]
            motion_diff = traj[obs_indices[1:, 0], :] - traj[obs_indices[:-1, 0], :]            # [Z - 1, 2]
            v[obs_indices[1:].squeeze(), :] = motion_diff / (obs_indices[1:, :] - obs_indices[:-1, :])    # [Z - 1, 2]
        return vel          # [N, T, 2]

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

    @staticmethod
    def impute_interpolate(trajs: Tensor, obs_mask: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # obs_mask [N, T]
        imputed_trajs = torch.zeros_like(trajs)
        for imputed_traj, traj, mask in zip(imputed_trajs, trajs, obs_mask):
            obs_indices = torch.nonzero(mask)       # [Z, 1]
            traj_points = traj[mask]                # [Z, 2]
            # TODO: figure this out using the interp function
        raise NotImplementedError

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

    def trajectory_processing_without_occlusion(self, trajs: Tensor, scene_map: TorchGeometricMap, m_by_px: float):
        trajs = scene_map.to_map_points(trajs)
        obs_mask = torch.ones(trajs.shape[:-1])
        obs_mask[..., self.T_obs] = False
        scene_map.set_homography(torch.eye(3))

        last_obs_positions = trajs[:, self.T_obs-1, :]
        center_agent_idx = torch.randint(0, trajs.shape[0], (1,))
        center_point = last_obs_positions[center_agent_idx].squeeze()

        scaling = self.traj_scale * m_by_px
        trajs = (trajs - center_point) * scaling
        scene_map.homography_translation(center_point)
        scene_map.homography_scaling(1/scaling)

        # cropping the scene map
        self.crop_scene_map(scene_map=scene_map)

        return trajs, obs_mask, center_agent_idx, scene_map.image

    def trajectory_processing_with_occlusion(
            self, trajs: Tensor,
            ego: Tensor, occluder: Tensor, tgt_idx: Tensor,
            scene_map: TorchGeometricMap, px_by_m: float, m_by_px: float
    ):
        # mapping trajectories to the scene map coordinate system
        trajs = scene_map.to_map_points(trajs)
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
        last_obs_positions = self.last_observed_positions(trajs=trajs, last_obs_indices=last_obs_indices)       # [N, 2]
        # identifying the target agent's last observed position (the agent for whom an occlusion was simulated)
        tgt_last_obs_pos = last_obs_positions[tgt_idx]

        # identifying the candidate agents to use as the center point of the instance.
        # We want to keep the simulated occlusion case in view, so we select agents who aren't too far from the
        # occlusion target.
        dist_threshold_mask = self.points_within_distance(
            target_point=tgt_last_obs_pos.squeeze(), points=last_obs_positions,
            distance=self.distance_threshold_occluded_target * px_by_m
        )

        # sampling an agent to be the center point. we're selecting from agents who are close to the
        # simulated occlusion, and who have enough observations available
        keep_mask = torch.logical_and(sufficiently_observed_mask, dist_threshold_mask)
        center_agent_idx = self.random_index(keep_mask).squeeze()
        center_point = last_obs_positions[center_agent_idx]

        # shifting and scaling (to metric) of the trajectories, visibility polygon and scene map coordinates
        scaling = self.traj_scale * m_by_px

        trajs = (trajs - center_point) * scaling
        ego_visipoly = sg.Polygon((torch.from_numpy(ego_visipoly.coords) - center_point) * scaling)
        scene_map.homography_translation(center_point.squeeze())
        scene_map.homography_scaling(1/scaling)

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

        return trajs, obs_mask, last_obs_indices, keep_mask, center_agent_idx, occlusion_map, dist_transformed_occlusion_map, probability_map, nlog_probability_map, scene_map.image

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
        trajs = torch.empty([len(lookup_row['targets']), self.T_total, 2])      # [N, T, 2]
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

        if np.isnan(occlusion_case['ego_point']).any():
            trajs, obs_mask, center_agent_idx, scene_map_image = self.trajectory_processing_without_occlusion(
                trajs=trajs, scene_map=scene_map, m_by_px=m_by_px
            )
            last_obs_indices = torch.full([ids.shape[0]], self.T_obs - 1)
            keep_mask = torch.full([ids.shape[0]], True)
            tgt_idx = center_agent_idx

            occlusion_map = torch.full([self.map_resolution, self.map_resolution], True)
            dist_transformed_occlusion_map = torch.zeros([self.map_resolution, self.map_resolution])
            probability_map = torch.zeros([self.map_resolution, self.map_resolution])
            nlog_probability_map = torch.zeros([self.map_resolution, self.map_resolution])

        else:
            tgt_idx = torch.from_numpy(occlusion_case['target_agent_indices']).squeeze()
            trajs, obs_mask, last_obs_indices, keep_mask, center_agent_idx,\
                occlusion_map, dist_transformed_occlusion_map,\
                probability_map, nlog_probability_map, scene_map_image = self.trajectory_processing_with_occlusion(
                    trajs=trajs,
                    ego=torch.from_numpy(occlusion_case['ego_point']).to(torch.float32).unsqueeze(0),
                    occluder=torch.from_numpy(np.vstack(occlusion_case['occluders'][0])).to(torch.float32),
                    tgt_idx=tgt_idx,
                    scene_map=scene_map,
                    px_by_m=px_by_m, m_by_px=m_by_px
                )

        # identifying agents who are outside the global scene map
        inside_map_mask = ~torch.any(torch.any(torch.abs(trajs) >= 0.95 * 0.5 * self.map_side, dim=-1), dim=-1)
        keep_mask = torch.logical_and(keep_mask, inside_map_mask)

        # further cutting down of agents if we have too many. We make sure not to remove the center of the instance
        # nor the agent who was the subject of the occlusion
        if torch.sum(keep_mask) > self.max_train_agent:
            keep_indices = torch.Tensor([tgt_idx, center_agent_idx]).to(torch.int64).unique()
            candidate_indices = torch.nonzero(keep_mask).squeeze()
            candidate_indices = candidate_indices[(candidate_indices[:, None] != keep_indices).all(dim=1)]

            kps = torch.randperm(candidate_indices.shape[0])[:self.max_train_agent - keep_indices.shape[0]]

            keep_indices = torch.cat((keep_indices, candidate_indices[kps]))
            keep_mask[:] = False
            keep_mask[keep_indices] = True

        # removing agent surplus
        ids = ids[keep_mask]
        trajs = trajs[keep_mask]
        obs_mask = obs_mask[keep_mask]
        last_obs_indices = last_obs_indices[keep_mask]

        obs_mask = obs_mask.to(torch.bool)

        # TODO: Implement simple random occlusion masks
        # TODO: provide velocity estimations, cv extrapolation, imputations, etc

        agent_grid = self.agent_grid(ids=ids)
        timestep_grid = self.timestep_grid(ids=ids)

        obs_trajs = trajs[obs_mask, ...]
        obs_vel = self.observed_velocity(trajs=trajs, obs_mask=obs_mask)[obs_mask, ...]
        obs_ids = agent_grid[obs_mask, ...]
        obs_timesteps = timestep_grid[obs_mask, ...]
        last_obs_positions = self.last_observed_positions(trajs=trajs, last_obs_indices=last_obs_indices)
        last_obs_timesteps = self.last_observed_timesteps(last_obs_indices=last_obs_indices)

        pred_mask = self.predict_mask(last_obs_indices=last_obs_indices).T
        pred_trajs = trajs.transpose(0, 1)[pred_mask, ...]
        pred_vel = self.true_velocity(trajs=trajs).transpose(0, 1)[pred_mask, ...]
        pred_ids = agent_grid.T[pred_mask, ...]
        pred_timesteps = timestep_grid.T[pred_mask, ...]

        data_dict = {
            'trajectories': trajs,
            'observation_mask': obs_mask,

            'identities': ids,
            'timesteps': self.timesteps,

            'obs_identity_sequence': obs_ids,
            'obs_position_sequence': obs_trajs,
            'obs_velocity_sequence': obs_vel,
            'obs_timestep_sequence': obs_timesteps,
            'last_obs_positions': last_obs_positions,
            'last_obs_timesteps': last_obs_timesteps,

            'pred_identity_sequence': pred_ids,
            'pred_position_sequence': pred_trajs,
            'pred_velocity_sequence': pred_vel,
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

        # visualization stuff
        fig, ax = plt.subplots(1, 6)
        visualize.visualize_training_instance(
            draw_ax=ax[0], instance_dict=self.temp_generator.__getitem__(idx)
        )

        self.visualize(
            data_dict=data_dict,
            draw_ax=ax[1],
            draw_ax_sequences=ax[2],
            draw_ax_dist_transformed_map=ax[3],
            draw_ax_probability_map=ax[4],
            draw_ax_nlog_probability_map=ax[5]
        )
        plt.show()

        return data_dict

    def visualize_sequences(self, data_dict: Dict, draw_ax: matplotlib.axes.Axes):
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

        homogeneous_last_obs_pos = torch.cat((last_obs_pos, torch.ones([*last_obs_pos.shape[:-1], 1])), dim=-1).transpose(-1, -2)
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
            s = 5 * (timestep + self.T_obs)
            draw_ax.scatter(pos[0], pos[1], marker='x', s=5 + s, color=c)
            old_pos = pos - vel
            draw_ax.plot([old_pos[0], pos[0]], [old_pos[1], pos[1]], color=c, linestyle='--', alpha=0.8)

        homogeneous_pred_trajs = torch.cat((pred_trajs, torch.ones([*pred_trajs.shape[:-1], 1])), dim=-1).transpose(-1, -2)
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
        scene_map_img = data_dict['scene_map']              # [C, H, W]
        occlusion_map_img = data_dict['occlusion_map']      # [H, W]

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


if __name__ == '__main__':
    from utils.utils import prepare_seed
    from tqdm import tqdm
    print(sdd_conf.REPO_ROOT)

    n_calls = 10000
    # config_str = 'sdd_agentformer_pre'
    config_str = 'sdd_occlusion_agentformer_pre'

    # ####################################################################################################################
    # config = Config(config_str)
    # prepare_seed(config.seed)
    # log = open(os.path.join(config.log_dir, 'log.txt'), "a+")
    # time_str = get_timestring()
    # print_log("time str: {}".format(time_str), log)
    # print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    # print_log("torch version : {}".format(torch.__version__), log)
    # print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    #
    # generator = AgentFormerDataGeneratorForSDD(config, log, split='train')
    #
    # generator.shuffle()
    # for i in tqdm(range(n_calls)):
    #     if i < 2280:
    #         continue
    #     if i == 2280:
    #         generator.index = 2280
    #
    #     fig, ax = plt.subplots(1, 5)
    #     visualize.visualize_training_instance(
    #         draw_ax=ax[0], instance_dict=generator.dataset.__getitem__(generator.sample_list[generator.index])
    #     )
    #
    #     data_dict = generator()
    #
    #     generator.visualize(
    #         draw_ax=ax[1], data_dict=data_dict, draw_ax_dt_map=ax[2], draw_ax_p_occl_map=ax[3], draw_ax_min_log_p_occl_map=ax[4]
    #     )
    #
    #     plt.show()
    #
    #
    # ####################################################################################################################

    config = Config(config_str)
    prepare_seed(config.seed)

    log = open(os.path.join(config.log_dir, 'log.txt'), "a+")
    time_str = get_timestring()
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)


    generator = TorchDataGeneratorSDD(parser=config, log=log, split='train')

    for idx in range(10):
        generator.__getitem__(idx)




