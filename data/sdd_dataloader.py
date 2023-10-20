import os.path
import random
from io import TextIOWrapper

import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.colors as colors
import numpy
from matplotlib.path import Path
import torch
import sys
import numpy as np
import skgeom as sg
from scipy.ndimage import distance_transform_edt
from scipy.special import log_softmax, softmax

from data.map import GeometricMap
from utils.config import Config
from utils.utils import print_log, get_timestring

from typing import Dict, Optional
from numpy.typing import NDArray

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
from src.visualization.plot_utils import plot_sg_polygon
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
        try:
            self.map_res = parser.get('global_map_encoder').get('map_resolution', 800)
        except:
            self.map_res = 800
        self.map_crop_coords = np.array(
            [[-self.map_side, -self.map_side], [self.map_side, self.map_side]]
        ) * self.traj_scale / 2

    def shuffle(self) -> None:
        random.shuffle(self.sample_list)

    def is_epoch_end(self) -> bool:
        if self.index >= self.num_total_samples:
            self.index = 0
            return True
        else:
            return False

    def check_n_agents_and_subsample(self, ids: NDArray):
        # ids is an NDArray of shape [N]
        # TODO: check that we are never eliminating the target agent for occlusion
        if ids.shape[0] > self.max_train_agent:     # TODO: add 'self.training'
            keep_indices = np.sort(np.random.choice(ids.shape[0], self.max_train_agent, replace=False))
            return keep_indices
        return numpy.arange(ids.shape[0])

    def cropped_scene_map(self, scene_map: GeometricMap):
        # cropping the scene_map
        crop_coords = scene_map.to_map_points(self.map_crop_coords)
        scene_map.square_crop(
            crop_coords=crop_coords,
            side_length=self.map_side,
            resolution=self.map_res
        )
        return scene_map

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

        keep_indices = self.check_n_agents_and_subsample(ids)
        ids = ids[keep_indices]
        trajs = trajs[keep_indices]
        obs_mask = obs_mask[keep_indices]

        # calculating the mean of all points at t_0
        mean_point = np.mean(trajs[:, past_window.shape[0]-1, :], axis=0)

        # normalize
        trajs -= mean_point
        scene_map.translation(mean_point)

        scaling = self.traj_scale * m_per_px
        trajs *= scaling
        scene_map.scale(scaling=1/scaling)

        scene_map = self.cropped_scene_map(scene_map)

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
        # other agents (those who are insufficiently observed) are discarded
        keep = (np.sum(obs_mask, axis=1) >= 2)  # [N]
        trajs, ids, obs_mask = trajs[keep], ids[keep], obs_mask[keep]

        keep = self.check_n_agents_and_subsample(ids)
        ids = ids[keep]
        trajs = trajs[keep]
        obs_mask = obs_mask[keep]

        # performing all shifting / normalization only based on points we have observed
        # (using all past points, even unobserved ones, would constitute data leakage)
        last_obs_indices = trajs.shape[1] - np.argmax(obs_mask[:, ::-1], axis=1) - 1
        last_obs_points = trajs[np.arange(ids.shape[0]), last_obs_indices, :]
        mean_point = np.mean(last_obs_points, axis=0)

        # normalize
        ego -= mean_point
        trajs -= mean_point
        ego_visipoly = sg.Polygon(ego_visipoly.coords - mean_point)
        scene_map.translation(mean_point)

        scaling = self.traj_scale * m_per_px

        ego *= scaling
        trajs *= scaling
        ego_visipoly = sg.Polygon(ego_visipoly.coords * scaling)
        scene_map.scale(scaling=1/scaling)

        # # cropping the scene_map
        # box_coords = np.array([[-1, -1], [1, 1]])
        # k = 2.0
        # crop_coords = scene_map.to_map_points(box_coords * k)
        # scene_map.square_crop(crop_coords=crop_coords, h_scaling=k)
        scene_map = self.cropped_scene_map(scene_map)

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
            m_per_px=extracted_data['m/px']
        )

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
        data['timesteps'] = torch.from_numpy(
            np.arange(len(extracted_data['full_window'])) - len(extracted_data['past_window']) + 1
        )
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


if __name__ == '__main__':
    from utils.utils import prepare_seed
    print(sdd_conf.REPO_ROOT)

    n_calls = 100
    # config_str = 'sdd_agentformer_pre'
    config_str = 'sdd_occlusion_agentformer_pre'

    ####################################################################################################################
    config = Config(config_str)
    prepare_seed(config.seed)
    log = open(os.path.join(config.log_dir, 'log.txt'), "a+")
    time_str = get_timestring()
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)

    generator = AgentFormerDataGeneratorForSDD(config, log, split='train')

    generator.shuffle()
    for i in range(n_calls):

        fig, ax = plt.subplots(1, 5)

        visualize.visualize_training_instance(
            draw_ax=ax[0], instance_dict=generator.dataset.__getitem__(generator.sample_list[generator.index])
        )

        data_dict = generator()

        generator.visualize(
            draw_ax=ax[1], data_dict=data_dict, draw_ax_dt_map=ax[2], draw_ax_p_occl_map=ax[3], draw_ax_min_log_p_occl_map=ax[4]
        )

        plt.show()
