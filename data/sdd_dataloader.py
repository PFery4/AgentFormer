import os.path
import random
from io import TextIOWrapper

import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.path import Path
import torch
import sys
import numpy as np
import skgeom as sg

from data.map import GeometricMap
from utils.config import Config
from utils.utils import print_log, get_timestring

from typing import Tuple, List, Optional
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
        self.frame_skip = parser.get('frame_skip', 1)
        self.rand_rot_scene = parser.get('rand_rot_scene', False)
        self.max_train_agent = parser.get('max_train_agent', 100)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        self.sdd_config = sdd_conf.get_config(parser.sdd_config_file_name)
        if not parser.get('sdd_occlusion_data', False):
            full_dataset = StanfordDroneDataset(self.sdd_config)
            self.traj_processing = self.traj_processing_without_occlusion
        else:
            full_dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config)
            self.traj_processing = self.traj_processing_with_occlusion
        print(f"instantiating dataloader from {full_dataset.__class__} class")

        # TODO: investigate whether a split strategy such as the one used here won't possibly result in data leakage
        # No, it won't, so long as normalization does not involve data from the test/val splits.
        split_proportions = [0.7, 0.2, 0.1]
        train_size = int(split_proportions[0] * len(full_dataset))
        val_size = int(split_proportions[1] * len(full_dataset))
        test_size = len(full_dataset) - (train_size + val_size)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':
            self.dataset = train_dataset
        elif self.split == 'val':
            self.dataset = val_dataset
        elif self.split == 'test':
            self.dataset = test_dataset
        else:
            assert False, 'error'

        self.num_total_samples = len(self.dataset)

        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

        self.sample_list = list(range(self.num_total_samples))
        self.index = 0

        self.traj_scale = parser.traj_scale

    def shuffle(self) -> None:
        random.shuffle(self.sample_list)

    def is_epoch_end(self) -> bool:
        if self.index >= self.num_total_samples:
            self.index = 0
            return True
        else:
            return False

    # @staticmethod
    # def compute_agent_trajs_and_extract_ids(extracted_data: dict, scene_map: GeometricMap):
    #     # compute all agent trajectories (by transforming to map coords) and extract agent ids
    #     trajs = scene_map.to_map_points(
    #         scene_pts=np.stack(
    #             [agent.get_traj_section(extracted_data['full_window'])
    #              for agent in extracted_data['agents']]
    #         )
    #     )       # [N, T, 2]
    #     ids = np.stack([agent.id for agent in extracted_data['agents']])        # [N]
    #     return trajs, ids

    def check_n_agents_and_subsample(self, ids: NDArray, trajs:NDArray, obs_mask: NDArray):
        if ids.shape[0] > self.max_train_agent:     # todo: add 'self.training'
            keep_indices = np.sort(np.random.choice(ids.shape[0], self.max_train_agent, replace=False))
            return ids[keep_indices], trajs[keep_indices], obs_mask[keep_indices]
        return ids, trajs, obs_mask

    def traj_processing_without_occlusion(
            self, trajs: NDArray, **kwargs
    ):
        scene_map = kwargs['scene_map']             # GeometricMap
        past_window = kwargs['past_window']         # NDArray   [T_obs]
        ids = kwargs['ids']                         # NDArray   [N]

        scene_map.set_homography(np.eye(3))

        obs_mask = np.full(trajs.shape[:2], True)
        obs_mask[..., len(past_window):] = False

        self.check_n_agents_and_subsample(ids, trajs, obs_mask)

        # performing normalization based on points we have observed
        points = trajs[obs_mask, :]     # NDArray [P, 2]

        # normalize
        min = np.min(points, axis=0)
        max = np.max(points, axis=0)

        shift = (max + min) / 2
        trajs -= shift
        points -= shift
        scene_map.translation(shift)

        scale = np.max((max - min) / 2)
        trajs /= scale
        points /= scale
        scene_map.scale(scaling=scale)

        # cropping the scene_map
        box_coords = np.array([[-1, -1], [1, 1]])
        k = 2.0
        crop_coords = scene_map.to_map_points(box_coords * k)
        scene_map.square_crop(crop_coords=crop_coords, h_scaling=k)

        out_dict = {
            "ids": ids,
            "trajs": trajs,
            "obs_mask": obs_mask,
            "scene_map": scene_map,
            "occlusion_map": np.ones(scene_map.get_map_dimensions()),
            "ego": None,
            "ego_visipoly": None,
            "occluders": None,
            "occlusion_case": False
        }
        return out_dict

    def traj_processing_with_occlusion(
            self, trajs: NDArray, **kwargs
    ):
        if np.any(np.isnan(kwargs['ego'])):
            print(f"NAN NAN {kwargs['ego']=}")
            return self.traj_processing_without_occlusion(trajs=trajs, **kwargs)

        print(f"YES EGO {kwargs['ego']=}")

        scene_map = kwargs['scene_map']             # GeometricMap
        orig_ego = kwargs['ego']                    # NDArray   [2]
        orig_occluders = kwargs['occluders']        # List[List[NDArray]]
        past_window = kwargs['past_window']         # NDArray   [T_obs]
        ids = kwargs['ids']                         # NDArray   [N]

        # compute ego and occluder positions (transforming to map coords)
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

        self.check_n_agents_and_subsample(ids, trajs, obs_mask)

        # performing all shifting / normalization only based on points we have observed
        # (using all past points, even unobserved ones, would constitute data leakage)
        points = trajs[obs_mask, :]  # NDArray [P, 2]

        # normalize
        min = np.min(points, axis=0)
        max = np.max(points, axis=0)

        shift = (max + min) / 2
        ego -= shift
        trajs -= shift
        points -= shift
        ego_visipoly = sg.Polygon(ego_visipoly.coords - shift)
        scene_map.translation(shift)

        scale = np.max((max - min) / 2)
        ego /= scale
        trajs /= scale
        points /= scale
        ego_visipoly = sg.Polygon(ego_visipoly.coords / scale)
        scene_map.scale(scaling=scale)

        # cropping the scene_map
        box_coords = np.array([[-1, -1], [1, 1]])
        k = 2.0
        crop_coords = scene_map.to_map_points(box_coords * k)
        scene_map.square_crop(crop_coords=crop_coords, h_scaling=k)

        # compute occlusion map
        map_dims = scene_map.get_map_dimensions()
        occ_x = np.arange(map_dims[0])
        occ_y = np.arange(map_dims[1])
        xy = np.dstack((np.meshgrid(occ_x, occ_y))).reshape((-1, 2))
        mpath = Path(scene_map.to_map_points(ego_visipoly.coords))
        occlusion_map = mpath.contains_points(xy).reshape(*reversed(map_dims))          # NDArray [map_res, map_res]

        out_dict = {
            "ids": ids,
            "trajs": trajs,
            "obs_mask": obs_mask,
            "scene_map": scene_map,
            "occlusion_map": occlusion_map,
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
        scene_map.mirror_expand()

        # load trajectories and agent identities
        trajs = np.stack(
            [agent.get_traj_section(extracted_data['full_window'])
             for agent in extracted_data['agents']]
        )       # [N, T, 2]
        ids = np.stack([agent.id for agent in extracted_data['agents']])        # [N]

        # rotate_the_map
        if self.rand_rot_scene:
            data['theta'] = float(np.random.rand() * 2 * np.pi)
        else:
            data['theta'] = 0.0
        scene_map.rotate_around_center(data['theta'])

        trajs = scene_map.to_map_points(trajs)

        fig, ax = plt.subplots()
        visualize.visualize_training_instance(
            draw_ax=ax, instance_dict=extracted_data
        )
        plt.show()
        print("OK")

        processed_data = self.traj_processing_with_occlusion(
            trajs=trajs,
            scene_map=scene_map,
            ego=extracted_data['ego_point'],
            occluders=extracted_data['occluders'],
            past_window=extracted_data['past_window'],
            ids=ids
        )

        fig, ax = plt.subplots(1, 3)
        visualize.visualize_training_instance(
            draw_ax=ax[0], instance_dict=extracted_data
        )
        self.visualize(
            draw_ax=ax[1],
            scene_map=processed_data['scene_map'],
            trajs=processed_data['trajs'],
            occluders=processed_data['occluders'],
            ego=processed_data['ego'],
            ego_visipoly=processed_data['ego_visipoly'],
            plot_norm_box=True
        )
        ax[2].imshow(processed_data['occlusion_map'])
        plt.show()
        print(zblu)

        #############################################################################################################


        # TODO: MAKE SURE THE ASSIGNMENT IS PERFORMED CORRECTLY BELOW:
        data['full_motion_3D'] = torch.from_numpy(trajs)
        data['valid_id'] = torch.from_numpy(ids)
        data['obs_mask'] = torch.from_numpy(obs_mask)
        data['ego'] = torch.from_numpy(ego)
        data['occluders'] = torch.from_numpy(np.stack(occluders))       # [n_occluders, 2, 2]
        data['ego_visipoly'] = ego_visipoly                             # sg.Polygon
        data['occlusion_map'] = torch.from_numpy(occlusion_map)         # [H, W]
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
            scene_map: GeometricMap,
            trajs: np.array,
            occluders: Optional[List[np.array]] = None,
            ego: Optional[np.array] = None,
            ego_visipoly: Optional[sg.Polygon] = None,
            plot_norm_box: bool = False,
            plot_crop_box: bool = False
    ) -> None:
        draw_ax.set_xlim(0., scene_map.get_map_dimensions()[0])
        draw_ax.set_ylim(scene_map.get_map_dimensions()[1], 0.)
        draw_ax.imshow(scene_map.as_image())
        plot_trajs = scene_map.to_map_points(trajs)
        draw_ax.scatter(plot_trajs[..., 0], plot_trajs[..., 1], marker='x', s=20)
        plot_occl = []
        if occluders is not None:
            for occluder in occluders:
                p1 = scene_map.to_map_points(scene_pts=occluder[0])
                p2 = scene_map.to_map_points(scene_pts=occluder[1])
                plot_occl.append((p1, p2))
                draw_ax.plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]], c='black')
        if ego is not None:
            plot_ego = scene_map.to_map_points(ego)
            draw_ax.scatter(plot_ego[0], plot_ego[1], marker='D', c='yellow', s=30)
        if ego_visipoly is not None:
            plot_ego_visipoly = sg.Polygon(scene_map.to_map_points(ego_visipoly.coords))
            plot_scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
            plot_regions = sg.PolygonSet(plot_scene_boundary).difference(plot_ego_visipoly)
            [plot_sg_polygon(ax=draw_ax, poly=poly, edgecolor='red', facecolor='red', alpha=0.2)
             for poly in plot_regions.polygons]
        if plot_norm_box:
            plot_box = np.array([[-1, -1],
                                 [-1, 1],
                                 [1, 1],
                                 [1, -1],
                                 [-1, -1]])
            plot_box = scene_map.to_map_points(plot_box)
            draw_ax.plot(plot_box[..., 0], plot_box[..., 1], c='r')
        if plot_crop_box:
            k = 2.0         # todo: move constant to a member variable
            crop_coords = scene_map.to_map_points(np.array([[-1, -1], [1, 1]]) * k)
            crop_box = np.array([[crop_coords[0, 0], crop_coords[0, 1]],
                                 [crop_coords[0, 0], crop_coords[1, 1]],
                                 [crop_coords[1, 0], crop_coords[1, 1]],
                                 [crop_coords[1, 0], crop_coords[0, 1]],
                                 [crop_coords[0, 0], crop_coords[0, 1]]])
            draw_ax.plot(crop_box[..., 0], crop_box[..., 1], c='k')


if __name__ == '__main__':
    print(sdd_conf.REPO_ROOT)

    n_calls = 1
    # config_str = 'sdd_agentformer_pre'
    config_str = 'sdd_occlusion_agentformer_pre'

    ####################################################################################################################
    config = Config(config_str)
    log = open(os.path.join(config.log_dir, 'log.txt'), "a+")
    time_str = get_timestring()
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)

    generator = AgentFormerDataGeneratorForSDD(config, log, split='train')

    print(f"{generator.rand_rot_scene=}")

    generator.shuffle()
    for i in range(n_calls):
        print("\nCALLING")
        data_dict = generator()
        print(f"{data_dict['scene_map']=}")
        # [print(f"{k}: {type(v)}") for k, v in generator().items()]
