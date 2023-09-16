import os.path
import random
from io import TextIOWrapper

import matplotlib.pyplot as plt
from matplotlib.path import Path
import torch
import sys
import numpy as np
import skgeom as sg

from data.map import GeometricMap
from utils.config import Config
from utils.utils import print_log, get_timestring

from typing import Tuple, List

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

    @staticmethod
    def compute_agent_trajs_and_extract_ids(extracted_data: dict, scene_map: GeometricMap):
        # compute all agent trajectories (by transforming to map coords) and extract agent ids
        trajs = scene_map.to_map_points(
            scene_pts=np.stack(
                [agent.get_traj_section(extracted_data['full_window'])
                 for agent in extracted_data['agents']]
            )
        )       # [N, T, 2]
        ids = np.stack([agent.id for agent in extracted_data['agents']])        # [N]
        return trajs, ids

    def traj_processing_without_occlusion(
            self, extracted_data: dict, scene_map: GeometricMap
    ) -> Tuple[np.array, np.array, np.array, None, None, None, None]:
        trajs, ids = self.compute_agent_trajs_and_extract_ids(
            extracted_data=extracted_data, scene_map=scene_map
        )                           # [N, T, 2] , [N]

        full_window_occlusion_masks = np.tile(
            np.concatenate(
                (np.full_like(extracted_data['past_window'], True),
                 np.full_like(extracted_data['future_window'], False))
            ), (len(ids), 1)
        ).astype(bool)              # [N, T]

        return trajs, ids, full_window_occlusion_masks, None, None, None, None

    def traj_processing_with_occlusion(
            self, extracted_data: dict, scene_map: GeometricMap
    ) -> Tuple[np.array, np.array, np.array, np.array, List[np.array], sg.Polygon, np.array]:
        pass
    #
    #     trajs, ids = self.compute_agent_trajs_and_extract_ids(extracted_data=extracted_data, scene_map=scene_map)
    #
    #     # compute ego and occluder positions (transforming to map coords)
    #     ego = scene_map.to_map_points(scene_pts=extracted_data['ego_point'])
    #     occluders = []
    #     for occluder in extracted_data['occluders']:
    #         p1 = scene_map.to_map_points(scene_pts=occluder[0])
    #         p2 = scene_map.to_map_points(scene_pts=occluder[1])
    #         occluders.append((p1, p2))
    #     # ego = extracted_data['ego_point']
    #     # occluders = extracted_data['occluders']
    #
    #     # compute visibility polygon
    #     scene_boundary = poly_gen.default_rectangle(corner_coords=reversed(scene_map.get_map_dimensions()))
    #     ego_visipoly = visibility.compute_visibility_polygon(
    #         ego_point=ego,
    #         occluders=occluders,
    #         boundary=scene_boundary
    #     )
    #
    #     # obtain observation_mask
    #     full_window_occlusion_masks = visibility.occlusion_mask(
    #         points=trajs.reshape(-1, trajs.shape[-1]),
    #         ego_visipoly=ego_visipoly
    #     ).reshape(trajs.shape[:-1])     # [N, T]
    #     full_window_occlusion_masks[..., len(extracted_data['past_window']):] = False
    #
    #     # check for all agents that they have at least 2 observations available for the model to process
    #     # other agents (those who are insufficiently observed) are discarded
    #     keep = (np.sum(full_window_occlusion_masks, axis=1) >= 2)        # [N]
    #     trajs, ids, full_window_occlusion_masks = trajs[keep], ids[keep], full_window_occlusion_masks[keep]
    #
    #     if ids.shape[0] > self.max_train_agent:     # todo: add 'self.training'
    #         keep_indices = np.sort(np.random.choice(ids.shape[0], self.max_train_agent, replace=False))
    #         trajs, ids, full_window_occlusion_masks = trajs[keep_indices], ids[keep_indices], full_window_occlusion_masks[keep_indices]
    #
    #     # performing all shifting / normalization only based on points we have observed
    #     # (using all past points, even unobserved ones, would constitute data leakage)
    #     points = trajs[full_window_occlusion_masks, :]      # [P, 2]
    #
    #     print(f"{points, points.shape=}")
    #
    #     # normalize
    #     min = np.min(points, axis=0)
    #     max = np.max(points, axis=0)
    #
    #     shift = (max + min)/2
    #     scale = np.max((max - min)/2)
    #
    #     print(f"{shift, scale, 1/scale=}")
    #     print(f"{scene_map.homography=}")
    #
    #     print(f"{points[0]=}")
    #
    #     trajs -= shift
    #     points -= shift
    #     scene_map.set_homography(np.eye(3))
    #     scene_map.translation(shift)
    #
    #     # trajs /= scale
    #     # points /= scale
    #     #
    #     # k = 2.0
    #     #
    #     # scene_map.get_square_crop(center=shift, extent=k*scale)
    #
    #     # box = np.array([[-1, -1],
    #     #                 [-1, 1],
    #     #                 [1, 1],
    #     #                 [1, -1],
    #     #                 [-1, -1]])
    #
    #     # compute occlusion map
    #     map_dims = scene_map.get_map_dimensions()
    #     occ_x = np.arange(map_dims[0])
    #     occ_y = np.arange(map_dims[1])
    #     xy = np.dstack((np.meshgrid(occ_x, occ_y))).reshape((-1, 2))
    #     mpath = Path(ego_visipoly.coords)
    #     occlusion_map = mpath.contains_points(xy).reshape(*reversed(map_dims))
    #     occlusion_map[0, ...] = occlusion_map[1, ...]       # quick fix: some bug makes the first row all False (might need revision)
    #
    #     return trajs[keep], ids[keep], full_window_occlusion_masks[keep], ego, occluders, ego_visipoly, occlusion_map

    def convert_to_preprocessor_data(self, extracted_data: dict) -> dict:

        data = dict()

        heading = None
        # from the nuscenes implementation:
        # pred mask is a numpy array, of shape (n_agents,), with values either 1 or 0
        pred_mask = None

        # PLOTTING FOR EXAMPLE #######################################################################################
        fig, axes = plt.subplots(1, 7)
        visualize.visualize_training_instance(
            draw_ax=axes[0], instance_dict=extracted_data
        )
        # PLOTTING FOR EXAMPLE #######################################################################################

        # perform the loading of the map
        scene_map = GeometricMap(
            data=np.transpose(extracted_data['scene_image'], (2, 1, 0)),
            homography=np.eye(3)
        )

        # expand the map
        scene_map.mirror_expand()

        # rotate_the_map
        if self.rand_rot_scene:
            data['theta'] = float(np.random.rand() * 2 * np.pi)
            data['theta'] = np.pi/4
        else:
            data['theta'] = 0.0
        print(f"{data['theta'], data['theta']*180/np.pi=}")
        scene_map.rotate_around_center(data['theta'])

        ##############################################################################################################
        # trajs, ids, obs_mask, ego, occluders, ego_visipoly, occlusion_map = self.traj_processing(
        #     extracted_data=extracted_data,
        #     scene_map=scene_map
        # )
        ##############################################################################################################

        trajs = np.stack(
            [agent.get_traj_section(extracted_data['full_window'])
             for agent in extracted_data['agents']]
        )       # [N, T, 2]
        ids = np.stack([agent.id for agent in extracted_data['agents']])        # [N]

        # PLOTTING FOR EXAMPLE #######################################################################################
        axes[1].set_xlim(0., scene_map.get_map_dimensions()[0])
        axes[1].set_ylim(scene_map.get_map_dimensions()[1], 0.)
        axes[1].imshow(scene_map.as_image())
        plot_trajs = scene_map.to_map_points(trajs)
        axes[1].scatter(plot_trajs[..., 0], plot_trajs[..., 1], marker='x', s=20)
        # PLOTTING FOR EXAMPLE #######################################################################################

        trajs = scene_map.to_map_points(trajs)
        # compute ego and occluder positions (transforming to map coords)
        ego = scene_map.to_map_points(scene_pts=extracted_data['ego_point'])
        occluders = []
        for occluder in extracted_data['occluders']:
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
        full_window_occlusion_masks = visibility.occlusion_mask(
            points=trajs.reshape(-1, trajs.shape[-1]),
            ego_visipoly=ego_visipoly
        ).reshape(trajs.shape[:-1])  # [N, T]
        full_window_occlusion_masks[..., len(extracted_data['past_window']):] = False

        # check for all agents that they have at least 2 observations available for the model to process
        # other agents (those who are insufficiently observed) are discarded
        keep = (np.sum(full_window_occlusion_masks, axis=1) >= 2)  # [N]
        trajs, ids, full_window_occlusion_masks = trajs[keep], ids[keep], full_window_occlusion_masks[keep]

        if ids.shape[0] > self.max_train_agent:  # todo: add 'self.training'
            keep_indices = np.sort(np.random.choice(ids.shape[0], self.max_train_agent, replace=False))
            trajs, ids, full_window_occlusion_masks = trajs[keep_indices], ids[keep_indices], \
                                                      full_window_occlusion_masks[keep_indices]

        # PLOTTING FOR EXAMPLE #######################################################################################
        ax_idx = 2
        axes[ax_idx].set_xlim(0., scene_map.get_map_dimensions()[0])
        axes[ax_idx].set_ylim(scene_map.get_map_dimensions()[1], 0.)
        axes[ax_idx].imshow(scene_map.as_image())
        plot_trajs = scene_map.to_map_points(trajs)
        axes[ax_idx].scatter(plot_trajs[..., 0], plot_trajs[..., 1], marker='x', s=20)
        plot_occl = []
        for occluder in occluders:
            p1 = scene_map.to_map_points(scene_pts=occluder[0])
            p2 = scene_map.to_map_points(scene_pts=occluder[1])
            plot_occl.append((p1, p2))
            axes[ax_idx].plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]], c='black')
        plot_ego = scene_map.to_map_points(ego)
        axes[ax_idx].scatter(plot_ego[0], plot_ego[1], marker='D', c='yellow', s=30)
        plot_ego_visipoly = ego_visipoly
        plot_scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
        plot_regions = sg.PolygonSet(plot_scene_boundary).difference(plot_ego_visipoly)
        [plot_sg_polygon(ax=axes[ax_idx], poly=poly, edgecolor='red', facecolor='red', alpha=0.2)
         for poly in plot_regions.polygons]
        # PLOTTING FOR EXAMPLE #######################################################################################

        # performing all shifting / normalization only based on points we have observed
        # (using all past points, even unobserved ones, would constitute data leakage)
        points = trajs[full_window_occlusion_masks, :]  # [P, 2]

        # normalize
        min = np.min(points, axis=0)
        max = np.max(points, axis=0)

        shift = (max + min) / 2

        ego -= shift
        trajs -= shift
        points -= shift
        ego_visipoly = sg.Polygon(ego_visipoly.coords - shift)
        scene_map.translation(shift)

        # PLOTTING FOR EXAMPLE #######################################################################################
        ax_idx = 3
        axes[ax_idx].set_xlim(0., scene_map.get_map_dimensions()[0])
        axes[ax_idx].set_ylim(scene_map.get_map_dimensions()[1], 0.)
        axes[ax_idx].imshow(scene_map.as_image())
        plot_trajs = scene_map.to_map_points(trajs)
        axes[ax_idx].scatter(plot_trajs[..., 0], plot_trajs[..., 1], marker='x', s=20)
        plot_occl = []
        for occluder in occluders:
            p1 = scene_map.to_map_points(scene_pts=occluder[0])
            p2 = scene_map.to_map_points(scene_pts=occluder[1])
            plot_occl.append((p1, p2))
            axes[ax_idx].plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]], c='black')
        plot_ego = scene_map.to_map_points(ego)
        axes[ax_idx].scatter(plot_ego[0], plot_ego[1], marker='D', c='yellow', s=30)
        plot_ego_visipoly = sg.Polygon(scene_map.to_map_points(ego_visipoly.coords))
        plot_scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
        plot_regions = sg.PolygonSet(plot_scene_boundary).difference(plot_ego_visipoly)
        [plot_sg_polygon(ax=axes[ax_idx], poly=poly, edgecolor='red', facecolor='red', alpha=0.2)
         for poly in plot_regions.polygons]
        # PLOTTING FOR EXAMPLE #######################################################################################

        scale = np.max((max - min) / 2)
        ego /= scale
        trajs /= scale
        points /= scale
        ego_visipoly = sg.Polygon(ego_visipoly.coords / scale)
        scene_map.scale(scaling=scale)
        box_coords = np.array([[-1, -1], [1, 1]])
        k = 2.0
        crop_coords = scene_map.to_map_points(box_coords * k)

        # PLOTTING FOR EXAMPLE #######################################################################################
        ax_idx = 4
        axes[ax_idx].set_xlim(0., scene_map.get_map_dimensions()[0])
        axes[ax_idx].set_ylim(scene_map.get_map_dimensions()[1], 0.)
        axes[ax_idx].imshow(scene_map.as_image())
        plot_trajs = scene_map.to_map_points(trajs)
        axes[ax_idx].scatter(plot_trajs[..., 0], plot_trajs[..., 1], marker='x', s=20)
        plot_occl = []
        for occluder in occluders:
            p1 = scene_map.to_map_points(scene_pts=occluder[0])
            p2 = scene_map.to_map_points(scene_pts=occluder[1])
            plot_occl.append((p1, p2))
            axes[ax_idx].plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]], c='black')
        plot_ego = scene_map.to_map_points(ego)
        axes[ax_idx].scatter(plot_ego[0], plot_ego[1], marker='D', c='yellow', s=30)
        plot_ego_visipoly = sg.Polygon(scene_map.to_map_points(ego_visipoly.coords))
        plot_scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
        plot_regions = sg.PolygonSet(plot_scene_boundary).difference(plot_ego_visipoly)
        [plot_sg_polygon(ax=axes[ax_idx], poly=poly, edgecolor='red', facecolor='red', alpha=0.2)
         for poly in plot_regions.polygons]
        plot_box = np.array([[box_coords[0, 0], box_coords[0, 1]],
                             [box_coords[0, 0], box_coords[1, 1]],
                             [box_coords[1, 0], box_coords[1, 1]],
                             [box_coords[1, 0], box_coords[0, 1]],
                             [box_coords[0, 0], box_coords[0, 1]]])
        plot_box = scene_map.to_map_points(plot_box)
        axes[ax_idx].plot(plot_box[..., 0], plot_box[..., 1], c='r')
        crop_box = np.array([[crop_coords[0, 0], crop_coords[0, 1]],
                             [crop_coords[0, 0], crop_coords[1, 1]],
                             [crop_coords[1, 0], crop_coords[1, 1]],
                             [crop_coords[1, 0], crop_coords[0, 1]],
                             [crop_coords[0, 0], crop_coords[0, 1]]])
        axes[ax_idx].plot(crop_box[..., 0], crop_box[..., 1], c='k')
        # PLOTTING FOR EXAMPLE #######################################################################################

        # cropping the scene_map

        scene_map.square_crop(crop_coords=crop_coords, h_scaling=k)
        crop_coords = scene_map.to_map_points(box_coords * k)

        print(f"{ego_visipoly.coords=}")
        print(f"{scene_map.to_map_points(ego_visipoly.coords)=}")

        # compute occlusion map
        map_dims = scene_map.get_map_dimensions()
        occ_x = np.arange(map_dims[0])
        occ_y = np.arange(map_dims[1])
        xy = np.dstack((np.meshgrid(occ_x, occ_y))).reshape((-1, 2))
        mpath = Path(scene_map.to_map_points(ego_visipoly.coords))
        occlusion_map = mpath.contains_points(xy).reshape(*reversed(map_dims))

        # PLOTTING FOR EXAMPLE #######################################################################################
        ax_idx = 5
        axes[ax_idx].set_xlim(0., scene_map.get_map_dimensions()[0])
        axes[ax_idx].set_ylim(scene_map.get_map_dimensions()[1], 0.)
        axes[ax_idx].imshow(scene_map.as_image())
        plot_trajs = scene_map.to_map_points(trajs)
        axes[ax_idx].scatter(plot_trajs[..., 0], plot_trajs[..., 1], marker='x', s=20)
        plot_occl = []
        for occluder in occluders:
            p1 = scene_map.to_map_points(scene_pts=occluder[0])
            p2 = scene_map.to_map_points(scene_pts=occluder[1])
            plot_occl.append((p1, p2))
            axes[ax_idx].plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]], c='black')
        plot_ego = scene_map.to_map_points(ego)
        axes[ax_idx].scatter(plot_ego[0], plot_ego[1], marker='D', c='yellow', s=30)
        plot_ego_visipoly = sg.Polygon(scene_map.to_map_points(ego_visipoly.coords))
        plot_scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
        plot_regions = sg.PolygonSet(plot_scene_boundary).difference(plot_ego_visipoly)
        [plot_sg_polygon(ax=axes[ax_idx], poly=poly, edgecolor='red', facecolor='red', alpha=0.2)
         for poly in plot_regions.polygons]
        plot_box = np.array([[box_coords[0, 0], box_coords[0, 1]],
                             [box_coords[0, 0], box_coords[1, 1]],
                             [box_coords[1, 0], box_coords[1, 1]],
                             [box_coords[1, 0], box_coords[0, 1]],
                             [box_coords[0, 0], box_coords[0, 1]]])
        plot_box = scene_map.to_map_points(plot_box)
        axes[ax_idx].plot(plot_box[..., 0], plot_box[..., 1], c='r')
        crop_box = np.array([[crop_coords[0, 0], crop_coords[0, 1]],
                             [crop_coords[0, 0], crop_coords[1, 1]],
                             [crop_coords[1, 0], crop_coords[1, 1]],
                             [crop_coords[1, 0], crop_coords[0, 1]],
                             [crop_coords[0, 0], crop_coords[0, 1]]])
        axes[ax_idx].plot(crop_box[..., 0], crop_box[..., 1], c='k')

        axes[6].imshow(occlusion_map)
        # PLOTTING FOR EXAMPLE #######################################################################################

        plt.show()

        #############################################################################################################

        print(zblu)
        # TODO: MAKE SURE THE ASSIGNMENT IS PERFORMED CORRECTLY BELOW:
        data['full_motion_3D'] = torch.from_numpy(trajs)
        data['valid_id'] = torch.from_numpy(ids)
        data['obs_mask'] = torch.from_numpy(full_window_occlusion_masks)
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
