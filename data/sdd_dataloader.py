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

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
from src.visualization.plot_utils import plot_sg_polygon
import src.visualization.sdd_visualize as visualize
import src.occlusion_simulation.visibility as visibility
import src.occlusion_simulation.polygon_generation as poly_gen
import src.data.sdd_extract as sdd_extract


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

        self.sdd_config = sdd_extract.get_config(parser.sdd_config_file_name)
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

    def traj_processing_without_occlusion(self, extracted_data: dict, scene_map: GeometricMap):
        trajs, ids = self.compute_agent_trajs_and_extract_ids(extracted_data=extracted_data, scene_map=scene_map)

        full_window_occlusion_masks = np.tile(
            np.concatenate(
                (np.full_like(extracted_data['past_window'], True),
                 np.full_like(extracted_data['future_window'], False))
            ), (len(ids), 1)
        ).astype(bool)              # [N, T]

        return trajs, ids, full_window_occlusion_masks, None, None, None, None

    def traj_processing_with_occlusion(self, extracted_data: dict, scene_map: GeometricMap):
        trajs, ids = self.compute_agent_trajs_and_extract_ids(extracted_data=extracted_data, scene_map=scene_map)

        # compute ego and occluder positions (transforming to map coords)
        ego = scene_map.to_map_points(scene_pts=extracted_data['ego_point'])
        occluders = []
        for occluder in extracted_data['occluders']:
            p1 = scene_map.to_map_points(scene_pts=occluder[0])
            p2 = scene_map.to_map_points(scene_pts=occluder[1])
            occluders.append((p1, p2))

        # compute visibility polygon
        scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
        ego_visipoly = visibility.compute_visibility_polygon(
            ego_point=ego,
            occluders=occluders,
            boundary=scene_boundary
        )

        # obtain observation_mask
        full_window_occlusion_masks = visibility.occlusion_mask(
            points=trajs.reshape(-1, trajs.shape[-1]),
            ego_visipoly=ego_visipoly
        ).reshape(trajs.shape[:-1])     # [N, T]
        full_window_occlusion_masks[..., len(extracted_data['past_window']):] = False

        # check for all agents that they have at least 2 observations available for the model to process
        keep = (np.sum(full_window_occlusion_masks, axis=1) >= 2)        # [N]

        # compute occlusion map
        map_dims = scene_map.get_map_dimensions()
        occ_x = np.arange(map_dims[0])
        occ_y = np.arange(map_dims[1])
        xy = np.dstack((np.meshgrid(occ_x, occ_y))).reshape((-1, 2))
        mpath = Path(ego_visipoly.coords)
        occlusion_map = mpath.contains_points(xy).reshape(*reversed(map_dims))
        occlusion_map[0, ...] = occlusion_map[1, ...]       # lazy fix: some bug makes the first row all False

        return trajs[keep], ids[keep], full_window_occlusion_masks[keep], ego, occluders, ego_visipoly, occlusion_map

    def convert_to_preprocessor_data(self, extracted_data: dict) -> dict:
        # [print(f"{k}: {type(v)}") for k, v in extracted_data.items()]

        # fig, ax = plt.subplots(1)
        # visualize.visualize_training_instance(
        #     draw_ax=ax, instance_dict=extracted_data
        # )
        # print(f"{extracted_data['scene_image'].shape=}")
        # print(f"{extracted_data['ego_point']=}")
        # print(f"{extracted_data['occluders']=}")
        # plt.show()

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
        if self.rand_rot_scene:
            data['theta'] = float(np.random.rand() * 2 * np.pi)
        else:
            data['theta'] = 0.0
        scene_map.rotate_around_center(data['theta'])

        trajs, ids, obs_mask, ego, occluders, ego_visipoly, occlusion_map = self.traj_processing(
            extracted_data=extracted_data,
            scene_map=scene_map
        )

        # subsample identities if too many:
        if ids.shape[0] > self.max_train_agent:     # TODO: add 'self.training' to if statement
            keep_indices = np.sort(np.random.choice(ids.shape[0], self.max_train_agent, replace=False))
            trajs = trajs[keep_indices]
            ids = ids[keep_indices]
            obs_mask = obs_mask[keep_indices]

        # # plotting for example:
        # fig, axes = plt.subplots(1, 3)
        # visualize.visualize_training_instance(
        #     draw_ax=axes[0], instance_dict=extracted_data
        # )
        #
        # axes[1].set_xlim(0., scene_map.get_map_dimensions()[0])
        # axes[1].set_ylim(scene_map.get_map_dimensions()[1], 0.)
        # axes[1].imshow(scene_map.as_image())
        #
        # for traj, mask in zip(trajs, obs_mask):
        #     occluded = traj[~mask]
        #     axes[1].scatter(occluded[..., 0], occluded[..., 1], marker='x', s=30, c='black')
        #     axes[1].plot(traj[..., 0], traj[..., 1])
        # if ego is not None and occluders is not None and ego_visipoly is not None:
        #     for occluder in occluders:
        #         axes[1].plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]], c='black')
        #     axes[1].scatter(ego[0], ego[1], marker='D', c='yellow', s=30)
        #     scene_boundary = poly_gen.default_rectangle(corner_coords=(reversed(scene_map.get_map_dimensions())))
        #     occluded_regions = sg.PolygonSet(scene_boundary).difference(ego_visipoly)
        #     [plot_sg_polygon(ax=axes[1], poly=poly, edgecolor='red', facecolor='red', alpha=0.2)
        #      for poly in occluded_regions.polygons]
        #
        # axes[2].imshow(occlusion_map)
        # plt.show()

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


if __name__ == '__main__':
    print(sdd_extract.REPO_ROOT)

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
