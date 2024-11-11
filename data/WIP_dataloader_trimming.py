import sys

from torch.utils.data import Dataset

import cv2
import h5py
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import os.path
import pandas as pd
from scipy.ndimage import distance_transform_edt
import skgeom as sg
import struct
import torch
from torchvision import transforms
from tqdm import tqdm

from data.map import TorchGeometricMap
from data.sdd_dataloader import dataset_dict, TorchDataGeneratorSDD
from utils.config import Config, REPO_ROOT
from utils.sdd_visualize import visualize
from utils.utils import prepare_seed, print_log

import src.occlusion_simulation.visibility as visibility        # TODO: MANAGE IMPORTS FROM OTHER REPO IN A BETTER WAY

from typing import Dict, List, Optional
Tensor = torch.Tensor
from io import TextIOWrapper


class PresavedDataset(Dataset):
    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    # This is a "quick fix".
    # We performed the wrong px/m coordinate conversion when computing the distance transformed map.
    # Ideally we should correct this in the TorchDatasetGenerator class.
    # the real fix is to have the distance transformed map scaled by the proper factor:
    # TorchDatasetGenerator.map_side / TorchDatasetGenerator.map_resolution
    coord_conv_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'coordinates_conversion.txt')
    coord_conv_table = pd.read_csv(coord_conv_dir, sep=';', index_col=('scene', 'video'))

    def __init__(self, parser: Config, log: Optional[TextIOWrapper] = None, split: str = 'train'):

        # TODO: REMOVE QUICK FIX, FIX DIRECTLY PLEASE
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

        # map specific parameters
        self.map_side = parser.get('scene_side_length', 80.0)               # [m]
        self.map_resolution = parser.get('global_map_resolution', 800)      # [px]
        self.traj_scale = parser.traj_scale
        self.map_crop_coords = torch.Tensor(
            [[-self.map_side, -self.map_side],
             [self.map_side, self.map_side]]
        ) * self.traj_scale / 2
        self.map_homography = torch.Tensor(
            [[self.map_resolution / self.map_side, 0., self.map_resolution / 2],
             [0., self.map_resolution / self.map_side, self.map_resolution / 2],
             [0., 0., 1.]]
        )
        self.struct_format = f'{int(self.map_resolution * self.map_resolution / 8)}B'
        assert self.map_resolution % 8 == 0

        # scene map parameters
        self.padding_px = 2075
        self.padded_images_path = os.path.join(REPO_ROOT, 'datasets', 'SDD', f'padded_images_{self.padding_px}')
        self.to_torch_image = transforms.ToTensor()

        # timesteps specific parameters
        self.T_obs = parser.past_frames
        self.T_pred = parser.future_frames
        self.T_total = self.T_obs + self.T_pred
        self.timesteps = torch.arange(-self.T_obs, self.T_pred) + 1

        # dataset identification
        dataset_dir_name = f'{self.occlusion_process}_imputed' if self.impute else self.occlusion_process
        self.dataset_dir = os.path.join(self.presaved_datasets_dir, dataset_dir_name, self.split)

        assert os.path.exists(self.dataset_dir)
        prnt_str = f"Dataset directory is:\n{self.dataset_dir}"
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

        ################################################################################################################
        self.hdf5_file = os.path.join(self.dataset_dir, 'dataset_v2.h5')    # TODO: maybe don't hard set filename
        assert os.path.exists(self.hdf5_file)

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

        if False:       # handle difficult cases
        # if parser.get('difficult', False):
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

        elif False:             # handle validation subsampling
        # elif self.split == 'val' and parser.get('validation_set_size', None) is not None:
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

        assert self.__len__() != 0
        prnt_str = f'total number of samples: {self.__len__()}'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)
        prnt_str = f'------------------------------ done --------------------------------\n'
        print_log(prnt_str, log=log) if log is not None else print(prnt_str)

    @staticmethod
    def last_observed_indices(obs_mask: Tensor) -> Tensor:
        # obs_mask [N, T]
        return obs_mask.shape[1] - torch.argmax(torch.flip(obs_mask, dims=[1]), dim=1) - 1  # [N]

    @staticmethod
    def last_observed_positions(trajs: Tensor, last_obs_indices: Tensor) -> Tensor:
        # trajs [N, T, 2]
        # last_obs_indices [N]
        return trajs[torch.arange(trajs.shape[0]), last_obs_indices, :]  # [N, 2]

    def last_observed_timesteps(self, last_obs_indices: Tensor) -> Tensor:
        # last_obs_indices [N]
        return self.timesteps[last_obs_indices]  # [N]

    def agent_grid(self, ids: Tensor) -> Tensor:
        # ids [N]
        return torch.hstack([ids.unsqueeze(1)] * self.T_total)  # [N, T]

    def timestep_grid(self, ids: Tensor) -> Tensor:
        return torch.vstack([self.timesteps] * ids.shape[0])  # [N, T]

    def predict_mask(self, last_obs_indices: Tensor) -> Tensor:
        # last_obs_indices [N]
        predict_mask = torch.full([last_obs_indices.shape[0], self.T_total], False)  # [N, T]
        pred_indices = last_obs_indices + 1  # [N]
        for i, pred_idx in enumerate(pred_indices):
            predict_mask[i, pred_idx:] = True
        return predict_mask

    def crop_scene_map(self, scene_map: TorchGeometricMap):
        scene_map.crop(crop_coords=scene_map.to_map_points(self.map_crop_coords), resolution=self.map_resolution)
        scene_map.set_homography(matrix=self.map_homography)

    def add_instance_identifiers(self, data_dict: Dict, idx: int):
        data_dict['frame'] = self.h5_dataset['frame'][idx].astype(np.int64)
        data_dict['scene'] = self.h5_dataset['scene'].asstr()[idx]
        data_dict['video'] = self.h5_dataset['video'].asstr()[idx]
        data_dict['seq'] = f"{data_dict['scene']}_{data_dict['video']}"

    def add_occlusion_objects(self, data_dict: Dict, idx: int):
        data_dict['ego'] = torch.from_numpy(self.h5_dataset['ego'][idx]).view(1, 2)
        data_dict['occluder'] = torch.from_numpy(self.h5_dataset['occluder'][idx])

    def add_scene_map_transform_parameters(self, data_dict: Dict, idx: int):
        data_dict['theta'] = float(self.h5_dataset['theta'][idx])
        data_dict['center_point'] = torch.from_numpy(self.h5_dataset['center_point'][idx])

    def add_trajectory_data(self, data_dict):
        agent_grid = self.agent_grid(ids=data_dict['identities'])
        timestep_grid = self.timestep_grid(ids=data_dict['identities'])
        last_obs_indices = self.last_observed_indices(obs_mask=data_dict['observation_mask'].to(torch.int16))
        pred_mask = self.predict_mask(last_obs_indices=last_obs_indices)

        data_dict['obs_identity_sequence'] = agent_grid[data_dict['observation_mask'], ...]
        data_dict['obs_timestep_sequence'] = timestep_grid[data_dict['observation_mask'], ...]
        data_dict['obs_position_sequence'] = data_dict['trajectories'][data_dict['observation_mask'], ...]
        data_dict['obs_velocity_sequence'] = data_dict['observed_velocities'][data_dict['observation_mask'], ...]

        data_dict['last_obs_positions'] = self.last_observed_positions(
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

        scaling = self.traj_scale * self.coord_conv_table.loc[data_dict['scene'], data_dict['video']]['m/px']
        if torch.any(torch.isnan(data_dict['ego'])):
            dist_transformed_occlusion_map = torch.zeros([self.map_resolution, self.map_resolution])
            probability_map = torch.zeros([self.map_resolution, self.map_resolution])
            nlog_probability_map = torch.zeros([self.map_resolution, self.map_resolution])
        else:
            invert_occlusion_map = ~processed_occl_map
            dist_transformed_occlusion_map = (torch.where(
                invert_occlusion_map,
                torch.from_numpy(-distance_transform_edt(invert_occlusion_map)),
                torch.from_numpy(distance_transform_edt(processed_occl_map))
            ) * scaling).to(torch.float32)

            clipped_map = -torch.clamp(dist_transformed_occlusion_map, min=0.)
            probability_map = torch.nn.functional.softmax(clipped_map.view(-1), dim=0).view(clipped_map.shape)
            nlog_probability_map = -torch.nn.functional.log_softmax(clipped_map.view(-1), dim=0).view(clipped_map.shape)

        data_dict['dist_transformed_occlusion_map'] = dist_transformed_occlusion_map
        data_dict['probability_occlusion_map'] = probability_map
        data_dict['nlog_probability_occlusion_map'] = nlog_probability_map

    def add_scene_map_data(self, data_dict: Dict):
        # loading the scene map
        image_path = os.path.join(self.padded_images_path, f"{data_dict['seq']}_padded_img.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_torch_image(image)

        scene_map = TorchGeometricMap(
            map_image=image, homography=torch.eye(3)
        )
        scene_map.homography_translation(Tensor([self.padding_px, self.padding_px]))
        scene_map.rotate_around_center(theta=data_dict['theta'])
        scene_map.set_homography(torch.eye(3))
        center_point = data_dict['center_point']
        scaling = self.traj_scale * self.coord_conv_table.loc[data_dict['scene'], data_dict['video']]['m/px']
        scene_map.homography_translation(center_point)
        scene_map.homography_scaling(1 / scaling)
        self.crop_scene_map(scene_map=scene_map)

        data_dict['scene_map'] = scene_map.image

    def __len__(self):
        return self.lookup_indices.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        idx_start, idx_end = self.lookup_indices[idx]

        if self.h5_dataset is None:
            self.h5_dataset = h5py.File(self.hdf5_file, 'r')

        data_dict = dict()

        self.add_instance_identifiers(data_dict=data_dict, idx=idx)
        self.add_scene_map_transform_parameters(data_dict=data_dict, idx=idx)
        if self.occlusion_process == 'occlusion_simulation':
            self.add_occlusion_objects(data_dict=data_dict, idx=idx)

        for dset_name in self.lookup_datasets:
            data_dict[dset_name] = torch.from_numpy(self.h5_dataset[dset_name][idx_start:idx_end])
        data_dict['identities'] = data_dict['identities'].to(torch.int64)

        self.add_trajectory_data(data_dict=data_dict)
        if self.occlusion_process == 'occlusion_simulation':
            self.add_occlusion_map_data(data_dict=data_dict, idx=idx)
        # self.add_scene_map_data(data_dict=data_dict)

        data_dict['timesteps'] = self.timesteps
        data_dict['scene_orig'] = torch.zeros([2])
        data_dict['map_homography'] = self.map_homography

        if self.impute:
            assert 'true_trajectories' in data_dict.keys()
            data_dict['imputation_mask'] = data_dict['true_observation_mask'][data_dict['observation_mask']]

        # Quick Fix
        if self.quick_fix:
            data_dict = self.apply_quick_fix(data_dict)

        return data_dict

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

    # def get_instance_idx(self, instance_name: Optional[str] = None, instance_num: Optional[int] = None):
    #     assert instance_name is not None or instance_num is not None
    #
    #     if instance_name is not None and instance_name in self.instance_names:
    #         return self.instance_names.index(instance_name)
    #
    #     if instance_num is not None and instance_num in self.instance_nums:
    #         return self.instance_nums.index(instance_num)
    #
    #     return None


########################################################################################################################


def write_instance_to_hdf5_dataset_from_pickle(
        hdf5_file: h5py.File,
        instance_idx: int,
        setup_dict: Dict,
        instance_dict: Dict
):
    # # creating a separate group for the instance, where all shape changing instance elements will be stored
    # group = hdf5_file.create_group(name=instance_name)
    #
    # for key in instance_dict.keys():
    #     # processing instance elements which do not change shapes into their respective datasets
    #     if key in [
    #         'scene_map', 'occlusion_map', 'dist_transformed_occlusion_map', 'probability_occlusion_map',
    #         'nlog_probability_occlusion_map', 'map_homography'
    #     ]:
    #         hdf5_file[key][instance_idx, ...] = instance_dict[key]
    #     elif key in ['seq', 'frame']:
    #         hdf5_file[key][instance_idx] = instance_dict[key]
    #
    #     # processing the remaining instance elements into separate datasets within the group
    #     elif instance_dict[key] is not None:
    #         group.create_dataset(name=key, data=instance_dict[key])
    #     else:
    #         print(f"{instance_name}: {key} is None! passing this one")

    # print(instance_dict.keys())
    # print(instance_dict['observation_mask'])
    # print(instance_dict['trajectories'])
    # print(zblu)

    n_agents = instance_dict['identities'].shape[0]
    orig_index = hdf5_file['identities'].shape[0]

    hdf5_file['lookup_indices'][instance_idx, ...] = (orig_index, orig_index + n_agents)

    for key, value in setup_dict.items():

        dset = hdf5_file[key]
        data = instance_dict.get(key, None)

        if key == 'occlusion_map':
            bytes_occl_map = data.clone().detach().to(torch.int64)
            bytes_occl_map *= 2 ** torch.arange(7, -1, -1, dtype=torch.int64).repeat(100)
            bytes_occl_map = bytes_occl_map.reshape(800, 100, 8).sum(dim=-1)
            bytes_occl_map = struct.pack('80000B', *bytes_occl_map.flatten().tolist())

            dset[instance_idx] = np.void(bytes_occl_map)

        elif data is not None:
            print(f"Writing to hdf5 dataset: {key}")
            if None in dset.maxshape:
                dset.resize(dset.shape[0] + n_agents, axis=0)
                dset[orig_index:orig_index + n_agents, ...] = data
            else:
                dset[instance_idx, ...] = data
        else:
            print(f"Skipped:                 {key}")

    print()

########################################################################################################################


def load_old_data():
    # Loading old HDF5 dataset

    cfg = Config(CONFIG_STR)
    prepare_seed(RNG_SEED)
    torch_dataset = dataset_dict[DATASET_CLASS](parser=cfg, log=None, split=SPLIT)

    # fig, ax = plt.subplots(n_row, n_col)
    #
    # for i in range(n_row * n_col):
    #
    #     row_i, col_i = i // n_col, i % n_col
    #
    #     out_dict = torch_dataset.__getitem__(i + start_idx)
    #     if 'map_homography' not in out_dict.keys():
    #         out_dict['map_homography'] = torch_dataset.map_homography
    #
    #     # print(out_dict.keys())
    #     # print(out_dict['map_homography'])
    #
    #     visualize(
    #         data_dict=out_dict,
    #         draw_ax=ax[row_i, col_i]
    #     )
    # plt.show()

    out_dict = torch_dataset.__getitem__(0)
    for k, v in out_dict.items():
        out_str = f"{k}, {type(v)}"
        if isinstance(v, Tensor):
            out_str += f" ({v.shape}, {v.dtype})"
        print(out_str)
    print()


def load_new_hdf5():
    # Loading new HDF5 dataset
    print("PART TWO!")

    cfg = Config(CONFIG_STR)
    prepare_seed(RNG_SEED)
    torch_dataset = PresavedDataset(parser=cfg, log=None, split=SPLIT)

    fig, ax = plt.subplots(N_ROW, N_COL)

    for i in range(N_ROW * N_COL):

        row_i, col_i = i // N_COL, i % N_COL

        out_dict = torch_dataset.__getitem__(i + START_IDX)
        if 'map_homography' not in out_dict.keys():
            out_dict['map_homography'] = torch_dataset.map_homography

        # print(out_dict.keys())
        # print(out_dict['map_homography'])

        visualize(
            data_dict=out_dict,
            draw_ax=ax[row_i, col_i]
        )
    plt.show()


def compare_old_and_new():
    # Comparing old and new data

    def compare_data_dicts(dict_1, dict_2, verbose=False):

        dont_compare = [
            # 'scene_map',
            # 'instance_name',
        ]

        # print(dict_1['trajectories'][6])
        # print(dict_2['trajectories'][6])
        # print(dict_1['last_obs_positions'][6])
        # print(dict_2['last_obs_positions'][6])

        for key in sorted(dict_1.keys()):

            if key in dont_compare:
                continue

            obj_1, obj_2 = dict_1[key], dict_2[key]
            assert type(obj_1) == type(obj_2), f"{key}: {type(obj_1)=}, {type(obj_2)=}"

            if isinstance(obj_1, torch.Tensor):
                assert obj_1.shape == obj_2.shape, f"{key}: {obj_1.shape=}, {obj_2.shape=}"
                assert obj_1.dtype == obj_2.dtype, f"{key}: {obj_1.dtype=}, {obj_2.dtype=}"

                if obj_1.isnan().any():
                    assert obj_1.isnan().all() and obj_2.isnan().all(), f"{key}: {obj_1=}\n\n{obj_2=}"
                else:
                    assert torch.all(obj_1 == obj_2), f"{key}: {obj_1=}\n\n{obj_2=}\n\n{obj_1 == obj_2=}"
                print(f"IDENTICAL: {key}, {type(obj_1)}") if verbose else None

            elif isinstance(obj_1, np.number):
                assert obj_1.dtype == obj_2.dtype, f"{key}: {obj_1.dtype=}, {obj_2.dtype=}"
                assert obj_1 == obj_2, f"{key}: {obj_1=}, {obj_2=}"
                print(f"IDENTICAL: {key}, {type(obj_1)}") if verbose else None

            elif isinstance(obj_1, str):
                assert obj_1 == obj_2, f"{key}: {obj_1=}, {obj_2=}"
                print(f"IDENTICAL: {key}, {type(obj_1)}") if verbose else None

            elif isinstance(obj_1, float):
                assert obj_1 == obj_2, f"{key}: {obj_1=}, {obj_2=}"
                print(f"IDENTICAL: {key}, {type(obj_1)}") if verbose else None

            elif obj_1 is None:
                assert obj_2 is None, f"{key}: {obj_1=}, {obj_2=}"
                print(f"IDENTICAL: {key}, {type(obj_1)}") if verbose else None

            else:
                print(f"SKIPPED: {key}, {type(obj_1)}") if verbose else None

    cfg_1 = Config(CONFIG_STR)
    torch_dataset_1 = dataset_dict[DATASET_CLASS](parser=cfg_1, log=None, split=SPLIT)

    cfg_2 = Config(CONFIG_STR)
    torch_dataset_2 = PresavedDataset(parser=cfg_2, log=None, split=SPLIT)
    prepare_seed(RNG_SEED)

    out_dict_1 = torch_dataset_1.__getitem__(0)
    out_dict_2 = torch_dataset_2.__getitem__(0)
    all_keys = list(dict.fromkeys(list(out_dict_1.keys()) + list(out_dict_2.keys())))
    common_keys = [key for key in all_keys if all((key in out_dict_1.keys(), key in out_dict_2.keys()))]
    missing_keys = [key for key in all_keys if key not in common_keys]
    # assert len(missing_keys) == 0
    print(f"OLD KEYS:\n{sorted(list(out_dict_1.keys()))}\n")
    print(f"NEW KEYS:\n{sorted(list(out_dict_2.keys()))}\n")
    print(f"KEYS IN COMMON:\n{sorted(common_keys)}\n")
    print(f"MISSING KEYS:\n{sorted(missing_keys)}\n")
    print("\n\n\n")

    compare_data_dicts(out_dict_1, out_dict_2)

    prepare_seed(RNG_SEED)

    # fig_1, ax_1 = plt.subplots(N_ROW, N_COL)
    # fig_2, ax_2 = plt.subplots(N_ROW, N_COL)

    from time import time
    time_1, time_2 = [], []

    for i in range(N_COMPARISON):

        # row_i, col_i = i // N_COL, i % N_COL

        t1 = time()
        out_dict_1 = torch_dataset_1.__getitem__(i + START_IDX)
        t2 = time()
        out_dict_2 = torch_dataset_2.__getitem__(i + START_IDX)
        t3 = time()

        if 'map_homography' not in out_dict_1.keys():
            out_dict_1['map_homography'] = torch_dataset_1.map_homography
        if 'map_homography' not in out_dict_2.keys():
            out_dict_2['map_homography'] = torch_dataset_2.map_homography

        compare_data_dicts(out_dict_1, out_dict_2)
        print(f"{i}: PASSED TESTS!")

        time_1.append(t2-t1)
        time_2.append(t3-t2)

        # print(out_dict.keys())
        # print(out_dict['map_homography'])

        # visualize(
        #     data_dict=out_dict_1,
        #     draw_ax=ax_1[row_i, col_i]
        # )
        # visualize(
        #     data_dict=out_dict_2,
        #     draw_ax=ax_2[row_i, col_i]
        # )

    print(f"{time_1=}")
    print(f"{time_2=}")
    print(f"{np.mean(time_1)=}")
    print(f"{np.mean(time_2)=}")
    plt.show()


if __name__ == '__main__':
    print("Hello!")

    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_fully_observed', 'pickle', 'train'          # TODO: TRY TO REMAIN CONSISTENT WITH POSITION / VELOCITIES
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_fully_observed', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_fully_observed', 'pickle', 'train'
    CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'train'
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'pickle', 'train'

    N_ROW, N_COL = 4, 6
    N_COMPARISON = 100
    RNG_SEED = 42
    START_IDX = 0

    # load_old_data()
    # load_new_hdf5()
    compare_old_and_new()

    print("Goodbye!")
