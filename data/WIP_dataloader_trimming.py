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
from tqdm import tqdm

from data.trajectory_operations import last_observed_indices, last_observed_positions
from data.map import \
    compute_distance_transformed_map, compute_probability_map, compute_nlog_probability_map, \
    HomographyMatrix, MapManager, MAP_DICT
from data.sdd_dataloader import dataset_dict, TorchDataGeneratorSDD, HDF5PresavedDatasetSDD
from utils.config import Config, REPO_ROOT
from utils.sdd_visualize import visualize
from utils.utils import prepare_seed, print_log

import src.occlusion_simulation.visibility as visibility        # TODO: MANAGE IMPORTS FROM OTHER REPO IN A BETTER WAY

from typing import Dict, List, Optional
Tensor = torch.Tensor
from io import TextIOWrapper


# TODO: remove this file once done with everything else

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
    torch_dataset = HDF5PresavedDatasetSDD(parser=cfg, log=None, split=SPLIT)

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
    torch_dataset_2 = HDF5PresavedDatasetSDD(parser=cfg_2, log=None, split=SPLIT)
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
