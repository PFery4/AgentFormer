import h5py
import matplotlib.pyplot as plt
import numpy as np
import os.path
import struct
import torch
from tqdm import tqdm

from data.sdd_dataloader import dataset_dict
from utils.config import Config
from utils.sdd_visualize import visualize
from utils.utils import prepare_seed

from typing import Dict
Tensor = torch.Tensor


########################################################################################################################

def instantiate_hdf5_dataset(save_path, setup_dict: Dict):
    assert os.path.exists(os.path.dirname(save_path))

    print(f"\nthe setup dictionary for the hdf5 dataset is:")
    [print(f"{k}: {v}") for k, v in setup_dict.items()]
    print()

    with h5py.File(save_path, 'w') as hdf5_file:

        # creating separate datasets for instance elements which do not change shapes
        for k, v in setup_dict.items():
            hdf5_file.create_dataset(k, **v)


def write_instance_to_hdf5_dataset(
        hdf5_file: h5py.File,
        instance_idx: int,
        setup_dict: Dict,
        instance_dict: Dict
):

    n_agents = instance_dict['identities'].shape[0]
    orig_index = hdf5_file['identities'].shape[0]

    hdf5_file['lookup_indices'][instance_idx, ...] = (orig_index, orig_index + n_agents)

    print(instance_dict['true_observation_mask'])
    print(zbluy)

    for key, value in setup_dict.items():

        dset = hdf5_file[key]
        data = instance_dict[key]

        print(f"Writing to hdf5 dataset: {key}")

        if key == 'occlusion_map':
            bytes_occl_map = data.clone().detach().to(torch.int64)
            bytes_occl_map *= 2 ** torch.arange(7, -1, -1, dtype=torch.int64).repeat(100)   # TODO: FIX
            bytes_occl_map = bytes_occl_map.reshape(800, 100, 8).sum(dim=-1)                # TODO: FIX
            bytes_occl_map = struct.pack('80000B', *bytes_occl_map.flatten().tolist())      # TODO: FIX

            dset[instance_idx] = np.void(bytes_occl_map)

        elif data is not None:
            if None in dset.maxshape:
                dset.resize(dset.shape[0] + n_agents, axis=0)
                dset[orig_index:orig_index + n_agents, ...] = data
            else:
                dset[instance_idx, ...] = data
        else:
            print(f"Skipped:                 {key}")

    print()


def save_new_hdf5():
    ###################################################################################################################
    # Saving new hdf5 dataset
    save_start_idx = 0
    save_split = SPLIT
    save_end_idx = N_COMPARISON
    save_temp_len = save_end_idx - save_start_idx

    save_config = Config(cfg_id=CONFIG_STR)

    save_temp_dir = os.path.abspath(os.path.dirname(__file__))
    save_temp_name = 'test_dataset'
    save_temp_path = os.path.join(save_temp_dir, f"{save_temp_name}.h5")

    print(f"Presaving a dataset from the \'{save_config}\' file.")
    print(f"Beginning saving process of {SPLIT} split.")

    prepare_seed(RNG_SEED)

    generator = dataset_dict[DATASET_CLASS](parser=save_config, log=None, split=save_split)

    indices = range(save_start_idx, save_end_idx, 1)
    print(f"Saving Dataset instances between the range [{save_start_idx}-{save_end_idx}].")

    T_TOTAL = generator.T_total
    MAP_RESOLUTION = generator.map_resolution
    assert MAP_RESOLUTION % 8 == 0

    HDF5_SETUP_DICT = {
        # key, value <--> dataset name, dataset metadata dict
        'frame': {'shape': save_temp_len, 'dtype': 'i2'},
        'scene': {'shape': save_temp_len, 'dtype': h5py.string_dtype(encoding='utf-8')},
        'video': {'shape': save_temp_len, 'dtype': h5py.string_dtype(encoding='utf-8')},
        'theta': {'shape': save_temp_len, 'dtype': 'f8'},
        'center_point': {'shape': (save_temp_len, 2), 'dtype': 'f4'},
        'ego': {'shape': (save_temp_len, 2), 'dtype': 'f4'},
        'occluder': {'shape': (save_temp_len, 2, 2), 'dtype': 'f4'},
        'occlusion_map': {'shape': save_temp_len, 'dtype': f'V{int(MAP_RESOLUTION*MAP_RESOLUTION/8)}'},
        'lookup_indices': {'shape': (save_temp_len, 2), 'dtype': 'i4'},
        'identities': {'shape': (0,), 'maxshape': (None,), 'chunks': (1,), 'dtype': 'i2'},
        'trajectories': {'shape': (0, T_TOTAL, 2), 'maxshape': (None, T_TOTAL, 2), 'chunks': (1, T_TOTAL, 2), 'dtype': 'f4'},
        'observation_mask': {'shape': (0, T_TOTAL), 'maxshape': (None, T_TOTAL), 'chunks': (1, T_TOTAL), 'dtype': '?'},
        'observed_velocities': {'shape': (0, T_TOTAL, 2), 'maxshape': (None, T_TOTAL, 2), 'chunks': (1, T_TOTAL, 2), 'dtype': 'f4'},
        'velocities': {'shape': (0, T_TOTAL, 2), 'maxshape': (None, T_TOTAL, 2), 'chunks': (1, T_TOTAL, 2), 'dtype': 'f4'},
        'true_observation_mask': {'shape': (0, T_TOTAL), 'maxshape': (None, T_TOTAL), 'chunks': (1, T_TOTAL), 'dtype': '?'},
        'true_trajectories': {'shape': (0, T_TOTAL, 2), 'maxshape': (None, T_TOTAL, 2), 'chunks': (1, T_TOTAL, 2), 'dtype': 'f4'},
    }

    setup_keys = [
        # key, value <--> dataset name, dataset metadata dict
        'frame',
        'video',
        'theta',
        'center_point',
        'occluder',
        'occlusion_map',
        'lookup_indices',
        'trajectories',
        'observation_mask',
        'observed_velocities',
        'velocities',
        'true_observation_mask',
        'true_trajectories',
    ]
    hdf5_setup_dict = {key: HDF5_SETUP_DICT[key] for key in setup_keys}

    instantiate_hdf5_dataset(
        save_path=save_temp_path,
        setup_dict=hdf5_setup_dict
    )

    for i, idx in enumerate(tqdm(indices)):

        data_dict = generator.__getitem__(idx)

        with h5py.File(save_temp_path, 'a') as hdf5_file:
            write_instance_to_hdf5_dataset(
                hdf5_file=hdf5_file,
                instance_idx=idx,
                # process_keys=list(hdf5_setup_dict.keys()),
                setup_dict=hdf5_setup_dict,
                instance_dict=data_dict
            )

    print(f"Done!")


if __name__ == '__main__':

    print("Hello!")

    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_fully_observed', 'pickle', 'train'          # TODO: TRY TO REMAIN CONSISTENT WITH POSITION / VELOCITIES
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_fully_observed', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_fully_observed', 'pickle', 'train'
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'train'
    CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_occlusion_simulation_imputed', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'pickle', 'train'

    N_COMPARISON = 100
    RNG_SEED = 42
    START_IDX = 0

    save_new_hdf5()

    print("Goodbye!")
