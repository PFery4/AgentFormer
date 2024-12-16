import argparse
import h5py
import numpy as np
import os.path
import struct
import torch
from tqdm import tqdm

from data.sdd_dataloader import TorchDataGeneratorSDD
from utils.config import Config, REPO_ROOT
from utils.utils import prepare_seed

from typing import Dict, Optional
Tensor = torch.Tensor


DEFAULT_DATASETS_DIR = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')
DEFAULT_FILENAME = 'dataset_v2.h5'


def prepare_dataset_setup_dict(dataset: TorchDataGeneratorSDD, save_size: Optional[int] = None) -> Dict:
    if save_size is None:
        save_size = len(dataset)

    assert dataset.map_resolution % 8 == 0
    t_len, map_res = dataset.T_total, dataset.map_resolution

    # prepare the setup dictionary for instantiation of the HDF5 dataset file
    basic_setup_dict = {
        # key, value <--> dataset name, dataset metadata dict
        'frame': {'shape': save_size, 'chunks': (512,), 'dtype': 'i2'},
        'scene': {'shape': save_size, 'chunks': (64,), 'dtype': h5py.string_dtype(encoding='utf-8')},
        'video': {'shape': save_size, 'chunks': (64,), 'dtype': h5py.string_dtype(encoding='utf-8')},
        'theta': {'shape': save_size, 'chunks': (128,), 'dtype': 'f8'},
        'center_point': {'shape': (save_size, 2), 'chunks': (128, 2), 'dtype': 'f4'},
        'lookup_indices': {'shape': (save_size, 2), 'chunks': (128, 2), 'dtype': 'i4'},
        'identities': {'shape': (0,), 'maxshape': (None,), 'chunks': (512,), 'dtype': 'i2'},
        'trajectories': {'shape': (0, t_len, 2), 'maxshape': (None, t_len, 2), 'chunks': (16, t_len, 2), 'dtype': 'f4'},
        'observation_mask': {'shape': (0, t_len), 'maxshape': (None, t_len), 'chunks': (64, t_len), 'dtype': '?'},
        'observed_velocities': {'shape': (0, t_len, 2), 'maxshape': (None, t_len, 2),
                                'chunks': (16, t_len, 2), 'dtype': 'f4'},
        'velocities': {'shape': (0, t_len, 2), 'maxshape': (None, t_len, 2), 'chunks': (16, t_len, 2), 'dtype': 'f4'},
    }
    occlusion_setup_dict = {
        'is_occluded': {'shape': save_size, 'chunks': (512,), 'dtype': '?'},
        'ego': {'shape': (save_size, 2), 'chunks': (128, 2), 'dtype': 'f4'},
        'occluder': {'shape': (save_size, 2, 2), 'chunks': (64, 2, 2), 'dtype': 'f4'},
        'occlusion_map': {'shape': save_size, 'chunks': (1,), 'dtype': f'V{int(map_res * map_res / 8)}'},
    }
    impute_setup_dict = {
        'true_observation_mask': {'shape': (0, t_len), 'maxshape': (None, t_len), 'chunks': (64, t_len), 'dtype': '?'},
        'true_trajectories': {'shape': (0, t_len, 2), 'maxshape': (None, t_len, 2),
                              'chunks': (16, t_len, 2), 'dtype': 'f4'},
    }

    setup_dict = {**basic_setup_dict}
    if dataset.occlusion_process == 'occlusion_simulation':
        setup_dict = {**basic_setup_dict, **occlusion_setup_dict}
        if dataset.impute:
            setup_dict = {**basic_setup_dict, **occlusion_setup_dict, **impute_setup_dict}
    return setup_dict


def instantiate_hdf5_dataset(save_path, setup_dict: Dict):
    assert os.path.exists(os.path.dirname(save_path))
    assert 'identities' in setup_dict.keys()
    assert 'lookup_indices' in setup_dict.keys()

    with h5py.File(save_path, 'w') as hdf5_file:

        # creating separate datasets for instance elements which do not change shapes
        for k, v in setup_dict.items():
            hdf5_file.create_dataset(k, **v)


def write_instance_to_hdf5_dataset(
        hdf5_file: h5py.File,
        instance_idx: int,
        setup_dict: Dict,
        instance_dict: Dict,
        verbose: bool = False
):

    n_agents = instance_dict['identities'].shape[0]
    orig_index = hdf5_file['identities'].shape[0]

    hdf5_file['lookup_indices'][instance_idx, ...] = (orig_index, orig_index + n_agents)

    for key, value in setup_dict.items():

        if key == 'lookup_indices':
            continue

        dset = hdf5_file[key]
        data = instance_dict[key]

        if verbose:
            description = f"{data.shape, data.dtype}" if isinstance(data, torch.Tensor) else f"{data}"
            print(f"Writing to hdf5 dataset: {key}, {description}")

        if key == 'occlusion_map':
            bytes_occl_map = data.clone().detach().to(torch.int64)
            map_height, map_width = bytes_occl_map.shape
            n_bytes = map_width // 8

            bytes_occl_map *= 2 ** torch.arange(7, -1, -1, dtype=torch.int64).repeat(n_bytes)
            bytes_occl_map = bytes_occl_map.reshape(map_height, n_bytes, 8).sum(dim=-1)
            bytes_occl_map = struct.pack(f'{int(map_height * n_bytes)}B', *bytes_occl_map.flatten().tolist())

            dset[instance_idx] = np.void(bytes_occl_map)

        elif data is not None:
            if None in dset.maxshape:
                dset.resize(dset.shape[0] + n_agents, axis=0)
                dset[orig_index:orig_index + n_agents, ...] = data
            else:
                dset[instance_idx, ...] = data
        else:
            if verbose:
                print(f"Skipped:                 {key}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None,
                        help="config file (specified as either name or path")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-10)
    parser.add_argument('--save_path', type=os.path.abspath, default=None)
    parser.add_argument('--size_setting', type=str, default='generator',
                        help='\'generator\': use the length of the generator to set the presaved dataset file size.\n'
                             '\'indices\': use --start_idx and --end_idx to infer the presaved dataset file size.')
    parser.add_argument('--process_rgb_map', type=bool, default=False,
                        help="Whether to have the <TorchDataGeneratorSDD> class process the RGB scene map."
                             "This significantly increases the program's running time and memory usage."
                             "The saving process does not use the RGB scene map, setting this flag to True will not"
                             "modify the program's output in any way (so it should be kept as False).")
    args = parser.parse_args()

    assert args.split in ['train', 'val', 'test']
    assert args.size_setting in ['generator', 'indices']

    cfg = Config(cfg_id=args.cfg)
    cfg.__setattr__('with_rgb_map', args.process_rgb_map)

    if args.save_path is None:
        # Assign default save path
        save_dir_name = cfg.occlusion_process
        save_dir_name += '_imputed' if cfg.get('impute', False) else ''

        args.save_path = os.path.abspath(
            os.path.join(DEFAULT_DATASETS_DIR, save_dir_name, args.split, DEFAULT_FILENAME)
        )
    assert args.save_path.endswith('.h5')

    print(f"Presaving a dataset from the \'{args.cfg}\' file (\'{args.split}\' split).\n")
    print(f"Dataset will be saved under:\n{args.save_path}\n")

    print(f"Setting RNG seed to: {cfg.seed}\n")
    prepare_seed(cfg.seed)

    print(f"Instantiating Dataset object:\n")
    generator = TorchDataGeneratorSDD(parser=cfg, split=args.split)

    if args.end_idx < 0:
        print(f"Setting end index to the length of the Dataset: {len(generator)}\n")
        args.end_idx = len(generator)
    assert args.end_idx > args.start_idx

    hdf5_setup_dict = prepare_dataset_setup_dict(
        dataset=generator,
        save_size=len(generator) if args.size_setting == 'generator' else args.end_idx - args.start_idx
    )

    print(f"Target HDF5 file has the following characteristics:")
    [print(f"{k}: {v}") for k, v in hdf5_setup_dict.items()]
    print()

    if not os.path.exists(args.save_path):
        print("Dataset file does not exist, creating a new file\n")
        instantiate_hdf5_dataset(save_path=args.save_path, setup_dict=hdf5_setup_dict)
    else:
        print("Dataset file already exists, continuing from there...\n")

    indices = range(args.start_idx, args.end_idx, 1)
    print(f"Saving Dataset instances between the range [{args.start_idx}-{args.end_idx}].")

    with h5py.File(args.save_path, 'a') as hdf5_file:

        for i, idx in enumerate(tqdm(indices)):

            data_dict = generator.__getitem__(idx)

            write_instance_to_hdf5_dataset(
                hdf5_file=hdf5_file,
                instance_idx=idx,
                setup_dict=hdf5_setup_dict,
                instance_dict=data_dict,
                verbose=False
            )

    print("\n\nDone, Goodbye!")
