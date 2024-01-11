import argparse
import os.path
import pickle
import numpy as np
import h5py
from tqdm import tqdm
from typing import Dict
from data.sdd_dataloader import TorchDataGeneratorSDD
from utils.config import Config, REPO_ROOT
from utils.utils import prepare_seed

"""

Due to the fact that the preprocessing pipeline of the SDD dataset is quite demanding,
we had no choice but to save the preprocessed data.

"""


def instantiate_hdf5_dataset(save_path, setup_dict: Dict):
    assert os.path.exists(save_path)

    print(f"\nthe setup dictionary for the hdf5 dataset is:")
    [print(f"{k}: {v}") for k, v in setup_dict.items()]
    print()

    with h5py.File(os.path.join(save_path, 'dataset.h5'), 'w') as hdf5_file:

        # creating separate datasets for instance elements which do not change shapes
        for k, v in setup_dict.items():
            hdf5_file.create_dataset(k, shape=v['shape'], dtype=v['dtype'])
        # hdf5_file.create_dataset('scene_map', (len_dataset, 3, 800, 800), dtype='f4')
        # hdf5_file.create_dataset('occlusion_map', (len_dataset, 800, 800), dtype='b1')
        # hdf5_file.create_dataset('dist_transformed_occlusion_map', (len_dataset, 800, 800), dtype='f4')
        # hdf5_file.create_dataset('probability_occlusion_map', (len_dataset, 800, 800), dtype='f4')
        # hdf5_file.create_dataset('nlog_probability_occlusion_map', (len_dataset, 800, 800), dtype='f4')
        # hdf5_file.create_dataset('map_homography', (len_dataset, 3, 3), dtype='f4')
        # hdf5_file.create_dataset('seq', len_dataset, dtype=h5py.string_dtype(encoding='utf-8'))
        # hdf5_file.create_dataset('frame', len_dataset, dtype='i8')


def write_to_hdf5_dataset(hdf5_file: h5py.File, instance_idx: int, instance_name: str, instance_dict: Dict):
    # creating a separate group for the instance, where all shape changing instance elements will be stored
    if instance_name in hdf5_file.keys():
        group = hdf5_file[instance_name]
    else:
        group = hdf5_file.create_group(name=instance_name)

    for key in instance_dict.keys():
        # processing instance elements which do not change shapes into their respective datasets
        if key in [
            'scene_map', 'occlusion_map', 'dist_transformed_occlusion_map', 'probability_occlusion_map',
            'nlog_probability_occlusion_map', 'map_homography'
        ]:
            hdf5_file[key][instance_idx, ...] = instance_dict[key]
        elif key in ['seq', 'frame']:
            hdf5_file[key][instance_idx] = instance_dict[key]

        # processing the remaining instance elements into separate datasets within the group
        elif instance_dict[key] is not None:
            group.create_dataset(name=key, data=instance_dict[key])
        else:
            print(f"{instance_name}: {key} is None! passing this one")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-10)
    parser.add_argument('--tiny_dataset', type=bool, default=False)
    parser.add_argument('--no_rgb_map', type=bool, default=False)
    parser.add_argument('--no_prob_map', type=bool, default=False)
    parser.add_argument('--save_directory', default=None)
    parser.add_argument('--data_format', default='pickle')
    args = parser.parse_args()

    assert args.split in ['train', 'val', 'test']
    assert args.data_format in ['pickle', 'hdf5']
    if args.end_idx > 0:
        assert args.end_idx > args.start_idx

    # dataset_id = args.occlusion_process
    cfg = args.cfg
    split = args.split
    if args.tiny_dataset:
        tiny, start_idx, end_idx = True, None, None
    else:
        tiny, start_idx, end_idx = False, args.start_idx, args.end_idx

    config = Config(cfg_id=cfg)

    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    del_keys = ['timesteps',
                'scene_orig']
    if config.occlusion_process == 'fully_observed':
        print(f"Fully observed dataset: no map data will be saved")
        del_keys.extend(
            ['occlusion_map',
             'dist_transformed_occlusion_map',
             'probability_occlusion_map',
             'nlog_probability_occlusion_map',
             'scene_map',
             'map_homography']
        )
    if args.no_rgb_map:
        del_keys.extend(['scene_map'])
    if args.no_prob_map:
        del_keys.extend(['probability_occlusion_map'])
    if not config.get('impute', False):
        del_keys.extend(
            ['true_trajectories',
             'true_observation_mask']
        )

    # removing eventual duplicates
    del_keys = list(dict.fromkeys(del_keys))

    print(f"Presaving a dataset from the \'{cfg}\' file.")
    print(f"Beginning saving process of {split} split.")

    prepare_seed(config.seed)

    generator = TorchDataGeneratorSDD(parser=config, log=None, split=split)

    save_dir_name = config.occlusion_process
    if config.get('impute', False):
        save_dir_name += '_imputed'
    if tiny:
        save_dir_name += '_tiny'

    if args.save_directory is None:
        save_path = os.path.join(presaved_datasets_dir, save_dir_name, split)
    else:
        save_path = os.path.normpath(args.save_directory)

    print(f"root directory of the dataset will be:\n{save_path}")
    os.makedirs(save_path, exist_ok=True)

    indices = None
    if tiny:
        indices = np.linspace(0, len(generator), num=50, endpoint=False).astype(int)
        print("Creating a tiny version of the dataset: only 50 instances will be saved.")
    else:
        if end_idx < 0:
            end_idx = len(generator)
        indices = range(start_idx, end_idx, 1)
        print(f"Saving Dataset instances between the range [{start_idx}-{end_idx}].")

    if args.data_format == 'pickle':
        for idx in tqdm(indices):

            filename = f'{idx:08}.pickle'
            data_dict = generator.__getitem__(idx)

            for key in del_keys:
                del data_dict[key]

            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.data_format == 'hdf5':

        len_dataset = len(generator) if not tiny else 50
        C = config.get('map_channels', 3)
        H = W = config.get('global_map_resolution', 800)

        hdf5_setup_dict = {
            'scene_map':                        {'shape': (len_dataset, C, H, W), 'dtype': 'f4'},
            'occlusion_map':                    {'shape': (len_dataset, H, W), 'dtype': 'b1'},
            'dist_transformed_occlusion_map':   {'shape': (len_dataset, H, W), 'dtype': 'f4'},
            'probability_occlusion_map':        {'shape': (len_dataset, H, W), 'dtype': 'f4'},
            'nlog_probability_occlusion_map':   {'shape': (len_dataset, H, W), 'dtype': 'f4'},
            'map_homography':                   {'shape': (len_dataset, 3, 3), 'dtype': 'f4'},
            'seq':                              {'shape': len_dataset, 'dtype': h5py.string_dtype(encoding='utf-8')},
            'frame':                            {'shape': len_dataset, 'dtype': 'i8'},
        }
        for key in del_keys:
            if key in hdf5_setup_dict.keys():
                del hdf5_setup_dict[key]

        if not os.path.exists(os.path.join(save_path, 'dataset.h5')):
            print("Dataset file does not exist, creating a new file")
            instantiate_hdf5_dataset(save_path=save_path, setup_dict=hdf5_setup_dict)
        else:
            print(f"Will use the hdf5 dataset file 'dataset.h5' stored under:\n{save_path}")

        for i, idx in enumerate(tqdm(indices)):

            filename = f'{idx:08}'
            data_dict = generator.__getitem__(idx)

            for key in del_keys:
                del data_dict[key]

            with h5py.File(os.path.join(save_path, 'dataset.h5'), 'a') as hdf5_file:
                write_to_hdf5_dataset(
                    hdf5_file=hdf5_file,
                    instance_idx=idx if not tiny else i,
                    instance_name=filename,
                    instance_dict=data_dict
                )

    print(f"Done!")
    print(f"Goodbye !")
