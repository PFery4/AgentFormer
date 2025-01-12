"""
This file contains deprecated class implementations from sdd_dataloader.py
"""


import glob
import h5py
import numpy as np
import os.path
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset

from utils.config import REPO_ROOT, Config

from typing import Dict, Optional


class PresavedDatasetSDD(Dataset):
    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    # This is a "quick fix".
    # We performed the wrong px/m coordinate conversion when computing the distance transformed map.
    # Ideally we should correct this in the TorchDatasetGenerator class.
    # the real fix is to have the distance transformed map scaled by the proper factor:
    # TorchDatasetGenerator.map_side / TorchDatasetGenerator.map_resolution
    coord_conv_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'coordinates_conversion.txt')
    coord_conv_table = pd.read_csv(coord_conv_dir, sep=';', index_col=('scene', 'video'))

    def __init__(self, parser: Config, split: str = 'train'):

        self.quick_fix = parser.get('quick_fix', False)   # again, the quick fix should be result upstream, in the TorchDatasetGenerator class.

        self.split = split

        assert split in ['train', 'val', 'test']
        assert parser.dataset == 'sdd', f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"
        assert parser.occlusion_process in ['fully_observed', 'occlusion_simulation']

        print("\n-------------------------- loading %s data --------------------------" % split)

        # occlusion process specific parameters
        self.occlusion_process = parser.get('occlusion_process', 'fully_observed')
        self.impute = parser.get('impute', False)
        assert not self.impute or self.occlusion_process != 'fully_observed'
        self.momentary = parser.get('momentary', False)
        assert not (self.momentary and self.occlusion_process != 'fully_observed')

        # map specific parameters
        self.map_side = parser.get('scene_side_length', 80.0)               # [m]
        self.map_resolution = parser.get('global_map_resolution', 800)      # [px]
        self.map_homography = torch.Tensor(
            [[self.map_resolution / self.map_side, 0., self.map_resolution / 2],
             [0., self.map_resolution / self.map_side, self.map_resolution / 2],
             [0., 0., 1.]]
        )

        # timesteps specific parameters
        self.T_obs = parser.past_frames
        self.T_pred = parser.future_frames
        self.T_total = self.T_obs + self.T_pred
        self.timesteps = torch.arange(-self.T_obs, self.T_pred) + 1

        # dataset identification
        self.dataset_name = None
        self.dataset_dir = None
        self.set_dataset_name_and_dir()
        assert os.path.exists(self.dataset_dir)
        print(f"Dataset directory is:\n{self.dataset_dir}")

    def set_dataset_name_and_dir(self):
        try_name = self.occlusion_process
        if self.impute:
            try_name += '_imputed'
        if self.momentary:
            try_name += f'_momentary_{self.T_obs}'
        try_dir = os.path.join(self.presaved_datasets_dir, try_name, self.split)
        if os.path.exists(try_dir):
            self.dataset_name = try_name
            self.dataset_dir = try_dir
        else:
            print("Couldn't find full dataset path, trying with tiny dataset instead...")
            try_name += '_tiny'
            self.dataset_name = try_name
            self.dataset_dir = os.path.join(self.presaved_datasets_dir, self.dataset_name, self.split)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError

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


class PickleDatasetSDD(PresavedDatasetSDD):

    def __init__(self, parser: Config, split: str = 'train'):
        super().__init__(parser=parser, split=split)
        print("PICKLE PICKLE PICKLE DATASET DATASET DATASET")

        self.pickle_files = sorted(glob.glob1(self.dataset_dir, "*.pickle"))

        if parser.get('difficult', False):
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
            difficult_instances = [f'{instance:08}.pickle' for instance in difficult_instances]

            difficult_mask = np.in1d(self.pickle_files, difficult_instances)

            difficult_pickle_files = list(np.array(self.pickle_files)[difficult_mask])
            self.pickle_files = difficult_pickle_files

        elif self.split == 'val' and parser.get('validation_set_size', None) is not None:
            assert len(self.pickle_files) > parser.validation_set_size
            # Training set might be quite large. When that is the case, we prefer to validate after every
            # <parser.validation_freq> batches rather than every epoch (i.e., validating multiple times per epoch).
            # train / val split size ratios are typically ~80/20. Our desired validation set size should then be:
            #       <parser.validation_freq> * 20/80
            # We let the user choose the validation set size.
            # The dataset will then be effectively reduced to <parser.validation_set_size>
            required_val_set_size = parser.validation_set_size
            keep_instances = np.linspace(0, len(self.pickle_files)-1, num=required_val_set_size).round().astype(int)

            assert np.all(keep_instances[1:] != keep_instances[:-1])        # verifying no duplicates

            keep_pickle_files = [self.pickle_files[i] for i in keep_instances]

            print(
                f"Val set size too large! --> {len(self.pickle_files)} "
                f"(validating after every {parser.validation_freq} batch).\n"
                f"Reducing val set size to {len(keep_pickle_files)}."
            )
            self.pickle_files = keep_pickle_files
            assert len(self.pickle_files) == required_val_set_size

            print(f"{self.pickle_files=}")

        assert self.__len__() != 0
        print(f'total number of samples: {self.__len__()}')
        print(f'------------------------------ done --------------------------------\n')

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx: int) -> Dict:
        filename = self.pickle_files[idx]

        with open(os.path.join(self.dataset_dir, filename), 'rb') as f:
            data_dict = pickle.load(f)

        data_dict['instance_name'] = filename
        data_dict['timesteps'] = self.timesteps
        data_dict['scene_orig'] = torch.zeros([2])

        if self.impute:
            if data_dict['true_trajectories'] is None:
                data_dict['true_trajectories'] = data_dict['trajectories']
            if data_dict['true_observation_mask'] is None:
                data_dict['true_observation_mask'] = data_dict['observation_mask']
            data_dict['imputation_mask'] = data_dict['true_observation_mask'][data_dict['observation_mask']]

        # Quick Fix
        if self.quick_fix and self.occlusion_process == 'occlusion_simulation':
            data_dict = self.apply_quick_fix(data_dict)

        return data_dict


class HDF5DatasetSDD(PresavedDatasetSDD):

    def __init__(self, parser: Config, split: str = 'train'):
        super().__init__(parser=parser, split=split)

        self.hdf5_file = os.path.join(self.dataset_dir, 'dataset.h5')
        assert os.path.exists(self.hdf5_file)

        # For integrating the hdf5 dataset into the Pytorch class,
        # we follow the principles recommended by Piotr Januszewski:
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.h5_dataset = None
        with h5py.File(self.hdf5_file, 'r') as h5_file:
            instances = sorted([int(key) for key in h5_file.keys() if key.isdecimal()])
            self.instance_names = [f'{instance:08}' for instance in instances]
            self.instance_nums = list(range(len(self.instance_names)))

            self.separate_dataset_keys = [key for key in h5_file.keys() if not key.isdecimal() and key not in ['seq', 'frame']]

        if parser.get('difficult', False):
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

        elif self.split == 'val' and parser.get('validation_set_size', None) is not None:
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

            print(
                f"Val set size too large! --> {len(self.instance_names)} "
                f"(validating after every {parser.validation_freq} batch).\n"
                f"Reducing val set size to {len(keep_instances)}."
            )
            self.instance_names = keep_instance_names
            self.instance_nums = keep_instance_nums
            assert len(self.instance_names) == required_val_set_size

        assert len(self.instance_names) == len(self.instance_nums)

        assert self.__len__() != 0
        print(f'total number of samples: {self.__len__()}')
        print(f'------------------------------ done --------------------------------\n')

    def __len__(self):
        return len(self.instance_names)

    def __getitem__(self, idx: int) -> Dict:
        instance_name = self.instance_names[idx]
        instance_num = self.instance_nums[idx]

        if self.h5_dataset is None:
            self.h5_dataset = h5py.File(self.hdf5_file, 'r')

        data_dict = dict()

        data_dict['seq'] = self.h5_dataset['seq'].asstr()[instance_num]
        data_dict['frame'] = self.h5_dataset['frame'][instance_num]

        # reading instance elements which do not change shapes from their respective datasets
        for key in self.separate_dataset_keys:
            data_dict[key] = torch.from_numpy(self.h5_dataset[key][instance_num])

        # reading remaining instance elements from the corresponding group
        for key in self.h5_dataset[instance_name].keys():
            data_dict[key] = torch.from_numpy(self.h5_dataset[instance_name][key][()])

        data_dict['instance_name'] = instance_name
        data_dict['timesteps'] = self.timesteps
        data_dict['scene_orig'] = torch.zeros([2])

        if self.impute:
            if 'true_trajectories' not in data_dict.keys():
                data_dict['true_trajectories'] = data_dict['trajectories']
            if 'true_observation_mask' not in data_dict.keys():
                data_dict['true_observation_mask'] = data_dict['observation_mask']
            data_dict['imputation_mask'] = data_dict['true_observation_mask'][data_dict['observation_mask']]

        # Quick Fix
        if self.quick_fix and self.occlusion_process == 'occlusion_simulation':
            data_dict = self.apply_quick_fix(data_dict)

        return data_dict

    def get_instance_idx(self, instance_name: Optional[str] = None, instance_num: Optional[int] = None):
        assert instance_name is not None or instance_num is not None

        if instance_name is not None and instance_name in self.instance_names:
            return self.instance_names.index(instance_name)

        if instance_num is not None and instance_num in self.instance_nums:
            return self.instance_nums.index(instance_num)

        return None

