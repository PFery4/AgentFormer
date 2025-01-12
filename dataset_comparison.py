import os
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.sdd_dataloader import TorchDataGeneratorSDD, HDF5PresavedDatasetSDD
from data.sdd_dataloader_deprecated import PickleDatasetSDD, HDF5DatasetSDD
from utils.config import Config
from utils.utils import prepare_seed

from typing import Dict, List, Optional, Tuple
Tensor = torch.Tensor


DEFAULT_RNG = 42


def print_instance_keys(dataset):
    print(f"KEYS OF {type(dataset)} INSTANCE DICTIONARY:\n")
    out_dict = dataset.__getitem__(0)
    for k, v in out_dict.items():
        out_str = f"{k}, {type(v)}"
        if isinstance(v, Tensor):
            out_str += f" ({v.shape}, {v.dtype})"
        print(out_str)
    print()


def difference_of_data_dicts(
        dict_1: Dict,
        dict_2: Dict,
        keys: List
) -> Dict:

    comparison_results = {key: None for key in keys}

    for key in keys:

        obj_1, obj_2 = dict_1[key], dict_2[key]
        assert type(obj_1) == type(obj_2), f"{key}: {type(obj_1)=}, {type(obj_2)=}"

        if isinstance(obj_1, torch.Tensor):
            assert obj_1.shape == obj_2.shape, f"{key}: {obj_1.shape=}, {obj_2.shape=}"
            assert obj_1.dtype == obj_2.dtype, f"{key}: {obj_1.dtype=}, {obj_2.dtype=}"

            if obj_1.isnan().any():
                diff = 0. if (obj_1.isnan().all() and obj_2.isnan().all()) else 100000.
            elif obj_1.dtype == torch.bool:
                diff = torch.sum(torch.logical_xor(obj_1, obj_2)).item()
            else:
                diff = torch.max(torch.abs(obj_1 - obj_2)).item()

        elif isinstance(obj_1, np.number):
            assert obj_1.dtype == obj_2.dtype, f"{key}: {obj_1.dtype=}, {obj_2.dtype=}"

            diff = np.abs(obj_1 - obj_2)

        elif isinstance(obj_1, str):
            diff = 0. if obj_1 == obj_2 else 1000.

        elif isinstance(obj_1, float):
            diff = abs(obj_1 - obj_2)

        elif isinstance(obj_1, bool):
            diff = int(obj_1 ^ obj_2)

        elif obj_1 is None:
            diff = 0. if (obj_1 is None and obj_2 is None) else 10000.

        else:
            raise NotImplementedError(f"No implementation for handling of: {type(obj_1)}")

        comparison_results[key] = diff

    return comparison_results


def get_common_and_uncommon_instance_keys(
        dataset_1,
        dataset_2
) -> Tuple[List[str], List[str]]:
    # returns the key names that both datasets have in common
    out_dict_1 = dataset_1.__getitem__(0)
    out_dict_2 = dataset_2.__getitem__(0)

    all_keys = list(dict.fromkeys(list(out_dict_1.keys()) + list(out_dict_2.keys())))
    common_keys = [key for key in all_keys if all((key in out_dict_1.keys(), key in out_dict_2.keys()))]
    uncommon_keys = [key for key in all_keys if key not in common_keys]

    return common_keys, uncommon_keys


def compare_2_datasets(
        dataset_1,
        dataset_2,
        compare_keys: List[str],
        indices: Optional[range] = None,       # indices to compare
        start_idx: Optional[int] = 0
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Comparing old and new data
    if indices is None:
        indices = range(len(dataset_1))

    comparison_df = pd.DataFrame(index=indices, columns=compare_keys)

    prepare_seed(DEFAULT_RNG)

    from time import time
    time_1, time_2 = [], []

    for i, idx in enumerate(tqdm(indices)):

        t1 = time()
        out_dict_1 = dataset_1.__getitem__(idx + start_idx)
        t2 = time()
        out_dict_2 = dataset_2.__getitem__(idx + start_idx)
        t3 = time()

        if 'map_homography' not in out_dict_1.keys():
            out_dict_1['map_homography'] = dataset_1.map_homography
        if 'map_homography' not in out_dict_2.keys():
            out_dict_2['map_homography'] = dataset_2.map_homography

        comp_dict = difference_of_data_dicts(out_dict_1, out_dict_2, keys=compare_keys)
        comparison_df.iloc[i] = comp_dict

        time_1.append(t2-t1)
        time_2.append(t3-t2)

    return comparison_df, np.array(time_1), np.array(time_2)


def main(args: argparse.Namespace):
    if args.save_path is not None:
        assert os.path.exists(os.path.dirname(os.path.abspath(args.save_path)))
        assert args.save_path.endswith('.csv')

    cfg = Config(cfg_id=args.cfg)
    cfg.__setattr__('with_rgb_map', False)

    if args.legacy:
        if args.split == 'test':
            dset_class_1 = HDF5DatasetSDD
        else:
            dset_class_1 = PickleDatasetSDD
    else:
        dset_class_1 = TorchDataGeneratorSDD

    dset_class_2 = HDF5PresavedDatasetSDD

    torch_dataset_1 = dset_class_1(parser=cfg, split=args.split)
    torch_dataset_2 = dset_class_2(parser=cfg, split=args.split, legacy_mode=args.legacy)

    print(f"Dataset 1 is of type: {type(torch_dataset_1)}")
    print(f"Dataset 2 is of type: {type(torch_dataset_2)}")

    assert len(torch_dataset_1) == len(torch_dataset_2)
    print(f"Compared Datasets Length: {len(torch_dataset_1)}")

    if args.end_idx < 0:
        print(f"Setting end index to the length of the Dataset: {len(torch_dataset_1)}\n")
        args.end_idx = len(torch_dataset_1)
    assert args.start_idx >= 0
    assert args.end_idx > args.start_idx

    indices = range(args.start_idx, args.end_idx, 1)
    print(f"Comparing Dataset instances between the range [{args.start_idx}-{args.end_idx}].")

    rng_seed = cfg.yml_dict.get('seed', DEFAULT_RNG)
    prepare_seed(rng_seed)
    print(f"RNG seed set to: {rng_seed}\n")

    print_instance_keys(dataset=torch_dataset_1)
    print_instance_keys(dataset=torch_dataset_2)

    common_keys, uncommon_keys = get_common_and_uncommon_instance_keys(
        dataset_1=torch_dataset_1, dataset_2=torch_dataset_2
    )

    print("Instance keys not shared in common between the two datasets:")
    [print(key) for key in uncommon_keys]
    print()

    comp_df, time_1, time_2 = compare_2_datasets(
        dataset_1=torch_dataset_1, dataset_2=torch_dataset_2,
        compare_keys=common_keys, indices=indices, start_idx=0
    )

    print()
    print(comp_df)
    print()
    print(f"{np.mean(time_1), np.mean(time_2)=}")
    print()

    if args.save_path is not None:
        print(f"saving comparison report under:\n{os.path.abspath(args.save_path)}")
        comp_df.to_csv(os.path.abspath(args.save_path))


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None,
                        help="Dataset config file (specified as either name or path")
    parser.add_argument('--split', type=str, default='train',
                        help="\'train\' | \'val\' | \'test\'")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-10)
    parser.add_argument('--legacy', action='store_true', default=False)
    parser.add_argument('--save_path', type=os.path.abspath, default=None,
                        help="path of a \'.csv\' file to save the comparison report")
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
