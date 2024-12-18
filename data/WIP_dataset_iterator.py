import argparse
import time
import psutil

import torch

from data.sdd_dataloader import TorchDataGeneratorSDD, HDF5PresavedDatasetSDD
from utils.config import Config, REPO_ROOT
from utils.utils import prepare_seed

Tensor = torch.Tensor


########################################################################################################################


# TODO: FIGURE OUT WHETHER TO KEEP THIS FILE FOR SOME UTILITY OR NOT (do we care about just iterating over a dataset?)


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_fully_observed', 'pickle', 'train'          # TODO: TRY TO REMAIN CONSISTENT WITH POSITION / VELOCITIES
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_fully_observed', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_fully_observed', 'pickle', 'train'
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'test'
    # CONFIG_STR, DATASET_CLASS, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'train'
    # config_str, dataset_class, split = 'dataset_occlusion_simulation', 'pickle', 'train'

    # CONFIG_STR, _, SPLIT = 'dataset_fully_observed', 'torch_preprocess', 'train'
    # CONFIG_STR, _, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'train'
    # CONFIG_STR, _, SPLIT = 'dataset_occlusion_simulation_imputed', 'torch_preprocess', 'train'
    # CONFIG_STR, _, SPLIT = 'dataset_fully_observed', 'torch_preprocess', 'val'
    # CONFIG_STR, _, SPLIT = 'dataset_occlusion_simulation', 'torch_preprocess', 'val'
    # CONFIG_STR, _, SPLIT = 'dataset_occlusion_simulation_imputed', 'torch_preprocess', 'val'
    # CONFIG_STR, _, SPLIT = 'dataset_fully_observed_no_rand_rot', 'torch_preprocess', 'test'
    # CONFIG_STR, _, SPLIT = 'dataset_occlusion_simulation_no_rand_rot', 'torch_preprocess', 'test'
    # CONFIG_STR, _, SPLIT = 'dataset_occlusion_simulation_imputed_no_rand_rot', 'torch_preprocess', 'test'

    CONFIG_STR, SPLIT = args.cfg, args.split

    RNG_SEED = 42
    START_IDX = 0

    cfg = Config(CONFIG_STR)
    cfg.__setattr__('with_rgb_map', False)
    # torch_dataset = PresavedDataset(parser=cfg, split=SPLIT)
    # torch_dataset = TorchDataGeneratorSDD(parser=cfg, split=SPLIT)
    torch_dataset = HDF5PresavedDatasetSDD(parser=cfg, split=SPLIT)

    print(f"{len(torch_dataset)=}")

    # N_COMPARISON = 50
    N_ITERATIONS = len(torch_dataset)

    prepare_seed(RNG_SEED)

    for i in range(N_ITERATIONS):

        time_bf = time.time()
        data = torch_dataset.__getitem__(idx=i)
        time_af = time.time()
        process = psutil.Process()

        print(f"DATA DICT KEYS: {i}")
        print(f"\t\t(loading time: {time_af-time_bf} s)")
        print(f"\t\t(process memory: {process.memory_info().rss / (1000 ** 2)} MB)")
        # [print(k) for k in data.keys()]
        print("\n\n")

    print("Goodbye!")
