

"""

Due to the fact that the preprocessing pipeline of the SDD dataset is quite demanding,
we had no choice but to save the preprocessed data.

Total dataset sizes:
- occlusion dataset:
- fully observed dataset:

"""

if __name__ == '__main__':
    import argparse
    import os.path
    import pickle
    import numpy as np
    from tqdm import tqdm
    from data.sdd_dataloader import TorchDataGeneratorSDD
    from utils.config import Config, REPO_ROOT
    from utils.utils import prepare_seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--occlusion_process', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-10)
    args = parser.parse_args()

    assert args.occlusion_process in ['occlusion_simulation', 'fully_observed']
    assert args.split in ['train', 'val', 'test']
    if args.end_idx > 0:
        assert args.end_idx > args.start_idx

    dataset_id = args.occlusion_process
    split = args.split
    start_idx = args.start_idx
    end_idx = args.end_idx

    config_str = 'sdd_baseline_occlusionformer_pre'
    config = Config(config_str)

    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    del_keys = ['timesteps',
                'scene_orig']
    if args.occlusion_process == 'fully_observed':
        del_keys.extend(
            ['occlusion_map',
             'dist_transformed_occlusion_map',
             'probability_occlusion_map',
             'nlog_probability_occlusion_map',
             'scene_map',
             'map_homography']
        )

    print(f"Presaving the \'{dataset_id}\' dataset.")
    print(f"Beginning saving process of {split} split.")

    prepare_seed(config.seed)

    generator = TorchDataGeneratorSDD(parser=config, log=None, split=split)

    save_path = os.path.join(presaved_datasets_dir, dataset_id, split)
    print(f"root directory of the dataset will be:\n{save_path}")
    os.makedirs(save_path, exist_ok=True)

    if end_idx < 0:
        end_idx = len(generator)

    indices = range(start_idx, end_idx, 1)

    for idx in tqdm(indices):

        filename = f'{idx:08}.pickle'
        data_dict = generator.__getitem__(idx)

        for key in del_keys:
            del data_dict[key]

        with open(os.path.join(save_path, filename), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done!")
    print(f"Goodbye !")
