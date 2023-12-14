

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
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-10)
    parser.add_argument('--tiny_dataset', type=bool, default=False)
    parser.add_argument('--no_rgb_map', type=bool, default=False)
    parser.add_argument('--save_directory', default=None)
    args = parser.parse_args()

    assert args.split in ['train', 'val', 'test']
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
    elif args.no_rgb_map:
        del_keys.extend(['scene_map'])
    if not config.get('impute', False):
        del_keys.extend(
            ['true_trajectories',
             'true_observation_mask']
        )

    print(f"Presaving a dataset from the \'{cfg}\' file.")
    print(f"Beginning saving process of {split} split.")

    prepare_seed(config.seed)

    generator = TorchDataGeneratorSDD(parser=config, log=None, split=split)

    save_dir_name = f"{config.occlusion_process}_tiny" if tiny else config.occlusion_process

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

    for idx in tqdm(indices):

        filename = f'{idx:08}.pickle'
        data_dict = generator.__getitem__(idx)

        for key in del_keys:
            del data_dict[key]

        with open(os.path.join(save_path, filename), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done!")
    print(f"Goodbye !")
