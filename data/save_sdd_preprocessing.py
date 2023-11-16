

"""

Due to the fact that the preprocessing pipeline of the SDD dataset is quite demanding,
we had no choice but to save the preprocessed data.

Total dataset sizes:
- occlusion dataset:
- fully observed dataset:

"""

if __name__ == '__main__':
    import os.path
    import pickle
    import numpy as np
    from tqdm import tqdm
    from data.sdd_dataloader import TorchDataGeneratorSDD
    from utils.config import Config, REPO_ROOT
    from utils.utils import prepare_seed

    config_str = 'sdd_baseline_occlusionformer_pre'

    config = Config(config_str)

    splits = ['train', 'val', 'test']
    presaved_datasets_dir = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')

    for split in splits:
        dataset_id = 'occlusion_simulations'
        print(f"Presaving the \'{dataset_id}\' dataset.")
        print(f"Beginning saving process of {split} split.")

        prepare_seed(config.seed)
        generator = TorchDataGeneratorSDD(parser=config, log=None, split=split)

        save_path = os.path.join(presaved_datasets_dir, dataset_id, split)
        print(f"root directory of the dataset will be:\n{save_path}")
        os.makedirs(save_path, exist_ok=True)

        # indices = np.linspace(0, len(generator), 50, endpoint=False).astype(int)
        indices = range(len(generator))

        for idx in tqdm(indices):

            filename = f'{idx:08}.pickle'
            data_dict = generator.__getitem__(idx)

            del data_dict['timesteps']
            del data_dict['scene_orig']

            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Done!")

    config = Config(config_str)

    for split in splits:
        dataset_id = 'fully_observed'
        config.occlusion_process = 'fully_observed'
        print(f"Presaving the \'{dataset_id}\' dataset.")
        print(f"Beginning saving process of {split} split.")

        prepare_seed(config.seed)
        generator = TorchDataGeneratorSDD(parser=config, log=None, split=split)

        save_path = os.path.join(presaved_datasets_dir, dataset_id, split)
        print(f"root directory of the dataset will be:\n{save_path}")
        os.makedirs(save_path, exist_ok=True)

        # indices = np.linspace(0, len(generator), 50, endpoint=False).astype(int)
        indices = range(len(generator))

        for idx in tqdm(indices):

            filename = f'{idx:08}.pickle'
            data_dict = generator.__getitem__(idx)

            del data_dict['timesteps']
            del data_dict['scene_orig']

            del data_dict['occlusion_map']
            del data_dict['dist_transformed_occlusion_map']
            del data_dict['probability_occlusion_map']
            del data_dict['nlog_probability_occlusion_map']
            del data_dict['scene_map']
            del data_dict['map_homography']

            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Done!")

    print(f"Goodbye !")

