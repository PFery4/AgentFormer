import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from data.sdd_dataloader import dataset_dict
from utils.config import Config, REPO_ROOT


DEFAULT_DATASETS_DIR = os.path.join(REPO_ROOT, 'datasets', 'SDD', 'pre_saved_datasets')
DEFAULT_FILENAME = 'trajectories_info.csv'

torch.set_default_dtype(torch.float32)


def main(args: argparse.Namespace):
    assert args.dataset_class in ['hdf5', 'torch']
    if args.legacy:
        assert args.dataset_class == 'hdf5', "Legacy mode is only available with presaved HDF5 datasets" \
                                             "(use: --dataset_class hdf5)"

    data_cfg = Config(cfg_id=args.cfg)
    data_cfg.__setattr__('with_rgb_map', False)

    dataset_class = dataset_dict[args.dataset_class]
    dataset_kwargs = dict(parser=data_cfg, split=args.split)
    if args.legacy:
        dataset_kwargs.update(legacy_mode=True)
    assert data_cfg.dataset == 'sdd'
    sdd_test_set = dataset_class(**dataset_kwargs)
    test_loader = DataLoader(dataset=sdd_test_set, shuffle=False, num_workers=0)

    if args.save_path is None:
        # assign default save path
        save_dir_name = data_cfg.occlusion_process
        save_dir_name += '_imputed' if data_cfg.get('impute', False) else ''

        args.save_path = os.path.abspath(
            os.path.join(DEFAULT_DATASETS_DIR, save_dir_name, args.split, DEFAULT_FILENAME)
        )

    print(f"saving file under:\n{os.path.dirname(args.save_path)}")

    df_indices = ['idx', 'filename', 'agent_id']
    df_columns = df_indices.copy()
    df_columns += [
        'occlusion_pattern',                    # specified as a byte, increasing timesteps, 0=occluded, 1=observed
        'travelled_distance_Tobs_t0',           # [m] distance travelled over the entirety of the past
        'travelled_distance_Tobs_tlastobs',     # [m] distance travelled [-Tobs; last observed timestep]
        'distance_Tobs_t0',                     # [m] distance between -T_obs and t_0
        'distance_Tobs_tlastobs',               # [m] distance between -T_obs and last observed timestep
    ]
    out_df = pd.DataFrame(columns=df_columns)

    def occl_pattern(obs_mask):
        return (obs_mask.long() * 10**(torch.arange(7, -1, -1))).sum(dim=-1)

    def dist(points_a, points_b):   # [*, 2] -> [*]
        return (points_a - points_b).pow(2).sum(dim=-1).pow(0.5)

    def travl_Tobs_t0(trajectories):        # [*, T, 2] -> [*]
        return (trajectories[..., 1:, :] - trajectories[..., :-1, :]).pow(2).sum(dim=-1).pow(0.5).sum(dim=-1)

    def travl(trajs, obs_mask):
        vel = torch.zeros_like(trajs)  # [N, T, 2]
        for traj, mask, v in zip(trajs, obs_mask, vel):
            obs_indices = torch.nonzero(mask)  # [Z, 1]
            motion_diff = traj[obs_indices[1:, 0], :] - traj[obs_indices[:-1, 0], :]  # [Z - 1, 2]
            v[obs_indices[1:].squeeze(), :] = motion_diff / (obs_indices[1:, :] - obs_indices[:-1, :])  # [Z - 1, 2]
        return vel.pow(2).sum(dim=-1).pow(0.5).sum(-1)

    for i, data in enumerate(pbar := tqdm(test_loader)):

        filename = data['instance_name'][0]
        pbar.set_description(f"Processing: {filename}")

        valid_ids = data['identities'].squeeze(0)

        trajs = data['trajectories'][..., :8, :].squeeze(0)
        obsmask = data['observation_mask'][..., :8].squeeze(0)
        last_pos = data['last_obs_positions'].squeeze(0)

        occlusion_patterns = occl_pattern(obsmask)
        d_Tobs_t0 = dist(trajs[..., 7, :], trajs[..., 0, :])
        t_Tobs_t0 = travl_Tobs_t0(trajs)
        d_Tobs_tlast = dist(last_pos, trajs[..., 0, :])
        t_Tobs_tlast = travl(trajs, obsmask)

        for i_agent, valid_id in enumerate(valid_ids):

            df_row = {
                'idx': i,
                'filename': filename,
                'agent_id': int(valid_id),
                'occlusion_pattern': int(occlusion_patterns[i_agent]),
                'travelled_distance_Tobs_t0': float(t_Tobs_t0[i_agent]),
                'travelled_distance_Tobs_tlastobs': float(t_Tobs_tlast[i_agent]),
                'distance_Tobs_t0': float(d_Tobs_t0[i_agent]),
                'distance_Tobs_tlastobs': float(d_Tobs_tlast[i_agent]),
            }

            assert len(df_row) == len(df_columns)
            out_df.loc[len(out_df)] = df_row

    out_df[['idx', 'agent_id']] = out_df[['idx', 'agent_id']].astype(int)
    out_df.set_index(keys=df_indices, inplace=True)
    print(out_df)

    print(f"saving trajectory measures under:\n{args.save_path}")
    out_df.to_csv(args.save_path, sep=',', encoding='utf-8')


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None,
                        help="Dataset config file (specified as either name or path")
    parser.add_argument('--split', type=str, default='test',
                        help="\'train\' | \'val\' | \'test\'")
    parser.add_argument('--dataset_class', type=str, default='hdf5',
                        help="\'torch\' | \'hdf5\'")
    parser.add_argument('--save_path', type=os.path.abspath, default=None)
    parser.add_argument('--legacy', action='store_true', default=False)
    args = parser.parse_args()

    main(args=args)

    print("Goodbye!")
