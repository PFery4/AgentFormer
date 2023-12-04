import sys
import argparse
import os.path
import pickle
import yaml

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.config import Config
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from data.sdd_dataloader import PresavedDatasetSDD
from model.agentformer_loss import index_mapping_gt_seq_pred_seq
from model.model_lib import model_dict

Tensor = torch.Tensor


# METRICS #############################################################################################################


def compute_samples_ADE(
        pred_positions: Tensor,         # [K, P, 2]
        gt_positions: Tensor,           # [P, 2]
        identities: Tensor,             # [N]
        identity_mask: Tensor,          # [N, P]
) -> Tensor:        # [N, K]

    diff = gt_positions - pred_positions        # [K, P, 2]
    dists = diff.pow(2).sum(-1).sqrt()          # [K, P]

    scores_tensor = torch.zeros([identities.shape[0], pred_positions.shape[0]])     # [N, K]
    for i, (mask, identity) in enumerate(zip(identity_mask, identities)):
        masked_dist = dists[:, mask]                # [K, p]
        ades = torch.mean(masked_dist, dim=-1)      # [K]
        scores_tensor[i, :] = ades

    return scores_tensor        # [N, K]


def compute_samples_FDE(
        pred_positions: Tensor,         # [K, P, 2]
        gt_positions: Tensor,           # [P, 2]
        identities: Tensor,             # [N]
        identity_mask: Tensor,          # [N, P]
) -> Tensor:        # [N, K]

    diff = gt_positions - pred_positions        # [K, P, 2]
    dists = diff.pow(2).sum(-1).sqrt()          # [K, P]

    scores_tensor = torch.zeros([identities.shape[0], pred_positions.shape[0]])     # [N, K]
    for i, (mask, identity) in enumerate(zip(identity_mask, identities)):
        masked_dist = dists[:, mask]            # [K, p]
        fdes = masked_dist[:, -1]               # [K]
        scores_tensor[i, :] = fdes

    return scores_tensor        # [N, K]


def compute_occlusion_area_occupancy():
    # Park et al.'s DAO applied on the occlusion map
    pass


def compute_occlusion_area_count():
    # Park et al.'s DAC applied on the occlusion map
    pass


def compute_rf():
    pass


def compute_mean_score(scores_tensor: Tensor) -> Tensor:
    # [N agents, K modes] -> [N agents]
    return torch.mean(scores_tensor, dim=-1)


def compute_min_score(scores_tensor: Tensor) -> Tensor:
    # [N agents, K modes] -> [N agents]
    return torch.min(scores_tensor, dim=-1)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default=None)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    split = args.data_split
    checkpoint_name = args.checkpoint_name
    cfg = Config(cfg_id=args.cfg, tmp=args.tmp, create_dirs=False)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # device
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print("Torch CUDA is available")
        num_of_devices = torch.cuda.device_count()
        if num_of_devices:
            print(f"Number of CUDA devices: {num_of_devices}")
            current_device = torch.cuda.current_device()
            current_device_id = torch.cuda.device(current_device)
            current_device_name = torch.cuda.get_device_name(current_device)
            print(f"Current device: {current_device}")
            print(f"Current device id: {current_device_id}")
            print(f"Current device name: {current_device_name}")
            print()
            device = torch.device('cuda', index=current_device)
            torch.cuda.set_device(current_device)
        else:
            print("No CUDA devices!")
            sys.exit()
    else:
        print("Torch CUDA is not available!")
        sys.exit()

    # log
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    # dataloader
    if cfg.dataset == 'sdd':
        sdd_test_set = PresavedDatasetSDD(parser=cfg, log=log, split=split)
        test_loader = DataLoader(dataset=sdd_test_set, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    # model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    if checkpoint_name is not None:
        cp_path = cfg.model_path % checkpoint_name
        print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    # saving model predictions
    if checkpoint_name is not None:
        save_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    else:
        save_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, 'untrained', split)
    log_str = f'saving predictions under the following directory:\n{save_dir}\n\n'
    print_log(log_str, log=log)
    mkdir_if_missing(save_dir)

    for i, data in enumerate(test_loader):

        seq_name, frame, filename = data['seq'][0], int(data['frame'][0]), data['filename'][0]
        log_str = f'saving predictions of instance #{i}\t\t| ' \
                  f'file name: {filename}\t\t| ' \
                  f'sequence name: {seq_name}\t\t| ' \
                  f'frame number: {frame}'
        print_log(log_str, log=log)

        out_dict = dict()
        with torch.no_grad():
            model.set_data(data)
            # recon_pred, _ = model.inference(mode='recon', sample_num=1)         # [B, P, 2]   # unused
            samples_pred, model_data = model.inference(
                mode='infer', sample_num=cfg.sample_k, need_weights=False
            )                                                                   # [B * sample_k, P, 2]

        for key, value in model_data.items():
            if key == 'valid_id' or 'last_obs_' in key or 'pred_' in key or '_dec_' in key:
                out_dict[key] = value

        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # preparing the table of outputs
    df_indices = ['idx', 'agent_id']
    mode_ades = [f'K{i}_ADE' for i in range(cfg.sample_k)]
    mode_fdes = [f'K{i}_FDE' for i in range(cfg.sample_k)]
    df_columns = df_indices + mode_ades + mode_fdes
    score_df = pd.DataFrame(columns=df_columns)

    # computing performance metrics from saved predictions
    for i, in_data in enumerate(test_loader):

        filename = in_data['filename'][0]

        with open(os.path.join(save_dir, filename), 'rb') as f:
            pred_data = pickle.load(f)

        valid_ids = pred_data['valid_id'][0]                        # [N]
        gt_identities = pred_data['pred_identity_sequence'][0]      # [P]
        gt_timesteps = pred_data['pred_timestep_sequence'][0]       # [P]
        gt_positions = pred_data['pred_position_sequence'][0]       # [P, 2]

        infer_pred_identities = pred_data['infer_dec_agents'][0]        # [P]
        infer_pred_timesteps = pred_data['infer_dec_timesteps']         # [P]
        infer_pred_positions = pred_data['infer_dec_motion']            # [K, P, 2]
        infer_pred_past_mask = pred_data['infer_dec_past_mask']         # [P]

        idx_map = index_mapping_gt_seq_pred_seq(
            ag_gt=gt_identities,
            tsteps_gt=gt_timesteps,
            ag_pred=infer_pred_identities,
            tsteps_pred=infer_pred_timesteps
        )

        gt_identities = gt_identities[idx_map]      # [P]
        gt_timesteps = gt_timesteps[idx_map]        # [P]
        gt_positions = gt_positions[idx_map, :]     # [P, 2]

        assert torch.all(infer_pred_identities == gt_identities)
        assert torch.all(infer_pred_timesteps == gt_timesteps)

        identity_mask = valid_ids.unsqueeze(1) == infer_pred_identities.unsqueeze(0)        # [N, P]

        # COMPUTE SCORES
        infer_ade_scores = compute_samples_ADE(
            pred_positions=infer_pred_positions,
            gt_positions=gt_positions,
            identities=valid_ids,
            identity_mask=identity_mask
        )           # [N, K]
        infer_fde_scores = compute_samples_FDE(
            pred_positions=infer_pred_positions,
            gt_positions=gt_positions,
            identities=valid_ids,
            identity_mask=identity_mask
        )           # [N, K]

        # APPEND SCORE VALUES TO TABLE
        for i_agent, valid_id in enumerate(valid_ids):
            # i, agent_id, K{i}
            df_row = [i, int(valid_id)]
            df_row.extend(infer_ade_scores[i_agent].tolist())
            df_row.extend(infer_fde_scores[i_agent].tolist())
            assert len(df_row) == len(df_columns)
            score_df.loc[len(score_df)] = df_row

    # postprocessing on the table
    score_df[['idx', 'agent_id']] = score_df[['idx', 'agent_id']].astype(int)
    score_df.set_index(keys=['idx', 'agent_id'], inplace=True)
    score_df['min_ADE'] = score_df[mode_ades].min(axis=1)
    score_df['mean_ADE'] = score_df[mode_ades].mean(axis=1)
    score_df['min_FDE'] = score_df[mode_fdes].min(axis=1)
    score_df['mean_FDE'] = score_df[mode_fdes].mean(axis=1)
    score_df['rF'] = score_df['mean_FDE'] / score_df['min_FDE']
    score_df['rF'] = score_df['rF'].fillna(value=1.0)

    # saving the table, and the score summary
    df_save_name = os.path.join(save_dir, 'prediction_scores.csv')
    print(f"saving prediction scores table under:\n{df_save_name}")
    score_df.to_csv(df_save_name, sep=',', encoding='utf-8')

    score_dict = {x: float(score_df[x].mean()) for x in score_df.columns}
    yml_save_name = os.path.join(save_dir, 'prediction_scores.yml')
    print(f"saving prediction scores summary under:\n{yml_save_name}")
    with open(yml_save_name, 'w') as f:
        yaml.dump(score_dict, f)

    print(score_df)
    [print(f"{key}:\t\t{val}") for key, val in score_dict.items()]
