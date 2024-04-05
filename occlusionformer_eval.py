import sys
import argparse
import os.path
import pickle
import yaml
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.config import Config, REPO_ROOT
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from utils.performance_metrics import \
    compute_samples_ADE,\
    compute_samples_FDE,\
    compute_pred_lengths,\
    compute_occlusion_area_occupancy,\
    compute_occlusion_area_count,\
    compute_occlusion_map_area
from data.sdd_dataloader import dataset_dict
from model.agentformer_loss import index_mapping_gt_seq_pred_seq
from model.model_lib import model_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default='best_val')        # can be 'best_val' / 'untrained' / <model_id>
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_class', default='hdf5')          # [hdf5, pickle, torch_preprocess]
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
    assert cfg.dataset == 'sdd'
    dataset_class = dataset_dict[args.dataset_class]
    if cfg.dataset == 'sdd':
        sdd_test_set = dataset_class(parser=cfg, log=log, split=split)
        test_loader = DataLoader(dataset=sdd_test_set, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    # model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    if model_id in ['const_velocity', 'oracle']:
        checkpoint_name = 'untrained'
    elif checkpoint_name == 'best_val':
        # TODO: Make sure the loading will work in case we submit a checkpoint_name instead of 'best_val'
        checkpoint_name = cfg.get_best_val_checkpoint_name()
        print(f"Best validation checkpoint name is: {checkpoint_name}")
        cp_path = cfg.model_path % checkpoint_name
        print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    # loading model predictions
    if checkpoint_name == 'untrained':
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, 'untrained', split)
    else:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    assert os.path.exists(saved_preds_dir)
    log_str = f'loading predictions from the following directory:\n{saved_preds_dir}\n\n'
    print_log(log_str, log=log)

    # preparing the table of outputs
    metrics_to_compute = [
        'ADE',
        'FDE',
        'ADE_px',
        'FDE_px',
    ]
    if cfg.occlusion_process == 'occlusion_simulation':
        metrics_to_compute.extend(
            [
                'pred_length',
                'past_pred_length',
                'past_ADE',
                'past_FDE',
                'past_ADE_px',
                'past_FDE_px',
                'all_ADE',
                'all_ADE_px',
                'OAC',
                'OAO',
                'OAC_t0',
                'occlusion_area'
            ]
        )
    # if cfg.get('impute', False):
    #     assert all([metric in metrics_to_compute for metric in ['past_ADE', 'past_FDE']])
    #     metrics_to_compute.extend(['future_ADE', 'future_ADE_px'])

    metric_columns = {
        'ADE': [f'K{i}_ADE' for i in range(cfg.sample_k)],
        'FDE': [f'K{i}_FDE' for i in range(cfg.sample_k)],
        'ADE_px': [f'K{i}_ADE_px' for i in range(cfg.sample_k)],
        'FDE_px': [f'K{i}_FDE_px' for i in range(cfg.sample_k)],
        'pred_length': ['pred_length'],
        'past_pred_length': ['past_pred_length'],
        'past_ADE': [f'K{i}_past_ADE' for i in range(cfg.sample_k)],
        'past_FDE': [f'K{i}_past_FDE' for i in range(cfg.sample_k)],
        'past_ADE_px': [f'K{i}_past_ADE_px' for i in range(cfg.sample_k)],
        'past_FDE_px': [f'K{i}_past_FDE_px' for i in range(cfg.sample_k)],
        'all_ADE': [f'K{i}_all_ADE' for i in range(cfg.sample_k)],
        'all_ADE_px': [f'K{i}_all_ADE_px' for i in range(cfg.sample_k)],
        'OAO': ['OAO'],
        'OAC': ['OAC'],
        'OAC_t0': ['OAC_t0'],
        'occlusion_area': ['occlusion_area']
    }
    coord_conv_table = pd.read_csv(
        # os.path.join(REPO_ROOT, '..', 'occlusion-prediction', 'config', 'coordinates_conversion.txt'),
        # ^ location of the coordinates_conversion.txt file in the occlusion simulation repo. copied into datasets/SDD
        os.path.join(REPO_ROOT, 'datasets', 'SDD', 'coordinates_conversion.txt'),
        sep=';', index_col=('scene', 'video')
    )

    df_indices = ['idx', 'filename', 'agent_id']
    df_columns = df_indices.copy()
    for metric_name in metrics_to_compute:
        df_columns.extend(metric_columns[metric_name])

    score_df = pd.DataFrame(columns=df_columns)

    # computing performance metrics from saved predictions
    for i, in_data in enumerate(pbar := tqdm(test_loader)):

        filename = in_data['instance_name'][0]
        pbar.set_description(f"Processing: {filename}")

        with open(os.path.join(saved_preds_dir, filename), 'rb') as f:
            pred_data = pickle.load(f)

        valid_ids = pred_data['valid_id'][0].to(model.device)                        # [N]
        gt_identities = pred_data['pred_identity_sequence'][0].to(model.device)      # [P]
        gt_timesteps = pred_data['pred_timestep_sequence'][0].to(model.device)       # [P]
        gt_positions = pred_data['pred_position_sequence'][0].to(model.device)       # [P, 2]

        infer_pred_identities = pred_data['infer_dec_agents'][0].to(model.device)        # [P]
        infer_pred_timesteps = pred_data['infer_dec_timesteps'].to(model.device)         # [P]
        infer_pred_positions = pred_data['infer_dec_motion'].to(model.device)            # [K, P, 2]
        infer_pred_past_mask = pred_data['infer_dec_past_mask'].to(model.device)         # [P]

        if cfg.get('impute', False):
            true_gt_pred_mask = ~in_data['true_observation_mask'][0]         # [N, T_total]
            impute_mask = in_data['observation_mask'][0]                # [N, T_total]
            true_last_obs_indices = in_data['true_observation_mask'][0].shape[1] - torch.argmax(torch.flip(
                in_data['true_observation_mask'][0], dims=[1]
            ).to(torch.float32), dim=1) - 1     # [N]
            for i_last_obs, true_last_obs_index in enumerate(true_last_obs_indices):
                impute_mask[i_last_obs, :true_last_obs_index+1] = False
                true_gt_pred_mask[i_last_obs, :true_last_obs_index+1] = False

            # retrieving ground truth
            true_gt_positions = in_data['true_trajectories'][0][true_gt_pred_mask].to(gt_positions.device)      # [P, 2]
            identities_grid = torch.hstack([valid_ids.unsqueeze(1)] * in_data['timesteps'][0].shape[0])      # [N, T_total]
            true_gt_identities = identities_grid[true_gt_pred_mask].to(gt_identities.device)                  # [P]
            timesteps_grid = torch.vstack([in_data['timesteps'][0]] * valid_ids.shape[0])                    # [N, T_total]
            true_gt_timesteps = timesteps_grid[true_gt_pred_mask].to(gt_timesteps.device)                    # [P]

            # prediction, with the imputed part of the prediction appended to it
            true_infer_pred_identities = identities_grid[impute_mask].to(infer_pred_identities.device)
            true_infer_pred_timesteps = timesteps_grid[impute_mask].to(infer_pred_timesteps.device)
            true_infer_pred_positions = in_data['trajectories'][0][impute_mask].repeat(cfg.sample_k, 1, 1).to(infer_pred_positions.device)

            true_infer_pred_identities = torch.cat([true_infer_pred_identities, infer_pred_identities], dim=-1)     # [P]
            true_infer_pred_timesteps = torch.cat([true_infer_pred_timesteps, infer_pred_timesteps], dim=-1)        # [P]
            true_infer_pred_positions = torch.cat([true_infer_pred_positions, infer_pred_positions], dim=-2)        # [K, P, 2]
            true_infer_pred_past_mask = (true_infer_pred_timesteps <= 0).to(infer_pred_past_mask.device)            # [P]

            gt_identities = true_gt_identities
            gt_timesteps = true_gt_timesteps
            gt_positions = true_gt_positions

            infer_pred_identities = true_infer_pred_identities
            infer_pred_timesteps = true_infer_pred_timesteps
            infer_pred_positions = true_infer_pred_positions
            infer_pred_past_mask = true_infer_pred_past_mask

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

        future_mask = infer_pred_timesteps > 0                                              # [P]
        identity_and_future_mask = torch.logical_and(identity_mask, future_mask)            # [N, P]

        if {'past_pred_length', 'past_ADE', 'past_FDE', 'OAO', 'OAC'}.intersection(metrics_to_compute):
            past_mask = infer_pred_timesteps <= 0                                               # [P]
            identity_and_past_mask = torch.logical_and(identity_mask, past_mask)                # [N, P]

        if {'OAO', 'OAC', 'OAC_t0', 'occlusion_area'}.intersection(metrics_to_compute):
            occlusion_map = in_data['dist_transformed_occlusion_map'][0].to(model.device)
            map_homography = in_data['map_homography'][0].to(model.device)

            map_infer_pred_positions = torch.cat(
                [infer_pred_positions,
                 torch.ones((*infer_pred_positions.shape[:-1], 1), device=model.device)], dim=-1
            ).transpose(-1, -2)
            map_infer_pred_positions = (map_homography @ map_infer_pred_positions).transpose(-1, -2)[..., :-1]

            # import matplotlib.pyplot as plt
            #
            # fig, ax = plt.subplots()
            # ax.imshow(occlusion_map)
            #
            # map_infer_pred_positions = map_infer_pred_positions.cpu()
            # past_pos = in_data['obs_position_sequence'].to(model.device)
            # past_pos = torch.cat([past_pos, torch.ones((*past_pos.shape[:-1], 1), device=model.device)], dim=-1).transpose(-1, -2)
            # past_pos = (map_homography @ past_pos).transpose(-1, -2)[..., :-1]
            # past_pos = past_pos.cpu()
            #
            # ax.plot(past_pos[..., 0], past_pos[..., 1], c='r')
            # ax.scatter(past_pos[..., 0], past_pos[..., 1], c='r', marker='x', alpha=0.5)
            # ax.plot(map_infer_pred_positions[..., 0], map_infer_pred_positions[..., 1], c='b')
            # ax.scatter(map_infer_pred_positions[..., 0], map_infer_pred_positions[..., 1], c='b', marker='x', alpha=0.5)
            #
            # plt.show()

        computed_metrics = {metric_name: None for metric_name in metrics_to_compute}

        for metric_name in metrics_to_compute:
            if metric_name == 'ADE':
                computed_metrics['ADE'] = compute_samples_ADE(
                    pred_positions=infer_pred_positions,
                    gt_positions=gt_positions,
                    identity_mask=identity_and_future_mask
                )       # [N, K]
                if 'ADE_px' in metrics_to_compute:
                    scene, video = in_data['seq'][0].split('_')
                    px_by_m = coord_conv_table.loc[scene, video]['px/m']
                    computed_metrics['ADE_px'] = computed_metrics['ADE'] * px_by_m

            if metric_name == 'FDE':
                computed_metrics['FDE'] = compute_samples_FDE(
                    pred_positions=infer_pred_positions,
                    gt_positions=gt_positions,
                    identity_mask=identity_and_future_mask
                )       # [N, K]
                if 'FDE_px' in metrics_to_compute:
                    scene, video = in_data['seq'][0].split('_')
                    px_by_m = coord_conv_table.loc[scene, video]['px/m']
                    computed_metrics['FDE_px'] = computed_metrics['FDE'] * px_by_m

            if metric_name == 'pred_length':
                computed_metrics['pred_length'] = compute_pred_lengths(
                    identity_mask=identity_mask
                )       # [N, 1]

            if metric_name == 'past_pred_length':
                computed_metrics['past_pred_length'] = compute_pred_lengths(
                    identity_mask=identity_and_past_mask
                )       # [N, 1]

            if metric_name == 'past_ADE':
                computed_metrics['past_ADE'] = compute_samples_ADE(
                    pred_positions=infer_pred_positions,
                    gt_positions=gt_positions,
                    identity_mask=identity_and_past_mask
                )       # [N, K]
                if 'past_ADE_px' in metrics_to_compute:
                    scene, video = in_data['seq'][0].split('_')
                    px_by_m = coord_conv_table.loc[scene, video]['px/m']
                    computed_metrics['past_ADE_px'] = computed_metrics['past_ADE'] * px_by_m

            if metric_name == 'past_FDE':
                computed_metrics['past_FDE'] = compute_samples_FDE(
                    pred_positions=infer_pred_positions,
                    gt_positions=gt_positions,
                    identity_mask=identity_and_past_mask
                )       # [N, K]
                if 'past_FDE_px' in metrics_to_compute:
                    scene, video = in_data['seq'][0].split('_')
                    px_by_m = coord_conv_table.loc[scene, video]['px/m']
                    computed_metrics['past_FDE_px'] = computed_metrics['past_FDE'] * px_by_m

            if metric_name == 'OAO':
                computed_metrics['OAO'] = compute_occlusion_area_occupancy(
                    pred_positions=map_infer_pred_positions,
                    occlusion_map=occlusion_map,
                    identity_mask=identity_and_past_mask
                )        # [N, 1]

            if metric_name == 'OAC':
                computed_metrics['OAC'] = compute_occlusion_area_count(
                    pred_positions=map_infer_pred_positions,
                    occlusion_map=occlusion_map,
                    identity_mask=identity_and_past_mask
                )        # [N, 1]

            if metric_name == 'OAC_t0':
                computed_metrics['OAC_t0'] = compute_occlusion_area_count(
                    pred_positions=map_infer_pred_positions,
                    occlusion_map=occlusion_map,
                    identity_mask=torch.logical_and(identity_mask, infer_pred_timesteps == 0)
                )        # [N, 1]

            if metric_name == 'all_ADE':
                computed_metrics['all_ADE'] = compute_samples_ADE(
                    pred_positions=infer_pred_positions,
                    gt_positions=gt_positions,
                    identity_mask=identity_mask
                )       # [N, K]
                if 'all_ADE_px' in metrics_to_compute:
                    scene, video = in_data['seq'][0].split('_')
                    px_by_m = coord_conv_table.loc[scene, video]['px/m']
                    computed_metrics['all_ADE_px'] = computed_metrics['all_ADE'] * px_by_m

            if metric_name == 'occlusion_area':
                occl_map_area = compute_occlusion_map_area(
                    occlusion_map=occlusion_map,
                    homography_matrix=map_homography
                )   # float
                computed_metrics['occlusion_area'] = torch.ones([valid_ids.shape[0], 1]) * occl_map_area    # [N, 1]

        assert all([val is not None for val in computed_metrics.values()])

        # APPEND SCORE VALUES TO TABLE
        for i_agent, valid_id in enumerate(valid_ids):
            # i, agent_id, K{i}
            df_row = [i, filename, int(valid_id)]

            for metric_name in metrics_to_compute:
                df_row.extend(computed_metrics[metric_name][i_agent].tolist())

            assert len(df_row) == len(df_columns)
            score_df.loc[len(score_df)] = df_row

    # postprocessing on the table
    score_df[['idx', 'agent_id']] = score_df[['idx', 'agent_id']].astype(int)
    score_df.set_index(keys=df_indices, inplace=True)

    for metric_name in [x for x in [
        'ADE', 'ADE_px', 'FDE', 'FDE_px',
        'past_ADE', 'past_ADE_px', 'past_FDE', 'past_FDE_px',
        'all_ADE', 'all_ADE_px'] if x in metrics_to_compute
    ]:
        mode_ades = metric_columns[metric_name]
        score_df[f'min_{metric_name}'] = score_df[mode_ades].min(axis=1)
        score_df[f'mean_{metric_name}'] = score_df[mode_ades].mean(axis=1)

    # saving the table, and the score summary
    df_save_name = os.path.join(saved_preds_dir, 'prediction_scores.csv')
    print(f"saving prediction scores table under:\n{df_save_name}")
    score_df.to_csv(df_save_name, sep=',', encoding='utf-8')

    score_dict = {x: float(score_df[x].mean()) for x in score_df.columns}
    yml_save_name = os.path.join(saved_preds_dir, 'prediction_scores.yml')
    print(f"saving prediction scores summary under:\n{yml_save_name}")
    with open(yml_save_name, 'w') as f:
        yaml.dump(score_dict, f)

    print("\n\n")
    print(score_df)
    print()
    [print(f"{key}{' '.ljust(20-len(key))}| {val}") for key, val in score_dict.items()]
