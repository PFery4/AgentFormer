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
from data.sdd_dataloader import dataset_dict
from model.agentformer_loss import index_mapping_gt_seq_pred_seq
from model.model_lib import model_dict

Tensor = torch.Tensor


# METRICS #############################################################################################################


def compute_samples_ADE(
        pred_positions: Tensor,         # [K, P, 2]
        gt_positions: Tensor,           # [P, 2]
        identity_mask: Tensor,          # [N, P]
) -> Tensor:                            # [N, K]

    diff = gt_positions - pred_positions        # [K, P, 2]
    dists = diff.pow(2).sum(-1).sqrt()          # [K, P]

    scores_tensor = torch.zeros([identity_mask.shape[0], pred_positions.shape[0]])     # [N, K]
    for i, mask in enumerate(identity_mask):
        masked_dist = dists[:, mask]                # [K, p]
        ades = torch.mean(masked_dist, dim=-1)      # [K]
        scores_tensor[i, :] = ades

    return scores_tensor        # [N, K]


def compute_samples_FDE(
        pred_positions: Tensor,         # [K, P, 2]
        gt_positions: Tensor,           # [P, 2]
        identity_mask: Tensor,          # [N, P]
) -> Tensor:                            # [N, K]

    diff = gt_positions - pred_positions        # [K, P, 2]
    dists = diff.pow(2).sum(-1).sqrt()          # [K, P]

    scores_tensor = torch.zeros([identity_mask.shape[0], pred_positions.shape[0]])     # [N, K]
    for i, mask in enumerate(identity_mask):
        masked_dist = dists[:, mask]            # [K, p]
        try:
            fdes = masked_dist[:, -1]               # [K]
        except IndexError:
            fdes = torch.full([pred_positions.shape[0]], float('nan'))
        scores_tensor[i, :] = fdes

    return scores_tensor        # [N, K]


def compute_pred_lengths(
        identity_mask: Tensor           # [N, P]
) -> Tensor:                            # [N, 1]
    return torch.sum(identity_mask, dim=-1).unsqueeze(-1)


def compute_points_out_of_map(
        map_dims: torch.Size,   # (H, W)
        points: Tensor          # [*, 2]
) -> Tensor:                    # [*]
    # returns a bool mask that is True for points that lie outside the map.
    # we assume that <points> are already expressed in pixel coordinates.
    return torch.logical_or(
        points < torch.tensor([0., 0.], device=points.device),
        points >= torch.tensor(map_dims[::-1], device=points.device)
    ).any(-1)


def compute_points_in_occlusion_zone(
        occlusion_map: Tensor,      # [H, W]
        points: Tensor              # [*, 2]
) -> Tensor:                        # [*]
    # returns a bool mask that is True for points that are in the occlusion zone.
    # we assume that <points> are already expressed in pixel coordinates.
    H, W = occlusion_map.shape

    x = points[..., 0]          # [*]
    y = points[..., 1]          # [*]
    x = x.clamp(1e-4, W - 1e-4)
    y = y.clamp(1e-4, H - 1e-4)
    x = x.to(torch.int64)
    y = y.to(torch.int64)

    points_in_occlusion_zone = occlusion_map[y, x]     # [*]

    return points_in_occlusion_zone <= 0.0


def agent_mode_sequence_tensor(
        mode_tensor: Tensor,        # [K, P]
        identity_mask: Tensor,      # [N, P]
) -> Tensor:                        # [N, K]
    return torch.logical_and(
        identity_mask.unsqueeze(1),         # [N, 1, P]
        mode_tensor.unsqueeze(0)            # [1, K, P]
    )                                       # [N, K, P]


def compute_occlusion_area_count(
        pred_positions: Tensor,         # [K, P, 2]
        occlusion_map: Tensor,          # [H, W]
        identity_mask: Tensor,          # [N, P]
) -> Tensor:                            # [N, 1]
    """
    Park et al.'s Drivable Area Count metric, applied to the occlusion map.
    Note that we dismiss all modes that go outside the map; i.e., for each agent we first look at
    which modes go outside the map. From the remaining "legal" predictions (L), we compute the number of
    predictions which go out of the occlusion zone (M). The OAC is then equal to:
    (L-M)/L

    Note that L might be a smaller number than the originally predicted amount of modes (K), as some of them might
    leave the map, and therefore be considered "illegal".
    """
    # we assume that <pred_positions> are already expressed in pixel coordinates.

    points_out_of_map = compute_points_out_of_map(
        map_dims=occlusion_map.shape, points=pred_positions
    )       # [K, P]
    points_out_of_occlusion_zone = ~compute_points_in_occlusion_zone(
        occlusion_map=occlusion_map, points=pred_positions
    )       # [K, P]
    samples_out_of_map = agent_mode_sequence_tensor(
        mode_tensor=points_out_of_map, identity_mask=identity_mask
    ).any(dim=-1)       # [N, P]
    samples_out_of_occlusion_zone = agent_mode_sequence_tensor(
        mode_tensor=points_out_of_occlusion_zone, identity_mask=identity_mask
    ).any(dim=-1)       # [N, P]

    l_tensor = torch.sum(~samples_out_of_map, dim=-1)        # [N]
    m_tensor = torch.sum(torch.logical_and(samples_out_of_occlusion_zone, ~samples_out_of_map), dim=-1)     # [N]
    oac = ((l_tensor - m_tensor) / l_tensor)

    oac[identity_mask.sum(-1) == 0] = float('nan')

    return oac.unsqueeze(-1)     # [N, 1]


def compute_occlusion_area_occupancy(
        pred_positions: Tensor,             # [K, P, 2]
        occlusion_map: Tensor,              # [H, W]
        identity_mask: Tensor,              # [N, P]
) -> Tensor:                                # [N, 1]
    """
    Park et al.'s Drivable Area Occupancy metric, applied to the occlusion map.
    Note that we dismiss all modes that go outside the map; i.e., for each agent we first look at
    which modes go outside the map. From the remaining "legal" predictions (L), we compute the OAO as:
                count_traj / (len(past_traj) * count_occlusion_zone * L)

    where:
        - count_traj is the number of points lying within the occlusion zone across all predictions made for
        the agent in question
        - len(past_traj) is equal to the number of timesteps predicted over the past for that prediction
        (we do need to normalize by that number, as we have varying past sequence lengths)
        - count_occlusion_zone is the number of pixels of the occlusion zone

    Note that L might be a smaller number than the originally predicted amount of modes (K), as some of them might
    leave the map, and therefore be considered "illegal".
    """
    # we assume that <pred_positions> are already expressed in pixel coordinates
    points_out_of_map = compute_points_out_of_map(
        map_dims=occlusion_map.shape, points=pred_positions
    )       # [K, P]
    samples_out_of_map = agent_mode_sequence_tensor(
        mode_tensor=points_out_of_map, identity_mask=identity_mask
    ).any(dim=-1)       # [N, K]
    points_in_occlusion_zone = compute_points_in_occlusion_zone(
        occlusion_map=occlusion_map, points=pred_positions
    )       # [K, P]
    count_traj = agent_mode_sequence_tensor(
        mode_tensor=points_in_occlusion_zone, identity_mask=identity_mask
    ).sum(dim=-1)       # [N, K]
    count_traj[samples_out_of_map] = 0.0       # [N, K]

    l_tensor = torch.sum(~samples_out_of_map, dim=-1)       # [N]
    len_past_traj = torch.sum(identity_mask, dim=-1)        # [N]
    count_occlusion_zone = torch.sum(occlusion_map <= 0.0)  # []

    oao = count_traj.sum(dim=-1) / (l_tensor * len_past_traj * count_occlusion_zone)

    return oao.unsqueeze(-1)        # [N, 1]

# dummy_H = 10
# dummy_W = 10
# # dummy_occl_map = torch.meshgrid(torch.arange(dummy_H), torch.arange(dummy_W))        # [H, W]
# # dummy_occl_map = dummy_occl_map[0] + 2 * dummy_occl_map[1] - 15
#
# dummy_occl_map = torch.full([dummy_H, dummy_W], -10.)
# dummy_occl_map[:, 5:] = 8.
# dummy_occl_map[5:, :] = 8.
#
# # dummy_occl_map += torch.randn_like(dummy_occl_map)
#
# print(f"{dummy_occl_map=}")
# print()
# dummy_identity_mask = torch.tensor([[False, False, False, False, False],
#                                     [True, True, True, False, False],
#                                     [False, False, False, True, True]])
#
# dummy_pred_pos = torch.tensor([[[3.5, 3.5],
#                                 [4.5, 3.5],
#                                 [4.5, 3.5],
#                                 [2.2, 2.2],
#                                 [2.2, 2.2]],
#
#                                [[2.5, 3.5],
#                                 [2.5, 3.5],
#                                 [12000, 9.],
#                                 [-4, 2.2],
#                                 [7., 2.2]],
#
#                                [[2.5, 2.5],
#                                 [2.5, 2.5],
#                                 [2.5, 2.5],
#                                 [2.2, 2.2],
#                                 [7., 2.2]]])         # [3, 5, 2]
#
# print(f"{dummy_pred_pos=}")
# print()
#
# oac = compute_occlusion_area_count(
#     pred_positions=dummy_pred_pos,
#     occlusion_map=dummy_occl_map,
#     identity_mask=dummy_identity_mask
# )
#
# oao = compute_occlusion_area_occupancy(
#     pred_positions=dummy_pred_pos,
#     occlusion_map=dummy_occl_map,
#     identity_mask=dummy_identity_mask
# )
#
# print(f"{oac=}")
# print(f"{oao=}")
#
# raise NotImplementedError


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
                'OAO'
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
    }
    oao_factor = 100_000.
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

        valid_ids = pred_data['valid_id'][0]                        # [N]
        gt_identities = pred_data['pred_identity_sequence'][0]      # [P]
        gt_timesteps = pred_data['pred_timestep_sequence'][0]       # [P]
        gt_positions = pred_data['pred_position_sequence'][0]       # [P, 2]

        infer_pred_identities = pred_data['infer_dec_agents'][0]        # [P]
        infer_pred_timesteps = pred_data['infer_dec_timesteps']         # [P]
        infer_pred_positions = pred_data['infer_dec_motion']            # [K, P, 2]
        infer_pred_past_mask = pred_data['infer_dec_past_mask']         # [P]

        if cfg.get('impute', False):
            true_gt_pred_mask = ~in_data['true_observation_mask'][0]         # [N, T_total]
            impute_mask = in_data['observation_mask'][0]                # [N, T_total]
            true_last_obs_indices = in_data['true_observation_mask'][0].shape[1] - torch.argmax(torch.flip(
                in_data['true_observation_mask'][0], dims=[1]
            ).to(torch.float32), dim=1) - 1     # [N]
            for i, true_last_obs_index in enumerate(true_last_obs_indices):
                impute_mask[i, :true_last_obs_index+1] = False
                true_gt_pred_mask[i, :true_last_obs_index+1] = False

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

        if {'OAO', 'OAC'}.intersection(metrics_to_compute):
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
                computed_metrics['OAO'] = oao_factor * compute_occlusion_area_occupancy(
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

        print(f"{identity_mask=}")
        print(f"{infer_pred_timesteps=}")
        print(f"{identity_and_future_mask=}")
        print(f"{identity_and_past_mask=}")
        print(f"{infer_pred_positions=}")
        print(f"{gt_positions=}")

        print(f"{computed_metrics['ADE']=}")
        print(f"{computed_metrics['past_ADE']=}")
        print(f"{computed_metrics['all_ADE']=}")

        # remove once tests have been completed
        raise NotImplementedError
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

    if 'ADE' in metrics_to_compute:
        mode_ades = metric_columns['ADE']
        score_df['min_ADE'] = score_df[mode_ades].min(axis=1)
        score_df['mean_ADE'] = score_df[mode_ades].mean(axis=1)
    if 'ADE_px' in metrics_to_compute:
        mode_ades = metric_columns['ADE_px']
        score_df['min_ADE_px'] = score_df[mode_ades].min(axis=1)
        score_df['mean_ADE_px'] = score_df[mode_ades].mean(axis=1)
    if 'FDE' in metrics_to_compute:
        mode_fdes = metric_columns['FDE']
        score_df['min_FDE'] = score_df[mode_fdes].min(axis=1)
        score_df['mean_FDE'] = score_df[mode_fdes].mean(axis=1)
        score_df['rF'] = score_df['mean_FDE'] / score_df['min_FDE']
        # score_df['rF'] = score_df['rF'].fillna(value=1.0)
    if 'FDE_px' in metrics_to_compute:
        mode_fdes = metric_columns['FDE_px']
        score_df['min_FDE_px'] = score_df[mode_fdes].min(axis=1)
        score_df['mean_FDE_px'] = score_df[mode_fdes].mean(axis=1)
        score_df['rF_px'] = score_df['mean_FDE_px'] / score_df['min_FDE_px']
        # score_df['rF_px'] = score_df['rF_px'].fillna(value=1.0)
    if 'past_ADE' in metrics_to_compute:
        mode_min_ades = metric_columns['past_ADE']
        score_df['min_past_ADE'] = score_df[mode_min_ades].min(axis=1)
        score_df['mean_past_ADE'] = score_df[mode_min_ades].mean(axis=1)
    if 'past_ADE_px' in metrics_to_compute:
        mode_min_ades = metric_columns['past_ADE_px']
        score_df['min_past_ADE_px'] = score_df[mode_min_ades].min(axis=1)
        score_df['mean_past_ADE_px'] = score_df[mode_min_ades].mean(axis=1)
    if 'past_FDE' in metrics_to_compute:
        mode_min_fdes = metric_columns['past_FDE']
        score_df['min_past_FDE'] = score_df[mode_min_fdes].min(axis=1)
        score_df['mean_past_FDE'] = score_df[mode_min_fdes].mean(axis=1)
    if 'past_FDE_px' in metrics_to_compute:
        mode_min_fdes = metric_columns['past_FDE_px']
        score_df['min_past_FDE_px'] = score_df[mode_min_fdes].min(axis=1)
        score_df['mean_past_FDE_px'] = score_df[mode_min_fdes].mean(axis=1)
    if 'future_ADE' in metrics_to_compute:
        mode_min_ades = metric_columns['future_ADE']
        score_df['min_future_ADE'] = score_df[mode_min_ades].min(axis=1)
        score_df['mean_future_ADE'] = score_df[mode_min_ades].mean(axis=1)
    if 'future_ADE_px' in metrics_to_compute:
        mode_min_ades = metric_columns['future_ADE_px']
        score_df['min_future_ADE_px'] = score_df[mode_min_ades].min(axis=1)
        score_df['mean_future_ADE_px'] = score_df[mode_min_ades].mean(axis=1)

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
