import torch
Tensor = torch.Tensor


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
