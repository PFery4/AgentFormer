import torch
from scipy.interpolate import interp1d

Tensor = torch.Tensor


def last_observed_indices(
        obs_mask: Tensor    # [N, T]
) -> Tensor:                # [N]
    return obs_mask.shape[1] - torch.argmax(torch.flip(obs_mask, dims=[1]), dim=1) - 1


def last_observed_positions(
        trajs: Tensor,              # [N, T, 2]
        last_obs_indices: Tensor    # [N]
) -> Tensor:                        # [N, 2]
    return trajs[torch.arange(trajs.shape[0]), last_obs_indices, :]


def true_velocity(
        trajs: Tensor   # [N, T, 2]
) -> Tensor:            # [N, T, 2]
    vel = torch.zeros_like(trajs)
    vel[:, 1:, :] = trajs[:, 1:, :] - trajs[:, :-1, :]
    return vel


def observed_velocity(
        trajs: Tensor,      # [N, T, 2]
        obs_mask: Tensor    # [N, T]
) -> Tensor:                # [N, T, 2]
    vel = torch.zeros_like(trajs)
    for traj, mask, v in zip(trajs, obs_mask, vel):
        obs_indices = torch.nonzero(mask)  # [Z, 1]
        motion_diff = traj[obs_indices[1:, 0], :] - traj[obs_indices[:-1, 0], :]  # [Z - 1, 2]
        v[obs_indices[1:].squeeze(), :] = motion_diff / (obs_indices[1:, :] - obs_indices[:-1, :])  # [Z - 1, 2]
    return vel


def cv_extrapolate(
        trajs: Tensor,              # [N, T, 2]
        obs_vel: Tensor,            # [N, T, 2]
        last_obs_indices: Tensor    # [N]
) -> Tensor:                        # [N, T, 2]
    xtrpl_trajs = trajs.detach().clone()
    for traj, vel, obs_idx in zip(xtrpl_trajs, obs_vel, last_obs_indices):
        last_pos = traj[obs_idx]
        last_vel = vel[obs_idx]
        extra_seq = last_pos + torch.arange(traj.shape[0] - obs_idx).unsqueeze(1) * last_vel
        traj[obs_idx:] = extra_seq
    return xtrpl_trajs


def impute_and_cv_predict(
        trajs: Tensor,      # [N, T, 2]
        obs_mask: Tensor,   # [N, T]
        timesteps: Tensor   # [T]
) -> Tensor:                # [N, T, 2]
    imputed_trajs = torch.zeros_like(trajs)
    for idx, (traj, mask) in enumerate(zip(trajs, obs_mask)):
        # if none of the values are observed, then skip this trajectory altogether
        if mask.sum() == 0:
            continue
        f = interp1d(timesteps[mask], traj[mask], axis=0, fill_value='extrapolate')
        interptraj = f(timesteps)
        imputed_trajs[idx, ...] = torch.from_numpy(interptraj)
    return imputed_trajs


def points_within_distance(
        target_point: Tensor,   # [2]
        points: Tensor,         # [N, 2]
        distance: Tensor        # [1]
) -> Tensor:                    # [N]
    distances = torch.linalg.norm(points - target_point, dim=1)
    return distances <= distance
