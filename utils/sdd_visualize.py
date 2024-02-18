import torch
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.colors as colors

from model.agentformer_loss import index_mapping_gt_seq_pred_seq
from utils.performance_metrics import compute_samples_ADE, compute_samples_FDE

from typing import Dict, Optional, List
Tensor = torch.Tensor


def apply_homography(points: Tensor, homography: Tensor) -> Tensor:
    # TODO: this function probably shouldn't live in this file...
    assert points.shape[-1] == 2
    assert homography.shape == torch.Size([3, 3])
    homogeneous_points = torch.cat((points, torch.ones([*points.shape[:-1], 1])), dim=-1).transpose(-1, -2)
    return (homography @ homogeneous_points).transpose(-1, -2)[..., :-1]


def verify_data_dicts_consistency(input_dict: Dict, pred_dict: Dict) -> None:
    # TODO: This function probably shouldn't live in this file...
    # print()
    # print(f"{input_dict.keys()=}")
    # print(f"{pred_dict.keys()=}")
    # print()

    if 'identities' in input_dict.keys() and 'valid_id' in pred_dict.keys():
        assert torch.all(input_dict['identities'].cpu() == pred_dict['valid_id'].squeeze().cpu())
    if 'map_homography' in input_dict.keys() and 'map_homography' in pred_dict.keys():
        assert torch.all(input_dict['map_homography'].cpu() == pred_dict['map_homography'].cpu())


def visualize_sequences(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'identities' in data_dict.keys()

    assert 'obs_identity_sequence' in data_dict.keys()
    assert 'obs_position_sequence' in data_dict.keys()
    assert 'obs_velocity_sequence' in data_dict.keys()
    assert 'obs_timestep_sequence' in data_dict.keys()
    assert 'last_obs_positions' in data_dict.keys()

    assert 'pred_identity_sequence' in data_dict.keys()
    assert 'pred_position_sequence' in data_dict.keys()
    assert 'pred_velocity_sequence' in data_dict.keys()
    assert 'pred_timestep_sequence' in data_dict.keys()

    ids = data_dict['identities']

    obs_ids = data_dict['obs_identity_sequence']
    obs_trajs = data_dict['obs_position_sequence']
    obs_vel = data_dict['obs_velocity_sequence']
    obs_timesteps = data_dict['obs_timestep_sequence']
    last_obs_pos = data_dict['last_obs_positions']

    pred_ids = data_dict['pred_identity_sequence']
    pred_trajs = data_dict['pred_position_sequence']
    pred_vel = data_dict['pred_velocity_sequence']
    pred_timesteps = data_dict['pred_timestep_sequence']

    homography = torch.eye(3)
    vel_homography = torch.eye(3)
    if 'map_homography' in data_dict.keys():
        homography = data_dict['map_homography']
        vel_homography = homography.clone()
        vel_homography[:2, 2] = 0.

    color = plt.cm.rainbow(np.linspace(0, 1, ids.shape[0]))

    homogeneous_last_obs_pos = torch.cat((last_obs_pos, torch.ones([*last_obs_pos.shape[:-1], 1])),
                                         dim=-1).transpose(-1, -2)
    plot_last_obs_pos = (homography @ homogeneous_last_obs_pos).transpose(-1, -2)[..., :-1]

    for i, (ag_id, last_pos) in enumerate(zip(ids, plot_last_obs_pos)):
        c = color[i].reshape(1, -1)
        draw_ax.scatter(last_pos[0], last_pos[1], s=70, facecolors='none', edgecolors=c, alpha=0.3)

    homogeneous_obs_trajs = torch.cat((obs_trajs, torch.ones([*obs_trajs.shape[:-1], 1])), dim=-1).transpose(-1, -2)
    plot_obs_trajs = (homography @ homogeneous_obs_trajs).transpose(-1, -2)[..., :-1]

    homogeneous_obs_vel = torch.cat((obs_vel, torch.ones([*obs_vel.shape[:-1], 1])), dim=-1).transpose(-1, -2)
    plot_obs_vel = (vel_homography @ homogeneous_obs_vel).transpose(-1, -2)[..., :-1]

    for i, (ag_id, pos, vel, timestep) in enumerate(zip(obs_ids, plot_obs_trajs, plot_obs_vel, obs_timesteps)):
        c = color[np.nonzero(ids == ag_id).flatten()].reshape(1, -1)
        s = 5 * (timestep + 8)
        draw_ax.scatter(pos[0], pos[1], marker='x', s=5 + s, color=c)
        old_pos = pos - vel
        draw_ax.plot([old_pos[0], pos[0]], [old_pos[1], pos[1]], color=c, linestyle='--', alpha=0.8)

    homogeneous_pred_trajs = torch.cat((pred_trajs, torch.ones([*pred_trajs.shape[:-1], 1])), dim=-1).transpose(-1,
                                                                                                                -2)
    plot_pred_trajs = (homography @ homogeneous_pred_trajs).transpose(-1, -2)[..., :-1]

    homogeneous_pred_vel = torch.cat((pred_vel, torch.ones([*pred_vel.shape[:-1], 1])), dim=-1).transpose(-1, -2)
    plot_pred_vel = (vel_homography @ homogeneous_pred_vel).transpose(-1, -2)[..., :-1]

    for i, (ag_id, pos, vel, timestep) in enumerate(zip(pred_ids, plot_pred_trajs, plot_pred_vel, pred_timesteps)):
        c = color[np.nonzero(ids == ag_id).flatten()].reshape(1, -1)
        s = 5 * (timestep)
        draw_ax.scatter(pos[0], pos[1], marker='*', s=5 + s, color=c)
        old_pos = pos - vel
        draw_ax.plot([old_pos[0], pos[0]], [old_pos[1], pos[1]], color=c, linestyle=':', alpha=0.8)


def visualize_trajectories(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'identities' in data_dict.keys()
    assert 'trajectories' in data_dict.keys()
    assert 'observation_mask' in data_dict.keys()

    ids = data_dict['identities']
    trajs = data_dict['trajectories']
    obs_mask = data_dict['observation_mask']

    homography = torch.eye(3)
    if 'map_homography' in data_dict.keys():
        homography = data_dict['map_homography']

    homogeneous_trajs = torch.cat((trajs, torch.ones([*trajs.shape[:-1], 1])), dim=-1).transpose(-1, -2)
    plot_trajs = (homography @ homogeneous_trajs).transpose(-1, -2)[..., :-1]

    if 'true_trajectories' in data_dict.keys() and data_dict['true_trajectories'] is not None:
        assert 'true_observation_mask' in data_dict.keys() and data_dict['true_observation_mask'] is not None
        true_trajs = data_dict['true_trajectories']
        homogeneous_true_trajs = torch.cat((true_trajs, torch.ones([*true_trajs.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        plot_true_trajs = (homography @ homogeneous_true_trajs).transpose(-1, -2)[..., :-1]

        color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, ids.shape[0])))
        impute_mask = (obs_mask != data_dict['true_observation_mask'])
        for true_traj, mask in zip(plot_true_trajs, impute_mask):
            c = next(color_iter).reshape(1, -1)
            draw_ax.scatter(true_traj[:, 0][mask], true_traj[:, 1][mask], marker='x', s=20, color=c, alpha=0.5)

    color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, ids.shape[0])))
    for traj, mask, agent_id in zip(plot_trajs, obs_mask, ids):
        c = next(color_iter).reshape(1, -1)
        draw_ax.plot(traj[:, 0][mask], traj[:, 1][mask], color=c)
        draw_ax.plot(traj[:, 0][~mask], traj[:, 1][~mask], color='black')
        draw_ax.scatter(traj[:, 0][mask], traj[:, 1][mask], marker='x', s=25, color=c, label=f'{agent_id}')
        draw_ax.scatter(traj[:, 0][~mask][:-1], traj[:, 1][~mask][:-1], marker='*', s=40, color='black')
        draw_ax.scatter(traj[:, 0][~mask][-1], traj[:, 1][~mask][-1], marker='*', s=40, color=c)


def visualize_predictions(
        pred_dict: Dict, draw_ax: matplotlib.axes.Axes,
        display_modes_mask: Optional[torch.Tensor] = None,
        alpha_displayed_modes: float = 0.5,
        alpha_non_displayed_modes: float = 0.0,
        display_mode_ids: bool = False
) -> None:
    assert 'valid_id' in pred_dict.keys()
    assert 'infer_dec_agents' in pred_dict.keys()
    # assert 'infer_dec_timesteps' in pred_dict.keys()
    assert 'infer_dec_motion' in pred_dict.keys()
    if display_modes_mask is not None:
        assert display_modes_mask.shape[0] == pred_dict['valid_id'].shape[-1]
        assert display_modes_mask.shape[1] == pred_dict['infer_dec_agents'].shape[0]
        # [N, K]
    else:
        display_modes_mask = torch.full(
            [pred_dict['valid_id'].shape[-1],
             pred_dict['infer_dec_agents'].shape[0]], True
        )

    valid_ids = pred_dict['valid_id'][0]                    # [N]
    pred_ids = pred_dict['infer_dec_agents']                # [K, P]
    # pred_timesteps = pred_dict['infer_dec_timesteps']       # [P]
    pred_trajs = pred_dict['infer_dec_motion']              # [K, P, 2]

    homography = torch.eye(3)
    if 'map_homography' in pred_dict.keys():
        homography = pred_dict['map_homography']
    homography = homography.to(pred_trajs.device)

    color = plt.cm.rainbow(np.linspace(0, 1, valid_ids.shape[0]))

    homogeneous_pred_trajs = torch.cat(
        (pred_trajs, torch.ones([*pred_trajs.shape[:-1], 1], device=pred_trajs.device)), dim=-1
    ).transpose(-1, -2)

    plot_pred_trajs = (homography @ homogeneous_pred_trajs).transpose(-1, -2)[..., :-1].cpu()
    valid_ids = valid_ids.cpu()
    pred_ids = pred_ids.cpu()

    assert torch.eq(pred_ids, pred_ids[0].repeat(pred_ids.shape[0], 1)).all()

    for i, ag_id in enumerate(valid_ids):

        c = color[i].reshape(1, -1)
        agent_mask = (pred_ids[0] == ag_id)

        for k, (pred_traj, agent_sequence) in enumerate(zip(plot_pred_trajs, pred_ids)):
            # agent_mask = (agent_sequence == ag_id)
            agent_traj = pred_traj[agent_mask]

            # TODO: better way to apply style
            c = 'black' if display_modes_mask[i, k] else color[i].reshape(1, -1)
            alpha = alpha_displayed_modes if display_modes_mask[i, k] else alpha_non_displayed_modes

            draw_ax.plot(agent_traj[..., 0], agent_traj[..., 1], color=c, alpha=alpha)
            draw_ax.scatter(agent_traj[..., 0], agent_traj[..., 1], marker='*', s=10, color=c, alpha=alpha)
            if display_mode_ids:
                draw_ax.annotate(f"{k}", (agent_traj[-1, 0], agent_traj[-1, 1]), alpha)


def visualize_scene_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'scene_map' in data_dict.keys()

    scene_map_img = data_dict['scene_map']  # [C, H, W]
    draw_ax.imshow(scene_map_img.permute(1, 2, 0))


def visualize_occlusion_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'occlusion_map' in data_dict.keys()

    occlusion_map_img = data_dict['occlusion_map']  # [H, W]
    occlusion_map_render = np.full(
        (*occlusion_map_img.shape, 4), (1., 0, 0, 0.3)
    ) * (~occlusion_map_img)[..., None].numpy()
    draw_ax.imshow(occlusion_map_render)


def visualize_dist_transformed_occlusion_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'dist_transformed_occlusion_map' in data_dict.keys()

    dt_occlusion_map = data_dict['dist_transformed_occlusion_map']

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax)
    colors_visible = plt.cm.Purples(np.linspace(0.5, 1, 256))
    colors_occluded = plt.cm.Reds(np.linspace(1, 0.5, 256))
    all_colors = np.vstack((colors_occluded, colors_visible))
    color_map = colors.LinearSegmentedColormap.from_list('color_map', all_colors)
    divnorm = colors.TwoSlopeNorm(
        vmin=np.min([-1, torch.min(dt_occlusion_map)]),
        vcenter=0.0,
        vmax=np.max([1, torch.max(dt_occlusion_map)])
    )
    cax = divider.append_axes('right', size='5%', pad=0.05)
    img = draw_ax.imshow(dt_occlusion_map, norm=divnorm, cmap=color_map)
    draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')


def visualize_map(tensor_map: Tensor, draw_ax: matplotlib.axes.Axes) -> None:
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(draw_ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    img = draw_ax.imshow(tensor_map, cmap='Greys')
    draw_ax.get_figure().colorbar(img, cax=cax, orientation='vertical')


def visualize_probability_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'probability_occlusion_map' in data_dict.keys()

    probability_map = data_dict['probability_occlusion_map']
    visualize_map(tensor_map=probability_map, draw_ax=draw_ax)


def visualize_nlog_probability_map(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    assert 'nlog_probability_occlusion_map' in data_dict.keys()

    nlog_probability_map = data_dict['nlog_probability_occlusion_map']
    visualize_map(tensor_map=nlog_probability_map, draw_ax=draw_ax)


def define_axes_boundaries(data_dict: Dict, draw_ax: matplotlib.axes.Axes) -> None:
    default_height, default_width = (800., 800.)

    if 'occlusion_map' in data_dict.keys():
        # [H, W]
        draw_ax.set_xlim(0., data_dict['occlusion_map'].shape[1])
        draw_ax.set_ylim(data_dict['occlusion_map'].shape[0], 0.)
    elif 'scene_map' in data_dict.keys():
        # [C, H, W]
        draw_ax.set_xlim(0., data_dict['scene_map'].shape[2])
        draw_ax.set_ylim(data_dict['scene_map'].shape[1], 0.)
    else:
        draw_ax.set_xlim(0., default_width)
        draw_ax.set_ylim(default_height, 0.)
        draw_ax.set_aspect('equal')


def visualize(
        data_dict: Dict,
        draw_ax: matplotlib.axes.Axes,
        draw_ax_sequences: Optional[matplotlib.axes.Axes] = None,
        draw_ax_dist_transformed_map: Optional[matplotlib.axes.Axes] = None,
        draw_ax_probability_map: Optional[matplotlib.axes.Axes] = None,
        draw_ax_nlog_probability_map: Optional[matplotlib.axes.Axes] = None,
) -> None:
    define_axes_boundaries(data_dict=data_dict, draw_ax=draw_ax)
    has_scene_map = 'scene_map' in data_dict.keys()
    has_occl_map = 'occlusion_map' in data_dict.keys()
    if has_scene_map:
        visualize_scene_map(data_dict=data_dict, draw_ax=draw_ax)
    if has_occl_map:
        visualize_occlusion_map(data_dict=data_dict, draw_ax=draw_ax)
    visualize_trajectories(data_dict=data_dict, draw_ax=draw_ax)

    if draw_ax_sequences is not None:
        define_axes_boundaries(data_dict=data_dict, draw_ax=draw_ax_sequences)
        if has_scene_map:
            visualize_scene_map(data_dict=data_dict, draw_ax=draw_ax_sequences)
        if has_occl_map:
            visualize_occlusion_map(data_dict=data_dict, draw_ax=draw_ax_sequences)
        visualize_sequences(data_dict=data_dict, draw_ax=draw_ax_sequences)

    if draw_ax_dist_transformed_map is not None:
        visualize_dist_transformed_occlusion_map(data_dict=data_dict, draw_ax=draw_ax_dist_transformed_map)

    if draw_ax_probability_map is not None:
        visualize_probability_map(data_dict=data_dict, draw_ax=draw_ax_probability_map)

    if draw_ax_nlog_probability_map is not None:
        visualize_nlog_probability_map(data_dict=data_dict, draw_ax=draw_ax_nlog_probability_map)


def visualize_input_and_predictions(
        draw_ax: matplotlib.axes.Axes,
        data_dict: Dict,
        pred_dict: Dict,
        show_rgb_map: bool = False,
        show_obs_agent_ids: Optional[List[int]] = None,     # default: show for everyone
        show_gt_agent_ids: Optional[List[int]] = None,      # default: show for everyone
        show_pred_agent_ids: Optional[List[int]] = None,     # default: show for everyone
        past_pred_alpha: float = 0.5,
        future_pred_alpha: float = 0.5
):
    verify_data_dicts_consistency(input_dict=data_dict, pred_dict=pred_dict)

    homography = (data_dict.get('map_homography', torch.eye(3))).cpu()
    identities = data_dict.get('identities').cpu()      # [N]

    # extracting the observed trajectory data
    obs_id_sequence = data_dict['obs_identity_sequence'].cpu()      # [O]
    obs_pos_sequence = data_dict['obs_position_sequence'].cpu()     # [O, 2]
    # obs_time_sequence = data_dict['obs_timestep_sequence'].cpu()    # [O]
    last_obs_pos = data_dict['last_obs_positions'].cpu()            # [N, 2]

    # extracting the ground truth future trajectory data
    gt_id_sequence = data_dict['pred_identity_sequence'].cpu()      # [P]
    gt_pos_sequence = data_dict['pred_position_sequence'].cpu()     # [P, 2]
    # gt_time_sequence = data_dict['pred_timestep_sequence'].cpu()    # [P]

    # extracting the prediction trajectory data
    pred_id_sequence = pred_dict['infer_dec_agents'].cpu()          # [K, P]
    pred_pos_sequence = pred_dict['infer_dec_motion'].cpu()         # [K, P, 2]
    # pred_time_sequence = pred_dict['infer_dec_timesteps'].cpu()     # [P]
    pred_past_mask = pred_dict['infer_dec_past_mask'].to(bool).cpu()    # [P]

    # converting position data to pixel coordinates
    obs_pos_sequence = apply_homography(obs_pos_sequence, homography)
    last_obs_pos = apply_homography(last_obs_pos, homography)
    gt_pos_sequence = apply_homography(gt_pos_sequence, homography)
    pred_pos_sequence = apply_homography(pred_pos_sequence, homography)

    # for each part of the trajectory data to display, determine which agents to show
    show_obs_agent_ids = identities.tolist() if show_obs_agent_ids is None else show_obs_agent_ids
    show_gt_agent_ids = identities.tolist() if show_gt_agent_ids is None else show_gt_agent_ids
    show_pred_agent_ids = identities.tolist() if show_pred_agent_ids is None else show_pred_agent_ids
    show_obs_agent_ids = [x for x in show_obs_agent_ids if x in identities]
    show_gt_agent_ids = [x for x in show_gt_agent_ids if x in identities]
    show_pred_agent_ids = [x for x in show_pred_agent_ids if x in identities]

    # define colors for every agent
    colors = plt.cm.rainbow(np.linspace(0, 1, identities.shape[0]))

    # plotting the map
    define_axes_boundaries(data_dict=data_dict, draw_ax=draw_ax)
    if show_rgb_map and 'scene_map' in data_dict.keys(): visualize_scene_map(data_dict=data_dict, draw_ax=draw_ax)
    if 'occlusion_map' in data_dict.keys(): visualize_occlusion_map(data_dict=data_dict, draw_ax=draw_ax)

    # plotting past trajectories
    for agent_id in show_obs_agent_ids:
        # extracting the index of agent within the identity sequence and determining the color
        identities_index = np.argwhere(identities == agent_id).item()
        color = colors[identities_index].reshape(1, -1)

        # plotting last observed position
        draw_ax.scatter(last_obs_pos[identities_index, 0], last_obs_pos[identities_index, 1], s=40, facecolors='none', edgecolors=color, alpha=0.4)

        # plotting the past trajectory
        agent_sequence = (obs_id_sequence == agent_id)
        agent_pos_sequence = obs_pos_sequence[agent_sequence, :]

        draw_ax.plot(agent_pos_sequence[:, 0], agent_pos_sequence[:, 1], c=color, marker='x', linestyle='-', alpha=0.4, label=agent_id)

    # plotting ground truth future trajectories
    for agent_id in show_gt_agent_ids:
        # extracting the index of agent within the identity sequence and determining the color
        identities_index = np.argwhere(identities == agent_id).item()
        color = colors[identities_index].reshape(1, -1)

        # plotting the ground truth future trajectory
        agent_sequence = (gt_id_sequence == agent_id)
        agent_pos_sequence = gt_pos_sequence[agent_sequence, :]     # [p, 2]
        agent_pos_sequence = np.concatenate((last_obs_pos[identities_index, :].unsqueeze(0), agent_pos_sequence), axis=0)

        draw_ax.plot(agent_pos_sequence[:, 0], agent_pos_sequence[:, 1], c='black', linestyle='-', alpha=0.5)
        draw_ax.scatter(agent_pos_sequence[1:-1, 0], agent_pos_sequence[1:-1, 1], facecolors='none', edgecolors='black', marker='X', alpha=0.5)
        draw_ax.scatter(agent_pos_sequence[-1, 0], agent_pos_sequence[-1, 1], facecolors='none', edgecolors=color, marker='X', alpha=0.5)

    # plotting predictions
    for agent_id in show_pred_agent_ids:
        # extracting the index of agent within the identity sequence and determining the color
        identities_index = np.argwhere(identities == agent_id).item()
        color = colors[identities_index].reshape(1, -1)

        # plotting the predictions
        for i, agent_sequence in enumerate(pred_id_sequence == agent_id):
            prediction = pred_pos_sequence[i, agent_sequence, :]
            past_mask = pred_past_mask[agent_sequence]
            past_pred = prediction[past_mask, :]
            future_pred = prediction[~past_mask, :]

            draw_ax.plot(
                [last_obs_pos[identities_index, 0], prediction[0, 0]],
                [last_obs_pos[identities_index, 1], prediction[0, 1]],
                c=color, alpha=past_pred_alpha if past_mask[0] else future_pred_alpha
            )
            draw_ax.plot(
                past_pred[:, 0], past_pred[:, 1],
                marker='*', linestyle='-', c=color, alpha=past_pred_alpha
            )
            draw_ax.plot(
                future_pred[:, 0], future_pred[:, 1],
                marker='*', linestyle='-', c=color, alpha=future_pred_alpha
            )
            if past_mask.sum():
                t0_index = np.where(past_mask[:-1] != past_mask[1:])[0].item()
                draw_ax.plot(
                    prediction[t0_index:t0_index+2, 0],
                    prediction[t0_index:t0_index+2, 1],
                    linestyle='-', c=color, alpha=future_pred_alpha
                )


def write_scores_per_mode(
        draw_ax: matplotlib.axes.Axes,
        pred_dict: Dict,
        show_agent_ids: Optional[List[int]] = None,     # default, show for everyone
        write_mode_number: bool = False,
        write_ade_score: bool = False,
        write_fde_score: bool = False
):
    identities = pred_dict['valid_id'].squeeze()  # [N]
    homography = pred_dict['map_homography']    # [3, 3]
    gt_timesteps = pred_dict['pred_timestep_sequence'][0]  # [P]
    gt_identities = pred_dict['pred_identity_sequence'][0]  # [P]
    gt_positions = pred_dict['pred_position_sequence'][0]  # [P, 2]
    pred_timesteps = pred_dict['infer_dec_timesteps']  # [P]
    pred_identities = pred_dict['infer_dec_agents'][0]  # [P]
    pred_positions = pred_dict['infer_dec_motion']  # [K, P, 2]

    idx_map = index_mapping_gt_seq_pred_seq(
        ag_gt=gt_identities, tsteps_gt=gt_timesteps,
        ag_pred=pred_identities, tsteps_pred=pred_timesteps
    )

    # gt_identities = gt_identities[idx_map]
    # gt_timesteps = gt_timesteps[idx_map]
    gt_positions = gt_positions[idx_map, :]

    identity_mask = identities.unsqueeze(1) == pred_identities.unsqueeze(0)  # [N, P]

    if write_ade_score:
        ades = compute_samples_ADE(
            pred_positions=pred_positions, gt_positions=gt_positions, identity_mask=identity_mask
        )       # [N, K]
    if write_fde_score:
        fdes = compute_samples_FDE(
            pred_positions=pred_positions, gt_positions=gt_positions, identity_mask=identity_mask
        )       # [N, K]

    timestep_mask = pred_timesteps == torch.max(pred_timesteps)
    pred_identities = pred_identities[timestep_mask]
    pred_positions = pred_positions[:, timestep_mask, :]

    pred_positions = apply_homography(pred_positions.cpu(), homography)

    for ag_i, identity in enumerate(pred_identities):
        if identity not in show_agent_ids:
            continue
        for k, pred_mode in enumerate(pred_positions[:, ag_i, :]):  # [2]
            score_index = identities == identity
            scores = []
            if write_mode_number:
                scores.append(f"{k}")
            if write_ade_score:
                scores.append(f"{ades[score_index, k].item():.2f}")
            if write_fde_score:
                scores.append(f"{fdes[score_index, k].item():.2f}")
            draw_ax.text(pred_mode[0], pred_mode[1], " | ".join(scores))


def show_example_instances_dataloader():
    from utils.config import Config
    from utils.utils import prepare_seed
    from data.sdd_dataloader import TorchDataGeneratorSDD, PresavedDatasetSDD

    from src.data.sdd_dataloader import StanfordDroneDatasetWithOcclusionSim
    import src.visualization.sdd_visualize as simulation_visualize
    import src.data.config as sdd_conf

    config_str = 'sdd_baseline_copy_for_test_pre'
    # dataset_type = TorchDataGeneratorSDD
    dataset_type = PresavedDatasetSDD

    compare = False
    split = 'train'

    config = Config(config_str)
    # config.occlusion_process = 'fully_observed'
    prepare_seed(config.seed)

    generator = dataset_type(parser=config, log=None, split=split)

    num_figures = 2

    if generator.occlusion_process == 'occlusion_simulation':
        num_figures += 3        # adding figures for the maps

    if compare:
        sdd_config = sdd_conf.get_config(config.sdd_config_file_name)
        compare_generator = StanfordDroneDatasetWithOcclusionSim(sdd_config, split=split)
        num_figures += 1        # adding an extra figure

    for idx in range(len(generator)):

        # 225, 615, 735
        # (run this on the v3 simulators, with max_train_agent 16 to
        # observe preprocessing of cases with too many agents)
        # if idx < 735:
        #     continue

        data_dict = generator.__getitem__(idx)

        fig, ax = plt.subplots(1, num_figures)

        if compare:
            draw_ax_compare = ax[-1]
            simulation_visualize.visualize_training_instance(
                draw_ax=draw_ax_compare, instance_dict=compare_generator.__getitem__(idx)
            )

        draw_ax_dist_transformed_map = None
        draw_ax_probability_map = None
        draw_ax_nlog_probability_map = None
        if 'dist_transformed_occlusion_map' in data_dict.keys():
            draw_ax_dist_transformed_map = ax[2]
        if 'probability_occlusion_map' in data_dict.keys():
            draw_ax_probability_map = ax[3]
        if 'nlog_probability_occlusion_map' in data_dict.keys():
            draw_ax_nlog_probability_map = ax[4]

        visualize(
            data_dict=data_dict,
            draw_ax=ax[0],
            draw_ax_sequences=ax[1],
            draw_ax_dist_transformed_map=draw_ax_dist_transformed_map,
            draw_ax_probability_map=draw_ax_probability_map,
            draw_ax_nlog_probability_map=draw_ax_nlog_probability_map
        )

        # print(f"{data_dict['identities']=}")
        # print(f"{data_dict['obs_timestep_sequence']=}")
        # print(f"{data_dict['pred_timestep_sequence']=}")

        plt.show()
