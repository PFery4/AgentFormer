import torch

from typing import Dict


def compute_motion_mse(
        data: Dict,
        cfg: Dict
):
    # checking that the predicted sequence and the ground truth have the same timestep / agent order
    idx_map = index_mapping_gt_seq_pred_seq(
        ag_gt=data['pred_identity_sequence'][0],
        tsteps_gt=data['pred_timestep_sequence'][0],
        ag_pred=data['train_dec_agents'][0],
        tsteps_pred=data['train_dec_timesteps']
    )
    gt_identities = data['pred_identity_sequence'][:, idx_map]      # [B, P]
    gt_timesteps = data['pred_timestep_sequence'][:, idx_map]       # [B, P]
    gt_positions = data['pred_position_sequence'][:, idx_map, :]    # [B, P, 2]

    assert torch.all(data['train_dec_agents'] == gt_identities),\
        f"{data['train_dec_agents']=}\n\n{gt_identities=}"
    assert torch.all(data['train_dec_timesteps'] == gt_timesteps),\
        f"{data['train_dec_timesteps']=}\n\n{gt_timesteps=}"

    diff = gt_positions - data['train_dec_motion']

    if cfg.get('weight_past', False):
        past_mask = data['train_dec_past_mask']
        diff[:, past_mask, :] *= cfg.weight_past

    loss_unweighted = diff.pow(2).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= gt_timesteps.shape[1]        # normalize wrt prediction sequence length
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def index_mapping_gt_seq_pred_seq(
        ag_gt: torch.Tensor,
        tsteps_gt: torch.Tensor,
        ag_pred: torch.Tensor,
        tsteps_pred: torch.Tensor
) -> torch.Tensor:
    """
    given 4 sequences of shape (T):
    returns a tensor of indices that provides the mapping between ground truth (agent, timestep) pairs and predicted
    (agent, timestep) pairs.
    """
    gt_seq = torch.stack([tsteps_gt.detach().clone(), ag_gt.detach().clone()], dim=1)
    pred_seq = torch.stack([tsteps_pred.detach().clone(), ag_pred.detach().clone()], dim=1)
    return torch.cat([torch.nonzero(torch.all(gt_seq == elem, dim=1)) for elem in pred_seq]).squeeze()


def compute_z_kld(data: Dict, cfg: Dict):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['agent_num']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data: Dict, cfg: Dict):
    # 'infer_dec_motion' [K, P, 2]       (K modes, sequence length P)
    idx_map = index_mapping_gt_seq_pred_seq(
        ag_gt=data['pred_identity_sequence'][0],
        tsteps_gt=data['pred_timestep_sequence'][0],
        ag_pred=data['infer_dec_agents'][0],
        tsteps_pred=data['infer_dec_timesteps']
    )
    gt_identities = data['pred_identity_sequence'][:, idx_map]      # [B, P]
    gt_timesteps = data['pred_timestep_sequence'][:, idx_map]       # [B, P]
    gt_positions = data['pred_position_sequence'][:, idx_map, :]    # [B, P, 2]

    # checking that the predicted sequence and the ground truth have the same timestep / agent order
    assert torch.all(data['infer_dec_agents'] == gt_identities),\
        f"{data['infer_dec_agents']=}\n\n{gt_identities=}"
    assert torch.all(data['infer_dec_timesteps'] == gt_timesteps),\
        f"{data['infer_dec_timesteps']=}\n\n{gt_timesteps=}"

    diff = data['infer_dec_motion'] - gt_positions

    if cfg.get('weight_past', False):
        past_mask = data['infer_dec_past_mask']
        diff[:, past_mask, :] *= cfg.weight_past

    dist = diff.pow(2).sum(-1)
    dist = torch.stack(
        [dist[:, gt_identities.squeeze() == ag_id].sum(dim=-1)
         for ag_id in torch.unique(gt_identities)]
    )       # [N, K]        N agents, K modes
    loss_unweighted, _ = dist.min(dim=1)     # [N]
    if cfg.get('normalize', True):
        loss_unweighted /= (torch.unique(gt_identities).unsqueeze(1) == gt_identities).sum(dim=-1)
        loss_unweighted = loss_unweighted.mean()
    else:
        raise NotImplementedError
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_occlusion_map_loss(
        points: torch.Tensor,               # [B, P, 2]
        mask: torch.Tensor,                 # [P]
        loss_map: torch.Tensor,             # [B, H, W]
        homography_matrix: torch.Tensor,    # [B, 3, 3]
        kernel_func=None
):
    H, W = loss_map.shape[-2:]

    # transforming points to scene coordinate system
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1)).to(points.device)], dim=-1).transpose(-1, -2)
    points = (homography_matrix @ points).transpose(-1, -2)[:, mask, :]     # [B, ⊆P, 3] <==> [B, p, 3]

    x = points[..., 0]          # [B, p]
    y = points[..., 1]          # [B, p]
    x = x.clamp(1e-4, W - (1 + 1e-4))
    y = y.clamp(1e-4, H - (1 + 1e-4))

    x0 = torch.floor(x).long()
    x1 = x0+1
    y0 = torch.floor(y).long()
    y1 = y0+1

    Ia = loss_map[:, y0, x0]
    Ib = loss_map[:, y1, x0]
    Ic = loss_map[:, y0, x1]
    Id = loss_map[:, y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    interp_vals = (wa * Ia + wb * Ib + wc * Ic + wd * Id)

    if kernel_func is not None:
        interp_vals = kernel_func(interp_vals)

    loss_unweighted = interp_vals.sum()

    return loss_unweighted


def compute_train_occlusion_map_loss(data: Dict, cfg: Dict):
    points = data['train_dec_motion']                       # [B, P, 2]
    mask = data['train_dec_past_mask']                      # [P]
    loss_map = data['occlusion_loss_map']                   # [B, H, W]
    homography_matrix = data['map_homography']              # [B, 3, 3]

    loss_unweighted = compute_occlusion_map_loss(
        points=points, mask=mask, loss_map=loss_map, homography_matrix=homography_matrix,
        kernel_func=cfg.get('kernel', None)
    )

    if cfg.get('normalize', True) and mask.sum() != 0:
        loss_unweighted /= mask.sum()
    loss = loss_unweighted * cfg['weight']

    return loss, loss_unweighted


def compute_infer_occlusion_map_loss(data: Dict, cfg: Dict):
    points = data['infer_dec_motion']                       # [B * K, P, 2]
    mask = data['infer_dec_past_mask']                      # [P]
    loss_map = data['occlusion_loss_map']                   # [B, H, W]
    homography_matrix = data['map_homography']              # [B, 3, 3]

    loss_unweighted = compute_occlusion_map_loss(
        points=points, mask=mask, loss_map=loss_map, homography_matrix=homography_matrix,
        kernel_func=cfg.get('kernel', None)
    )

    if cfg.get('normalize', True) and mask.sum() != 0:
        loss_unweighted /= (mask.sum() * points.shape[0])
    loss = loss_unweighted * cfg['weight']

    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'occl_map': compute_train_occlusion_map_loss,
    'infer_occl_map': compute_train_occlusion_map_loss
}

if __name__ == '__main__':
    # cfg = {'normalize': True, 'weight': 3.0}
    # compute_occlusion_map_loss(data=None, cfg=cfg)
    print(f'Hello World!')

