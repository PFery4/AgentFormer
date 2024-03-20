import torch
from numpy import pi

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

    loss_unweighted = diff.pow(2).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= gt_timesteps.shape[1]        # normalize wrt prediction sequence length
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def gaussian_twodee_nll(mu, sig, rho, targets, eps: float = 1e-20):
    """
    referring to : https://github.com/quancore/social-lstm/blob/bdddf36e312c9dde7a5f1d447000e59918346520/helper.py#L250
    mu.shape -> (*, 2)
    sig.shape -> (*, 2)
    rho.shape -> (*, 1)
    targets.shape -> (*, 2)
    reduction -> ["mean", "sum", "tensor"]
    """
    raise NotImplementedError
    mux, muy = mu[..., 0].unsqueeze(-1), mu[..., 1].unsqueeze(-1)  # shape [*, 1]
    sigx, sigy = sig[..., 0].unsqueeze(-1), sig[..., 1].unsqueeze(-1)  # shape [*, 1]
    gt_x, gt_y = targets[..., 0].unsqueeze(-1), targets[..., 1].unsqueeze(-1)  # shape [*, 1]

    norm_x, norm_y = (gt_x - mux) / sigx, (gt_y - muy) / sigy

    z = norm_x ** 2 + norm_y ** 2 - 2 * rho * norm_x * norm_y
    min_r = 1 - rho ** 2

    result = torch.exp(-z / (2 * min_r))
    denom = 2 * pi * sigx * sigy * torch.sqrt(min_r)

    result = result / denom

    result = -torch.log(torch.clamp(result, min=eps))

    return result


def gaussian_twodee_nll_2(mu, sig, rho, targets, eps: float = 1e-20):
    """
    referring to : https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood
    mu.shape -> (*, 2)
    sig.shape -> (*, 2)
    rho.shape -> (*, 1)
    targets.shape -> (*, 2)
    """
    raise NotImplementedError
    mux, muy = mu[..., 0].unsqueeze(-1), mu[..., 1].unsqueeze(-1)  # shape [*, 1]
    sigx, sigy = sig[..., 0].unsqueeze(-1), sig[..., 1].unsqueeze(-1)  # shape [*, 1]
    gt_x, gt_y = targets[..., 0].unsqueeze(-1), targets[..., 1].unsqueeze(-1)  # shape [*, 1]

    sigx_sq, sigy_sq = sigx ** 2, sigy ** 2
    min_r = 1 - rho ** 2
    normx, normy = (gt_x - mux) / torch.clamp(sigx, min=eps), (gt_y - muy) / torch.clamp(sigy, min=eps)

    # const = torch.ones_like(rho) * 4 * pi ** 2

    result = torch.log(torch.clamp(sigx_sq, min=eps)) + \
             torch.log(torch.clamp(sigy_sq, min=eps)) + \
             torch.log(torch.clamp(min_r, min=eps)) + \
             (normx ** 2 + normy ** 2 - 2 * rho * normx * normy) / min_r    # + torch.log(const)

    return result


def multivariate_gaussian_nll(mu, Sig, targets, eps: float = 1e-20):
    """
    mu.shape -> (*, N)
    Sig.shape -> (*, N, N)
    targets.shape -> (*, N)
    """
    raise NotImplementedError
    norm = (targets - mu).unsqueeze(-1)     # [*, N, 1]

    t1 = (norm.transpose(-2, -1) @ torch.linalg.inv(torch.clamp(Sig, min=eps)) @ norm).view(mu.shape[:-1])      # [*]
    t2 = torch.log(torch.linalg.det(torch.clamp(Sig, min=eps)))                                                 # [*]
    # t3 = torch.log(torch.sum(torch.ones_like(mu) * (2 * pi)**2, dim=-1))

    result = t1 + t2    # + t3

    return result


def compute_gauss_nll(data: Dict, cfg: Dict):
    loss_unweighted = multivariate_gaussian_nll(mu=data['train_dec_mu'],
                                                Sig=data['train_dec_Sig'],
                                                targets=data['fut_motion_orig'])
    raise NotImplementedError
    if cfg.get('mask', True):
        mask = data['fut_mask']
        loss_unweighted *= mask
    loss_unweighted = torch.mean(loss_unweighted) if cfg.get('normalize', True) else torch.sum(loss_unweighted)
    loss_unweighted = 0.5 * loss_unweighted         # why 0.5? -> https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood
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


def compute_occlusion_map_loss(data: Dict, cfg: Dict):
    points = data['train_dec_motion']                       # [B, P, 2]
    mask = data['train_dec_past_mask']                      # [P]
    nlog_p_map = data['nlog_probability_occlusion_map']     # [B, H, W]
    homography_matrix = data['map_homography']              # [B, 3, 3]
    H, W = nlog_p_map.shape[-2:]

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

    Ia = nlog_p_map[:, y0[-1], x0[-1]]
    Ib = nlog_p_map[:, y1[-1], x0[-1]]
    Ic = nlog_p_map[:, y0[-1], x1[-1]]
    Id = nlog_p_map[:, y1[-1], x1[-1]]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    interp_vals = (wa * Ia + wb * Ib + wc * Ic + wd * Id)

    loss_unweighted = interp_vals.sum()
    if cfg.get('normalize', True) and mask.sum() != 0:
        loss_unweighted /= mask.sum()
    loss = loss_unweighted * cfg['weight']

    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'nll': compute_gauss_nll,
    'occl_map': compute_occlusion_map_loss
}

if __name__ == '__main__':
    # cfg = {'normalize': True, 'weight': 3.0}
    # compute_occlusion_map_loss(data=None, cfg=cfg)
    print(f'Hello World!')

