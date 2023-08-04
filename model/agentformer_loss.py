import torch
from numpy import pi


def compute_motion_mse(
        data: dict,
        cfg: dict
):

    # print(f"INDEX MAPPING")
    idx_map = index_mapping_gt_seq_pred_seq(
        ag_gt=data['fut_agents'],
        tsteps_gt=data['fut_timesteps'],
        ag_pred=data['train_dec_agents'],
        tsteps_pred=data['train_dec_timesteps']
    )
    # print(f"{idx_map=}")

    # modded_ag_gt = data['fut_agents'][idx_map]
    # modded_t_gt = data['fut_timesteps'][idx_map]
    # print(f"{data['fut_agents']=}")
    # print(f"{data['fut_timesteps']=}")
    # print(f"{modded_ag_gt=}")
    # print(f"{modded_t_gt=}")
    # print(f"{data['train_dec_agents']=}")
    # print(f"{data['train_dec_timesteps']=}")

    # assert torch.all(modded_t_gt == data['train_dec_timesteps'])
    # assert torch.all(modded_ag_gt == data['train_dec_agents'])
    # assert torch.all(data['fut_timesteps'] == data['train_dec_timesteps'])
    # assert torch.all(data['fut_agents'] == data['train_dec_agents'])

    # print(f"{data['fut_sequence'][idx_map].shape=}")
    # print(f"{data['train_dec_motion'].squeeze(1).shape=}")

    diff = data['fut_sequence'][idx_map] - data['train_dec_motion'].squeeze(1)

    loss_unweighted = diff.pow(2).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['agent_num']    # Should we not normalize by number of predicted timesteps instead?
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
    mux, muy = mu[..., 0].unsqueeze(-1), mu[..., 1].unsqueeze(-1)  # shape (*, 1)
    sigx, sigy = sig[..., 0].unsqueeze(-1), sig[..., 1].unsqueeze(-1)  # shape (*, 1)
    gt_x, gt_y = targets[..., 0].unsqueeze(-1), targets[..., 1].unsqueeze(-1)  # shape (*, 1)

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
    mux, muy = mu[..., 0].unsqueeze(-1), mu[..., 1].unsqueeze(-1)  # shape (*, 1)
    sigx, sigy = sig[..., 0].unsqueeze(-1), sig[..., 1].unsqueeze(-1)  # shape (*, 1)
    gt_x, gt_y = targets[..., 0].unsqueeze(-1), targets[..., 1].unsqueeze(-1)  # shape (*, 1)

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
    norm = (targets - mu).unsqueeze(-1)     # (*, N, 1)

    t1 = (norm.transpose(-2, -1) @ torch.linalg.inv(torch.clamp(Sig, min=eps)) @ norm).view(mu.shape[:-1])      # (*)
    t2 = torch.log(torch.linalg.det(torch.clamp(Sig, min=eps)))                                                 # (*)
    # t3 = torch.log(torch.sum(torch.ones_like(mu) * (2 * pi)**2, dim=-1))

    result = t1 + t2    # + t3

    return result


def compute_gauss_nll(data, cfg):
    loss_unweighted = multivariate_gaussian_nll(mu=data['train_dec_mu'],
                                                Sig=data['train_dec_Sig'],
                                                targets=data['fut_motion_orig'])
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


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
    # print(f"{data['fut_sequence'].shape=}")
    # print(f"{data['fut_sequence'].unsqueeze(1).shape=}")
    # print(f"{data['train_dec_motion'].shape=}")
    # print(f"{data['infer_dec_motion'].shape=}")
    # print(f"{data['infer_dec_agents'].shape=}")
    # print(f"{data['infer_dec_timesteps'].shape=}")

    idx_map = index_mapping_gt_seq_pred_seq(
        ag_gt=data['fut_agents'],
        tsteps_gt=data['fut_timesteps'],
        ag_pred=data['infer_dec_agents'],
        tsteps_pred=data['infer_dec_timesteps']
    )

    # print("diff = data['infer_dec_motion'] - data['fut_sequence'].unsqueeze(1)")
    diff = data['infer_dec_motion'] - data['fut_sequence'][idx_map].unsqueeze(1)
    # print(f"{diff.shape=}")

    dist = diff.pow(2).sum(-1)
    # print(f"{dist.shape=}")

    dist = torch.stack(
        [(dist[data['infer_dec_agents'] == ag_id]).sum(0)/torch.sum(data['infer_dec_agents'] == ag_id)
         for ag_id in torch.unique(data['infer_dec_agents'])], dim=0
    )
    # print(f"{dist.shape=}")

    loss_unweighted = dist.min(dim=1)[0]
    # print(f"{loss_unweighted.shape=}")

    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']

    # print(f"{loss.shape=}")
    # print(f"{loss_unweighted.shape=}")
    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'nll': compute_gauss_nll
}
