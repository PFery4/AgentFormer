import torch
from numpy import pi


def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'] - data['train_dec_motion']
    if cfg.get('mask', True):
        mask = data['fut_mask']
        diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def gaussian_twodee_nll(mu, sig, rho, targets, reduction: str = "mean", eps: float = 1e-20):
    """
    referring to : https://github.com/quancore/social-lstm/blob/bdddf36e312c9dde7a5f1d447000e59918346520/helper.py#L250
    mu.shape -> (*, 2)
    sig.shape -> (*, 2)
    rho.shape -> (*, 1)
    targets.shape -> (*, 2)
    reduction -> ["mean", "sum"]
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

    return getattr(torch, reduction)(result)


def gaussian_twodee_nll_2(mu, sig, rho, targets, reduction: str = "mean", eps: float = 1e-20):
    """
    referring to : https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood
    mu.shape -> (*, 2)
    sig.shape -> (*, 2)
    rho.shape -> (*, 1)
    targets.shape -> (*, 2)
    reduction -> ["mean", "sum"]
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

    return 0.5 * getattr(torch, reduction)(result)


def multivariate_gaussian_nll(mu, Sig, targets, reduction: str = "mean", eps: float = 1e-20):
    """
    mu.shape -> (*, N)
    Sig.shape -> (*, N, N)
    targets.shape -> (*, N)
    reduction -> ["mean", "sum"]
    """
    norm = (targets - mu).unsqueeze(-1)     # (*, N, 1)

    t1 = norm.transpose(-2, -1) @ torch.linalg.inv(torch.clamp(Sig, min=eps)) @ norm
    t2 = torch.log(torch.linalg.det(torch.clamp(Sig, min=eps)))
    # t3 = torch.log(torch.ones(mu.shape[-1]) * (2 * pi)**2)

    result = t1 + t2    # + t3

    return 0.5 * getattr(torch, reduction)(result)

def compute_gauss_nll(data, cfg):

    if cfg.get('mask', True):
        mask = data['fut_mask']

    raise NotImplementedError


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'nll': compute_gauss_nll
}
