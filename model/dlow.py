import torch
from torch import nn
from torch.nn import functional as F
# from utils.torch import *
from utils.config import Config
from model.agentformer_loss import index_mapping_gt_seq_pred_seq
from model.common.mlp import MLP
# from model.common.dist import *
from model.common.dist import Normal
from model import model_lib
from model.agentformer_loss import compute_occlusion_map_loss


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist_dlow'].kl(data['p_z_dist_infer']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['agent_num']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def diversity_loss(data, cfg):
    loss_unweighted = 0
    pred_motions = data['infer_dec_motion'].view(data['infer_dec_motion'].shape[0], -1)    # [K, P * 2]

    ids_masks = torch.repeat_interleave(
        (data['infer_dec_agents'][0].unsqueeze(0) == data['infer_dec_agents'].unique().unsqueeze(1)), repeats=2, dim=-1
    )           # [N, P * 2]

    for id_mask in ids_masks:
        pred_seq = pred_motions[:, id_mask]
        dist = F.pdist(pred_seq, 2) ** 2
        loss_unweighted += (-dist / cfg['d_scale']).exp().mean()

    if cfg.get('normalize', True):
        loss_unweighted /= data['agent_num']
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def recon_loss(data, cfg):
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


def compute_infer_occlusion_map_loss(data, cfg):
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
    'kld': compute_z_kld,
    'diverse': diversity_loss,
    'recon': recon_loss,
    'infer_occl_map': compute_infer_occlusion_map_loss
}


class DLow(nn.Module):
    """ DLow (Diversifying Latent Flows)"""
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg
        self.nk = cfg.sample_k
        self.nz = cfg.nz
        self.share_eps = cfg.get('share_eps', True)
        self.train_w_mean = cfg.get('train_w_mean', False)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())

        pred_cfg = Config(cfg.pred_cfg, tmp=False, create_dirs=False)
        pred_model = model_lib.model_dict[pred_cfg.model_id](pred_cfg)
        self.pred_model_dim = pred_cfg.tf_model_dim

        assert cfg.pred_checkpoint_name is not None
        cp_path = pred_cfg.model_path % cfg.pred_checkpoint_name
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location='cpu')
        pred_model.load_state_dict(model_cp['model_dict'])
        pred_model.eval()
        self.pred_model = [pred_model]

        # Dlow's Q net
        self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, self.nk * self.nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, self.nk * self.nz)

    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data

    def main(self, mean=False, need_weights=False):
        pred_model = self.pred_model[0]
        if pred_model.global_map_attention:
            self.data['global_map_encoding'] = pred_model.global_map_encoder(self.data['input_global_map'])

        pred_model.context_encoder(self.data)

        if not mean:
            if self.share_eps:
                eps = torch.randn([1, self.nz]).to(self.device)                 # [1, nz]
                eps = eps.repeat((self.data['agent_num'] * self.nk, 1))         # [N * nk, nz]
            else:
                eps = torch.randn([self.data['agent_num'], self.nz]).to(self.device)    # [N, nz]
                eps = eps.repeat_interleave(self.nk, dim=0)                             # [N * nk, nz]

        qnet_h = self.q_mlp(self.data['agent_context'])                                 # [B, N, q_mlp.outdim]

        A = self.q_A(qnet_h).view(*qnet_h.shape[:-1], self.nk, self.nz).permute(0, 2, 1, 3).reshape(
            -1, self.data['agent_num'], self.nz)                  # [B * nk, N, nz]
        b = self.q_b(qnet_h).view(*qnet_h.shape[:-1], self.nk, self.nz).permute(0, 2, 1, 3).reshape(
            -1, self.data['agent_num'], self.nz)                  # [B * nk, N, nz]
        z = b if mean else A*eps + b                    # [B * nk, N, nz]
        logvar = (A ** 2 + 1e-8).log()                  # [B * nk, N, nz]
        self.data['q_z_dist_dlow'] = Normal(mu=b, logvar=logvar)

        pred_model.future_decoder(
            self.data, mode='infer', sample_num=self.nk, autoregress=True, z=z, need_weights=need_weights
        )
        return self.data

    def forward(self):
        return self.main(mean=self.train_w_mean)

    def inference(self, mode, sample_num, need_weights=False):
        self.main(mean=True, need_weights=need_weights)
        res = self.data[f'infer_dec_motion']            # [B * sample_num, P, 2]

        if mode == 'recon':
            res = res[0, ...]

        return res, self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict

    def step_annealer(self):
        pass