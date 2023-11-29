import glob
import sys
import argparse
import os.path
import pickle

import torch
from torch.utils.data import DataLoader

from utils.config import Config
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from data.sdd_dataloader import PresavedDatasetSDD
from model.agentformer_loss import index_mapping_gt_seq_pred_seq
from model.model_lib import model_dict

Tensor = torch.Tensor


# METRICS #############################################################################################################


def compute_sequence_ADE(
        pred_positions: Tensor,         # [B * sample_num, P, 2]
        pred_identities: Tensor,        # [B * sample_num, P]
        pred_timesteps: Tensor,         # [P]
        gt_positions: Tensor,           # [B, P, 2]
        gt_identities: Tensor,          # [B, P]
        gt_timesteps: Tensor,           # [B, P]
        unique_identities: Tensor,      # [B, N]
):
    # TODO: this function needs verification
    # TODO: perhaps assume correct index mapping within the function (move index mapping elsewhere)
    idx_map = index_mapping_gt_seq_pred_seq(
        ag_gt=gt_identities[0],
        tsteps_gt=gt_timesteps[0],
        ag_pred=pred_identities[0],
        tsteps_pred=pred_timesteps
    )

    # NOTE: careful, in place modification might be weird for subsequent calls of the function...
    gt_identities = gt_identities[:, idx_map]
    gt_timesteps = gt_timesteps[:, idx_map]
    gt_positions = gt_positions[:, idx_map, :]

    assert torch.all(pred_identities == gt_identities)
    assert torch.all(pred_timesteps == gt_timesteps)

    diff = gt_positions - pred_positions        # [B * sample_num, P, 2]
    dists = diff.pow(2).sum(-1).sqrt()          # [B * sample_num, P]

    identity_masks = gt_identities == unique_identities.unique().unsqueeze(1)       # [N, P]

    identity_list = []
    sample_mode_list = []
    score_list = []
    for mask, identity in zip(identity_masks, unique_identities.unique()):
        # mask [P]
        masked_dist = dists[:, mask]              # [B * sample_num, p]
        ades = torch.mean(masked_dist, dim=-1)    # [B * sample_num]

        identity_list.extend([identity] * ades.shape[0])
        sample_mode_list.extend(list(range(ades.shape[0])))
        score_list.extend(ades.tolist())

    return score_list, identity_list, sample_mode_list

def compute_sequence_FDE():
    pass


def compute_occlusion_area_occupancy():
    # Park et al.'s DAO applied on the occlusion map
    pass


def compute_occlusion_area_count():
    # Park et al.'s DAC applied on the occlusion map
    pass


def compute_rf():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default=None)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--log_file', default=None)
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
    if cfg.dataset == 'sdd':
        sdd_test_set = PresavedDatasetSDD(parser=cfg, log=log, split=split)
        test_loader = DataLoader(dataset=sdd_test_set, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    # model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    if checkpoint_name is not None:
        cp_path = cfg.model_path % checkpoint_name
        print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    # saving model predictions
    if checkpoint_name is not None:
        save_dir = os.path.join(cfg.result_dir, checkpoint_name, split)
    else:
        save_dir = os.path.join(cfg.result_dir, 'untrained', split)
    log_str = f'saving predictions under the following directory:\n{save_dir}\n\n'
    print_log(log_str, log=log)
    mkdir_if_missing(save_dir)

    for i, data in enumerate(test_loader):

        seq_name, frame, filename = data['seq'][0], int(data['frame'][0]), data['filename'][0]
        log_str = f'saving predictions of instance #{i}\t\t| ' \
                  f'file name: {filename}\t\t| ' \
                  f'sequence name: {seq_name}\t\t| ' \
                  f'frame number: {frame}'
        print_log(log_str, log=log)

        out_dict = dict()
        with torch.no_grad():
            model.set_data(data)
            recon_pred, _ = model.inference(mode='recon', sample_num=1)         # [B, P, 2]
            samples_pred, model_data = model.inference(
                mode='infer', sample_num=cfg.sample_k, need_weights=False
            )                                                                   # [B * sample_k, P, 2]

        for key, value in model_data.items():
            if key == 'valid_id' or 'last_obs_' in key or 'pred_' in key or '_dec_' in key:
                out_dict[key] = value

        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # computing performance metrics from saved predictions
    # TODO: CONTINUE HERE
    for i, filename in enumerate(glob.glob1(save_dir, '*.pickle')):
        print(f"{filename=}")
        with open(os.path.join(save_dir, filename), 'rb') as f:
            data_dict = pickle.load(f)
        print(f"{data_dict.keys()=}")


