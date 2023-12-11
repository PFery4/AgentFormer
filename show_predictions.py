import sys
import os
import argparse

import torch
from torch.utils.data import DataLoader

from data.sdd_dataloader import PresavedDatasetSDD
from model.model_lib import model_dict
from utils.config import Config
from utils.utils import prepare_seed, print_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default=None)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
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

    # loading model predictions
    if checkpoint_name is not None:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    else:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, 'untrained', split)
    assert os.path.exists(saved_preds_dir)
    log_str = f'loading predictions from the following directory:\n{saved_preds_dir}\n\n'
    print_log(log_str, log=log)

    # TODO: CONTINUE HERE
    