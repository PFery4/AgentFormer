import sys

import pandas as pd
import torch
import argparse

from model.model_lib import model_dict
from utils.config import Config
from utils.utils import prepare_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--checkpoint_name', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    checkpoint_name = args.checkpoint_name
    cfg = Config(cfg_id=args.cfg, tmp=False, create_dirs=False)
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

    # model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    if checkpoint_name is not None:
        cp_path = cfg.model_path % checkpoint_name
        print(f'loading model from checkpoint: {cp_path}')
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    # providing a summary of model parameters
    param_df = pd.DataFrame(columns=['module', 'params'])

    for i, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue

        param_count = param.numel()
        param_df.loc[i] = {'module': name, 'params': param_count}

    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.max_colwidth', None,
            'display.width', None
    ):
        print(param_df)
    print(f"\nTOTAL PARAMETER COUNT:\n{param_df['params'].sum()}")
