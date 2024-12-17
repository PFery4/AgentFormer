import argparse
import pandas as pd
import torch

from model.model_lib import model_dict
from utils.config import ModelConfig
from utils.utils import prepare_seed, get_cuda_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None,
                        help="Model config file (specified as either name or path)")
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help="None | 'best_val' | 'untrained' | <model_id>")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    checkpoint_name = args.checkpoint_name
    cfg = ModelConfig(cfg_id=args.cfg, tmp=False, create_dirs=False)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # device
    device = get_cuda_device()

    # model
    if checkpoint_name == 'best_val':
        checkpoint_name = cfg.get_best_val_checkpoint_name()
        print(f"Best validation checkpoint name is: {checkpoint_name}")

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
