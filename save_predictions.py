import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sdd_dataloader import dataset_dict
from model.model_lib import model_dict
from utils.config import Config, ModelConfig
from utils.utils import prepare_seed, print_log, mkdir_if_missing, get_cuda_device


def main(args: argparse.Namespace):
    if args.legacy:
        assert args.dataset_class == 'hdf5', "Legacy mode is only available with presaved HDF5 datasets" \
                                             "(use: --dataset_class hdf5)"

    cfg = ModelConfig(cfg_id=args.cfg, tmp=args.tmp, create_dirs=False)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # device
    device = get_cuda_device(device_index=args.gpu)

    # log
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    # dataloader
    dataset_class = dataset_dict[args.dataset_class]
    data_cfg_id = args.dataset_cfg if args.dataset_cfg is not None else cfg.dataset_cfg
    dataset_cfg = Config(cfg_id=data_cfg_id)
    dataset_cfg.__setattr__('with_rgb_map', False)
    dataset_kwargs = dict(parser=dataset_cfg, split=args.data_split)
    if args.legacy:
        dataset_kwargs.update(legacy_mode=True)
    assert dataset_cfg.dataset == 'sdd'
    if dataset_cfg.dataset == 'sdd':
        sdd_test_set = dataset_class(**dataset_kwargs)
        test_loader = DataLoader(dataset=sdd_test_set, shuffle=False, num_workers=0)

    # model
    model_id = cfg.get('model_id', 'agentformer')
    for key in ['future_frames', 'motion_dim', 'forecast_dim', 'global_map_resolution']:
        assert key in dataset_cfg.yml_dict.keys()
        cfg.yml_dict[key] = dataset_cfg.__getattribute__(key)

    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    if model_id in ['const_velocity', 'oracle']:
        args.checkpoint_name = 'untrained'
    else:
        if args.checkpoint_name == 'best_val':
            args.checkpoint_name = cfg.get_best_val_checkpoint_name()
            print(f"Best validation checkpoint name is: {args.checkpoint_name}")
        cp_path = cfg.model_path % args.checkpoint_name
        print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    # saving model predictions
    save_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, args.checkpoint_name, args.data_split)
    log_str = f'saving predictions under the following directory:\n{save_dir}\n\n'
    print_log(log_str, log=log)
    mkdir_if_missing(save_dir)

    for i, data in enumerate(pbar := tqdm(test_loader)):
        filename = data['instance_name'][0]
        pbar.set_description(f"Saving: {filename}")

        out_dict = dict()
        with torch.no_grad():
            model.set_data(data)
            # recon_pred, _ = model.inference(mode='recon', sample_num=1)         # [B, P, 2]   # unused
            samples_pred, model_data = model.inference(
                mode='infer', sample_num=cfg.sample_k, need_weights=False
            )  # [B * sample_k, P, 2]

        for key, value in model_data.items():
            if key == 'valid_id' or 'last_obs_' in key or 'pred_' in key or '_dec_' in key:
                out_dict[key] = value

        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None,
                        help="Model config file (specified as either name or path")
    parser.add_argument('--dataset_cfg', type=str, default=None,
                        help="Dataset config file (specified as either name or path")
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', type=str, default='best_val',
                        help="'best_val' | 'untrained' | <model_id>")
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dataset_class', type=str, default='hdf5', help="'torch' | 'hdf5'")
    parser.add_argument('--legacy', action='store_true', default=False)
    args = parser.parse_args()

    main(args=args)
    print("\nDone, goodbye!")
