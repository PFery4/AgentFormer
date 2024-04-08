import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sdd_dataloader import dataset_dict
from model.model_lib import model_dict
from utils.config import Config
from utils.utils import prepare_seed, print_log, mkdir_if_missing, get_cuda_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default='best_val')        # can be 'best_val' / 'untrained' / <model_id>
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--dataset_class', default='hdf5')          # [hdf5, pickle, torch_preprocess]
    args = parser.parse_args()

    split = args.data_split
    checkpoint_name = args.checkpoint_name
    args.gpu = int(args.gpu) if args.gpu is not None else args.gpu

    cfg = Config(cfg_id=args.cfg, tmp=args.tmp, create_dirs=False)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # device
    device = get_cuda_device(device_index=args.gpu)

    # log
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    # dataloader
    assert cfg.dataset == 'sdd'
    dataset_class = dataset_dict[args.dataset_class]
    if cfg.dataset == 'sdd':
        sdd_test_set = dataset_class(parser=cfg, log=log, split=split)
        test_loader = DataLoader(dataset=sdd_test_set, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    # model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    if model_id in ['const_velocity', 'oracle']:
        checkpoint_name = 'untrained'
    else:
        if checkpoint_name == 'best_val':
            checkpoint_name = cfg.get_best_val_checkpoint_name()
            print(f"Best validation checkpoint name is: {checkpoint_name}")
        cp_path = cfg.model_path % checkpoint_name
        print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    # saving model predictions
    save_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    log_str = f'saving predictions under the following directory:\n{save_dir}\n\n'
    print_log(log_str, log=log)
    mkdir_if_missing(save_dir)

    for i, data in enumerate(pbar := tqdm(test_loader)):

        # seq_name, frame, filename = data['seq'][0], int(data['frame'][0]), data['filename'][0]
        # log_str = f"saving predictions of instance #{i}\t\t| " \
        #           f"file name: {filename}\t\t| " \
        #           f"sequence name: {seq_name.ljust(20, ' ')}\t\t| " \
        #           f"frame number: {frame}"
        # print_log(log_str, log=log)
        filename = data['instance_name'][0]
        pbar.set_description(f"Saving: {filename}")

        out_dict = dict()
        with torch.no_grad():
            model.set_data(data)
            # recon_pred, _ = model.inference(mode='recon', sample_num=1)         # [B, P, 2]   # unused
            samples_pred, model_data = model.inference(
                mode='infer', sample_num=cfg.sample_k, need_weights=False
            )                                                                   # [B * sample_k, P, 2]

        for key, value in model_data.items():
            if key == 'valid_id' or 'last_obs_' in key or 'pred_' in key or '_dec_' in key:
                out_dict[key] = value

        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nDone, goodbye!")
