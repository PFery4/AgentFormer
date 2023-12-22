import sys
import os.path
import argparse

import matplotlib.pyplot as plt
import torch
import pickle

from data.sdd_dataloader import PresavedDatasetSDD
from utils.sdd_visualize import visualize, visualize_predictions
from utils.config import Config
from utils.utils import prepare_seed, print_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default='best_val')        # can be 'best_val' / 'untrained' / <model_id>
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--instances', type=int, nargs='+')
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
    log = open(os.path.join(cfg.log_dir, 'log_pred_vis.txt'), 'w')

    # dataloader
    if cfg.dataset == 'sdd':
        sdd_test_set = PresavedDatasetSDD(parser=cfg, log=log, split=split)
    else:
        raise NotImplementedError

    # prediction directory
    if checkpoint_name == 'best_val':
        checkpoint_name = cfg.get_best_val_checkpoint_name()
        print(f"Best validation checkpoint name is: {checkpoint_name}")

    if checkpoint_name == 'untrained':
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, 'untrained', split)
    else:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    assert os.path.exists(saved_preds_dir)
    log_str = f'loading predictions from the following directory:\n{saved_preds_dir}\n\n'
    print_log(log_str, log=log)

    # loading model predictions
    for instance in args.instances:

        pickle_name = f"{instance:08}.pickle"

        data_file = os.path.join(sdd_test_set.dataset_dir, pickle_name)
        pred_file = os.path.join(saved_preds_dir, pickle_name)

        assert os.path.exists(data_file)
        assert os.path.exists(pred_file)

        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)

        with open(pred_file, 'rb') as f:
            pred_dict = pickle.load(f)

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f"{args.cfg}/{checkpoint_name}: {instance}")

        visualize(data_dict=data_dict, draw_ax=ax)
        visualize_predictions(pred_dict=pred_dict, draw_ax=ax)

        plt.show()

    print("Goodbye!")
