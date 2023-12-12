import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import train
from data.sdd_dataloader import TorchDataGeneratorSDD, PresavedDatasetSDD
from model.model_lib import model_dict
from utils.torch import get_scheduler
from utils.config import Config
from utils.utils import prepare_seed, get_timestring, AverageMeter
from train import print_log

if __name__ == '__main__':

    model_runs = 50
    cfg_str = 'sdd_baseline_copy_for_test_pre'
    profile_model = False
    profile_dataset = True
    presaved_dataset = False

    cfg = Config(cfg_id=cfg_str, tmp=True, create_dirs=True)

    prepare_seed(cfg.seed)

    torch.set_default_dtype(torch.float32)

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

    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir)
    print(f"{cfg.tb_dir=}")

    dataset_class = PresavedDatasetSDD if presaved_dataset else TorchDataGeneratorSDD
    sdd_dataset = dataset_class(parser=cfg, log=log, split='train')
    print(f"dataset class is of type: {type(sdd_dataset)}")
    training_loader = DataLoader(dataset=sdd_dataset, shuffle=True, num_workers=1)

    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    print(f"{device=}")
    model.set_device(device)
    model.train()

    ###################################################################################################################
    since_train = time.time()
    if profile_model:
        print(f"NOW PROFILING THE MODEL")
        train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
        train_loss_meter['total_loss'] = AverageMeter()

        data_iter = iter(training_loader)

        for i in range(model_runs):

            data = next(data_iter)

            total_loss, loss_dict, loss_unweighted_dict = train.train_one_batch(
                model=model, data=data, optimizer=optimizer
            )

            train.update_loss_meters(
                loss_meter=train_loss_meter,
                total_loss=total_loss,
                loss_unweighted_dict=loss_unweighted_dict
            )

            # print(f"{i, total_loss=}")

        print(f"Passing through {model_runs} instances in: {time.time() - since_train} seconds")

    if profile_dataset:
        print(f"NOW PROFILING THE DATASET")
        import numpy as np

        indices = np.random.randint(len(sdd_dataset), size=model_runs)

        for i, idx in enumerate(indices):
            data = sdd_dataset.__getitem__(idx)
            print(f"{i=}")

        print(f"Passing through {model_runs} instances in: {time.time() - since_train} seconds")

    print("Goodbye!")
