import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import train
from data.sdd_dataloader import TorchDataGeneratorSDD, PresavedDatasetSDD, PickleDatasetSDD, HDF5DatasetSDD
from model.model_lib import model_dict
from utils.torch import get_scheduler
from utils.config import Config
from utils.utils import prepare_seed, get_timestring, AverageMeter
from train import print_log


def analyze_cProfile(filename):
    # This script might come in handy when it comes to optimizing model performance

    import re
    import pandas as pd

    def remove_extraneaous_semi_colons(string):
        start_idx = string.replace(';', 'Γ', 4).find(';') + 1
        string = string[:start_idx] + string[start_idx:].replace(';', ' ')
        return string

    with open(filename) as f:
        lines = f.readlines()

    lines.pop(-1)
    # print(f"{lines=}")

    clean_lines = []

    columns = lines[0]
    columns = re.sub('[ \t]+', ';', columns)
    columns = columns.replace('\n', '')
    columns = columns[1:]
    columns = columns.split(';')
    columns[4] = 'percall2'

    data_dict = {x: [] for x in columns}

    for line in lines[1:]:
        clean_line = re.sub('[ \t]+', ';', line)
        if clean_line[0] == ';':
            clean_line = clean_line[1:]
        clean_line = remove_extraneaous_semi_colons(clean_line)
        clean_line = clean_line.replace('\n', '')
        clean_lines.append(clean_line)
        # print(f"{clean_line=}")
        assert clean_line.count(';') == 5

        data = clean_line.split(';')
        for key, val in zip(columns, data):
            data_dict[key].append(val)

    df = pd.DataFrame(data=data_dict)

    df[['tottime', 'percall', 'cumtime', 'percall2']] = df[['tottime', 'percall', 'cumtime', 'percall2']].astype(float)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 150):
        print(df.sort_values(by=['cumtime'], ascending=False))


if __name__ == '__main__':

    model_runs = 50
    # cfg_str = 'sdd_baseline_copy_for_test_pre'
    cfg_str = 'original_agentformer_pre'
    profile_model = True
    profile_dataset = False
    dataset_class = PickleDatasetSDD        # [PickleDatasetSDD, HDF5DatasetSDD, TorchDataGeneratorSDD]

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
