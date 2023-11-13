import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import train
from data.sdd_dataloader import TorchDataGeneratorSDD
from model.model_lib import model_dict
from utils.torch import get_scheduler
from utils.config import Config
from utils.utils import prepare_seed, get_timestring, AverageMeter
from train import print_log

if __name__ == '__main__':

    cfg_str = 'sdd_test_config'
    cfg = Config(cfg_id=cfg_str, tmp=True, create_dirs=True)
    wait = 10
    warmup = 10
    active = 5


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

    sdd_dataset = TorchDataGeneratorSDD(parser=cfg, log=log, split='train')
    training_loader = DataLoader(dataset=sdd_dataset, shuffle=True, num_workers=2)

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

    model.set_device(device)
    print(f"{device=}")
    model.train()

    since_train = time.time()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./test_log/profiling_baseline'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        for i, data in enumerate(training_loader):

            prof.step()
            if i >= wait + warmup + active:
                break

            total_loss, loss_dict, loss_unweighted_dict = train.train_one_batch(
                model=model, data=data, optimizer=optimizer
            )

            train.update_loss_meters(
                train_loss_meter=train_loss_meter, total_loss=total_loss, loss_unweighted_dict=loss_unweighted_dict
            )

            if i % cfg.print_freq == 0:
                ep = time.time() - since_train
                losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
                train.logging(
                    cfg=cfg_str,
                    epoch=1,
                    total_epoch=cfg.num_epochs,
                    iter=i,
                    total_iter=len(training_loader),
                    ep=ep,
                    seq=data['seq'][0],
                    frame=data['frame'][0],
                    losses_str=losses_str,
                    log=log
                )
                tb_x = i + 1
                for name, meter in train_loss_meter.items():
                    tb_logger.add_scalar(f'model_{name}', meter.avg, tb_x)

        print("Goodbye!")
