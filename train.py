import os
import sys
import argparse
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import data_generator
from data.sdd_dataloader import TorchDataGeneratorSDD
from model.model_lib import model_dict
from utils.torch import get_scheduler
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring, memory_report

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def logging(cfg, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log):
    ep_time_str = convert_secs2time(ep)
    eta_time_str = convert_secs2time(ep / (iter + 1) * (total_iter * (total_epoch - epoch) - (iter + 1)))
    prnt_str = f"{cfg} |Epo: {epoch:02d}/{total_epoch:02d}, " \
               f"It: {iter:04d}/{total_iter:04d}, " \
               f"Ep: {ep_time_str:s}, ETA: {eta_time_str}," \
               f"seq: {seq:s}, frame: {frame}," \
               f"{losses_str}"
    print_log(prnt_str, log)


def train(epoch_index: int):
    since_train = time.time()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()

    for i, data in enumerate(training_loader):
        # providing the data dictionary to the model
        model.set_data(data=data)

        # zeroing the gradients
        optimizer.zero_grad()

        # making a prediction
        model_data = model()

        # computing loss and updating model parameters
        total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
        total_loss.backward()
        optimizer.step()

        # memory_report('BEFORE UPDATING LOSS METERS')
        train_loss_meter['total_loss'].update(total_loss.item())
        for key in loss_unweighted_dict.keys():
            train_loss_meter[key].update(loss_unweighted_dict[key])
        # memory_report('AFTER UPDATING LOSS METERS')

        if i % cfg.print_freq == 0:
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            logging(
                cfg=args.cfg,
                epoch=epoch_index,
                total_epoch=cfg.num_epochs,
                iter=i,
                total_iter=len(training_loader),
                ep=ep,
                seq=data['seq'][0],
                frame=data['frame'][0],
                losses_str=losses_str,
                log=log
            )
            tb_x = epoch_index * len(training_loader) + i + 1
            for name, meter in train_loss_meter.items():
                tb_logger.add_scalar(f'model_{name}', meter.avg, tb_x)

        """ save model """
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:

            scheduler.step()
            model.step_annealer()

            save_name = f"epoch_{epoch_index + 1}_batch_{i + 1}"
            cp_path = cfg.model_path % save_name
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(), 'epoch': epoch_index + 1, 'batch': i + 1}
            torch.save(model_cp, cp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp, create_dirs=True)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    # DELFTBLUE GPU ##################################################################################################
    # device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda', index=args.gpu)
    #     torch.cuda.set_device(args.gpu)
    # else:
    #     device = torch.device('cpu')
    #
    # print("-" * 120)
    # print(f"{torch.cuda.is_available()=}")
    # print(f"{torch.cuda.device_count()=}")
    # print(f"{torch.cuda.current_device()=}")
    # print(f"{torch.cuda.device(torch.cuda.current_device())=}")
    # print(f"{torch.cuda.get_device_name(torch.cuda.current_device())=}")
    # print(f"{device=}")
    # print("-" * 120)
    #
    # device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    # if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
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
    # DELFTBLUE GPU ##################################################################################################

    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir)
    print(f"{cfg.tb_dir=}")

    """ data """
    if cfg.dataset == "sdd":
        sdd_dataset = TorchDataGeneratorSDD(parser=cfg, log=log, split='train')
        training_loader = DataLoader(dataset=sdd_dataset, shuffle=True, num_workers=2)
    else:
        generator = data_generator(cfg, log, split='train', phase='training')

    """ model """
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    if args.start_epoch > 0:
        cp_path = cfg.model_path % args.start_epoch
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])

    """ start training """
    model.set_device(device)
    model.train()
    for i in range(args.start_epoch, cfg.num_epochs):
        train(i)
        # """ save model """
        # if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
        #     cp_path = cfg.model_path % (i + 1)
        #     model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
        #                 'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
        #     torch.save(model_cp, cp_path)
