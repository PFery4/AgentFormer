import os
import sys
import argparse
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from csv import DictWriter

from data.dataloader import data_generator
from data.sdd_dataloader import TorchDataGeneratorSDD, PresavedDatasetSDD
from model.model_lib import model_dict
from utils.torch import get_scheduler
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring, memory_report

from typing import Optional
from io import TextIOWrapper

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def logging(cfg, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log: Optional[TextIOWrapper]):
    ep_time_str = convert_secs2time(ep)
    eta_time_str = convert_secs2time(ep / (iter + 1) * (total_iter * (total_epoch - epoch) - (iter + 1)))
    prnt_str = f"{cfg} |Epo: {epoch:02d}/{total_epoch:02d}, " \
               f"It: {iter:04d}/{total_iter:04d}, " \
               f"Ep: {ep_time_str:s}, ETA: {eta_time_str}," \
               f"seq: {seq:s}, frame: {frame}," \
               f"{losses_str}"
    if log is not None:
        print_log(prnt_str, log)
    else:
        print(prnt_str)


def train_one_batch(model, data, optimizer):
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

    return total_loss, loss_dict, loss_unweighted_dict


def update_loss_meters(loss_meter, total_loss, loss_unweighted_dict):
    loss_meter['total_loss'].update(total_loss.item())
    for key in loss_unweighted_dict.keys():
        loss_meter[key].update(loss_unweighted_dict[key])


def train(epoch_index: int, batch_idx: int = 0):
    since_train = time.time()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()

    if batch_idx != 0:
        log_str = f"Continuing from where we last stopped! Skipping the first {batch_idx} instances for this epoch."
        print_log(log_str, log=log)

    data_iter = iter(training_loader)
    for i in range(batch_idx):
        next(data_iter)

    for i, data in enumerate(data_iter, start=batch_idx):
        # # providing the data dictionary to the model
        # model.set_data(data=data)
        #
        # # zeroing the gradients
        # optimizer.zero_grad()
        #
        # # making a prediction
        # model_data = model()
        #
        # # computing loss and updating model parameters
        # total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
        # total_loss.backward()
        # optimizer.step()
        total_loss, loss_dict, loss_unweighted_dict = train_one_batch(model=model, data=data, optimizer=optimizer)

        # memory_report('BEFORE UPDATING LOSS METERS')
        # train_loss_meter['total_loss'].update(total_loss.item())
        # for key in loss_unweighted_dict.keys():
        #     train_loss_meter[key].update(loss_unweighted_dict[key])
        update_loss_meters(
            loss_meter=train_loss_meter,
            total_loss=total_loss,
            loss_unweighted_dict=loss_unweighted_dict
        )
        # memory_report('AFTER UPDATING LOSS METERS')

        if i % cfg.print_freq == cfg.print_freq - 1:
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
            with open(csv_train_logfile, 'a') as f:
                dict_writer = DictWriter(f, fieldnames=csv_field_names)
                meter_dict = {name: meter.avg for name, meter in train_loss_meter.items()}
                row_dict = {**{'tb_x': tb_x}, **meter_dict}
                dict_writer.writerow(row_dict)
                f.close()

        """ perform validation, and model saving """
        if (cfg.validation_freq > 0 and i % cfg.validation_freq == cfg.validation_freq - 1) or \
                (i + 1) == len(training_loader):

            log_str = f"VALIDATING:\nat training step nr {i+1}\nvalidating over {len(validation_loader)} instances."
            print_log(log_str, log=log)
            val_time = time.time()

            # TODO: figure out how frequently we should step the scheduler.
            #   Does it make sense to have this bit of code here, in the validation section?
            #   Or should we not define a new frequency parameter to manage this subprocess?
            scheduler.step()
            model.step_annealer()

            val_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
            val_loss_meter['total_loss'] = AverageMeter()

            model.eval()

            with torch.no_grad():
                for i_val, val_data in enumerate(validation_loader):
                    model.set_data(val_data)
                    model_data = model()

                    total_val_loss, val_loss_dict, val_loss_unweighted_dict = model.compute_loss()

                    update_loss_meters(
                        loss_meter=val_loss_meter,
                        total_loss=total_val_loss,
                        loss_unweighted_dict=val_loss_unweighted_dict
                    )

            tb_x = epoch_index * len(training_loader) + i + 1
            for name, meter in val_loss_meter.items():
                tb_logger.add_scalar(f'model_val_{name}', meter.avg, tb_x)
            tb_logger.flush()
            with open(csv_val_logfile, 'a') as f:
                dict_writer = DictWriter(f, fieldnames=csv_field_names)
                meter_dict = {name: meter.avg for name, meter in val_loss_meter.items()}
                row_dict = {**{'tb_x': tb_x}, **meter_dict}
                dict_writer.writerow(row_dict)
                f.close()

            val_duration = time.time() - val_time
            log_str = f"Epoch {epoch_index}, batch {i}: Validation loss is {val_loss_meter['total_loss'].avg}\n" \
                      f"Validation took: {convert_secs2time(val_duration)}"
            print_log(log_str, log=log)

            save_name = f"epoch_{epoch_index}_batch_{i}"
            cp_path = cfg.model_path % save_name
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'epoch_idx': epoch_index, 'batch_idx': i,
                        'val_loss': val_loss_meter['total_loss'].avg}
            log_str = f"saving model at:\n{cp_path}"
            print_log(log_str, log=log)
            torch.save(model_cp, cp_path)
            with open(csv_models, 'a+') as f:
                dict_writer = DictWriter(f, fieldnames=csv_models_field_names)
                row_dict = {
                    'epoch': epoch_index,
                    'batch': i,
                    'model_name': save_name,
                    'val_loss': val_loss_meter['total_loss'].avg
                }
                dict_writer.writerow(row_dict)

            print_log("\n\n", log=log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--checkpoint_name', default=None)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp, create_dirs=False)
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
    csv_train_logfile = os.path.join(cfg.log_dir, "train_losses.csv")
    csv_val_logfile = os.path.join(cfg.log_dir, "val_losses.csv")
    csv_field_names = ['tb_x', 'total_loss'] + [*cfg.loss_cfg.keys()]
    for csv_file in [csv_train_logfile, csv_val_logfile]:
        with open(csv_file, 'a+') as f:
            dict_writer = DictWriter(f, fieldnames=csv_field_names)
            row_dict = {name: name for name in csv_field_names}
            dict_writer.writerow(row_dict)
    csv_models = os.path.join(cfg.model_dir, 'models.csv')
    csv_models_field_names = ['epoch', 'batch', 'model_name', 'val_loss']
    with open(csv_models, 'a+') as f:
        dict_writer = DictWriter(f, fieldnames=csv_models_field_names)
        row_dict = {name: name for name in csv_models_field_names}
        dict_writer.writerow(row_dict)

    """ data """
    if cfg.dataset == "sdd":
        # sdd_dataset = TorchDataGeneratorSDD(parser=cfg, log=log, split='train')
        sdd_train_set = PresavedDatasetSDD(parser=cfg, log=log, split='train')
        training_loader = DataLoader(dataset=sdd_train_set, shuffle=True, num_workers=2)

        sdd_val_set = PresavedDatasetSDD(parser=cfg, log=log, split='val')
        validation_loader = DataLoader(dataset=sdd_val_set, shuffle=False, num_workers=2)
    else:
        generator = data_generator(cfg, log, split='train', phase='training')
        raise NotImplementedError

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

    start_epoch_idx = 0
    start_batch_idx = 0

    model.set_device(device)

    if args.checkpoint_name is not None:
        cp_path = cfg.model_path % args.checkpoint_name
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])
        if 'epoch_idx' in model_cp:
            start_epoch_idx = model_cp['epoch_idx']
        if 'batch_idx' in model_cp:
            start_batch_idx = model_cp['batch_idx'] + 1

    """ start training """

    model.train()
    for epoch_i in range(start_epoch_idx, cfg.num_epochs):
        log_str = f"Beginning Epoch: {epoch_i}\n"
        print_log(log_str, log=log)

        model.train(True)
        if epoch_i == start_epoch_idx:
            train(epoch_index=epoch_i, batch_idx=start_batch_idx)
        else:
            train(epoch_index=epoch_i, batch_idx=0)

    log_str = "Done for now, Goodbye!"
    print_log(log_str, log=log)
