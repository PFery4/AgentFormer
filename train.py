import os
import sys
import argparse
import time
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import data_generator
from data.sdd_dataloader import AgentFormerDataGeneratorForSDD
from model.model_lib import model_dict
from utils.torch import get_scheduler
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def logging(cfg, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log):
	print_log('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, ETA: {:s}, seq {:s}, frame {:05d}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter)), seq, frame, losses_str), log)


def train(epoch):
    global tb_ind
    since_train = time.time()
    generator.shuffle()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()
    last_generator_index = 0

    # WIP CODE FOR EXPLORATION
    cnt = 0
    cnt_stop = 100
    # WIP CODE FOR EXPLORATION
    while not generator.is_epoch_end():
        data = generator()
        # # WIP CODE
        # if cnt != cnt_stop:
        #     cnt += 1
        #     continue
        # # WIP CODE
        if data is not None:
            # print()
            # [print(f"{k}: {type(v)}") for k, v in data.items()]
            # print(f"{len(data['pre_motion_3D'])=}")
            # print(f"{data['pre_motion_3D'][0]=}")
            # print(f"{data['fut_motion_3D'][0]=}")
            # print(f"{data['pre_motion_mask'][0]=}")
            # print(f"{data['fut_motion_mask'][0]=}")
            # print(f"{data['heading']=}")
            # print(f"{data['valid_id'][0]=}")
            # print(f"{data['traj_scale']=}")
            # print(f"{data['pred_mask']=}")
            # print(f"{data['scene_map'].data.shape=}")
            # print(f"{data['scene_map'].data=}")
            # print(f"{data['seq']=}")
            # print(f"{data['frame']=}")
            # print()

            seq, frame = data['seq'], data['frame']
            model.set_data(data)

            # print("#" * 120)
            # print("BEFORE:")
            # print(f"{model.data['batch_size']=}")
            # print(f"{model.data['agent_num']=}")
            # [print(f"{k}: {type(v)}" + (f": {v.shape}" if isinstance(v, torch.Tensor) else "")) for k, v in model.data.items()]
            # print()

            model_data = model()
            total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()

            # print("AFTER:")
            # print(f"{model.data['batch_size']=}")
            # print(f"{model.data['agent_num']=}")
            # [print(f"{k}: {type(v)}" + (f": {v.shape}" if isinstance(v, torch.Tensor) else "")) for k, v in model.data.items()]
            # print()

            """ optimize """
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_meter['total_loss'].update(total_loss.item())
            for key in loss_unweighted_dict.keys():
                train_loss_meter[key].update(loss_unweighted_dict[key])

        if generator.index - last_generator_index > cfg.print_freq:
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            logging(args.cfg, epoch, cfg.num_epochs, generator.index, generator.num_total_samples, ep, seq, frame, losses_str, log)
            for name, meter in train_loss_meter.items():
                tb_logger.add_scalar('model_' + name, meter.avg, tb_ind)
            tb_ind += 1
            last_generator_index = generator.index

        # WIP CODE FOR EXPLORATION
        if cnt == cnt_stop:
            print(cnt)
            print(zblu)
        cnt += 1
        # WIP CODE FOR EXPLORATION

    scheduler.step()
    model.step_annealer()


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
    # device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.gpu)
        torch.cuda.set_device(args.gpu)

        # print("-" * 120)
        # print(f"{torch.cuda.is_available()=}")
        # print(f"{torch.cuda.device_count()=}")
        # print(f"{torch.cuda.current_device()=}")
        # print(f"{torch.cuda.device(torch.cuda.current_device())=}")
        # print(f"{torch.cuda.get_device_name(torch.cuda.current_device())=}")
        # print(f"{device=}")
        # print("-" * 120)

    else:
        device = torch.device('cpu')

    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir)
    tb_ind = 0

    """ data """
    if cfg.dataset == "sdd":
        generator = AgentFormerDataGeneratorForSDD(cfg, log, split="train", phase="training")
    else:
        generator = data_generator(cfg, log, split='train', phase='training')
    # print(f"{type(generator)=}")


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
        """ save model """
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
            cp_path = cfg.model_path % (i + 1)
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
            torch.save(model_cp, cp_path)

