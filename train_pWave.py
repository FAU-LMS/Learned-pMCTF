import argparse
import glob
import random
import sys
import os
import logging
import time
import datetime

import torch
import torch.nn as nn

from torchvision import transforms
import torch.backends.cudnn as cudnn
import os.path as osp

from pMCTF.datasets.image import VideoFolder
from pMCTF.utils.logger import get_root_logger, get_env_info, dict2str
from optim_factory import configure_optimizers, AverageMeter
from pMCTF.utils.visualizer import Visualizer
from pMCTF.utils.util import cosine_scheduler
from pMCTF.models.pWave import pWave
import math
from torch.utils.data import DataLoader


lamda_list = [1, 35]


def get_cur_lamda(lamda_list, q_index, qp_num):
    min_l = lamda_list[0]
    max_l = lamda_list[1]
    step = (math.log(max_l) - math.log(min_l)) / (qp_num - 1)
    cur_lamda = math.exp(math.log(min_l) + step * q_index)
    return cur_lamda * 0.003


def train_one_epoch(
        model: nn.Module,
        train_dataloader: DataLoader,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        clip_max_norm: float = 0,
        logger=None,
        visualizer=None,
        num_training_steps_per_epoch=0,
        lr_schedule_values=None,
        update_freq=1,
        start_steps=0,
        wd_schedule_values=None,
):
    model.train()
    device = next(model.parameters()).device
    num_iters = len(train_dataloader)
    past_iters = len(train_dataloader)*epoch
    # logs per epoch
    log_freq = math.ceil(len(train_dataloader)/50)

    q_index_num = model.get_qp_num()

    for data_iter_step, d in enumerate(train_dataloader):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it]  # * param_group["lr_scale"] # layer scale
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        d = d.to(device)

        q_index = random.randint(0, q_index_num-1)
        cur_lmbda = get_cur_lamda(lamda_list, q_index, q_index_num)

        if not model.lossy:
            d = torch.round(d)

        optimizer.zero_grad()

        out_net = model(d, q_index)
        out_criterion = model.compute_loss(out_net, d, cur_lmbda)

        out_criterion["loss"] /= update_freq
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.step()

        if data_iter_step % log_freq == 0:
            out_info = f"Train epoch {epoch}: [ " \
                  f"{data_iter_step*len(d)}/{len(train_dataloader.dataset)}" \
                  f" ({100. * data_iter_step / len(train_dataloader):.0f}%)]"

            for loss_name in out_criterion.keys():
                out_info += f'\t{loss_name}: {out_criterion[loss_name].item():.3f} |'

            out_info += f" QP {model.QP[0].item():.3f}-{model.QP[1].item():.3f}"
            out_info += f"| QP_ll {model.QP_ll[0].item():.3f}-{model.QP_ll[1].item():.3f}"

            logger.info(out_info)
        if data_iter_step == num_iters-1 and epoch > 0:
            model.compute_visuals(d, out_net['x_hat'])
            current_visuals = model.get_current_visuals()
            if current_visuals is not None:
                visualizer.display_current_results(current_visuals, epoch)
    return past_iters + data_iter_step + 1


@torch.no_grad()
def test_epoch(epoch: int,
               test_dataloader: DataLoader,
               model: nn.Module,
               logger=None,
               visualizer=None):
    model.eval()
    device = next(model.parameters()).device

    losses = {}
    q_index_num = model.get_qp_num()
    q_index = q_index_num - 1
    cur_lmbda = get_cur_lamda(lamda_list, q_index, q_index_num)

    with torch.no_grad():
        for idx, d in enumerate(test_dataloader):
            d = d.to(device)

            if not model.lossy:
                d = torch.round(d)

            out_net = model(d, q_index)
            out_criterion = model.compute_loss(out_net, d, cur_lmbda)

            if not losses:
                # initialize
                for loss_name in out_criterion.keys():
                    losses[loss_name] = AverageMeter()
            for loss_name, loss_val in out_criterion.items():
                losses[loss_name].update(loss_val)
            if idx == 0:
                model.compute_visuals(d, out_net['x_hat'])
                current_visuals = model.get_current_visuals()
                if current_visuals is not None:
                    visualizer.display_current_results(current_visuals, epoch, True)

    test_info = f"Test epoch {epoch}: Average losses:"
    for loss_name, loss_val in losses.items():
        test_info += f"\t{loss_name}: {loss_val.avg:.3f} |"

    logger.info(test_info)

    return losses["loss"].avg


def save_checkpoint(state, is_best, exp_path, filename="checkpoint"):
    if is_best:
        savename = osp.join(exp_path, filename + '_best_loss.pth.tar')
    else:
        savename = osp.join(exp_path, filename + '.pth.tar')

    torch.save(state, savename)


def init_loggers(opt):
    opt = vars(opt)
    curr_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(opt['exp_path'], f"train_{osp.basename(opt['exp_path'])}_{curr_time}.log")
    logger = get_root_logger(logger_name='compressai', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    return logger


def load_datasets(args, device):
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop((args.patch_size))]
    )

    args.dataset = args.dataset[0]
    train_dataset = VideoFolder(args.dataset, split="train", transform=train_transforms, max_frames=1)
    test_dataset = VideoFolder(args.dataset, split="valid", transform=test_transforms, max_frames=1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    return train_dataloader, train_dataset, test_dataloader, test_dataset


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-d", "--dataset", type=str, required=True, nargs='+', help="Training dataset path")
    parser.add_argument("-e", "--epochs", default=31,  type=int,help="Number of epochs (default: %(default)s)")
    parser.add_argument("--save-freq", type=int, default=1,
                        help="epoch frequency for saving checkpoints (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=16, help="Test batch size (default: %(default)s)")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                        help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm",  default=1.0, type=float,
                        help="gradient clipping max norm (default: %(default)s")
    parser.add_argument("--experiments_root", default='experiments', type=str,
                        help="folder for storing model checkpoints and log files (default: %(default)s")
    parser.add_argument("--exp_postfix", default='', type=str,
                        help="custom postfix for the experiment name (default: %(default)s")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument("--lossless",action="store_true",
                        help="Lossless compression, i.e., integer to integer wavelet transform")
    parser.add_argument("--not_strict", action="store_true",
                        help="do not load checkpoint strictly, ignore some param mismatches "
                             "intentionally (default: %(default)s")

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.benchmark = True

    total_batch_size = args.batch_size * args.update_freq
    # mkdir for experiments and logger
    exp_name = f"lifting-fast_BS{total_batch_size}_LR{args.learning_rate}{args.exp_postfix}"
    exp_path = osp.join(args.experiments_root, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    args.exp_path = exp_path
    args.name = osp.basename(exp_path)

    # initialize loggers
    logger = init_loggers(args)
    visualizer = Visualizer(args)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_dataloader, train_dataset, test_dataloader, test_dataset = load_datasets(args, device)

    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    logger.info('Training statistics:'
                f'\n\tNumber of train images: {len(train_dataset)}'
                f'\n\tTotal batch size: {total_batch_size}'
                f'\n\tNumber of steps per epoch: {num_training_steps_per_epoch}'
                f'\n\tUpdate frequency: {args.update_freq}'
                f'\n\tTotal epochs: {args.epochs}')

    # create model
    net = pWave(lossy=not args.lossless)

    net = net.to(device)

    optimizer, _ = configure_optimizers(net, args)

    warmup_epochs = int(args.epochs * 0.055)
    lr_schedule_values = cosine_scheduler(
        optimizer.param_groups[0]['lr'], args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=warmup_epochs, warmup_steps=-1
    )

    weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logger.info(f'Loading {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1

        load_strict = not args.not_strict
        net.load_state_dict(checkpoint["state_dict"], strict=load_strict)
        if load_strict:
            optimizer.load_state_dict(checkpoint["optimizer"])

    num_parameters = sum(p.numel() for p in net.parameters())  # if p.requires_grad
    num_parameters_trafo = sum(p.numel() for pname, p in net.named_parameters() if pname.startswith(("wavelet_transform",
                                                                                                     "dequantModule")))
    num_parameters_longcontext = sum(p.numel() for pname, p in net.named_parameters() if
                                     pname.startswith("context_prediction"))
    num_parameters_contextfusion = sum(p.numel() for pname, p in net.named_parameters() if
                                     pname.startswith("context_fusion"))

    print(net)
    logger.info(f"Number of parameters (TOTAL): {num_parameters/1000000:0.3f}M \n"
                f"Number of params wavelet trafo and postprocessing module: {num_parameters_trafo/1000000:0.3f}M \n"
                f"Number of params long context prediction: {num_parameters_longcontext/1000000:0.3f}M \n"
                f"Number of params context fusion: {num_parameters_contextfusion/1000000:0.3f} \n"
                )

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        start_time = time.time()
        train_iter = train_one_epoch(
            net,
            train_dataloader,
            epoch,
            optimizer,
            args.clip_max_norm,
            logger,
            visualizer,
            num_training_steps_per_epoch,
            lr_schedule_values,
            args.update_freq,
            epoch*num_training_steps_per_epoch,
            wd_schedule_values,
        )
        seconds_left = datetime.timedelta(seconds=int(time.time() - start_time))
        seconds_left = seconds_left*(args.epochs-(epoch+1))
        logger.info(f"Remaining Time: {seconds_left.days} days {seconds_left.seconds//3600} hours {(seconds_left.seconds // 60)%60} minutes")

        loss = test_epoch(epoch, test_dataloader, net, logger, visualizer)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if epoch % args.save_freq == 0 or epoch == args.epochs-1 or is_best:
            filename = 'state_epoch' + str(epoch)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                exp_path,
                filename=filename,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
