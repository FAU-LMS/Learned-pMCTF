import math
import os
import os.path as osp
import argparse
import time

import torch
import datetime
import logging
import random
import numpy as np
from typing import List
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import json

from pMCTF.datasets.video import VideoYCbCr
from torchvision import transforms
from pMCTF.utils.logger import init_loggers

from pMCTF.models.video.pMCTF_L import pMCTF

torch.backends.cudnn.enabled = True

print_step = 100
cal_step = 10


def parse_config(args):
    config = json.load(open(args.config))
    if 'total_epochs' in config:
        args.total_epochs = config['total_epochs']
    if 'num_stages' in config:
        args.num_stages = config['num_stages']
    if 'train_lambda' in config:
        args.train_lmbda_list = config['train_lambda']

    if 'lr' in config:
        args.lr_list = config["lr"]
    if 'num_frames' in config:
        args.num_frame_list = config["num_frames"]
    if 'num_epochs' in config:
        args.num_epochs_list = config["num_epochs"]
    if 'parts' in config:
        args.parts = config["parts"]
    if 'frame_interval' in config:
        args.frame_interval = config["frame_interval"]


def save_checkpoint(state, exp_path, filename="checkpoint"):
    savename = osp.join(exp_path, filename + '.pth.tar')
    torch.save(state, savename)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_datasets(args, device, logger):
    tuplet=7
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(args.patchsize)])

    train_dataset = VideoYCbCr(args.dataset, split="train", transform=train_transforms, lossless=args.lossless,
                               num_frames=args.num_frame_list[0])

    logger.info(f"Training dataset consisting of {len(train_dataset)} videos with {tuplet} images each.")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    return train_loader


def get_cur_lamda(lamda_list, q_index, qp_num):
    min_l = lamda_list[0]
    max_l = lamda_list[1]
    step = (math.log(max_l) - math.log(min_l)) / (qp_num - 1)
    cur_lamda = math.exp(math.log(min_l) + step * q_index)
    return cur_lamda * 0.003


def train(video_net: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_lmbda_list: List[float],
          stage_num: int,
          train_loader: DataLoader,
          epoch: int,
          global_step: int,
          num_frames: int,
          max_interval: int = 1,
          logger=None):

    num_stages_tmp = 1
    while 2**num_stages_tmp < num_frames:
        num_stages_tmp += 1

    logger.info(f"Epoch {epoch} ")
    video_net.train()
    device = next(video_net.parameters()).device

    batch_count = 0
    cal_cnt = 0

    sumloss = 0
    # consider motion estimation distortion in first two stages only
    d_me = 1.0 if stage_num < 2 else 0.0
    # reconstruction distortion starting from second stage
    d_rec = 1.0 if stage_num >= 2 else 0.0
    # rate of motion vectors from stage 2
    r_mv = 1.0 if stage_num >= 1 else 0.0
    # rate of main latent features stage 3-7
    r_latent = 1.0 if stage_num >= 2 else 0.0
    random_interval = True if stage_num >= 3 else False

    # random rate only in stage 6
    random_rate = True
    q_index_num = video_net.get_qp_num()

    q_index = q_index_num - 1
    cur_lmbda = get_cur_lamda(train_lmbda_list, q_index, q_index_num)

    log_dict = {"psnr_L": 0,  "psnr_H": 0, "mse_L": 0, "mse_H": 0,
                "bpp_L": 0, "bpp_H": 0, "bpp_mv": 0, "bpp_total": 0, "bits_total": 0,
                "me_mse": 0, "warp_psnr": 0, "warp_psnr_inv": 0, "rd_loss": 0}

    t0 = datetime.datetime.now()

    if random_interval:
        train_loader.dataset.use_random_interval()
        current_interval = train_loader.dataset.set_current_interval()
    for batch_idx, data in enumerate(train_loader):
        data = [d[:, 0, :, :].unsqueeze(1).to(device) for d in data]

        global_step += 1
        batch_count += 1

        # VARIABLE RATE: Choose random lambda value and q_scales for this iteration
        if random_rate:
            q_index = random.randint(0, q_index_num-1)
            cur_lmbda = get_cur_lamda(train_lmbda_list, q_index, q_index_num)

        num_frames_stage = num_frames//2

        frames_coded = [None] * num_frames
        mvs = [None] * num_frames
        results_bit = [None] * num_frames

        if random_interval:
            if current_interval == 1:
                me_num = 0
            else:
                me_num = current_interval // (video_net.num_me_stages - 1)
        else:
            me_num = min(video_net.num_me_stages - 1, max_interval - 1)

        for stage_idx in range(num_stages_tmp):
            dpb = {"mv_feature": None, "ref_mv_y": None}
            # temporal decomposition level
            for group_idx in range(num_frames_stage):
                group_step = 2 ** stage_idx
                frame_idx = group_idx * 2 * group_step

                code_lt = (stage_idx + 1) == num_stages_tmp

                if stage_idx == 0:
                    ref_frame, cur_frame = data[frame_idx], data[frame_idx+group_step]
                else:
                    ref_frame, cur_frame = frames_coded[frame_idx], frames_coded[frame_idx + group_step]

                # code temporal Lowpass only in last temporal stage
                result = video_net(ref_frame, cur_frame, q_index, dpb=dpb,
                                   code_lt=code_lt, stage_idx=me_num+stage_idx)
                dpb = result["dpb"]

                frames_coded[frame_idx] = result["L_t"]
                if code_lt:
                    results_bit[frame_idx] = {"bpp_L": result["bpp_L"], "me_mse_inv": result["me_mse_inv"]}

                frames_coded[frame_idx+group_step] = result["H_t"]
                mvs[frame_idx+group_step] = result["mv_hat"]

                results_bit[frame_idx+group_step] = {"bpp_H": result["bpp_H"], "bpp_me": result["bpp_me"],
                                                     "me_mse": result["me_mse"], "bpp": result["bpp"],
                                                     "bit": result["bit"]}

            num_frames_stage = num_frames_stage // 2

        for stage_idx in reversed(range(num_stages_tmp)):
            if stage_idx == num_stages_tmp-1:
                num_frames_stage = 1
            else:
                num_frames_stage = num_frames_stage*2
            for group_idx in reversed(range(num_frames_stage)):
                group_step = 2 ** stage_idx
                frame_idx = group_idx * 2 * group_step

                L_t_rec = frames_coded[frame_idx]
                H_t_rec = frames_coded[frame_idx+group_step]
                mv_hat = mvs[frame_idx+group_step]
                ref_frame, cur_frame = video_net.inverse_MCTF(L_t_rec, H_t_rec, mv_hat, stage_idx=me_num+stage_idx)

                frames_coded[frame_idx] = ref_frame#.detach()
                frames_coded[frame_idx + group_step] = cur_frame#.detach()

        rd_loss = 0
        for frame_idx in range(num_frames):
            # get loss
            frame_rec = frames_coded[frame_idx]
            d_latent = video_net.mse(frame_rec, data[frame_idx])

            result_b = results_bit[frame_idx]

            rate_latent = result_b["bpp_L"] if frame_idx == 0 else result_b["bpp_H"]

            rate_mv = 0 if frame_idx == 0 else result_b["bpp_me"]
            distortion_me = 0 if frame_idx == 0 else result_b["me_mse"]
            # PER FRAME RATE DISTORTION LOSS
            rate_loss = r_latent * rate_latent + r_mv * rate_mv
            distortion = d_rec * d_latent + d_me * distortion_me

            rd_loss += cur_lmbda * distortion + rate_loss

            if global_step % cal_step == 0:
                if frame_idx == 0:
                    log_dict["mse_L"] += d_latent
                    log_dict["psnr_L"] += 20 * np.log10(255.0) - 10 * torch.log10(d_latent)
                    log_dict["bpp_L"] += result["bpp_L"]
                    log_dict["warp_psnr_inv"] += 20 * np.log10(255.0) - 10 * torch.log10(result["me_mse_inv"])
                else:
                    log_dict["bpp_H"] += result["bpp_H"] / (num_frames - 1)
                    log_dict["bpp_mv"] += result["bpp_me"] / (num_frames - 1)

                    log_dict["bpp_total"] += result["bpp"] / (num_frames - 1)
                    log_dict["bits_total"] += result["bit"] / (num_frames - 1)

                    log_dict["mse_H"] += d_latent / (num_frames - 1)
                    log_dict["me_mse"] += result["me_mse"] / (num_frames - 1)  # warp loss

                    log_dict["psnr_H"] += (20 * np.log10(255.0) - 10 * torch.log10(d_latent)) / (num_frames - 1)
                    log_dict["warp_psnr"] += (20 * np.log10(255.0) - 10 * torch.log10(result["me_mse"])) / (num_frames - 1)
                log_dict["rd_loss"] += rd_loss / (num_frames - 1)
        optimizer.zero_grad()
        rd_loss.backward()

        torch.nn.utils.clip_grad_norm_(video_net.parameters(), 5.0)
        optimizer.step()

        # compute statistics every cal_step (10) steps
        if global_step % cal_step == 0:
            cal_cnt += 1

        if (global_step % print_step) == 0:
            log_dict = {k: v.mean() for k, v in log_dict.items()}

            t1 = datetime.datetime.now()
            deltatime = t1 - t0

            out_info = f"Train epoch {epoch}: [ " \
                       f"{batch_idx*len(data)}/{len(train_loader.dataset)}" \
                       f" ({100. * batch_idx / len(train_loader):.0f}%)]"
            for loss_name in log_dict.keys():
                out_info += f'\t{loss_name}: {log_dict[loss_name].item() / cal_cnt:.3f} |'
            out_info += f"time: {(deltatime.seconds + 1e-6 * deltatime.microseconds) / batch_count}"

            logger.info(out_info)

            sumloss += log_dict["rd_loss"].item() / cal_cnt
            cal_cnt = 0
            for k, v in log_dict.items():
                log_dict[k] = 0
            t0 = t1

        if random_interval:
            current_interval = train_loader.dataset.set_current_interval()  # new interval per batch

    log = 'Train Epoch : {:02} Average Loss:\t {:.6f}\t'.format(epoch, sumloss / batch_count)
    logger.info(log)
    return global_step


def parse_args(argv):
    parser = argparse.ArgumentParser(description='pMCTF-L')
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path")
    parser.add_argument('--iframe_path', type=str, required=True,
                        help='Path to I frame model pWave++')
    parser.add_argument('--config', type=str, default="configs/train_mctf_gop16.json",
                        help='path to config helper file')
    parser.add_argument('--checkpoint', default=None,
                        help='load pretrained model from checkpoint path')
    parser.add_argument('--resume', action='store_true',
                        help='load pretrained model from checkpoint path')
    parser.add_argument('--start_stage', type=int, default=-1,
                        help="If --resume is given, continue training from stage specified in start_stage")
    parser.add_argument('--not_strict', action='store_true',
                        help="Do not load net strictly")
    # dataloader arguments
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--patchsize", type=int, nargs=2, default=(128, 128),
                        help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--exp_postfix", default='', type=str,
                        help="custom postfix for the experiment name (default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument('--lossless', action='store_true',
                        help="lossless wavelet transforms")

    parser.add_argument('--num_me_stages', default=2, type=int,
                        help="Number of temporal decomposition levels for which dedicated ME/MC "
                             "networks are available")

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    parse_config(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.benchmark = True

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Create experiment path
    exp_name = f"pMCTF_L_BS{args.batch_size}{args.exp_postfix}"
    exp_path = osp.join("experiments", exp_name)
    args.name = exp_name
    os.makedirs(exp_path, exist_ok=True)
    args.exp_path = exp_path

    # initialize loggers
    logger, _ = init_loggers(args)
    logger.setLevel(logging.INFO)

    video_net = pMCTF(lossy=not args.lossless,
                      num_me_stages=args.num_me_stages)

    num_parameters_p = sum(p.numel() for p in video_net.parameters())  # if p.requires_grad

    num_parameters_mctf = sum(p.numel() for name, p in video_net.named_parameters()
                              if name.startswith(("optic_flow", "temporal_filtering", "mv", "bit_estimator_z_mv")))

    logger.info(f"Number of parameters (TOTAL): {num_parameters_p/1000000:0.3f}M \n"
                )
    logger.info(f"Number of parameters MCTF: {num_parameters_mctf/1000000:0.3f}M \n")
    logger.info(f"Percentage MCTF params: {num_parameters_mctf/num_parameters_p*100} \%\n")

    logger.info(f'Loading I frame model {args.iframe_path}')
    i_frame_state_dict = torch.load(args.iframe_path, map_location=device)

    train_loader = load_datasets(args, device, logger)

    global_step = 0
    assert sum(args.num_epochs_list) == args.total_epochs
    stage_num = 0
    stage_duration = args.num_epochs_list[stage_num]
    epochs_cur_stage = 0

    video_net.make_inter_trainable()
    video_net.quant_stage = False  # do not use temporal layer adaptive scaling until last training stage
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, video_net.parameters()), lr=args.lr_list[0])

    video_net = video_net.to(device)
    start_epoch = 0
    if args.checkpoint:
        logger.info(f'Loading {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        video_net.load_state_dict(state_dict, strict=not args.not_strict)
        if args.resume:
            if args.start_stage > 0:
                start_epoch = sum(args.num_epochs_list[:args.start_stage - 1])
            else:
                start_epoch = checkpoint["epoch"] + 1
            epochs_ = 0
            for stage_n, stage_dur in enumerate(args.num_epochs_list):
                epochs_ += stage_dur
                if epochs_ >= start_epoch:
                    if epochs_ != start_epoch:
                        epochs_cur_stage = stage_dur - (epochs_ - start_epoch)
                    if epochs_cur_stage > 0:
                        stage_num = stage_n
                    else:
                        stage_num = stage_n + 1
                    logger.info(f"START TRAINING IN STAGE {stage_num+1}")
                    stage_duration = args.num_epochs_list[stage_num]
                    num_frames = args.num_frame_list[stage_num]
                    if num_frames > 7:
                        # switch to vimeo-32 for GOP 8 and 16
                        train_loader = load_datasets(args, device, logger, num_frames)
                    train_loader.dataset.update_num_frames(num_frames, logger)

                    if args.frame_interval[stage_num] > 1:
                        train_loader.dataset.update_interval(args.frame_interval[stage_num], logger)

                    if stage_num >= 2:
                        if args.parts[stage_num] == "All":
                            logger.info(f"MAKE ENTIRE VIDEO NET TRAINABLE")
                            video_net.make_all_trainable()
                            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, video_net.parameters()),
                                                    lr=args.lr_list[stage_num])
                        elif args.parts[stage_num] == "MCTF":
                            logger.info(f"Use different ME network for second stage")
                            video_net.make_mctf_trainable()
                            video_net = video_net.to(device)
                            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, video_net.parameters()),
                                                    lr=args.lr_list[stage_num])
                    if stage_num == 5 and args.parts[stage_num] == "All":
                        logger.info(f"ENABLE TEMPORAL LAYER ADAPTIVE QUALITY SCALING")
                        video_net.quant_stage = True
                    if num_frames > 8 and num_frames != args.num_frame_list[stage_num - 1]:
                        logger.info(f"Add another ME Stage for {num_frames} frames")
                        me_stage = int(math.log2(num_frames))
                        # initalize new MCTF stage by last available stage
                        video_net.make_mctf_trainable(start_idx=me_stage - 1, copy_idx=me_stage - 2)
                        video_net.make_all_trainable()
                    if (args.parts[stage_num] == args.parts[stage_num-1] or epochs_cur_stage > 0) \
                        and "optimizer" in checkpoint and start_epoch < 15:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    adjust_learning_rate(optimizer, args.lr_list[stage_num])
                    break
    else:
        spynet_checkpoint = torch.hub.load_state_dict_from_url(
            url='http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch',
            file_name='spynet-sintel-final')
        # spynet_checkpoint = torch.load("./checkpoints/spynet/spynet-sintel-final", map_location=device)
        spynet_checkpoint = {k.replace(".moduleBasic.0.", ".conv1."): v for k, v in spynet_checkpoint.items()}
        spynet_checkpoint = {k.replace(".moduleBasic.2.", ".conv2."): v for k, v in spynet_checkpoint.items()}
        spynet_checkpoint = {k.replace(".moduleBasic.4.", ".conv3."): v for k, v in spynet_checkpoint.items()}
        spynet_checkpoint = {k.replace(".moduleBasic.6.", ".conv4."): v for k, v in spynet_checkpoint.items()}
        spynet_checkpoint = {k.replace(".moduleBasic.8.", ".conv5."): v for k, v in spynet_checkpoint.items()}
        video_net.optic_flow.load_state_dict(spynet_checkpoint, strict=True)
        video_net.load_from_iframe(i_frame_state_dict)

    video_net = video_net.to(device)

    for epoch in range(start_epoch, args.total_epochs):
        start_time = time.time()
        if epochs_cur_stage == stage_duration:
            # Enter next training stage
            epochs_cur_stage = 0
            stage_num += 1
            stage_duration = args.num_epochs_list[stage_num]

            num_frames = args.num_frame_list[stage_num]
            if num_frames > 7:
                # switch to vimeo-32 for GOP 8 and 16
                train_loader = load_datasets(args, device, logger, num_frames)
            train_loader.dataset.update_num_frames(num_frames, logger)

            if args.frame_interval[stage_num] > 1:
                train_loader.dataset.update_interval(args.frame_interval[stage_num], logger)

            logger.info(f"ENTERING STAGE {stage_num+1}")

            if stage_num == 2 and args.parts[stage_num] == "All":
                logger.info(f"MAKE ENTIRE VIDEO NET TRAINABLE")
                video_net.make_all_trainable()
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, video_net.parameters()),
                                        lr=args.lr_list[stage_num])

            if stage_num == 3 and args.parts[stage_num] == "MCTF":
                logger.info(f"Use different ME network for second stage")
                video_net.make_mctf_trainable()
                video_net = video_net.to(device)
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, video_net.parameters()),
                                        lr=args.lr_list[stage_num])

            if stage_num == 4 and args.parts[stage_num] == "All" and args.parts[stage_num-1] != "All":
                logger.info(f"MAKE ENTIRE VIDEO NET TRAINABLE AND USE RANDOM FRAME INTERVAL")
                video_net.make_all_trainable()
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, video_net.parameters()),
                                        lr=args.lr_list[stage_num])
                train_loader.dataset.use_random_interval()

            if stage_num == 5 and args.parts[stage_num] == "All":
                logger.info(f"ENABLE TEMPORAL LAYER ADAPTIVE QUALITY SCALING")
                video_net.quant_stage = True

            if num_frames > 8 and num_frames != args.num_frame_list[stage_num-1]:
                logger.info(f"Add another ME Stage for {num_frames} frames")
                me_stage = int(math.log2(num_frames))
                # initalize new MCTF stage by last available stage
                video_net.make_mctf_trainable(start_idx=me_stage - 1, copy_idx=me_stage - 2)
                video_net.make_all_trainable()

            logger.info(f"Update learning rate to {args.lr_list[stage_num]}")
            adjust_learning_rate(optimizer, args.lr_list[stage_num])

        global_step = train(video_net,
                            optimizer,
                            args.train_lmbda_list,
                            stage_num,
                            train_loader,
                            epoch, global_step,
                            args.num_frame_list[stage_num],
                            args.frame_interval[stage_num],
                            logger)

        epochs_cur_stage += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": video_net.state_dict(),
                "optimizer": optimizer.state_dict()
            },
            exp_path,
            filename='state_epoch' + str(epoch),
        )

        seconds_left = datetime.timedelta(seconds=int(time.time() - start_time))
        seconds_left = seconds_left*(args.total_epochs-(epoch+1))
        logger.info(f"Remaining Time: {seconds_left.days} days {seconds_left.seconds//3600} hours {(seconds_left.seconds // 60)%60} minutes")


if __name__ == "__main__":
    main(sys.argv[1:])

