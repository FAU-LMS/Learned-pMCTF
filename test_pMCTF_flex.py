# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import glob
import os
import json
import time
import sys


import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pMCTF.models.video.pMCTF_L import pMCTF
from pMCTF.utils.video_eval_utils import str2bool, create_folder, generate_log_json, dump_json
from pMCTF.utils.stream_helper import get_padding_size, get_state_dict
from pMCTF.utils.yuv_reader import YUVReader
from pMCTF.utils.util import ycbcr2rgb, yuv_420_to_444
from pytorch_msssim import ms_ssim


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument("--force_intra", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--last_frames", action='store_true')
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str, default="output.json", required=False)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--seq_num', type=int, default=-1)
    parser.add_argument('--ds_name', type=str, default=None)

    # for wavelet model
    parser.add_argument('--lossless', action='store_true',
                        help="lossless wavelet transform")
    parser.add_argument('--two_stage_me', action='store_true', help="Use different ME network depending on "
                                                                    "temporal decomposition stage")
    parser.add_argument('--skip_decoding', action='store_true', help="Skip decoding when writing stream ")
    parser.add_argument('--num_me_stages', default=1, type=int, help="Number of ME/MC options")
    parser.add_argument("--q_index_num", default=1, type=int, help="Specify number of q_indices (default: %(default)s)")
    parser.add_argument("--q_index", default=-1, type=int, help="Test one specific q_index only (default: %(default)s)")

    args = parser.parse_args(argv)
    return args


def read_image_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0) / 255
    return input_image


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    Image.fromarray(img.squeeze()).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr.item()


def run_test(video_net, args, device):
    frame_num_eval = args['frame_num']
    gop_size = args['gop_size']
    num_stages_tmp = 1
    while 2**num_stages_tmp < gop_size:
        num_stages_tmp += 1
    assert 2**num_stages_tmp == gop_size

    if frame_num_eval % gop_size > 0:
        frame_num = frame_num_eval + (gop_size - frame_num_eval % gop_size)
    else:
        frame_num = frame_num_eval
    if args["last_frames"]:
        start_frame = args['frame_num_seq'] - frame_num  # count from 0
    else:
         start_frame = 0

    assert frame_num % gop_size == 0
    gop_num = frame_num // gop_size
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0

    print("CODING ", args['vid_path'])

    src_reader = YUVReader(args['vid_path'], args['src_width'], args['src_height'], start_index=start_frame)

    frame_types = [None] * frame_num
    psnrs = [None] * frame_num
    msssims = [None] * frame_num
    rgb_psnrs = [None] * frame_num
    bits = [None] * frame_num
    bpps = [None] * frame_num
    bpp_mv = [None] * frame_num
    frame_pixel_num = 0

    start_time = time.time()
    p_frame_number = 0
    overall_p_decoding_time = 0
    overall_p_encoding_time = 0

    q_index = args['q_idx']
    with torch.no_grad():
        for gop_idx in range(gop_num):
            print(f"CODING GOP {gop_idx+1}")
            frames_coded = [None] * gop_size
            frames_orig = [None] * gop_size
            num_frames = gop_size
            # CODE GOP (forward MCTF + spatial wavelet)
            hp_last_stage = 0
            for stage_idx in range(num_stages_tmp):
                num_frames = num_frames // 2
                hp_bpp_cur_stage = []
                dpb = {"mv_feature": None, "ref_mv_y": None}
                for group_idx in range(num_frames):
                    group_step = 2**stage_idx
                    frame_idx_gop = group_idx*2*group_step

                    frame_idx = gop_idx*gop_size + frame_idx_gop
                    print(f"Coding Frame {frame_idx} and {frame_idx+group_step} in Stage {stage_idx}")

                    if stage_idx == 0:
                        # first temporal decomposition level: read original frames
                        ycbcr_ref = src_reader.read_one_frame()
                        ycbcr_cur = src_reader.read_one_frame()
                        ycbcr_ref = [np_image_to_tensor(curr) for curr in ycbcr_ref]
                        ycbcr_cur = [np_image_to_tensor(curr)for curr in ycbcr_cur]
                        y_ref, cb_ref, cr_ref = ycbcr_ref
                        chroma_ref = torch.cat((cb_ref, cr_ref), dim=0)

                        y_cur, cb_cur, cr_cur = ycbcr_cur
                        chroma_cur = torch.cat((cb_cur, cr_cur), dim=0)

                        y_ref = y_ref.unsqueeze(0).to(device)
                        y_cur = y_cur.unsqueeze(0).to(device)
                        chroma_ref = chroma_ref.unsqueeze(1).to(device)
                        chroma_cur = chroma_cur.unsqueeze(1).to(device)

                        frames_orig[frame_idx_gop] = [y_ref, chroma_ref]
                        frames_orig[frame_idx_gop+group_step] = [y_cur, chroma_cur]

                        pic_height = y_ref.shape[2]
                        pic_width = y_ref.shape[3]
                        if frame_pixel_num == 0:
                            frame_pixel_num = pic_height * pic_width
                        else:
                            assert frame_pixel_num == pic_height * pic_width

                        psize = 128

                        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, p=psize)

                        frame_start_time = time.time()
                        # pad if necessary
                        y_ref_p = torch.nn.functional.pad(y_ref, (padding_l, padding_r, padding_t, padding_b),
                                                          mode="constant", value=0)
                        y_cur_p = torch.nn.functional.pad(y_cur, (padding_l, padding_r, padding_t, padding_b),
                                                          mode="constant", value=0)

                        chroma_ref_p = torch.nn.functional.pad(chroma_ref, (
                        padding_l // 2, padding_r // 2, padding_t // 2, padding_b // 2),
                                                               mode="constant", value=0)
                        chroma_cur_p = torch.nn.functional.pad(chroma_cur, (
                        padding_l // 2, padding_r // 2, padding_t // 2, padding_b // 2),
                                                               mode="constant", value=0)

                    else:
                        # Frames in frames_coded are already padded
                        y_ref_p, chroma_ref_p, mv_ref = frames_coded[frame_idx_gop]
                        y_cur_p, chroma_cur_p, mv_cur = frames_coded[frame_idx_gop+group_step]
                        # further decompose lowpass frames with no associated MVs
                        assert mv_ref is None
                        assert mv_cur is None

                    bin_path = os.path.join(args['bin_folder'], f"{frame_idx_gop+group_step}.bin") \
                        if write_stream else None

                    code_lt = (stage_idx + 1) == num_stages_tmp

                    me_num = min(video_net.num_me_stages-1, stage_idx)

                    y_ref_p = y_ref_p.to(device)
                    y_cur_p = y_cur_p.to(device)
                    chroma_ref_p = chroma_ref_p.to(device)
                    chroma_cur_p = chroma_cur_p.to(device)

                    result = video_net.encode_one_stage(ref_frame=[y_ref_p, chroma_ref_p],
                                                        cur_frame=[y_cur_p, chroma_cur_p],
                                                        output_path=bin_path,
                                                        pic_height=pic_height, pic_width=pic_width,
                                                        stage_idx=me_num,
                                                        code_lt=code_lt,
                                                        psize=psize,
                                                        skip_decoding=args["skip_decoding"],
                                                        dpb=dpb,
                                                        q_index=q_index)

                    frames_coded[frame_idx_gop] = [result["L_t"], result["L_tc"], None]
                    frames_coded[frame_idx_gop+group_step] = [result["H_t"], result["H_tc"], result["mv_hat"]]

                    dpb= result["dpb"]

                    frame_end_time = time.time()

                    frame_types[frame_idx+group_step] = 1
                    p_frame_number += 1
                    overall_p_decoding_time += result['decoding_time']
                    overall_p_encoding_time += result['encoding_time']
                    curr_bits = result["bit_H"] + result["bit_ME"]
                    if isinstance(curr_bits, torch.Tensor):
                        curr_bits = curr_bits.item()
                    tmp = result["bit_ME"] / curr_bits
                    print(f"percentage MV: {tmp*100} %")

                    bpps[frame_idx+group_step] = curr_bits / frame_pixel_num
                    bits[frame_idx+group_step] = curr_bits
                    if isinstance(result["bit_ME"], torch.Tensor):
                        result["bit_ME"] = result["bit_ME"].item()
                    bpp_mv[frame_idx+group_step] = result["bit_ME"] / frame_pixel_num

                    print(f"Frame {frame_idx+group_step}: {bpps[frame_idx+group_step] } bpp")
                    hp_bpp_cur_stage.append(bpps[frame_idx+group_step])

                    if code_lt:
                        # last frame in current GOP
                        frame_types[frame_idx] = 0
                        curr_bits = result["bit_L"]
                        if isinstance(curr_bits, torch.Tensor):
                            curr_bits = curr_bits.item()
                        bpps[frame_idx] = curr_bits / frame_pixel_num
                        bits[frame_idx] = curr_bits
                        bpp_mv[frame_idx] = 0
                # stage completed
                print(f"STAGE {stage_idx} completed")
                if hp_last_stage != 0:
                    hp_cur_stage = np.mean(hp_bpp_cur_stage)
                    print(f"Average bpp HP {hp_cur_stage} current stage and {hp_last_stage}  prev stage ")
                hp_last_stage = np.mean(hp_bpp_cur_stage)

            # TEMPORAL DECODING
            for stage_idx in reversed(range(num_stages_tmp)):
                if stage_idx == num_stages_tmp-1:
                    num_frames = 1
                else:
                    num_frames = num_frames*2
                for group_idx in reversed(range(num_frames)):
                    group_step = 2**stage_idx
                    frame_idx_gop = group_idx*2*group_step

                    frame_idx = gop_idx*gop_size + frame_idx_gop
                    print(f"Temporal Decoding {frame_idx} and {frame_idx + group_step} in Stage {stage_idx}")

                    L_t, L_tc, mv_ref = frames_coded[frame_idx_gop]
                    H_t, H_tc, mv_hat = frames_coded[frame_idx_gop + group_step]

                    assert mv_ref is None

                    me_num = min(video_net.num_me_stages-1, stage_idx)

                    ref_frame, cur_frame = video_net.inverse_MCTF(L_t, H_t, mv_hat, stage_idx=me_num)
                    ref_frame_c, cur_frame_c = video_net.inverse_MCTF(L_tc, H_tc, mv_hat, stage_idx=me_num,
                                                                      downscale=True)
                    frames_coded[frame_idx_gop] = [ref_frame, ref_frame_c, None]
                    frames_coded[frame_idx_gop + group_step] = [cur_frame, cur_frame_c, None]

            # CALCULATE PSNR, MS-SSIM
            for frame_idx_gop in range(gop_size):
                frame_idx = gop_idx * gop_size + frame_idx_gop

                cur_frame, cur_frame_c, mv_ref = frames_coded[frame_idx_gop]
                y_cur, chroma_cur = frames_orig[frame_idx_gop]

                assert mv_ref is None
                cur_frame_rec = torch.round(cur_frame.clamp_(0, 255.0))
                cur_frame_c = torch.round(cur_frame_c.clamp_(0, 255.0))

                y_hat_cur = F.pad(cur_frame_rec, (-padding_l, -padding_r, -padding_t, -padding_b))
                y_psnr_cur = PSNR(y_hat_cur, y_cur)

                c_hat_cur = F.pad(cur_frame_c, (-padding_l // 2, -padding_r // 2, -padding_t // 2, -padding_b // 2))

                cb_psnr_cur = PSNR(c_hat_cur[0:0 + 1, :, :, :], chroma_cur[0:0 + 1, :, :, :])
                cr_psnr_cur = PSNR(c_hat_cur[1:1 + 1, :, :, :], chroma_cur[1:1 + 1, :, :, :])

                # get RGB-psnr
                ycbcr_444_hat = yuv_420_to_444((y_hat_cur, *c_hat_cur.chunk(2, 0)))
                ycbcr_444_orig = yuv_420_to_444((y_cur, *chroma_cur.chunk(2, 0)))

                x_rgb = torch.round(ycbcr2rgb(ycbcr_444_orig))
                x_hat_rgb = torch.round(ycbcr2rgb(ycbcr_444_hat))
                rgb_psnr = PSNR(x_rgb, x_hat_rgb)

                if pic_height > 128 and pic_width > 128:
                    msssim = ms_ssim(x_hat_rgb, x_rgb, data_range=255.0).item()
                else:
                    msssim = 0

                psnrs[frame_idx] = (6.0 * y_psnr_cur + cb_psnr_cur + cr_psnr_cur) / 8.0
                rgb_psnrs[frame_idx] = rgb_psnr
                msssims[frame_idx] = msssim

                if verbose >= 2:
                    print(f"frame {frame_idx} (SeqIdx {start_frame + frame_idx}), {frame_end_time - frame_start_time:.3f} seconds,",
                          f"bpps: {bpps[frame_idx] :.3f}, YUV-PSNR: {psnrs[frame_idx ] :.4f}, RGB-PSNR: {rgb_psnrs[frame_idx] :.4f},"
                          f"MS-SSIM: {msssims[frame_idx ] :.4f}, Y-PSNR: {y_psnr_cur :.4f},  Cb-PSNR: {cb_psnr_cur :.4f}, Cr-PSNR: {cr_psnr_cur :.4f}  ")

                if save_decoded_frame:
                    save_path = os.path.join(args['decoded_frame_folder'], f'{start_frame + frame_idx}.png')
                    save_torch_image(x_hat_rgb, save_path)

    test_time = time.time() - start_time
    if verbose >= 1 and p_frame_number > 0:
        print(f"decoding {p_frame_number} P frames, "
              f"average {overall_p_decoding_time/p_frame_number * 1000:.0f} ms.")
        print(f"encoding {p_frame_number} P frames, "
              f"average {overall_p_encoding_time/p_frame_number * 1000:.0f} ms.")

    pad_frame_num = frame_num - frame_num_eval
    if pad_frame_num > 0:
        print(f"Considering {frame_num_eval} of {frame_num} coded frames in evaluation")
        frame_types = frame_types[pad_frame_num:]
        bits = bits[pad_frame_num:]
        bpp_mv = bpp_mv[pad_frame_num:]
        psnrs = psnrs[pad_frame_num:]
        rgb_psnrs = rgb_psnrs[pad_frame_num:]
        msssims = msssims[pad_frame_num:]

    log_result = generate_log_json(frame_num_eval, frame_types, bits, bpp_mv, psnrs, rgb_psnrs, msssims,
                                   frame_pixel_num, test_time)
    return log_result


def encode_one(args, device):
    p_state_dict = get_state_dict(args['model_path'])

    video_net = pMCTF(lossy=not args["lossless"], num_me_stages=args["num_me_stages"])

    video_net.load_state_dict(p_state_dict, strict=True)
    video_net = video_net.to(device)
    video_net.eval()

    if args['write_stream']:
        if video_net is not None:
            video_net.update(force=True)

    sub_dir_name = args['video_path']
    gop_size = args['gop']
    frame_num = args['frame_num']

    bin_folder = os.path.join(args['stream_path'], sub_dir_name)
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args['decoded_frame_path'], sub_dir_name)
        create_folder(decoded_frame_folder)
    else:
        decoded_frame_folder = None
    img_path = os.path.join(args['dataset_path'], sub_dir_name)
    args['vid_path'] = img_path + ".yuv"
    args['gop_size'] = gop_size
    args['frame_num'] = frame_num
    args['bin_folder'] = bin_folder
    args['decoded_frame_folder'] = decoded_frame_folder

    result = run_test(video_net, args, device=device)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']

    return result


def worker(use_cuda, args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(seed=0)
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"
    result = encode_one(args, device)
    return result


def main(argv):
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args(argv)

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    with open(args.test_config) as f:
        config = json.load(f)

    count_frames = 0
    count_sequences = 0

    q_index_num = pMCTF.get_qp_num()

    res_summary = {"bpp": [], "psnr-rgb": [], "psnr-yuv": [], "quality": [], "ms-ssim-rgb": [], "bpp_mv": []}
    if args.q_index >= 0:
        assert args.q_index < q_index_num
        q_index = [args.q_index]
    else:
        q_index_n = args.q_index_num
        q_index_max = q_index_num - 1
        step = int(q_index_max/q_index_n) + 1
        if (q_index_n-1)*step > q_index_max:
            step = step - 1
        q_index = [x*step for x in range(q_index_n)]
        q_index[-1] = q_index_max
    print(f"Testing q_index list: {q_index}")

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    if args.ds_name is not None:
        config = {args.ds_name: config[args.ds_name]}
        if args.seq_num >= 0:
            tmp = list(config[args.ds_name]['sequences'].items())[args.seq_num]
            config[args.ds_name]['sequences'] = {tmp[0]: tmp[1]}

    output_paths = []
    for q_idx in q_index:
        results = []
        for ds_name in config:
            if config[ds_name]['test'] == 0:
                continue
            for seq_name in config[ds_name]['sequences']:
                count_sequences += 1
                cur_args = {}
                if not args.force_intra:
                    cur_args['model_path'] = args.model_path
                    cur_args['num_me_stages'] = args.num_me_stages
                cur_args['q_idx'] = q_idx
                cur_args['force_intra'] = args.force_intra
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                if args.force_intra:
                    cur_args['gop'] = 1
                if args.force_intra_period > 0:
                    cur_args['gop'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                cur_args['frame_num_seq'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}_MCTF'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose

                cur_args['lossless'] = args.lossless
                cur_args['q_index'] = args.q_index
                cur_args["q_index_num"] = args.q_index_num
                cur_args["two_stage_me"] = args.two_stage_me
                cur_args["skip_decoding"] = args.skip_decoding
                cur_args["last_frames"] = args.last_frames

                count_frames += cur_args['frame_num']

                results.append(worker(args.cuda, cur_args))

        log_result = {}
        for ds_name in config:
            if config[ds_name]['test'] == 0:
                continue
            log_result[ds_name] = {}
            for seq in config[ds_name]['sequences']:
                log_result[ds_name][seq] = {}
                for res in results:
                    if ds_name == res['ds_name'] and seq == res['video_path']:
                        log_result[ds_name][seq] = res
        for ds_name in log_result.keys():

            res = {"test_time": [], "ave_p_frame_bpp": [],
                   "ave_p_frame_psnr": [], "ave_p_frame_psnr_rgb": [], "ave_p_frame_msssim": [],
                   "ave_all_frame_bpp": [], "ave_all_frame_bpp_mv": [],  "ave_all_frame_psnr": [],
                   "ave_all_frame_psnr_rgb": [], "ave_all_frame_msssim": []}

            for seq_name in log_result[ds_name].keys():
                for metrickey in res.keys():
                    res[metrickey].append(
                        log_result[ds_name][seq_name][metrickey])
            for metrickey in res.keys():
                res[metrickey] = np.mean(res[metrickey])
            log_result[ds_name]["AVERAGE"] = res
            if len(q_index) > 1:
                res_summary["bpp"].append(res["ave_all_frame_bpp"])
                res_summary["bpp_mv"].append(res["ave_all_frame_bpp_mv"])
                res_summary["psnr-rgb"].append(res["ave_all_frame_psnr_rgb"])
                res_summary["psnr-yuv"].append(res["ave_all_frame_psnr"])
                res_summary["ms-ssim-rgb"].append(res["ave_all_frame_msssim"])
                res_summary["quality"].append(str(q_idx))

        if args.ds_name is not None:
            output_path = f"results/{args.ds_name}/output_"
            if args.seq_num >= 0:
                # sequence name
                output_path += list(config[args.ds_name]["sequences"].keys())[0]
            # quality
            tmp = args.model_path.split('/')
            exp_name = tmp[-2]
            chkpt_name = tmp[-1].replace(".pth.tar", "")
            res_summary["name"] = chkpt_name
            output_path += f"{exp_name}_{chkpt_name}"
            if args.force_intra_period > 0:
                output_path += f"_GOP{args.force_intra_period}"
                res_summary["name"] += f" GOP{args.force_intra_period}"
            summary_path = output_path
            output_path += f"_qidx{q_idx}"
            curr_time = time.strftime('%Y%m%d_%H%M', time.localtime())
            output_path += f"_{curr_time}.json"
        else:
            output_path = args.output_path
            summary_path = args.output_path
        out_json_dir = os.path.dirname(output_path)
        if len(out_json_dir) > 0:
            create_folder(out_json_dir, True)
        output_paths.append(output_path)
        with open(output_path, 'w') as fp:
            dump_json(log_result, fp, float_digits=6, indent=2)

        total_minutes = (time.time() - begin_time) / 60
        print('Test finished')
        print(f'Tested {count_frames} frames from {count_sequences} sequences')
        print(f'Total elapsed time: {total_minutes:.1f} min')

    with open(f"{summary_path}_summary.json", 'w') as fp:
        dump_json(res_summary, fp, float_digits=6, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
