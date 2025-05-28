import math
import os
import time

import torch.nn as nn
import torch
import os.path as osp

from copy import deepcopy
from timm.models.layers import trunc_normal_

from pMCTF.layers import RoundNoGradient

from pMCTF.layers.video.video_net import ME_Spynet, flow_warp, bilineardownsacling, \
    get_hyper_enc_model, get_hyper_dec_model, bilinearupsacling, MvDec, MvEnc
from pMCTF.layers.video.wavelet_transform_temporal_mctf import TemporalLifting

from pMCTF.utils.stream_helper import get_downsampled_shape, encode_p, decode_p, \
    get_rounded_q

from pMCTF.entropy_models.entropy_models import BitEstimator
from pMCTF.entropy_models.gaussian_model import CompressionModel
from pMCTF.layers.video.four_part_prior import MVCoderQuad
from pMCTF.layers.video.layers import DepthConvBlock

from pMCTF.models.pWave import pWave


class pMCTF(nn.Module):
    """
        Variable Rate Wavelet-Video Coder based on Motion Compensated Temporal Filtering (MCTF)
        with temporal layer adaptivity
    """

    def __init__(self,
                 bitdepth=8,
                 decomp_levels=4,
                 lossy=True,
                 two_stage_me=True,
                 num_me_stages=2,
                 quant_stage=True,
                 **kwargs):

        super(pMCTF, self).__init__()
        self.bitdepth = bitdepth
        self.dynamic_range = 2**bitdepth-1
        self.lossy = lossy

        self.lp_coder = pWave(bitdepth, decomp_levels, lossy)
        self.hp_coder = pWave(bitdepth, decomp_levels, lossy)
        self.mse = nn.MSELoss(reduction='mean')

        # VIDEO ---------------------------------------------------
        channel_mv = 64
        channel_N = 64
        channel_M = 32

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        # MOTION ESTIMATION
        self.optic_flow = ME_Spynet(L=6)

        # MV VECTOR CODING
        self.mv_encoder = nn.ModuleList([MvEnc(2, channel_mv)  # with DepthConvBlock
                                         for _ in range(num_me_stages)])
        self.mv_decoder = nn.ModuleList([MvDec(2, channel_mv)
                                         for _ in range(num_me_stages)])

        self.mv_hyper_prior_encoder = nn.ModuleList([get_hyper_enc_model(channel_N, channel_mv)
                                                     for _ in range(num_me_stages)])
        self.mv_hyper_prior_decoder = nn.ModuleList([get_hyper_dec_model(channel_N, channel_mv)
                                                     for _ in range(num_me_stages)])

        self.mv_y_prior_fusion_adaptor_0 = nn.ModuleList([DepthConvBlock(channel_mv * 1, channel_mv * 2)
                                                          for _ in range(num_me_stages)])
        self.mv_y_prior_fusion_adaptor_1 = nn.ModuleList([DepthConvBlock(channel_mv * 2, channel_mv * 2)
                                                          for _ in range(num_me_stages)])

        self.mv_y_prior_fusion = nn.ModuleList([nn.Sequential(
            DepthConvBlock(channel_mv * 2, channel_mv * 3),
            DepthConvBlock(channel_mv * 3, channel_mv * 3),
        ) for _ in range(num_me_stages)])
        self.mv_y_spatial_prior = nn.ModuleList([nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3),
            DepthConvBlock(channel_mv * 3, channel_mv * 3),
            DepthConvBlock(channel_mv * 3, channel_mv * 2),
        ) for _ in range(num_me_stages)])

        self.mv_y_spatial_prior_adaptor_1 = nn.ModuleList([nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
                                                           for _ in range(num_me_stages)])
        self.mv_y_spatial_prior_adaptor_2 = nn.ModuleList([nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
                                                           for _ in range(num_me_stages)])
        self.mv_y_spatial_prior_adaptor_3 = nn.ModuleList([nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
                                                           for _ in range(num_me_stages)])

        self.mv_y_q_scale_enc = nn.ParameterList([nn.Parameter(torch.ones((2, 1, 1, 1)))
                                                  for _ in range(num_me_stages)])
        self.mv_y_q_scale_dec = nn.ParameterList([nn.Parameter(torch.ones((2, 1, 1, 1)))
                                                  for _ in range(num_me_stages)])

        self.mv_bit_est = nn.ModuleList([BitEstimator(channel_mv) for _ in range(num_me_stages)])
        self.em = CompressionModel(y_distribution="laplace")
        self.mv_coder = MVCoderQuad(enc_dec_quant=True)

        # MCTF
        self.temporal_filtering = nn.ModuleList([TemporalLifting(lossy=self.lossy) for _ in range(num_me_stages)])
        self.quant_stage = quant_stage
        if self.quant_stage:
            self.hp_q_scale = nn.ParameterList([nn.Parameter(torch.ones((2, 1, 1, 1))) for _ in range(num_me_stages)])

        self.two_stage_me = two_stage_me
        self.num_me_stages = num_me_stages

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def freeze_inter(self):
        for k, p in self.named_parameters():
            if k.startswith(("mv_", "optic_flow", "temporal_filtering", "bit_estimator_z_mv")):
                # Spynet, MV enc/dec and MV hyperprior enc/dec
                p.requires_grad = False
            else:
                p.requires_grad = True

    def make_mctf_trainable(self, start_idx=1, copy_idx=0):
        """ Training stage 4: Add mutliple temporal wavelet transform modules for every temporal decomposition levels.
        """
        for i in range(start_idx, self.num_me_stages):
            self.mv_encoder[i].load_state_dict(self.mv_encoder[copy_idx].state_dict())
            self.mv_decoder[i].load_state_dict(self.mv_decoder[copy_idx].state_dict())

            self.mv_hyper_prior_encoder[i].load_state_dict(self.mv_hyper_prior_encoder[copy_idx].state_dict())
            self.mv_hyper_prior_decoder[i].load_state_dict(self.mv_hyper_prior_decoder[copy_idx].state_dict())

            self.mv_y_prior_fusion_adaptor_0[i].load_state_dict(self.mv_y_prior_fusion_adaptor_0[copy_idx].state_dict())
            self.mv_y_prior_fusion_adaptor_1[i].load_state_dict(self.mv_y_prior_fusion_adaptor_1[copy_idx].state_dict())
            self.mv_y_prior_fusion[i].load_state_dict(self.mv_y_prior_fusion[copy_idx].state_dict())
            self.mv_y_spatial_prior[i].load_state_dict(self.mv_y_spatial_prior[copy_idx].state_dict())

            self.mv_y_spatial_prior_adaptor_1[i].load_state_dict(self.mv_y_spatial_prior_adaptor_1[copy_idx].state_dict())
            self.mv_y_spatial_prior_adaptor_2[i].load_state_dict(self.mv_y_spatial_prior_adaptor_2[copy_idx].state_dict())
            self.mv_y_spatial_prior_adaptor_3[i].load_state_dict(self.mv_y_spatial_prior_adaptor_3[copy_idx].state_dict())

            self.mv_y_q_scale_enc[i] = deepcopy(self.mv_y_q_scale_enc[copy_idx])
            self.mv_y_q_scale_dec[i] = deepcopy(self.mv_y_q_scale_dec[copy_idx])

            self.mv_bit_est[i].load_state_dict(self.mv_bit_est[copy_idx].state_dict())
            self.temporal_filtering[i].load_state_dict(self.temporal_filtering[copy_idx].state_dict())

        for k, p in self.named_parameters():
            # p.requires_grad = True
            if k.startswith(("mv_", "temporal_filtering")):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def make_inter_trainable(self):
        # all parameters related to motion estimation only
        for k, p in self.named_parameters():
            if k.startswith(("temporal_filtering", "mv")):
                # Spynet, MV enc/dec and MV hyperprior enc/dec
                p.requires_grad = True
            else:
                p.requires_grad = False

    def make_all_trainable(self):
        for k, p in self.named_parameters():
            if not k.startswith("optic_flow"):
                p.requires_grad = True

    def make_optic_flow_trainable(self):
        for k, p in self.named_parameters():
            if k.startswith("optic_flow"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def fix_wavelet_transform(self):
        for p in self.hp_coder.wavelet_transform.parameters():
            p.requires_grad = False
        for p in self.lp_coder.wavelet_transform.parameters():
            p.requires_grad = False

    def motion_compensation(self, ref_frame, mv):
        warpframe = flow_warp(ref_frame, mv)
        return warpframe

    def get_one_q_scale(self, q_scale, q_index):
        min_q = q_scale[0:1, :, :, :]
        max_q = q_scale[1:2, :, :, :]
        step = (torch.log(max_q) - torch.log(min_q)) / (self.get_qp_num() - 1)
        q = torch.exp(torch.log(min_q) + step * q_index)
        return q

    def get_curr_q(self, q_scale, q_index):
        if isinstance(q_index, list):
            q_step = [self.get_one_q_scale(q_scale, i) for i in q_index]
            q_step = torch.cat(q_step, dim=0)
        else:
            q_step = self.get_one_q_scale(q_scale, q_index)

        return q_step

    @staticmethod
    def get_index_tensor(q_index, device):
        if not isinstance(q_index, list):
            q_index = [q_index]
        return torch.tensor(q_index, dtype=torch.int32, device=device)

    @staticmethod
    def get_qp_num():
        return 21  # 64

    def get_mv_y_q(self, q_index, stage_idx=0, inference=False):
        mv_y_q_scale_enc = self.mv_y_q_scale_enc[stage_idx]
        mv_y_q_enc = self.get_curr_q(mv_y_q_scale_enc, q_index)
        if inference:
            mv_y_q_enc, _ = get_rounded_q(mv_y_q_enc.cpu())
        mv_y_q_scale_dec = self.mv_y_q_scale_dec[stage_idx]
        mv_y_q_dec = self.get_curr_q(mv_y_q_scale_dec, q_index)
        if inference:
            mv_y_q_dec, _ = get_rounded_q(mv_y_q_dec.cpu())
        return mv_y_q_enc, mv_y_q_dec

    def mv_prior_param_decoder(self, mv_z_hat, dpb, me_num):
        mv_params = self.mv_hyper_prior_decoder[me_num](mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            mv_params = self.mv_y_prior_fusion_adaptor_0[me_num](mv_params)
        else:
            mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
            mv_params = self.mv_y_prior_fusion_adaptor_1[me_num](mv_params)
        mv_params = self.mv_y_prior_fusion[me_num](mv_params)
        return mv_params

    def compute_and_code_motion(self, ref_frame, cur_frame, q_index, dpb, stage_idx=0, me_downsample=1):
        me_num = min(self.num_me_stages-1, stage_idx)
        mv_y_q_enc, mv_y_q_dec = self.get_mv_y_q(q_index, me_num)
        # curr_mv_y_q = self.get_curr_mv_y_q(self.mv_y_q_scale[me_num], me_num)
        if self.training and cur_frame.size(0) != 3:
            mv_cur = cur_frame.tile((1, 3, 1, 1)) / self.dynamic_range
            mv_ref = ref_frame.tile((1, 3, 1, 1)) / self.dynamic_range
        else:
            # estimate motion on Y only
            mv_cur = cur_frame[0, :, :, :].tile((1, 3, 1, 1)) / self.dynamic_range
            mv_ref = ref_frame[0, :, :, :].tile((1, 3, 1, 1)) / self.dynamic_range

        if me_downsample > 1:
            mv_cur = bilineardownsacling(mv_cur, factor=me_downsample)
            mv_ref = bilineardownsacling(mv_ref, factor=me_downsample)

        # ESTIMATE AND ENCODE+DECODE MOTION
        est_mv = self.optic_flow(mv_cur, mv_ref)  # MOTION ESTIMATION
        mv_y = self.mv_encoder[me_num](est_mv, dpb["mv_feature"], mv_y_q_enc)

        mv_z = self.mv_hyper_prior_encoder[me_num](mv_y)
        mv_z_hat = self.mv_coder.quant(mv_z)

        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, me_num)

        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.mv_coder.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1[me_num], self.mv_y_spatial_prior_adaptor_2[me_num],
            self.mv_y_spatial_prior_adaptor_3[me_num], self.mv_y_spatial_prior[me_num])

        mv_hat, mv_feature  = self.mv_decoder[me_num](mv_y_hat, mv_y_q_dec)

        if me_downsample > 1:
            mv_hat = bilinearupsacling(mv_hat, factor=me_downsample) * me_downsample

        if self.training:
            mv_y_for_bit = self.em.add_noise(mv_y_res)
            mv_z_for_bit = self.em.add_noise(mv_z)
        else:
            mv_y_for_bit = mv_y_q
            mv_z_for_bit = mv_z_hat
        bits_mv_y = self.em.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_mv_z = self.em.get_z_bits(mv_z_for_bit, self.mv_bit_est[me_num])

        pixel_num = ref_frame.size(2) * ref_frame.size(3)
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp_mv_y = torch.mean(bpp_mv_y) if self.training else torch.sum(bpp_mv_y)
        bpp_mv_z = torch.mean(bpp_mv_z) if self.training else torch.sum(bpp_mv_z)
        return mv_hat, {"mv_feature": mv_feature, "mv_y_hat": mv_y_hat}, bpp_mv_y, bpp_mv_z

    def forward(self, ref_frame, cur_frame, q_index, code_lt, dpb, stage_idx=0):
        return self.forward_one_stage(ref_frame, cur_frame, q_index, code_lt, dpb, stage_idx=stage_idx)

    def forward_MCTF(self, ref_frame, cur_frame, mv_hat, stage_idx=0):
        me_num = min(self.num_me_stages - 1, stage_idx)
        if ref_frame.size(0) > mv_hat.size(0):
            mv_hat = mv_hat.tile((ref_frame.size(0), 1, 1, 1))
        pred_frame = self.motion_compensation(ref_frame, mv_hat)
        if not self.lossy:
            pred_frame = RoundNoGradient.apply(pred_frame)
        pred_frame = self.temporal_filtering[me_num].predict_filter(pred_frame)
        H_t = cur_frame - pred_frame

        inv_pred_frame = self.motion_compensation(H_t, -mv_hat)
        if not self.lossy:
            inv_pred_frame = RoundNoGradient.apply(inv_pred_frame)
        inv_pred_frame = self.temporal_filtering[me_num].update_filter(inv_pred_frame)
        L_t = ref_frame + inv_pred_frame
        return L_t, H_t, pred_frame, inv_pred_frame

    def inverse_MCTF(self, L_t, H_t, mv_hat, downscale=False, stage_idx=0):
        me_num = min(self.num_me_stages - 1, stage_idx)
        if downscale:
            mv_hat = bilineardownsacling(mv_hat) / 2
        if L_t.size(0) > mv_hat.size(0):
            mv_hat = mv_hat.tile((L_t.size(0), 1, 1, 1))
        inv_pred_frame = self.motion_compensation(H_t, -mv_hat)  # MC-1
        if not self.lossy:
            inv_pred_frame = RoundNoGradient.apply(inv_pred_frame)
        inv_pred_frame = self.temporal_filtering[me_num].update_filter(inv_pred_frame)
        ref_frame = L_t - inv_pred_frame
        pred_frame = self.motion_compensation(ref_frame, mv_hat)
        if not self.lossy:
            pred_frame = RoundNoGradient.apply(pred_frame)
        pred_frame = self.temporal_filtering[me_num].predict_filter(pred_frame)
        cur_frame = H_t + pred_frame
        return ref_frame, cur_frame

    def forward_one_stage(self, ref_frame, cur_frame, q_index, code_lt, dpb, mv_hat=None, stage_idx=0, me_downsample=1):
        if mv_hat is not None:
            bpp_mv_y, bpp_mv_z = None, None
            ref_mv = {"mv_feature": None, "mv_y_hat": None}
            mv_hat = bilineardownsacling(mv_hat) / 2
        else:
            mv_hat, ref_mv, bpp_mv_y, bpp_mv_z = self.compute_and_code_motion(ref_frame, cur_frame, q_index, dpb,
                                                                      stage_idx=stage_idx, me_downsample=me_downsample)

        L_t, H_t, pred_frame, inv_pred_frame = self.forward_MCTF(ref_frame, cur_frame, mv_hat, stage_idx)

        if self.quant_stage:
            hp_q_scale = self.hp_q_scale[stage_idx]
            qp_scale = self.get_curr_q(hp_q_scale, q_index)
        else:
            qp_scale = None

        res_H = self.hp_coder.forward(H_t, q_index, qp_scale=qp_scale)

        me_mse = self.mse(pred_frame, cur_frame)

        ret_dict = {"bpp_mv_y": bpp_mv_y,
                    "bpp_mv_z": bpp_mv_z,
                    "bpp_me": bpp_mv_z + bpp_mv_y if bpp_mv_z is not None else None,
                    "me_mse": me_mse,
                    "bpp": res_H["bpp_total"] + bpp_mv_z + bpp_mv_y if bpp_mv_z is not None else res_H["bpp_total"] ,
                    "bpp_H": res_H["bpp_total"],
                    "bit_H": res_H["bits_total"],
                    "bit_ME": (bpp_mv_y + bpp_mv_z) * (ref_frame.size(2) * ref_frame.size(3)) if bpp_mv_z is not None else None,
                    "mse_H": res_H["mse"],
                    "mv_hat": mv_hat,
                    "dpb": { "mv_feature": ref_mv["mv_feature"], "ref_mv_y": ref_mv["mv_y_hat"]} ,
                    "H_t": res_H["x_hat"],
                    }
        if code_lt:
            res_L = self.lp_coder.forward(L_t, q_index)

            ret_dict["bpp"] = ret_dict["bpp"]
            ret_dict["bpp_L"] = res_L["bpp_total"]
            ret_dict["bit_L"] = res_L["bits_total"]
            ret_dict["mse_L"] = res_L["mse"]
            ret_dict["me_mse_inv"] = self.mse(inv_pred_frame, ref_frame)
            ret_dict["L_t"] = res_L["x_hat"]
        else:
            ret_dict["L_t"] = L_t

        ret_dict["bit"] = ret_dict["bpp"] * (ref_frame.size(2) * ref_frame.size(3))
        return ret_dict

    def load_from_iframe(self, i_frame_state_dict):
        if self.lp_coder.QP.size(0) != i_frame_state_dict["QP"].size(0):
            qp_size = self.lp_coder.QP.size(0)
            for i in range(qp_size):
                self.lp_coder.QP[i].copy_(i_frame_state_dict["QP"][-1].clone())
                self.hp_coder.QP[i].copy_(i_frame_state_dict["QP"][-1].clone())
                self.lp_coder.QP_ll[i].copy_(i_frame_state_dict["QP_ll"].clone())
                self.hp_coder.QP_ll[i].copy_(i_frame_state_dict["QP_ll"].clone())
            del i_frame_state_dict["QP"]
            del i_frame_state_dict["QP_ll"]
            load_strict = False
        else:
            load_strict = True
        self.lp_coder.load_state_dict(i_frame_state_dict, strict=load_strict)
        self.hp_coder.load_state_dict(i_frame_state_dict, strict=load_strict)

    @torch.no_grad()
    def compress_one_stage(self, ref_frame, cur_frame, code_lt, mv_hat, ischroma,
                           sideinfo=None, file_name=None, stage_idx=0, q_index=0, skip_decoding=False):
        if ischroma:
            mv_hat = bilineardownsacling(mv_hat) / 2
        L_t, H_t, pred_frame, inv_pred_frame = self.forward_MCTF(ref_frame, cur_frame, mv_hat, stage_idx)

        if self.quant_stage:
            hp_q_scale = self.hp_q_scale[stage_idx]
            qp_scale = self.get_curr_q(hp_q_scale, q_index)
        else:
            qp_scale = None

        H_t_hat = self.hp_coder.compress(H_t, sideinfo, file_name, q_index=q_index,
                                         skip_decoding=skip_decoding, qp_scale=qp_scale)

        if code_lt:
            file_name_l = file_name.replace(osp.basename(file_name), "0_C_main.bin" if ischroma else "0_main.bin")
            L_t_hat = self.lp_coder.compress(L_t, sideinfo, file_name_l, q_index=q_index,
                                             skip_decoding=skip_decoding)
        else:
            L_t_hat = None

        return {"L_t": L_t, "H_t": H_t, "H_t_hat": H_t_hat, "L_t_hat": L_t_hat}

    @torch.no_grad()
    def decompress_one_stage(self, file_name, code_lt, ischroma, psize=128, q_index=0, stage_idx=0):
        if self.quant_stage:
            hp_q_scale = self.hp_q_scale[stage_idx]
            qp_scale = self.get_curr_q(hp_q_scale, q_index)
        else:
            qp_scale = None
        H_t = self.hp_coder.decompress(file_name, padding=psize//2 if ischroma else psize,
                                       q_index=q_index, qp_scale=qp_scale)

        if code_lt:
            file_name_l = file_name.replace(osp.basename(file_name), "0_C_main.bin" if ischroma else "0_main.bin")
            L_t = self.lp_coder.decompress(file_name_l, padding=psize//2 if ischroma else psize,
                                           q_index=q_index)
        else:
            L_t = None

        return {"L_t": L_t, "H_t": H_t}

    def update(self, force=False):
        self.em.update(force)
        for i in range(self.num_me_stages):
            self.mv_bit_est[i].update(force, entropy_coder=self.em.entropy_coder)
        self.lp_coder.update(force)
        self.hp_coder.update(force)

    def compress_mv(self, ref_frame, cur_frame, dpb, stage_idx=0, q_index=0, me_downsample=1):
        me_num = min(self.num_me_stages - 1, stage_idx)
        mv_y_q_enc, mv_y_q_dec = self.get_mv_y_q(q_index, me_num, inference=True)

        # estimate motion on Y only
        mv_x = cur_frame.tile((1, 3, 1, 1)) / self.dynamic_range
        mv_ref = ref_frame.tile((1, 3, 1, 1)) / self.dynamic_range

        if me_downsample > 1:
            mv_x = bilineardownsacling(mv_x, factor=me_downsample)
            mv_ref = bilineardownsacling(mv_ref, factor=me_downsample)

        est_mv = self.optic_flow(mv_x, mv_ref)

        mv_y = self.mv_encoder[me_num](est_mv, dpb["mv_feature"], mv_y_q_enc)
        mv_z = self.mv_hyper_prior_encoder[me_num](mv_y)
        mv_z_hat = torch.round(mv_z)

        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, me_num)

        mv_y_q_w_0, mv_y_q_w_1, mv_y_q_w_2, mv_y_q_w_3, \
        mv_scales_w_0, mv_scales_w_1, mv_scales_w_2, mv_scales_w_3, mv_y_hat = self.mv_coder.compress_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1[me_num], self.mv_y_spatial_prior_adaptor_2[me_num],
            self.mv_y_spatial_prior_adaptor_3[me_num], self.mv_y_spatial_prior[me_num])

        mv_hat, mv_feature = self.mv_decoder[me_num](mv_y_hat, mv_y_q_dec)

        if me_downsample > 1:
            mv_hat = bilinearupsacling(mv_hat, factor=me_downsample) * me_downsample

        self.em.entropy_coder.reset()
        _ = self.mv_bit_est[me_num].encode(mv_z_hat)
        _ = self.em.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        _ = self.em.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        _ = self.em.gaussian_encoder.encode(mv_y_q_w_2, mv_scales_w_2)
        _ = self.em.gaussian_encoder.encode(mv_y_q_w_3, mv_scales_w_3)

        self.em.entropy_coder.flush()

        bit_stream = self.em.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "mv_hat": mv_hat,
            "mv_feature": mv_feature,
            "mv_y_hat": mv_y_hat
        }
        return result

    def decompress_mv(self, string, dtype, height, width, dpb, stage_idx=0, q_index=0, me_downsample=1):
        me_num = min(self.num_me_stages - 1, stage_idx)
        mv_y_q_enc, mv_y_q_dec = self.get_mv_y_q(q_index, me_num, inference=True)

        self.em.entropy_coder.set_stream(string)
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.mv_bit_est[me_num].decode_stream(mv_z_size, dtype, device)
        mv_z_hat = mv_z_hat.to(device)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, me_num)

        mv_y_hat = self.mv_coder.decompress_four_part_prior(mv_params,
                                                            self.mv_y_spatial_prior_adaptor_1[me_num],
                                                            self.mv_y_spatial_prior_adaptor_2[me_num],
                                                            self.mv_y_spatial_prior_adaptor_3[me_num],
                                                            self.mv_y_spatial_prior[me_num],
                                                            gaussian_encoder=self.em.gaussian_encoder)
        mv_hat, mv_feature = self.mv_decoder[me_num](mv_y_hat, mv_y_q_dec)

        if me_downsample > 1:
            mv_hat = bilinearupsacling(mv_hat, factor=me_downsample) * me_downsample

        return {
            "mv_hat": mv_hat,
            "mv_feature": mv_feature,
            "mv_y_hat": mv_y_hat
        }

    def encode_one_stage(self, ref_frame, cur_frame, code_lt, dpb, output_path=None,
                         pic_width=None, pic_height=None, psize=128,
                         skip_decoding=False, stage_idx=0, q_index=0, me_downsample=1):
        ref_y, ref_chroma = ref_frame
        cur_y, cur_chroma = cur_frame

        if output_path is None:
            result = self.forward_one_stage(ref_y, cur_y, q_index, code_lt, dpb,stage_idx=stage_idx, me_downsample=me_downsample)
            result_c = self.forward_one_stage(ref_chroma, cur_chroma, q_index, code_lt, dpb,
                                              mv_hat=result["mv_hat"], stage_idx=stage_idx, me_downsample=me_downsample)
            ret_dict = {
                "L_t": result["L_t"],
                "H_t": result["H_t"],
                "L_tc": result_c["L_t"],
                "H_tc": result_c["H_t"],
                "bit_L": result["bit_L"] + result_c["bit_L"] if code_lt else None,
                "bit_H": result["bit_H"] + result_c["bit_H"],
                "bit_Lc": result_c["bit_L"] if code_lt else None,
                "bit_Hc": result_c["bit_H"],
                "bit_ME": result["bit_ME"],
                "mv_hat": result["mv_hat"],
                "dpb": {"mv_feature": result["mv_feature"],
                        "ref_mv_y": result["ref_mv_y"]},
                "decoding_time": 0,
                "encoding_time": 0
            }

            return ret_dict
        else:
            start = time.time()
            # mv_y_q_scale, mv_y_q_index = get_rounded_q(self.mv_y_q_scale[min(self.num_me_stages-1, stage_idx)].cpu())
            mv_y_q_index = 0

            mv_out = output_path.replace('.bin', '_mv.bin')
            out_enc = self.compress_mv(ref_y, cur_y, dpb, stage_idx=stage_idx, q_index=q_index, me_downsample=me_downsample)
            encode_p(out_enc['bit_stream'], mv_y_q_index, mv_out)

            mv_hat = out_enc["mv_hat"]
            mv_feature = out_enc["mv_feature"]
            mv_y_hat = out_enc["mv_y_hat"]

            # LUMA
            file_name = output_path

            del out_enc
            out_enc = self.compress_one_stage(ref_y, cur_y, code_lt, mv_hat, ischroma=False,
                                              sideinfo=[1, 1, pic_height, pic_width],
                                              stage_idx=stage_idx,
                                              file_name=file_name, q_index=q_index, skip_decoding=skip_decoding)

            bits_H = (os.path.getsize(file_name)) * 8.0
            bits_me = os.path.getsize(mv_out) * 8.0
            if code_lt:
                bits_L = (os.path.getsize(file_name.replace(osp.basename(file_name), "0_main.bin"))) * 8.0

            file_name_c = output_path.replace('.bin', '_C_main.bin')

            out_enc_c = self.compress_one_stage(ref_chroma.to(ref_y.device), cur_chroma.to(ref_y.device), code_lt,
                                                mv_hat,  ischroma=True,
                                                sideinfo=[1, 2, pic_height//2, pic_width//2],
                                                file_name=file_name_c,
                                                stage_idx=stage_idx, q_index=q_index, skip_decoding=skip_decoding)

            encoding_time = time.time() - start
            bits_H_c = os.path.getsize(file_name_c) * 8.0

            if code_lt:
                bits_L_c = (os.path.getsize(file_name.replace(osp.basename(file_name), "0_C_main.bin"))) * 8.0

            if not skip_decoding:
                start = time.time()
                mv_y_q_index, string = decode_p(mv_out)
                ds = 1
                decoded = self.decompress_mv(string,
                                             ref_y.dtype,
                                             ref_y.size(2)/ds, ref_y.size(3)/ds,  # mv_hat has padded size
                                             dpb,
                                             stage_idx=stage_idx, q_index=q_index)
                mv_hat = decoded["mv_hat"]
                mv_feature = decoded["mv_feature"]

                out_dec = self.decompress_one_stage(file_name, code_lt, ischroma=False, psize=psize, q_index=q_index, stage_idx=stage_idx)
                out_dec_c = self.decompress_one_stage(file_name_c, code_lt, ischroma=True, psize=psize, q_index=q_index, stage_idx=stage_idx)
                decoding_time = time.time() - start
                L_t_rec = out_dec["L_t"]["x_hat"] if code_lt else out_enc["L_t"]
                H_t_rec = out_dec["H_t"]["x_hat"]
                L_tc_rec = out_dec_c["L_t"]["x_hat"] if code_lt else out_enc_c["L_t"]
                H_tc_rec = out_dec_c["H_t"]["x_hat"]
            else:
                decoding_time = 0
                L_t_rec = out_enc["L_t_hat"] if code_lt else out_enc["L_t"]
                H_t_rec = out_enc["H_t_hat"]
                L_tc_rec = out_enc_c["L_t_hat"] if code_lt else out_enc_c["L_t"]
                H_tc_rec = out_enc_c["H_t_hat"]

            result = {
                "L_t": L_t_rec,
                "H_t": H_t_rec,
                "L_tc": L_tc_rec,
                "H_tc": H_tc_rec,
                "bit_H": bits_H + bits_H_c,
                "bit_L": bits_L + bits_L_c if code_lt else None,
                "bit_Lc": bits_L_c if code_lt else None,
                "bit_Hc": bits_H_c,
                "bit_ME": bits_me,
                "mv_hat": mv_hat,
                "dpb": {"mv_feature": mv_feature,
                        "ref_mv_y": mv_y_hat},
                "decoding_time": decoding_time,
                "encoding_time": encoding_time
            }

            return result
