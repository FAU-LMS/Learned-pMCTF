# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pMCTF.layers.video.video_net import LowerBound
import math

import torch
from torch import nn


class MVCoderQuad(nn.Module):
    def __init__(self, enc_dec_quant=False):
        super().__init__()
        self.enc_dec_quant = enc_dec_quant
        self.masks = {}

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n

        return torch.round(x)

    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)

    @staticmethod
    def separate_prior_enc_dec(params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = LowerBound.apply(quant_step, 0.5)
        #quant_step = torch.clamp(quant_step, 0.5, None)
        q_enc = 1. / quant_step
        q_dec = quant_step
        return q_enc, q_dec, scales, means

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def get_mask_four_parts(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
            mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)

            micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
            mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
            mask_1 = mask_1[:height, :width]
            mask_1 = torch.unsqueeze(mask_1, 0)
            mask_1 = torch.unsqueeze(mask_1, 0)

            micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
            mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
            mask_2 = mask_2[:height, :width]
            mask_2 = torch.unsqueeze(mask_2, 0)
            mask_2 = torch.unsqueeze(mask_2, 0)

            micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
            mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
            mask_3 = mask_3[:height, :width]
            mask_3 = torch.unsqueeze(mask_3, 0)
            mask_3 = torch.unsqueeze(mask_3, 0)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)


    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        if self.enc_dec_quant:
            q_enc, q_dec, scales, means = self.separate_prior_enc_dec(common_params)
        else:
            quant_step, scales, means = self.separate_prior(common_params)
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        if self.enc_dec_quant:
            y = y * q_enc
        else:
            quant_step = torch.clamp_min(quant_step, 0.5)
            y = y / quant_step
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        y_res_0_0, y_q_0_0, y_hat_0_0, s_hat_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, s_hat_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)
        y_res_2_2, y_q_2_2, y_hat_2_2, s_hat_2_2 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_2)
        y_res_3_3, y_q_3_3, y_hat_3_3, s_hat_3_3 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_3)
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)

        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)

        y_res_0_3, y_q_0_3, y_hat_0_3, s_hat_0_3 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_3)
        y_res_1_2, y_q_1_2, y_hat_1_2, s_hat_1_2 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_2)
        y_res_2_1, y_q_2_1, y_hat_2_1, s_hat_2_1 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_1)
        y_res_3_0, y_q_3_0, y_hat_3_0, s_hat_3_0 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_0)
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)

        y_res_0_2, y_q_0_2, y_hat_0_2, s_hat_0_2 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_2)
        y_res_1_3, y_q_1_3, y_hat_1_3, s_hat_1_3 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_3)
        y_res_2_0, y_q_2_0, y_hat_2_0, s_hat_2_0 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_0)
        y_res_3_1, y_q_3_1, y_hat_3_1, s_hat_3_1 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_1)
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)

        y_res_0_1, y_q_0_1, y_hat_0_1, s_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, s_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)
        y_res_2_3, y_q_2_3, y_hat_2_3, s_hat_2_3 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_3)
        y_res_3_2, y_q_3_2, y_hat_3_2, s_hat_3_2 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_2)

        y_res = self.combine_four_parts(y_res_0_0, y_res_0_1, y_res_0_2, y_res_0_3,
                                        y_res_1_0, y_res_1_1, y_res_1_2, y_res_1_3,
                                        y_res_2_0, y_res_2_1, y_res_2_2, y_res_2_3,
                                        y_res_3_0, y_res_3_1, y_res_3_2, y_res_3_3)
        y_q = self.combine_four_parts(y_q_0_0, y_q_0_1, y_q_0_2, y_q_0_3,
                                      y_q_1_0, y_q_1_1, y_q_1_2, y_q_1_3,
                                      y_q_2_0, y_q_2_1, y_q_2_2, y_q_2_3,
                                      y_q_3_0, y_q_3_1, y_q_3_2, y_q_3_3)
        y_hat = self.combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                        y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                        y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                        y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)
        scales_hat = self.combine_four_parts(s_hat_0_0, s_hat_0_1, s_hat_0_2, s_hat_0_3,
                                             s_hat_1_0, s_hat_1_1, s_hat_1_2, s_hat_1_3,
                                             s_hat_2_0, s_hat_2_1, s_hat_2_2, s_hat_2_3,
                                             s_hat_3_0, s_hat_3_1, s_hat_3_2, s_hat_3_3)

        if self.enc_dec_quant:
            y_hat = y_hat * q_dec
        else:
            y_hat = y_hat * quant_step

        if write:
            y_q_w_0 = y_q_0_0 + y_q_1_1 + y_q_2_2 + y_q_3_3
            y_q_w_1 = y_q_0_3 + y_q_1_2 + y_q_2_1 + y_q_3_0
            y_q_w_2 = y_q_0_2 + y_q_1_3 + y_q_2_0 + y_q_3_1
            y_q_w_3 = y_q_0_1 + y_q_1_0 + y_q_2_3 + y_q_3_2
            scales_w_0 = s_hat_0_0 + s_hat_1_1 + s_hat_2_2 + s_hat_3_3
            scales_w_1 = s_hat_0_3 + s_hat_1_2 + s_hat_2_1 + s_hat_3_0
            scales_w_2 = s_hat_0_2 + s_hat_1_3 + s_hat_2_0 + s_hat_3_1
            scales_w_3 = s_hat_0_1 + s_hat_1_0 + s_hat_2_3 + s_hat_3_2
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3,\
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior, gaussian_encoder):
        if self.enc_dec_quant:
            _, quant_step, scales, means = self.separate_prior_enc_dec(common_params)
        else:
            quant_step, scales, means = self.separate_prior(common_params)
            quant_step = torch.clamp_min(quant_step, 0.5)
        dtype = means.dtype
        device = means.device
        _, _, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        scales_r = scales_0 * mask_0 + scales_1 * mask_1 + scales_2 * mask_2 + scales_3 * mask_3
        y_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_0 = (y_q_r + means_0) * mask_0
        y_hat_1_1 = (y_q_r + means_1) * mask_1
        y_hat_2_2 = (y_q_r + means_2) * mask_2
        y_hat_3_3 = (y_q_r + means_3) * mask_3
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        scales_r = scales_0 * mask_3 + scales_1 * mask_2 + scales_2 * mask_1 + scales_3 * mask_0
        y_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_3 = (y_q_r + means_0) * mask_3
        y_hat_1_2 = (y_q_r + means_1) * mask_2
        y_hat_2_1 = (y_q_r + means_2) * mask_1
        y_hat_3_0 = (y_q_r + means_3) * mask_0
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        scales_r = scales_0 * mask_2 + scales_1 * mask_3 + scales_2 * mask_0 + scales_3 * mask_1
        y_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_2 = (y_q_r + means_0) * mask_2
        y_hat_1_3 = (y_q_r + means_1) * mask_3
        y_hat_2_0 = (y_q_r + means_2) * mask_0
        y_hat_3_1 = (y_q_r + means_3) * mask_1
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        scales_r = scales_0 * mask_1 + scales_1 * mask_0 + scales_2 * mask_3 + scales_3 * mask_2
        y_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_1 = (y_q_r + means_0) * mask_1
        y_hat_1_0 = (y_q_r + means_1) * mask_0
        y_hat_2_3 = (y_q_r + means_2) * mask_3
        y_hat_3_2 = (y_q_r + means_3) * mask_2
        y_hat_curr_step = torch.cat((y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat