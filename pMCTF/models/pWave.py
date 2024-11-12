import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from copy import deepcopy
from timm.models.layers import trunc_normal_
from pMCTF.utils.util import get_state_dict

from pMCTF.layers import (
    LiftingScheme2D,
    PostProcess,
    SubbandContext,
    ContextFusionSubband,
    ContextFusionFourStep,
    RoundNoGradient,
    ClampNoGradient)

from pMCTF.utils.util import normalize_tensor
from pMCTF.entropy_models.gaussian_model import CompressionModel

from pMCTF.utils.stream_helper import get_state_dict, encode_image, decode_image


class pWave(nn.Module):
    """  End-to-end wavelet image coder pWave++
    Analysis and synthesis transform are a forward and inverse wavelet transform implemented via lifting scheme.
    pWave++: Parallelized version of iWave++.
    LL subband: Autoregressive context model.
    Remaining subbands:  Sequentially coded using four-step context fusion model and inter-subband long-term context.
    """

    def __init__(self,
                 bitdepth=8,
                 decomp_levels=4,
                 lossy=True):

        super(pWave, self).__init__()
        self.bitdepth = 8
        self.dynamic_range = float(2 ** bitdepth)
        self.lossy = lossy

        self.in_channels = 1

        self.decomp_levels = decomp_levels
        self.mse = nn.MSELoss(reduction='mean')

        self.wavelet_transform = LiftingScheme2D(bitdepth=bitdepth, lossy=self.lossy,
                                                 in_channels=self.in_channels)

        self.visual_names = ['l', 'h', ('ll', 'hl'), ('lh', 'hh'), 'x', 'x_hat']
        self.visuals = []

        if self.lossy:
            self.clip_value = 8192.
        else:
            self.clip_value = torch.iinfo(torch.int16).max

        self.context_prediction = SubbandContext(in_channels=1, decomp_levels=self.decomp_levels)

        self.dequantModule = PostProcess(in_channels=self.in_channels, out_channels=self.in_channels)

        self.num_params = 2  # scales, means

        self.em = CompressionModel(y_distribution="laplace")  # following DCVC-DC

        context_fusion = ContextFusionFourStep
        nfeats = 112

        # context fusion model (includes entropy parameter prediction)
        self.context_fusion = nn.ModuleDict({
            str(lvl):
                nn.ModuleDict({
                    subband: context_fusion(in_channels=1, num_features=nfeats, num_parameters=self.num_params,
                                            lossy=lossy, ctx_channels=2 if lvl < self.decomp_levels-1 else 1)
                    for subband in ["lh", "hl", "hh"]})
                for lvl in range(self.decomp_levels)})

        self.context_fusion[str(self.decomp_levels - 1)]['ll'] = \
            ContextFusionSubband(num_features=128, num_parameters=self.num_params, context=False,
                                 in_channels=1)

        self.QP = nn.Parameter(torch.ones((2, 1, 1, 1), dtype=torch.float) * 1 / 16)
        self.QP_ll = nn.Parameter(torch.ones((2, 1, 1, 1), dtype=torch.float) * 1 / 16)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d) and (m.weight.size(-1) == m.weight.size(-2)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def fix_wavelet_transform(self):
        for p in self.wavelet_transform.parameters():
            p.requires_grad = False

    def get_q_scale(self, ridx):
        return self.QP[ridx]

    def compute_visuals(self, x, x_hat, normalize=True):
        """Calculate additional output images HTML visualization"""
        subbands = self.encode(x)
        self.visuals = {}
        for lvl in range(self.decomp_levels):
            ims = subbands[lvl]
            ims = [normalize_tensor(im, im_name=im_name) if normalize else im for im_name, im in ims.items()]
            if len(ims) == 4:
                [ll, lh, hl, hh] = ims
                l_h = torch.cat((ll, lh), dim=3)
            else:
                [ll, lh, hl, hh, l, h] = ims
                l_h = torch.cat((l, h), dim=3)

            ll_lh_hl_hh = torch.cat((ll, lh), dim=3)

            ll_lh_hl_hh = torch.cat((ll_lh_hl_hh,
                                     torch.cat((hl, hh), dim=3)), dim=2)
            self.visuals[lvl] = {f"l_h_lvl{lvl + 1}": l_h,
                                 f"ll_lh_hl_hh_lvl{lvl + 1}": ll_lh_hl_hh}
        self.visuals['x'] = normalize_tensor(x, im_name="x") if normalize else x
        self.visuals['x_hat'] = x_hat.clamp(0, self.dynamic_range - 1)
        if normalize:
            self.visuals['x_hat'] = normalize_tensor(self.visuals['x_hat'], im_name="x_hat")

    def get_current_visuals(self):
        """Return visualization images. During training, the images will be saved and can be accessed via a HTML file"""
        return self.visuals

    def make_all_trainable(self):
        for p in self.parameters():
            p.requires_grad = True

    def encode(self, x):
        """ Training encoding function.
            The analysis transform is a (multi-level) wavelet decomposition. """
        subbands = {}
        ll = x
        for lvl in range(self.decomp_levels):
            subband_dict = self.wavelet_transform.forward_lift_2d(ll)
            subbands[lvl] = subband_dict
            ll = subband_dict['ll']
        return subbands

    def decode(self, subbands):
        """ Training decoding function.
            The synthesis transform is a (multi-level) inverse wavelet decomposition. """
        for lvl in range(self.decomp_levels - 1, -1, -1):
            y = self.wavelet_transform.backward_lift_2d(subbands[lvl])
            if lvl > 0:
                subbands[lvl - 1]['ll'] = y
        return y

    def compute_loss(self, output, target, lmda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp"] = output["bits"]["bits_total"].sum() / num_pixels
        out["mse"] = self.mse(output["x_hat"], target)
        out["loss"] = lmda * out["mse"] + out["bpp"]
        return out

    def quantize_subbands(self, subbands, q_scale, q_scale_ll):
        subbands_hat = {}
        for lvl in range(self.decomp_levels-1, -1, -1):
            subbands_hat[lvl] = {}
            subband_list = ['ll', 'lh', 'hl', 'hh'] if lvl == self.decomp_levels-1 else ['lh', 'hl', 'hh']
            for sidx, subband in enumerate(subband_list):
                QP = q_scale_ll if subband == 'll' else q_scale
                if self.lossy:
                    subbands_hat[lvl][subband] = RoundNoGradient.apply(
                        ClampNoGradient.apply(subbands[lvl][subband] * QP, -self.clip_value, self.clip_value))
                else:
                    subbands_hat[lvl][subband] = ClampNoGradient.apply(
                        subbands[lvl][subband], -self.clip_value, self.clip_value)

        return subbands_hat

    def quantize_subband(self, subband, q_scale):
        if self.lossy:
            ret = ClampNoGradient.apply(subband * q_scale, -self.clip_value, self.clip_value)
        else:
            ret = ClampNoGradient.apply(subband, -self.clip_value, self.clip_value)
        return ret

    def dequantize_subbands(self, subbands_hat, q_scale, q_scale_ll):
        subbands_rec = {}
        for lvl in range(self.decomp_levels-1, -1, -1):
            subbands_rec[lvl] = {}
            subband_list = ['ll', 'lh', 'hl', 'hh'] if lvl == self.decomp_levels-1 else ['lh', 'hl', 'hh']
            for sidx, subband in enumerate(subband_list):
                QP = q_scale_ll if subband == 'll' else q_scale
                if self.lossy:
                    subbands_rec[lvl][subband] = subbands_hat[lvl][subband] / QP
                else:
                    subbands_rec[lvl][subband] = subbands_hat[lvl][subband]
        return subbands_rec

    def dequantize_subband(self, subband, q_scale):
        if self.lossy:
            subband = subband / q_scale
        return subband

    def get_one_q_scale(self, q_scale, q_index):
        # duplicate from video model
        min_q = q_scale[0:1, :, :, :]
        max_q = q_scale[1:2, :, :, :]
        step = (torch.log(max_q) - torch.log(min_q)) / (self.get_qp_num() - 1)
        q = torch.exp(torch.log(min_q) + step * q_index)
        return q

    def get_curr_q(self, q_scale, q_index):
        # duplicate from video model
        if isinstance(q_index, list):
            q_step = [self.get_one_q_scale(q_scale, i) for i in q_index]
            q_step = torch.cat(q_step, dim=0)
        else:
            q_step = self.get_one_q_scale(q_scale, q_index)

        return q_step

    @staticmethod
    def get_qp_num():
        return 21  # 64

    def forward(self, x, q_index=None, qp_scale=None):
        if q_index is not None:
            qp = self.get_curr_q(self.QP, q_index)
            qp_ll = self.get_curr_q(self.QP_ll, q_index)
            if qp_scale is not None:
                # depends on temporal decomposition level (stage idx)
                qp = qp * qp_scale
                qp_ll = qp_ll * qp_scale
            return self.forward_one_channel(x, qp, qp_ll)
        else:
            return self.forward_one_channel(x)

    def forward_one_channel(self, x, q_scale=None, q_scale_ll=None):
        if q_scale is None:
            q_scale = self.QP[-1]
            q_scale_ll = self.QP_ll#[-1]
        # ANALYSIS TRANSFORM: forward wavelet transform
        y = self.encode(x)

        # CODE quantized subbands
        subbands_hat = {lvl: {} for lvl in range(self.decomp_levels)}
        bits_ret = {lvl: {} for lvl in range(self.decomp_levels - 1, -1, -1)}

        ll = y[self.decomp_levels - 1]['ll']
        # rounding AFTER mean has been subtracted
        ll = self.quantize_subband(ll, q_scale_ll)  # multiply by q_scale
        ll_hat = RoundNoGradient.apply(ll)

        params = self.context_fusion[str(self.decomp_levels - 1)]['ll'](ll_hat)
        scales, means = params.chunk(2, dim=1)
        # different quantization for autoregressive part
        bits_ll = self.em.get_y_laplace_bits(ll_hat - means, scales)

        subbands_hat[self.decomp_levels - 1]['ll'] = ll_hat

        bits_ret[self.decomp_levels - 1]['ll'] = bits_ll
        bits_ret["bits_total"] = torch.sum(bits_ll, dim=(1, 2, 3))

        self.context_prediction.init_sequential(list(ll.size()), ll.device)
        context = self.context_prediction.forward_one_subband(ll_hat, "ll", self.decomp_levels - 1)["context"]

        # fuse contexts and use as input for entropy bottleneck
        for lvl in range(self.decomp_levels - 1, -1, -1):
            for sidx, subband in enumerate(['lh', 'hl', 'hh']):
                context = context.chunk(3, dim=1)
                context = context[sidx]
                # previously coded LH/HL/HH subband
                prev_subband = subbands_hat[lvl+1][subband] if lvl < self.decomp_levels - 1 else None

                s_curr = y[lvl][subband]
                s_curr = self.quantize_subband(s_curr, q_scale)

                s_res, s_q, s_hat, scales = self.context_fusion[str(lvl)][subband](s_curr, context=context,
                                                                                   prev_subband=prev_subband)
                subbands_hat[lvl][subband] = s_hat
                bits_curr = self.em.get_y_laplace_bits(s_q, scales)
                bits_ret[lvl][subband] = bits_curr
                bits_ret["bits_total"] += torch.sum(bits_curr, dim=(1, 2, 3))

                context = self.context_prediction.forward_one_subband(s_hat, subband, lvl)["context"]

        # DEQUANTIZE
        subbands_rec = self.dequantize_subbands(subbands_hat, q_scale, q_scale_ll)  # divide by q_scale

        # SYNTHESIS TRANSFORM: inverse wavelet transform
        x_hat = self.decode(subbands_rec)

        # Post Processing
        if self.lossy:
            x_hat = self.dequantModule(x_hat/self.dynamic_range) * self.dynamic_range

        ret_dict = {
            "x_hat": x_hat,
            "bits": bits_ret,
            "likelihoods": bits_ret,
            "subbands": subbands_hat,
            "bpp_total": bits_ret["bits_total"].sum() / (x_hat.size(2) * x_hat.size(3) * x_hat.size(0)),
            "bits_total": bits_ret["bits_total"].sum() / x_hat.size(0),
            "mse": self.mse(x, x_hat)
        }

        return ret_dict

    def spatial_wavelet_dec(self, x, q_scale, q_scale_ll):
        # compute approxtimation of forward pass (skip actual coding of subbands)
        if q_scale is None:
            q_scale = self.QP[-1]
            q_scale_ll = self.QP_ll
        # ANALYSIS TRANSFORM: forward wavelet transform
        y = self.encode(x)

        # CODE quantized subbands
        subbands_hat = {lvl: {} for lvl in range(self.decomp_levels)}

        ll = y[self.decomp_levels - 1]['ll']
        # rounding AFTER mean has been subtracted
        ll = self.quantize_subband(ll, q_scale_ll)  # multiply by q_scale
        subbands_hat[self.decomp_levels - 1]['ll'] = RoundNoGradient.apply(ll)

        for lvl in range(self.decomp_levels - 1, -1, -1):
            for sidx, subband in enumerate(['lh', 'hl', 'hh']):

                s_curr = y[lvl][subband]
                s_curr = self.quantize_subband(s_curr, q_scale)

                # caution: real s_hat depends on predicted means of context fusion model
                subbands_hat[lvl][subband] = RoundNoGradient.apply(s_curr)

        # DEQUANTIZE
        subbands_rec = self.dequantize_subbands(subbands_hat, q_scale, q_scale_ll)  # divide by q_scale

        # SYNTHESIS TRANSFORM: inverse wavelet transform
        x_hat = self.decode(subbands_rec)

        # Post Processing: De-quantization
        if self.lossy:
            x_hat = self.dequantModule(x_hat/self.dynamic_range) * self.dynamic_range

        return x_hat

    def forward_ycbcr(self, x):
        if isinstance(x, list):
            in_y, in_cb, in_cr = x
            c420 = True
        else:
            # Y
            in_y = deepcopy(x[:, 0, :, :]).unsqueeze(1)

            in_cb = deepcopy(x[:, 1, :, :].unsqueeze(1))
            in_cr = deepcopy(x[:, 2, :, :].unsqueeze(1))

        out_net_y = self.forward_one_channel(in_y)  # Y

        # CB and CR
        out_net_cb = self.forward_one_channel(in_cb)
        out_net_cr = self.forward_one_channel(in_cr)

        # plot_correlation() in .utils.py
        ret_dict = {
            "x_hat": torch.cat((out_net_y["x_hat"], out_net_cb["x_hat"], out_net_cr["x_hat"]), dim=1),
            "bits": {"bits_y": out_net_y["bits"]["bits_total"],
                     "bits_cb": out_net_cb["bits"]["bits_total"],
                     "bits_cr": out_net_cr["bits"]["bits_total"]},
        }
        return ret_dict

    def update(self, force=False):
        self.em.update(force)

    @torch.no_grad()
    def compress(self, x, sideinfo=None, file_name=None, q_index=None, skip_decoding=False, qp_scale=None):
        _, num_channels, height, width = sideinfo
        if q_index is None:
            q_scale = self.QP[-1]
            q_scale_ll = self.QP_ll #[-1]
        else:
            q_scale = self.get_curr_q(self.QP, q_index)
            q_scale_ll = self.get_curr_q(self.QP_ll, q_index)
            if qp_scale is not None:
                # depends on temporal decomposition level (stage idx)
                q_scale = q_scale * qp_scale
                q_scale_ll = q_scale_ll * qp_scale
        if num_channels == 3:
            # RGB image
            x_in = torch.cat((x[:, 0:0+1, :, :], x[:, 1:1+1, :, :], x[:, 2:2+1, :, :]), dim=0)
        else:
            # Y or UV
            x_in = x

        # for ycbcr_idx in range(3):
        y = self.encode(x_in)

        # CODE quantized subbands
        subbands_hat = {lvl: {} for lvl in range(self.decomp_levels)}

        ll = y[self.decomp_levels - 1]['ll']
        # rounding AFTER means have been subtracted
        ll = self.quantize_subband(ll, q_scale_ll).round()  # multiply by q_scale

        self.em.entropy_coder.reset()
        if not skip_decoding:
            ll_hat = self._compress_subband_ar(ll, 'll', 3, encoder=self.em.gaussian_encoder)
        else:
            params = self.context_fusion[str(self.decomp_levels - 1)]['ll'](ll)
            scales, means = params.chunk(2, dim=1)
            ll_res, ll_q, ll_hat = self.em.process(ll, means)
            ll_hat = (ll_res.round() + means).round()
            self.em.gaussian_encoder.encode(ll_res.round(), scales)
        subbands_hat[self.decomp_levels - 1]['ll'] = ll_hat

        self.context_prediction.init_sequential(list(ll.size()), ll.device)
        context = self.context_prediction.forward_one_subband(ll_hat, "ll", self.decomp_levels - 1)["context"]

        # fuse contexts and use as input for entropy bottleneck
        for lvl in range(self.decomp_levels - 1, -1, -1):
            for sidx, subband in enumerate(['lh', 'hl', 'hh']):
                context = context.chunk(3, dim=1)
                context = context[sidx]
                # previously coded LH/HL/HH subband
                prev_subband = subbands_hat[lvl + 1][subband] if lvl < self.decomp_levels - 1 else None

                s_curr = y[lvl][subband]
                s_curr = self.quantize_subband(s_curr, q_scale)

                out = self.context_fusion[str(lvl)][subband].compress(s_curr, context=context,
                                                                      prev_subband=prev_subband)
                x_q_0, x_q_1, x_q_2, x_q_3, s_w_0, s_w_1, s_w_2, s_w_3, s_hat = out
                subbands_hat[lvl][subband] = s_hat

                self.em.gaussian_encoder.encode(x_q_0, s_w_0)
                self.em.gaussian_encoder.encode(x_q_1, s_w_1)
                self.em.gaussian_encoder.encode(x_q_2, s_w_2)
                self.em.gaussian_encoder.encode(x_q_3, s_w_3)

                context = self.context_prediction.forward_one_subband(s_hat, subband, lvl)["context"]

        # DEQUANTIZE
        subbands_rec = self.dequantize_subbands(subbands_hat, q_scale, q_scale_ll)  # divide by q_scale

        # SYNTHESIS TRANSFORM: inverse wavelet transform
        x_hat = self.decode(subbands_rec)

        # Post Processing: De-quantization
        if self.lossy:
            x_hat = self.dequantModule(x_hat / self.dynamic_range) * self.dynamic_range

        self.em.entropy_coder.flush()
        bit_stream = self.em.entropy_coder.get_encoded_stream()
        encode_image(height, width, num_channels, bit_stream, file_name)

        if num_channels == 3:
            x_hat = torch.cat((x_hat[0:0 + 1, :, :, :], x_hat[1:1 + 1, :, :, :], x_hat[2:2 + 1, :, :, :]), dim=1)
        return x_hat # , subbands_hat


    @torch.no_grad()
    def decompress(self, file_name, padding=64, q_index=None, qp_scale=None):
        if q_index is None:
            q_scale = self.QP[-1]
            q_scale_ll = self.QP_ll #[-1]
        else:
            q_scale = self.get_curr_q(self.QP, q_index)
            q_scale_ll = self.get_curr_q(self.QP_ll, q_index)
            if qp_scale is not None:
                # depends on temporal decomposition level (stage idx)
                q_scale = q_scale * qp_scale
                q_scale_ll = q_scale_ll * qp_scale
        height, width, num_channel, bit_stream = decode_image(file_name)

        self.em.entropy_coder.set_stream(bit_stream)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        p = padding  # maximum 6 strides of 2
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p

        kernel_size = 3  # context prediction kernel size (masked convolution kernel size)
        padding = (kernel_size - 1) // 2

        # ycbcr_ret = None
        # for ycbcr_idx in range(3):
        lvl = self.decomp_levels - 1
        subband_h = new_h // (2 ** (lvl + 1))
        subband_w = new_w // (2 ** (lvl + 1))
        subband_size = [num_channel, 1, subband_h, subband_w]

        entropy_coder = self.em.gaussian_encoder
        ll_rec = self._decompress_subband_ar(subband_size, padding, "ll", self.decomp_levels-1,
                                             entropy_coder, device, dtype)

        subband_ret = {lvl: {} for lvl in range(self.decomp_levels - 1, -1, -1)}
        subband_ret[self.decomp_levels-1]["ll"] = ll_rec

        self.context_prediction.init_sequential(list(ll_rec.size()), ll_rec.device)
        context = self.context_prediction.forward_one_subband(ll_rec, "ll", self.decomp_levels - 1)["context"]

        for lvl in range(self.decomp_levels - 1, -1, -1):
            for sidx, subband in enumerate(['lh', 'hl', 'hh']):
                context = context.chunk(3, dim=1)
                context = context[sidx]
                # previously coded LH/HL/HH subband
                prev_subband = subband_ret[lvl + 1][subband] if lvl < self.decomp_levels - 1 else None

                s_hat = self.context_fusion[str(lvl)][subband].decompress(self.em.gaussian_encoder, context=context,
                                                                          prev_subband=prev_subband)
                subband_ret[lvl][subband] = s_hat

                context = self.context_prediction.forward_one_subband(s_hat, subband, lvl)["context"]

        subbands_rec = self.dequantize_subbands(subband_ret, q_scale, q_scale_ll)
        x_hat = self.decode(subbands_rec)

        # Post Processing
        if self.lossy:
            x_hat = self.dequantModule(x_hat / self.dynamic_range) * self.dynamic_range
        if num_channel == 3:
            x_hat = torch.cat((x_hat[0:0 + 1, :, :, :], x_hat[1:1 + 1, :, :, :], x_hat[2:2 + 1, :, :, :]), dim=1)
        return {"x_hat": x_hat}

    def _compress_subband_ar(self, y, subband, lvl, encoder):
        symbols = y

        B, C, H, W = symbols.size()
        subband_hat = torch.zeros_like(y)
        padding = 1  # kernel size = 3
        symbols = F.pad(symbols, (padding, padding, padding, padding))
        elems = np.arange(np.prod(y.size()))
        pbar = tqdm(elems, total=len(elems), desc=f"{subband} lvl {lvl} encode", unit="elem(s)")
        for h in range(H):  # Warning, this is slow...
            for w in range(W):
                params = self.context_fusion[str(lvl)][subband].forward_sequential(
                    symbols, h, w)
                scales, means = params.chunk(2, dim=1)

                ll_res, ll_q, ll_hat = self.em.process(symbols[:, 0:0+1, h + padding:h + padding + 1, w + padding:w + padding+1], means)
                ll_hat = (ll_res.round() + means).round()
                subband_hat[:, :, h, w] = ll_hat[:, :, 0, 0]
                encoder.encode(ll_res.round(), scales)

        pbar.update()
        # finished sequential encoding
        self.context_fusion[str(lvl)][subband].sequential_init = False
        pbar.close()
        return subband_hat

    def _decompress_subband_ar(self, subband_size, padding, subband, lvl, decoder, device, dtype):
        B, C, H, W = subband_size
        assert C == 1
        curr_subband = torch.zeros(size=[B, C, H, W]).to(device)
        curr_subband = F.pad(curr_subband, (padding, padding, padding, padding))

        elems = np.arange(np.prod(subband_size))
        pbar = tqdm(elems, total=len(elems), desc=f"{subband} lvl {lvl} decode", unit="elem(s)")

        for h in range(H):  # Warning, this is slow...
            for w in range(W):
                # calculate entropy parameters via context fusion module
                symbols_in = curr_subband
                condition = self.context_fusion[str(lvl)][subband].forward_sequential(
                    symbols_in, h, w)

                scale, mean = condition.chunk(2, dim=1)
                rec = decoder.decode_stream(scale, dtype, device)

                rec = rec + mean
                # pbar.update()
                curr_subband[:, :, h + padding, w + padding] = rec.round()[:, :, 0, 0]

        # finished sequential encoding
        pbar.close()
        curr_subband = F.pad(curr_subband, (-padding, -padding, -padding, -padding))
        self.context_fusion[str(lvl)][subband].sequential_init = False
        return curr_subband

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scales = ckpt["QP"]

        return q_scales.reshape(-1)

