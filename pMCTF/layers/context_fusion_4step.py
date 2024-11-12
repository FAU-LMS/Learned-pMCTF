import torch
import torch.nn as nn

from .convs import get_conv2d
from pMCTF.layers.lifting_1d import RoundNoGradient
from pMCTF.layers.video.layers import DepthConvBlock


class ContextResidual(nn.Module):
    def __init__(self,num_features):
        super(ContextResidual, self).__init__()
        self.conv1 = get_conv2d(kernel_size=3, in_ch=num_features, out_ch=num_features)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = get_conv2d(kernel_size=3, in_ch=num_features, out_ch=num_features)

    def forward(self, x):
        output = self.conv1(x)
        output = self.lrelu(output)
        output = self.conv2(output)
        return output + x


class ContextFusionFourStep(nn.Module):
    def __init__(self,
                 in_channels=1,
                 ctx_channels=1,
                 num_features=128,
                 num_parameters=2,
                 ctx=True,
                 lossy=True,
                 lower_subband=True):
        super(ContextFusionFourStep, self).__init__()

        self.num_ch = num_features
        self.num_parameters = num_parameters
        self.ctx_channels = ctx_channels
        self.ctx = ctx
        self.lossy = lossy

        self.masks = {}

        if self.ctx:
            curr_ch = self.num_ch
            self.y_hierarchical_prior_enc = nn.Sequential(
                ContextResidual(self.num_ch),
                ContextResidual(self.num_ch)
            )
            self.conv1_context = get_conv2d(kernel_size=3, in_ch=ctx_channels, out_ch=num_features)
            if ctx_channels > 1 and lower_subband:
                self.lower_level_subband = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                                         get_conv2d(kernel_size=3, in_ch=in_channels,
                                                                    out_ch=in_channels))
        else:
            curr_ch = in_channels
        self.y_hierarchical_prior_out = DepthConvBlock(curr_ch, self.num_parameters)

        #self.y_hierarchical_prior_out = nn.Sequential(
        #    ContextResidual(self.num_ch),
        #    get_conv2d(kernel_size=1, in_ch=self.num_ch, out_ch=self.num_parameters)
        #)

        self.y_spatial_prior_1 = nn.Sequential(
            get_conv2d(kernel_size=3, in_ch=in_channels, out_ch=num_features),
            ContextResidual(self.num_ch)
        )
        self.y_spatial_prior_1_out = nn.Sequential(
            ContextResidual(self.num_ch),
            ContextResidual(self.num_ch),
            get_conv2d(kernel_size=1, in_ch=self.num_ch, out_ch=self.num_parameters)
        )

        self.y_spatial_prior_2 = nn.Sequential(
            get_conv2d(kernel_size=3, in_ch=in_channels, out_ch=num_features),
            ContextResidual(self.num_ch)
        )
        self.y_spatial_prior_2_out = nn.Sequential(
            ContextResidual(self.num_ch),
            ContextResidual(self.num_ch),
            get_conv2d(kernel_size=1, in_ch=self.num_ch, out_ch=self.num_parameters)
        )

        self.y_spatial_prior_3 = nn.Sequential(
            get_conv2d(kernel_size=3, in_ch=in_channels, out_ch=num_features),
            ContextResidual(self.num_ch)
        )
        self.y_spatial_prior_3_out = nn.Sequential(
            ContextResidual(self.num_ch),
            ContextResidual(self.num_ch),
            get_conv2d(kernel_size=1, in_ch=self.num_ch, out_ch=self.num_parameters)
        )

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

    def quant(self, x):
        if self.training:
            return RoundNoGradient.apply(x)
        else:
            return torch.round(x)

    def process_with_mask(self, y, scales, means, mask):
        if not self.lossy:
            means = RoundNoGradient.apply(means)
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def forward(self, x, context=None, prev_subband=None, write=False):
        if self.ctx:
            if prev_subband is not None:
                prev_subband = self.lower_level_subband(prev_subband)
                context = torch.cat((context, prev_subband), dim=1)
            context = self.conv1_context(context)
            context = self.y_hierarchical_prior_enc(context)
        else:
            context = torch.zeros_like(x)
        hierarchical_params = self.y_hierarchical_prior_out(context)
        scales_0, means_0 = hierarchical_params.chunk(2, dim=1)

        dtype = x.dtype
        device = x.device
        _, _, H, W = x.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        # STEP 1
        x_res_0, x_q_0, x_hat_0, s_hat_0 = self.process_with_mask(x, scales_0, means_0, mask_0)
        x_hat_so_far = x_hat_0
        x_0_out = self.y_spatial_prior_1(x_hat_so_far)
        x_0_out = x_0_out + context
        params = self.y_spatial_prior_1_out(x_0_out)
        scales_1, means_1 = params.chunk(2, dim=1)

        # STEP 2
        x_res_1, x_q_1, x_hat_1, s_hat_1 = self.process_with_mask(x, scales_1, means_1, mask_1)
        x_hat_so_far = x_hat_so_far + x_hat_1
        x_1_out = self.y_spatial_prior_2(x_hat_so_far)
        x_1_out = x_1_out + context
        params = self.y_spatial_prior_2_out(x_1_out)
        scales_2, means_2 = params.chunk(2, dim=1)

        # STEP 3
        x_res_2, x_q_2, x_hat_2, s_hat_2 = self.process_with_mask(x, scales_2, means_2, mask_2)
        x_hat_so_far = x_hat_so_far + x_hat_2
        x_2_out = self.y_spatial_prior_3(x_hat_so_far)
        x_2_out = x_2_out + context
        params = self.y_spatial_prior_3_out(x_2_out)
        scales_3, means_3 = params.chunk(2, dim=1)

        # STEP 4
        x_res_3, x_q_3, x_hat_3, s_hat_3 = self.process_with_mask(x, scales_3, means_3, mask_3)

        x_hat = x_hat_so_far + x_hat_3
        x_res = x_res_0 + x_res_1 + x_res_2 + x_res_3
        x_q = x_q_0 + x_q_1 + x_q_2 + x_q_3
        s_hat = s_hat_0 + s_hat_1 + s_hat_2 + s_hat_3

        if write:
            return x_q_0, x_q_1, x_q_2, x_q_3, s_hat_0, s_hat_1, s_hat_2, s_hat_3, x_hat

        return x_res, x_q, x_hat, s_hat

    def compress(self, x, context=None, prev_subband=None):
        return self.forward(x, context, prev_subband, write=True)

    def decompress(self, gaussian_encoder, context=None, prev_subband=None):
        if prev_subband is not None:
            prev_subband = self.lower_level_subband(prev_subband)
            context = torch.cat((context, prev_subband), dim=1)
        if self.ctx:
            context = self.conv1_context(context)
            context = self.y_hierarchical_prior_enc(context)
        else:
            context = torch.zeros(context, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)

        hierarchical_params = self.y_hierarchical_prior_out(context)
        scales_0, means_0 = hierarchical_params.chunk(2, dim=1)

        dtype = scales_0.dtype
        device = scales_0.device
        _, _, H, W = scales_0.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        scales_r = scales_0 * mask_0
        x_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        x_hat_0 = (x_q_r + means_0) * mask_0

        x_hat_so_far = x_hat_0
        x_0_out = self.y_spatial_prior_1(x_hat_so_far)
        x_0_out = x_0_out + context
        params = self.y_spatial_prior_1_out(x_0_out)
        scales_1, means_1 = params.chunk(2, dim=1)

        scales_r = scales_1 * mask_1
        x_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        x_hat_1 = (x_q_r + means_1) * mask_1
        x_hat_so_far = x_hat_so_far + x_hat_1

        x_1_out = self.y_spatial_prior_2(x_hat_so_far)
        x_1_out = x_1_out + context
        params = self.y_spatial_prior_2_out(x_1_out)
        scales_2, means_2 = params.chunk(2, dim=1)

        scales_r = scales_2 * mask_2
        x_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        x_hat_2 = (x_q_r + means_2) * mask_2
        x_hat_so_far = x_hat_so_far + x_hat_2

        x_2_out = self.y_spatial_prior_3(x_hat_so_far)
        x_2_out = x_2_out + context
        params = self.y_spatial_prior_3_out(x_2_out)
        scales_3, means_3 = params.chunk(2, dim=1)

        scales_r = scales_3 * mask_3
        x_q_r = gaussian_encoder.decode_stream(scales_r, dtype, device)
        x_hat_3 = (x_q_r + means_3) * mask_3
        x_hat_so_far = x_hat_so_far + x_hat_3

        return x_hat_so_far
