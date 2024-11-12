import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import get_conv2d, get_masked_conv2d


class MaskResidual(nn.Module):
    def __init__(self, num_features):
        super(MaskResidual, self).__init__()
        self.conv1 = get_masked_conv2d(kernel_size=3, mask_type="B", in_ch=num_features, out_ch=num_features)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = get_masked_conv2d(kernel_size=3, mask_type="B", in_ch=num_features, out_ch=num_features)
        self.conv1_input, self.conv2_input = None, None
        self.maskedWeight1, self.maskedWeight2 = None, None

    def forward(self, x):
        output = self.conv1(x)
        output = self.lrelu(output)
        output = self.conv2(output)
        return output + x

    def init_sequential(self, y_hat):
        self.conv1_input = torch.zeros_like(y_hat)
        self.conv2_input = torch.zeros_like(y_hat)
        self.maskedWeight1 = self.conv1.weight * self.conv1.mask
        self.maskedWeight2 = self.conv2.weight * self.conv2.mask

    def forward_sequential(self, x, h, w, kernel_size, padding):
        # save output of previous masked conv layer
        self.conv1_input[:, :, h+padding:h+padding+1, w+padding:w+padding+1] = x[:, :, :, :]
        tmp = self.conv1_input[:, :, h: h + kernel_size, w: w + kernel_size]
        tmp = F.conv2d(tmp, self.maskedWeight1, bias=self.conv1.bias)
        tmp = self.lrelu(tmp)  # relu

        self.conv2_input[:, :, h+padding:h+padding+1, w+padding:w+padding+1] = tmp[:, :, :, :]
        tmp = self.conv2_input[:, :, h: h + kernel_size, w: w + kernel_size]
        tmp = F.conv2d(tmp, self.maskedWeight2, bias=self.conv2.bias)
        return tmp + x


class ContextResidual(nn.Module):
    def __init__(self, num_features):
        super(ContextResidual, self).__init__()
        self.conv1 = get_conv2d(kernel_size=3, in_ch=num_features, out_ch=num_features)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = get_conv2d(kernel_size=3, in_ch=num_features, out_ch=num_features)

    def forward(self, x):
        output = self.conv1(x)
        output = self.lrelu(output)
        output = self.conv2(output)
        return output + x


class ContextFusionSubband(nn.Module):
    # num_features = N
    def __init__(self,
                 in_channels=1,
                 ctx_channels=1,
                 num_features=128,
                 context=False,
                 num_parameters=9,
                 prev_ctxs=False, pretrained=None,
                 ll_ctx=False,
                 lower_subband=True,
                 adaptive_quant=False):
        super(ContextFusionSubband, self).__init__()
        self.sequential_init = False
        self.maskedWeight1 = None
        self.maskedConv2_input = None
        self.maskedWeight2 = None
        self.residual_blocks = 2

        self.num_features = num_features
        self.context = context
        self.num_parameters = num_parameters

        self.maskedConv1 = get_masked_conv2d(kernel_size=3, mask_type="A", in_ch=in_channels, out_ch=num_features)
        self.residualBlocks = nn.ModuleList(MaskResidual(self.num_features)
                                            for _ in range(self.residual_blocks))
        self.ctx_channels = ctx_channels

        self.maskedConv2 = get_masked_conv2d(kernel_size=3, mask_type="B", in_ch=num_features, out_ch=num_features)

        if self.context:
            self.conv1_context = get_conv2d(kernel_size=3, in_ch=ctx_channels, out_ch=num_features)
            if ctx_channels > 1 and not ll_ctx and lower_subband:
                self.lower_level_subband_up = nn.Upsample(scale_factor=2, mode="nearest")
                self.lower_level_subband_conv = get_conv2d(kernel_size=3, in_ch=in_channels, out_ch=in_channels)
            self.residualBlocksContext = nn.ModuleList(ContextResidual(self.num_features)
                                                       for _ in range(self.residual_blocks))

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.convs = nn.ModuleList([get_conv2d(kernel_size=1, in_ch=num_features, out_ch=num_features),
                                   get_conv2d(kernel_size=1, in_ch=num_features, out_ch=num_features),
                                   get_conv2d(kernel_size=1, in_ch=num_features, out_ch=num_parameters)])

    def forward(self, x, context=None, prev_subband=None, channel_idx=0):
        x = self.maskedConv1(x)
        conv1 = x
        if self.context:
            if prev_subband is not None:
                prev_subband = self.lower_level_subband_up(prev_subband)
                prev_subband = self.lower_level_subband_conv(prev_subband)
                context = torch.cat((context, prev_subband), dim=1)
            context = self.conv1_context(context)

            x = x + context

        for idx in range(self.residual_blocks):
            x = self.residualBlocks[idx](x)
            if self.context:
                context = self.residualBlocksContext[idx](context)
                x = x + context

        x = x + conv1
        x = self.maskedConv2(x)
        x = self.lrelu(x)

        num_convs = len(self.convs)
        for idx, curr_conv in enumerate(self.convs):
            x = curr_conv(x)
            if idx < num_convs-1:
                x = self.lrelu(x)
        # returns entropy paramaters
        return x

    def get_quant(self, ctx):
        ctx = self.ctx_conv_quant(ctx)
        quant_step = self.res_quant(ctx)
        quant_step = self.pred_quant(quant_step)
        quant_step = self.lower_bound(quant_step)
        return quant_step

    def init_sequential(self, y_hat):
        self.maskedWeight1 = self.maskedConv1.weight * self.maskedConv1.mask

        new_size = list(y_hat.size())
        new_size[1] = self.num_features
        self.mask1 = self.maskedConv1.mask[0, :, :, :].unsqueeze(0).repeat(new_size[0], 1, 1, 1)
        self.maskedConv2_input = torch.zeros(new_size, dtype=y_hat.dtype, device=y_hat.device)
        self.maskedWeight2 = self.maskedConv2.weight * self.maskedConv2.mask
        for resBlock in self.residualBlocks:
            resBlock.init_sequential(self.maskedConv2_input)

        self.sequential_init = True

    def context_forward(self, context, prev_subband=None, channel_idx=0):
        context_ret = []
        if prev_subband is not None:
            prev_subband = self.lower_level_subband_up(prev_subband)
            prev_subband = self.lower_level_subband_conv(prev_subband)
            context = torch.cat((context, prev_subband), dim=1)
        if channel_idx == 0:
            context = self.conv1_context(context)
        elif channel_idx == 1:
            context = self.conv1_context_cb(context)
        else:
            context = self.conv1_context_cr(context)

        context_ret.append(context)
        for idx, resBlock in enumerate(self.residualBlocksContext):
            context = resBlock(context)
            context_ret.append(context)
        return context_ret

    def forward_sequential(self, y_hat, h, w, context_list=None, channel_idx=0):
        if not self.sequential_init:
            self.init_sequential(y_hat)
        # input y_hat already divided by QP or rather dynamic_range in this case
        # y_hat was padded before
        kernel_size = 3  # kernel_size always 3 for now
        padding = 1
        y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size].detach().clone()
        # make sure that y_crop is zero in case mask is -0 or nonzero in non-causal regions
        # possible reason for decoder drift
        y_crop[self.mask1 == 0] = 0
        # masked conv 1
        tmp = F.conv2d(y_crop, self.maskedWeight1, bias=self.maskedConv1.bias)
        conv1 = tmp
        if context_list:
            tmp = tmp + context_list[0][:, :, h:h+1, w:w+1]

        for idx, resBlock in enumerate(self.residualBlocks):
            tmp = resBlock.forward_sequential(tmp, h, w, kernel_size, padding)
            if context_list:
                tmp = tmp + context_list[idx+1][:, :, h:h+1, w:w+1]
        tmp = tmp + conv1

        self.maskedConv2_input[:, :, h+padding:h+padding+1, w+padding:w+padding+1] = tmp[:, :, :, :]
        tmp = self.maskedConv2_input[:, :, h: h + kernel_size, w: w + kernel_size]
        tmp = F.conv2d(tmp, self.maskedWeight2, bias=self.maskedConv2.bias)
        tmp = self.lrelu(tmp)

        # finish with 1 x 1 convolutions, fully connected across channel dim and independent of spatial size
        num_convs = len(self.convs)
        for idx, curr_conv in enumerate(self.convs):
            tmp = curr_conv(tmp)
            if idx < num_convs-1:
                tmp = self.lrelu(tmp)

        return tmp
