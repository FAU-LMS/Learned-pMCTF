import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from pMCTF.layers.video.layers import subpel_conv1x1, conv3x3,\
    ResidualBlockWithStride, ResidualBlockUpsample, DepthConvBlock


backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU


# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
# pylint: enable=W0221


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=feature.device, dtype=feature.dtype).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=feature.device, dtype=feature.dtype).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


class MEBasic(nn.Module):
    def __init__(self, in_ch=8):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x, modes=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self, in_ch=8, L=6):
        super().__init__()
        self.L = L
        self.moduleBasic = torch.nn.ModuleList([MEBasic(in_ch=in_ch) for _ in range(self.L)])

    def forward(self, im1, im2, modes=None):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                   flow_warp(im2_list[img_index], flow_up),
                                                   flow_up], 1), modes)

        return flow


class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        # self.adaptor_1 = DepthConvBlock(channel * 2, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        out = self.enc_2(out)
        out = self.adaptor_0(out)
        return self.enc_3(out)


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

    def forward(self, x, quant_step):
        feature = self.dec_1(x)
        out = self.dec_2(feature)
        out = out * quant_step
        mv = self.dec_3(out)
        return mv  # , feature


def get_hyper_enc_model(y_channel, z_channel):
    enc = nn.Sequential(
        conv3x3(y_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
    )
    return enc


def get_hyper_dec_model(y_channel, z_channel):
    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel, y_channel * 3 // 2),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel * 3 // 2, y_channel * 3 // 2, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel * 3 // 2, y_channel * 2),
    )

    return dec


