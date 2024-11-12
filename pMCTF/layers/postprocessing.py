import torch.nn as nn
import torch
from .convs import get_conv2d


class ResBlock(nn.Module):
    def __init__(self, intermediate_channels):
        super(ResBlock, self).__init__()
        self.conv1 = get_conv2d(kernel_size=3, in_ch=intermediate_channels, out_ch=intermediate_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = get_conv2d(kernel_size=3, in_ch=intermediate_channels, out_ch=intermediate_channels)

    def forward(self, input):
        output = self.conv1(input)
        output = self.lrelu(output)
        output = self.conv2(output)
        return output + input


class PostProcess(nn.Module):
    """
         Post-Processing/De-quantization module (iWave++) or quality enhancament, also used by ANFIC
    """
    def __init__(self, num_res=6, intermediate_channels=64, in_channels=1, out_channels=1):
        super(PostProcess, self).__init__()
        self.num_res = num_res

        self.resBlocks = nn.ModuleList(ResBlock(intermediate_channels) for _ in range(num_res))

        # remaining layers
        self.conv1 = get_conv2d(kernel_size=3, in_ch=in_channels, out_ch=intermediate_channels)
        self.conv2 = get_conv2d(kernel_size=3, in_ch=intermediate_channels, out_ch=intermediate_channels)
        self.conv3 = get_conv2d(kernel_size=3, in_ch=intermediate_channels, out_ch=out_channels)

    def forward(self, x):
        tmp = self.conv1(x)  # conv1
        conv1 = tmp
        for idx, resBlock in enumerate(self.resBlocks):
            tmp = resBlock(tmp)

        tmp = self.conv2(tmp) + conv1
        tmp = self.conv3(tmp)

        return x + tmp


class PostProcessCtx(nn.Module):
    """
         Post-Processing/De-quantization module (iWave++) or quality enhancament, also used by ANFIC
         Extension to recon generation net (DMC)
    """
    def __init__(self, intermediate_channels=64, in_channels=1, out_channels=1, ctx_channel=64):
        super(PostProcessCtx, self).__init__()

        self.conv1 = get_conv2d(kernel_size=3, in_ch=in_channels + ctx_channel, out_ch=intermediate_channels)
        self.resBlocks = nn.Sequential(*[ResBlock(intermediate_channels) for _ in range(6)])
        # TODO use unet?

        self.conv2 = get_conv2d(kernel_size=3, in_ch=intermediate_channels, out_ch=intermediate_channels)
        self.conv3 = get_conv2d(kernel_size=3, in_ch=intermediate_channels, out_ch=out_channels)

    def forward(self, x, ctx):
        tmp = self.conv1(torch.cat((ctx, x), dim=1))  # conv1
        conv1 = tmp
        tmp = self.resBlocks(tmp)

        feature = self.conv2(tmp) + conv1
        tmp = self.conv3(feature)

        recon = x + tmp
        return feature, recon
