import torch.nn as nn
import torch
from copy import deepcopy
from typing import Any
from torch import Tensor
import torch.nn.functional as F
from .layers import MaskedConv2d


class DynamicDWConv(nn.Module):
    """ Homogeneous Dynamic Convolution
        from https://github.com/Atten4Vis/DemystifyLocalViT
        weights are shared across positions

        Usage in DW Block:

        if dynamic:
            self.conv = DynamicDWConv(dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
        else :
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)

        """
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        if self.dim < 4:
            reduction = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # output size (1,1) -> Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # linear projection (1x1 conv) to reduce channel dim
        self.conv1 = nn.Conv2d(self.dim, self.dim // reduction, kernel_size=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(self.dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        # linear projection to generate kernel weights
        self.conv2 = nn.Conv2d(self.dim // reduction,
                               self.dim * kernel_size * kernel_size,
                               kernel_size=(1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.dim))
        else:
            self.bias = None

    def forward(self, x, onehot=None):
        b, c, h, w = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding,
                     groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])

        # weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        # (out_channel, in_channel, kernel_size, kernel_size)
        #weight = weight.view(self.out_dim, self.in_dim // self.groups, self.kernel_size, self.kernel_size)
        #x = F.conv2d(x, weight, self.bias, stride=self.stride,
        #             padding=self.padding, groups=self.groups)
        return x


class MaskedConv2dDynamicDW(nn.Module):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, dim, kernel_size, mask_type="A",
                 bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        if self.dim < 4:
            reduction = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # output size (1,1) -> Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # linear projection (1x1 conv) to reduce channel dim
        self.conv1 = nn.Conv2d(self.dim, self.dim // reduction, kernel_size=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(self.dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        # linear projection to generate kernel weights
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, kernel_size=(1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.dim))
        else:
            self.bias = None

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer("mask", torch.ones((self.dim, 1, kernel_size, kernel_size)))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        weight *= self.mask.repeat(b, 1, 1, 1)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


def get_conv2d(kernel_size: int, in_ch: int, out_ch: int, stride: int = 1, conditional: bool = False,
               init_weights: torch.Tensor = None, padding: bool = True, kernel_size2: int = 0,
               dynamic: bool = False, groups: int = 1) -> nn.Module:
    """2d convolution with padding"""
    kernel_dims = (kernel_size, kernel_size) if kernel_size2 == 0 else (kernel_size, kernel_size2)
    if conditional:
        ret = ConditionalConv(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_dims, padding=padding)
        if init_weights is not None:
            ret.conv2d.weight.data = init_weights
    else:
        padding = (kernel_size - 1) // 2 if padding else 0
        if dynamic and in_ch == out_ch:
            # in_ch if out_ch >= in_ch else 1
            ret = DynamicDWConv(in_ch, kernel_size, padding=padding, groups=in_ch)
        else:
            ret = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_dims, stride=(stride, stride),
                            padding=(padding, padding), groups=groups)
        if init_weights is not None and not isinstance(ret, DynamicDWConv):
            if groups > 1:
                init_weights = torch.repeat_interleave(init_weights, groups, dim=0)
            ret.weight.data = init_weights
    return ret


class ConditionalConv(nn.Module):
    """
        Conditional Convolution for conditioning on the RD parameter lambda from a finite set of lambdas
        (conditioning on continuous lambda did not work well for Choi et al.)
        Current lambda is one hot encoded in a vector of length num_lambdas
        bias and scaling parameter are conditioned on lambda

        Loss function: average over all lambda values for every iteration
        Real training: Randomly select lambda from all available lambda values (uniform random)
                       for each image (additionally select quantization bin randomly for every image)

        Additional rate adaption: mixed bin sizes (quantization step size)
    """

    def __init__(self, in_channels, out_channels, kernel_size, mask_type="A", num_lambdas=5, padding=True, masked=False):
        super(ConditionalConv, self).__init__()

        # two fully connected layers receiving the one-hot encoded lambda value as input
        self.fc_bias = nn.Linear(in_features=num_lambdas, out_features=out_channels)
        self.fc_scaling = nn.Linear(in_features=num_lambdas, out_features=out_channels)
        padding = kernel_size[0]//2 if padding else 0
        self.sp = nn.Softplus()
        if not masked:
            self.conv2d = nn.Conv2d(kernel_size=kernel_size, padding=(padding, padding),
                                    in_channels=in_channels, out_channels=out_channels, bias=False)
        else:
            self.conv2d = MaskedConv2d(kernel_size=kernel_size, padding=(padding, padding), in_channels=in_channels,
                                       out_channels=out_channels, mask_type=mask_type, bias=False)

    def forward(self, x, onehot):
        # one scaling value for every conv2d output channel
        scaling = self.sp(self.fc_scaling(onehot))
        # one bias value for every conv2d output channel
        bias = self.fc_bias(onehot)
        return scaling[:, :, None, None]*self.conv2d(x) + bias[:, :, None, None]

    # TODO: forward sequential


def get_masked_conv2d(kernel_size: int, in_ch: int, out_ch: int, stride: int = 1, conditional: bool = False,
                      padding: bool = True, kernel_size2: int = 0, mask_type: str = "A") -> nn.Module:
    """2d masked convolution with padding"""
    kernel_dims = (kernel_size, kernel_size) if kernel_size2 == 0 else (kernel_size, kernel_size2)
    if conditional:
        return ConditionalConv(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_dims, padding=padding,
                               mask_type=mask_type, masked=True)
    else:
        padding = (kernel_size - 1) // 2 if padding else 0
        return MaskedConv2d(in_ch, out_ch, kernel_size=kernel_dims, stride=(stride, stride), mask_type=mask_type,
                            padding=(padding, padding))


def get_conv3d(kernel_size: int, in_ch: int, out_ch: int, stride: int = 1,
               init_weights: torch.Tensor = None, padding: bool = True, kernel_size2: int = 0,
               depthwise: bool = False, masked: bool = False, mask_type: str = "A") -> nn.Module:
    """ 3d convolution with padding
        Pytorch 3d conv input (N,C_in,D,H,W) -> output  (N,C_out,D_out,H_out,W_out)
    """
    if depthwise:
        groups = in_ch
        # and out_ch are a multiple of in_ch
    else:
        groups = 1
    # kernel size for dimension (depth, height, width)
    # kernel_size2 only for 3 times 1 convs
    kernel_dims = (kernel_size, kernel_size, kernel_size) if kernel_size2 == 0 \
        else (kernel_size, kernel_size, kernel_size2)

    padding = (kernel_size - 1) // 2 if padding else 0

    if masked:
        ret = MaskedConv3d(in_ch, out_ch, kernel_size=kernel_dims, stride=(stride, stride, stride),
                           padding=(padding, padding, padding), groups=groups, mask_type=mask_type)
    else:
        # zero padding in all 3 dimensions
        ret = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_dims, stride=(stride, stride, stride),
                        padding=(padding, padding, padding), groups=groups)
    if init_weights is not None and not isinstance(ret, DynamicDWConv):
        ret.weight.data = init_weights
    return ret


class MaskedConv3d(nn.Conv3d):
    r"""Masked 3D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, d, h, w = self.mask.size()
        assert d == 3
        self.mask[:, :, 1, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, 1, h // 2 + 1:] = 0
        self.mask[:, :, 2, :, :] = 0

    def forward(self, x: Tensor, onehot: Any = None) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)