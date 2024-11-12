import torch
import torch.nn as nn
import numpy as np
from .convs import get_conv2d

import torch.nn.functional as F
from pMCTF.layers.layers import RoundNoGradient, conv3x3


def split(x):
    x_e = x[:, :, ::2, :]  # x[n, 2m]
    x_o = x[:, :, 1::2, :]  # x[n, 2m+1]
    return x_e, x_o


def merge(x_e, x_o):
    dims = list(x_e.size())
    dims[2] = dims[2] * 2
    x = torch.zeros(dims).to(x_e.device)
    x[:, :, ::2, :] = x_e[:, :]
    x[:, :, 1::2, :] = x_o[:, :]
    return x


class PredictUpdate(nn.Module):
    def __init__(self, in_ch):
        super(PredictUpdate, self).__init__()
        self.in_channels = in_ch
        num_ch = 16
        self.conv1 = conv3x3(in_ch=in_ch, out_ch=num_ch)
        self.conv2 = conv3x3(in_ch=num_ch, out_ch=num_ch)
        self.conv3 = conv3x3(in_ch=num_ch, out_ch=num_ch)

        self.conv4 = conv3x3(in_ch=num_ch, out_ch=1)

    def forward(self, x):
        conv1 = self.conv1(x)

        x = torch.tanh(conv1)
        x = self.conv2(x)

        x = torch.tanh(x)
        x = self.conv3(x)

        x = conv1 + x

        x = self.conv4(x)

        return x


class iWave1D(nn.Module):
    """
         Prediction-first lifting scheme with uniform scaling after transform
         and rate-distortion loss
    """
    def __init__(self, in_channels=1, bitdepth=8, lossy=True):

        super(iWave1D, self).__init__()

        self.bitdepth = bitdepth
        self.dynamic_range = float(2**self.bitdepth)
        self.in_channels = in_channels
        self.lossy = lossy

        self.lifting_coeffs = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971,
                               0.869864451624781, 1.149604398860241]  # bior4.4

        # for skip path
        # weights have size (1, 1, 3, 1) = (out_channel, in_channel, kernel_size[0], kernel_size[1])
        init_weights = torch.tensor([[0.0], [self.lifting_coeffs[0]], [self.lifting_coeffs[0]]])
        self.conv_P1 = get_conv2d(kernel_size=3, in_ch=self.in_channels, out_ch=self.in_channels, padding=False,
                                  init_weights=init_weights.view(1, 1, 3, 1), kernel_size2=1, groups=self.in_channels,
                                  )

        init_weights = torch.tensor([[self.lifting_coeffs[1]], [self.lifting_coeffs[1]], [0.0]])
        self.conv_U1 = get_conv2d(kernel_size=3, in_ch=self.in_channels, out_ch=self.in_channels, padding=False,
                                  init_weights=init_weights.view(1, 1, 3, 1), kernel_size2=1, groups=self.in_channels,
                                  )

        init_weights = torch.tensor([[0.0], [self.lifting_coeffs[2]], [self.lifting_coeffs[2]]])
        self.conv_P2 = get_conv2d(kernel_size=3, in_ch=self.in_channels, out_ch=self.in_channels, padding=False,
                                  init_weights=init_weights.view(1, 1, 3, 1), kernel_size2=1, groups=self.in_channels,
                                  )

        init_weights = torch.tensor([[self.lifting_coeffs[3]], [self.lifting_coeffs[3]], [0.0]])
        self.conv_U2 = get_conv2d(kernel_size=3, in_ch=self.in_channels, out_ch=self.in_channels, padding=False,
                                  init_weights=init_weights.view(1, 1, 3, 1), kernel_size2=1, groups=self.in_channels,
                                  )

        self.reflectionPadSkip = nn.ReflectionPad2d((0, 0, 1, 1))

        self.P_1 = PredictUpdate(self.in_channels)
        self.P_2 = PredictUpdate(self.in_channels)
        self.U_1 = PredictUpdate(self.in_channels)
        self.U_2 = PredictUpdate(self.in_channels)

        self.scaling = True
        if self.scaling:
            self.scale_l = torch.tensor(self.lifting_coeffs[5], requires_grad=True) # self.lifting_coeffs[5]
            self.scale_h = torch.tensor(self.lifting_coeffs[4], requires_grad=True) # self.lifting_coeffs[4]

    def forward_lift(self, x):
        x_e, x_o = split(x)  # l, h
        skip_path = self.reflectionPadSkip(x_e)
        skip_path = self.conv_P1(skip_path)
        # P_1
        l = self.P_1(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + l * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_o = x_o + tmp  # h/x_o  -> x_o = x_o + P_1(x_e)

        skip_path = self.reflectionPadSkip(x_o)
        skip_path = self.conv_U1(skip_path)
        # U_1
        h = self.U_1(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + h * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_e = x_e + tmp  # l/x_e -> x_e = x_e + U_1(x_o)

        skip_path = self.reflectionPadSkip(x_e)
        skip_path = self.conv_P2(skip_path)
        # P_2
        l = self.P_2(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + l * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_o = x_o + tmp  # h/x_o -> x_o = x_o + P_2(x_e)

        skip_path = self.reflectionPadSkip(x_o)
        skip_path = self.conv_U2(skip_path)
        # U_2
        h = self.U_2(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + h * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_e = x_e + tmp  # l/x_e -> x_e = x_e + U_2(x_o)

        if self.scaling and self.lossy:
            x_e = x_e * self.scale_l
            x_o = x_o * self.scale_h

        return x_e, x_o  # l, h

    def backward_lift(self, l, h):
        if self.scaling and self.lossy:
            l = l / self.scale_l
            h = h / self.scale_h

        skip_path = self.reflectionPadSkip(h)
        skip_path = self.conv_U2(skip_path)
        # U_2
        h_tmp = self.U_2(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + h_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e  -> x_e = x_e - U_2(x_o)

        skip_path = self.reflectionPadSkip(l)
        skip_path = self.conv_P2(skip_path)
        # P_2
        l_tmp = self.P_2(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + l_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_2(x_e)

        skip_path = self.reflectionPadSkip(h)
        skip_path = self.conv_U1(skip_path)
        # U_1
        h_tmp = self.U_1(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + h_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e -> x_e = x_e - U_2(x_o)

        skip_path = self.reflectionPadSkip(l)
        skip_path = self.conv_P1(skip_path)
        # P_1
        l_tmp = self.P_1(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + l_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_1(x_e)

        x = merge(l, h)
        return x

    def backward_lift_tmp(self, l, h):
        if self.scaling and self.lossy:
            l = l / self.scale_l
            h = h / self.scale_h

        skip_path = F.pad(h, (0, 0, 1, 1, 0, 0, 0, 0))
        skip_path = self.conv_U2(skip_path)
        # U_2
        h_tmp = self.U_2(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + h_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e  -> x_e = x_e - U_2(x_o)

        skip_path = F.pad(l,  (0, 0, 1, 1, 0, 0, 0, 0))
        skip_path = self.conv_P2(skip_path)
        # P_2
        l_tmp = self.P_2(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + l_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_2(x_e)

        skip_path = F.pad(h,  (0, 0, 1, 1, 0, 0, 0, 0))
        skip_path = self.conv_U1(skip_path)
        # U_1
        h_tmp = self.U_1(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + h_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e -> x_e = x_e - U_2(x_o)

        skip_path = F.pad(l,  (0, 0, 1, 1, 0, 0, 0, 0))
        skip_path = self.conv_P1(skip_path)
        # P_1
        l_tmp = self.P_1(skip_path/self.dynamic_range) * self.dynamic_range
        tmp = skip_path + l_tmp * 0.1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_1(x_e)

        x = merge(l, h)
        return x


class Haar(nn.Module):
    """
         Haar wavelet transform - lifting implementation
    """
    def __init__(self, lossy=True):

        super(Haar, self).__init__()

        self.lossy = lossy

    def forward_lift(self, x):
        x_e, x_o = split(x)  # l, h

        # P = 1
        l = 1*x_e
        if not self.lossy:
            l = RoundNoGradient.apply(l)
        x_o = x_o + l  # h/x_o  -> x_o = x_o + P_1(x_e)

        # U = 1/2
        h = x_o * 0.5
        if not self.lossy:
            h = RoundNoGradient.apply(h)
        x_e = x_e + h  # l/x_e -> x_e = x_e + U_1(x_o)

        return x_e, x_o  # l, h

    def backward_lift(self, l, h):

        # U = 1/2
        tmp = 0.5*h
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e  -> x_e = x_e - U_2(x_o)

        # P = 1
        tmp = l * 1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_2(x_e)

        x = merge(l, h)
        return x


class CDF97(nn.Module):
    """
         Prediction-first lifting scheme with uniform scaling after transform
         and rate-distortion loss
    """
    def __init__(self, in_channels=1, bitdepth=8, lossy=True):
        super(CDF97, self).__init__()
        self.bitdepth = bitdepth
        self.dynamic_range = float(2**self.bitdepth)
        self.in_channels = in_channels
        self.lossy = lossy

        self.coeffs = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971,
                       0.869864451624781, 1.149604398860241]  # bior4.4

        # for skip path
        # weights have size (1, 1, 3, 1) = (out_channel, in_channel, kernel_size[0], kernel_size[1])
        self.register_buffer("P1_weights", torch.tensor([[0.0], [self.coeffs[0]], [self.coeffs[0]]]).view(1, 1, 3, 1))
        self.register_buffer("U1_weights", torch.tensor([[self.coeffs[1]], [self.coeffs[1]], [0.0]]).view(1, 1, 3, 1))
        self.register_buffer("P2_weights", torch.tensor([[0.0], [self.coeffs[2]], [self.coeffs[2]]]).view(1, 1, 3, 1))
        self.register_buffer("U2_weights", torch.tensor([[self.coeffs[3]], [self.coeffs[3]], [0.0]]).view(1, 1, 3, 1))

        self.reflectionPad = nn.ReflectionPad2d((0, 0, 1, 1))

        self.scale_l = self.coeffs[5]
        self.scale_h = self.coeffs[4]

    def forward_lift(self, x):
        x_e, x_o = split(x)  # l, h
        tmp = self.reflectionPad(x_e)
        tmp = F.conv2d(tmp, self.P1_weights)
        # P_1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_o = x_o + tmp  # h/x_o  -> x_o = x_o + P_1(x_e)

        tmp = self.reflectionPad(x_o)
        tmp = F.conv2d(tmp, self.U1_weights)
        # U_1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_e = x_e + tmp  # l/x_e -> x_e = x_e + U_1(x_o)

        tmp = self.reflectionPad(x_e)
        tmp = F.conv2d(tmp, self.P2_weights)
        # P_2
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_o = x_o + tmp  # h/x_o -> x_o = x_o + P_2(x_e)

        tmp = self.reflectionPad(x_o)
        tmp = F.conv2d(tmp, self.U2_weights)
        # U_2
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x_e = x_e + tmp  # l/x_e -> x_e = x_e + U_2(x_o)

        if self.lossy:
            x_e = x_e * self.scale_l
            x_o = x_o * self.scale_h

        return x_e, x_o  # l, h

    def backward_lift(self, l, h):
        if self.lossy:
            l = l / self.scale_l
            h = h / self.scale_h

        tmp = self.reflectionPad(h)
        tmp = F.conv2d(tmp, self.U2_weights)
        # U_2
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e  -> x_e = x_e - U_2(x_o)

        tmp = self.reflectionPad(l)
        tmp = F.conv2d(tmp, self.P2_weights)
        # P_2
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_2(x_e)

        tmp = self.reflectionPad(h)
        tmp = F.conv2d(tmp, self.U1_weights)
        # U_1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        l = l - tmp  # l/x_e -> x_e = x_e - U_2(x_o)

        tmp = self.reflectionPad(l)
        tmp = F.conv2d(tmp, self.P1_weights)
        # P_1
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        h = h - tmp  # h/x_o -> x_o = x_o - P_1(x_e)

        x = merge(l, h)
        return x