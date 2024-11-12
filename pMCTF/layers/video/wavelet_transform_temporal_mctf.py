import math

import torch
import torch.nn as nn


from pMCTF.layers.lifting_1d import PredictUpdate
from pMCTF.layers.layers import RoundNoGradient


class TemporalLifting(nn.Module):
    def __init__(self, bitdepth=8, lossy=True, in_channels=1):
        super(TemporalLifting, self).__init__()
        self.bitdepth = bitdepth
        self.dynamic_range = float(2**self.bitdepth)

        self.scale = 0.1
        self.in_channels = in_channels
        self.lossy = lossy

        self.P_t = PredictUpdate(self.in_channels)
        self.U_t = PredictUpdate(self.in_channels)

        self.scale_p = torch.tensor(1/math.sqrt(2), requires_grad=True)
        self.scale_u = torch.tensor(0.5, requires_grad=True)

    def predict_filter(self, x):
        tmp = self.P_t(x)
        tmp = tmp * self.scale
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x = x + tmp
        if self.lossy:
            x = x * self.scale_p
        return x

    def update_filter(self, x):
        tmp = self.U_t(x)
        tmp = tmp * self.scale
        if not self.lossy:
            tmp = RoundNoGradient.apply(tmp)
        x = x + tmp
        if self.lossy:
            x = x * self.scale_u
        return x

