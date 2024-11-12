# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn

from .entropy_models import GaussianEncoder, EntropyCoder
from pMCTF.layers.lifting_1d import RoundNoGradient


class CompressionModel(nn.Module):
    def __init__(self, y_distribution, ec_thread=False, stream_part=1):
        super().__init__()

        self.y_distribution = y_distribution
        self.entropy_coder = None
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.masks = {}

    def quant(self, x):
        if self.training:
            return RoundNoGradient.apply(x)
        else:
            return torch.round(x)

    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale[q_index]
        return q_basic * q_scale

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.clamp_min(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)

    def process(self, y, means):
        y_q = self.quant(y)
        y_res = (y_q - means)
        y_hat = y_res + means
        return y_res, y_q, y_hat

    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        noise = noise.clone().detach()
        return x + noise
