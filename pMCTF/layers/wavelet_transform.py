import torch
import torch.nn as nn
import numpy as np
from .lifting_1d import iWave1D
from .lifting_1d import Haar


class LiftingScheme2D(nn.Module):
    def __init__(self, non_separable=False, bitdepth=8, lossy=True, in_channels=1, haar=False):
        super(LiftingScheme2D, self).__init__()
        self.bitdepth = bitdepth
        self.dynamic_range = float(2**self.bitdepth)
        self.non_separable = non_separable

        if haar:
            self.lift_h = Haar(lossy=lossy)
            self.lift_v = self.lift_h
        else:
            self.lift_h = iWave1D(bitdepth=bitdepth, lossy=lossy, in_channels=in_channels)
            self.lift_v = iWave1D(bitdepth=bitdepth, lossy=lossy, in_channels=in_channels) \
                if non_separable else self.lift_h

        self.ll_subband = None

    def forward_lift_2d(self, x):
        # ROW FILTERING
        # tensor dimensions: batch size x channels x height/rows x width/columns
        # N x C x H x W
        l, h = self.lift_h.forward_lift(x)

        # COLUMN FILTERING
        l = l.permute((0, 1, 3, 2))
        ll, lh = self.lift_v.forward_lift(l)
        ll = ll.permute((0, 1, 3, 2))
        lh = lh.permute((0, 1, 3, 2))

        h = h.permute((0, 1, 3, 2))
        hl, hh = self.lift_v.forward_lift(h)
        hl = hl.permute((0, 1, 3, 2))
        hh = hh.permute((0, 1, 3, 2))

        subband_dict = {'ll': ll, 'lh': lh, 'hl': hl, 'hh': hh, 'l': l, 'h': h}
        return subband_dict

    def backward_lift_2d(self, subbands):
        ll = subbands['ll'].permute((0, 1, 3, 2))
        lh = subbands['lh'].permute((0, 1, 3, 2))
        l = self.lift_v.backward_lift(ll, lh)
        l = l.permute((0, 1, 3, 2))

        hl = subbands['hl'].permute((0, 1, 3, 2))
        hh = subbands['hh'].permute((0, 1, 3, 2))
        h = self.lift_v.backward_lift(hl, hh)
        h = h.permute((0, 1, 3, 2))

        x = self.lift_h.backward_lift(l, h)
        return x

    def backward_lift_2d_tmp(self, subbands):
        ll = subbands['ll'].permute((0, 1, 3, 2))
        lh = subbands['lh'].permute((0, 1, 3, 2))
        l = self.lift_v.backward_lift_tmp(ll, lh)
        l = l.permute((0, 1, 3, 2))

        hl = subbands['hl'].permute((0, 1, 3, 2))
        hh = subbands['hh'].permute((0, 1, 3, 2))
        h = self.lift_v.backward_lift_tmp(hl, hh)
        h = h.permute((0, 1, 3, 2))

        x = self.lift_h.backward_lift_tmp(l, h)
        return x

