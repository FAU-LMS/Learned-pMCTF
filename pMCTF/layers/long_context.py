import torch
import torch.nn as nn

from pMCTF.utils.util import plotImage, normalize_tensor
from .convs import get_conv2d


class LSTM2D(nn.Module):
    """ convolutional LSTM as used by iWave++ (from pixelRNN paper)"""
    def __init__(self, input_channels, hidden_size):
        super(LSTM2D, self).__init__()
        # https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-recurrent-neural-networks-part-3/
        self.conv_in = get_conv2d(kernel_size=3, in_ch=input_channels, out_ch=hidden_size)
        self.conv_hidden = get_conv2d(kernel_size=3, in_ch=hidden_size, out_ch=hidden_size)

    def forward(self, x, hidden, cell_state):
        x = self.conv_in(x)
        hidden = self.conv_hidden(hidden)

        x_h = x + hidden

        # regular LSTM: fully connected layers applied to concatenated input for obtaining
        # input+forget gate, c_tilde and o
        forget_gate = torch.sigmoid(x_h)
        input_gate = torch.sigmoid(x_h)
        c_tilde = torch.tanh(x_h)  # = g

        # element wise multiplication
        cell_state = forget_gate*cell_state + input_gate*c_tilde

        o = torch.sigmoid(x_h)
        hidden = o * torch.tanh(cell_state)
        return hidden, cell_state


def get_upsample_conv(num_channels):
    return nn.ConvTranspose2d(num_channels, num_channels, kernel_size=(3, 3),
                              stride=(2, 2), output_padding=(1, 1), padding=(1, 1))


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, mode="nearest"):
        super(UpsampleModule, self).__init__()
        if mode == "transpose":
            # iWave++ default
            self.up = nn.Identity()
            self.conv = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=(3, 3),
                                           stride=(2, 2), output_padding=(1, 1), padding=(1, 1))
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv = get_conv2d(3, num_channels, num_channels)
        elif mode == 'bilinear':
            # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/12
            # align_corners=True important for translation equivariance
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = get_conv2d(3, num_channels, num_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class SubbandContext(nn.Module):
    def __init__(self, in_channels=1, decomp_levels=4,
                 ctx_per_band=True, init_states=False):
        super(SubbandContext, self).__init__()

        self.decomp_levels = decomp_levels
        self.init_states = init_states
        self.out_channels = 3*in_channels if ctx_per_band else in_channels
        self.ctx_per_band = ctx_per_band
        self.in_channels = in_channels

        hidden_size=32
        self.hidden_size = hidden_size

        self.LSTM1 = LSTM2D(in_channels, hidden_size=hidden_size)
        self.LSTM2 = LSTM2D(hidden_size, hidden_size=hidden_size)
        self.LSTM3 = LSTM2D(hidden_size, hidden_size=self.out_channels)

        self.sequential_init = False
        self.lstm1, self.lstm2, self.lstm3 = None, None, None
        if self.decomp_levels > 1:
            self.deconv_h1 = nn.ModuleList(UpsampleModule(hidden_size)
                                           for _ in range(self.decomp_levels - 1))
            self.deconv_c1 = nn.ModuleList(UpsampleModule(hidden_size)
                                           for _ in range(self.decomp_levels - 1))
            self.deconv_h2 = nn.ModuleList(UpsampleModule(hidden_size)
                                           for _ in range(self.decomp_levels - 1))
            self.deconv_c2 = nn.ModuleList(UpsampleModule(hidden_size)
                                           for _ in range(self.decomp_levels - 1))
            self.deconv_h3 = nn.ModuleList(UpsampleModule(self.out_channels)
                                           for _ in range(self.decomp_levels - 1))
            self.deconv_c3 = nn.ModuleList(UpsampleModule(self.out_channels)
                                           for _ in range(self.decomp_levels - 1))

    def context_one_band(self, x, lstm1, lstm2, lstm3):
        h1, c1 = self.LSTM1(x, *lstm1)
        h2, c2 = self.LSTM2(h1, *lstm2)
        h3, c3 = self.LSTM3(h2, *lstm3)
        return [h1, c1], [h2, c2], [h3, c3]

    def forward(self, subband_dict, init_states=None):
        hidden_states = {idx: {} for idx in range(self.decomp_levels)}
        ll = subband_dict[self.decomp_levels-1]['ll']
        subband_shape = list(ll.size())
        if self.ctx_per_band:
            subband_shape[1] = 3*subband_shape[1]
        if init_states:
            lstm1, lstm2, lstm3 = init_states["lstm1"], init_states["lstm2"], init_states["lstm3"]
        else:
            lstm3 = [torch.zeros(size=subband_shape, dtype=ll.dtype, device=ll.device),
                     torch.zeros(size=subband_shape, dtype=ll.dtype, device=ll.device)]
            subband_shape[1] = self.hidden_size  # hidden state feature maps
            lstm1 = [torch.zeros(size=subband_shape, dtype=ll.dtype, device=ll.device),
                     torch.zeros(size=subband_shape, dtype=ll.dtype, device=ll.device)]
            lstm2 = [torch.zeros(size=subband_shape, dtype=ll.dtype, device=ll.device),
                     torch.zeros(size=subband_shape, dtype=ll.dtype, device=ll.device)]
        lstm1, lstm2, lstm3 = self.context_one_band(ll, lstm1, lstm2, lstm3)
        hidden_states[self.decomp_levels-1]['lh'] = lstm3[0]

        if self.init_states:
            hidden_states["init_next"] = {
                "lstm1": [lstm_state.detach().clone() for lstm_state in lstm1],
                "lstm2": [lstm_state.detach().clone() for lstm_state in lstm2],
                "lstm3": [lstm_state.detach().clone() for lstm_state in lstm3],
            }
            hidden_states[self.decomp_levels-1]["ll"] = lstm3[0]

        for lvl in range(self.decomp_levels-1, -1, -1):
            lh = subband_dict[lvl]['lh']
            lstm1, lstm2, lstm3 = self.context_one_band(lh, lstm1, lstm2, lstm3)
            hidden_states[lvl]['hl'] = lstm3[0]

            hl = subband_dict[lvl]['hl']
            lstm1, lstm2, lstm3 = self.context_one_band(hl, lstm1, lstm2, lstm3)
            hidden_states[lvl]['hh'] = lstm3[0]

            hh = subband_dict[lvl]['hh']
            lstm1, lstm2, lstm3 = self.context_one_band(hh, lstm1, lstm2, lstm3)

            if lvl > 0:
                lstm1[0] = self.deconv_h1[lvl-1](lstm1[0])
                lstm1[1] = self.deconv_c1[lvl-1](lstm1[1])

                lstm2[0] = self.deconv_h2[lvl-1](lstm2[0])
                lstm2[1] = self.deconv_c2[lvl-1](lstm2[1])

                lstm3[0] = self.deconv_h3[lvl-1](lstm3[0])
                lstm3[1] = self.deconv_c3[lvl-1](lstm3[1])
                hidden_states[lvl-1]['lh'] = lstm3[0]

        return hidden_states

    def init_sequential(self, subband_size, device, init_states=None):
        if init_states:
            self.lstm1, self.lstm2, self.lstm3 = init_states["lstm1"], init_states["lstm2"], init_states["lstm3"]
        else:
            subband_size_ = subband_size.copy()
            if self.ctx_per_band:
                subband_size_[1] = 3
            self.lstm3 = [torch.zeros(size=subband_size_,  dtype=torch.float32, device=device),
                          torch.zeros(size=subband_size,  dtype=torch.float32, device=device)]
            subband_size_[1] = self.hidden_size # hidden state feature maps
            self.lstm1 = [torch.zeros(size=subband_size_, dtype=torch.float32, device=device),
                          torch.zeros(size=subband_size_, dtype=torch.float32, device=device)]
            self.lstm2 = [torch.zeros(size=subband_size_, dtype=torch.float32, device=device),
                          torch.zeros(size=subband_size_, dtype=torch.float32, device=device)]
        self.sequential_init = True

    def forward_sequential(self, subband, subband_name, lvl):
        # subband = reconstructed subband processed before "subband_name"
        self.lstm1, self.lstm2, self.lstm3 = self.context_one_band(subband, self.lstm1, self.lstm2, self.lstm3)

        if subband_name == 'lh' and lvl != (self.decomp_levels-1):
            # get current context by processing previous subband in coding order
            # for lh, previous subband is from previous decomposition level
            self.lstm1[0] = self.deconv_h1[lvl](self.lstm1[0])
            self.lstm1[1] = self.deconv_c1[lvl](self.lstm1[1])

            self.lstm2[0] = self.deconv_h2[lvl](self.lstm2[0])
            self.lstm2[1] = self.deconv_c2[lvl](self.lstm2[1])

            self.lstm3[0] = self.deconv_h3[lvl](self.lstm3[0])
            self.lstm3[1] = self.deconv_c3[lvl](self.lstm3[1])
        if subband_name == 'lh' and lvl == self.decomp_levels-1:
            # subband = LL = previously decoded subband
             init_next = {
                "lstm1": [lstm_state.detach().clone() for lstm_state in self.lstm1],
                "lstm2": [lstm_state.detach().clone() for lstm_state in self.lstm2],
                "lstm3": [lstm_state.detach().clone() for lstm_state in self.lstm3],
            }
        else:
            init_next = None

        return {"context": self.lstm3[0], "init_next": init_next}

    def forward_one_subband(self, subband, subband_name, lvl):
        # subband = reconstructed subband processed before "subband_name"
        self.lstm1, self.lstm2, self.lstm3 = self.context_one_band(subband, self.lstm1, self.lstm2, self.lstm3)

        if subband_name == 'hh' and lvl > 0:
            # get current context by processing previous subband in coding order
            # for lh, previous subband is from previous decomposition level
            self.lstm1[0] = self.deconv_h1[lvl-1](self.lstm1[0])
            self.lstm1[1] = self.deconv_c1[lvl-1](self.lstm1[1])

            self.lstm2[0] = self.deconv_h2[lvl-1](self.lstm2[0])
            self.lstm2[1] = self.deconv_c2[lvl-1](self.lstm2[1])

            self.lstm3[0] = self.deconv_h3[lvl-1](self.lstm3[0])
            self.lstm3[1] = self.deconv_c3[lvl-1](self.lstm3[1])
        if subband_name == 'll' and lvl == self.decomp_levels-1:
            # subband = LL = previously decoded subband
             init_next = {
                "lstm1": [lstm_state.detach().clone() for lstm_state in self.lstm1],
                "lstm2": [lstm_state.detach().clone() for lstm_state in self.lstm2],
                "lstm3": [lstm_state.detach().clone() for lstm_state in self.lstm3],
            }
        else:
            init_next = None

        return {"context": self.lstm3[0], "init_next": init_next}


