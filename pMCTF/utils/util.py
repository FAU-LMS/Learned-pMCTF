"""This module contains simple helper functions """
from __future__ import print_function
import torch
from PIL import Image
import math
import importlib
from torch import Tensor
import argparse
from typing import Tuple, Union
import matplotlib.pyplot as plt

import sys
import numpy as np
import logging
import traceback
from mpl_toolkits import axes_grid1
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def rgb2ycbcr(rgb, bitdpeth=8):
    """Convert RGB images to JPEG YCbCr color format. Y, Cb and Cr have the full value range.
        Args:
            rgb: tensor of shape C x H x W or numpy array of shape H X W X C, where C=3
    """
    delta = 128 if bitdpeth == 8 else 32768
    if isinstance(rgb, np.ndarray):
        ycbcr = np.zeros_like(rgb)
    else:
        # torch.tensor
        rgb = rgb.permute(1, 2, 0)
        ycbcr = torch.zeros_like(rgb)

    ycbcr[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ycbcr[:, :, 1] = (rgb[:, :, 2] - ycbcr[:, :, 0]) * 0.564 + delta
    ycbcr[:, :, 2] = (rgb[:, :, 0] - ycbcr[:, :, 0]) * 0.713 + delta

    if isinstance(ycbcr, torch.Tensor):
        return ycbcr.permute(2, 0, 1)
    return ycbcr


def ycbcr2rgb(ycbcr, bitdpeth=8):
    """Convert JPEG YCbCr image to RGB color format.
        Args:
            rgb: tensor of shape C x H x W or numpy array of shape H X W X C, where C=3
    """
    delta = 128 if bitdpeth == 8 else 32768
    if isinstance(ycbcr, np.ndarray):
        rgb = np.zeros_like(ycbcr)
    elif len(ycbcr.size()) == 4:
        # torch.tensor
        rgb = torch.zeros_like(ycbcr)
        for idx in range(ycbcr.size(0)):
            rgb[idx, 0, :, :] = ycbcr[idx, 0, :, :] + 1.403 * (ycbcr[idx, 2, :, :] - delta)
            rgb[idx, 1, :, :] = ycbcr[idx, 0, :, :] - 0.714 * (ycbcr[idx, 2, :, :] - delta) - 0.344 * (ycbcr[idx, 1, :, :] - delta)
            rgb[idx, 2, :, :] = ycbcr[idx, 0, :, :] + 1.773 * (ycbcr[idx, 1, :, :] - delta)
        return rgb
    else:
        ycbcr = ycbcr.permute(1, 2, 0)
        rgb = torch.zeros_like(ycbcr)

    rgb[:, :, 0] = ycbcr[:, :, 0] + 1.403 * (ycbcr[:, :, 2] - delta)
    rgb[:, :, 1] = ycbcr[:, :, 0] - 0.714 * (ycbcr[:, :, 2] - delta) - 0.344 * (ycbcr[:, :, 1] - delta)
    rgb[:, :, 2] = ycbcr[:, :, 0] + 1.773 * (ycbcr[:, :, 1] - delta)

    if isinstance(rgb, torch.Tensor):
        return rgb.permute(2, 0, 1)

    return rgb


def ycbcr2rgb_709(ycbcr):
    """Convert 3D YCbCr 4:4:4 image array to 3D RGB image array.
    Different conversion compared to ycbcr2rgb_citrix(...)!

    :param ndarray: 3D RGB array
    :param int flavor: flavor of conversion (601 for BT.601, 709 for BT.709)
    :return: 3D YCbCr array
    :rtype: ndarray
    """
    if len(ycbcr.size()) == 4:
        ycbcr = ycbcr[0, :, :, :].permute(1, 2, 0)
        ycbcr = ycbcr.cpu().numpy()

    height, width, channels = ycbcr.shape
    assert channels == 3

    Y, Cb, Cr = np.dsplit(ycbcr.astype(np.int32), 3)
    C = Y - 16
    D = Cb - 128
    E = Cr - 128

    R = ((298 * C + 459 * E + 128) >> 8)
    G = ((298 * C - 55 * D - 136 * E + 128) >> 8)
    B = ((298 * C + 541 * D + 128) >> 8)

    rgb = np.dstack((R, G, B))

    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    rgb_out = torch.from_numpy(rgb).float()
    rgb_out = rgb_out.permute(2, 0, 1)
    return rgb_out.unsqueeze(0)


def yuv_420_to_444(
    yuv: Tuple[Tensor, Tensor, Tensor],
    mode: str = "bilinear",
    return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    Args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    if mode in ("bilinear", "nearest"):

        def _upsample(tensor):
            return F.interpolate(tensor, scale_factor=2, mode=mode, align_corners=False)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)




def rgb2yuv_lossless(x):
    x = x.int()

    r, g, b = x.chunk(3, -3)

    co = r - b
    tmp = b + torch.bitwise_right_shift(co, 1)
    cg = g - tmp
    y = tmp + torch.bitwise_right_shift(cg, 1)

    yuv = torch.cat((y, co, cg), dim=-3)

    return yuv.float()


def yuv2rgb_lossless(x):
    x = x.int()

    y, co, cg = x.chunk(3, -3)

    tmp = y - torch.bitwise_right_shift(cg, 1)
    g = cg + tmp
    b = tmp - torch.bitwise_right_shift(co, 1)
    r = b + co
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb.float()


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def image_export(Y, U, V, filename, access_mode='wb'):
    # write yuv image
    try:
        f = open(filename, access_mode)
    except:
        print('Could not open '+filename)

    if Y.dtype == np.uint8:
        Y.astype(np.uint8).tofile(f)
        U.astype(np.uint8).tofile(f)
        V.astype(np.uint8).tofile(f)
    else:
        Y.astype(np.uint16).tofile(f)
        U.astype(np.uint16).tofile(f)
        V.astype(np.uint16).tofile(f)

    f.close()


def image_export_400(Y, filename, access_mode='wb', header=None):
    # write yuv image
    try:
        f = open(filename, access_mode)
    except:
        print('Could not open '+filename)

    if header is not None:
        f.write(header.encode())
    if isinstance(Y, list):
        for im in Y:
            im.tofile(f)
        f.close()
        return

    if Y.dtype == np.uint8:
        Y.astype(np.uint8).tofile(f)
    elif Y.dtype == np.int:
        Y.astype(np.int).tofile(f)
    else:
        Y.astype(np.uint16).tofile(f)
    f.close()


def image_import(filename, width, height, POC=0, bitdepth=np.uint8, colorformat=420, as444=False, as420=False):
    assert(bitdepth == np.uint8 or bitdepth == np.uint16 or np.int16)
    assert(colorformat == 420 or colorformat == 444 or colorformat == 400)

    try:
        f = open(filename, "rb")

        bytes_per_sample = 1
        if bitdepth in [np.uint16, np.int16]:
            bytes_per_sample = 2

        # go to desired position
        if colorformat == 420:
            f.seek(int(width * height * 1.5 * POC * bytes_per_sample))
        if colorformat == 444:
            f.seek(width * height * 3 * POC * bytes_per_sample)
        if colorformat == 400:
            f.seek(width * height * POC * bytes_per_sample)

        count = width * height
        Y = np.fromfile(f, dtype=bitdepth, count=int(count))
        Y = Y.reshape(height, width)

        if colorformat == 420:
            width /= 2
            height /= 2
            width = int(width)
            height = int(height)
            count = int(width * height)
        elif colorformat == 400:
            f.close()
            if as420:
                width = int(height/2)
                height = int(height/2)
                Cb = np.zeros((height, width), dtype=bitdepth)
                Cr = np.zeros((height, width), dtype=bitdepth)
                return (Y, Cb, Cr)
            return Y
        Cb = np.fromfile(f, dtype=bitdepth, count=count)
        Cr = np.fromfile(f, dtype=bitdepth, count=count)
        Cb = Cb.reshape(height, width)
        Cr = Cr.reshape(height, width)

        f.close()

        if as444:
            if colorformat==444:
                return np.dstack((Y,Cb,Cr))
            else:
                # convert to 444 image
                return None  # conversion.YCbCr4202YCbCr444(Y, Cb, Cr, bitdepth=bitdepth)
        else: # return channels seperately for 420
            return (Y, Cb, Cr)

    except Exception as e:
        logging.error(traceback.format_exc())
        print('Could not open ' + filename)
        sys.exit()


def tensor2numpy(tensor, batchidx=0):
    return np.transpose(tensor[batchidx, :, :, :].cpu().detach().numpy(), [1, 2, 0])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def normalize_tensor(im, im_name='lh'):
    """Normalize tensor value range to [-1, 1] for plotting"""
    im_max = torch.max(im)
    im_min = torch.min(im)
    if im_name in ['ll', 'x', 'x_hat']:
        outputHigh = 1
        outputLow = -1
        if im_max > 1 or im_min < -1:
            im = (outputHigh - outputLow) * (im - im_min) / (im_max-im_min) + outputLow
    else:
        if torch.abs(im_max) > torch.abs(im_min):
            outputHigh = 1
            outputLow = torch.sign(im_min) * torch.abs(im_min)/torch.abs(im_max)
        else:
            im = im*-1
            im_max = torch.max(im)
            im_min = torch.min(im)
            outputHigh = 1
            outputLow = torch.sign(im_min) * torch.abs(im_min)/torch.abs(im_max)
        if im_max > 1 or im_min < -1:
            im = (outputHigh - outputLow) *(im - im_min)/(im_max-im_min) + outputLow
    return im


def normalize_np(im):
    """Normalize tensor value range to [-1, 1] for plotting"""
    if isinstance(im, torch.Tensor):
        im_max = torch.max(im)
        im_min = torch.min(im)
        if im_max > 1 or im_min < -1:
            im = 2*(im - im_min)/(im_max-im_min) - 1
        return im.cpu().numpy()

    im_max = np.max(im[:])
    im_min = np.min(im[:])

    if np.abs(im_max) > np.abs(im_min):
        outputHigh = 1
        outputLow = np.sign(im_min) * np.abs(im_min) / np.abs(im_max)
    else:
        outputHigh = np.sign(im_max) * np.abs(im_max) / np.abs(im_min)
        outputLow = -1

    if im_max > 1 or im_min < -1:
        if im_min >= 0:
            im = (im - im_min)/(im_max-im_min)
        else:
            im = (outputHigh-outputLow)*(im - im_min)/(im_max-im_min) + outputLow

    if im_max < 1:
        im_max = im_max + 2
        im_min = im_min + 2
        im = im + 2
        im = (im - im_min) / (im_max - im_min)

    return im


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plotImage(tensor, idx, plt_show=True, fignum=1):
    imgNP = np.transpose(tensor[idx, :, :, :].cpu().detach().numpy(), [1, 2, 0])

    plt.figure(fignum)
    plt.imshow(imgNP)
    if plt_show:
        plt.show()


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if normalize:
            image_numpy = normalize_tensor(image_tensor[0]).clamp(-1.0, 1.0).cpu().float().numpy()
        else:
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt
