# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import random

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from pMCTF.utils.util import rgb2ycbcr, ycbcr2rgb
from torch.utils.data import Dataset


class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:
    .. code-block::
        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...
    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.
    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.
    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
        tuplet=7,
        max_frames=2,
        rgb=False,
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        list_path = Path(root) / self._list_filename(split, tuplet)

        if not list_path.is_file():
            raise RuntimeError(f'Invalid file "{list_path}"')

        with open(list_path) as f:
            self.sample_folders = [
                Path(root) / "sequences" / line.rstrip()  # im{idx}.png"
                for line in f
                if line.strip() != "" and (Path(root) / "sequences" / line.rstrip()).is_dir()
                # for idx in range(1, tuplet + 1)
            ]

        if split == "valid":
            self.sample_folders = self.sample_folders[:min(len(self.sample_folders), 100)]

        self.max_frames = max_frames  # Training GOP (group of pictures)
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform
        self.channel_idx = 0
        self.rgb = rgb
        self.ycbcr = False  # train on Y, Cb and Cr channel (False: Y only)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        frames = np.concatenate(
            [self._load_img(p) for p in frame_paths], axis=-1
        )
        frames = self.transform(frames)  # torch.chunk(-, self.max_frames)
        if self.max_frames != 1 and frames.dim() == 3:
            frames = frames.unsqueeze(0)

        if self.rnd_temp_order and isinstance(frames, list):
            if random.random() < 0.5:
                return frames[::-1]

        return frames  # 4-d tensor: Channels x Temporal/Depth x Height x Width

    def __len__(self):
        return len(self.sample_folders)

    def _load_img(self, p):
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        if not self.rgb:
            img = rgb2ycbcr(img)
            if not self.ycbcr:
                img = img[:, :, self.channel_idx, np.newaxis]
        return img

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        list_suffix = {"train": "trainlist", "valid": "testlist", "test": "testlist"}[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"
