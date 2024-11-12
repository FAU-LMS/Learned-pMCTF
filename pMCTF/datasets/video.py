# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pMCTF.utils.util import rgb2ycbcr, rgb2yuv_lossless
import random
import glob


class VideoYCbCr(Dataset):
    """ Load vimeo-septuplet
    """
    def __init__(self,
                 rootpath,
                 num_frames=2,
                 patchsize=256,
                 split="train",
                 rnd_interval=False,
                 lossless=False,
                 transform=None,
                 use_idx_list=False,
                 rnd_temp_order=False):

        self.patchsize = patchsize

        self.num_frames = num_frames
        self.num_frames_max = num_frames
        self.rnd_interval = rnd_interval
        self.use_idx_list = use_idx_list
        self.rnd_temp_order = rnd_temp_order
        self.lossless = lossless
        self.current_interval = 1
        self.frame_ids_curr = None

        self.sample_folders = list(glob.iglob(str(Path(rootpath) / "sequences") + '/*/*/', recursive=True))

        if split == "valid":
            self.sample_folders = self.sample_folders[:min(len(self.sample_folders), 10)]

        self.transform = transform
        self.max_interval = 1

    def update_num_frames(self, num_frames, logger):
        """
        different training stages require different frame numbers
        """
        assert 1 <= num_frames <= 7
        logger.info(f"Setting number of frames per batch element from {self.num_frames} to {num_frames}.")
        self.num_frames = num_frames
        self.num_frames_max = num_frames

    def set_random_frame_num(self):
        tmp = random.randint(0, self.num_frames_max//4)
        if tmp == 0:
            self.num_frames = 2
        elif tmp == 1:
            self.num_frames = 4

    def update_interval(self, max_interval, logger):
        """
        different training stages require different frame numbers
        """
        # assert 1 <= max_interval <= (12) // self.num_frames
        logger.info(f"Setting max interval between frames from {self.max_interval} to {max_interval}.")
        self.max_interval = max_interval
        self.rnd_interval = True if max_interval > 1 else False

    def use_random_interval(self):
        self.rnd_interval = True
        if self.num_frames == 4:
            self.use_idx_list = True

    def set_current_interval(self):
        if self.use_idx_list:
            return self.get_frame_ids()
        self.current_interval = random.randint(1, self.max_interval)
        return self.current_interval

    def get_frame_ids(self):
        choice = random.random()
        if choice < 0.2:
            frames = [0, 2, 4, 6]
            interval = 2
        elif choice < 0.4:
            frames = [0, 1, 3, 5]
            interval = 2
        else:
            start_idx = random.randint(0, 3)  # LP frame idx (first frame)
            frames = [start_idx + i for i in range(4)]
            interval = 1
        self.frame_ids_curr = frames

        return interval

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            frames: List containing `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = Path(self.sample_folders[index])
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        if self.use_idx_list:
            frame_ids = self.frame_ids_curr
            frame_paths = [samples[frame_id] for frame_id in frame_ids]
        else:
            # max_interval = (len(samples) + 2) // self.num_frames
            interval = self.current_interval if self.rnd_interval else self.max_interval
            if self.current_interval == 4 and self.num_frames >= 3:
                frame_paths = [samples[0], samples[4], samples[6]]
            else:
                frame_paths = (samples[::interval])[:self.num_frames]

        frames = np.concatenate(
            [self._load_img(p) for p in frame_paths], axis=-1
        )

        frames = self.transform(frames)
        frames = torch.chunk(frames, chunks=self.num_frames, dim=0)

        if self.lossless:
            frames = [rgb2yuv_lossless(img) for img in frames]
        else:
            frames = [rgb2ycbcr(img) for img in frames]

        if not self.use_idx_list and self.rnd_temp_order and isinstance(frames, list):
            if random.random() < 0.5:
                return frames[::-1]

        return frames

    def _load_img(self, p):
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        return img


