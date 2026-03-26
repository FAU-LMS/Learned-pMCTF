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
import warnings
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pMCTF.utils.util import rgb2ycbcr, rgb2yuv_lossless
import random
import glob


def _is_instereo2k_root(root, split="train"):
    """Return True if *root* follows the InStereo2K directory layout."""
    sub = "train" if split == "train" else "valid"
    p = Path(root) / sub
    return (p / "left").is_dir() and (p / "right").is_dir()


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


class InStereo2KVideoDataset(Dataset):
    """Load InStereo2K stereo pairs as 2-frame video sequences (GOP = 2).

    Each stereo pair (left image, right image) is treated as a 2-frame video
    clip, so the dataset is always used with ``num_frames=2`` and
    ``num_me_stages=1``.  The class exposes the same update/callback interface
    as ``VideoYCbCr`` so that ``train_pMCTF_L.py`` can call
    ``dataset.update_num_frames(…)`` etc. without crashing.

    Directory layout expected at *root*::

        root/
            train/
                left/   (e.g. Instereo2K_0.png, …)
                right/  (e.g. Instereo2K_0.png, …)
            valid/
                left/
                right/

    Args:
        root (str | Path): root dataset directory.
        split (str): ``"train"`` or ``"valid"``/``"test"``.
        transform (callable): torchvision transform applied to the stacked
            ``[H, W, 6]`` RGB array (ToTensor + RandomCrop).
        lossless (bool): use integer-to-integer YUV conversion (default: False).
        num_frames (int): ignored – always 2 for stereo pairs.
    """

    def __init__(self, root, split="train", transform=None, lossless=False,
                 num_frames=2, **kwargs):
        # --- compatibility attributes expected by train_pMCTF_L.py ---
        self.num_frames = 2
        self.num_frames_max = 2
        self.max_interval = 1
        self.rnd_interval = False
        self.use_idx_list = False
        self.rnd_temp_order = False
        self.current_interval = 1
        self.frame_ids_curr = None
        self.lossless = lossless
        self.transform = transform

        if split == "train":
            data_dir = Path(root) / "train"
        else:
            data_dir = Path(root) / "valid"

        left_dir = data_dir / "left"
        right_dir = data_dir / "right"

        if not left_dir.is_dir() or not right_dir.is_dir():
            raise RuntimeError(
                f'InStereo2K dataset not found at "{data_dir}". '
                "Expected sub-directories left/ and right/."
            )

        left_images = sorted(left_dir.glob("*.png"))
        right_images = sorted(right_dir.glob("*.png"))
        assert len(left_images) == len(right_images), (
            f"Mismatched image counts: left={len(left_images)}, right={len(right_images)}"
        )

        self.sample_pairs = list(zip(left_images, right_images))

        if split in ("valid", "test"):
            self.sample_pairs = self.sample_pairs[: min(len(self.sample_pairs), 10)]

    # ------------------------------------------------------------------
    # Compatibility stubs – VideoYCbCr interface used by train_pMCTF_L.py
    # ------------------------------------------------------------------

    def update_num_frames(self, num_frames, logger):
        """InStereo2K is always GOP-2; num_frames update is silently ignored."""
        logger.info(
            "InStereo2K dataset: ignoring update_num_frames – keeping num_frames=2 (stereo pair)."
        )

    def update_interval(self, max_interval, logger):
        """No frame-interval concept for stereo pairs."""
        logger.info(
            "InStereo2K dataset: frame interval not applicable (stereo pairs are always consecutive)."
        )

    def set_current_interval(self):
        return 1

    def use_random_interval(self):
        pass  # no-op

    def set_random_frame_num(self):
        pass  # no-op

    def get_frame_ids(self):
        return 1

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, index):
        # Be robust to occasional corrupted/truncated PNG files.
        # Retry with another stereo pair instead of crashing the DataLoader.
        max_retry = 8
        cur_index = index
        for retry in range(max_retry):
            left_path, right_path = self.sample_pairs[cur_index]
            try:
                left_img = self._load_img(left_path)    # [H, W, 3] float32 RGB
                right_img = self._load_img(right_path)  # [H, W, 3] float32 RGB

                # Stack along channel dim → [H, W, 6], then apply joint transform
                frames = np.concatenate([left_img, right_img], axis=-1)
                frames = self.transform(frames)                    # [6, H_crop, W_crop]
                frames = torch.chunk(frames, chunks=2, dim=0)     # tuple of [3, H_crop, W_crop]

                if self.lossless:
                    frames = [rgb2yuv_lossless(img) for img in frames]
                else:
                    frames = [rgb2ycbcr(img) for img in frames]

                return frames
            except (OSError, ValueError) as exc:
                if retry == 0:
                    warnings.warn(
                        f"InStereo2KVideoDataset: failed to read stereo pair "
                        f"('{left_path}', '{right_path}'): {exc}. "
                        "Skipping this sample and retrying with another one."
                    )
                cur_index = random.randint(0, len(self.sample_pairs) - 1)

        raise RuntimeError(
            f"InStereo2KVideoDataset: failed to read stereo pair after {max_retry} retries. "
            f"Last attempted files: '{left_path}', '{right_path}'"
        )

    def _load_img(self, p):
        with Image.open(p) as img_file:
            img = np.asarray(img_file.convert("RGB"), dtype=np.float32)
        return img
