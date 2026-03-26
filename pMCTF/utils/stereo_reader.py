# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
StereoImageReader
-----------------
A drop-in replacement for ``YUVReader`` that reads InStereo2K stereo PNG
pairs (left + right) as a sequential stream of YCbCr 4:2:0 frames.

Frames are interleaved in the order:
    left_0, right_0, left_1, right_1, …

so a GOP-2 reader consumes one stereo pair per group-of-pictures.

The returned (Y, Cb, Cr) numpy arrays match the format produced by
``YUVReader.read_one_frame()``:
    Y  : float32  [H,   W  ]
    Cb : float32  [H/2, W/2]  (bilinear 4:2:0 downsampling)
    Cr : float32  [H/2, W/2]
"""

from pathlib import Path

import numpy as np
from PIL import Image


def _rgb_to_ycbcr420(img: np.ndarray):
    """Convert an [H, W, 3] float32 RGB array to Y, Cb420, Cr420 arrays.

    Uses the same BT.601-like coefficients as ``pMCTF.utils.util.rgb2ycbcr``
    so that the test-time colour space matches training.
    """
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 128.0
    Cr = (R - Y) * 0.713 + 128.0

    # 4:2:0 downsampling – average 2×2 blocks (box filter)
    # Handle odd dimensions gracefully by cropping to even size first.
    h, w = Y.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)

    Cb_420 = 0.25 * (
        Cb[:h_even:2, :w_even:2] +
        Cb[1:h_even:2, :w_even:2] +
        Cb[:h_even:2, 1:w_even:2] +
        Cb[1:h_even:2, 1:w_even:2]
    )
    Cr_420 = 0.25 * (
        Cr[:h_even:2, :w_even:2] +
        Cr[1:h_even:2, :w_even:2] +
        Cr[:h_even:2, 1:w_even:2] +
        Cr[1:h_even:2, 1:w_even:2]
    )
    Y_out = Y[:h_even, :w_even]

    return Y_out.astype(np.float32), Cb_420.astype(np.float32), Cr_420.astype(np.float32)


class StereoImageReader:
    """Read InStereo2K stereo PNG pairs as a sequential stream of YCbCr frames.

    Frames are interleaved as ``left_0, right_0, left_1, right_1, …`` so that
    a GOP-2 consumer (gop_size=2) processes exactly one stereo pair per GOP.

    The interface mirrors ``pMCTF.utils.yuv_reader.YUVReader``:
    ``read_one_frame()`` returns ``(Y, Cb, Cr)`` numpy float32 arrays.

    Args:
        src_dir  : base directory that contains ``left/`` and ``right/``
                   sub-directories.
        width    : ignored – actual width is read from the PNG header.
        height   : ignored – actual height is read from the PNG header.
        start_index: first *interleaved* frame index (default 0).
    """

    def __init__(self, src_dir, width=None, height=None, start_index=0):
        src_dir = Path(src_dir)
        left_dir  = src_dir / "left"
        right_dir = src_dir / "right"

        if not left_dir.is_dir() or not right_dir.is_dir():
            raise RuntimeError(
                f'StereoImageReader: could not find left/ and right/ '
                f'sub-directories under "{src_dir}".'
            )

        left_images  = sorted(left_dir.glob("*.png"))
        right_images = sorted(right_dir.glob("*.png"))

        if len(left_images) != len(right_images):
            raise RuntimeError(
                f"Mismatched stereo image counts: "
                f"left={len(left_images)}, right={len(right_images)}"
            )

        # Interleave: [left_0, right_0, left_1, right_1, …]
        self.frame_paths = []
        for l, r in zip(left_images, right_images):
            self.frame_paths.append(l)
            self.frame_paths.append(r)

        self.current_frame_index = start_index
        self.eof = False

    def read_one_frame(self, src_format="yuv420"):
        """Return the next frame as (Y, Cb420, Cr420) float32 numpy arrays.

        Returns ``(None, None, None)`` after all frames have been consumed.
        """
        if self.eof or self.current_frame_index >= len(self.frame_paths):
            self.eof = True
            return None, None, None

        p = self.frame_paths[self.current_frame_index]
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

        Y, Cb, Cr = _rgb_to_ycbcr420(img)

        self.current_frame_index += 1
        return Y, Cb, Cr

    def close(self):
        self.current_frame_index = 0
