# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from pMCTF.utils.util import image_import


class YUVReader():
    def __init__(self, src_file, width, height, start_index=0):
        assert os.path.exists(src_file)
        self.src_file = src_file
        self.width = width
        self.height = height
        self.current_frame_index = start_index

        self.eof = False

    def read_one_frame(self, src_format="rgb"):
        def _none_exist_frame():
            if src_format == "rgb":
                return None
            return None, None, None
        if self.eof:
            return _none_exist_frame()

        Y, Cb, Cr = image_import(self.src_file, self.width, self.height, POC=self.current_frame_index,
                                 bitdepth=np.uint8, colorformat=420)

        height, width = Y.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return Y, Cb, Cr

    def close(self):
        self.current_frame_index = 0
