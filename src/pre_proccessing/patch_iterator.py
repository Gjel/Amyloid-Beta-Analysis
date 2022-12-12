from typing import Union

import pyvips as pv
import numpy as np

from .util import *
from .get_patch import get_patch_from_image, get_random_patch
from .label_enum import LabelEnum


class PatchIterator:
    """
    An iterator that iterates over all patches in an image. It iterates from left to right first and then from top to
    bottom.
    """
    def __init__(self, image_path: str, size: int = SIZE, stride: int = STRIDE, label_type: LabelEnum = LabelEnum.NO):
        """
        Constructor for the PatchIterator
        :param image_path: the path to the image you want to iterate over
        :param size: The size of the patches you want to extract (default 256). Patches are always square.
        :param stride: The stride of the sliding window (default 128)
        """
        self._image = pv.Image.new_from_file(image_path)
        self.row = 0
        self.column = 0
        self.patch_size = size
        self.stride = stride
        self.max_column, self.max_row = calc_num_patches(self._image.width, self._image.height, size, stride)
        self.label_type = label_type

    def __iter__(self):
        self.row = 0
        self.column = 0
        return self

    def __next__(self) -> tuple[np.ndarray, Union[None, bool, np.ndarray]]:
        if self.row == self.max_row:
            raise StopIteration
        patch, label = get_patch_from_image(
            self._image,
            self.column * self.stride,
            self.row * self.stride,
            self.patch_size,
            self.label_type,
        )
        self.column += 1
        if self.column == self.max_column:
            self.row += 1
            self.column = 0
        return patch, label

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, Union[None, bool, np.ndarray]]:
        """
        A method that returns a batch of patches from the image.
        :param batch_size: The size of the batch
        :return: a ndarray of shape (batch_size, patch_size, patch_size, 3) and optional labels.
        """
        batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3), dtype=np.uint8) # TODO: remove magic number
        label_batch = None
        if self.label_type == LabelEnum.CLASS:
            label_batch = np.zeros(batch_size, dtype=bool)
        elif self.label_type == LabelEnum.PIXEL:
            label_batch = np.zeros((batch_size, self.patch_size, self.patch_size), dtype=bool)
        for i in range(batch_size):
            batch[i], labels = self.__next__()
            if self.label_type != LabelEnum.NO:
                label_batch[i] = labels
        return batch, label_batch


class RandomPatchIterator:
    """
    An iterator that returns random patches from an image dataset.
    """
    def __init__(self, data_path: str = DATA_ROOT, size: int = SIZE, label_type: LabelEnum = LabelEnum.NO):
        """
        Constructor for the PatchIterator
        :param data_path: the path to the dataset you want to iterate over
        :param size: The size of the patches you want to extract (default 256). Patches are always square.
        :param stride: The stride of the sliding window (default 128)
        """
        self.data_path = data_path
        self.patch_size = size
        self.label_type = label_type

    def __iter__(self):
        return self

    def __next__(self):
        return get_random_patch(self.data_path, self.patch_size, self.label_type)

    def get_batch(self, batch_size: int) -> np.ndarray:
        """
        A method that returns a batch of patches from the dataset.
        :param batch_size: The size of the batch
        :return: a ndarray of shape (batch_size, patch_size, patch_size, 3).
        """
        batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3), dtype=np.uint8)  # TODO: remove magic number
        label_batch = None
        if self.label_type == LabelEnum.CLASS:
            label_batch = np.zeros(batch_size, dtype=bool)
        elif self.label_type == LabelEnum.PIXEL:
            label_batch = np.zeros((batch_size, self.patch_size, self.patch_size), dtype=bool)
        for i in range(batch_size):
            batch[i], labels = self.__next__()
            if self.label_type != LabelEnum.NO:
                label_batch[i] = labels
        return batch, label_batch
        # batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3), dtype=np.uint8)  # TODO: remove magic number
        # for i in range(batch_size):
        #     batch[i] = self.__next__()
        # return batch
