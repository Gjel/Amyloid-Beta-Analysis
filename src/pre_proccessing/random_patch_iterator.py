import numpy as np

from .label_enum import LabelEnum
from .util import *
from .get_patch import get_random_patch

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
        for i in range(batch_size):
            batch[i] = self.__next__()
        return batch