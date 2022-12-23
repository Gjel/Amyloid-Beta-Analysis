import h5py
import numpy as np

from src.util import PatchCoordinateIterator, LabelEnum

def get_patch(
        file: h5py.File,
        name: str,
        left: int,
        top: int,
        size: int,
):
    data = file[name]['labels']
    return data[top:top+size, left:left+size]

def get_patch_class(
        file: h5py.File,
        name: str,
        left: int,
        top: int,
        size: int,
        avg: bool = False,
):
    patch = get_patch(file, name, left, top, size)
    if avg:
        return patch.mean() >= 0.5
    else:
        i = int(size/2)
        return patch[i, i]

def get_vsi_path(
        file: h5py.File,
        name: str,
):
    return file[name]['vsi_path'][()]


class LabelIterator:
    def __init__(self, data: h5py.Dataset, size: int, stride: int, label_type: LabelEnum):
        self.data = data
        self.size = size
        self.stride = stride
        self.label_type = label_type

        height, width = self.data.shape
        self.it = PatchCoordinateIterator(width, height, self.size, self.stride)

    def __iter__(self):
        self.it = self.it.__iter__()
        return self

    def __next__(self):
        top, left = self.it.__next__()
        patch = self.data[top:top + self.size, left:left + self.size]
        if self.label_type == LabelEnum.PIXEL:
            return patch
        elif self.label_type == LabelEnum.CLASS:
            i = int(self.size / 2)
            return patch[i, i]

    def get_batch(self, batch_size):
        if self.label_type == LabelEnum.PIXEL:
            batch = np.zeros((batch_size, self.size, self.size), dtype=bool)
        elif self.label_type == LabelEnum.CLASS:
            batch = np.zeros((batch_size,), dtype=bool)
        for i in range(batch_size):
            try:
                batch[i] = self.__next__()
            except StopIteration:
                return batch[:i]
        return batch









