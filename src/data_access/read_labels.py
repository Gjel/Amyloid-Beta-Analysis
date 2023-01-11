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
    def __init__(self, data: h5py.Dataset, size: int, stride: int, batch_size: int, label_type: LabelEnum):
        self.data = data
        self.size = size
        self.stride = stride
        self.batch_size = batch_size
        self.label_type = label_type

        height, width = self.data.shape
        self.patch_it = PatchCoordinateIterator(width, height, self.size, self.stride)

    def __iter__(self):
        self.patch_it = self.patch_it.__iter__()
        return self

    def next_label(self):
        top, left = self.patch_it.__next__()
        patch = self.data[top:top + self.size, left:left + self.size]
        if self.label_type == LabelEnum.PIXEL:
            return patch
        elif self.label_type == LabelEnum.CLASS_MIDDLE:
            i = int(self.size / 2)
            return patch[i, i]
        elif self.label_type == LabelEnum.CLASS_AVG:
            return patch.mean()

    def __next__(self):
        if self.label_type == LabelEnum.PIXEL:
            batch = np.zeros((self.batch_size, self.size, self.size), dtype=np.float32)
        elif self.label_type == LabelEnum.CLASS_MIDDLE or self.label_type == LabelEnum.CLASS_AVG:
            batch = np.zeros((self.batch_size,), dtype=np.float32)
        for i in range(self.batch_size):
            try:
                batch[i] = self.next_label()
            except StopIteration:
                return batch[:i]
        return batch

    def set_current_patch(self, row: int, column: int):
        self.patch_it.row = row
        self.patch_it.column = column

    def get_current_patch_coords(self):
        return self.patch_it.row, self.patch_it.column









