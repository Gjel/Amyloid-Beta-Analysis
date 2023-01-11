import bioformats
import numpy as np

from src.util import PatchCoordinateIterator

FULL_INDEX = 13
DOWNSAMPLE_INDEX = 20

def get_patch(
        reader: bioformats.ImageReader,
        left: int,
        top: int,
        size: int,
):
    reader.rdr.setSeries(FULL_INDEX)
    patch = reader.rdr.openBytesXYWH(0, left, top, size, size)
    patch = np.transpose(patch.reshape((size, size, 3)), (2, 0, 1))
    return patch / 255.0

def get_downsampled(
        reader: bioformats.ImageReader,
):
    reader.rdr.setSeries(DOWNSAMPLE_INDEX)
    return reader.read()

class VsiIterator:
    def __init__(self, image_path: str, size: int, stride: int, batch_size: int):
        self.path = image_path
        self.size = size
        self.stride = stride
        self.batch_size = batch_size

        self.reader = None
        self.patch_it = None

    def __enter__(self):
        self.reader = bioformats.ImageReader(self.path)
        self.reader.rdr.setSeries(FULL_INDEX)
        width = self.reader.rdr.getSizeX()
        height = self.reader.rdr.getSizeY()
        self.patch_it = PatchCoordinateIterator(width, height, self.size, self.stride)
        return self

    def open(self):
        self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()

    def close(self):
        self.__exit__(None, None, None)

    def __iter__(self):
        self.patch_it = self.patch_it.__iter__()
        return self

    def next_patch(self):
        top, left = self.patch_it.__next__()
        return get_patch(self.reader, left, top, self.size)

    def __next__(self):
        batch = np.zeros((self.batch_size, 3, self.size, self.size), np.float32)
        for i in range(self.batch_size):
            try:
                batch[i] = self.next_patch()
            except StopIteration:
                return batch[:i]
        return batch

    def set_current_patch(self, row: int, column: int):
        self.patch_it.row = row
        self.patch_it.column = column

    def get_current_patch_coords(self):
        return self.patch_it.row, self.patch_it.column








