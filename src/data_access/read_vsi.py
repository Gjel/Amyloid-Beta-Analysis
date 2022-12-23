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
    patch = patch.reshape((size, size, 3))
    return patch

def get_downsampled(
        reader: bioformats.ImageReader,
):
    reader.rdr.setSeries(DOWNSAMPLE_INDEX)
    return reader.read()

class VsiIterator:
    def __init__(self, image_path: str, size: int, stride: int):
        self.path = image_path
        self.size = size
        self.stride = stride

        self.reader = None
        self.it = None

    def __enter__(self):
        self.reader = bioformats.ImageReader(self.path)
        self.reader.rdr.setSeries(FULL_INDEX)
        width = self.reader.rdr.getSizeX()
        height = self.reader.rdr.getSizeY()
        self.it = PatchCoordinateIterator(width, height, self.size, self.stride)
        return self

    def open(self):
        self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()

    def close(self):
        self.__exit__(None, None, None)

    def __iter__(self):
        self.it = self.it.__iter__()
        return self

    def __next__(self):
        top, left = self.it.__next__()
        return get_patch(self.reader, left, top, self.size)

    def get_batch(self, batch_size):
        batch = np.zeros((batch_size, self.size, self.size, 3), dtype=np.uint8)
        for i in range(batch_size):
            try:
                batch[i] = self.__next__()
            except StopIteration:
                return batch[:i]
        return batch







