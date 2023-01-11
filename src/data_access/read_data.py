import os
import h5py

from src.data_access import VsiIterator, LabelIterator
from src.util import LabelEnum


def rec_read(group, lvl=0):
    if not isinstance(group, h5py.Group):
        return
    print(lvl * '\t', os.path.basename(group.name))
    for sub_name in group:
        rec_read(group[sub_name], lvl + 1)


class ImageIterator:
    def __init__(self, image_group: h5py.Group, size: int, stride: int, batch_size: int, label_type: LabelEnum):
        self.image_group = image_group
        self.size = size
        self.stride = stride
        self.batch_size = batch_size
        self.label_type = label_type

        self.label_it = LabelIterator(image_group['labels'], size, stride, batch_size, label_type)
        self.vsi_it = VsiIterator(self.get_vsi_path(), size, stride, batch_size)

    def get_vsi_path(self):
        root = os.path.dirname(self.image_group.file.filename)
        return os.path.join(root, self.image_group['vsi_path'].asstr()[()])

    def __enter__(self):
        self.vsi_it.open()
        return self

    def open(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vsi_it.close()

    def close(self):
        self.__exit__(None, None, None)

    def __iter__(self):
        self.vsi_it = self.vsi_it.__iter__()
        self.label_it = self.label_it.__iter__()
        return self

    def __next__(self):
        patch = self.vsi_it.__next__()
        label = self.label_it.__next__()
        return patch, label

    def set_current_patch(self, row: int, column: int):
        self.vsi_it.set_current_patch(row, column)
        self.label_it.set_current_patch(row, column)

    def get_current_patch_coords(self):
        vsi_coords = self.vsi_it.get_current_patch_coords()
        label_coords = self.label_it.get_current_patch_coords()
        if vsi_coords != label_coords:
            raise ValueError(f"Patch iterators are out of sync: vsi {vsi_coords}, label {label_coords}")
        return vsi_coords


class DataIterator:
    def __init__(self, data_group: h5py.Group, size: int, stride: int, batch_size: int, label_type: LabelEnum):
        self.data_group = data_group
        self.size = size
        self.stride = stride
        self.batch_size = batch_size
        self.label_type = label_type

        self.image_list = [key for key in data_group.keys()]
        self.num_images = len(self.image_list)
        self.current_image_index = 0
        self.image_it = self.init_image_it()

    def init_image_it(self):
        return ImageIterator(
            self.data_group[self.image_list[self.current_image_index]],
            self.size,
            self.stride,
            self.batch_size,
            self.label_type,
        )

    def __enter__(self):
        self.image_it.__enter__()
        return self

    def open(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.image_it.close()

    def close(self):
        self.__exit__(None, None, None)

    def __iter__(self):
        self.image_it = self.image_it.__iter__()
        return self

    def __next__(self):
        try:
            return self.image_it.__next__()
        except StopIteration:
            self.next_image()
            return self.image_it.__next__()

    def next_image(self):
        self.image_it.close()
        self.current_image_index += 1
        if self.current_image_index == self.num_images:
            raise StopIteration
        self.image_it = self.init_image_it()















