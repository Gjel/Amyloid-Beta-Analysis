import numpy as np
import h5py
from paquo.images import QuPathProjectImageEntry
from shapely.geometry import Polygon
from rasterio.features import rasterize
from rasterio.transform import Affine

from src.data_preparation.project import Project
from src.util import PatchCoordinateIterator, from_qupath

HDF5_PATH = 'F:/entry_point.hdf5'

PATCH_SIZE = 1024

def get_patch_pixel_labels(
        image_entry: QuPathProjectImageEntry,
        left: int,
        top: int,
        size: int,
) -> np.ndarray:
    """
    A function that creates a numpy array that maps the annotation of gray matter
    :param image_entry: The Qupath image entry that the patch is taken from
    :param left: Left edge of extract area
    :param top: Top edge of extract area
    :param size: Width and height of the extract area
    :return: a size x size numpy array where a cell is true if that pixel is annotated as grey matter and false
    otherwise.
    """
    square = Polygon([(left, top), (left+size, top), (left+size, top+size), (left, top+size), (left, top)])
    roi = image_entry.hierarchy.annotations[0].roi

    intersection = square.intersection(roi)
    labels = np.zeros((size, size), dtype=np.uint8)
    if not intersection.is_empty:
        transform = Affine(1, 0, left, 0, 1, top)
        rasterize([intersection], out=labels, transform=transform, default_value=1)
    return labels.astype(bool)


def write_ground_truth(image_entry: QuPathProjectImageEntry, target_group: h5py.Group):
    image_name = from_qupath(image_entry.image_name).base
    if image_name not in target_group:
        print(image_name, 'created')
        image_group = target_group.create_group(image_name)
    else:
        print(image_name, 'skipped')
        return
    data = image_group.create_dataset(
        'grey matter labels',
        (image_entry.height, image_entry.width),
        dtype=bool,
        compression='lzf',
        chunks=True,
    )
    s = PATCH_SIZE
    it = PatchCoordinateIterator(image_entry.width, image_entry.height, s, s)
    for n, m in it:
        ground_truth = get_patch_pixel_labels(image_entry, m, n, s)
        data[n:n+s, m:m+s] = ground_truth

def main(file: h5py.File):
    group = file['init']
    project = Project.get_project()
    for entry in project.images:
        write_ground_truth(entry, group)


data_file = h5py.File(HDF5_PATH, 'a')
main(data_file)
data_file.close()
