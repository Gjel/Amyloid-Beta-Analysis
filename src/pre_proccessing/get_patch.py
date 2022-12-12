import os
import random
from typing import Union

import pyvips as pv
import numpy as np
from shapely.geometry import Point, MultiPoint

from .util import *
from .label_enum import LabelEnum
from .project import Project

def get_patch_from_path(
        file_path: str,
        left: int,
        top: int,
        size: int = SIZE,
        label_type: LabelEnum = LabelEnum.NO,
) -> tuple[np.ndarray, Union[None, bool, np.ndarray]]:
    """
    A function that returns an image patch as a numpy array.
    :param file_path: The path to the images you want to take a patch from
    :param left: Left edge of extract area
    :param top: Top edge of extract area
    :param size: Width and height of the extract area
    :param label_type: The type of label that should be attached to the patch
    :return: The image patch as a numpy array with optional label
    """
    img = pv.Image.new_from_file(file_path, access='random')
    return get_patch_from_image(img, left, top, size, label_type)


def get_patch_from_image(
        image: pv.Image,
        left: int,
        top: int,
        size: int = SIZE,
        label_type: LabelEnum = LabelEnum.NO,
) -> tuple[np.ndarray, Union[None, bool, np.ndarray]]:
    """
    A function that returns an image patch as a numpy array.
    :param image: The image you want to take a patch from
    :param left: Left edge of extract area
    :param top: Top edge of extract area
    :param size: Width and height of the extract area
    :param label_type: The type of label that should be attached to the patch
    :return: The image patch as a numpy array with optional label
    """
    # get image
    crop = image.crop(left, top, size, size)
    patch = crop.numpy()

    # get label
    label = None
    if label_type == LabelEnum.CLASS:
        label = get_patch_class(image, left, top, size)
    elif label_type == LabelEnum.PIXEL:
        label = get_patch_pixel_labels(image, left, top, size)
    return patch, label


def get_random_patch(
        data_path: str = DATA_ROOT,
        size: int = SIZE,
        label_type: LabelEnum = LabelEnum.NO,
) -> tuple[np.ndarray, Union[None, bool, np.ndarray]]:
    """
    A function that returns a random patch from a random image in the dataset
    :param data_path:
    :param size: Width and height of the extract area
    :param label_type: The type of label that should be attached to the patch
    :return: a random patch from a random image in the dataset
    """
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    file = os.path.join(data_path, random.choice(files))
    image = pv.Image.new_from_file(file, access='random')
    left = np.random.randint(0, image.width - size)
    top = np.random.randint(0, image.height - size)
    patch, label_type = get_patch_from_image(image, left, top, size, label_type)
    return patch, label_type


def get_patch_class(
        image: pv.Image,
        left: int,
        top: int,
        size: int = SIZE,
) -> bool:
    """
    A function that return the class of the patch by checking whether the centre pixel is considered grey matter
    :param image: The pyvips image that the patch is taken from
    :param left: Left edge of extract area
    :param top: Top edge of extract area
    :param size: Width and height of the extract area
    :return: True if the centre pixel is annotated as gray matter, false otherwise.
    """
    image_entry = Project.get_image_from_path(image.filename)
    roi = image_entry.hierarchy.annotations[0].roi
    x = (2 * left + size) / 2
    y = (2 * top + size) / 2
    mid = Point(x, y)
    return not mid.intersection(roi).is_empty


def get_patch_pixel_labels(
        image: pv.Image,
        left: int,
        top: int,
        size: int = SIZE,
) -> np.ndarray:
    """
    A function that creates a numpy array that maps the annotation of gray matter
    :param image: The pyvips image that the patch is taken from
    :param left: Left edge of extract area
    :param top: Top edge of extract area
    :param size: Width and height of the extract area
    :return: a size x size numpy array where a cell is true if that pixel is annotated as grey matter and false
    otherwise.
    """
    x = np.arange(left, left + size)
    y = np.arange(top, top + size)
    points = MultiPoint(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]))

    image_entry = Project.get_image_from_path(image.filename)
    roi = image_entry.hierarchy.annotations[0].roi

    intersection = points.intersection(roi)
    labels = np.zeros((size, size), dtype=bool)
    if not intersection.is_empty:
        for point in intersection:
            labels[int(point.x) - left, int(point.y) - top] = True
    return labels



