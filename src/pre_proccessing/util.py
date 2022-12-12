import math

SIZE = 256
STRIDE = 128
DATA_ROOT = "F:/non-vsi/"

def calc_num_patches(width: int, height: int, size: int = SIZE, stride: int = STRIDE):
    """
    Calculates the amount of patches that can be taken from an image width the provided width and height, based on the size and stride.
    :param width: Width of the image in pixels
    :param height: height of the image in pixels
    :param size: Width and height of the patch
    :param stride: The stride of the sliding window
    :return: number of patches in the width and height respectively
    """
    minus_term = int(size / stride) - 1
    w_patches = math.floor(width / stride) - minus_term
    h_patches = math.floor(height / stride) - minus_term
    return w_patches, h_patches









