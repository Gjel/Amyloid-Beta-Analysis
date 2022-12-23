import random

import h5py

TEST_PATH = 'F:/hdf5/test.hdf5'
DATA_PATH = 'F:/segmentation-labels.hdf5'

TRAIN = 0.7
VAL = 0.1
TEST = 0.2

def create_data():
    with h5py.File(TEST_PATH, 'w') as file:
        for i in range(10):
            file.create_group(str(i))


def shuffle_data(file: h5py.File):
    key_list = [key for key in file['init'].keys()]
    random.shuffle(key_list)
    s = len(key_list)
    for i, key in enumerate(key_list):
        if i < TRAIN*s:
            file.move(f'/init/{key}', f'/train/{key}')
        elif i < (TRAIN + VAL) * s:
            file.move(f'/init/{key}', f'/validation/{key}')
        else:
            file.move(f'/init/{key}', f'/test/{key}')


def reset_data(file: h5py.File):
    for subset in ['train', 'validation', 'test']:
        for image_group_name in file[subset]:
            file.move(f'/{subset}/{image_group_name}', f'/init/{image_group_name}')



