import os

import h5py

from src.util import from_qupath, is_qupath

DATA_PATH = 'F:/segmentation-labels.hdf5'
TEST_PATH = 'F:/hdf5/test.hdf5'

def create_test_hdf5():
    with h5py.File(TEST_PATH, 'a') as file:
        for i in range(5):
            file.create_group(str(i))

def read_group_names():
    file = h5py.File(TEST_PATH, 'r')
    for group_name in file:
        print(group_name)
    file.close()

def main():
    file = h5py.File(DATA_PATH, 'a')
    for group_name in file:
        if is_qupath(group_name):
            new_name = from_qupath(group_name).base
            file.move(group_name, new_name)
            print(group_name, 'changed to', new_name)
        else:
            print(group_name, 'skipped')
            continue

    file.close()


# create_test_hdf5()
read_group_names()
print('')
main()
read_group_names()
