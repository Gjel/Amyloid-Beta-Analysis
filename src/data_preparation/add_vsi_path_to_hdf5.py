import os

import h5py

from src.util import Name

DATA_PATH = 'F:/segmentation-labels.hdf5'
TEST_PATH = 'F:/hdf5/test.hdf5'

def create_data():
    file = h5py.File(TEST_PATH, 'w')
    for i in range(5):
        file.create_group(str(i))
    #
    # data = 'this sentence'
    # file['string'] = data
    file.close()

def read_data():
    file = h5py.File(DATA_PATH, 'r')
    for group_name in file:
        group = file[group_name]
        try:
            print(group_name, '->', group['vsi_path'].asstr()[()])
        except KeyError:
            print(group_name, '->', 'no path')
    # item = file['string'].asstr()[()]
    # print(item)
    file.close()

def main():
    file = h5py.File(DATA_PATH, 'a')
    for group_name in file:
        group = file[group_name]
        path = os.path.join('Abeta images 100+', Name(group_name).to_vsi())
        group['vsi_path'] = path
    file.close()


read_data()
main()
print('')
read_data()
