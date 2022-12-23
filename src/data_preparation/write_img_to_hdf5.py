import javabridge
import bioformats
import h5py

from src.util.patch_coordinate_iterator import PatchCoordinateIterator

FULL_INDEX = 13
DOWNSAMPLE_INDEX = 20
DOWNSAMPLE_LEVEL = 128

PATCH_SIZE = 1024

def write_full_img(src: str, target: str):
    with bioformats.ImageReader(src) as reader:
        with h5py.File(target, 'w') as file:
            reader.rdr.setSeries(FULL_INDEX)
            img_width = reader.rdr.getSizeX()
            img_height = reader.rdr.getSizeY()
            full = file.create_dataset('full', shape=(img_height, img_width, 3), chunks=True, dtype=np.uint8, compression="gzip", compression_opts=6)
            s = PATCH_SIZE
            it = PatchCoordinateIterator(img_width, img_height, s, s)
            for n, m in it:
                print('row', it.row, 'of', it.num_height, ', col', it.column, 'of', it.num_width)
                patch = reader.rdr.openBytesXYWH(0, m, n, s, s)
                patch = patch.reshape((s, s, 3))
                full[n:n+s, m:m+s, :] = patch

def write_downsample_img(src: str, target: str):
    with bioformats.ImageReader(src) as reader:
        with h5py.File(target, 'w') as file:
            reader.rdr.setSeries(DOWNSAMPLE_INDEX)
            img_width = reader.rdr.getSizeX()
            img_height = reader.rdr.getSizeY()
            downsample = file.create_dataset('downsample', shape=(img_height, img_width, 3), chunks=True, dtype=np.uint8)
            downsample[:, :, :] = reader.read()

VSI_PATH = 'F:/Abeta images 100+/Image_2013-095_F2_BA4.vsi'
HDF5_PATH = 'F:/hdf5/test.hdf5'

javabridge.start_vm(class_path=bioformats.JARS)
logback = javabridge.JClassWrapper("loci.common.LogbackTools")
logback.enableLogging()
logback.setRootLevel("ERROR")

write_full_img(VSI_PATH, HDF5_PATH)
write_downsample_img(VSI_PATH, HDF5_PATH)

javabridge.kill_vm()