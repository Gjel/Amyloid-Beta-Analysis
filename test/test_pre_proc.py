import unittest
import numpy as np
import pyvips as pv

from src import pre_proccessing as pp

FILE_PATH = "F:/non-vsi/Image_2013-095_F2_BA4.vsi - 20x_BF_01.ome.tif"

class MyTestCase(unittest.TestCase):
    def test_calc_num_patches(self):
        N, M = 1000, 1000
        w, h = pp.calc_num_patches(M, N)
        mat = np.zeros((N, M))
        for i in range(h):
            for j in range(w):
                minimat = mat[i * pp.STRIDE:i * pp.STRIDE + pp.SIZE, j * pp.STRIDE:j * pp.STRIDE + pp.SIZE]
                self.assertEqual(minimat.shape, (256, 256), f"patch ({j}, {i}) is misshapen")

    def test_get_patch(self):
        patch, label = pp.get_patch_from_path(FILE_PATH, 10000, 10000)
        test = pv.Image.new_from_file("test/static/test_get_patch.png").numpy()
        self.assertTrue(np.array_equal(test, patch))

    def test_patch_iterator(self):
        it = pp.PatchIterator(FILE_PATH)
        patch_batch, label_batch = it.get_batch(128)
        self.assertEqual((128, 256, 256, 3), patch_batch.shape)

    def test_get_random_patch(self):
        patch_1, label_1 = pp.get_random_patch()
        self.assertEqual((256, 256, 3), patch_1.shape)
        patch_2, label_2 = pp.get_random_patch()
        self.assertEqual((256, 256, 3), patch_2.shape)
        self.assertFalse(np.array_equal(patch_1, patch_2))

    def test_random_patch_iterator(self):
        it = pp.RandomPatchIterator()
        patch_batch, label_batch = it.get_batch(128)
        self.assertEqual((128, 256, 256, 3), patch_batch.shape)

    def test_get_patch_class(self):
        image = pv.Image.new_from_file(FILE_PATH)
        is_grey = pp.get_patch_class(image, 10000, 10000)
        self.assertFalse(is_grey)
        is_grey = pp.get_patch_class(image, 100000, 10000)
        self.assertTrue(is_grey)

    def test_get_patch_pixel_label(self):
        image = pv.Image.new_from_file(FILE_PATH)
        label = pp.get_patch_pixel_labels(image, 10000, 10000)
        self.assertFalse(label[0, 0])
        label = pp.get_patch_pixel_labels(image, 100000, 10000)
        self.assertTrue(label[0, 0])

    def test_class_batch(self):
        it = pp.PatchIterator(FILE_PATH, label_type=pp.LabelEnum.CLASS)
        patch_batch, label_batch = it.get_batch(128)
        self.assertEqual((128, 256, 256, 3), patch_batch.shape)
        self.assertEqual((128,), label_batch.shape)

    def test_pixel_batch(self):
        it = pp.PatchIterator(FILE_PATH, label_type=pp.LabelEnum.PIXEL)
        it.row = 4
        it.column = 100
        patch_batch, label_batch = it.get_batch(128)
        self.assertEqual((128, 256, 256, 3), patch_batch.shape)
        self.assertEqual((128, 256, 256), label_batch.shape)

    def test_get_project(self):
        project = pp.Project.get_project()
        self.assertEqual(len(project.images), 463)

    def test_get_image_name_from_path(self):
        name = src.data_preparation.project.get_image_name_from_path(FILE_PATH)
        self.assertEqual('Image_2013-095_F2_BA4.vsi - 20x_BF_01', name)

    def test_get_image_from_project(self):
        name = src.data_preparation.project.get_image_name_from_path(FILE_PATH)
        image = pp.Project.get_image(name)
        self.assertIsNotNone(image)


if __name__ == '__main__':
    unittest.main()
