import unittest

from src.util.patch_coordinate_iterator import PatchCoordinateIterator


class TestPatchCoordinateIterator(unittest.TestCase):
    def test_simple(self):
        h, w, s = 3, 3, 1
        it = PatchCoordinateIterator(w, h, s, s)
        for n, m in it:
            with self.subTest(f"test n, m = {it.row}, {it.column}"):
                self.assertLessEqual(n + s, h)
                self.assertLessEqual(m + s, w)

    def test_spare_space(self):
        h, w, s = 11, 11, 3
        it = PatchCoordinateIterator(w, h, s, s)
        for n, m in it:
            with self.subTest(f"test n, m = {n}, {m}"):
                self.assertLessEqual(n + s, h)
                self.assertLessEqual(m + s, w)


if __name__ == '__main__':
    unittest.main()
