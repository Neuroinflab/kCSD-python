from __future__ import print_function, division, absolute_import
from kcsd.sKCSD_utils import check_estimated_shape
import os
import unittest
import numpy as np

class testCheckEstimatedShape(unittest.TestCase):
    def test_unchanged(self):
        array = np.ones((1, 5))
        out = check_estimated_shape(array)
        self.assertEqual(array.shape, out.shape)

    def test_changed(self):
        array = np.ones((5, ))
        out = check_estimated_shape(array)
        self.assertEqual(out.shape, (5, 1))


if __name__ == '__main__':
    unittest.main()
