#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division, absolute_import
import os
import unittest
import numpy as np

from kcsd import oKCSD2D, oKCSD3D

try:
    basestring
except NameError:
    basestring = str


n_src = 5

class testsKCD(unittest.TestCase):
    def setUpClass(self):
        ele_pos = np.array([[-0.2, -0.2], [0, 0], [0, 1], [1, 0], [1, 1],
                            [0.5, 0.5], [1.2, 1.2]])
        pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
        own_src = np.array([[1,2,3,4,5,6,7,8,9,10], [0,0,1,1,2,2,1,1,1,1]])
        k = oKCSD2D(ele_pos, pots, own_est = own_src)
        k.cross_validate()
        ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                            (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),
                            (0.5, 0.5, 0.5)])
        pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])
        own_src = np.array([[1,2,3,4,5,6,7,8,9,10], [0,0,1,1,2,2,1,1,1,1], [1,1,1,1,1,5,3,4,2,5]])
        k = oKCSD3D(ele_pos, pots, own_est = own_src)
        k.cross_validate()

if __name__ == '__main__':
  unittest.main()
