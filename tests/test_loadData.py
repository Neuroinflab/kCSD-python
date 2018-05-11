#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division, absolute_import
import sys
import os
import unittest
import numpy as np
try:
    basestring
except NameError:
    basestring = str
from sKCSD_paper import loadData

class testData(unittest.TestCase):
    def setUp(self):
        self.data = loadData.Data("Data/gang_7x7_200")

    def test_path_expansion_morphology(self):
        path = self.data.path
        a = 'Data/gang_7x7_200/morphology' in self.data.sub_dir_path(path)
        self.assertTrue(a)

    def test_get_fname_string(self):
        path = self.data.get_fname('Data/gang_7x7_200/LFP', ['myLFP'])
        self.assertTrue(isinstance(path, basestring))

    def test_get_fname_list(self):
        path = self.data.get_fname('Data/gang_7x7_200/LFP',
                                   ['myLFP', 'yourLFP'])
        self.assertTrue(isinstance(path, list))

    def test_get_paths_LFP(self):
        self.assertEqual('Data/gang_7x7_200/LFP/myLFP',
                         self.data.path_LFP)

    def test_get_paths_morpho(self):
        path = 'Data/gang_7x7_200/morphology/Badea2011Fig2Du.CNG.swc'
        self.assertEqual(path, self.data.path_morphology)

    def test_get_paths_ele_pos(self):
        path = 'Data/gang_7x7_200/electrode_positions/elcoord_x_y_z'
        self.assertEqual(path, self.data.path_ele_pos)

    def test_load_morpho_correct(self):
        self.assertTrue(isinstance(self.data.morphology, np.ndarray))

    def test_load_ele_pos_correct(self):
        self.assertTrue(isinstance(self.data.ele_pos, np.ndarray))

    def test_load_LFP_correct(self):
        self.assertTrue(isinstance(self.data.LFP, np.ndarray))

    def test_load_morpho_no_file(self):
        self.data.load(path='gugu', what="morphology")
        self.assertFalse(self.data.morphology)

    def test_load_morpho_no_array(self):
        self.data.load(path='test_loadData.py', what="morphology")
        self.assertFalse(self.data.morphology)

    def test_reload_morpho(self):
        path = 'Data/gang_7x7_200/morphology/Badea2011Fig2Du.CNG.swc'
        self.data.load(path=path, what='morphology')
        self.assertTrue(isinstance(self.data.morphology, np.ndarray))

    def test_load_ele_pos_no_file(self):
        self.data.load(path='gugu', what="electrode_positions")
        self.assertFalse(self.data.ele_pos)

    def test_load_ele_pos_no_array(self):
        self.data.load(path='test_loadData.py',
                       what="electrode_positions")
        self.assertFalse(self.data.ele_pos)

    def test_reload_ele_pos(self):
        path = 'Data/gang_7x7_200/electrode_positions/elcoord_x_y_z'
        self.data.load(path=path, what="electrode_positions")
        self.assertTrue(isinstance(self.data.ele_pos, np.ndarray))

    def test_load_LFP_no_file(self):
        self.data.load(path='gugu', what="LFP")
        self.assertFalse(self.data.LFP)

    def test_load_LFP_no_array(self):
        self.data.load(path='test_loadData.py', what="LFP")
        self.assertFalse(self.data.LFP)

    def test_reload_LFP(self):
        self.data.load(path='Data/gang_7x7_200/LFP/myLFP', what="LFP")
        self.assertTrue(isinstance(self.data.LFP, np.ndarray))


if __name__ == '__main__':
    unittest.main()
