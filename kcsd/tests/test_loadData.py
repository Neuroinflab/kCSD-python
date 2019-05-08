#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division, absolute_import
from kcsd import sample_data_path
from kcsd.sKCSD_utils import LoadData
import os
import unittest
import numpy as np
try:
    basestring
except NameError:
    basestring = str

class testData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(sample_data_path, "gang_7x7_200")
        cls.data = LoadData(cls.path)
        
    def test_path_expansion_morphology(self):
        path = self.data.path
        m_path = ''
        for p in self.data.sub_dir_path(path):
            if 'morphology' in p:
                m_path = p
        a = 'data/gang_7x7_200/morphology' in m_path
        self.assertTrue(a)

    def test_get_fname_string(self):
        path = self.data.get_fname('data/gang_7x7_200/LFP', ['myLFP'])
        self.assertTrue(isinstance(path, basestring))

    def test_get_fname_list(self):
        path = self.data.get_fname('data/gang_7x7_200/LFP',
                                   ['myLFP', 'yourLFP'])
        self.assertTrue(isinstance(path, list))

    def test_get_paths_LFP(self):
        print(self.data.path_LFP)
        self.assertTrue('data/gang_7x7_200/LFP/myLFP' in self.data.path_LFP)

    def test_get_paths_morpho(self):
        path = os.path.join(self.path,'morphology/Badea2011Fig2Du.CNG.swc')
        self.assertTrue(path in self.data.path_morphology)

    def test_get_paths_ele_pos(self):
        path = os.path.join(self.path, 'electrode_positions/elcoord_x_y_z')
        self.assertTrue(path in self.data.path_ele_pos)

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
        path = os.path.join(self.path, 'morphology/Badea2011Fig2Du.CNG.swc')
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
        path = os.path.join(self.path, 'electrode_positions/elcoord_x_y_z')
        self.data.load(path=path, what="electrode_positions")
        self.assertTrue(isinstance(self.data.ele_pos, np.ndarray))

    def test_load_LFP_no_file(self):
        self.data.load(path='gugu', what="LFP")
        self.assertFalse(self.data.LFP)

    def test_load_LFP_no_array(self):
        self.data.load(path='test_loadData.py', what="LFP")
        self.assertFalse(self.data.LFP)

    def test_reload_LFP(self):
        lfp_path = os.path.join(self.path,'LFP/myLFP')
        self.data.load(path=lfp_path, what="LFP")
        self.assertTrue(isinstance(self.data.LFP, np.ndarray))


if __name__ == '__main__':
    unittest.main()
