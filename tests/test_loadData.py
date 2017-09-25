#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division
import sys
import os
import unittest

import numpy as np

try:
  basestring
except NameError:
  basestring = str

sys.path.append('..')
import utility_functions as utils
from loadData import Data

class testData(unittest.TestCase):
    
    def setUp(self):
        self.data = Data("Data/gang_7x7_200")
    
    def test_path_expansion_morphology(self):
        self.assertEqual(1,'Data/gang_7x7_200/morphology' in self.data.sub_dir_path(self.data.path))
        
    def test_get_fname_string(self):
        self.assertTrue(isinstance(self.data.get_fname('Data/gang_7x7_200/LFP',['myLFP']),basestring))

    def test_get_fname_list(self):
        self.assertTrue(isinstance(self.data.get_fname('Data/gang_7x7_200/LFP',['myLFP','yourLFP']),list))

    def test_get_paths_LFP(self):
        self.assertTrue('Data/gang_7x7_200/LFP/myLFP',self.data.path_LFP)
        
    def test_get_paths_morpho(self):
        self.assertTrue('Data/gang_7x7_200/morphology/Badea2011Fig2Du.CNG.swc',self.data.path_morphology)
        
    def test_get_paths_ele_pos(self):
        self.assertTrue('Data/gang_7x7_200/LFP/electrode_positions/elcoord_x_y_z',self.data.path_ele_pos)
    def test_load_morpho_correct(self):
        self.assertTrue(isinstance(self.data.morphology,np.ndarray))
    def test_load_morpho_no_file(self):
        self.data.load_morpho('gugu')
        self.assertFalse(self.data.morphology)
    def test_load_morpho_no_array(self):
        self.data.load_morpho('test_loadData.txt')
        self.assertFalse(self.data.morphology)
    def test_reload_morpho(self):
        self.data.load_morpho()
        self.assertTrue(isinstance(self.data.morphology,np.ndarray))
        
        
if __name__ == '__main__':
  unittest.main()
