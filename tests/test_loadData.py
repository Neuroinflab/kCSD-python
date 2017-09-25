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
    def test_load_ele_pos_correct(self):
        self.assertTrue(isinstance(self.data.ele_pos,np.ndarray))
    def test_load_LFP_correct(self):
        self.assertTrue(isinstance(self.data.LFP,np.ndarray))
        
    def test_load_morpho_no_file(self):
        self.data.load(path='gugu',what="morphology")
        self.assertFalse(self.data.morphology)
    def test_load_morpho_no_array(self):
        self.data.load(path='test_loadData.py',what="morphology")
        self.assertFalse(self.data.morphology)
    def test_reload_morpho(self):
        self.data.load(path='Data/gang_7x7_200/morphology/Badea2011Fig2Du.CNG.swc',what='morphology')
        self.assertTrue(isinstance(self.data.morphology,np.ndarray))
         
    def test_load_ele_pos_no_file(self):
        self.data.load(path='gugu',what="electrode_positions")
        self.assertFalse(self.data.ele_pos)
    def test_load_ele_pos_no_array(self):
        self.data.load(path='test_loadData.py',what="electrode_positions")
        self.assertFalse(self.data.ele_pos)
    def test_reload_ele_pos(self):
        self.data.load(path='Data/gang_7x7_200/electrode_positions/elcoord_x_y_z',what="electrode_positions")
        self.assertTrue(isinstance(self.data.ele_pos,np.ndarray))
        
    def test_load_LFP_no_file(self):
        self.data.load(path='gugu',what="LFP")
        self.assertFalse(self.data.LFP)
    def test_load_LFP_no_array(self):
        self.data.load(path='test_loadData.py',what="LFP")
        self.assertFalse(self.data.LFP)
    def test_reload_LFP(self):
        self.data.load(path='Data/gang_7x7_200/LFP/myLFP',what="LFP")
        self.assertTrue(isinstance(self.data.LFP,np.ndarray)) 
        
if __name__ == '__main__':
  unittest.main()
