#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division, absolute_import
import os
import unittest

from kcsd import sKCSD, sample_data_path
from kcsd.utility_functions import LoadData

try:
    basestring
except NameError:
    basestring = str


n_src = 5

class testsKCD(unittest.TestCase):
    @classmethod
    def setUpClass(cls, n_src=1000):
        sc = 1e6
        cls.data = LoadData(os.path.join(sample_data_path, "ball_and_stick_8"))
        cls.data.morphology[:,2:6] = cls.data.morphology[:,2:6]/sc
        cls.reco = sKCSD(cls.data.ele_pos/sc,cls.data.LFP/1e3,cls.data.morphology,n_src_init=n_src)
        cls.segments = cls.reco.values(transformation='segments')
        cls.loops = cls.reco.values(transformation=None)
        cls.cartesian = cls.reco.values(transformation='3D')
        cls.data2 = LoadData(os.path.join(sample_data_path, "Simple_with_branches"))
        cls.data2.morphology[:, 2:6] = cls.data2.morphology[:, 2:6]/sc
        cls.data2.ele_pos = cls.data2.ele_pos/sc
        cls.data2.LFP = cls.data2.LFP/1e3
        cls.reco2 = sKCSD(cls.data2.ele_pos, cls.data2.LFP, cls.data2.morphology, n_src_init=n_src)
        
    def test_values_segments_shape(self): 
        self.assertTrue(len(self.segments.shape) == 2)
    
    def test_values_segments_length(self):
        self.assertTrue(self.segments.shape[0] == len(self.data.morphology)-1)
        
    def test_values_loops_shape(self): 
        self.assertTrue(len(self.loops.shape) == 2)
        
    def test_values_loops_length(self):
        self.assertTrue(self.loops.shape[0] == 2*(len(self.data.morphology)-1)+1) #plus 1, because we include 0

    def test_values_cartesian_shape(self): 
        self.assertTrue(len(self.cartesian.shape) == 4)

    def test_get_src_ele_dists_1(self):
        self.assertTrue(len(self.reco2.src.ele_help) == 2)

   
if __name__ == '__main__':
  unittest.main()
