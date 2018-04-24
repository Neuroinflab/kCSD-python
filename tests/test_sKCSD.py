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
  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import utility_functions as utils
from tutorials.loadData import Data
from corelib.sKCSD import sKCSD

n_src = 1000

class testsKCD(unittest.TestCase):
  @classmethod
  def setUpClass(cls,n_src=1000):
    sc = 1e6
    sKCSD.skmonaco_available = False
    cls.data = Data("Data/ball_and_stick_8")
    cls.data.morphology[:,2:6] = cls.data.morphology[:,2:6]/sc
    cls.reco = sKCSD(cls.data.ele_pos/sc,cls.data.LFP,cls.data.morphology,n_src_init=n_src)
    cls.segments = cls.reco.values(transformation='segments')
    cls.loops = cls.reco.values(transformation=None)
    cls.cartesian = cls.reco.values(transformation='3D')

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
   
            
if __name__ == '__main__':
  unittest.main()
