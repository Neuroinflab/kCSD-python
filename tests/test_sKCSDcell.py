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
from sKCSDcell import sKCSDcell

class testsKCDcell(unittest.TestCase):
    def setUp(self,n_src=1000):
        self.data = Data("Data/Simple_with_branches")
        self.cell1 = sKCSDcell(self.data.morphology,self.data.ele_pos,n_src)
        self.cell2 = sKCSDcell(self.data.morphology,self.data.ele_pos,10)
    def test_all_sources_distributed(self):
        self.cell1.distribute_srcs_3D_morph()
        self.assertTrue(self.cell1.src_distributed,self.cell1.n_src)
    def test_all_sources_distributed_few_sources(self):
        self.cell2.distribute_srcs_3D_morph()
        self.assertTrue(self.cell2.src_distributed,self.cell2.n_src)
if __name__ == '__main__':
  unittest.main()
