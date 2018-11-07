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
    def setUpClass(cls, n_src=5):
        """
        Check, if it is possible to read in data. 
        This test will be expanded and more neurons read-in.
        """
        
        cls.data2 = LoadData(os.path.join(sample_data_path, "Simple_with_branches"))
        cls.data2.morphology[:, 2:6] = cls.data2.morphology[:, 2:6]/sc
        cls.data2.ele_pos = cls.data2.ele_pos/sc
        cls.data2.LFP = cls.data2.LFP[:,:10]/1e3
        cls.reco2 = sKCSD(cls.data2.ele_pos, cls.data2.LFP, cls.data2.morphology, n_src_init=n_src, dist_table_density=5)
        
    
   
if __name__ == '__main__':
  unittest.main()
