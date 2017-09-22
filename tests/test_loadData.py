#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division
import sys
import os
import unittest
import utility_functions as utils

sys.path.append('..')

from loadData import Data
class testData(unittest.TestCase):
    
    def setUp(self):
        self.data = Data("Data/gang_7x7_200")
    
    def test_path_expansion_morphology(self):
        self.assertEqual(1,'Data/gang_7x7_200/morphology' in self.data.sub_dir_path(self.data.path))
    
if __name__ == '__main__':
  unittest.main()
