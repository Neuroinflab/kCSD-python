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
  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import utility_functions as utils
from corelib.loadData import Data
from corelib.sKCSDcell import sKCSDcell

class testsKCDcell(unittest.TestCase):
    def setUp(self,n_src=1000):
        self.data = Data("Data/gang_7x7_200")
        self.cell = sKCSDcell(self.data.morphology,self.data.ele_pos,n_src)
        self.cell.distribute_srcs_3D_morph()
        self.branch_points = []
        for loop in self.cell.loops:
            if loop[0] != loop[1]+1 and loop[0] != loop[1]-1:
                self.branch_points.append(loop.tolist())

    def test_all_sources_distributed(self):
        self.assertEqual(self.cell.src_distributed,self.cell.n_src)
    def test_all_sources_distributed_few_sources(self):
        self.cell2 = sKCSDcell(self.data.morphology,self.data.ele_pos,10)
        self.cell2.distribute_srcs_3D_morph()
        self.assertEqual(self.cell2.src_distributed,self.cell2.n_src)
    def test_if_lost_branch(self):
        segments = self.cell.morphology[1:,:].shape[0]
        self.assertTrue(segments,np.unique(self.cell.loops[:,0]).shape[0])
        
    def test_correct_branches(self):
        branches = np.array(self.branch_points)
        bad_points = 0
        for b in branches:
            if self.cell.morphology[b[0],6] == b[0]:
                pass
            elif self.cell.morphology[b[0],6] == b[1]+1:
                pass
            elif self.cell.morphology[b[0],6] == -1:
                pass
            else:
                bad_points += 1
            
    
        self.assertTrue(bad_points==0)

    def test_all_connections_forward_and_backwards(self):
        branches = self.branch_points[:]
        for b in self.branch_points:
            if [b[1],b[0]] in self.branch_points:
                if b in branches:
                    branches.remove(b)
                if [b[1],b[0]] in branches:
                    branches.remove([b[1],b[0]])
        self.assertFalse(len(branches))

    def test_all_branches_found(self):
        branches = []
        for line in self.cell.morphology[1:]:
            if line[0] != line[6]+1:
                branches.append([line[0]-1,line[6]-1])
        branch_count = len(branches)
        for b in branches:
            if b in self.branch_points:
                branch_count -=1
        self.assertTrue(branch_count==0)
            
            
if __name__ == '__main__':
  unittest.main()
