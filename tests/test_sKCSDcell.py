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
  
sys.path.insert(0,
os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import utility_functions as utils
from corelib.loadData import Data
from corelib.sKCSDcell import sKCSDcell


def in_between(a,b):
  return a+(b-a)/2

class testsKCDcell(unittest.TestCase):
  
  @classmethod
  def setUpClass(cls):
    sc = 1e6
    cls.data = Data("Data/gang_7x7_200")
    cls.data.morphology[:,2:6] = cls.data.morphology[:,2:6]/sc
    n_src = 1000
    cls.cell = sKCSDcell(cls.data.morphology,cls.data.ele_pos/sc,n_src)
    cls.cell.distribute_srcs_3D_morph()
    cls.branch_points = []
    for loop in cls.cell.loops:
      if loop[0] != loop[1]+1 and loop[0] != loop[1]-1:
        cls.branch_points.append(loop.tolist())

    data = Data("Data/ball_and_stick_8")
    data.morphology[:,2:6] = data.morphology[:,2:6]/sc
    cls.cell_small = sKCSDcell(data.morphology,data.ele_pos/sc,10)
    cls.cell_small.distribute_srcs_3D_morph()
  # Test if morphology loop is done correctly
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
            
  def test_calculate_total_distance(self):
    
    self.assertTrue(np.isclose(self.cell_small.calculate_total_distance(),2*505.97840005636186e-6))

  def test_get_xyz_x_small(self):
    a =  in_between(self.cell_small.est_pos[1,0],self.cell_small.est_pos[2,0])
    b =  in_between(self.cell_small.est_xyz[1,0],self.cell_small.est_xyz[2,0])
   
    x,y,z = self.cell_small.get_xyz(a)
    self.assertTrue(x == b)

  def test_get_xyz_y_small(self):
    a =  in_between(self.cell_small.est_pos[1,0],self.cell_small.est_pos[2,0])
    b =  in_between(self.cell_small.est_xyz[1,1],self.cell_small.est_xyz[2,1])
    x,y,z = self.cell_small.get_xyz(a)
    self.assertTrue(y == b)

  def test_get_xyz_z_small(self):
    a =  in_between(self.cell_small.est_pos[1,0],self.cell_small.est_pos[2,0])
    b =  in_between(self.cell_small.est_xyz[1,2],self.cell_small.est_xyz[2,2])
    x,y,z = self.cell_small.get_xyz(a)
    self.assertTrue(np.isclose(z,b))

  def test_get_xyz_x(self):
    a =  in_between(self.cell.est_pos[1,0],self.cell.est_pos[2,0])
    b =  in_between(self.cell.est_xyz[1,0],self.cell.est_xyz[2,0])
   
    x,y,z = self.cell.get_xyz(a)
    self.assertTrue(np.isclose(x,b))

  def test_get_xyz_y(self):
    a =  in_between(self.cell.est_pos[1,0],self.cell.est_pos[2,0])
    b =  in_between(self.cell.est_xyz[1,1],self.cell.est_xyz[2,1])
    x,y,z = self.cell.get_xyz(a)
    self.assertTrue(np.isclose(y,b))

  def test_get_xyz_z(self):
    a =  in_between(self.cell.est_pos[1,0],self.cell.est_pos[2,0])
    b =  in_between(self.cell.est_xyz[1,2],self.cell.est_xyz[2,2])
    x,y,z = self.cell.get_xyz(a)
    self.assertTrue(np.isclose(z,b))
  
    
  def test_segments_small(self):
    self.assertTrue(self.cell_small.segment_counter == len(self.cell_small.morphology)-1)

  def test_segments(self):  
    self.assertTrue(self.cell.segment_counter == len(self.cell.morphology)-1)

    
if __name__ == '__main__':
  unittest.main()
