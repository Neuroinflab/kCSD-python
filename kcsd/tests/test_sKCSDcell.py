#!/usr/bin/env python
# encoding: utf-8
import os
import unittest
import numpy as np

from kcsd import sKCSDcell, sample_data_path
from kcsd.sKCSD_utils import LoadData

try:
    basestring
except NameError:
    basestring = str


def in_between(a,b):
    return a+(b-a)/2

class testsKCDcell(unittest.TestCase):
  
    @classmethod
    def setUpClass(cls):
        sc = 1e6
        #Very branchy neuron (for testing morphology loop)
        cls.data = LoadData(os.path.join(sample_data_path, "gang_7x7_200"))
        cls.data.morphology[:,2:6] = cls.data.morphology[:,2:6]/sc
        n_src = 1000
        cls.cell = sKCSDcell(cls.data.morphology,cls.data.ele_pos/sc,n_src)
        cls.branch_points = []
        for loop in cls.cell.loops:
            if loop[0] != loop[1]+1 and loop[0] != loop[1]-1:
                cls.branch_points.append(loop.tolist())
        cls.morpho = cls.cell.morphology[:,2:5]
        cls.coor3D, cls.zero = cls.cell.point_coordinates(cls.morpho)

        #ball and stick neuron
        data = LoadData(os.path.join(sample_data_path, "ball_and_stick_8"))
        data.morphology[:,2:6] = data.morphology[:,2:6]/sc
        cls.cell_small = sKCSDcell(data.morphology,data.ele_pos/sc, 104)
        cls.cell_small_segment_coordinates_loops = cls.cell_small.coordinates_3D_loops()
        cls.small_points = np.zeros((len(cls.cell_small.morphology),))
        dic = cls.cell_small_segment_coordinates_loops
        for seg in dic:
            ps = dic[seg]
            for p in ps:
                cls.small_points[p[2]] += 1
        cls.cell_small_segment_coordinates = cls.cell_small.coordinates_3D_segments()

        #Y-shaped neuron
        data = LoadData(os.path.join(sample_data_path, "Y_shaped_neuron"))
        data.morphology[:,2:6] = data.morphology[:,2:6]/sc
        cls.cell_y = sKCSDcell(data.morphology,data.ele_pos/sc, 200)
        cls.cell_y_segment_coordinates_loops = cls.cell_y.coordinates_3D_loops()
        cls.cell_y_segment_coordinates = cls.cell_y.coordinates_3D_segments()

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
            elif self.cell.morphology[b[1],6] == b[0]+1:
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
        self.assertTrue(np.isclose(self.cell_small.max_dist,.00102))

    def test_calculate_total_distance_max_dist_small(self):
        #self.max_dist = self.est_pos.max()#add test
        self.assertTrue(np.isclose(self.cell_small.max_dist,self.cell_small.calculate_total_distance()))

    def test_calculate_total_distance_max_dist(self):
        #self.max_dist = self.est_pos.max()#add test
        self.assertTrue(np.isclose(self.cell.max_dist,self.cell.calculate_total_distance()))

                    
    def test_max_est_pos_total_distance_small(self):
        #self.max_dist = self.est_pos.max()#add test
        self.assertTrue(np.isclose(self.cell_small.max_dist,self.cell_small.est_pos.max()))

    def test_max_est_pos_total_distance(self):
        #self.max_dist = self.est_pos.max()#add test
        self.assertTrue(np.isclose(self.cell.max_dist,self.cell.est_pos.max()))

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

    def test_get_xyz_source_x_small(self):
        a =  in_between(self.cell_small.source_pos[1,0],
                        self.cell_small.source_pos[2,0])
        b =  in_between(self.cell_small.source_xyz[1,0],
                        self.cell_small.source_xyz[2,0])
        x,y,z = self.cell_small.get_xyz(a)
        self.assertTrue(x == b)

    def test_get_xyz_source_y_small(self):
        a =  in_between(self.cell_small.source_pos[1,0],
                        self.cell_small.source_pos[2,0])
        b =  in_between(self.cell_small.source_xyz[1,1],
                        self.cell_small.source_xyz[2,1])
        x,y,z = self.cell_small.get_xyz(a)
        self.assertTrue(y == b)

    def test_get_xyz_source_z_small(self):
        a =  in_between(self.cell_small.source_pos[1,0],
                        self.cell_small.source_pos[2,0])
        b =  in_between(self.cell_small.source_xyz[1,2],
                        self.cell_small.source_xyz[2,2])
        x,y,z = self.cell_small.get_xyz(a)
        self.assertTrue(np.isclose(z, b))

    def test_points_in_between(self):
        point0 = [0,0,0]
        point1 = [1,1,1]
        last = False
        out = self.cell.points_in_between(point0,point1,last)
        self.assertTrue(np.array_equal(out,np.array([point0])))

    def test_points_in_between_with_last(self):
        point0 = [0,0,0]
        point1 = [1,1,1]
        last = True
        expected = [point1, point0]
        out = self.cell.points_in_between(point0,point1,last)
        self.assertTrue(np.array_equal(out,np.array(expected)))
    
    def test_get_grid_small_z(self):
        self.assertTrue(self.cell_small.dims[2] == len(self.cell_small.morphology))

    def test_get_grid_small_y(self):
        self.assertTrue(self.cell_small.dims[1] == 1)
    
    def test_get_grid_small_x(self):
        self.assertTrue(self.cell_small.dims[0] == 1)

    def test_get_grid(self):
        self.assertFalse(self.cell.dims[-1] == len(self.cell.morphology))

    def test_get_grid_large_z(self):
        if self.cell.dxs[2]:
            self.assertTrue(self.cell.dims[2] == 1 + np.floor((self.cell.zmax-self.cell.zmin)/self.cell.dxs[2]))
        else:
            self.assertTrue(self.cell.dims[2] == 1)

    def test_get_grid_large_y(self):
        if self.cell.dxs[1]:
            self.assertTrue(self.cell.dims[1] == 1 + np.floor((self.cell.ymax-self.cell.ymin)/self.cell.dxs[1]))
        else:
            self.assertTrue(self.cell.dims[1] == 1)
    
    def test_get_grid_large_x(self):
        if self.cell.dxs[0]:
            self.assertTrue(self.cell.dims[0] == 1 + np.floor((self.cell.xmax-self.cell.xmin)/self.cell.dxs[0]))
        else:
            self.assertTrue(self.cell.dims[0] == 0)
    
    def test_dxs_small_x(self):
        self.assertTrue(self.cell_small.dxs[0]>self.cell_small.tolerance or self.cell_small.dxs[0] == 0)
    
    def test_dxs_small_y(self):
        self.assertTrue(self.cell_small.dxs[1]>self.cell_small.tolerance or self.cell_small.dxs[1] == 0)

    def test_dxs_small_z(self):
        self.assertTrue(self.cell_small.dxs[2]>self.cell_small.tolerance or self.cell_small.dxs[2] == 0)

    def test_point_coordinates_morpho_x(self):
        self.assertTrue(self.zero[0] == np.floor((self.morpho[0,0]-self.cell.xmin)/self.cell.dxs[0]))

    def test_point_coordinates_morpho_y(self):
        self.assertTrue(self.zero[1] == np.floor((self.morpho[0,1]-self.cell.ymin)/self.cell.dxs[1]))
        
    def test_point_coordinates_morpho_z(self):
        if self.cell.dxs[2]:
            self.assertTrue(self.zero[2] == np.floor((self.morpho[0,2]-self.cell.zmin)/self.cell.dxs[2]))
        else:
             self.assertTrue(self.zero[2] == 0)
            
    def test_point_coordinates_morpho_x_max(self):
        self.assertTrue(max(self.coor3D[:,0]) < self.cell.dims[0])

    def test_point_coordinates_morpho_y_max(self):
        self.assertTrue(max(self.coor3D[:,1]) < self.cell.dims[1])

    def test_point_coordinates_morpho_z_max(self):
        self.assertTrue(max(self.coor3D[:,2]) < self.cell.dims[2])
    
    def test_point_coordinates_morpho_x_min(self):
        self.assertTrue(min(self.coor3D[:,0]) == 0 or self.zero[0] == 0)
        
    def test_point_coordinates_morpho_y_min(self):
        self.assertTrue(min(self.coor3D[:,1]) == 0 or self.zero[1] == 0)
        
    def test_point_coordinates_morpho_z_min(self):
        self.assertTrue(min(self.coor3D[:,2])  == 0 or self.zero[2] == 0)

    def test_coordinates_3D_loops_is_every_point_except_ending_at_least_twice_small(self):
        idxs = np.where(self.small_points[:-1]<2)[0]
        self.assertFalse(idxs.size>0)

    def test_coordinates_3D_loops_last_point_once_small(self):
        self.assertTrue(self.small_points[-1] == 1)

    def test_coordinates_3D_loops_last_loop_two_counts(self):
        l = len(self.cell_small_segment_coordinates_loops)-1
        self.assertTrue(len(self.cell_small_segment_coordinates_loops[l]) == 2)

    def test_coordinates_3D_loops_last_loop_difference(self):
        l = len(self.cell_small_segment_coordinates_loops)-1
        p0 = self.cell_small_segment_coordinates_loops[l][0]
        p1 = self.cell_small_segment_coordinates_loops[l][1]
        self.assertTrue(p0[2] == p1[2] + 1 or p0[2] == p1[2] -1) 
    
    def test_coordinates_3D_loops_every_but_last_loop_one_point(self):
        l = len(self.cell_small_segment_coordinates_loops)
        for i in range(l-1):
            self.assertTrue(len(self.cell_small_segment_coordinates_loops[i]) == 1)

    def test_coordinates_3D_loops_small_change(self):
        l = len(self.cell_small_segment_coordinates)
        for i in range(1,l-1):
            a = self.cell_small_segment_coordinates[i][0][2]
            b = self.cell_small_segment_coordinates[i-1][0][2]
            self.assertTrue(a == b + 1 or a == b-1 )
  

    def test_coordinates_3D_small_length(self):
        l = len(self.cell_small_segment_coordinates)
        self.assertTrue(l == self.cell_small.morphology.shape[0]-1)

    def test_coordinates_3D_y_length(self):
        l = len(self.cell_y_segment_coordinates)
        self.assertTrue(l == self.cell_y.morphology.shape[0]-1)

    def test_coordinates_3D_small_last(self):
        l = len(self.cell_small_segment_coordinates)
        self.assertTrue(len(self.cell_small_segment_coordinates[l-1]) == 2)

    def test_coordinates_3D_y_last(self):
        l = len(self.cell_y_segment_coordinates)
        self.assertTrue(len(self.cell_y_segment_coordinates[l-1]) == 4)

    def test_coordinates_3D_small_change(self):
        l = len(self.cell_small_segment_coordinates)
        for i in range(1,l-1):
            self.assertTrue(self.cell_small_segment_coordinates[i][0][2] == self.cell_small_segment_coordinates[i-1][0][2]+1)
  
    def test_coordinates_3D_loops_y_point_difference_x(self):
        l = len(self.cell_y_segment_coordinates_loops)-1
        d = self.cell_y_segment_coordinates_loops
        for i in range(l):
            if len(d[i]) > 1:
                for j in range(1,len(d[i])):
                    x0, y0, z0 = d[i][j-1]
                    x1, y1, z1 = d[i][j]
                    self.assertTrue(x0 == x1 or x0 == x1+1 or x0 == x1-1)

    def test_coordinates_3D_loops_y_point_difference_y(self):
        l = len(self.cell_y_segment_coordinates_loops)-1
        d = self.cell_y_segment_coordinates_loops
        for i in range(l):
            if len(d[i]) > 1:
                for j in range(1,len(d[i])):
                    x0, y0, z0 = d[i][j-1]
                    x1, y1, z1 = d[i][j]
                    self.assertTrue(y0 == y1 or y0 == y1+1 or y0 == y1-1)
          
    def test_coordinates_3D_loops_y_point_difference_z(self):
        l = len(self.cell_y_segment_coordinates_loops)-1
        d = self.cell_y_segment_coordinates_loops
        for i in range(l):
            if len(d[i]) > 1:
                for j in range(1,len(d[i])):
                    x0, y0, z0 = d[i][j-1]
                    x1, y1, z1 = d[i][j]
                    self.assertTrue(z0 == z1 or z0 == z1+1 or z0 == z1-1)

    def test_continuity_3D_loops_y(self):
        l = len(self.cell_y_segment_coordinates_loops)-1
        d = self.cell_y_segment_coordinates_loops
        for i in range(1,l):
            d0 = d[i-1]
            p1 = d[i][-1]
            if len(d0) > 1:
                p0 = d0[0]
            else:
                p0 = d0[-1]
            x0, y0, z0 = p0
            x1, y1, z1 = p1
            self.assertTrue(x0 == x1 or x0 == x1-1 or x0 == x1+1)
            self.assertTrue(y0 == y1 or y0 == y1-1 or y0 == y1+1)
            self.assertTrue(z0 == z1 or z0 == z1-1 or z0 == z1+1)
        
    def test_coordinates_3D_loops_every_but_last_loop_one_point(self):
        l = len(self.cell_small_segment_coordinates_loops)
        for i in range(l-1):
            self.assertTrue(len(self.cell_small_segment_coordinates_loops[i]) == 1)

    def test_corrected_x_1(self):
        x = self.cell.max_dist/2
        new_x = self.cell.corrected_x(x)
        self.assertTrue(new_x == x)

    def test_corrected_x_2(self):
        x = self.cell.max_dist/2 + self.cell.max_dist
        new_x = self.cell.corrected_x(x)
        self.assertTrue(new_x == x-self.cell.max_dist)

    def test_corrected_x_3(self):
        x = self.cell.max_dist/2 - self.cell.max_dist
        new_x = self.cell.corrected_x(x)
        self.assertTrue(new_x == x + self.cell.max_dist)

    def test_get_src_ele_mesh_1(self):
        electrodes = np.arange(0, self.cell_small.ele_pos.shape[0], 1, dtype=int)
        out = self.cell_small.get_src_ele_mesh(electrodes)
        self.assertTrue(isinstance(out, list))
        
    def test_get_src_ele_mesh_2(self):
        n = self.cell_small.ele_pos.shape[0]
        electrodes = np.arange(0, n, 1, dtype=int)
        out = self.cell_small.get_src_ele_mesh(electrodes)
        for i in range(n):
            self.assertTrue(out[1][0, i] == electrodes[i])

    def test_get_src_ele_mesh_2(self):
        n = self.cell_small.ele_pos.shape[0]
        electrodes = np.arange(0, n, 1, dtype=int)
        out = self.cell_small.get_src_ele_mesh(electrodes)
        for j in range(len(self.cell_small.source_pos)):
            for i in range(n):
                self.assertTrue(out[1][j, i] == electrodes[i])

    def test_get_src_ele_mesh_3(self):
        n = self.cell_small.ele_pos.shape[0]
        electrodes = np.arange(0, n, 1, dtype=int)
        out = self.cell_small.get_src_ele_mesh(electrodes)
        for j in range(len(self.cell_small.source_pos)):
            for i in range(n):
                self.assertTrue(out[0][j, i] == self.cell_small.source_pos[j])

    def test_get_src_ele_dists_1(self):
        out = self.cell_small.get_src_ele_dists()
        for j in range(len(self.cell_small.source_pos)):
            for i in range(len(self.cell_small.ele_pos)):
                self.assertTrue(out[0][j, i] == self.cell_small.source_pos[j])

    def test_get_src_ele_dists_2(self):
        out = self.cell_small.get_src_ele_dists()
        self.assertTrue(out[1].shape[2] == 3)

    def test_get_src_ele_dists_3(self):
        out = self.cell_small.get_src_ele_dists()
        for j in range(len(self.cell_small.source_pos)):
            for i in range(len(self.cell_small.ele_pos)):
                for k in range(3):
                    self.assertTrue(out[1][j, i, k] == self.cell_small.ele_pos[i, k])

    def test_get_src_estm_dists_1(self):
        out = self.cell_small.get_src_estm_dists()
        self.assertTrue(out.shape[0] == len(self.cell_small.source_pos))

    def test_get_src_estm_dists_2(self):
        out = self.cell_small.get_src_estm_dists()
        self.assertTrue(out.shape[1] == len(self.cell_small.est_pos)-1)
    
    def test_get_src_estm_dists_3(self):
        out = self.cell_small.get_src_estm_dists()
        for i in range(len(self.cell_small.source_pos)):
            for j in range(len(self.cell_small.est_pos)-1):
                self.assertTrue(out[i, j ] == abs(self.cell_small.source_pos[i] - self.cell_small.est_pos[j+1]))

    def test_get_src_estm_dists_pot_1(self):
        out = self.cell_small.get_src_estm_dists_pot()
        self.assertTrue(isinstance(out, list))

if __name__ == '__main__':
    unittest.main()
