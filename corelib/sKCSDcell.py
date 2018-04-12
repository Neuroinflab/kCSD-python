# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import numpy as np
from scipy import interpolate
from collections import Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import corelib.utility_functions as utils
import corelib.loadData as ld
from  corelib.bresenhamline import bresenhamline

#testing

class sKCSDcell(object):
    """
    KCSD3D - The 3D variant for the Kernel Current Source Density method.

    This estimates the Current Source Density, for a given configuration of 
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, morphology, ele_pos, n_src,tolerance=2e-6):
        """
        
        """
        self.morphology = morphology
        self.ele_pos = ele_pos
        self.n_src = n_src
        self.min_dist = 0 #counter
        self.max_dist = 0 #counter
        self.src_distributed = 0 #counter
        self.morph_points_dist = 0 #counter
        total_dist = self.calculate_total_distance()
        self.loop_pos = np.linspace(0, total_dist, n_src) #positions of sources on the morphology (1D), necessary for source division

        self.est_pos = [] #loop positions on the 1D morphology
        
        rep = Counter(self.morphology[:,6])
        self.branching = [int(key) for key in rep.keys() if rep[key]>1]

        self.loop_xyz = np.zeros(shape=(n_src+self.morphology.shape[0]*2,3))#for morphology plotting
        self.source_xyz = np.zeros(shape=(n_src,3))
        self.loops = [] #loop positions -- which 2 points are connected, mostly for debug
        self.xmin =  np.min(self.morphology[:,2])
        self.xmax = np.max(self.morphology[:,2])
        self.ymin =  np.min(self.morphology[:,3])
        self.ymax = np.max(self.morphology[:,3])
        self.zmin =  np.min(self.morphology[:,4])
        self.zmax = np.max(self.morphology[:,4])
        self.tolerance = tolerance
        self.segments = {}
        self.segment_counter = 0
        
    def distribute_srcs_3D_morph(self):
        self.loops = []
        for morph_pnt in range(1,self.morphology.shape[0]):
            
            if self.morphology[morph_pnt-1,0]==self.morphology[morph_pnt,6]:
                self.distribute_src_cylinder(morph_pnt, morph_pnt-1)
            elif self.morphology[morph_pnt,6] in self.branching:
                #go back up
                last_branch = int(self.morphology[morph_pnt,6])-1
                last_point = morph_pnt - 1
                while True:
                    parent = int(self.morphology[last_point,6]) - 1
                    self.distribute_src_cylinder(parent, last_point)
                    if parent == last_branch:
                        break
                    last_point = parent
                self.distribute_src_cylinder(morph_pnt,int(self.morphology[morph_pnt,6])-1)

        last_point = morph_pnt
        
        while True:
            parent = int(self.morphology[last_point,6]) - 1
            self.distribute_src_cylinder(parent, last_point)
            if int(self.morphology[parent,6]) == -1:
                break
            last_point = parent

        self.loops = np.array(self.loops)
        self.est_pos = np.zeros((len(self.loops)+1,1))
        self.est_xyz = np.zeros((len(self.loops)+1,3))
        self.est_xyz[0,:] = self.morphology[0,2:5]
        for i,loop in enumerate(self.loops):
            length = 0
            for j in [2,3,4]:
                length += (self.morphology[loop[1]][j]-self.morphology[loop[0]][j])**2
            self.est_pos[i+1] = self.est_pos[i] + length**0.5
            self.est_xyz[i+1,:] = self.morphology[loop[1],2:5]
        self.loop_pos = self.loop_pos.reshape(-1,1)
        self.max_dist = self.est_pos.max()
        return self.est_pos

    def add_segment(self, mp1, mp2):
        key1 = "%d_%d"%(mp1, mp2)
        key2 = "%d_%d"%(mp2, mp1)
        
        if key1 not in  self.segments:
            self.segments[key1] = self.segment_counter
            self.segments[key2] = self.segment_counter
            self.segment_counter += 1
            
    def distribute_src_cylinder(self,mp1, mp2):

        self.add_segment(mp1,mp2)
        xyz1 = self.morphology[mp1,2:5]
        xyz2 = self.morphology[mp2,2:5]
        self.loops.append([mp2,mp1])
        self.max_dist += np.linalg.norm(xyz1-xyz2)
        in_range = [idx for idx in range(self.src_distributed,self.n_src) 
                    if self.loop_pos[idx]<=self.max_dist or np.isclose(self.loop_pos[idx],self.max_dist)]
        
        self.src_distributed += len(in_range)
 
        if len(in_range)>0:
            for src_idx in in_range:
                self.source_xyz[src_idx,:] = xyz1-(xyz2-xyz1)*(self.loop_pos[src_idx] - self.max_dist)/(self.max_dist - self.min_dist)
                self.loop_xyz[src_idx+self.morph_points_dist,:] =  xyz1-(xyz2-xyz1)*(self.loop_pos[src_idx]  -self.max_dist)/(self.max_dist - self.min_dist)
        self.min_dist = self.max_dist
        #Add morphology point to the loop
        
        self.loop_xyz[self.morph_points_dist+self.src_distributed] = self.morphology[mp1,2:5]
        self.morph_points_dist +=1

    def get_xyz(self, x):
        return interpolate.interp1d(self.est_pos[:,0],self.est_xyz, kind='linear',axis=0)(x)
    
    def calculate_total_distance(self):
        total_dist = 0
        for i in range(1,self.morphology.shape[0]):
            xyz1 = self.morphology[i,2:5]
            xyz2 = self.morphology[int(self.morphology[i,6])-1,2:5]
            total_dist+=np.linalg.norm(xyz2-xyz1)
        total_dist*=2
        return total_dist
      
    def points_in_between(self,p1,p0,last):
    
        new_p1 = np.ndarray((1,3),dtype=np.int) #bresenhamline only works with 2D vectors with coordinates
        new_p0 = np.ndarray((1,3),dtype=np.int)
        for i in range(3):
            new_p1[0,i] = p1[i]
            new_p0[0,i] = p0[i]
            
        intermediate_points = bresenhamline(new_p0,new_p1,-1)
        if last:
            return np.concatenate((new_p0,intermediate_points))

        else:
            return intermediate_points        
   
    def get_grid(self):
        vals = [[self.xmin,self.xmax],[self.ymin,self.ymax],[self.zmin, self.zmax]]
        dx = np.zeros((self.est_xyz.shape[0]-1,self.est_xyz.shape[1]))
        dims = np.ones((3,),dtype=np.int)
        dxs = np.zeros((3,))
        for i in range(self.est_xyz.shape[1]):
            dx[:,i] = abs(self.est_xyz[1:,i]-self.est_xyz[:-1,i])
            try:
                dxs[i] = min(dx[dx[:,i]>self.tolerance,i])
            except ValueError:
                pass
                
            dims[i] = 1
            if dxs[i]:
                dims[i] += np.floor((vals[i][1]-vals[i][0])/dxs[i])
        return dims, dxs
                
    def point_coordinates(self,morpho):
        
        minis =  np.array([self.xmin,self.ymin,self.zmin])
        dims, dxs = self.get_grid()
        zero_coords = np.zeros((3,),dtype=int)
        coor_3D = np.zeros((morpho.shape[0]-1,morpho.shape[1]),dtype=np.int)
        for i,dx in enumerate(dxs):
            if dx:
                coor_3D[:,i] = np.floor((morpho[1:,i] - minis[None,i])/dxs[None,i])
                zero_coords[i] = np.floor((morpho[0,i] - minis[i])/dxs[i])
        return coor_3D, zero_coords
         
    def coordinates_3D_loops(self):
        
        coor_3D, p0 = self.point_coordinates(self.est_xyz)
        segment_coordinates = {}
        
        for i, p1 in enumerate(coor_3D):
            last = (i+1 ==len(coor_3D))
            segment_coordinates[i] = self.points_in_between(p0,p1,last)
            p0 = p1
        return segment_coordinates

    def coordinates_3D_segments(self):

        coor_3D, p0 = self.point_coordinates(self.morphology[:,2:5])
        segment_coordinates = {}
        
        parentage = self.morphology[1:,6]-2        
        i = 0
        p1 = coor_3D[0]
        while True:
            last = (i+1 ==len(coor_3D))

            segment_coordinates[i] = self.points_in_between(p0,p1,last)
            if i+1 == len(coor_3D):
                break
            if i:
                p0_idx = int(parentage[i+1])
                p0 = coor_3D[p0_idx]
            else:
                p0 = p1
            
            i = i+1
            p1 = coor_3D[i]
        return segment_coordinates
                
    def transform_to_3D(self, estimated, what="loop"):
        dims, dxs = self.get_grid()
        if what == "loop":
            coor_3D = self.coordinates_3D_loops()
        elif what == "morpho":
            coor_3D = self.coordinates_3D_segments()
        else:
            sys.exit('Do not understand morphology %s\n'%what)
            
        n_time = estimated.shape[-1]
        weights = np.zeros((dims))
        new_dims = list(dims)+[n_time]
        result = np.zeros(new_dims)

        for i in coor_3D:
            coor = coor_3D[i]
            
            for p in coor:
                x,y,z, = p
                
                result[x,y,z,:] += estimated[i,:]
                weights[x,y,z] += 1
                
        non_zero_weights = np.array(np.where(weights>0)).T
        
        for (x,y,z) in non_zero_weights:
            result[x,y,z,:] = result[x,y,z,:]/weights[x,y,z]
        return result

    def draw_cell2D(self,axis=2):
        
        resolution, dxs = self.get_grid()
        
        xgrid = np.linspace(self.xmin, self.xmax, resolution[0])
        ygrid = np.linspace(self.ymin, self.ymax, resolution[1])
        zgrid = np.linspace(self.zmin, self.zmax, resolution[2])
        #morphology = self.morphology_2D_for_images(axis=axis)
        if axis == 0:
            image = np.ones(shape=(resolution[1], resolution[2], 4), dtype=np.uint8) * 255
            extent = [1e6*self.ymin, 1e6*self.ymax, 1e6*self.zmin, 1e6*self.zmax]
        elif axis == 1:
            image = np.ones(shape=(resolution[0], resolution[2], 4), dtype=np.uint8) * 255
            extent = [1e6*self.xmin, 1e6*self.xmax, 1e6*self.zmin, 1e6*self.zmax]
        elif axis == 2:
            image = np.ones(shape=(resolution[0], resolution[1], 4), dtype=np.uint8) * 255
            extent = [1e6*self.xmin, 1e6*self.xmax, 1e6*self.ymin, 1e6*self.ymax]
        else:
            sys.exit('In drawing 2D morphology unknown axis ' + str(axis))
            
        
        image[:, :, 3] = 0
        xs = []
        ys = []
        x0,y0 = 0,0

        for p in range(self.loop_xyz.shape[0]):
            x = (np.abs(xgrid-self.loop_xyz[p,0])).argmin()
            y = (np.abs(ygrid-self.loop_xyz[p,1])).argmin()
            z = (np.abs(zgrid-self.loop_xyz[p,2])).argmin()
            if axis == 0:
                xi, yi = y,z
            elif axis == 1:
                xi, yi =  x,z
            elif axis == 2:
                xi, yi = x,y
            xs.append(xi)
            ys.append(yi)
            image[xi,yi,:] = np.array([0,0,0,1])
            if x0 !=0:
                
                idx_arr = self.points_in_between([xi,yi,0],[x0,y0,0])#getlinepoints(xi,yi,x0,y0)

                for i in range(len(idx_arr)):

                    image[idx_arr[i,0]-1:idx_arr[i,0]+1,idx_arr[i,1]-1:idx_arr[i,1]+1,:] = np.array([0,0,0,20])
            x0, y0 = xi, yi        
        
        return image,extent
    
    def plot3Dloop(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        X,Y,Z = self.source_xyz[:,0],self.source_xyz[:,1],self.source_xyz[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.plot(X,Y,Z)
        X,Y,Z = self.ele_pos[:,0], self.ele_pos[:,1], self.ele_pos[:,2]
        ax.scatter(X,Y,Z)
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
        plt.grid()
        plt.show()
     
if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    data_dir = os.path.join(path,"tutorials/Data/gang_7x7_200")
    data = ld.Data(data_dir)
    morphology = data.morphology
    data.morphology[:,2:5] = data.morphology[:,2:5]/1e6
    ele_pos = data.ele_pos
    n_src = 512
    cell = sKCSDcell(morphology,ele_pos,n_src)
    cell.distribute_srcs_3D_morph()
  
