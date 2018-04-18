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
    sKCSDcell -- construction of the morphology loop for sKCSD method (Cserpan et al., 2017).

    This calculates the morphology loop and helps transform CSDestimates/potential 
    estimates from loop/segment space to 3D. The method implented here is based 
    on the original paper by Dorottya Cserpan et al., 2017.
    """
    def __init__(self, morphology, ele_pos, n_src,tolerance=2e-6):
        """
        Parameters
        ----------
        morphology : np.array
            morphology array (swc format)
        ele_pos : np.array
            electrode positions
        n_src : int
            number of sources
        tolerance : float
            minimum size of dendrite used to calculate 3 D grid parameters
        """
        self.morphology = morphology #morphology file
        self.ele_pos = ele_pos #electrode_positions
        self.n_src = n_src #number of sources
        self.max_dist = 0 #maximum distance
        self.segments = {} #segment dictionary -- keys are loops (eg '2_3' -- segment 2
        self.segment_counter = 0 #which segment we're on
        rep = Counter(self.morphology[:,6])
        self.branching = [int(key) for key in rep.keys() if rep[key]>1] #branchpoints
        self.morphology_loop() #make the morphology loop
        self.source_pos = np.zeros((n_src,1))
        self.source_pos[:,0] = np.linspace(0, self.max_dist, n_src) #positions of sources on the morphology (1D), necessary for source division
        self.source_xyz = np.zeros(shape=(n_src,3))#Cartesian coordinates of the sources

        #max and min points of the neuron's morphology
        self.xmin =  np.min(self.morphology[:,2])
        self.xmax = np.max(self.morphology[:,2])
        self.ymin =  np.min(self.morphology[:,3])
        self.ymax = np.max(self.morphology[:,3])
        self.zmin =  np.min(self.morphology[:,4])
        self.zmax = np.max(self.morphology[:,4])
        self.tolerance = tolerance #smallest dendrite used in visualisation

    
    def add_segment(self, mp1, mp2):
        """Add indices (mp1, mp2) of morphology points defining a segment
        to a dictionary of segments. 
        This dictionary is used for CSD/potential trasformation from
        loops to segments.

        Parameters
        ----------
        mp1: int
        mp2: int

        """
        
        key1 = "%d_%d"%(mp1, mp2)
        key2 = "%d_%d"%(mp2, mp1)
        
        if key1 not in  self.segments:
            self.segments[key1] = self.segment_counter
            self.segments[key2] = self.segment_counter
            self.segment_counter += 1

    def add_loop(self, mp1, mp2):
        """Add indices of morphology points defining a loop to list of loops.
        Increase maximum distance counter.

        Parameters
        ----------
        mp1: int
        mp2: int

        """
        self.add_segment(mp1,mp2)
        xyz1 = self.morphology[mp1,2:5]
        xyz2 = self.morphology[mp2,2:5]
        self.loops.append([mp2,mp1])
        self.max_dist += np.linalg.norm(xyz1-xyz2)
            
    def morphology_loop(self):
        """Cover the morphology of the cell with loops.

        Parameters
        ----------
        None
        """
        #loop over morphology
        self.loops = []
        for morph_pnt in range(1,self.morphology.shape[0]):
            if self.morphology[morph_pnt-1,0]==self.morphology[morph_pnt,6]:
                self.add_loop(morph_pnt, morph_pnt-1)
            elif self.morphology[morph_pnt,6] in self.branching:
                last_branch = int(self.morphology[morph_pnt,6])-1
                last_point = morph_pnt - 1
                while True:
                    parent = int(self.morphology[last_point,6]) - 1
                    self.add_loop(parent, last_point)
                    if parent == last_branch:
                        break
                    last_point = parent
                    
                self.add_loop(morph_pnt,int(self.morphology[morph_pnt,6])-1)
                
        last_point = morph_pnt
        while True:
            parent = int(self.morphology[last_point,6]) - 1
            self.add_loop(parent, last_point)
            if int(self.morphology[parent,6]) == -1:
                break
            last_point = parent

        #find estimation points
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
         
    def distribute_srcs_3D_morph(self):
        """
        Calculate 3D coordinates of sources placed on the morphology loop.

        Parameters
        ----------
        None
        """
        for i,x in enumerate(self.source_pos):
            self.source_xyz[i] = self.get_xyz(x)
        return self.source_pos
            
    def get_xyz(self, x):
        """Find cartesian coordinates of a point (x) on the morphology loop. Use
        morphology point cartesian coordinates (from the morphology file, 
        self.est_xyz) for interpolation.

        Parameters
        ----------
        x : float

        Returns
        -------
        tuple of length 3
        """
        return interpolate.interp1d(self.est_pos[:,0],self.est_xyz, kind='slinear',axis=0)(x)
    
    def calculate_total_distance(self):
        """
        Calculates doubled total legth of the cell.

        Parameteres
        -----------
        None
        """
        
        total_dist = 0
        for i in range(1,self.morphology.shape[0]):
            xyz1 = self.morphology[i,2:5]
            xyz2 = self.morphology[int(self.morphology[i,6])-1,2:5]
            total_dist+=np.linalg.norm(xyz2-xyz1)
        total_dist*=2
        return total_dist
      
    def points_in_between(self,p1,p0,last):
        """Wrapper for the Bresenheim algorythm, which accepts only 2D vector
        coordinates. last -- p0 is included in output

        Parameters
        ----------
        p1, p0: sequence of length 3
        last : int

        Return
        -----
        np.array 
        points between p0 and p1 including (last=True) or not including p0
        """
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
        """Calculate parameters of the 3D grid used to transform CSD 
        (or potential) according to eq. (22). self.tolerance is used 
        to specify smalles possible size of neurite.
        
        Parameters
        ----------
        None

        Returns
        -------

        dims: np.array of 3 ints
        CSD/potential array 3D coordinates
        dxs: np.array of 3 floats
        space grain of the 3D CSD/potential
        """
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
        """
        Calculate indices of points in morpho in the 3D grid calculated
        using self.get_grid()

        Parameters
        ----------
        morpho : np.array
           array with a morphology (either segements or morphology loop)

        Returns
        -------
        coor_3D : np.array
        zero_coords : np.array
           indices of morpho's initial point
        """
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
        """
        Find points of each loop in 3D grid 
        (for CSD/potential calculation in 3D).
        
        Parameters
        ----------
        None

        Returns
        -------
        segment_coordinates : np.array
           Indices of points of 3D grid for each loop
        
        """
        coor_3D, p0 = self.point_coordinates(self.est_xyz)
        segment_coordinates = {}
        
        for i, p1 in enumerate(coor_3D):
            last = (i+1 ==len(coor_3D))
            segment_coordinates[i] = self.points_in_between(p0,p1,last)
            p0 = p1
        return segment_coordinates

    def coordinates_3D_segments(self):
        """
        Find points of each segment in 3D grid 
        (for CSD/potential calculation in 3D).
        
        Parameters
        ----------
        None

        Returns
        -------
        segment_coordinates : np.array
           Indices of points of 3D grid for each segment
        
        """

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
        """ 
        Transform potential/csd/ground truth values in segment or loop space 
        to 3D.

        Parameters
        ----------
        estimated : np.array
        what : string
           "loop" -- estimated is in loop space
           "morpho" -- estimated in in segment space

        Returns
        -------
        result : np.array
        """
        dims, dxs = self.get_grid()
        if what == "loop":
            coor_3D = self.coordinates_3D_loops()
        elif what == "morpho":
            coor_3D = self.coordinates_3D_segments()
        else:
            sys.exit('Do not understand morphology %s\n'%what)
            
        n_time = estimated.shape[-1]
        new_dims = list(dims)+[n_time]
        result = np.zeros(new_dims)

        for i in coor_3D:
            coor = coor_3D[i]
            
            for p in coor:
                x,y,z, = p
                result[x,y,z,:] += estimated[i,:]

        return result

    def draw_cell2D(self,axis=2):
        """
        Cell morphology in 3D grid in projection of axis.

        Parameters
        ----------
        axis : int
          0: x axis, 1: y axis, 2: z axis 
        """
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

        for p in range(self.source_xyz.shape[0]):
            x = (np.abs(xgrid-self.source_xyz[p,0])).argmin()
            y = (np.abs(ygrid-self.source_xyz[p,1])).argmin()
            z = (np.abs(zgrid-self.source_xyz[p,2])).argmin()
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
                
                idx_arr = self.points_in_between([xi,yi,0],[x0,y0,0],0)

                for i in range(len(idx_arr)):

                    image[idx_arr[i,0]-1:idx_arr[i,0]+1,idx_arr[i,1]-1:idx_arr[i,1]+1,:] = np.array([0,0,0,20])
            x0, y0 = xi, yi        
        
        return image,extent
    
   
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
  
