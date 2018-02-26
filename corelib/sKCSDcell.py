# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import collections
import utility_functions as utils
import os
import loadData as ld
#testing

class sKCSDcell(object):
    """
    KCSD3D - The 3D variant for the Kernel Current Source Density method.

    This estimates the Current Source Density, for a given configuration of 
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, morphology, ele_pos, n_src):
        """
        
        """
        self.morphology = morphology
        self.ele_pos = ele_pos
        self.n_src = n_src
        self.min_dist = 0
        self.max_dist = 0
        self.src_distributed = 0
        self.morph_points_dist = 0
        self.total_dist = self.calculate_total_distance()
        self.loop_pos = np.linspace(0,self.total_dist,n_src) #positions of sources on the morphology (1D)
        self.est_pos = []
        rep = collections.Counter(self.morphology[:,6])
        self.branching = [key for key in rep.keys() if rep[key]>1]
        self.source_xyz = np.zeros(shape=(n_src,3))
        self.loop_xyz = np.zeros(shape=(n_src+self.morphology.shape[0]*2,3))
        self.source_xyz_borders = []
        self.loops = []
        self.xmin =  np.min(self.morphology[:,2])
        self.xmax = np.max(self.morphology[:,2])
        self.ymin =  np.min(self.morphology[:,3])
        self.ymax = np.max(self.morphology[:,3])
        self.zmin =  np.min(self.morphology[:,4])
        self.zmax = np.max(self.morphology[:,4])
        self.dims = None
        self.dxs = None
        self.minis = minis = [self.xmin,self.ymin,self.zmin]
        
    def correct_min_max(self, xmin,xmax,radius):
        if xmin == xmax:
            xmin = xmin-radius
            xmax = xmax +radius
        return xmin, xmax
    
       
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
                    self.distribute_src_cylinder(parent,last_point)
                    if parent == last_branch:
                        break
                    last_point = parent
                self.distribute_src_cylinder(morph_pnt,int(self.morphology[morph_pnt,6])-1 )

        last_point = morph_pnt
        while True:
            parent = int(self.morphology[last_point,6]) - 1
            self.distribute_src_cylinder(parent,last_point)
            if int(self.morphology[parent,6]) == -1:
                break
            last_point = parent
        self.loops = np.array(self.loops)
        self.est_pos = np.zeros((len(self.loops),1))
        self.est_xyz = np.zeros((len(self.loops),3))
        self.est_xyz[0,:] = self.morphology[0,2:5]
        for i,loop in enumerate(self.loops[1:]):
            length = 0
            for j in [2,3,4]:
                length += (self.morphology[loop[1]][j]-self.morphology[loop[0]][j])**2
            self.est_pos[i+1] = self.est_pos[i] + length**0.5
            self.est_xyz[i+1,:] = self.morphology[loop[0],2:5]
        self.loop_pos = self.loop_pos.reshape(-1,1)

    def get_grid(self):
        vals = [[self.xmin,self.xmax],[self.ymin,self.ymax],[self.zmin, self.zmax]]
        dx = np.zeros((self.est_xyz.shape[0]-1,self.est_xyz.shape[1]))
        self.dims = np.ones((3,),dtype=np.int)
        self.dxs = np.zeros((3,))
        for i in range(self.est_xyz.shape[1]):
            dx[:,i] = abs(self.est_xyz[1:,i]-self.est_xyz[:-1,i])
            if not len(dx[dx[:,i]>1e-6,i]):
                continue
            self.dxs[i] = min(dx[dx[:,i]>2e-6,i])
            self.dims[i] = np.floor((vals[i][1]-vals[i][0])/self.dxs[i])+1

    def distribute_src_cylinder(self,mp1, mp2):
        xyz1 = self.morphology[mp1,2:5]
        xyz2 = self.morphology[mp2,2:5]
        self.loops.append([mp2,mp1])
        self.max_dist += np.linalg.norm(xyz1-xyz2)
        in_range = [idx for idx in range(self.src_distributed,self.n_src) 
                    if self.loop_pos[idx]<=self.max_dist or np.isclose(self.loop_pos[idx],self.max_dist)]
        
        self.src_distributed += len(in_range)
 
        if len(in_range)>0:
            for src_idx in in_range:
                self.source_xyz[src_idx,:] = xyz1-(xyz2-xyz1)*(self.loop_pos[src_idx]
                    -self.max_dist)/(self.max_dist-self.min_dist)
                self.loop_xyz[src_idx+self.morph_points_dist,:] =  xyz1-(xyz2-xyz1)*(self.loop_pos[src_idx]
                    -self.max_dist)/(self.max_dist-self.min_dist)
        self.min_dist = self.max_dist
        #Add morphology point to the loop
        self.loop_xyz[self.morph_points_dist+self.src_distributed] = self.morphology[mp1,2:5]
        self.morph_points_dist +=1
    
    def calculate_total_distance(self):
        total_dist = 0
        for i in range(1,self.morphology.shape[0]):
            xyz1 = self.morphology[i,2:5]
            xyz2 = self.morphology[int(self.morphology[i,6])-1,2:5]
            total_dist+=np.linalg.norm(xyz2-xyz1)
        total_dist*=2
        return total_dist
    
    def get_xyz(self):

        return self.source_xyz[:,0],self.source_xyz[:,1],self.source_xyz[:,2]
    
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
     
    def plot3Dmorph(self):
        for i in range(1,self.morphology.shape[0]):
            print("s")
    
    def getlinepoints(self,x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append([x,y])
        return np.array(points_in_line)
    
    def draw_cell2D(self,axis=2, resolution=(176,225,100)):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
       
        xgrid = np.linspace(self.xmin, self.xmax, resolution[0])
        ygrid = np.linspace(self.ymin, self.ymax, resolution[1])
        zgrid = np.linspace(self.zmin, self.zmax, resolution[2])
    
        if axis == 0:
            image = np.ones(shape=(resolution[1], resolution[2], 4), dtype=np.uint8) * 255
            extent = [self.ymin,self.ymax,self.zmin,self.zmax,]
        elif axis == 1:
            image = np.ones(shape=(resolution[0], resolution[2], 4), dtype=np.uint8) * 255
            extent = [self.xmin,self.xmax,self.zmin,self.zmax,]
        elif axis == 2:
            image = np.ones(shape=(resolution[0], resolution[1], 4), dtype=np.uint8) * 255
            extent = [self.xmin,self.xmax,self.ymin,self.ymax,]
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
                idx_arr = self.getlinepoints(xi,yi,x0,y0)
                for i in range(len(idx_arr)):

                    image[idx_arr[i,0]-1:idx_arr[i,0]+1,idx_arr[i,1]-1:idx_arr[i,1]+1,:] = np.array([0,0,0,20])
            x0, y0 = xi, yi

       
        plt.imshow(image,extent=extent,aspect='auto',origin="lower")
        plt.show()
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
    cell.get_grid()
    #cell.plot3Dloop()
    #cell.draw_cell2D(axis=0,resolution = (176,225,17))
    #cell.draw_cell2D(axis=1,resolution = (176,225,17))
    
    #cell.draw_cell2D(axis=2,resolution = (176,225,17))
