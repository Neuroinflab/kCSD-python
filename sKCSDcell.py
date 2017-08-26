# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import utility_functions as utils
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
        self.loop_pos = np.arange(0,self.total_dist,self.total_dist/n_src)
        rep = collections.Counter(self.morphology[:,6])
        self.branching = [key for key in rep.keys() if rep[key]>1]
        self.source_xyz = np.zeros(shape=(n_src,3))
        self.loop_xyz = np.zeros(shape=(n_src+self.morphology.shape[0]*2,3))
        self.repeated = []
        self.counter= {}
        self.source_xyz_borders = []
        
        self.xmin =  np.min(self.morphology[:,2])
        self.xmax = np.max(self.morphology[:,2])
        self.ymin =  np.min(self.morphology[:,4])
        self.ymax = np.max(self.morphology[:,4])
        self.zmin =  np.min(self.morphology[:,3])
        self.zmax = np.max(self.morphology[:,3])
        

    def distribute_srcs_3D_morph(self):
        for morph_pnt in range(1,self.morphology.shape[0]):
            if self.morphology[morph_pnt-1,0]==self.morphology[morph_pnt,6]:
                self.distribute_src_cylinder(morph_pnt, morph_pnt-1)
            elif self.morphology[morph_pnt,6] in self.branching:
                last_branch = int(self.morphology[morph_pnt,6])-1
                for morph_pnt2 in range(morph_pnt-1,last_branch,-1):
                    if morph_pnt2 not in self.repeated and self.morphology[
                            morph_pnt2-1,0]==self.morphology[morph_pnt2,6]:
                        self.repeated.append(morph_pnt2)
                        self.distribute_src_cylinder(morph_pnt2-1,morph_pnt2)
                self.distribute_src_cylinder(morph_pnt,int(self.morphology[morph_pnt,6])-1 )
                if self.morphology[morph_pnt,6]==1: 
                    self.counter[self.morphology[morph_pnt,6]] = morph_pnt
        last_soma = self.counter[1]
        for morph_pnt2 in range(morph_pnt-1,last_soma,-1):
            if morph_pnt2 not in self.repeated and self.morphology[
                    morph_pnt2-1,0]==self.morphology[morph_pnt2,6]:
                self.repeated.append(morph_pnt2)
                self.distribute_src_cylinder(morph_pnt2-1,morph_pnt2)
                
    
    def distribute_src_cylinder(self,mp1, mp2):
        xyz1 = self.morphology[mp1,2:5]
        xyz2 = self.morphology[mp2,2:5]
        self.max_dist += np.linalg.norm(xyz1-xyz2)
        in_range = [idx for idx in range(self.src_distributed,self.n_src) 
            if self.loop_pos[idx]<=self.max_dist]
        self.src_distributed += len(in_range)
        if mp1 in self.branching:
                self.counter[mp1] = mp1
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
        print self.src_distributed, self.n_src
        if self.src_distributed == self.n_src:
            print "correct"
        X,Y,Z = self.source_xyz[:,0],self.source_xyz[:,1],self.source_xyz[:,2]
        return X,Y,Z
    
    def plot3Dloop(self):
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
            print "s"
    
    def getlinepoints(self,x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append([x,y])
        return np.array(points_in_line)
    
    def draw_cell2D(self,axis = 0, resolution = 1000):
        print self.morph_points_dist
        print self.src_distributed
        print self.branching
        print len(self.morphology)
        image = np.zeros(shape = (resolution, resolution))
        xgrid = np.arange(self.xmin, self.xmax, (self.xmax-self.xmin)/resolution)
        ygrid = np.arange(self.ymin, self.ymax, (self.ymax-self.ymin)/resolution)
        zgrid = np.arange(self.zmin, self.zmax, (self.zmax-self.zmin)/resolution)
        print self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax
        xs = []
        ys = []
        x0,y0 = 0,0
        for p in range(self.loop_xyz.shape[0]):
            z = (np.abs(zgrid-self.loop_xyz[p,1])).argmin()
            y = (np.abs(ygrid-self.loop_xyz[p,2])).argmin()
            x = (np.abs(xgrid-self.loop_xyz[p,0])).argmin()
            if axis == 0:
                xi, yi = y,z
            elif axis == 1:
                xi, yi = -z, x
            elif axis == 2:
                xi, yi = x,y    
            xs.append(xi)
            ys.append(yi)
            image[xi,yi] = 255
            if x0 !=0:
                idx_arr = self.getlinepoints(xi,yi,x0,y0)
                for i in range(len(idx_arr)):
                    image[idx_arr[i,0]-2:idx_arr[i,0]+2,idx_arr[i,1]-2:idx_arr[i,1]+2] = 255
            x0, y0 = xi, yi
        #plt.imshow(image)
        plt.plot(self.source_xyz[:,2],self.source_xyz[:,1],'ro')
        plt.plot(self.loop_xyz[:,2],self.loop_xyz[:,1],'g-')
        plt.plot(self.morphology[:,4],self.morphology[:,3],'o')
        plt.show()
        
if __name__ == '__main__':
    morphology = np.loadtxt('../morphology/Badea2011Fig2Du.CNG.swc')  
    ele_pos = utils.load_elpos("..\simData_skCSD\gang_7x7_200\elcoord_x_y_z")      
    #morphology = np.loadtxt('data/morpho1.swc')
    n_src = 10000
    cell = sKCSDcell(morphology,ele_pos,n_src)
    cell.distribute_srcs_3D_morph()
    cell.plot3Dloop()
    #cell.draw_cell2D(axis=1)
    plt.savefig("movied.png")


