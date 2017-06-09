# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import collections


class sKCSDcell(object):
    def __init__(self, morphology, n_src):
        """
        """
        self.n_src = n_src
        self.morphology = morphology
        self.min_dist = 0
        self.max_dist = 0
        self.src_distributed = 0
        self.total_dist = self.calculate_total_distance()
        self.loop_pos = np.arange(0,self.total_dist,self.total_dist/n_src)
        rep = collections.Counter(self.morphology[:,6])
        self.branching = [key for key in rep.keys() if rep[key]>1]
        self.xyz = np.zeros(shape=(n_src,3))
        self.repeated = []
        self.counter= {}

    def distribute_srcs_3D_morph(self):
        for morph_pnt in range(1,self.morphology.shape[0]):
            if self.morphology[morph_pnt-1,0]==self.morphology[morph_pnt,6]:
                self.distribute_src_cylinder(morph_pnt, morph_pnt-1)
            elif self.morphology[morph_pnt,6] in self.branching:
                n_back = morph_pnt-self.counter[self.morphology[morph_pnt,6]]
                print morph_pnt,self.morphology[morph_pnt,6],"!!!",morph_pnt,morph_pnt-n_back-1
                for morph_pnt2 in range(morph_pnt-1,morph_pnt-n_back,-1):
                    if morph_pnt2 not in self.repeated and self.morphology[
                            morph_pnt2-1,0]==self.morphology[morph_pnt2,6]:
                        self.repeated.append(morph_pnt2)
                        self.distribute_src_cylinder(morph_pnt2, morph_pnt2-1)
                self.distribute_src_cylinder(morph_pnt,self.morphology[morph_pnt,6]-1 )

                if self.morphology[morph_pnt,6]==1: 
                    self.counter[self.morphology[morph_pnt,6]] = morph_pnt
                
    
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
            #self.src_distributed = in_range[-1]+1
            for src_idx in in_range:
                self.xyz[src_idx,:] = xyz1-(xyz2-xyz1)*(self.loop_pos[src_idx]
                    -self.max_dist)/(self.max_dist-self.min_dist)
        self.min_dist = self.max_dist
    
    
    def calculate_total_distance(self):
        total_dist = 0
        for i in range(1,self.morphology.shape[0]):
            xyz1 = self.morphology[i,2:5]
            xyz2 = self.morphology[self.morphology[i,6]-1,2:5]
            total_dist+=np.linalg.norm(xyz2-xyz1)
        total_dist*=2
        return total_dist
    
    def get_xyz(self):
        print self.src_distributed, self.n_src
        if self.src_distributed == self.n_src:
            print "correct"
        X,Y,Z = self.xyz[:,0],self.xyz[:,2],self.xyz[:,1]
        return X,Y,Z
    
    def plot3Dloop(self):
        X,Y,Z = self.xyz[:,0],self.xyz[:,2],self.xyz[:,1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.plot(X,Y,Z)
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
    #morphology = np.loadtxt('retina_ganglion.swc')        
    morphology = np.loadtxt('data/morpho1.swc')
    n_src = 1000
    cell = sKCSDcell(morphology,n_src)
    cell.distribute_srcs_3D_morph()
    cell.plot3Dloop()


