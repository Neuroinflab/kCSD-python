"""
This script is used to generate Current Source Density Estimates, 
using the sKCSD method Cserpan et.al (2017).

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Joanna Jedrzejewska-Szmek, Jan Maka, Chaitanya Chintaluri
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import os
from scipy.spatial import distance
from scipy import special, interpolate, integrate
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from skmonaco import mcmiser
    skmonaco_available = True
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
except ImportError:
    skmonaco_available = False
    
from corelib.KCSD import KCSD1D
from corelib.sKCSDcell import sKCSDcell
import corelib.utility_functions as utils
import corelib.basis_functions as basis
#testing

    
class sKCSD3D(KCSD1D):
    """KCSD3D - The 3D variant for the Kernel Current Source Density method.

    This estimates the Current Source Density, for a given configuration of 
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots,morphology, **kwargs):
        """Initialize KCSD3D Class.

        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        morphology: numpy array
            morphology of the cell
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the medium
                Defaults to 1.
            n_src_init : int
                requested number of sources
                Defaults to 1000
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                neuron radius
                Defaults to 10 um.
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.

        Returns
        -------
        None

        Raises
        ------
        LinAlgError 
            Could not invert the matrix, try changing the ele_pos slightly
        KeyError
            Basis function (src_type) not implemented. See basis_functions.py for available
        """
        self.morphology = morphology
        super(KCSD1D, self).__init__(ele_pos, pots, **kwargs)
        return
    
    def parameters(self, **kwargs):

        self.src_type = kwargs.pop('src_type', 'gauss')
        self.sigma = kwargs.pop('sigma', 1.0)
        self.h = kwargs.pop('h', 1e-5)
        self.n_src_init = kwargs.pop('n_src_init', 1000)
        self.lambd = kwargs.pop('lambd', 1e-4)
        self.R_init = kwargs.pop('R_init', 2.3e-5) #microns
        self.dist_table_density = kwargs.pop('dist_table_density',100)
        self.dim = 'skCSD'
        self.tolerance = kwargs.pop('tolerance',2e-06)
        if kwargs:
            raise TypeError('Invalid keyword arguments:', kwargs.keys())

    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:         
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy, self.ngz = self.estm_x.shape
        self.estm_x, self.estm_y, self.estm_z : Pts. at which CSD is requested

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def place_basis(self):
        """Places basis sources of the defined type.
        Checks if a given source_type is defined, if so then defines it
        self.basis, This function gives locations of the basis sources, 
        Defines
        source_type : basis_fuctions.basis_2D.keys()
        self.R based on R_init
        self.dist_max as maximum distance between electrode and basis
        self.nsx, self.nsy, self.nsz = self.src_x.shape
        self.src_x, self.src_y, self.src_z : Locations at which basis sources are placed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        #If Valid basis source type passed?
        source_type = self.src_type
        try:
            self.basis = basis.basis_1D[source_type]
        except:
            print('Invalid source_type for basis! available are:', basis.basis_1D.keys())
            raise KeyError
        #Mesh where the source basis are placed is at self.src_x
        self.R = self.R_init
        self.cell = sKCSDcell(self.morphology,self.ele_pos,self.n_src_init,self.tolerance)
        self.src_x = self.cell.distribute_srcs_3D_morph()
        self.n_src = self.cell.n_src
        self.n_estm = len(self.src_x)
        
        return        

    def create_src_dist_tables(self):
        """Creates distance tables between sources, electrode and estm points

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.src_ele_dists = distance.cdist(self.cell.source_xyz, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(self.cell.source_pos, self.cell.est_pos,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R
        return


    def forward_model(self, x, R, h, sigma, src_type):
        """FWD model functions
        Evaluates potential at point (x,0) by a basis source located at (0,0)
        Utlizies sk monaco monte carlo method if available, otherwise defaults
        to scipy integrate

        Parameters
        ----------
        x : float
        R : float
        h : float
        sigma : float
        src_type : basis_3D.key

        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        assert 2**1.5*R < self.cell.max_dist
        if skmonaco_available:
            pot, err = mcmiser(self.int_pot_1D, 
                               npoints=1e5,
                               xl= -2**1.5*R, 
                               xu=2**1.5*R+self.cell.max_dist,
                               seed=42, 
                               nprocs=num_cores, 
                               args=(x, R, src_type))
        else:
            pot, err = integrate.quad(self.int_pot_1D, 
                                      -2**1.5*R, 
                                      2**1.5*R+self.cell.max_dist,
                                      args=(x, R, src_type))

        return pot/(4.0*np.pi*sigma)

    
    def potential_at_the_electrodes(self):
        
        estimation = np.zeros((self.n_ele,self.n_time))
        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        for t in range(self.n_time):
            beta = np.dot(k_inv, self.pots[:, t])
            for i in range(self.n_ele):
                
                estimation[:, t] += self.k_pot[:, i]*beta[i]  # C*(x) Eq 18
        return estimation

    def values(self, estimate='CSD',segments=False,no_transformation=False):
        '''In skCSD CSD is calculated on the morphology, which is 1D, and
        the CSD needs to be translated to cartesian coordinates.

        '''
        #estimate self.n_src_init x self.n_time

        estimated = super(sKCSD3D,self).values(estimate=estimate)
        if no_transformation:
            return estimated
        if segments:
            result = np.zeros((self.cell.morphology.shape[0]-1,estimated.shape[1]))
            weights = np.zeros((self.cell.morphology.shape[0]-1))

            for i, loop in enumerate(self.cell.loops):
                key = "%d_%d"%(loop[0],loop[1])
                seg_no = self.cell.segments[key]
                
                result[seg_no,:] += estimated[i,:]
                weights[seg_no] += 1

            return result/weights[:,None]
        
        return self.cell.transform_to_3D(estimated,what="loop")
   
    def int_pot_1D(self, xp, x, R, basis_func):
        """FWD model function.
        Returns contribution of a point sp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)
        Eq 26 kCSD by Jan,2012

        Parameters
        ----------
        xp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        h : float
            thickness of slice
        basis_func : method
            Fuction of the basis source

        Returns
        -------
        pot : float
        """
        if xp > self.cell.max_dist:
            xp = xp - self.cell.max_dist
        elif xp < 0:
            xp = xp +  self.cell.max_dist

        dist = 0
        xp_coor = self.cell.get_xyz(xp)
        point = (x,0,0)
        for i,p in enumerate(point):
            dist += (p-xp_coor[i])**2
        pot = basis_func(xp, R)/np.sqrt(dist)# xp is the distance
        
        return pot

    

if __name__ == '__main__':
    import loadData as ld
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    data_dir = os.path.join(path,"tutorials/Data/gang_7x7_200")
    data = ld.Data(data_dir)
    scaling_factor = 1000000
    ele_pos = data.ele_pos/scaling_factor
    pots = data.LFP[:,:200]
    params = {}
    morphology = data.morphology 
    morphology[:,2:6] = morphology[:,2:6]/scaling_factor
    R_init = 32/scaling_factor
   
    k = sKCSD3D(ele_pos, pots,morphology,n_src_init=1000, src_type='gauss_lim', R_init=R_init)
    #k.cross_validate()
    
    if sys.version_info >= (3, 0):
        path = os.path.join(data_dir,"preprocessed_data/Python_3")
    else:
        path = os.path.join(data_dir,"preprocessed_data/Python_2")

    if not os.path.exists(path):
        print("Creating",path)
        os.makedirs(path)
        
    utils.save_sim(path,k)
    #est_csd = k.values("CSD")
    #est_pot = k.values("POT")
    
    
    
