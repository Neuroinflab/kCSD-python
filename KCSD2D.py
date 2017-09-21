"""
This script is used to generate Current Source Density Estimates, 
using the kCSD method Jan et.al (2012) for 2D case.

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Chaitanya Chintaluri, 
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
"""
from __future__ import print_function, division
import numpy as np
from scipy import integrate, interpolate
from scipy.spatial import distance
from numpy.linalg import LinAlgError

from KCSD import KCSD
import utility_functions as utils
import basis_functions as basis

class KCSD2D(KCSD):
    """KCSD2D - The 2D variant for the Kernel Current Source Density method.

    This estimates the Current Source Density, for a given configuration of 
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        """Initialize KCSD2D Class.

        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
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
                thickness of analyzed tissue slice
                Defaults to 1.
            xmin, xmax, ymin, ymax : floats
                boundaries for CSD estimation space
                Defaults to min(ele_pos(x)), and max(ele_pos(x))
                Defaults to min(ele_pos(y)), and max(ele_pos(y))
            ext_x, ext_y : float
                length of space extension: x_min-ext_x ... x_max+ext_x
                length of space extension: y_min-ext_y ... y_max+ext_y 
                Defaults to 0.
            gdx, gdy : float
                space increments in the estimation space
                Defaults to 0.01(xmax-xmin)
                Defaults to 0.01(ymax-ymin)
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
        super(KCSD2D, self).__init__(ele_pos, pots, **kwargs)
        return
        
    def estimate_at(self):
        """Defines locations where the estimation is wanted
        Defines:         
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy = self.estm_x.shape
        self.estm_x, self.estm_y : Locations at which CSD is requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        #Number of points where estimation is to be made.
        nx = (self.xmax - self.xmin)/self.gdx
        ny = (self.ymax - self.ymin)/self.gdy
        #Making a mesh of points where estimation is to be made.
        self.estm_x, self.estm_y = np.mgrid[self.xmin:self.xmax:np.complex(0,nx), 
                                            self.ymin:self.ymax:np.complex(0,ny)]
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy = self.estm_x.shape
        return

    def place_basis(self):
        """Places basis sources of the defined type.
        Checks if a given source_type is defined, if so then defines it
        self.basis, This function gives locations of the basis sources, 
        Defines
        source_type : basis_fuctions.basis_2D.keys()
        self.R based on R_init
        self.dist_max as maximum distance between electrode and basis
        self.nsx, self.nsy = self.src_x.shape
        self.src_x, self.src_y : Locations at which basis sources are placed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        source_type = self.src_type
        try:
            self.basis = basis.basis_2D[source_type]
        except:
            print('Invalid source_type for basis! available are:', basis.basis_2D.keys())
            raise KeyError
        #Mesh where the source basis are placed is at self.src_x 
        (self.src_x, self.src_y, self.R) = utils.distribute_srcs_2D(self.estm_x,
                                                                    self.estm_y,
                                                                    self.n_src_init,
                                                                    self.ext_x, 
                                                                    self.ext_y,
                                                                    self.R_init ) 
        self.n_src = self.src_x.size
        self.nsx, self.nsy = self.src_x.shape
        return        

    def create_src_dist_tables(self):
        src_loc = np.array((self.src_x.ravel(), self.src_y.ravel()))
        est_loc = np.array((self.estm_x.ravel(), self.estm_y.ravel()))
        self.src_ele_dists = distance.cdist(src_loc.T, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(src_loc.T, est_loc.T,  'euclidean')
        self.dist_max = max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R
        return

    def forward_model(self, x, R, h, sigma, src_type):
        """FWD model functions
        Evaluates potential at point (x,0) by a basis source located at (0,0)
        Eq 22 kCSD by Jan,2012

        Parameters
        ----------
        x : float
        R : float
        h : float
        sigma : float
        src_type : basis_2D.key

        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        pot, err = integrate.dblquad(self.int_pot_2D, 
                                     -R, R,
                                     lambda x: -R, 
                                     lambda x: R, 
                                     args=(x, R, h, src_type))
        pot *= 1./(2.0*np.pi*sigma)  #Potential basis functions bi_x_y
        return pot

    def int_pot_2D(self, xp, yp, x, R, h, basis_func):
        """FWD model function.
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)

        Parameters
        ----------
        xp, yp : floats or np.arrays
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
        y = ((x-xp)**2 + yp**2)**(0.5)
        if y < 0.00001:
            y = 0.00001
        dist = np.sqrt(xp**2 + yp**2)
        pot = np.arcsinh(h/y)*basis_func(dist, R) 
        return pot

if __name__ == '__main__':
    #Sample data, do not take this seriously
    ele_pos = np.array([[-0.2, -0.2],[0, 0], [0, 1], [1, 0], [1,1], [0.5, 0.5],
                        [1.2, 1.2]])
    pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
    k = KCSD2D(ele_pos, pots,
               gdx=0.05, gdy=0.05,
               xmin=-2.0, xmax=2.0,
               ymin=-2.0, ymax=2.0,
               src_type='gauss')
    k.cross_validate()
    #print(k.values('CSD'))
    #print(k.values('POT'))
