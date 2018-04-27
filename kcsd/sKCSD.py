"""
This script is used to generate Current Source Density Estimates, 
using the skCSD method Cserpan et.al (2017).

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
from kcsd import KCSD1D, sKCSDcell

from . import utility_functions as utils
from . import basis_functions as basis

try:
    from skmonaco import mcmiser
    skmonaco_available = True
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
except ImportError:
    skmonaco_available = False

class sKCSD(KCSD1D):
    """KCSD3D - The 3D variant for the Kernel Current Source Density method.

    This estimates the Current Source Density,
    using the skCSD method Cserpan et.al (2017).
    """
    def __init__(self, ele_pos, pots, morphology, **kwargs):
        """Initialize sKCSD Class.

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
            h : float
                tissue thickness, unused in sKCSD
                Defaults to 10 um
            n_src_init : int
                requested number of sources
                Defaults to 1000
            R_init : float
                demanded thickness of the basis element
                Defaults to 23 um
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.
            dist_table_density : int
                size of the potential interpolation table
                Defaults to 20
            tolerance : float
                minimum neurite size used for 3D tranformation of CSD and potential
                Defaults to 2 um
            
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
        This is done while construction of morphology loop in sKCSDcell
        Defines:
        self.n_estm = len(self.cell.estm_x)
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.cell = sKCSDcell(self.morphology,self.ele_pos,self.n_src_init,self.tolerance)
        self.n_estm = len(self.cell.est_pos)
        return

    def place_basis(self):
        """Places basis sources of the defined type.
        Checks if a given source_type is defined, if so then defines it
        self.basis, This function gives locations of the basis sources, 
        Defines
        source_type : basis_fuctions.basis_2D.keys()
        self.R based on R_init
       

        self.src_x: Locations at which basis sources are placed.
        self.n_src: amount of placed basis sources

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        #If Valid basis source type passed?
        self.R = self.R_init
        source_type = self.src_type
        try:
            self.basis = basis.basis_1D[source_type]
        except:
            print('Invalid source_type for basis! available are:', basis.basis_1D.keys())
            raise KeyError
        #Mesh where the source basis are placed is at self.src_x
       
        self.src_x = self.cell.distribute_srcs_3D_morph()
        self.n_src = self.cell.n_src
          
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
        src_loc = self.cell.source_xyz
        est_pos = self.cell.est_pos
        source_pos = self.src_x
        self.src_ele_dists = distance.cdist(src_loc, self.ele_pos, 'euclidean')
        self.src_estm_dists = distance.cdist(source_pos, est_pos,  'euclidean')
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
            pot, err = mcmiser(self.int_pot_1D_mc, 
                               npoints=1e5,
                               xl= [-2**1.5*R], 
                               xu=[2**1.5*R+self.cell.max_dist],
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
        """
        Reconstruction from CSD of potentials measured at the electrodes

        Parameters
        ----------
        None

        Returns
        -------
        estimation : np.array
            Potential generated by the CSD measured at the electrodes
        """
        estimation = np.zeros((self.n_ele,self.n_time))
        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        for t in range(self.n_time):
            beta = np.dot(k_inv, self.pots[:, t])
            for i in range(self.n_ele):
                
                estimation[:, t] += self.k_pot[:, i]*beta[i]  # C*(x) Eq 18
        return estimation

    def values(self, estimate='CSD', transformation='3D'):
        '''Computes the values of the quantity of interest

        Parameters
        ----------
        estimate: 'CSD' or 'POT'
            What quantity is to be estimated
            Defaults to 'CSD'
        transformation: '3D', 'segments', None
            Specify representation of the estimated quantity
            '3D' -- quantity is represented in cartesian coordinate system
            'segments' -- quantity is represented insegments
            None -- quantity is represented in the morphology loop
        
        Returns
        -------
        estimation : np.array
            estimated quantity
        '''
        estimated = super(sKCSD,self).values(estimate=estimate)
        
        if not transformation:
            return estimated
        elif transformation == 'segments':
            return self.cell.transform_to_segments(estimated)
        elif transformation == '3D':
            return self.cell.transform_to_3D(estimated,what="loop")

        raise Exception("Unknown transformation %s of %s"%(transformation, estimate))
   
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

        xp_coor = self.cell.get_xyz(xp)
        dist = ((x-xp_coor[0])**2+xp_coor[1]**2+xp_coor[2]**2)**0.5
        if dist < 0.00001:
            dist = 0.00001
        pot = basis_func(xp, R)/dist# xp is the distance
        return pot

    def int_pot_1D_mc(self, xyz, x, R, basis_func):
        """
        The same as int_pot_1D, just different input: x <-- xp (tuple)
        FWD model function, using Monte Carlo Method of integration
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)

        Parameters
        ----------
        xp : floats or np.arrays
            point or set of points where function should be calculated
        x :  float
            position at which potential is being measured
        R : float
            The size of the basis function
        basis_func : method
            Fuction of the basis source

        Returns
        -------
        pot : float
        """
        xp = xyz[0]
        return self.int_pot_1D(xp, x, R, basis_func)
if __name__ == '__main__':
    import argparse
    uwd = os.path.abspath('..')
    md = os.path.join(uwd,'tests/Data/ball_and_stick_8/morphology/Figure_2_rows_8.swc')
    lfpd = os.path.join(uwd,'tests/Data/ball_and_stick_8/LFP/MyLFP')
    eleposd = os.path.join(uwd,'tests/Data/ball_and_stick_8/electrode_positions/elcoord_x_y_x')
    
    parser = argparse.ArgumentParser(description='Calculate current/potential estimation using sKCSD')
    parser.add_argument('--morphology',type=str, metavar='morphology', default=md,
                    help='path to neuron morphology in swc file format')
    parser.add_argument('--LFP',type=str,metavar='LFP', default=lfpd,
                    help='path to LFP measurements')
    parser.add_argument('--electrode_positions',type=str,metavar='electrode_positions', default=eleposd,
                    help='path to electrode positions')
    parser.add_argument('--save_to',type=str,metavar='save_to', default=os.path.join(uwd,'tests/Data/ball_and_stick_8'),
                    help='path to results')
    parser.add_argument('--src_type',choices=set(('gauss', 'step', 'gauss_lim')), default='gauss', help='basis function type')
    parser.add_argument('--R_init',type=float, default=23e-6, help='width of basis function')
    parser.add_argument('--lambd',type=float, default=1e-1, help='regularization parameter for ridge regression')
    parser.add_argument('--n_src',type=int, default=300, help='requested number of sources')
    parser.add_argument('--sigma',type=float, default=1, help='space conductance of the tissue in S/m')
    parser.add_argument('--dist_table_density',type=int, default=100, help='size of the potential interpolation table')
    args = parser.parse_args()

    morphology = utils.load_swc(args.morphology)
    myLFP = np.loadtxt(args.LFP)
    electrode_positions = utils.load_elpos(args.electrode_positions)
   
    scaling_factor = 1000000
    ele_pos = electrode_positions/scaling_factor
    morphology[:,2:6] = morphology[:,2:6]/scaling_factor
    R_init = 32/scaling_factor
   
    k = sKCSD(ele_pos, myLFP, morphology, n_src_init=args.n_src, src_type=args.src_type, R_init=args.R_init, lambd=args.lambd, sigma=args.sigma, dist_table_density=args.dist_table_density)
    ker_dir = args.save_to
    if sys.version_info < (3,0):
        path = os.path.join(ker_dir, "preprocessed_data/Python_2")
    else:
        path = os.path.join(ker_dir, "preprocessed_data/Python_3")

    if not os.path.exists(path):
        print("Creating",path)
        os.makedirs(path)
    utils.save_sim(path,k)
    
    
