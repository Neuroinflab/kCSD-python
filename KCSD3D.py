"""
This script is used to generate Current Source Density Estimates, 
using the kCSD method Jan et.al (2012) for 3D case.

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Chaitanya Chintaluri  
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
import numpy as np
from scipy.spatial import distance
try:
    from skmonaco import mcmiser
    skmonaco_available = True
except ImportError:
    from scipy import integrate    
    skmonaco_available = False
    
from KCSD2D import KCSD2D
import utility_functions as utils
import basis_functions as basis
    
class KCSD3D(KCSD2D):
    """KCSD3D - The 3D variant for the Kernel Current Source Density method.

    This estimates the Current Source Density, for a given configuration of 
    electrod positions and recorded potentials, in the case of 2D recording
    electrodes. The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        """Initialize KCSD3D Class.

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
            xmin, xmax, ymin, ymax, zmin, zmax : floats
                boundaries for CSD estimation space
                Defaults to min(ele_pos(x)), and max(ele_pos(x))
                Defaults to min(ele_pos(y)), and max(ele_pos(y))
                Defaults to min(ele_pos(z)), and max(ele_pos(z))
            ext_x, ext_y : float
                length of space extension: xmin-ext_x ... xmax+ext_x
                length of space extension: ymin-ext_y ... ymax+ext_y 
                length of space extension: zmin-ext_z ... zmax+ext_z 
                Defaults to 0.
            gdx, gdy, gdz : float
                space increments in the estimation space
                Defaults to 0.01(xmax-xmin)
                Defaults to 0.01(ymax-ymin)
                Defaults to 0.01(zmax-zmin)
            lambd : float
                regularization parameter for ridge regression
                Defaults to 0.

        Returns
        -------
        None
        """
        super(KCSD3D, self).__init__(ele_pos, pots, **kwargs)

    def parameters(self, **kwargs):
        """Defining the default values of the method passed as kwargs
        Parameters
        ----------
        **kwargs
            Same as those passed to initialize the Class

        Returns
        -------
        None
        """
        self.src_type = kwargs.get('src_type', 'gauss')
        self.sigma = kwargs.get('sigma', 1.0)
        self.h = kwargs.get('h', 1.0)
        self.n_src_init = kwargs.get('n_src_init', 1000)
        self.ext_x = kwargs.get('ext_x', 0.0)
        self.ext_y = kwargs.get('ext_y', 0.0)
        self.ext_z = kwargs.get('ext_z', 0.0)
        self.lambd = kwargs.get('lambd', 0.0)
        self.R_init = kwargs.get('R_init', 0.23)
        #If no estimate plane given, take electrode plane as estimate plane
        self.xmin = kwargs.get('xmin', np.min(self.ele_pos[:, 0]))
        self.xmax = kwargs.get('xmax', np.max(self.ele_pos[:, 0]))
        self.ymin = kwargs.get('ymin', np.min(self.ele_pos[:, 1]))
        self.ymax = kwargs.get('ymax', np.max(self.ele_pos[:, 1]))
        self.zmin = kwargs.get('zmin', np.min(self.ele_pos[:, 2]))
        self.zmax = kwargs.get('zmax', np.max(self.ele_pos[:, 2]))
        #Space increment size in estimation
        self.gdx = kwargs.get('gdx', 0.01*(self.xmax - self.xmin)) 
        self.gdy = kwargs.get('gdy', 0.01*(self.ymax - self.ymin))
        self.gdz = kwargs.get('gdz', 0.01*(self.zmax - self.zmin))
        return

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
        #Number of points where estimation is to be made.
        nx = (self.xmax - self.xmin)/self.gdx
        ny = (self.ymax - self.ymin)/self.gdy
        nz = (self.zmax - self.zmin)/self.gdz
        #Making a mesh of points where estimation is to be made.
        self.estm_x, self.estm_y, self.estm_z = np.mgrid[self.xmin:self.xmax:np.complex(0,nx), 
                                                         self.ymin:self.ymax:np.complex(0,ny),
                                                         self.zmin:self.zmax:np.complex(0,nz)]
        self.n_estm = self.estm_x.size
        self.ngx, self.ngy, self.ngz = self.estm_x.shape
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
        if source_type not in basis.basis_3D.keys():
            raise Exception('Invalid source_type for basis! available are:', basis.basis_3D.keys())
        else:
            self.basis = basis.basis_3D.get(source_type)
        #Mesh where the source basis are placed is at self.src_x 
        (self.src_x, self.src_y, self.src_z, self.R) = utils.distribute_srcs_3D(self.estm_x,
                                                                                self.estm_y,
                                                                                self.estm_z,
                                                                                self.n_src_init,
                                                                                self.ext_x, 
                                                                                self.ext_y,
                                                                                self.ext_z,
                                                                                self.R_init)

        #Total diagonal distance of the area covered by the basis sources
        Lx = np.max(self.src_x) - np.min(self.src_x) + self.R
        Ly = np.max(self.src_y) - np.min(self.src_y) + self.R
        Lz = np.max(self.src_z) - np.min(self.src_z) + self.R
        self.dist_max = (Lx**2 + Ly**2 + Lz**2)**0.5
        self.n_src = self.src_x.size
        self.nsx, self.nsy, self.nsz = self.src_x.shape
        return        

    def values(self, estimate='CSD'):
        """Computes the values of the quantity of interest

        Parameters
        ----------
        estimate : 'CSD' or 'POT'
            What quantity is to be estimated
            Defaults to 'CSD'

        Returns
        -------
        estimated quantity of shape (ngx, ngy, ngz, nt)
        """
        if estimate == 'CSD': #Maybe used for estimating the potentials also.
            estimation_table = self.k_interp_cross 
        elif estimate == 'POT':
            estimation_table = self.k_interp_pot
        else:
            print 'Invalid quantity to be measured, pass either CSD or POT'
        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        estimation = np.zeros((self.n_estm, self.n_time))
        for t in xrange(self.n_time):
            beta = np.dot(k_inv, self.pots[:, t])
            for i in xrange(self.n_ele):
                estimation[:, t] += estimation_table[:, i] *beta[i] # C*(x) Eq 18
        estimation = estimation.reshape(self.ngx, self.ngy, self.ngz, self.n_time)
        return estimation

    def create_lookup(self, dist_table_density=100):
        """Creates a table for easy potential estimation from CSD.
        Updates and Returns the potentials due to a given basis 
        source like a lookup table whose 
        shape=(dist_table_density,)--> set in KCSD2D_Helpers.py

        Parameters
        ----------
        dist_table_density : int
            number of distance values at which potentials are computed.
            Default 100

        Returns
        -------
        None
        """
        dt_len = dist_table_density
        xs = utils.sparse_dist_table(self.R, self.dist_max, #Find pots at sparse points
                                     dt_len)
        dist_table = np.zeros(len(xs))
        for i, x in enumerate(xs):
            pos = (x/dt_len) * self.dist_max
            dist_table[i] = self.forward_model(pos, 
                                               self.R, 
                                               self.h, 
                                               self.sigma,
                                               self.basis)
        self.dist_table = utils.interpolate_dist_table(xs, dist_table, dt_len) #and then interpolated
        return self.dist_table #basis potentials in a look up table

    def update_b_pot(self):
        """Updates the b_pot  - array is (#_basis_sources, #_electrodes)
        Updates the  k_pot - array is (#_electrodes, #_electrodes) K(x,x') 
        Eq9,Jan2012
        Calculates b_pot - matrix containing the values of all
        the potential basis functions in all the electrode positions
        (essential for calculating the cross_matrix).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        src = np.array((self.src_x.ravel(), self.src_y.ravel(), self.src_z.ravel()))
        dists = distance.cdist(src.T, self.ele_pos, 'euclidean')
        self.b_pot = self.generated_potential(dists)
        self.k_pot = np.dot(self.b_pot.T, self.b_pot) #K(x,x') Eq9,Jan2012
        self.k_pot /= self.n_src
        return self.b_pot

    def update_b_src(self):
        """Updates the b_src in the shape of (#_est_pts, #_basis_sources)
        Updates the k_interp_cross - K_t(x,y) Eq17
        Calculate b_src - matrix containing containing the values of
        all the source basis functions in all the points at which we want to
        calculate the solution (essential for calculating the cross_matrix)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.b_src = np.zeros((self.ngx, self.ngy, self.ngz, self.n_src))
        for i in xrange(self.n_src):
            # getting the coordinates of the i-th source
            (i_x, i_y, i_z) = np.unravel_index(i, (self.nsx, self.nsy, self.nsz), order='C')
            x_src = self.src_x[i_x, i_y, i_z]
            y_src = self.src_y[i_x, i_y, i_z]
            z_src = self.src_z[i_x, i_y, i_z]
            self.b_src[:, :, :, i] = self.basis(self.estm_x, 
                                                self.estm_y,
                                                self.estm_z,
                                                [x_src, y_src, z_src],
                                                self.R)
        self.b_src = self.b_src.reshape(self.n_estm, self.n_src)
        self.k_interp_cross = np.dot(self.b_src, self.b_pot) #K_t(x,y) Eq17
        self.k_interp_cross /= self.n_src
        return self.b_src

    def update_b_interp_pot(self):
        """Compute the matrix of potentials generated by every source
        basis function at every position in the interpolated space.
        Updates b_interp_pot
        Updates k_interp_pot

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        src_loc = np.array((self.src_x.ravel(), 
                            self.src_y.ravel(), 
                            self.src_z.ravel()))
        est_loc = np.array((self.estm_x.ravel(), 
                            self.estm_y.ravel(), 
                            self.estm_z.ravel()))
        dists = distance.cdist(src_loc.T, est_loc.T,  'euclidean')
        self.b_interp_pot = self.generated_potential(dists).T
        self.k_interp_pot = np.dot(self.b_interp_pot, self.b_pot)
        self.k_interp_pot /= self.n_src
        return self.b_interp_pot

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
        src_type : basis_2D.key

        Returns
        -------
        pot : float
            value of potential at specified distance from the source
        """
        if skmonaco_available:
            pot, err = mcmiser(self.int_pot_3D_mc, 
                               npoints=1e5,
                               xl=[-R, -R, -R], 
                               xu=[R, R, R],
                               seed=42, 
                               nprocs=8, 
                               args=(x, R, h, src_type))
        else:
            pot, err = integrate.tplquad(self.int_pot_3D, 
                                         -R, 
                                         R,
                                         lambda x: -R, 
                                         lambda x: R,
                                         lambda x, y: -R, 
                                         lambda x, y: R,
                                         args=(x, R, h, src_type))
        pot *= 1./(4.0*np.pi*sigma)
        return pot

    def int_pot_3D(self, xp, yp, zp, x, R, h, basis_func):
        """FWD model function.
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)

        Parameters
        ----------
        xp, yp, zp : floats or np.arrays
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
        y = ((x-xp)**2 + yp**2 + zp**2)**0.5
        if y < 0.00001:
            y = 0.00001
        pot = 1.0/y
        pot *= basis_func(xp, yp, zp, [0, 0, 0], R)
        return pot

    def int_pot_3D_mc(self, xyz, x, R, h, basis_func):
        """
        The same as int_pot_3D, just different input: x,y,z <-- xyz (tuple)
        FWD model function, using Monte Carlo Method of integration
        Returns contribution of a point xp,yp, belonging to a basis source
        support centered at (0,0) to the potential measured at (x,0),
        integrated over xp,yp gives the potential generated by a
        basis source element centered at (0,0) at point (x,0)

        Parameters
        ----------
        xp, yp, zp : floats or np.arrays
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
        xp, yp, zp = xyz
        return self.int_pot_3D(xp, yp, zp, x, R, h, basis_func)

    def update_R(self, R):
        """Used in Cross validation

        Parameters
        ----------
        R : float

        Returns
        -------
        None
        """
        self.R = R
        Lx = np.max(self.src_x) - np.min(self.src_x) + self.R
        Ly = np.max(self.src_y) - np.min(self.src_y) + self.R
        Lz = np.max(self.src_z) - np.min(self.src_z) + self.R
        self.dist_max = np.sqrt((Lx**2 + Ly**2 + Lz**2))
        self.method()
        return

if __name__ == '__main__':
    ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                        (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),
                        (0.5, 0.5, 0.5)])
    pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])
    params = {}
    k = KCSD3D(ele_pos, pots,
               gdx=0.02, gdy=0.02, gdz=0.02,
               n_src_init=1000)
    k.cross_validate()
    #k.cross_validate(Rs=np.array(0.14).reshape(1))
    #k.cross_validate(Rs=np.array((0.01,0.02,0.04))) 
