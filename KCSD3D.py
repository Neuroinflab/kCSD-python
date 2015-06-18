import numpy as np
import utility_functions as utils
import KCSD3D_Helpers as defaults
from scipy import integrate
from scipy.spatial import distance
from KCSD2D import KCSD2D
from skmonaco import mcmiser

class KCSD3D(KCSD2D):
    def __init__(self, ele_pos, pots, src_type='gauss', params={}):
        super(KCSD3D, self).__init__(ele_pos, pots, src_type, params)

    def estimate_at(self, params):
        '''Locations where the estimation is wanted, this func must define
        self.space_X and self.space_Y
        '''
        #override defaults if params is passed
        for (prop, default) in defaults.KCSD3D_params.iteritems(): 
            setattr(self, prop, params.get(prop, default))
        #If no estimate plane given, take electrode plane as estimate plane
        xmin = params.get('xmin', np.min(self.ele_pos[:, 0]))
        xmax = params.get('xmax', np.max(self.ele_pos[:, 0]))
        ymin = params.get('ymin', np.min(self.ele_pos[:, 1]))
        ymax = params.get('ymax', np.max(self.ele_pos[:, 1]))
        zmin = params.get('zmin', np.min(self.ele_pos[:, 2]))
        zmax = params.get('zmax', np.max(self.ele_pos[:, 2]))
        #Space increment size in estimation
        gdX = params.get('gdX', 0.01 * (xmax - xmin)) 
        gdY = params.get('gdY', 0.01 * (ymax - ymin))
        gdZ = params.get('gdZ', 0.01 * (zmax - zmin))
        #Number of points where estimation is to be made.
        nx = (xmax - xmin)/gdX
        ny = (ymax - ymin)/gdY
        nz = (zmax - zmin)/gdZ
        #Making a mesh of points where estimation is to be made.
        self.space_X, self.space_Y, self.space_Z = np.mgrid[xmin:xmax:np.complex(0,nx), 
                                                            ymin:ymax:np.complex(0,ny),
                                                            zmin:zmax:np.complex(0,nz)]
        return

    def place_basis(self, source_type):
        '''Checks if a given source_type is defined, if so then defines it
        self.basis
        This function gives locations of the basis sources, and must define
        self.X_src, self.Y_src, self.R
        and
        self.dist_max '''
        #If Valid basis source type passed?
        if source_type not in defaults.basis_types.keys():
            raise Exception('Invalid source_type for basis! available are:', defaults.basis_types.keys())
        else:
            self.basis = defaults.basis_types.get(source_type)
        #Mesh where the source basis are placed is at self.X_src 
        (self.X_src, self.Y_src, self.Z_src, self.R) = defaults.make_src_3D(self.space_X,
                                                                            self.space_Y,
                                                                            self.space_Z,
                                                                            self.n_srcs_init,
                                                                            self.ext_x, 
                                                                            self.ext_y,
                                                                            self.ext_z,
                                                                            self.R_init)

        #Total diagonal distance of the area covered by the basis sources
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        Lz = np.max(self.Z_src) - np.min(self.Z_src) + self.R
        self.dist_max = (Lx**2 + Ly**2 + Lz**2)**0.5
        return        

    def values(self, estimate='CSD'):
        '''
        takes estimation_table as an input - default input is None
        if interested in csd (default), pass estimate='CSD'
        if interesting in pot pass estimate='POT'
        '''

        if estimate == 'CSD': #Maybe used for estimating the potentials also.
            estimation_table = self.k_interp_cross #pass self.interp_pot in such a case
        elif estimate == 'POT':
            estimation_table = self.k_interp_pot
        else:
            print 'Invalid quantity to be measured, pass either CSD or POT'

        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        nt = self.pots.shape[1] #Number of time points
        (nx, ny, nz) = self.space_X.shape
        estimation = np.zeros((nx * ny * nz, nt))

        for t in xrange(nt):
            beta = np.dot(k_inv, self.pots[:, t])
            for i in xrange(self.ele_pos.shape[0]):
                estimation[:, t] += beta[i] * estimation_table[:, i] # C*(x) Eq 18
        estimation = estimation.reshape(nx, ny, nz, nt)
        return estimation

    def create_lookup(self, dist_table_density=100):
        '''Updates and Returns the potentials due to a given basis source like a lookup
        table whose shape=(dist_table_density,)--> set in KCSD2D_Helpers.py

        '''
        dt_len = dist_table_density
        xs = utils.sparse_dist_table(self.R, 
                                     self.dist_max, #Find pots at sparse points
                                     dist_table_density)
        dist_table = np.zeros(len(xs))
        for i, x in enumerate(xs):
            pos = (x/dt_len) * self.dist_max
            # dist_table[i] = self.b_pot_3d_cont(pos, 
            #                                    self.R, 
            #                                    self.h, 
            #                                    self.sigma,
            #                                    self.basis)
            dist_table[i] = self.b_pot_3d_mc(pos, 
                                             self.R, 
                                             self.h, 
                                             self.sigma,
                                             self.basis)

        self.dist_table = utils.interpolate_dist_table(xs, dist_table, dt_len) #and then interpolated
        return self.dist_table #basis potentials in a look up table

    def update_b_pot(self):
        """
        Updates the b_pot  - array is (#_basis_sources, #_electrodes)
        Update  k_pot -- K(x,x') Eq9,Jan2012
        Calculates b_pot - matrix containing the values of all
        the potential basis functions in all the electrode positions
        (essential for calculating the cross_matrix).
        """
        src = np.array((self.X_src.ravel(), self.Y_src.ravel(), self.Z_src.ravel()))
        dists = distance.cdist(src.T, self.ele_pos, 'euclidean')
        self.b_pot = self.generated_potential(dists)
        self.k_pot = np.dot(self.b_pot.T, self.b_pot) #K(x,x') Eq9,Jan2012
        return self.b_pot

    def update_b_src(self):
        """
        Updates the b_src in the shape of (#_est_pts, #_basis_sources)
        Updates the k_interp_cross - K_t(x,y) Eq17
        Calculate b_src - matrix containing containing the values of
        all the source basis functions in all the points at which we want to
        calculate the solution (essential for calculating the cross_matrix)
        """
        (nsx, nsy, nsz) = self.X_src.shape #These should go elsewhere!
        n = nsz * nsy * nsx  # total number of sources
        (ngx, ngy, ngz) = self.space_X.shape
        ng = ngx * ngy *ngz

        self.b_src = np.zeros((ngx, ngy, ngz, n))
        for i in xrange(n):
            # getting the coordinates of the i-th source
            (i_x, i_y, i_z) = np.unravel_index(i, (nsx, nsy, ngz), order='F')
            x_src = self.X_src[i_x, i_y, i_z]
            y_src = self.Y_src[i_x, i_y, i_z]
            z_src = self.Z_src[i_x, i_y, i_z]
            self.b_src[:, :, :, i] = self.basis(self.space_X, 
                                                self.space_Y,
                                                self.space_Z,
                                                [x_src, y_src, z_src],
                                                self.R)

        self.b_src = self.b_src.reshape(ng, n)
        self.k_interp_cross = np.dot(self.b_src, self.b_pot) #K_t(x,y) Eq17
        return self.b_src

    def update_b_interp_pot(self):
        """
        Compute the matrix of potentials generated by every source
        basis function at every position in the interpolated space.
        Updates b_interp_pot
        Updates k_interp_pot
        """
        src = np.array((self.X_src.ravel(), 
                        self.Y_src.ravel(), 
                        self.Z_src.ravel()))
        est_loc = np.array((self.space_X.ravel(), 
                            self.space_Y.ravel(), 
                            self.space_Z.ravel()))
        dists = distance.cdist(src.T, est_loc.T,  'euclidean')
        self.b_interp_pot = self.generated_potential(dists).T
        self.k_interp_pot = np.dot(self.b_interp_pot, self.b_pot)
        return self.b_interp_pot

    def b_pot_3d_cont(self, x, R, h, sigma, src_type):
        """
        Returns the value of the potential at point (x,y,0) generated
        by a basis source located at (0,0,0)
        """
        pot, err = integrate.tplquad(defaults.int_pot_3D, 
                                     -R, 
                                     R,
                                     lambda x: -R, 
                                     lambda x: R,
                                     lambda x, y: -R, 
                                     lambda x, y: R,
                                     args=(x, R, h, src_type))
        pot *= 1./(2.0*np.pi*sigma)
        return pot

    def b_pot_3d_mc(self, x, R, h, sigma, src_type):
        """
        Calculate potential in the 3D case using Monte Carlo integration.
        It utilizes the MISER algorithm
        """
        pot, err = mcmiser(defaults.int_pot_3D_mc, 
                           npoints=1e5,
                           xl=[-R, -R, -R], 
                           xu=[R, R, R],
                           nprocs=8, 
                           args=(x, R, h, src_type))
        pot *= 1./(2.0*np.pi*sigma)
        return pot

    def update_R(self, R):
        '''Useful for Cross validation'''
        self.R = R
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        Lz = np.max(self.Z_src) - np.min(self.Z_src) + self.R
        self.dist_max = (Lx**2 + Ly**2 + Lz**2)**0.5
        self.method()
        return

if __name__ == '__main__':
    ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                        (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),
                        (0.5, 0.5, 0.5)])
    pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])

    params = {'gdX': 0.02, 'gdY': 0.02, 'gdZ': 0.02, 'n_srcs_init': 1000}
    
    k = KCSD3D(ele_pos, pots, params=params)
    #print k.values()
    #k.cross_validate()
    print k.cross_validate(Rs=np.array((0.01,0.02,0.04))) 
