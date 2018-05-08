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
from collections import Counter

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

try:
    from skmonaco import mcmiser
    skmonaco_available = True
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
except ImportError:
    skmonaco_available = False

from corelib.KCSD import KCSD1D
import corelib.utility_functions as utils
import corelib.basis_functions as basis
from corelib.bresenhamline import bresenhamline


class sKCSDcell(object):
    """
    sKCSDcell -- construction of the morphology loop for sKCSD method
    (Cserpan et al., 2017).

    This calculates the morphology loop and helps transform
    CSDestimates/potential estimates from loop/segment space to 3D.
    The method implented here is based on the original paper
    by Dorottya Cserpan et al., 2017.
    """
    def __init__(self, morphology, ele_pos, n_src, tolerance=2e-6):
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
        self.morphology = morphology  # morphology file
        self.ele_pos = ele_pos  # electrode_positions
        self.n_src = n_src  # number of sources
        self.max_dist = 0  # maximum distance
        self.segments = {}  # segment dictionary with loops as keys
        self.segment_counter = 0  # which segment we're on
        rep = Counter(self.morphology[:, 6])
        self.branching = [int(key) for key in rep.keys() if rep[key] > 1]
        self.morphology_loop()  # make the morphology loop
        self.source_pos = np.zeros((n_src, 1))
        # positions of sources on the morphology (1D),
        # necessary for source division
        self.source_pos[:, 0] = np.linspace(0, self.max_dist, n_src)
        # Cartesian coordinates of the sources
        self.source_xyz = np.zeros(shape=(n_src, 3))
        self.tolerance = tolerance  # smallest dendrite used for visualisation
        # max and min points of the neuron's morphology
        self.xmin = np.min(self.morphology[:, 2])
        self.xmax = np.max(self.morphology[:, 2])
        self.ymin = np.min(self.morphology[:, 3])
        self.ymax = np.max(self.morphology[:, 3])
        self.zmin = np.min(self.morphology[:, 4])
        self.zmax = np.max(self.morphology[:, 4])
        self.dxs = self.get_dxs()
        self.dims = self.get_grid()

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
        key1 = "%d_%d" % (mp1, mp2)
        key2 = "%d_%d" % (mp2, mp1)
        if key1 not in self.segments:
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
        self.add_segment(mp1, mp2)
        xyz1 = self.morphology[mp1, 2:5]
        xyz2 = self.morphology[mp2, 2:5]
        self.loops.append([mp2, mp1])
        self.max_dist += np.linalg.norm(xyz1 - xyz2)

    def morphology_loop(self):
        """Cover the morphology of the cell with loops.

        Parameters
        ----------
        None
        """
        # loop over morphology
        self.loops = []
        for morph_pnt in range(1, self.morphology.shape[0]):
            if self.morphology[morph_pnt-1, 0] == self.morphology[morph_pnt,
                                                                  6]:
                self.add_loop(morph_pnt, morph_pnt-1)
            elif self.morphology[morph_pnt, 6] in self.branching:
                last_branch = int(self.morphology[morph_pnt, 6])-1
                last_point = morph_pnt - 1
                while True:
                    parent = int(self.morphology[last_point, 6]) - 1
                    self.add_loop(parent, last_point)
                    if parent == last_branch:
                        break
                    last_point = parent
                self.add_loop(morph_pnt, int(self.morphology[morph_pnt,
                                                             6]) - 1)

        last_point = morph_pnt
        while True:
            parent = int(self.morphology[last_point, 6]) - 1
            self.add_loop(parent, last_point)
            if int(self.morphology[parent, 6]) == -1:
                break
            last_point = parent
        # find estimation points
        self.loops = np.array(self.loops)
        self.est_pos = np.zeros((len(self.loops)+1, 1))
        self.est_xyz = np.zeros((len(self.loops)+1, 3))
        self.est_xyz[0, :] = self.morphology[0, 2:5]
        for i, loop in enumerate(self.loops):
            length = 0
            for j in [2, 3, 4]:
                length += (
                    self.morphology[loop[1]][j] - self.morphology[loop[0]][j]
                )**2
            self.est_pos[i+1] = self.est_pos[i] + length**0.5
            self.est_xyz[i+1, :] = self.morphology[loop[1], 2:5]

    def distribute_srcs_3D_morph(self):
        """
        Calculate 3D coordinates of sources placed on the morphology loop.

        Parameters
        ----------
        None
        """
        for i, x in enumerate(self.source_pos):
            self.source_xyz[i] = self.get_xyz(x)
        return self.source_pos

    def get_xyz(self, x):
        """Find cartesian coordinates of a point (x) on the morphology loop.
        Use morphology point cartesian coordinates (from the morphology file,
        self.est_xyz) for interpolation.

        Parameters
        ----------
        x : float

        Returns
        -------
        tuple of length 3
        """
        return interpolate.interp1d(self.est_pos[:, 0], self.est_xyz,
                                    kind='linear', axis=0)(x)

    def calculate_total_distance(self):
        """
        Calculates doubled total legth of the cell.

        Parameteres
        -----------
        None
        """
        total_dist = 0
        for i in range(1, self.morphology.shape[0]):
            xyz1 = self.morphology[i, 2:5]
            xyz2 = self.morphology[int(self.morphology[i, 6]) - 1, 2:5]
            total_dist += np.linalg.norm(xyz2 - xyz1)
        total_dist *= 2
        return total_dist

    def points_in_between(self, p1, p0, last):
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
        # bresenhamline only works with 2D vectors with coordinates
        new_p1 = np.ndarray((1, 3), dtype=np.int)
        new_p0 = np.ndarray((1, 3), dtype=np.int)
        for i in range(3):
            new_p1[0, i] = p1[i]
            new_p0[0, i] = p0[i]
        intermediate_points = bresenhamline(new_p0, new_p1, -1)
        if last:
            return np.concatenate((new_p0, intermediate_points))
        else:
            return intermediate_points

    def get_dxs(self):
        """Calculate parameters of the 3D grid used to transform CSD
        (or potential) according to eq. (22). self.tolerance is used
        to specify smalles possible size of neurite.

        Parameters
        ----------
        None
        Returns
        -------
        dxs: np.array of 3 floats

        """
        dxs = np.zeros((3, ))
        for i in range(self.est_xyz.shape[1]):
            dx = abs(self.est_xyz[1:, i] - self.est_xyz[:-1, i])
            try:
                dxs[i] = min(dx[dx > self.tolerance])
            except ValueError:
                pass
        return dxs

    def get_grid(self):
        """Calculate size of the 3D grid used to transform CSD
        (or potential) according to eq. (22). self.tolerance is used
        to specify smalles possible size of neurite.

        Parameters
        ----------
        None

        Returns
        -------

        dims: np.array of 3 ints
        CSD/potential array 3D coordinates
        """
        vals = [
            [self.xmin, self.xmax],
            [self.ymin, self.ymax],
            [self.zmin, self.zmax]
        ]
        dims = np.ones((3, ), dtype=np.int)

        for i, dx in enumerate(self.dxs):
            dims[i] = 1
            if dx:
                dims[i] += np.floor((vals[i][1] - vals[i][0])/dx)
        return dims

    def point_coordinates(self, morpho):
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
        minis = np.array([self.xmin, self.ymin, self.zmin])
        zero_coords = np.zeros((3, ), dtype=int)
        coor_3D = np.zeros((morpho.shape[0]-1, morpho.shape[1]), dtype=np.int)
        for i, dx in enumerate(self.dxs):
            if dx:
                coor_3D[:, i] = np.floor((morpho[1:, i] - minis[i])/dx)
                zero_coords[i] = np.floor((morpho[0, i] - minis[i])/dx)
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
            last = (i+1 == len(coor_3D))
            segment_coordinates[i] = self.points_in_between(p0, p1, last)
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
        coor_3D, p0 = self.point_coordinates(self.morphology[:, 2:5])
        segment_coordinates = {}

        parentage = self.morphology[1:, 6]-2
        i = 0
        p1 = coor_3D[0]

        while True:
            last = (i+1 == len(coor_3D))
            segment_coordinates[i] = self.points_in_between(p0, p1, last)
            if i+1 == len(coor_3D):
                break
            if i:
                p0_idx = int(parentage[i + 1])
                p0 = coor_3D[p0_idx]
            else:
                p0 = p1
            i = i + 1
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

        if what == "loop":
            coor_3D = self.coordinates_3D_loops()
        elif what == "morpho":
            coor_3D = self.coordinates_3D_segments()
        else:
            sys.exit('Do not understand morphology %s\n' % what)

        n_time = estimated.shape[-1]
        new_dims = list(self.dims)+[n_time]
        result = np.zeros(new_dims)

        for i in coor_3D:

            coor = coor_3D[i]
            for p in coor:
                x, y, z, = p
                result[x, y, z, :] += estimated[i, :]
        return result

    def transform_to_segments(self, estimated):
        """
        Transform potential/csd/ground truth values in loop space
        to segment space.

        Parameters
        ----------
        estimated : np.array

        Returns
        -------
        result : np.array
        """
        result = np.zeros((self.morphology.shape[0]-1, estimated.shape[1]))
        for i, loop in enumerate(self.loops):
            key = "%d_%d" % (loop[0], loop[1])
            seg_no = self.segments[key]

            result[seg_no, :] += estimated[i, :]

        return result

    def draw_cell2D(self, axis=2):
        """
        Cell morphology in 3D grid in projection of axis.

        Parameters
        ----------
        axis : int
          0: x axis, 1: y axis, 2: z axis
        """
        resolution = self.dims
        xgrid = np.linspace(self.xmin, self.xmax, resolution[0])
        ygrid = np.linspace(self.ymin, self.ymax, resolution[1])
        zgrid = np.linspace(self.zmin, self.zmax, resolution[2])

        if axis == 0:
            image = np.ones(shape=(resolution[1],
                                   resolution[2], 4), dtype=np.uint8) * 255
            extent = [1e6*self.zmin, 1e6*self.zmax,
                      1e6*self.ymin, 1e6*self.ymax]
        elif axis == 1:
            image = np.ones(shape=(resolution[0],
                                   resolution[2], 4), dtype=np.uint8) * 255
            extent = [1e6*self.zmin, 1e6*self.zmax,
                      1e6*self.xmin, 1e6*self.xmax]
        elif axis == 2:
            image = np.ones(shape=(resolution[0],
                                   resolution[1], 4), dtype=np.uint8) * 255
            extent = [1e6*self.ymin, 1e6*self.ymax,
                      1e6*self.xmin, 1e6*self.xmax]
        else:
            sys.exit('In drawing 2D morphology unknown axis %d' % axis)
        image[:, :, 3] = 0
        xs = []
        ys = []
        x0, y0 = 0, 0
        for p in self.source_xyz:
            x = (np.abs(xgrid-p[0])).argmin()
            y = (np.abs(ygrid-p[1])).argmin()
            z = (np.abs(zgrid-p[2])).argmin()
            if axis == 0:
                xi, yi = y, z
            elif axis == 1:
                xi, yi = x, z
            elif axis == 2:
                xi, yi = x, y
            xs.append(xi)
            ys.append(yi)
            image[xi, yi, :] = np.array([0, 0, 0, 1])
            if x0 != 0:
                idx_arr = self.points_in_between([xi, yi, 0], [x0, y0, 0], 1)
                for i in range(len(idx_arr)):
                    image[idx_arr[i, 0] - 1:idx_arr[i, 0] + 1,
                          idx_arr[i, 1] - 1:idx_arr[i, 1] + 1, :] = np.array(
                              [0, 0, 0, 20])
            x0, y0 = xi, yi
        return image, extent


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
                minimum neurite size used for 3D tranformation of CSD
                and potential
                Defaults to 2 um

        Returns
        -------
        None

        Raises
        ------
        LinAlgError
            Could not invert the matrix,
            try changing the ele_pos slightly
        KeyError
            Basis function (src_type) not implemented.
            See basis_functions.py for available
        """
        self.morphology = morphology
        super(KCSD1D, self).__init__(ele_pos, pots, **kwargs)

    def parameters(self, **kwargs):
        self.src_type = kwargs.pop('src_type', 'gauss')
        self.sigma = kwargs.pop('sigma', 1.0)
        self.h = kwargs.pop('h', 1e-5)
        self.n_src_init = kwargs.pop('n_src_init', 1000)
        self.lambd = kwargs.pop('lambd', 1e-4)
        self.R_init = kwargs.pop('R_init', 2.3e-5)  # microns
        self.dist_table_density = kwargs.pop('dist_table_density', 100)
        self.dim = 'skCSD'
        self.tolerance = kwargs.pop('tolerance', 2e-06)

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
        self.cell = sKCSDcell(self.morphology, self.ele_pos,
                              self.n_src_init, self.tolerance)
        self.n_estm = len(self.cell.est_pos)

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
        self.R = self.R_init
        source_type = self.src_type
        try:
            self.basis = basis.basis_1D[source_type]
        except:
            print('Invalid source_type for basis! available are:',
                  basis.basis_1D.keys())
            raise KeyError
        self.src_x = self.cell.distribute_srcs_3D_morph()
        self.n_src = self.cell.n_src

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
        self.dist_max = max(np.max(self.src_ele_dists),
                            np.max(self.src_estm_dists)) + self.R

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
                               xl=[-2**1.5*R],
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
        estimation = np.zeros((self.n_ele, self.n_time))
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
        estimated = super(sKCSD, self).values(estimate=estimate)
        if not transformation:
            return estimated
        elif transformation == 'segments':
            return self.cell.transform_to_segments(estimated)
        elif transformation == '3D':
            return self.cell.transform_to_3D(estimated, what="loop")

        raise Exception("Unknown transformation %s of %s" %
                        (transformation, estimate))

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
            xp = xp + self.cell.max_dist
        xp_coor = self.cell.get_xyz(xp)
        dist = ((x - xp_coor[0])**2 + xp_coor[1]**2 + xp_coor[2]**2)**0.5
        if dist < 0.00001:
            dist = 0.00001
        pot = basis_func(xp, R)/dist  # xp is the distance
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
    md = os.path.join(uwd,
                      'tests/Data/ball_and_stick_8/morphology/Figure_2_rows_8.swc')
    lfpd = os.path.join(uwd,
                        'tests/Data/ball_and_stick_8/LFP/MyLFP')
    eleposd = os.path.join(uwd,
                           'tests/Data/ball_and_stick_8/electrode_positions/elcoord_x_y_x')
    parser = argparse.ArgumentParser(
        description='Calculate current/potential estimation using sKCSD')
    parser.add_argument('--morphology',
                        type=str,
                        metavar='morphology',
                        default=md,
                        help='path to neuron morphology in swc file format')
    parser.add_argument('--LFP',
                        type=str,
                        metavar='LFP',
                        default=lfpd,
                        help='path to LFP measurements')
    parser.add_argument('--electrode_positions',
                        type=str,
                        metavar='electrode_positions',
                        default=eleposd,
                        help='path to electrode positions')
    parser.add_argument('--save_to',
                        type=str,
                        metavar='save_to',
                        default=os.path.join(uwd,
                                             'tests/Data/ball_and_stick_8'),
                        help='path to results')
    parser.add_argument('--src_type',
                        choices=set(('gauss', 'step', 'gauss_lim')),
                        default='gauss',
                        help='basis function type')
    parser.add_argument('--R_init',
                        type=float,
                        default=23e-6,
                        help='width of basis function')
    parser.add_argument('--lambd',
                        type=float,
                        default=1e-1,
                        help='regularization parameter for ridge regression')
    parser.add_argument('--n_src',
                        type=int,
                        default=300,
                        help='requested number of sources')
    parser.add_argument('--sigma',
                        type=float,
                        default=1,
                        help='space conductance of the tissue in S/m')
    parser.add_argument('--dist_table_density',
                        type=int,
                        default=100,
                        help='size of the potential interpolation table')
    args = parser.parse_args()

    morphology = utils.load_swc(args.morphology)
    myLFP = np.loadtxt(args.LFP)
    electrode_positions = utils.load_elpos(args.electrode_positions)
    scaling_factor = 1000000
    ele_pos = electrode_positions/scaling_factor
    morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
    R_init = 32/scaling_factor

    k = sKCSD(ele_pos, myLFP, morphology,
              n_src_init=args.n_src, src_type=args.src_type,
              R_init=args.R_init, lambd=args.lambd, sigma=args.sigma,
              dist_table_density=args.dist_table_density)
    ker_dir = args.save_to
    if sys.version_info < (3, 0):
        path = os.path.join(ker_dir, "preprocessed_data/Python_2")
    else:
        path = os.path.join(ker_dir, "preprocessed_data/Python_3")

    if not os.path.exists(path):
        print("Creating", path)
        os.makedirs(path)
    utils.save_sim(path, k)
