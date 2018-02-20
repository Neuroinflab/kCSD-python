"""
@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import time

from builtins import int, range
from past.utils import old_div

import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.mlab import griddata
from matplotlib import colors, gridspec
from scipy.integrate import simps
import csd_profile as CSD
sys.path.append('../corelib')
from KCSD import KCSD1D, KCSD2D, KCSD3D


try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 1
    parallel_available = True
except ImportError:
    parallel_available = False


class ValidationClassKCSD(object):
    """
    Base class for validation of the kCSD method
    """
    def __init__(self, dim, **kwargs):
        """Initialize TestKCSD class

        Parameters
        ----------
        dim: int
            case dimention (1, 2 or 3 D)
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the medium
                Defaults to 0.3
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                thickness of analyzed cylindrical slice
                Defaults to 1.
            nr_basis: int
                number of basis sources
                Defaults to 300
            total_ele: int
                number of electrodes
                Defaults to 10
            ele_seed: int
                seed for random electrodes position placement
                Defaults to 10
            ext_x : float
                length of space extension: x_min-ext_x ... x_max+ext_x
                Defaults to 0.
            est_xres: int
                resolution of kcsd estimation
                Defaults to 100
            true_csd_xlims: list
                boundaries for ground truth space
                Defaults to [0., 1.]
            ele_xlims: list
                boundaries for electrodes placement
                Defaults to [0., 1.]
            csd_xres: int
                resolution of ground truth
                Defaults to 100

        Returns
        -------
        None
        """
        self.dim = dim
        self.general_parameters(**kwargs)
        self.dimension_parameters(**kwargs)
        return

    def general_parameters(self, **kwargs):
        """
        Defining the default values of the method passed as kwargs

        Parameters
        ----------
        **kwargs
            Same as those passed to initialize the Class

        Returns
        -------
        None
        """
        self.src_type = kwargs.get('src_type', 'gauss')
        self.sigma = kwargs.get('sigma', 0.3)
        self.h = kwargs.get('h', 0.25)
        self.R_init = kwargs.get('R_init', 0.23)
        self.n_src_init = kwargs.get('n_src_init', 300)
        self.total_ele = kwargs.get('total_ele', 10)
        self.config = kwargs.get('ele_config', 'regular')
        self.mask = kwargs.get('mask', False)
        return

    def dimension_parameters(self, **kwargs):
        """
        Defining the default values of the method passed as kwargs

        Parameters
        ----------
        dim: int
            dimention of analyzed case (1, 2 or 3D)
        **kwargs
            Same as those passed to initialize the Class

        Returns
        -------
        None
        """
        self.ext_x = kwargs.get('ext_x', 0.0)
        self.est_xres = kwargs.get('est_xres', 100)
        self.true_csd_xlims = kwargs.get('true_csd_xlims', [0., 1.])
        self.ele_xlims = kwargs.get('ele_xlims', [0.1, 0.9])
        self.ele_xres = kwargs.get('ele_xres', (self.total_ele))
        self.csd_xres = kwargs.get('csd_xres', 100)
        if self.dim >= 2:
            self.ext_y = kwargs.get('ext_y', 0.0)
            self.ele_ylims = kwargs.get('ele_ylims', [0.1, 0.9])
            self.basis_ylims = kwargs.get('basis_ylims', [0., 1.])
            self.kcsd_ylims = kwargs.get('kcsd_ylims', [0., 1.])
            self.true_csd_ylims = kwargs.get('true_csd_ylims', [0., 1.])
            self.est_yres = kwargs.get('est_yres', 100)
            self.ele_yres = kwargs.get('ele_yres',
                                       int(np.sqrt(self.total_ele)))
            self.csd_yres = kwargs.get('csd_yres', 100)
        if self.dim == 3:
            self.ext_z = kwargs.get('ext_z', 0.0)
            self.ele_zlims = kwargs.get('ele_zlims', [0.1, 0.9])
            self.basis_zlims = kwargs.get('basis_zlims', [0., 1.])
            self.kcsd_zlims = kwargs.get('kcsd_zlims', [0., 1.])
            self.true_csd_zlims = kwargs.get('true_csd_zlims', [0., 1.])
            self.est_zres = kwargs.get('est_zres', 100)
            self.ele_zres = kwargs.get('ele_zres',
                                       int(np.cbrt(self.total_ele)))
            self.csd_zres = kwargs.get('csd_zres', 100)
        return

    def broken_electrodes(self, ele_seed, n):
        """
        Produces electrodes positions for setup with n pseudo randomly
        (ele_seed) removed electrodes

        Parameters
        ----------
        ele_seed: int
            internal state of the random number generator
        n: int
            number of broken/missing electrodes

        Returns
        -------
        ele_pos: numpy array
            electrodes positions
        """
        if self.dim == 1:
            ele_grid = self.generate_electrodes()
        elif self.dim == 2:
            ele_x, ele_y = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                    np.complex(0, self.ele_yres),
                                    self.ele_ylims[0]:self.ele_ylims[1]:
                                    np.complex(0, self.ele_yres)]
            ele_x, ele_y = ele_x.flatten(), ele_y.flatten()
            ele_grid = np.vstack((ele_x, ele_y)).T
        else:
            ele_x, ele_y, ele_z = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                           np.complex(0, self.ele_xres),
                                           self.ele_ylims[0]:self.ele_ylims[1]:
                                           np.complex(0, self.ele_yres),
                                           self.ele_xlims[0]:self.ele_xlims[1]:
                                           np.complex(0, self.ele_zres)]
            ele_x, ele_y, ele_z = ele_x.flatten(), ele_y.flatten(), \
                ele_z.flatten()
            ele_grid = np.vstack((ele_x, ele_y, ele_z)).T
        random_indices = np.arange(0, ele_grid.shape[0])
        np.random.seed(ele_seed)
        np.random.shuffle(random_indices)
        ele_pos = ele_grid[random_indices[:self.total_ele - n]]
        return ele_pos

    def generate_electrodes(self):
        """
        Places electrodes linearly.

        Parameters
        ----------
        None

        Returns
        -------
        Linearly placed electrodes positions.
        (for 1D case: ele_x, 2D case: ele_x, ele_y, and
        3D case: ele_x, ele_y, ele_z)
        """
        if self.dim == 1:
            ele_x = np.linspace(self.ele_xlims[0], self.ele_xlims[1],
                                self.total_ele)
            return ele_x
        elif self.dim == 2:
            ele_x, ele_y = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                    np.complex(0, self.ele_yres),
                                    self.ele_ylims[0]:self.ele_ylims[1]:
                                    np.complex(0, self.ele_yres)]
            return ele_x.flatten(), ele_y.flatten()
        elif self.dim == 3:
            ele_x, ele_y, ele_z = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                           np.complex(0, self.ele_zres),
                                           self.ele_ylims[0]:self.ele_ylims[1]:
                                           np.complex(0, self.ele_zres),
                                           self.ele_zlims[0]:self.ele_zlims[1]:
                                           np.complex(0, self.ele_zres)]
            return ele_x.flatten(), ele_y.flatten(), ele_z.flatten()

    def electrode_config(self, csd_profile, noise=None, nr_broken_ele=0):
        """
        Produces electrodes positions and calculates potentials measured at
        these points

        Parameters
        ----------
        csd_profile: function
            Function to produce ground truth csd profile.
        noise: str, optional
            Determins if data contains noise. For noise='noise' white Gaussian
            noise is added to potentials.
            Default: None
        nr_broken_ele: int, optional
            Determines how many electrodes are broken.
            Default: 0.

        Returns
        -------
        ele_pos: numpy array
            Electrodes locations in 1D, 2D or 3D.
        pots: numpy array
            Potentials measured (calculated) on electrodes.
        """
        if self.dim == 1:
            csd_at, true_csd = self.generate_csd(csd_profile)
            if self.config == 'broken':
                ele_pos = self.broken_electrodes(10, nr_broken_ele)
            else:
                ele_pos = self.generate_electrodes()
            pots = self.calculate_potential(true_csd, csd_at, ele_pos)
            ele_pos = ele_pos.reshape((len(ele_pos), 1))
        if self.dim == 2:
            csd_at, true_csd = self.generate_csd(csd_profile)
            if self.config == 'broken':
                ele_pos = self.broken_electrodes(10, nr_broken_ele)
                ele_x, ele_y = ele_pos[:, 0], ele_pos[:, 1]
            else:
                ele_x, ele_y = self.generate_electrodes()
            pots = self.calculate_potential(true_csd, csd_at,
                                            ele_x, ele_y)
            ele_pos = np.vstack((ele_x, ele_y)).T
            pots = pots.reshape((len(pots), 1))
        if self.dim == 3:
            csd_at, true_csd = self.generate_csd(csd_profile)
            if self.config == 'broken':
                ele_pos = self.broken_electrodes(10, nr_broken_ele)
                ele_x, ele_y, ele_z = ele_pos[:, 0], ele_pos[:, 1], \
                    ele_pos[:, 2]
            else:
                ele_x, ele_y, ele_z = self.generate_electrodes()
            if parallel_available:
                pots = self.calculate_potential_parallel(true_csd,
                                                         ele_x, ele_y, ele_z,
                                                         csd_at)
            else:
                pots = self.calculate_potential(true_csd, ele_x, ele_y, ele_z,
                                                csd_at)
            ele_pos = np.vstack((ele_x, ele_y, ele_z)).T
        num_ele = ele_pos.shape[0]
        print('Number of electrodes:', num_ele)
        if noise == 'noise':
            pots = self.add_noise(pots)
        return ele_pos, pots.reshape((len(ele_pos), 1))

    def do_kcsd(self, ele_pos, pots, k, Rs=None, lambdas=None):
        """
        Estimates csd with kCSD method.

        Parameters
        ----------
        ele_pos: numpy array
            Electrodes positions.
        pots: numpy array
            Values of potentials at ele_pos.
        k: instance of the class
            Instance of KCSD1D, KCSD2D or KCSD3D class.
        Rs: numpy 1D array, optional
            Array of different values of basis parameter R used for
            cross validation.
            Default: None.
        lambdas: numpy 1D array, optional
            Regularization parameter.
            Default: None.

        Returns
        -------
        est_csd: numpy array
            Estimated csd (with kCSD method).
        est_pot: numpy array
            Estimated potentials.
        """
        k.cross_validate(Rs=Rs, lambdas=lambdas)
        est_csd = k.values('CSD')
        est_pot = k.values('POT')
        return est_csd, est_pot

    def calculate_rms(self, test_csd, est_csd):
        """
        Calculates normalized error of reconstruction.

        Parameters
        ----------
        test_csd: numpy array
            Values of true CSD at points of kCSD estimation.
        est_csd: numpy array
            CSD estimated with kCSD method.

        Returns
        -------
        rms: float
            Normalized error of reconstruction.
        """
        rms = np.linalg.norm((test_csd - est_csd))
        epsilon = 0.0000000001
        rms /= np.linalg.norm(test_csd) + epsilon
        return rms

    def calculate_point_error(self, test_csd, est_csd):
        """
        Calculates normalized error of reconstruction at every point of
        estimation space separetly.

        Parameters
        ----------
        test_csd: numpy array
            Values of true csd at points of kCSD estimation.
        est_csd: numpy array
            CSD estimated with kCSD method.

        Returns
        -------
        point_error: numpy array
            Normalized error of reconstruction calculated separetly at every
            point of estimation space.
        """
        epsilon = 0.0000000001
        point_error = np.linalg.norm(test_csd.reshape(test_csd.size, 1) -
                                     est_csd.reshape(est_csd.size, 1),
                                     axis=1)
        point_error /= np.linalg.norm(test_csd.reshape(test_csd.size, 1),
                                      axis=1) + epsilon
        if self.dim != 1:
            point_error = point_error.reshape(test_csd.shape)
        return point_error

    def sigmoid_mean(self, error):
        '''
        Calculates sigmoidal mean across errors of reconstruction for many
        different sources - used for error maps

        Parameters
        ----------
        error: numpy array
            Normalized point error of reconstruction.

        Returns
        -------
        error_mean: numpy array
            Sigmoidal mean error of reconstruction.
            error_mean -> 1    - very poor reconstruction
            error_mean -> 0    - perfect reconstruction
        '''
        sig_error = 2*(1./(1 + np.exp((-error))) - 1/2.)
        error_mean = np.mean(sig_error, axis=0)
        return error_mean

    def add_noise(self, pots, level=0.001):
        """
        Adds Gaussian noise to potentials.

        Parameters
        ----------
        pots: numpy array
            Potentials at measurement points.
        level: float, optional
            Noise level. Default: 0.001.

        Returns
        -------
        pots_noise: numpy array
            Potentials with added random Gaussian noise.
        """
        rstate = np.random.RandomState(self.csd_seed)
        noise = level*rstate.normal(np.mean(pots), np.std(pots), len(pots))
        pots_noise = pots + noise.reshape(len(noise), 1)
        return pots_noise


class SpectralStructure(object):
    """
    Class that enables examination of spectral structure of CSD reconstruction
    with kCSD method.
    """
    def __init__(self, k):
        """
        Initialize SpectralStructure class

        Parameters
        ----------
        k: instance of the class
            Instance of ValidationClassKCSD1D, ValidationClassKCSD2D or
            ValidationClassKCSD3D class.
        Defines:
            self.k: instance of the class
            Instance of ValidationClassKCSD1D, ValidationClassKCSD2D or
            ValidationClassKCSD3D class.

        Returns
        -------
        None
        """
        self.k = k
        return

    def svd(self):
        """
        Method that calculates singular value decomposition of total kernel
        matrix. K~*K^-1  from eq. 18 (Potworowski 2012)

        Parameters
        ----------
        None

        Returns
        -------
        u_svd: numpy array
            Left singular vectors of kernels product.
        sigma: numpy array
            Singular values of kernels product.
        v_svd: numpy array
            Right singular vectors of kernels product.
        """
        kernel = np.dot(self.k.k_interp_cross,
                        inv(self.k.k_pot +
                            self.k.lambd * np.identity(self.k.k_pot.shape[0])))
        u_svd, sigma, v_svd = np.linalg.svd(kernel, full_matrices=False)
        self.plot_svd_sigma(sigma)
        self.plot_svd_u(u_svd)
        self.plot_svd_v(v_svd)
        return u_svd, sigma, v_svd

    def picard_plot(self, b):
        """
        Creates Picard plot according to Hansen's book.

        Parameters
        ----------
        b: numpy array
            Right-hand side of the linear equation.

        Returns
        -------
        None
        """
        u, s, v = np.linalg.svd(self.k.k_pot + self.k.lambd *
                                np.identity(self.k.k_pot.shape[0]))
        picard = np.zeros(len(s))
        picard_norm = np.zeros(len(s))
        for i in range(len(s)):
            picard[i] = abs(np.dot(u[:, i].T, b))
            picard_norm[i] = abs(np.dot(u[:, i].T, b))/s[i]
        plt.figure(figsize=(10, 6))
        plt.plot(s, marker='.', label=r'$\sigma_{i}$')
        plt.plot(picard, marker='.', label='$|u(:, i)^{T}*b|$')
        plt.plot(picard_norm, marker='.',
                 label=r'$\frac{|u(:, i)^{T}*b|}{\sigma_{i}}$')
        plt.yscale('log')
        plt.legend()
        plt.title('Picard plot')
        plt.xlabel('i')
        plt.show()
        a = int(len(s) - int(np.sqrt(len(s)))**2)
        if a == 0:
            size = int(np.sqrt(len(s)))
        else:
            size = int(np.sqrt(len(s))) + 1
        fig, axs = plt.subplots(int(np.sqrt(len(s))),
                                size, figsize=(15, 13))
        axs = axs.ravel()
        beta = np.zeros(v.shape)
        fig.suptitle('vectors products of k_pot matrix')
        for i in range(len(s)):
            beta[i] = ((np.dot(u[:, i].T, b)/s[i]) * v[i, :])
            axs[i].plot(beta[i, :], marker='.')
            axs[i].set_title(r'$vec_{'+str(i+1)+'}$')
        plt.show()
        return

    def plot_evd_sigma(self, s):
        """
        Creates plot of eigenvalues of k_pot matrix.

        Parameters
        ----------
        s: numpy array
            Eigenvalues of k_pot matrix

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(s, '.')
        plt.title('Eigenvalues of k_pot matrix')
        plt.xlabel('Components number')
        plt.ylabel('Eigenvalues')
        plt.yscale('log')
        plt.show()
        return

    def evd(self):
        """
        Method that calculates eigenvalue decomposition of kernel (k_pot
        matrix)

        Parameters
        ----------
        k: instance of class (TestKCSD1D, TestKCSD2D or TestKCSD3D)

        Returns
        -------
        eigenvectors: numpy array
            Eigen vectors of k_pot matrix.
        eigrnvalues: numpy array
            Eigenvalues of k_pot matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.k.k_pot +
                                                   self.k.lambd * np.identity
                                                   (self.k.k_pot.shape[0]))
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.plot_evd(eigenvalues)
        return eigenvectors, eigenvalues

    def plot_svd_sigma(self, sigma):
        """
        Creates plot of singular values of kernels product (K~*K^-1)

        Parameters
        ----------
        sigma: numpy array
            Singular values of kernels product

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(sigma, 'b.')
        plt.title('Singular values of kernels product')
        plt.xlabel('Components number')
        plt.ylabel('Singular values')
        plt.yscale('log')
        plt.show()
        return

    def plot_u(self, u):
        """
        Creates plot of left singular vectors.

        Parameters
        ----------
        u: numpy array
            Left singular vectors.

        Returns
        -------
        None
        """
        a = int(u.shape[1] - int(np.sqrt(u.shape[1]))**2)
        if a == 0:
            size = int(np.sqrt(u.shape[1]))
        else:
            size = int(np.sqrt(u.shape[1])) + 1
        fig, axs = plt.subplots(int(np.sqrt(u.shape[1])),
                                size, figsize=(15, 13))
        axs = axs.ravel()
        fig.suptitle('Left singular vectors of k_pot matrix')
        for i in range(u.shape[1]):
            axs[i].plot(u[:, i], marker='.')
            axs[i].set_title(r'$u_{'+str(i+1)+'}$')
        plt.show()
        return

    def plot_v(self, v):
        """
        Creates plot of right singular values

        Parameters
        ----------
        v: right singular vectors

        Returns
        -------
        None
        """
        a = int(v.shape[1] - int(np.sqrt(v.shape[1]))**2)
        if a == 0:
            size = int(np.sqrt(v.shape[1]))
        else:
            size = int(np.sqrt(v.shape[1])) + 1
        fig, axs = plt.subplots(int(np.sqrt(v.shape[1])),
                                size, figsize=(15, 13))
        axs = axs.ravel()
        fig.suptitle('right singular vectors of k_pot matrix')
        for i in range(v.shape[1]):
            axs[i].plot(v[i, :], marker='.')
            axs[i].set_title(r'$v_{'+str(i+1)+'}$')
        plt.show()
        return

    def plot_svd_u(self, u_svd):
        """
        Creates plot of left singular values

        Parameters
        ----------
        u_svd: left singular vectors

        Returns
        -------
        None
        """
        a = int(u_svd.shape[1] - int(np.sqrt(u_svd.shape[1]))**2)
        if a == 0:
            size = int(np.sqrt(u_svd.shape[1]))
        else:
            size = int(np.sqrt(u_svd.shape[1])) + 1
        fig, axs = plt.subplots(int(np.sqrt(u_svd.shape[1])),
                                size, figsize=(15, 14))
        axs = axs.ravel()
        fig.suptitle('Left singular vectors of kernels product')
        for i in range(u_svd.shape[1]):
            axs[i].plot(u_svd[:, i], '.')
            axs[i].set_title(r'$u_{'+str(i+1)+'}$')
        plt.close()
        return

    def plot_svd_v(self, v_svd):
        """
        Creates plot of right singular values

        Parameters
        ----------
        v_svd: right singular vectors

        Returns
        -------
        None
        """
        a = int(v_svd.shape[1] - int(np.sqrt(v_svd.shape[1]))**2)
        if a == 0:
            size = int(np.sqrt(v_svd.shape[1]))
        else:
            size = int(np.sqrt(v_svd.shape[1])) + 1
        fig, axs = plt.subplots(int(np.sqrt(v_svd.shape[1])),
                                size, figsize=(15, 14))
        axs = axs.ravel()
        fig.suptitle('Right singular vectors of kernels product')
        for i in range(v_svd.shape[1]):
            axs[i].plot(v_svd[i, :], marker='.')
            axs[i].set_title(r'$v_{'+str(i+1)+'}$')
        plt.show()
        return

    def plot_evd(self, eigenvalues):
        """
        Creates plot of eigenvalues

        Parameters
        ----------
        eigenvalues: eigenvalues

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(eigenvalues, 'b.')
        plt. title('Eigenvalues of kernels product')
        plt.xlabel('Components number')
        plt.ylabel('Eigenvalues')
        plt.show()
        return


class ValidationClassKCSD1D(ValidationClassKCSD):
    """
    TestKCSD1D - The 1D variant of tests for kCSD method.
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        """
        Initialize TestKCSD1D class

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator
        **kwargs
            configuration parameters

        Returns
        -------
        None
        """
        super(ValidationClassKCSD1D, self).__init__(dim=1, **kwargs)
        self.csd_seed = csd_seed
        return

    def generate_csd(self, csd_profile):
        """
        Gives CSD profile at the requested spatial location,
        at 'res' resolution

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile

        Returns
        -------
        csd_x: numpy array, shape (csd_xres)
            positions at x axis (where is the ground truth)
        true_csd: numpy array, shape (csd_xres)
            csd at csd_x positions
        """
        csd_at = np.linspace(self.true_csd_xlims[0], self.true_csd_xlims[1],
                             self.csd_xres)
        true_csd = csd_profile(csd_at, self.csd_seed)
        return csd_at, true_csd

    def calculate_potential(self, true_csd, csd_at, ele_pos):
        """
        Calculates potentials

        Parameters
        ----------
        electrode_locations: numpy array, shape()
            locations of electrodes
        csd_xres: int
            resolution of ground truth

        Returns
        -------
        pots: numpy array, shape (total_ele)
            normalized values of potentials as in eq.:26 from Potworowski(2012)
        """
        pots = np.zeros(len(ele_pos))
        for index in range(len(ele_pos)):
            pots[index] = self.integrate(csd_at, true_csd, ele_pos[index])
        # eq.: 26 from Potworowski (2012)
        pots *= old_div(1, (2. * self.sigma))
        return pots

    def integrate(self, csd_at, csd, x0):
        """
        Calculates integrals (potential values) according to Simpson's rule in
        1D space

        Parameters
        ----------
        csd_x: numpy array, shape (csd_xres)
            position on x axis
        csd: numpy array, shape (csd_xres)
            values of csd (ground truth) at csd_x positions
        x0: float
            single electrode location/position

        Returns
        -------
        Integral: float
            calculated potential at x0 position
        """
        m = np.sqrt((csd_at - x0)**2 + self.h**2) - abs(csd_at - x0)
        y = csd * m
        Integral = simps(y, csd_at)
        return Integral

    def make_reconstruction(self, csd_profile, noise=None, nr_broken_ele=0):
        """
        Main method, makes the whole kCSD reconstruction

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile

        Returns
        -------
        rms: float
            error of reconstruction
        point_error: numpy array
            error of reconstruction calculated at every point of reconstruction
            space
        """
        csd_at, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile, noise,
                                              nr_broken_ele)
        kcsd = KCSD1D(ele_pos, pots, src_type='gauss', sigma=self.sigma,
                      h=self.h, n_src_init=self.n_src_init, ext_x=self.ext_x)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd)
        test_csd = csd_profile(kcsd.estm_x, self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (kcsd.lambd,
                                                                kcsd.R, rms)
        self.make_plot(kcsd, csd_at, true_csd, ele_pos, pots, est_csd,
                       est_pot, title)
        ss = SpectralStructure(kcsd)
        ss.picard_plot(pots)
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        return kcsd

    def make_plot(self, k, csd_at, true_csd, ele_pos, pots, est_csd,
                  est_pot, title):
        """
        Creates plots of ground truth, measured potentials and recontruction

        Parameters
        ----------
        k: object of the class
        csd_x: numpy array
            x coordinates of ground truth (true_csd)
        true_csd: numpy array
            ground truth data
        kcsd: object of the class
        est_csd: numpy array
            reconstructed csd
        ele_pos: numpy array
            positions of electrodes
        pots: numpy array
            potentials measured on electrodes
        rms: float
            error of reconstruction
        title: string
            title of the plot

        Returns
        -------
        None
        """
        # CSDs
        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.subplot(211)
        ax1.plot(csd_at, true_csd, 'g', label='TrueCSD')
        ax1.plot(k.estm_x, est_csd[:, 0], 'r--', label='kCSD')
        ax1.plot(ele_pos, np.zeros(len(pots)), 'ko')
        ax1.set_xlim(csd_at[0], csd_at[-1])
        ax1.set_xlabel('Depth [mm]')
        ax1.set_ylabel('CSD [mA/mm]')
        ax1.set_title('A) Currents')
        ax1.legend()
        # Potentials
        ax2 = plt.subplot(212)
        ax2.plot(ele_pos, pots, 'b.', label='TruePots')
        ax2.plot(k.estm_x, est_pot, 'y--', label='EstPots')
        ax2.set_xlim(csd_at[0], csd_at[-1])
        ax2.set_xlabel('Depth [mm]')
        ax2.set_ylabel('Potential [mV]')
        ax2.set_title('B) Potentials')
        ax2.legend()
        fig.suptitle(title)
        plt.show()
        return


class ValidationClassKCSD2D(ValidationClassKCSD):
    """
    TestKCSD2D - The 2D variant of tests for kCSD method.
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        """
        Initialize TestKCSD2D class

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator
        **kwargs
            configuration parameters

        Returns
        -------
        None
        """
        super(ValidationClassKCSD2D, self).__init__(dim=2, **kwargs)
#        self.general_parameters(**kwargs)
#        self.dimension_parameters(**kwargs)
        self.csd_seed = csd_seed
        return

    def generate_csd(self, csd_profile):
        """
        Gives CSD profile at the requested spatial location,
        at 'res' resolution

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile

        Returns
        -------
        csd_x: numpy array, shape (res_x, res_y)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y)
            y coordinates of ground truth data
        f: numpy array, shape (res_x, res_y)
            y coordinates of ground truth data
            calculated csd at locations indicated by csd_x and csd_y

        """
        csd_at = np.mgrid[self.true_csd_xlims[0]:self.true_csd_xlims[1]:
                          np.complex(0, self.csd_xres),
                          self.true_csd_ylims[0]:self.true_csd_ylims[1]:
                          np.complex(0, self.csd_yres)]
        f = csd_profile(csd_at, self.csd_seed)
        return csd_at, f

    def calculate_potential(self, true_csd, csd_at, ele_x, ele_y):
        """
        Computes the LFP generated by true_csd (ground truth)

        Parameters
        ----------
        true_csd: numpy array, shape (res_x, res_y)
            ground truth data (true_csd)
        csd_x: numpy array, shape (res_x, res_y)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y)
            y coordinates of ground truth data
        ele_x: numpy array, shape (len(ele_pos.shape[0]))
            x coordinates of electrodes
        ele_y: numpy array, shape (len(ele_pos.shape[0]))
            y coordinates of electrodes

        Returns
        -------
        pots: numpy array, shape (total_ele)
            calculated potentials
        """
        xlin = csd_at[0, :, 0]
        ylin = csd_at[1, 0, :]
        pots = np.zeros(len(ele_x))
        for ii in range(len(ele_x)):
            pots[ii] = self.integrate(ele_x[ii], ele_y[ii], true_csd,
                                      xlin, ylin, csd_at)
        pots /= 2 * np.pi * self.sigma
        return pots

    def integrate(self, x, y, true_csd, xlin, ylin, csd_at):
        """
        Integrates currents to calculate potentials on electrode in 2D space

        Parameters
        ----------
        x: float
            x coordinate of electrode
        y: float
            y coordinate of electrode
        true_csd: numpy array, shape (res_x, res_y)
            ground truth data (true_csd)
        xlin: numpy array, shape (res_x, 1)
            x range for coordinates of true_csd
        ylin: numpy array, shape (res_y, 1)
            y range for coordinates of true_csd
        csd_x: numpy array, shape (res_x, res_y)
            full x coordinates of true_csd
        csd_y: numpy array, shape (res_x, res_y)
            full y coordinates of true_csd

        Returns
        -------
        F: float
            potential on a single electrode
        """
        csd_x = csd_at[0]
        csd_y = csd_at[1]
        Ny = ylin.shape[0]
        m = np.sqrt((x - csd_x)**2 + (y - csd_y)**2)  # construct 2-D integrand
        m[m < 0.0000001] = 0.0000001             # I increased acuracy
        y = np.arcsinh(2 * self.h / m) * true_csd            # corrected
        integral_1D = np.zeros(Ny)           # do a 1-D integral over every row
        for i in range(Ny):
            integral_1D[i] = simps(y[:, i], ylin)      # I changed the integral
        F = simps(integral_1D, xlin)         # then an integral over the result
        return F

    def make_reconstruction(self, csd_profile, noise=None, nr_broken_ele=0):
        """
        Main method, makes the whole kCSD reconstruction

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile

        Returns
        -------
        None
        """
        csd_at, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile, noise,
                                              nr_broken_ele)
        kcsd = KCSD2D(ele_pos, pots, xmin=0., xmax=1., ymin=0.,
                      ymax=1., h=self.h, sigma=self.sigma,
                      n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.1, 0.31, 0.05))
        test_csd = csd_profile([kcsd.estm_x, kcsd.estm_y], self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, 0])
        title = 'csd_profile_' + csd_profile.__name__ + '_seed' +\
            str(self.csd_seed) + '_total_ele' + str(self.total_ele)
        self.make_plot(csd_at, test_csd, kcsd, est_csd, ele_pos, pots,
                       rms, title)
        ss = SpectralStructure(kcsd)
        ss.picard_plot(pots)
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, 0])
        return kcsd, point_error

    def plot_point_error(self, point_error, k):
        """
        Creates plot of point error

        Parameters
        ----------
        point_error: numpy array
        k: object of the class

        Returns
        -------
        None
        """

        plt.figure(figsize=(10, 6))
        plt.contourf(k.estm_x, k.estm_y, point_error, cmap=cm.Greys)
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.colorbar()
        plt.show()
        return

    def make_plot(self, csd_at, true_csd, kcsd, est_csd, ele_pos, pots,
                  rms, title):
        """
        Creates plot of ground truth data, calculated potentials and
        reconstruction

        Parameters
        ----------
        csd_x: numpy array
            x coordinates of ground truth (true_csd)
        csd_y: numpy array
            y coordinates of ground truth (true_csd)
        true_csd: numpy array
            ground truth data
        kcsd: object of the class
        est_csd: numpy array
            reconstructed csd
        ele_pos: numpy array
            positions of electrodes
        pots: numpy array
            potentials measured on electrodes
        rms: float
            error of reconstruction
        title: string
            title of the plot

        Returns
        -------
        None
        """
        csd_x = csd_at[0]
        csd_y = csd_at[1]
        suptitle = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (kcsd.lambd,
                                                                   kcsd.R, rms)
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle(suptitle)
        ax1 = plt.subplot(141, aspect='equal')
        t_max = np.max(np.abs(true_csd))
        levels = np.linspace(-1 * t_max, t_max, 16)
        im1 = ax1.contourf(csd_x, csd_y, true_csd, levels=levels,
                           cmap=cm.bwr)
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel('y [mm]')
        ax1.set_title('A) True CSD')
        ticks = np.linspace(-1 * t_max, t_max, 7, endpoint=True)
        plt.colorbar(im1, orientation='horizontal', format='%.2f',
                     ticks=ticks)

        ax2 = plt.subplot(143, aspect='equal')
        levels2 = np.linspace(0, 1, 10)
        t_max = np.max(np.abs(est_csd[:, :, 0]))
        levels_kcsd = np.linspace(-1 * t_max, t_max, 16, endpoint=True)
        im2 = ax2.contourf(kcsd.estm_x, kcsd.estm_y, est_csd[:, :, 0],
                           levels=levels_kcsd, alpha=1, cmap=cm.bwr)
        if self.mask is not False:
            ax2.contourf(kcsd.estm_x, kcsd.estm_y, self.mask, levels=levels2,
                         alpha=0.3, cmap='Greys')
            ax2.set_title('C) kCSD with error mask')
        else:
            ax2.set_title('C) kCSD')
        ax2.set_ylabel('y [mm]')
        ax2.set_xlim([0., 1.])
        ax2.set_ylim([0., 1.])
        ticks = np.linspace(-1 * t_max, t_max, 7, endpoint=True)
        plt.colorbar(im2, orientation='horizontal', format='%.2f',
                     ticks=ticks)

        ax3 = plt.subplot(142, aspect='equal')
        v_max = np.max(np.abs(pots))
        levels_pot = np.linspace(-1 * v_max, v_max, 32)
        X, Y, Z = self.grid(ele_pos[:, 0], ele_pos[:, 1], pots)
        im3 = plt.contourf(X, Y, Z, levels=levels_pot, cmap=cm.PRGn)
        plt.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c='k')
        ax3.set_xlim([0., 1.])
        ax3.set_ylim([0., 1.])
        ax3.set_title('B) Pots, Ele_pos')
        ticks = np.linspace(-1 * v_max, v_max, 7, endpoint=True)
        plt.colorbar(im3, orientation='horizontal', format='%.2f',
                     ticks=ticks)

        ax4 = plt.subplot(144, aspect='equal')
        difference = abs(true_csd-est_csd[:, :, 0])
        cmap = colors.LinearSegmentedColormap.from_list("",
                                                        ["white",
                                                         "darkorange"])
        im4 = ax4.contourf(kcsd.estm_x, kcsd.estm_y, difference,
                           cmap=cmap,
                           levels=np.linspace(0, np.max(difference), 15))
        if self.mask is not False:
            ax4.contourf(kcsd.estm_x, kcsd.estm_y, self.mask, levels=levels2,
                         alpha=0.3, cmap='Greys')
        ax4.set_xlabel('x [mm]')
        ax4.set_title('D) |True CSD - kCSD|')
        v = np.linspace(0, np.max(difference), 7, endpoint=True)
        plt.colorbar(im4, orientation='horizontal', format='%.2f', ticks=v)
        plt.show()
        return

    def grid(self, x, y, z, resX=100, resY=100):
        """
        Convert 3 column data to matplotlib grid

        Parameters
        ----------
        x
        y
        z

        Returns
        -------
        xi
        yi
        zi
        """
        z = z.flatten()
        xi = np.linspace(min(x), max(x), resX)
        yi = np.linspace(min(y), max(y), resY)
        zi = griddata(x, y, z, xi, yi, interp='linear')
        return xi, yi, zi


class ValidationClassKCSD3D(ValidationClassKCSD):
    """
    ValidationClassKCSD3D - The 3D variant of validation class for kCSD method.
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        """
        Initialize TestKCSD3D class

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator
        **kwargs
            configuration parameters

        Returns
        -------
        None
        """
        super(ValidationClassKCSD3D, self).__init__(dim=3, **kwargs)
#        self.general_parameters(**kwargs)
#        self.dimension_parameters(**kwargs)
        self.csd_seed = csd_seed
        return

    def generate_csd(self, csd_profile):
        """
        Gives CSD profile at the requested spatial location,
        at 'res' resolution

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile

        Returns
        -------
        csd_x: numpy array, shape (res_x, res_y, res_z)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y, res_z)
            y coordinates of ground truth data
        csd_z: numpy array, shape (res_x, res_y, res_z)
            z coordinates of ground truth data
        f: numpy array, shape (res_x, res_y, res_z)
            y coordinates of ground truth data
            calculated csd at locations indicated by csd_x and csd_y

        """
        csd_at = np.mgrid[self.true_csd_xlims[0]:self.true_csd_xlims[1]:
                          np.complex(0, self.csd_xres),
                          self.true_csd_ylims[0]:self.true_csd_ylims[1]:
                          np.complex(0, self.csd_yres),
                          self.true_csd_zlims[0]:self.true_csd_zlims[1]:
                          np.complex(0, self.csd_zres)]
        f = csd_profile(csd_at, self.csd_seed)
        return csd_at, f

    def integrate(self, x, y, z, csd, xlin, ylin, zlin, X, Y, Z):
        """
        Integrates currents to calculate potentials on electrode in 3D space

        Parameters
        ----------
        x: float
            x coordinate of electrode
        y: float
            y coordinate of electrode
        true_csd: numpy array, shape (res_x, res_y)
            ground truth data (true_csd)
        xlin: numpy array, shape (res_x, 1)
            x range for coordinates of true_csd
        ylin: numpy array, shape (res_y, 1)
            y range for coordinates of true_csd
        csd_x: numpy array, shape (res_x, res_y)
            full x coordinates of true_csd
        csd_y: numpy array, shape (res_x, res_y)
            full y coordinates of true_csd

        Returns
        -------
        F: float
            potential on a single electrode
        """
        Nz = zlin.shape[0]
        Ny = ylin.shape[0]
        m = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
        m[m < 0.0000001] = 0.0000001
        z = csd / m
        Iy = np.zeros(Ny)
        for j in range(Ny):
            Iz = np.zeros(Nz)
            for i in range(Nz):
                Iz[i] = simps(z[:, j, i], zlin)
            Iy[j] = simps(Iz, ylin)
        F = simps(Iy, xlin)
        return F

    def calculate_potential(self, true_csd, ele_xx, ele_yy, ele_zz, csd_at):
        """
        Computes the LFP generated by true_csd (ground truth)

        Parameters
        ----------
        true_csd: numpy array, shape (res_x, res_y, res_z)
            ground truth data (true_csd)
        csd_x: numpy array, shape (res_x, res_y, res_z)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y, res_z)
            y coordinates of ground truth data
        csd_z: numpy array, shape (res_x, res_y, res_z)
            z coordinates of ground truth data
        ele_xx: numpy array, shape (len(ele_pos.shape[0]))
            xx coordinates of electrodes
        ele_yy: numpy array, shape (len(ele_pos.shape[0]))
            yy coordinates of electrodes
        ele_zz: numpy array, shape (len(ele_pos.shape[0]))
            zz coordinates of electrodes

        Returns
        -------
        pots: numpy array, shape (total_ele)
            calculated potentials
        """
        xlin = csd_at[0, :, 0, 0]
        ylin = csd_at[1, 0, :, 0]
        zlin = csd_at[2, 0, 0, :]
        pots = np.zeros(len(ele_xx))
        for ii in range(len(ele_xx)):
            pots[ii] = self.integrate(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                      true_csd, xlin, ylin, zlin,
                                      csd_at[0], csd_at[1], csd_at[2])
        pots /= 4*np.pi*self.sigma
        return pots

    def calculate_potential_parallel(self, true_csd, ele_xx, ele_yy, ele_zz,
                                     csd_at):
        """
        Computes the LFP generated by true_csd (ground truth) using parallel
        computing

        Parameters
        ----------
        true_csd: numpy array, shape (res_x, res_y, res_z)
            ground truth data (true_csd)
        csd_x: numpy array, shape (res_x, res_y, res_z)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y, res_z)
            y coordinates of ground truth data
        csd_z: numpy array, shape (res_x, res_y, res_z)
            z coordinates of ground truth data
        ele_xx: numpy array, shape (len(ele_pos.shape[0]))
            xx coordinates of electrodes
        ele_yy: numpy array, shape (len(ele_pos.shape[0]))
            yy coordinates of electrodes
        ele_zz: numpy array, shape (len(ele_pos.shape[0]))
            zz coordinates of electrodes

        Returns
        -------
        pots: numpy array, shape (total_ele)
            calculated potentials
        """
        xlin = csd_at[0, :, 0, 0]
        ylin = csd_at[1, 0, :, 0]
        zlin = csd_at[2, 0, 0, :]
        pots = Parallel(n_jobs=num_cores)(delayed(self.integrate)
                                          (ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                           true_csd,
                                           xlin, ylin, zlin,
                                           csd_at[0], csd_at[1], csd_at[2])
                                          for ii in range(len(ele_xx)))
        pots = np.array(pots)
        pots /= 4*np.pi*self.sigma
        return pots

    def make_reconstruction(self, csd_profile, noise=None, nr_broken_ele=0):
        """
        Main method, makes the whole kCSD reconstruction

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile

        Returns
        -------
        None
        """
        csd_at, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile, noise,
                                              nr_broken_ele)
        kcsd = KCSD3D(ele_pos, pots, gdx=0.035, gdy=0.035, gdz=0.035,
                      h=self.h, sigma=self.sigma, xmax=1, xmin=0, ymax=1,
                      ymin=0, zmax=1, zmin=0, n_src_init=self.n_src_init)
        tic = time.time()
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        np.arange(0.08, 0.4, 0.05))
        ss = SpectralStructure(kcsd)
        ss.picard_plot(pots)
        toc = time.time() - tic
        test_csd = csd_profile([kcsd.estm_x, kcsd.estm_y, kcsd.estm_z],
                               self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
        point_error = self.calculate_point_error(test_csd,
                                                 est_csd[:, :, :, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS: %0.2E; CV_Error: %0.2E; "\
                "Time: %0.2f" % (kcsd.lambd, kcsd.R, rms, kcsd.cv_error, toc)
        self.make_plot(csd_at, test_csd, kcsd, est_csd, ele_pos,
                       pots, rms, title)
        return kcsd

    def make_plot(self, csd_at, true_csd, kcsd, est_csd, ele_pos,
                  pots, rms, fig_title):
        """
        Creates plot of ground truth data, calculated potentials and
        reconstruction

        Parameters
        ----------
        csd_x: numpy array
            x coordinates of ground truth (true_csd)
        csd_y: numpy array
            y coordinates of ground truth (true_csd)
        csd_z: numpy array
            z coordinates of ground truth (true_csd)
        true_csd: numpy array
            ground truth data
        kcsd: object of the class
        est_csd: numpy array
            reconstructed csd
        ele_pos: numpy array
            positions of electrodes
        pots: numpy array
            potentials measured on electrodes
        rms: float
            error of reconstruction
        title: string
            title of the plot

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(10, 16))
        z_steps = 5
        height_ratios = [1 for i in range(z_steps)]
        height_ratios.append(0.1)
        gs = gridspec.GridSpec(z_steps+1, 3, height_ratios=height_ratios)
        t_max = np.max(np.abs(true_csd))
        levels = np.linspace(-1*t_max, t_max, 16)
        ind_interest = np.mgrid[0:kcsd.estm_z.shape[2]:np.complex(0,
                                                                  z_steps+2)]
        ind_interest = np.array(ind_interest, dtype=np.int)[1:-1]
        for ii, idx in enumerate(ind_interest):
            ax = plt.subplot(gs[ii, 0])
            im = plt.contourf(kcsd.estm_x[:, :, idx], kcsd.estm_y[:, :, idx],
                              true_csd[:, :, idx], levels=levels,
                              cmap=cm.bwr_r)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            title = str(kcsd.estm_z[:, :, idx][0][0])[:4]
            ax.set_title(label=title, fontdict={'x': 0.8, 'y': 0.8})
            ax.set_aspect('equal')
        cax = plt.subplot(gs[z_steps, 0])
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal',
                            format='%.2f')
        cbar.set_ticks(levels[::3])
        cbar.set_ticklabels(np.around(levels[::3], decimals=2))
#        Potentials
        v_max = np.max(np.abs(pots))
        levels_pot = np.linspace(-1*v_max, v_max, 16)
        ele_res = int(np.ceil(len(pots)**(3**-1)))
        ele_x = ele_pos[:, 0].reshape(ele_res, ele_res, ele_res)
        ele_y = ele_pos[:, 1].reshape(ele_res, ele_res, ele_res)
        ele_z = ele_pos[:, 2].reshape(ele_res, ele_res, ele_res)
        pots = pots.reshape(ele_res, ele_res, ele_res)
        for idx in range(min(5, ele_res)):
            X, Y, Z = self.grid(ele_x[:, :, idx], ele_y[:, :, idx],
                                pots[:, :, idx])
            ax = plt.subplot(gs[idx, 1])
            im = plt.contourf(X, Y, Z, levels=levels_pot, cmap=cm.PRGn)
            plt.scatter(ele_x[:, :, idx], ele_y[:, :, idx], 5, c='k')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            title = str(ele_z[:, :, idx][0][0])[:4]
            ax.set_title(label=title, fontdict={'x': 0.8, 'y': 0.8})
            ax.set_aspect('equal')
            ax.set_xlim([0., 1.])
            ax.set_ylim([0., 1.])
        cax = plt.subplot(gs[z_steps, 1])
        cbar2 = plt.colorbar(im, cax=cax, orientation='horizontal',
                             format='%.2f')
        cbar2.set_ticks(levels_pot[::3])
        cbar2.set_ticklabels(np.around(levels_pot[::3], decimals=2))
        # #KCSD
        t_max = np.max(np.abs(est_csd[:, :, :, 0]))
        levels_kcsd = np.linspace(-1*t_max, t_max, 16)
        ind_interest = np.mgrid[0:kcsd.estm_z.shape[2]:np.complex(0,
                                                                  z_steps+2)]
        ind_interest = np.array(ind_interest, dtype=np.int)[1:-1]
        for ii, idx in enumerate(ind_interest):
            ax = plt.subplot(gs[ii, 2])
            im = plt.contourf(kcsd.estm_x[:, :, idx], kcsd.estm_y[:, :, idx],
                              est_csd[:, :, idx, 0], levels=levels_kcsd,
                              cmap=cm.bwr_r)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            title = str(kcsd.estm_z[:, :, idx][0][0])[:4]
            ax.set_title(label=title, fontdict={'x': 0.8, 'y': 0.8})
            ax.set_aspect('equal')
        cax = plt.subplot(gs[z_steps, 2])
        cbar3 = plt.colorbar(im, cax=cax, orientation='horizontal',
                             format='%.2f')
        cbar3.set_ticks(levels_kcsd[::3])
        cbar3.set_ticklabels(np.around(levels_kcsd[::3], decimals=2))
        fig.suptitle(fig_title)
        plt.show()
        return

    def grid(self, x, y, z, resX=100, resY=100):
        """
        Convert 3 column data to matplotlib grid
        """
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        xi = np.linspace(min(x), max(x), resX)
        yi = np.linspace(min(y), max(y), resY)
        zi = griddata(x, y, z, xi, yi, interp='linear')
        return xi, yi, zi


if __name__ == '__main__':
#    print('Checking 1D')
#    csd_profile = CSD.gauss_1d_mono
#    R_init = 0.23
#    csd_seed = 5
#    total_ele = 16
#    n_src_init = 100
#    ele_lims = [0.1, 0.9]  # range of electrodes space
#    true_csd_xlims = [0., 1.]
#
#    k = ValidationClassKCSD1D(csd_profile, csd_seed,
#                              total_ele=total_ele, n_src_init=n_src_init,
#                              h=0.25, R_init=R_init, ele_xlims=ele_lims,
#                              true_csd_xlims=true_csd_xlims, sigma=0.3,
#                              src_type='gauss', ext_x=0.1,
#                              config='regular')
#    k.make_reconstruction(csd_profile)

    print('Checking 2D')
    csd_profile = CSD.gauss_2d_small
    csd_seed = 7
    total_ele = 36
#    mask = np.load('/home/mkowalska/Marta/xCSD/branches/kCSD-marta/'
#                   'refactored_tests/mask.npy')
    k = ValidationClassKCSD2D(csd_profile, csd_seed, total_ele=total_ele,
                              h=50., sigma=1., config='regular', err_map='no',
                              n_src_init=400)
    k.make_reconstruction(csd_profile)

#    print('Checking 3D')
#    total_ele = 125
#    csd_seed = 20  # 0-49 are small sources, 50-99 are large sources
#    csd_profile = CSD.gauss_3d_small
#    tic = time.time()
#    k = ValidationClassKCSD3D(csd_profile, csd_seed, total_ele=total_ele, h=50,
#                              sigma=1, xmax=1, xmin=0, ymax=1, ymin=0, zmax=1,
#                              zmin=0, config='regular')
#    k.make_reconstruction(csd_profile)
#    toc = time.time() - tic
#    print('time', toc)
