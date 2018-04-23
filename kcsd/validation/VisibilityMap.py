"""
@author: mkowalska
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from builtins import range

import time
import numpy as np
import matplotlib.pyplot as plt

from kcsd import ValidateKCSD1D, ValidateKCSD2D, ValidateKCSD3D
from kcsd import csd_profile as CSD
from kcsd import KCSD1D, KCSD2D, KCSD3D

try:
    from joblib import Parallel, delayed
    import multiprocessing
    NUM_CORES = multiprocessing.cpu_count() - 1
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


class VisibilityMap1D(ValidateKCSD1D):
    """
    Class that produces error map for 1D CSD reconstruction.
    """
    def __init__(self, **kwargs):
        """
        Initialize ErrorMap1D class.

        Parameters
        ----------
        csd_profile: function
            Function to produce csd profile.
        csd_seed: int
            Seed for random generator to choose random CSD profile.
        **kwargs
            Configuration parameters.

        Returns
        -------
        None
        """
        super(VisibilityMap1D, self).__init__(1, **kwargs)
        return

    def calculate_error_map(self, csd_profile, total_ele, n=100, noise=None,
                            nr_broken_ele=None, Rs=None, lambdas=None):
        """
        Makes reconstructions for n random simulated ground truth profiles and
        returns errors of CSD estimation with kCSD method.

        Parameters
        ----------
        csd_profile: function
            Function to produce csd profile.
        n: int
            Number of simulations included in error map calculations.
            Default: 100.
        noise: string
            Determines if we want to generate data with noise.
            Default: None.
        nr_broken_ele: int
            How many electrodes are broken (excluded from analysis)
            Default: None.
        Rs: numpy 1D array
            Basis source parameter for crossvalidation.
            Default: None.
        lambdas: numpy 1D array
            Regularization parameter for crossvalidation.
            Default: None.

        Returns
        -------
        rms: float
            Error of reconstruction.
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        """
        if PARALLEL_AVAILABLE:
            err = Parallel(n_jobs=NUM_CORES)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, total_ele,
                                              noise=noise,
                                              nr_broken_ele=nr_broken_ele,
                                              Rs=Rs, lambdas=lambdas)
                                             for i in range(n))
            rms = np.array([item[0] for item in err])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i,
                                                       total_ele, noise,
                                                       nr_broken_ele, Rs=Rs,
                                                       lambdas=lambdas)
                rms[i] = data
                point_error.append(error)
        point_error = np.array(point_error)
        return rms, point_error

    def make_reconstruction(self, csd_profile, csd_seed, total_ele, noise=None,
                            nr_broken_ele=None, Rs=None, lambdas=None):
        """
        Makes the whole kCSD reconstruction.

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            Seed for random generator to choose random CSD profile.
        noise: string
            Determines if we want to generate data with noise.
            Default: None.
        nr_broken_ele: int
            How many electrodes are broken (excluded from analysis)
            Default: None.
        Rs: numpy 1D array
            Basis source parameter for crossvalidation.
            Default: None.
        lambdas: numpy 1D array
            Regularization parameter for crossvalidation.
            Default: None.

        Returns
        -------
        List of [rms, kcsd] and point_error
        rms: float
            Error of reconstruction.
        kcsd: object of a class
            Object of a class.
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        """
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed,
                                              total_ele, self.ele_lims, self.h,
                                              self.sigma,
                                              noise=None, nr_broken_ele=None,
                                              ele_seed=10)

        k = KCSD1D(ele_pos, pots, xmin=0., xmax=1., h=self.h,
                   sigma=self.sigma, n_src_init=self.n_src_init)
        k.cross_validate(Rs=Rs, lambdas=lambdas)
        est_csd = k.values('CSD')
        test_csd = csd_profile(k.estm_x, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        return rms, point_error

    def plot_error_map(self, point_error, ele_pos):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        ele_pos: numpy array
            Positions of electrodes.

        Returns
        -------
        None
        """
        mean_err = self.sigmoid_mean(point_error)
        plt.figure(figsize=(10, 6))
        plt.title('Sigmoidal mean point error for random sources')
        plt.plot(np.linspace(ele_pos[0], ele_pos[-1], mean_err.shape[0]),
                 mean_err, 'b.', label='mean error')
        plt.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        plt.xlabel('Depth [mm]')
        plt.ylabel('RMS Error')
        plt.legend()
        plt.show()
        return


class VisibilityMap2D(ValidateKCSD2D):
    """
    Class that produces error map for 2D CSD reconstruction.
    """
    def __init__(self, **kwargs):
        """
        Initialize ErrorMap2D class.

        Parameters
        ----------
        csd_profile: function
            Function to produce csd profile.
        csd_seed: int
            Seed for random generator to choose random CSD profile.
        **kwargs
            Configuration parameters.

        Returns
        -------
        None
        """
        super(VisibilityMap2D, self).__init__(1, **kwargs)
        return

    def make_reconstruction(self, csd_profile, csd_seed, total_ele, noise=None,
                            nr_broken_ele=None, Rs=None, lambdas=None):
        """
        Makes the whole kCSD reconstruction.

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            Seed for random generator to choose random CSD profile.
        noise: string
            Determines if we want to generate data with noise.
            Default: None.
        nr_broken_ele: int
            How many electrodes are broken (excluded from analysis)
            Default: None.
        Rs: numpy 1D array
            Basis source parameter for crossvalidation.
            Default: None.
        lambdas: numpy 1D array
            Regularization parameter for crossvalidation.
            Default: None.

        Returns
        -------
        List of [rms, kcsd] and point_error
        rms: float
            Error of reconstruction.
        kcsd: object of a class
            Object of a class.
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        """
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed,
                                              total_ele, self.ele_lims, self.h,
                                              self.sigma,
                                              noise=None, nr_broken_ele=None,
                                              ele_seed=10)

        k = KCSD2D(ele_pos, pots, xmin=0., xmax=1., ymin=0.,
                   ymax=1., h=self.h, sigma=self.sigma,
                   n_src_init=self.n_src_init)
        k.cross_validate(Rs=Rs, lambdas=lambdas)
        est_csd = k.values('CSD')
        test_csd = csd_profile([k.estm_x, k.estm_y], csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, 0])
        return rms, point_error

    def calculate_error_map(self, csd_profile, total_ele, n=100, noise=None,
                            nr_broken_ele=None, Rs=None, lambdas=None):
        """
        Makes reconstructions for n random simulated ground truth profiles and
        returns errors of CSD estimation with kCSD method.

        Parameters
        ----------
        csd_profile: function
            Function to produce csd profile.
        n: int
            Number of simulations included in error map calculations.
            Default: 100.
        noise: string
            Determines if we want to generate data with noise.
            Default: None.
        nr_broken_ele: int
            How many electrodes are broken (excluded from analysis)
            Default: None.
        Rs: numpy 1D array
            Basis source parameter for crossvalidation.
            Default: None.
        lambdas: numpy 1D array
            Regularization parameter for crossvalidation.
            Default: None.

        Returns
        -------
        rms: float
            Error of reconstruction.
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        """
        tic = time.time()
        if PARALLEL_AVAILABLE:
            err = Parallel(n_jobs=NUM_CORES)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, total_ele, noise,
                                              nr_broken_ele, Rs, lambdas)
                                             for i in range(n))
            rms = np.array([item[0] for item in err])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i,
                                                       total_ele, noise,
                                                       nr_broken_ele, Rs,
                                                       lambdas)
                rms[i] = data
                point_error.append(error)
        point_error = np.array(point_error)
        toc = time.time() - tic
        print('time: ', toc)
        return rms, point_error

    def plot_error_map(self, point_error, ele_pos):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        ele_pos: numpy array
            Positions of electrodes.

        Returns
        -------
        None
        """
        ele_x, ele_y = ele_pos[:, 0], ele_pos[:, 1]
#        x, y = np.mgrid[np.min(ele_x):np.max(ele_x):
#                        np.complex(0, self.est_xres),
#                        np.min(ele_y):np.max(ele_y):
#                        np.complex(0, self.est_yres)]
        x, y = np.mgrid[0:1:
                        np.complex(0, point_error.shape[1]),
                        0:1:
                        np.complex(0, point_error.shape[2])]
        mean_error = self.sigmoid_mean(point_error)
        plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(111, aspect='equal')
        levels = np.linspace(0, 1., 15)
        im = ax1.contourf(x, y, mean_error, levels=levels, cmap='Greys')
        plt.colorbar(im, fraction=0.046, pad=0.06)
        plt.scatter(ele_x, ele_y)
        ax1.set_xlabel('Depth x [mm]')
        ax1.set_ylabel('Depth y [mm]')
        ax1.set_title('Sigmoidal mean point error')
        plt.show()
        return


class VisibilityMap3D(ValidateKCSD3D):
    """
    Class that produces error map for 3D CSD reconstruction.
    """
    def __init__(self, **kwargs):
        """
        Initialize ErrorMap3D class.

        Parameters
        ----------
        csd_profile: function
            Function to produce csd profile.
        csd_seed: int
            Seed for random generator to choose random CSD profile.
        **kwargs
            Configuration parameters.

        Returns
        -------
        None
        """
        super(VisibilityMap3D, self).__init__(1, **kwargs)
        return

    def make_reconstruction(self, csd_profile, csd_seed, total_ele, noise=None,
                            nr_broken_ele=None, Rs=None, lambdas=None):
        """
        Makes the whole kCSD reconstruction.

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            Seed for random generator to choose random CSD profile.
        noise: string
            Determines if we want to generate data with noise.
            Default: None.
        nr_broken_ele: int
            How many electrodes are broken (excluded from analysis)
            Default: None.
        Rs: numpy 1D array
            Basis source parameter for crossvalidation.
            Default: None.
        lambdas: numpy 1D array
            Regularization parameter for crossvalidation.
            Default: None.

        Returns
        -------
        List of [rms, kcsd] and point_error
        rms: float
            Error of reconstruction.
        kcsd: object of a class
            Object of a class.
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        """
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed,
                                              total_ele, self.ele_lims, self.h,
                                              self.sigma,
                                              noise=None, nr_broken_ele=None,
                                              ele_seed=10)
        k = KCSD3D(ele_pos, pots, gdx=0.035, gdy=0.035, gdz=0.035,
                   h=self.h, sigma=self.sigma, xmax=1, xmin=0, ymax=1,
                   ymin=0, zmax=1, zmin=0, n_src_init=self.n_src_init)
        k.cross_validate(Rs=Rs, lambdas=lambdas)
        est_csd = k.values('CSD')
        test_csd = csd_profile([k.estm_x, k.estm_y, k.estm_z], csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, :, 0])
        return rms, point_error

    def calculate_error_map(self, csd_profile, total_ele, n=5, noise=None,
                            nr_broken_ele=None, Rs=None, lambdas=None):
        """
        Makes reconstructions for n random simulated ground truth profiles and
        returns errors of CSD estimation with kCSD method.

        Parameters
        ----------
        csd_profile: function
            Function to produce csd profile.
        n: int
            Number of simulations included in error map calculations.
            Default: 5.
        noise: string
            Determines if we want to generate data with noise.
            Default: None.
        nr_broken_ele: int
            How many electrodes are broken (excluded from analysis)
            Default: None.
        Rs: numpy 1D array
            Basis source parameter for crossvalidation.
            Default: None.
        lambdas: numpy 1D array
            Regularization parameter for crossvalidation.
            Default: None.

        Returns
        -------
        rms: float
            Error of reconstruction.
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        """
        tic = time.time()
        if PARALLEL_AVAILABLE:
            err = Parallel(n_jobs=NUM_CORES)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, total_ele, noise,
                                              nr_broken_ele, Rs, lambdas)
                                             for i in range(n))
            rms = np.array([item[0] for item in err])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i,
                                                       total_ele, noise,
                                                       nr_broken_ele, Rs,
                                                       lambdas)
                rms[i] = data
                point_error.append(error)
        point_error = np.array(point_error)
        toc = time.time() - tic
        print('time: ', toc)
        return rms, point_error

    def plot_error_map(self, point_error, ele_pos):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        point_error: numpy array
            Error of reconstruction calculated at every point of reconstruction
            space.
        ele_pos: numpy array
            Positions of electrodes.

        Returns
        -------
        None
        """
        ele_x, ele_y = ele_pos[0], ele_pos[1]
        error_mean = self.sigmoid_mean(point_error)
        plt.figure()
        plt.contourf(error_mean[:, :, 0])
        plt.scatter(ele_x, ele_y)
        plt.axis('equal')
        return


if __name__ == '__main__':
    print('Checking 1D')
    CSD_PROFILE = CSD.gauss_1d_mono
    ELE_LIMS = [0.1, 0.9]  # range of electrodes space
    TRUE_CSD_XLIMS = [0., 1.]
    k = VisibilityMap1D(h=0.25, R_init=0.3, ele_lims=ELE_LIMS,
                        true_csd_xlims=TRUE_CSD_XLIMS, sigma=0.3,
                        src_type='gauss', n_src_init=100, ext_x=0.1)
    rms, point_error = k.calculate_error_map(CSD_PROFILE, total_ele=32,
                                             Rs=np.arange(0.2, 0.5, 0.1))
    ele_pos = np.linspace(ELE_LIMS[0], ELE_LIMS[1], 32)

    print('Checking 2D')
    CSD_PROFILE = CSD.gauss_2d_small
    a = VisibilityMap2D(h=50., sigma=1., n_src_init=400)
    rms, point_error = a.calculate_error_map(CSD_PROFILE, total_ele=36)

    print('Checking 3D')
    CSD_PROFILE = CSD.gauss_3d_small
    a = VisibilityMap3D(h=50., sigma=1., n_src_init=729)
    a.calculate_error_map(CSD_PROFILE, total_ele=27)
