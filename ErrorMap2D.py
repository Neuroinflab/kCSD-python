"""
@author: mkowalska
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super

from builtins import range

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.ma as ma

from ValidationClassKCSD import ValidationClassKCSD2D, SpectralStructure
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD2D

try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 1
    parallel_available = True
except ImportError:
    parallel_available = False


class ErrorMap2D(ValidationClassKCSD2D):
    """
    Class that produces error map for 2D CSD reconstruction
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        self.n = kwargs.get('n', 1)
        super(ErrorMap2D, self).__init__(csd_profile, csd_seed, **kwargs)
        self.calculate_error_map_r(csd_profile)
        return

    def make_reconstruction(self, csd_profile, csd_seed, R, test_point):
        """
        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator
        R: float
        test_point: list

        Returns
        -------
        rms: float
            error of reconstruction
        """
        csd_at = np.mgrid[self.true_csd_xlims[0]:self.true_csd_xlims[1]:
                          np.complex(0, self.csd_xres),
                          self.true_csd_ylims[0]:self.true_csd_ylims[1]:
                          np.complex(0, self.csd_yres)]
        true_csd = csd_profile(csd_at, R, test_point[0], test_point[1],
                               'mono')

        ele_pos, pots = self.electrode_config_err(csd_profile, R,
                                                  test_point[0],
                                                  test_point[1], csd_seed,
                                                  noise='None', source='mono')
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD2D(ele_pos, pots, xmin=0., xmax=1., ymin=0., ymax=1.,
                      h=self.h, sigma=self.sigma, n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.1, 0.7, 0.02))
        test_csd = csd_profile(kcsd.estm_x, kcsd.estm_y, R, test_point[0],
                               test_point[1], 'mono', csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, 0])
        self.make_plot(csd_at, true_csd, kcsd, est_csd, ele_pos, pots,
                       rms, csd_profile, [R, test_point])
        return [rms, kcsd], point_error

    def make_reconstruction_random(self, csd_profile, csd_seed, noise='None'):
        """
        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator
        R: float
        test_point: list

        Returns
        -------
        rms: float
            error of reconstruction
        """
        self.csd_seed = csd_seed
        csd_at, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile)

        kcsd = KCSD2D(ele_pos, pots, xmin=0., xmax=1., ymin=0.,
                      ymax=1., h=self.h, sigma=self.sigma,
                      n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.3, 0.6, 0.05))
        test_csd = csd_profile([kcsd.estm_x, kcsd.estm_y], self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, 0])
        title = 'csd_profile_' + csd_profile.__name__ + '_seed' +\
            str(csd_seed) + '_total_ele' + str(self.total_ele)
        self.make_plot(csd_at, true_csd, kcsd, est_csd, ele_pos, pots,
                       rms, title)
        SpectralStructure(kcsd)
        return [rms, kcsd], point_error

    def calculate_error_map(self, csd_profile, csd_seed, test_res=256):
        test_x, test_y = np.mgrid[self.true_csd_xlims[0]:
                                  self.true_csd_xlims[1]:
                                  np.complex(0, int(np.sqrt(test_res))),
                                  self.true_csd_xlims[0]:
                                  self.true_csd_xlims[1]:
                                  np.complex(0, int(np.sqrt(test_res)))]
        test_points = np.vstack((test_x.flatten(), test_y.flatten())).T
        tic = time.time()
        for R in [self.R_init, self.R_init/2., self.R_init/4.,
                  3/4.*self.R_init, 3/2.*self.R_init]:
            if parallel_available:
                err = Parallel(n_jobs=num_cores)(delayed
                                                 (self.make_reconstruction)
                                                 (csd_profile, csd_seed, R,
                                                  test_points[i])
                                                 for i in range(len
                                                                (test_points)))
                data = np.array([item[0] for item in err])
                rms = np.array([item[0] for item in data])
                point_error = np.array([item[1] for item in err])
            else:
                rms = np.zeros(len(test_points))
                point_error = []
                for i in range(len(test_points)):
                    data, error = self.make_reconstruction(csd_profile,
                                                           csd_seed, R,
                                                           test_points[i])
                    rms[i] = data[0]
                    point_error.append(error)
            point_error = np.array(point_error)
            toc = time.time() - tic
            print('time: ', toc)
        return rms, point_error

    def calculate_error_map_r(self, csd_profile):
        n = 5
        tic = time.time()
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction_random)
                                             (csd_profile, i)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            rms = np.array([item[0] for item in data])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction_random(csd_profile, i)
                rms[i] = data[0]
                point_error.append(error)
        point_error = np.array(point_error)
        toc = time.time() - tic
        print('time: ', toc)
        self.plot_mean_error(point_error, 1, csd_profile)
        self.mean_error_threshold(point_error, 1, threshold=1)
        return rms, point_error

    def electrode_config_err(self, csd_profile, R, test_x, test_y, csd_seed,
                             noise='None', source='mono'):
        """
        Parameters
        ----------
        None

        Returns
        -------
        ele_pos: numpy array, shape (total_ele, 2)
            electrodes locations in 2D plane
        pots: numpy array, shape (total_ele, 1)
        """
        if self.config == 'broken':
            ele_x, ele_y = self.broken_electrode(10, self.n)
        else:
            ele_x, ele_y = self.generate_electrodes()
        csd_at, true_csd = self.generate_csd_err(csd_profile, R, test_x,
                                                 test_y, csd_seed,
                                                 self.csd_xres,
                                                 self.csd_yres,
                                                 source='mono')
        pots = self.calculate_potential(true_csd, csd_at, ele_x, ele_y)
        if noise == 'noise':
            pots = self.add_noise(csd_seed, pots, level=0.5)
        ele_pos = np.vstack((ele_x, ele_y)).T
        pots = pots.reshape((len(pots), 1))
        return ele_pos, pots

    def generate_csd_err(self, csd_profile, R, test_x, test_y, csd_seed,
                         res_x=50, res_y=50, source='mono'):
        csd_at = np.mgrid[self.true_csd_xlims[0]:self.true_csd_xlims[1]:
                          np.complex(0, res_x),
                          self.true_csd_ylims[0]:self.true_csd_ylims[1]:
                          np.complex(0, res_y)]
        f = csd_profile(csd_at, R, test_x, test_y, source)
        return csd_at, f

    def plot_error_map(self, rms, R, test_x, test_y):
        """
        Creates error map plot

        Parameters
        ----------
        rms
        test_x
        test_y

        Returns
        -------
        None
        """
        if self.config == 'broken':
            ele_x, ele_y = self.broken_electrode(10, self.n)
        else:
            ele_x, ele_y = self.generate_electrodes()
        rms = rms.reshape([test_x.shape[0], test_x.shape[1]])
        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        rms_max = np.max(np.abs(rms))
        levels = np.linspace(0, rms_max, 15)
        im = ax.contourf(test_x, test_y, rms, levels=levels, cmap='Greys')
        plt.colorbar(im)
        plt.scatter(ele_x, ele_y)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_xlim([0., 1.])
        ax.set_ylim([0., 1.])
        ax.set_title('Error plot for ground truth R=' + str(R))
        plt.show()
        return

    def plot_mean_error(self, point_error, R, csd_profile):
        """
        Creates error map plot

        Parameters
        ----------
        rms
        test_x
        test_y

        Returns
        -------
        None
        """
        if self.config == 'broken':
            ele_x, ele_y = self.broken_electrode(10, self.n)
        else:
            ele_x, ele_y = self.generate_electrodes()
#        x, y = np.mgrid[np.min(ele_x):np.max(ele_x):
#                        np.complex(0, self.est_xres),
#                        np.min(ele_y):np.max(ele_y):
#                        np.complex(0, self.est_yres)]
        x, y = np.mgrid[0:1:
                        np.complex(0, self.est_xres),
                        0:1:
                        np.complex(0, self.est_yres)]
        mean_err = np.mean(point_error, axis=0)
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111, aspect='equal')
        levels = np.linspace(0, 1., 15)
        im = ax.contourf(x, y, mean_err, levels=levels, cmap='Greys')
        plt.colorbar(im)
        plt.scatter(ele_x, ele_y)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_xlim([0., 1.])
        ax.set_ylim([0., 1.])
#        ax.set_title('Mean point error for ground truth R=' + str(R))
        plt.show()
        return

    def mean_error_threshold(self, point_error, R, threshold=1.):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        rms: numpy array
        R: float
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator

        Returns
        -------
        None
        """
        point_mask = ma.masked_array(point_error, point_error > threshold)
        mean_mask = ma.mean(point_mask, axis=0)
        mean_nr = ma.count(point_mask, axis=0)
        if self.config == 'broken':
            ele_x, ele_y = self.broken_electrode(10, self.n)
        else:
            ele_x, ele_y = self.generate_electrodes()
#        x, y = np.mgrid[np.min(ele_x):np.max(ele_x):
#                        np.complex(0, self.est_xres),
#                        np.min(ele_y):np.max(ele_y):
#                        np.complex(0, self.est_yres)]
        x, y = np.mgrid[0:1:
                        np.complex(0, self.est_xres),
                        0:1:
                        np.complex(0, self.est_yres)]
        plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(121, aspect='equal')
        levels = np.linspace(0, 1., 15)
        im = ax1.contourf(x, y, mean_mask, levels=levels, cmap='Greys')
        plt.colorbar(im, fraction=0.046, pad=0.06)
        plt.scatter(ele_x, ele_y)
        ax1.set_xlabel('Depth x [mm]')
        ax1.set_ylabel('Depth y [mm]')
#        ax1.set_title('Mean point error for random ground truth sources' +
#                      ' \n calculated for values below \n the threshold=' +
#                      str(threshold))
        ax1.legend()
        ax2 = plt.subplot(122, aspect='equal')
        levels = np.linspace(0, 100., 15)  # np.arange(np.max(mean_nr) + 1)
        im2 = ax2.contourf(x, y, mean_nr, levels=levels, cmap='Greys')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.scatter(ele_x, ele_y)
        ax2.set_xlabel('Depth x [mm]')
#        ax2.set_ylabel('Depth y [mm]')
#        ax2.set_title('Thresholded point error density')
        ax2.legend()
        plt.show()
        return


if __name__ == '__main__':
    csd_profile = CSD.gauss_2d_small
    csd_seed = 10
    total_ele = 36
    a = ErrorMap2D(csd_profile, csd_seed, total_ele=total_ele, h=50.,
                   sigma=1., n_src_init=400, config='regular', n=15)
