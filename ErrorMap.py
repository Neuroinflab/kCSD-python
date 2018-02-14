#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from ValidationClassKCSD import ValidationClassKCSD1D, ValidationClassKCSD2D, \
                                ValidationClassKCSD3D
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD1D, KCSD2D, KCSD3D

try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 1
    parallel_available = False
except ImportError:
    parallel_available = False


class ErrorMap1D(ValidationClassKCSD1D):

    def __init__(self, csd_profile, csd_seed, **kwargs):
        super(ErrorMap1D, self).__init__(csd_profile, csd_seed, **kwargs)
#        self.calculate_error_map(csd_profile)
        return

    def calculate_error_map(self, csd_profile, n=100, noise=None,
                            nr_broken_ele=0):
        """
        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator

        Returns
        -------
        None
        """
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, noise,
                                              nr_broken_ele)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            rms = np.array([item[0] for item in data])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i, noise,
                                                       nr_broken_ele)
                rms[i] = data[0]
                point_error.append(error)
        point_error = np.array(point_error)
        return rms, point_error

    def make_reconstruction(self, csd_profile, csd_seed, noise=None,
                            nr_broken_ele=0):
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
        ele_pos, pots = self.electrode_config(csd_profile, noise,
                                              nr_broken_ele)

        kcsd = KCSD1D(ele_pos, pots, xmin=0., xmax=1., h=self.h,
                      sigma=self.sigma, n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.3, 0.6, 0.1))
        test_csd = csd_profile(kcsd.estm_x, self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        return [rms, kcsd, ele_pos], point_error

    def plot_mean_error(self, point_error, ele_pos):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        point_error: numpy array

        Returns
        -------
        None
        """
        mean_err = self.sigmoid_mean(point_error)
        plt.figure(figsize=(10, 6))
        plt.title('Sigmoidal mean point error for random sources')
        plt.plot(np.linspace(self.ele_xlims[0], self.ele_xlims[-1],
                             self.est_xres), mean_err, 'b.',
                 label='mean error')
        plt.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        plt.xlabel('Depth [mm]')
        plt.ylabel('RMS Error')
        plt.legend()
        plt.show()
        return


class ErrorMap2D(ValidationClassKCSD2D):
    """
    Class that produces error map for 2D CSD reconstruction
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        super(ErrorMap2D, self).__init__(csd_profile, csd_seed, **kwargs)
#        self.calculate_error_map(csd_profile, noise='noise')
        return

    def make_reconstruction(self, csd_profile, csd_seed, noise=None,
                            nr_broken_ele=0):
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
        ele_pos, pots = self.electrode_config(csd_profile, noise,
                                              nr_broken_ele)

        kcsd = KCSD2D(ele_pos, pots, xmin=0., xmax=1., ymin=0.,
                      ymax=1., h=self.h, sigma=self.sigma,
                      n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.3, 0.6, 0.1))
        test_csd = csd_profile([kcsd.estm_x, kcsd.estm_y], self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, 0])
        return [rms, kcsd], point_error

    def calculate_error_map(self, csd_profile, n=100,  noise=None,
                            nr_broken_ele=0):
        tic = time.time()
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, noise,
                                              nr_broken_ele)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            rms = np.array([item[0] for item in data])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i, noise,
                                                       nr_broken_ele)
                rms[i] = data[0]
                point_error.append(error)
        point_error = np.array(point_error)
        toc = time.time() - tic
        print('time: ', toc)
        self.plot_mean_error(point_error, nr_broken_ele)
        return rms, point_error

    def plot_mean_error(self, point_error, nr_broken_ele=0):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        point_error: numpy array

        Returns
        -------
        None
        """
        if self.config == 'broken':
            ele_x, ele_y = self.broken_electrode(10, nr_broken_ele)
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


class ErrorMap3D(ValidationClassKCSD3D):
    """
    Class that produces error map for 3D CSD reconstruction
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        super(ErrorMap3D, self).__init__(csd_profile, csd_seed, **kwargs)
#        self.calculate_error_map(csd_profile)
        return

    def make_reconstruction(self, csd_profile, csd_seed, noise=None,
                            nr_broken_ele=0):
        '''
        Executes main method
        '''
        self.csd_seed = csd_seed
        csd_at, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile, noise,
                                              nr_broken_ele)
        kcsd = KCSD3D(ele_pos, pots, gdx=0.035, gdy=0.035, gdz=0.035,
                      h=self.h, sigma=self.sigma, xmax=1, xmin=0, ymax=1,
                      ymin=0, zmax=1, zmin=0, n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        np.arange(0.1, 0.4, 0.1))
        test_csd = csd_profile([kcsd.estm_x, kcsd.estm_y, kcsd.estm_z],
                               self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, :, 0])
        return [rms, kcsd], point_error

    def calculate_error_map(self, csd_profile, n=5, noise=None,
                            nr_broken_ele=0):
        tic = time.time()
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, noise,
                                              nr_broken_ele)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            rms = np.array([item[0] for item in data])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i, noise,
                                                       nr_broken_ele)
                rms[i] = data[0]
                point_error.append(error)
        point_error = np.array(point_error)
        toc = time.time() - tic
        print('time: ', toc)
        return rms, point_error

    def plot_mean_error(self):
        point_error = np.load('/point_error.npy')
        error_mean = self.sigmoid_mean(point_error)
        plt.figure()
        plt.contourf(error_mean[:, :, 0])
        plt.axis('equal')
        return


if __name__ == '__main__':
    print('Checking 1D')
    R_init = 0.3
    csd_profile = CSD.gauss_1d_mono
    csd_seed = 2
    ele_lims = [0.1, 0.9]  # range of electrodes space
    true_csd_xlims = [0., 1.]
    k = ErrorMap1D(csd_profile, csd_seed, total_ele=10, h=0.25,
                   R_init=R_init, ele_xlims=ele_lims,
                   true_csd_xlims=true_csd_xlims, sigma=0.3, src_type='gauss',
                   n_src_init=100, ext_x=0.1)

    print('Checking 2D')
    csd_profile = CSD.gauss_2d_small
    csd_seed = 10
    a = ErrorMap2D(csd_profile, csd_seed, total_ele=36, h=50.,
                   sigma=1., n_src_init=400, config='regular', n=15)

    print('Checking 3D')
    csd_profile = CSD.gauss_3d_small
    csd_seed = 10
    total_ele = 27
    a = ErrorMap3D(csd_profile, csd_seed, total_ele=total_ele, h=50.,
                   sigma=1., n_src_init=729, config='regular')
