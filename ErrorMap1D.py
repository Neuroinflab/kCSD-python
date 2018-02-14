"""
@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

from ValidationClassKCSD import ValidationClassKCSD1D
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD1D

try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 1
    parallel_available = True
except ImportError:
    parallel_available = False


class ErrorMap1D(ValidationClassKCSD1D):

    def __init__(self, csd_profile, csd_seed, **kwargs):
        super(ErrorMap1D, self).__init__(csd_profile, csd_seed, **kwargs)
        self.calculate_error_map_random(csd_profile, csd_seed)
        return

    def calculate_error_map_random(self, csd_profile, csd_seed):
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
        self.csd_seed = csd_seed
        n = 100
        tic = time.time()
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction_random)
                                             (csd_profile, i)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            rms = np.array([item[0] for item in data])
            kcsd = np.array([item[1] for item in data])
            ele_pos = np.array([item[2] for item in data])
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
        self.plot_rms(rms, R, csd_profile, ele_pos[0, :, 0])
        self.plot_mean_error(point_error, kcsd.R, csd_profile,
                             ele_pos[0, :, 0])
        print('time: ', toc)
        return

    def make_reconstruction_random(self, csd_profile, csd_seed):
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

        kcsd = KCSD1D(ele_pos, pots, xmin=0., xmax=1., h=self.h,
                      sigma=self.sigma, n_src_init=self.n_src_init)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.3, 0.6, 0.05))
        test_csd = csd_profile(kcsd.estm_x, self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        title = 'csd_profile_' + csd_profile.__name__ + '_seed' +\
            str(csd_seed) + '_total_ele' + str(self.total_ele)
        return [rms, kcsd, ele_pos], point_error

    def calculate_error_map(self, csd_profile, csd_seed):
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
        test_res = np.linspace(self.true_csd_xlims[0] - self.ext_x,
                               self.true_csd_xlims[1] + self.ext_x,
                               self.n_src_init)
        tic = time.time()
        R = self.R_init
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed(self.make_reconstruction)
                                             (csd_profile, [R, i])
                                             for i in test_res)
            rms = np.array([item[0] for item in err])
            point_error = np.array([item[1] for item in err])
            ele_pos = np.array([item[2] for item in err])
        else:
            rms = np.zeros(len(test_res))
            point_error = np.zeros((len(test_res), self.est_xres))
            for index, i in enumerate(test_res):
                rms[index], point_error[index] = self.make_reconstruction(
                        csd_profile, [R, i])
        toc = time.time() - tic
        print('time', toc)
        self.plot_rms(rms, R, csd_profile, ele_pos[0, :, 0])
        self.plot_mean_error(point_error, R, csd_profile, ele_pos[0, :, 0])
        return rms, point_error

    def make_reconstruction(self, csd_profile, csd_seed):
        """
        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator

        Returns
        -------
        rms: float
            error of reconstruction
        """
        self.csd_seed = csd_seed
        chrg_pos_x, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile)
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD1D(ele_pos, pots, src_type='gauss', sigma=self.sigma,
                      h=self.h, n_src_init=self.n_src_init, ext_x=self.ext_x)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd)
        test_csd = csd_profile(kcsd.estm_x, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (kcsd.lambd,
                                                                kcsd.R, rms)
        self.make_plot(kcsd, chrg_pos_x, true_csd, ele_pos, pots, est_csd,
                       est_pot, title)
#        SpectralStructure(kcsd)
        return rms, point_error, ele_pos

    def plot_rms(self, rms, R, csd_profile, ele_pos):
        """
        Creates plot of rms error calculated for the whole estimation space

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
#        ele_pos, pots = self.electrode_config(csd_profile)
        plt.figure(figsize=(10, 6))
        plt.title('Error plot for ground truth R=' + str(R))
        plt.plot(np.linspace(self.ele_xlims[0], self.ele_xlims[-1],
                             len(rms)), rms, 'b.', label='rms error')
        plt.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        plt.xlabel('Depth [mm]')
        plt.ylabel('RMS Error')
        plt.legend()
        plt.show()
        return

    def plot_mean_error(self, point_error, R, csd_profile, ele_pos):
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
        mean_err = self.sigmoid_mean(point_error)
        plt.figure(figsize=(10, 6))
        plt.title('Sigmoidal mean point error for ground truth R=' + str(R))
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


if __name__ == '__main__':
    R_init = 0.3
    csd_profile = CSD.gauss_1d_mono
    csd_seed = 2
    total_ele = 10  # [4, 8, 16, 32, 64, 128]
    n_src_init = 100
    ele_lims = [0.1, 0.9]  # range of electrodes space
    true_csd_xlims = [0., 1.]

    k = ErrorMap1D(csd_profile, csd_seed,
                   total_ele=total_ele, h=0.25,
                   R_init=R_init, ele_xlims=ele_lims,
                   true_csd_xlims=true_csd_xlims, sigma=0.3, src_type='gauss',
                   n_src_init=n_src_init, ext_x=0.1)
