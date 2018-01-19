"""
@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super

from future import standard_library

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.ma as ma

from TestKCSD import ValidationClassKCSD1D, SpectralStructure
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD1D
from save_paths import where_to_save_results, where_to_save_source_code, \
    TIMESTR

standard_library.install_aliases()
__abs_file__ = os.path.abspath(__file__)

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
        self.calculate_error_map(csd_profile, csd_seed, **kwargs)
        return

    def calculate_error_map(self, csd_profile, csd_seed, **kwargs):
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
        test_res = np.linspace(self.basis_xlims[0], self.basis_xlims[1],
                               self.nr_basis)
        tic = time.time()
        R = self.R_init
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed(self.make_reconstruction)
                                             (csd_profile, [R, i], **kwargs)
                                             for i in test_res)
            rms = np.array([item[0] for item in err])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(len(test_res))
            point_error = np.zeros((len(test_res), self.est_xres))
            for index, i in enumerate(test_res):
                rms[index], point_error[index] = self.make_reconstruction(
                        csd_profile, [R, i], **kwargs)
        toc = time.time() - tic
        print('time', toc)
        self.plot_rms(rms, R, csd_profile, csd_seed)
        self.plot_mean_error(point_error, R, csd_profile, csd_seed)
        self.mean_error_threshold(point_error, R, csd_profile, csd_seed,
                                  threshold=1)
        return

    def make_reconstruction(self, csd_profile, csd_seed, **kwargs):
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
        chrg_pos_x, true_csd = self.generate_csd(csd_profile, csd_seed)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD1D(ele_pos, pots, src_type='gauss', sigma=0.3, h=0.25,
                      n_src_init=100, ext_x=0.1)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd)
        test_csd = csd_profile(kcsd.estm_x, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (kcsd.lambd,
                                                                kcsd.R, rms)
        self.make_plot(kcsd, chrg_pos_x, true_csd, ele_pos, pots, est_csd,
                       est_pot, title)
        ss = SpectralStructure(kcsd, self.path)
        return rms, point_error

    def plot_rms(self, rms, R, csd_profile, csd_seed):
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
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        fig = plt.figure(figsize=(10, 6))
        plt.title('Error plot for ground truth R=' + str(R))
        plt.plot(np.linspace(self.kcsd_xlims[0], self.kcsd_xlims[-1],
                             len(rms)), rms, 'b.', label='error')
        plt.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        plt.xlabel('Depth [mm]')
        plt.ylabel('RMS Error')
        plt.legend()
        save_as = 'RMS_map_R_' + str(R)
        fig.savefig(os.path.join(self.path, save_as + '.png'))
        plt.close()
        return

    def plot_mean_error(self, point_error, R, csd_profile, csd_seed):
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
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        mean_err = np.mean(point_error, axis=0)
        fig = plt.figure(figsize=(10, 6))
        plt.title('Mean point error for ground truth R=' + str(R))
        plt.plot(np.linspace(self.kcsd_xlims[0], self.kcsd_xlims[-1],
                             self.est_xres), mean_err, 'b.',
                 label='mean error')
        plt.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        plt.xlabel('Depth [mm]')
        plt.ylabel('RMS Error')
        plt.legend()
        save_as = 'Mean_point_error_R_' + str(R) + '_nr_tested_sources_' +\
            str(self.nr_basis)
        fig.savefig(os.path.join(self.path, save_as + '.png'))
        plt.close()
        return

    def mean_error_threshold(self, point_error, R, csd_profile, csd_seed,
                             threshold=1.):
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
        point_mask = ma.masked_array(point_error, point_error >= threshold)
        mean_mask = ma.mean(point_mask, axis=0)
        mean_nr = ma.count(point_mask, axis=0)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
#        mean_err = np.mean(point_error, axis=0)
        fig = plt.figure(figsize=(12, 7))
        fig.suptitle('Mean point error for ground truth R=' + str(R) +
                     ' calculated for values above the threshold=' +
                     str(threshold))
        ax1 = plt.subplot(121)
        ax1.plot(np.linspace(self.kcsd_xlims[0], self.kcsd_xlims[-1],
                             self.est_xres), mean_mask, 'b.',
                 label='thresholded mean error')
#        ax1.plot(np.linspace(self.kcsd_xlims[0], self.kcsd_xlims[-1],
#                             self.est_xres), mean_err, 'y.',
#                 label='mean error')
        ax1.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        ax1.set_xlabel('Depth [mm]')
        ax1.set_ylabel('RMS Error')
        ax1.legend()
        ax2 = plt.subplot(122)
        ax2.plot(np.linspace(self.kcsd_xlims[0], self.kcsd_xlims[-1],
                             self.est_xres), mean_nr, 'b.',
                 label='nr of elements in mean')
        ax2.plot(ele_pos, np.zeros(len(ele_pos)), 'o', color='black',
                 label='electrodes locations')
        ax2.set_xlabel('Depth [mm]')
        ax2.set_ylabel('Nr of counts')
        ax2.legend()
        save_as = 'Thresholded_Mean_point_error_R_' + str(R) + \
            '_nr_tested_sources_' + str(self.nr_basis)
        fig.savefig(os.path.join(self.path, save_as + '.png'))
        plt.close()
        return


def save_source_code(save_path, TIMESTR):
    """
    Wilson G., et al. (2014) Best Practices for Scientific Computing,
    PLoS Biol 12(1): e1001745
    """
    with open(save_path + 'source_code_' + str(TIMESTR), 'w') as sf:
        sf.write(open(__abs_file__).read())
    return


def makemydir(directory):
    """
    Creates directory if it doesn't exist
    """
    try:
        os.makedirs(directory)
    except OSError:
        pass
    os.chdir(directory)


if __name__ == '__main__':
    makemydir(where_to_save_source_code)
    save_source_code(where_to_save_source_code, TIMESTR)
    csd_profile = CSD.basis_gauss
    R_init = 0.3
    csd_seed = [R_init, 0.5]
    total_ele = 10  # [4, 8, 16, 32, 64, 128]
    nr_basis = 32
    ele_lims = [0.1, 0.9]  # range of electrodes space
    kcsd_lims = [0.1, 0.9]  # CSD estimation space range
    true_csd_xlims = [0., 1.]
    basis_lims = true_csd_xlims  # basis sources coverage
    csd_res = 100  # number of estimation points
    ELE_PLACEMENT = 'regular'  # 'fractal_2'  # 'random'
    ele_seed = 50

    k = ErrorMap1D(csd_profile, csd_seed, ele_seed=ele_seed,
                   total_ele=total_ele, nr_basis=nr_basis, h=0.25,
                   R_init=R_init, ele_xlims=ele_lims, kcsd_xlims=kcsd_lims,
                   basis_xlims=basis_lims, est_points=csd_res,
                   true_csd_xlims=true_csd_xlims, sigma=0.3, src_type='gauss',
                   n_src_init=nr_basis, ext_x=0.1, TIMESTR=TIMESTR,
                   path=where_to_save_results)
