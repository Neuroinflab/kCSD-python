"""
@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super

# from builtins import str
from builtins import range
# from builtins import object
from past.utils import old_div
from future import standard_library

import os
import sys
import numpy as np
from scipy.integrate import simps
# import time
import matplotlib.pyplot as plt

from TestKCSD import TestKCSD
sys.path.append('../../corelib')
from KCSD import KCSD1D
import csd_profile as CSD

# from KCSD_crossValid_ext import KCSD1D_electrode_test as test
from save_paths import where_to_save_results, where_to_save_source_code, \
    TIMESTR

standard_library.install_aliases()
__abs_file__ = os.path.abspath(__file__)


class TestKCSD1D(TestKCSD):
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
        super(TestKCSD1D, self).__init__(dim=1, **kwargs)
        self.general_parameters(**kwargs)
        self.dimension_parameters(**kwargs)
        self.make_reconstruction(csd_profile, csd_seed)
        return

    def generate_csd(self, csd_profile, csd_seed):
        """
        Gives CSD profile at the requested spatial location,
        at 'res' resolution

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator

        Returns
        -------
        csd_x: numpy array, shape (csd_xres)
            positions at x axis (where is the ground truth)
        true_csd: numpy array, shape (csd_xres)
            csd at csd_x positions
        """
        csd_at = np.linspace(self.true_csd_xlims[0], self.true_csd_xlims[1],
                             self.csd_xres)
        true_csd = csd_profile(csd_at, csd_seed)
        return csd_at, true_csd

    def calculate_potential(self, true_csd, csd_at, ele_pos):
        """
        Calculates potentials

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator
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
        Calculates integrals (potential values) according to Simpson rule in
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

    def make_reconstruction(self, csd_profile, csd_seed):
        """
        Main method, makes the whole kCSD reconstruction

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            internal state of the random number generator

        Returns
        -------
        rms: float
            error of reconstruction
        point_error: numpy array
            error of reconstruction calculated at every point of reconstruction
            space
        """
        csd_at, true_csd = self.generate_csd(csd_profile, csd_seed)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD1D(ele_pos, pots, src_type='gauss', sigma=0.3, h=0.25,
                      n_src_init=100, ext_x=0.1)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd)
        test_csd = csd_profile(kcsd.estm_x, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (kcsd.lambd,
                                                                kcsd.R, rms)
        self.make_plot(kcsd, csd_at, true_csd, ele_pos, pots, est_csd,
                       est_pot, title)
        self.svd(kcsd)
        self.picard_plot(kcsd, pots)
        point_error = self.calculate_point_error(test_csd, est_csd[:, 0])
        return rms, point_error

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
        fig.savefig(os.path.join(self.path + '/', title + '.png'))
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
    csd_profile = CSD.sin
    R_init = 0.23
    csd_seed = 2
    total_ele = 16  # [4, 8, 16, 32, 64, 128]
    nr_basis = 100
    ele_lims = [0.1, 0.9]  # range of electrodes space
    kcsd_lims = [0.1, 0.9]  # CSD estimation space range
    true_csd_xlims = [0., 1.]
    basis_lims = true_csd_xlims  # basis sources coverage
    csd_res = 100  # number of estimation points
    ELE_PLACEMENT = 'regular'  # 'fractal_2'  # 'random'
    ele_seed = 50

    k = TestKCSD1D(csd_profile, csd_seed, ele_seed=ele_seed,
                   total_ele=total_ele, nr_basis=nr_basis, h=0.25,
                   R_init=R_init, ele_xlims=ele_lims, kcsd_xlims=kcsd_lims,
                   basis_xlims=basis_lims, est_points=csd_res,
                   true_csd_xlims=true_csd_xlims, sigma=0.3, src_type='gauss',
                   n_src_init=nr_basis, ext_x=0.1, TIMESTR=TIMESTR,
                   path=where_to_save_results, config='regular')
