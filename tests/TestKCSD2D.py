#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:27:04 2017

@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import range
from builtins import super
from future import standard_library

import os
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.mlab import griddata
from matplotlib import colors as cl

from TestKCSD import TestKCSD
from KCSD2D import KCSD2D
import csd_profile as CSD
from save_paths import where_to_save_results, where_to_save_source_code, \
    TIMESTR

standard_library.install_aliases()
__abs_file__ = os.path.abspath(__file__)


class TestKCSD2D(TestKCSD):
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

        Returns
        -------
        None
        """
        super(TestKCSD2D, self).__init__(dim=2, **kwargs)
        self.general_parameters(**kwargs)
        self.dimension_parameters(**kwargs)
        err_map = kwargs.get('err_map', 'yes')
        if err_map == 'no':
            self.make_reconstruction(csd_profile, csd_seed, **kwargs)
        return

    def electrode_config(self, csd_profile, csd_seed, noise='None'):
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
            ele_x, ele_y = self.broken_electrode(10, 5)
        else:
            ele_x, ele_y = self.generate_electrodes()
        csd_x, csd_y, true_csd = self.generate_csd(csd_profile, csd_seed,
                                                   self.csd_xres,
                                                   self.csd_yres)
        pots = self.calculate_potential(true_csd, csd_x, csd_y,
                                        ele_x, ele_y)
        if noise == 'noise':
            pots = self.add_noise(csd_seed, pots, level=0.5)
        ele_pos = np.vstack((ele_x, ele_y)).T
        pots = pots.reshape((len(pots), 1))
        return ele_pos, pots

    def generate_csd(self, csd_profile, csd_seed, res_x=50, res_y=50):
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
        csd_x: numpy array, shape (res_x, res_y)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y)
            y coordinates of ground truth data
        f: numpy array, shape (res_x, res_y)
            y coordinates of ground truth data
            calculated csd at locations indicated by csd_x and csd_y

        """
        csd_x, csd_y = np.mgrid[self.true_csd_xlims[0]:self.true_csd_xlims[1]:
                                np.complex(0, res_x),
                                self.true_csd_ylims[0]:self.true_csd_ylims[1]:
                                np.complex(0, res_y)]
        f = csd_profile(csd_x, csd_y, csd_seed)
        return csd_x, csd_y, f

    def calculate_potential(self, true_csd, csd_x, csd_y, ele_x, ele_y):
        """
        For Mihav's implementation to compute the LFP generated

        Parameters
        ----------
        true_csd: numpy array, shape (res_x, res_y)
            ground truth data (true_csd)
        csd_x: numpy array, shape (res_x, res_y)
            x coordinates of ground truth data
        csd_y: numpy array, shape (res_x, res_y)
            y coordinates of ground truth data
        ele_xx: numpy array, shape (len(ele_pos.shape[0]))
            x coordinates of electrodes
        ele_yy: numpy array, shape (len(ele_pos.shape[0]))
            y coordinates of electrodes

        Returns
        -------
        pots: numpy array, shape (total_ele)
            calculated potentials
        """
        xlin = csd_x[:, 0]
        ylin = csd_y[0, :]
        pots = np.zeros(len(ele_x))
        for ii in range(len(ele_x)):
            pots[ii] = self.integrate_2D(ele_x[ii], ele_y[ii], true_csd,
                                         xlin, ylin, csd_x, csd_y)
        pots /= 2 * np.pi * self.sigma
        return pots

    def integrate_2D(self, x, y, true_csd, xlin, ylin, csd_x, csd_y):
        """
        X,Y - parts of meshgrid - Mihav's implementation

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
        X: numpy array, shape (res_x, res_y)
            full x coordinates of true_csd
        Y: numpy array, shape (res_x, res_y)
            full y coordinates of true_csd

        Returns
        -------
        F: float
            potential on a single electrode
        """
        Ny = ylin.shape[0]
        m = np.sqrt((x - csd_x)**2 + (y - csd_y)**2)  # construct 2-D integrand
        m[m < 0.0000001] = 0.0000001             # I increased acuracy
        y = np.arcsinh(2 * self.h / m) * true_csd            # corrected
        integral_1D = np.zeros(Ny)           # do a 1-D integral over every row
        for i in range(Ny):
            integral_1D[i] = simps(y[:, i], ylin)      # I changed the integral
        F = simps(integral_1D, xlin)         # then an integral over the result
        return F

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
        print('csd_seed: ', csd_seed)
        csd_x, csd_y, true_csd = self.generate_csd(csd_profile, csd_seed,
                                                   self.csd_xres,
                                                   self.csd_yres)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed,
                                              noise='None')
#        print(pots)
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD2D(ele_pos, pots, xmin=0., xmax=1., ymin=0.,
                      ymax=1., **kwargs)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        Rs=np.arange(0.3, 0.6, 0.05)) #np.arange(0.1, 0.7, 0.02))
        self.picard_plot(kcsd, pots)
        test_csd = csd_profile(kcsd.estm_x, kcsd.estm_y, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, 0])
        self.make_plot(csd_x, csd_y, test_csd, kcsd, est_csd, ele_pos, pots,
                       rms, csd_profile, csd_seed, kcsd.R, kcsd.lambd)
        self.svd(kcsd)
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, 0])
        self.plot_point_error(point_error, kcsd)
        return

    def plot_point_error(self, point_error, k):
        """
        Creates point error plot

        Parameters
        ----------
        point_error: numpy array
        k: object of the class

        Returns
        -------
        None"""

        fig = plt.figure(figsize=(10, 6))
        plt.contourf(k.estm_x, k.estm_y, point_error, cmap=cm.Greys)
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.colorbar()
        save_as = 'Point_error_map'
        fig.savefig(os.path.join(self.path, save_as + '.png'))
        plt.close()
        return

    def make_plot(self, csd_x, csd_y, true_csd, kcsd, est_csd, ele_pos, pots,
                  rms, csd_profile, csd_seed, R, lambd):
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
        title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (kcsd.lambd,
                                                                kcsd.R, rms)
        fig = plt.figure(figsize=(15, 7))
#        fig.suptitle('R=' + str(R) + ', Lambda=' + str(lambd) +
#                     ', RMS=' + str(rms))
        fig.suptitle(title)
        ax1 = plt.subplot(141, aspect='equal')
        t_max = np.max(np.abs(true_csd))
        levels = np.linspace(-1 * t_max, t_max, 16)
        im1 = ax1.contourf(csd_x, csd_y, true_csd, levels=levels,
                           cmap=cm.bwr)
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel('y [mm]')
        ax1.set_title('A) True CSD')
        plt.colorbar(im1, orientation='horizontal', format='%.2f')
        ax2 = plt.subplot(143, aspect='equal')
        mask = np.load('/home/mkowalska/Marta/xCSD/branches/kCSD-marta/refactored_tests/mask.npy')
        levels2 = np.linspace(0, 1, 10)
        t_max = np.max(np.abs(est_csd[:, :, 0]))
        levels_kcsd = np.linspace(-1 * t_max, t_max, 16)
        im2 = ax2.contourf(kcsd.estm_x, kcsd.estm_y, est_csd[:, :, 0],
                           levels=levels_kcsd, alpha=1, cmap=cm.bwr)
        im2b = ax2.contourf(kcsd.estm_x, kcsd.estm_y, mask, levels=levels2,
                            alpha=0.3, cmap='Greys')
#        ax2.set_xlabel('x [mm]')
        ax2.set_ylabel('y [mm]')
        ax2.set_xlim([0., 1.])
        ax2.set_ylim([0., 1.])
        ax2.set_title('C) kCSD with error mask')
        plt.colorbar(im2, orientation='horizontal', format='%.2f')
#        plt.colorbar(im2b, orientation='vertical',
#                     format='%.2f')
        v_max = np.max(np.abs(pots))
        levels_pot = np.linspace(-1 * v_max, v_max, 64)
        X, Y, Z = self.grid(ele_pos[:, 0], ele_pos[:, 1], pots)
        ax3 = plt.subplot(142, aspect='equal')
        im3 = plt.contourf(X, Y, Z, levels=levels_pot, cmap=cm.PRGn)
        plt.scatter(ele_pos[:, 0], ele_pos[:, 1], 10)
        ax3.set_xlim([0., 1.])
        ax3.set_ylim([0., 1.])
        ax3.set_title('B) Pots, Ele_pos')
        plt.colorbar(im3, orientation='horizontal', format='%.2f')

        ax4 = plt.subplot(144, aspect='equal')
        difference = true_csd-est_csd[:, :, 0]
        im4 = ax4.contourf(kcsd.estm_x, kcsd.estm_y, difference,
                           levels=levels, cmap='PuOr')
        im4b = ax4.contourf(kcsd.estm_x, kcsd.estm_y, mask, levels=levels2,
                            alpha=0.3, cmap='Greys')
        ax4.set_xlabel('x [mm]')
#        ax4.set_ylabel('y [mm]')
        ax4.set_title('D) True CSD - kCSD')
        plt.colorbar(im4, orientation='horizontal', format='%.2f')
        save_as = 'csd_profile_' + csd_profile.__name__ + '_seed' +\
            str(csd_seed) + '_total_ele' + str(self.total_ele)
        fig.savefig(os.path.join(self.path, save_as + '.png'))
#        plt.close()
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


def save_source_code(save_path, TIMESTR):
    """
    Wilson G., et al. (2014) Best Practices for Scientific Computing,
    PLoS Biol 12(1): e1001745

    Parameters
    ----------
    save_path: string
    TIMESTR: string

    Returns
    -------
    None
    """
    with open(save_path + 'source_code_' + str(TIMESTR), 'w') as sf:
        sf.write(open(__abs_file__).read())
    return


def makemydir(directory):
    """
    Creates directory if it doesn't exist

    Parameters
    ----------
    directory: string
    """
    try:
        os.makedirs(directory)
    except OSError:
        pass
    os.chdir(directory)


if __name__ == '__main__':
    makemydir(where_to_save_source_code)
    save_source_code(where_to_save_source_code, TIMESTR)
    csd_profile = CSD.gauss_2d_small
    csd_seed = 12
    total_ele = 36
    a = TestKCSD2D(csd_profile, csd_seed, total_ele=total_ele, h=50., sigma=1.,
                   config='regular', err_map='no', nr_basis=400)
