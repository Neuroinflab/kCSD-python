#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:15:49 2017

@author: mkowalska
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import range
from builtins import super
from future import standard_library

import time
import numpy as np
import os
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.mlab import griddata
import matplotlib.cm as cm

from TestKCSD import TestKCSD
from KCSD import KCSD3D
import csd_profile as CSD
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


class TestKCSD3D(TestKCSD):
    """
    TestKCSD3D - The 3D variant of tests for kCSD method.
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
        super(TestKCSD3D, self).__init__(dim=3, **kwargs)
        self.general_parameters(**kwargs)
        self.dimension_parameters(**kwargs)
        self.make_reconstruction(csd_profile, csd_seed, **kwargs)
        return

    def electrode_config(self, csd_profile, csd_seed, noise=None):
        """
        Produces electrodes positions and potentials measured at these points

        Parameters
        ----------
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            internal state of the random number generator
        noise: string
            determins if data contains noise

        Returns
        -------
        ele_pos: numpy array, shape (total_ele, 2)
            electrodes locations in 2D plane
        pots: numpy array, shape (total_ele, 1)
        """
        ele_x, ele_y, ele_z = self.generate_electrodes()
        csd_x, csd_y, csd_z, true_csd = self.generate_csd(csd_profile,
                                                          csd_seed)
        if self.config == 'broken':
            ele_pos = self.broken_electrode(10, 5)
            ele_x, ele_y, ele_z = ele_pos[:, 0], ele_pos[:, 1], ele_pos[0, 2]
        if parallel_available:
            pots = self.calculate_potential_3D_parallel(true_csd,
                                                        ele_x, ele_y, ele_z,
                                                        csd_x, csd_y, csd_z)
        else:
            pots = self.calculate_potential_3D(true_csd, ele_x, ele_y, ele_z,
                                               csd_x, csd_y, csd_z)
        if noise == 'True':
            pots = self.add_noise(csd_seed, pots, level=0.5)
        ele_pos = np.vstack((ele_x, ele_y, ele_z)).T     # Electrode configs
        num_ele = ele_pos.shape[0]
        print('Number of electrodes:', num_ele)
        return ele_pos, pots.reshape((len(ele_pos), 1))

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
        csd_x, csd_y, csd_z = np.mgrid[self.true_csd_xlims[0]:
                                       self.true_csd_xlims[1]:
                                       np.complex(0, self.csd_xres),
                                       self.true_csd_ylims[0]:
                                       self.true_csd_ylims[1]:
                                       np.complex(0, self.csd_yres),
                                       self.true_csd_zlims[0]:
                                       self.true_csd_zlims[1]:
                                       np.complex(0, self.csd_zres)]
        f = csd_profile(csd_x, csd_y, csd_z, seed=csd_seed)
        return csd_x, csd_y, csd_z, f

    def integrate_3D(self, x, y, z, csd, xlin, ylin, zlin,
                     X, Y, Z):
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

    def calculate_potential_3D(self, true_csd, ele_xx, ele_yy, ele_zz,
                               csd_x, csd_y, csd_z):
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
        xlin = csd_x[:, 0, 0]
        ylin = csd_y[0, :, 0]
        zlin = csd_z[0, 0, :]
        pots = np.zeros(len(ele_xx))
        print(ele_xx)
        for ii in range(len(ele_xx)):
            pots[ii] = self.integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                         true_csd,
                                         xlin, ylin, zlin,
                                         csd_x, csd_y, csd_z)
        pots /= 4*np.pi*self.sigma
        return pots

    def calculate_potential_3D_parallel(self, true_csd, ele_xx, ele_yy, ele_zz,
                                        csd_x, csd_y, csd_z):
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
        xlin = csd_x[:, 0, 0]
        ylin = csd_y[0, :, 0]
        zlin = csd_z[0, 0, :]
        pots = Parallel(n_jobs=num_cores)(delayed(self.integrate_3D)
                                          (ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                           true_csd,
                                           xlin, ylin, zlin,
                                           csd_x, csd_y, csd_z)
                                          for ii in range(len(ele_xx)))
        pots = np.array(pots)
        pots /= 4*np.pi*self.sigma
        return pots

    def make_reconstruction(self, csd_profile, csd_seed, **kwargs):
        """
        Main method, makes the whole kCSD reconstruction

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
        csd_x, csd_y, csd_z, true_csd = self.generate_csd(csd_profile,
                                                          csd_seed)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        print(max(ele_pos[:, 0]), min(ele_pos[:, 0]))
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD3D(ele_pos, pots, gdx=0.03, gdy=0.03, gdz=0.03,
                      h=50, sigma=1, xmax=1, xmin=0, ymax=1, ymin=0, zmax=1,
                      zmin=0, n_src_init=12200)
        tic = time.time()
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        np.arange(0.08, 0.5, 0.025))
        toc = time.time() - tic
        test_csd = csd_profile(kcsd.estm_x, kcsd.estm_y, kcsd.estm_z, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
#        point_error = self.calculate_point_error(test_csd,
#                                                 est_csd[:, :, :, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS: %0.2E; CV_Error: %0.2E; "\
                "Time: %0.2f" % (kcsd.lambd, kcsd.R, rms, kcsd.cv_error, toc)
        self.make_plot(csd_x, csd_y, csd_z, test_csd, kcsd, est_csd, ele_pos,
                       pots, rms, title)
        return

    def make_plot(self, csd_x, csd_y, csd_z, true_csd, kcsd, est_csd, ele_pos,
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
            ax.hold(True)
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
    total_ele = 216
    csd_seed = 20  # 0-49 are small sources, 50-99 are large sources
    csd_profile = CSD.gauss_3d_small
    tic = time.time()
    TestKCSD3D(csd_profile, csd_seed, total_ele=total_ele, h=50, sigma=1,
               xmax=1, xmin=0, ymax=1, ymin=0, zmax=1, zmin=0, config='regular')
    toc = time.time() - tic
    print('time', toc)
