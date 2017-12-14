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

from TestKCSD import TestKCSD
from KCSD3D import KCSD3D
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

        Returns
        -------
        None
        """
        super(TestKCSD3D, self).__init__(dim=3, **kwargs)
        self.general_parameters(**kwargs)
        self.dimension_parameters(**kwargs)
        self.make_reconstruction(csd_profile, csd_seed, **kwargs)
        return

    def electrode_config(self, csd_profile, csd_seed):
        """
        What is the configuration of electrode positions,
        between what and what positions
        """
        ele_x, ele_y, ele_z = self.generate_electrodes()
        csd_x, csd_y, csd_z, true_csd = self.generate_csd(csd_profile,
                                                          csd_seed)
        if parallel_available:
            pots = self.calculate_potential_3D_parallel(true_csd,
                                                        ele_x, ele_y, ele_z,
                                                        csd_x, csd_y, csd_z)
        else:
            pots = self.calculate_potential_3D(true_csd, ele_x, ele_y, ele_z,
                                               csd_x, csd_y, csd_z)
        ele_pos = np.vstack((ele_x, ele_y, ele_z)).T     # Electrode configs
        num_ele = ele_pos.shape[0]
        print('Number of electrodes:', num_ele)
        return ele_pos, pots.reshape((len(ele_pos), 1))

    def generate_csd(self, csd_profile, csd_seed,
                     res_x=50, res_y=50, res_z=50):
        """
        Gives CSD profile at the requested spatial location, at 'res'
        resolution
        """
        csd_x, csd_y, csd_z = np.mgrid[self.true_csd_xlims[0]:
                                       self.true_csd_xlims[1]:
                                       np.complex(0, res_x),
                                       self.true_csd_ylims[0]:
                                       self.true_csd_ylims[1]:
                                       np.complex(0, res_y),
                                       self.true_csd_zlims[0]:
                                       self.true_csd_zlims[1]:
                                       np.complex(0, res_z)]
        f = csd_profile(csd_x, csd_y, csd_z, seed=csd_seed)
        return csd_x, csd_y, csd_z, f

    def integrate_3D(self, x, y, z, csd, xlin, ylin, zlin,
                     X, Y, Z):
        """
        X,Y - parts of meshgrid - Mihav's implementation
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
        For Mihav's implementation to compute the LFP generated
        """
        xlin = csd_x[:, 0, 0]
        ylin = csd_y[0, :, 0]
        zlin = csd_z[0, 0, :]
        pots = np.zeros(len(ele_xx))
        tic = time.time()
        for ii in range(len(ele_xx)):
            pots[ii] = self.integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                         true_csd,
                                         xlin, ylin, zlin,
                                         csd_x, csd_y, csd_z)
#            print('Electrode:', ii)
        pots /= 4*np.pi*self.sigma
        toc = time.time() - tic
        print(toc, 'Total time taken - series, sims')
        return pots

    def calculate_potential_3D_parallel(self, true_csd, ele_xx, ele_yy, ele_zz,
                                        csd_x, csd_y, csd_z):
        """
        For Mihav's implementation to compute the LFP generated
        """
        xlin = csd_x[:, 0, 0]
        ylin = csd_y[0, :, 0]
        zlin = csd_z[0, 0, :]
#        tic = time.time()
        pots = Parallel(n_jobs=num_cores)(delayed(self.integrate_3D)
                                          (ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                           true_csd,
                                           xlin, ylin, zlin,
                                           csd_x, csd_y, csd_z)
                                          for ii in range(len(ele_xx)))
        pots = np.array(pots)
        pots /= 4*np.pi*self.sigma
#        toc = time.time() - tic
#        print toc, 'Total time taken - parallel, sims '
        return pots

    def make_reconstruction(self, csd_profile, csd_seed, **kwargs):
        csd_x, csd_y, csd_z, true_csd = self.generate_csd(csd_profile,
                                                          csd_seed,
                                                          self.csd_xres,
                                                          self.csd_yres)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD3D(ele_pos, pots, **kwargs)
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        np.arange(0.19, 0.3, 0.04))
        test_csd = csd_profile(kcsd.estm_x, kcsd.estm_y, kcsd.estm_z, csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
        print('rms', rms)
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, :, 0])
        print('point error', point_error)
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
    total_ele = 27
#    Normal run
    csd_seed = 20  # 0-49 are small sources, 50-99 are large sources
    csd_profile = CSD.gauss_3d_small
    tic = time.time()
#    tracker = SummaryTracker()
    TestKCSD3D(csd_profile, csd_seed, total_ele=total_ele, h=50., sigma=1.)
    toc = time.time() - tic
    print('time', toc)
#    tracker.print_diff()
