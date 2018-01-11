#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:09:20 2017

@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super

from builtins import range
from future import standard_library

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.ma as ma

from TestKCSD3D import TestKCSD3D
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD3D
from save_paths import where_to_save_results, where_to_save_source_code, \
    TIMESTR

standard_library.install_aliases()
__abs_file__ = os.path.abspath(__file__)

try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 4
    parallel_available = True
except ImportError:
    parallel_available = False


class ErrorMap3D(TestKCSD3D):
    """
    Class that produces error map for 3D CSD reconstruction
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        self.test_res = 256
        self.n = kwargs.get('n', 1)
        super(ErrorMap3D, self).__init__(csd_profile, csd_seed, **kwargs)
        self.calculate_error_map(csd_profile, **kwargs)
        return

    def make_reconstruction(self, csd_profile, csd_seed, **kwargs):
        '''
        Executes main method
        '''
        csd_x, csd_y, csd_z, true_csd = self.generate_csd(csd_profile,
                                                          csd_seed,
                                                          self.csd_xres,
                                                          self.csd_yres,
                                                          self.csd_zres)
        ele_pos, pots = self.electrode_config(csd_profile, csd_seed)
        pots = pots.reshape(len(pots), 1)
        kcsd = KCSD3D(ele_pos, pots, gdx=0.03, gdy=0.03, gdz=0.03,
                      h=50, sigma=1, xmax=1, xmin=0, ymax=1, ymin=0, zmax=1,
                      zmin=0, n_src_init=8000)
#        tic = time.time()
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        np.arange(0.08, 0.5, 0.025))
#        toc = time.time() - tic
        test_csd = csd_profile(kcsd.estm_x, kcsd.estm_y, kcsd.estm_z, csd_seed)
        if csd_seed == 0:
            np.save(self.path + '/estm_x.npy', kcsd.estm_x)
            np.save(self.path + '/estm_y.npy', kcsd.estm_y)
            np.save(self.path + '/estm_z.npy', kcsd.estm_z)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, :, 0])
#        u_svd, sigma, v_svd = self.svd(kcsd)
#        np.save(self.path + '/u_svd_test' + str(csd_seed) + '.npy', u_svd)
#        np.save(self.path + '/sigma_test' + str(csd_seed) + '.npy', sigma)
#        np.save(self.path + '/v_svd_test' + str(csd_seed) + '.npy', v_svd)
        return [rms, kcsd.R, kcsd.lambd], point_error

    def calculate_error_map(self, csd_profile, **kwargs):
        n = 150
        tic = time.time()
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i, noise='None',
                                              **kwargs)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            np.save(self.path + '/data.npy', data)
            rms = np.array([item[0] for item in data])
            np.save(self.path + '/rms.npy', rms)
            cv_r = np.array([item[1] for item in data])
            np.save(self.path + '/cv_r.npy', cv_r)
            cv_l = np.array([item[2] for item in data])
            np.save(self.path + '/cv_l.npy', cv_l)
            point_error = np.array([item[1] for item in err])
            np.save(self.path + '/point_error.npy', point_error)
        else:
            rms = np.zeros(n)
            for i in range(n):
                rms[i] = self.make_reconstruction(csd_profile, i, **kwargs)
        np.save(self.path + '/rms.npy', rms)
        toc = time.time() - tic
        print('time: ', toc)
#        self.plot_mean_error(point_error, 1, csd_profile)
#        self.mean_error_threshold(point_error, 1, threshold=1)
        return

    def sigmoid_mean(self, error):
        sig_error = 2*(1./(1 + np.exp((-error))) - 1/2.)
        error_mean = np.mean(sig_error, axis=0)
        return error_mean

    def plot_mean_error(self):
        point_error = np.load(self.path + '/point_error.npy')
        error_mean = self.sigmoid_mean(point_error)
        plt.figure()
        plt.contourf(error_mean[:, :, 0])
        plt.axis('equal')
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
    csd_profile = CSD.gauss_3d_small
    csd_seed = 10
    total_ele = 216
    a = ErrorMap3D(csd_profile, csd_seed, total_ele=total_ele, h=50.,
                   sigma=1., nr_basis=10650, config='regular', n=15)
