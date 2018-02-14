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

from ValidationClassKCSD import ValidationClassKCSD3D, SpectralStructure
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD3D

try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 4
    parallel_available = True
except ImportError:
    parallel_available = False


class ErrorMap3D(ValidationClassKCSD3D):
    """
    Class that produces error map for 3D CSD reconstruction
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        super(ErrorMap3D, self).__init__(csd_profile, csd_seed, **kwargs)
        self.calculate_error_map(csd_profile)
        return

    def make_reconstruction(self, csd_profile, csd_seed):
        '''
        Executes main method
        '''
        self.csd_seed = csd_seed
        csd_at, true_csd = self.generate_csd(csd_profile)
        ele_pos, pots = self.electrode_config(csd_profile)
        kcsd = KCSD3D(ele_pos, pots, gdx=0.035, gdy=0.035, gdz=0.035,
                      h=self.h, sigma=self.sigma, xmax=1, xmin=0, ymax=1,
                      ymin=0, zmax=1, zmin=0, n_src_init=self.n_src_init)
        tic = time.time()
        est_csd, est_pot = self.do_kcsd(ele_pos, pots, kcsd,
                                        np.arange(0.15, 0.45, 0.025))
        toc = time.time() - tic
        test_csd = csd_profile([kcsd.estm_x, kcsd.estm_y, kcsd.estm_z],
                               self.csd_seed)
        rms = self.calculate_rms(test_csd, est_csd[:, :, :, 0])
        point_error = self.calculate_point_error(test_csd, est_csd[:, :, :, 0])
        title = "Lambda: %0.2E; R: %0.2f; RMS: %0.2E; CV_Error: %0.2E; "\
                "Time: %0.2f" % (kcsd.lambd, kcsd.R, rms, kcsd.cv_error, toc)
        self.make_plot(csd_at, test_csd, kcsd, est_csd, ele_pos, pots, rms,
                       title)
#        SpectralStructure(kcsd)
        return [rms, kcsd], point_error

    def calculate_error_map(self, csd_profile):
        n = 5
        tic = time.time()
        if parallel_available:
            err = Parallel(n_jobs=num_cores)(delayed
                                             (self.make_reconstruction)
                                             (csd_profile, i)
                                             for i in range(n))
            data = np.array([item[0] for item in err])
            rms = np.array([item[0] for item in data])
            point_error = np.array([item[1] for item in err])
        else:
            rms = np.zeros(n)
            point_error = []
            for i in range(n):
                data, error = self.make_reconstruction(csd_profile, i)
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
    csd_profile = CSD.gauss_3d_small
    csd_seed = 10
    total_ele = 27
    a = ErrorMap3D(csd_profile, csd_seed, total_ele=total_ele, h=50.,
                   sigma=1., n_src_init=1000, config='regular')
