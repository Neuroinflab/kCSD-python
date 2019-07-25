"""
@author: mkowalska
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import range

import numpy as np
from kcsd import csd_profile as CSD
from kcsd import ValidateKCSD2D
from figure_properties import *

try:
    from joblib import Parallel, delayed
    import multiprocessing
    NUM_CORES = multiprocessing.cpu_count() - 1
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


def set_axis(ax, letter=None):
    """
    Formats the plot's caption.

    Parameters
    ----------
    ax: Axes object.
    x: float
        X-position of caption.
    y: float
        Y-position of caption.
    letter: string
        Caption of the plot.
        Default: None.

    Returns
    -------
    ax: modyfied Axes object.
    """
    ax.text(
        -0.05,
        1.05,
        letter,
        fontsize=20,
        weight='bold',
        transform=ax.transAxes)
    return ax


def make_reconstruction(KK, csd_profile, csd_seed, total_ele,
                        ele_lims=None, noise=0, nr_broken_ele=None,
                        Rs=None, lambdas=None, method='cross-validation'):
    """
    Main method, makes the whole kCSD reconstruction.

    Parameters
    ----------
    KK: instance of the class
        Instance of ValidateKCSD2D class.
    csd_profile: function
        Function to produce csd profile.
    csd_seed: int
        Seed for random generator to choose random CSD profile.
    total_ele: int
        Number of electrodes.
    ele_lims: list
        Electrodes limits.
        Default: None.
    noise: float
        Determines the level of noise in the data.
        Default: 0.
    nr_broken_ele: int
        How many electrodes are broken (excluded from analysis)
        Default: None.
    Rs: numpy 1D array
        Basis source parameter for crossvalidation.
        Default: None.
    lambdas: numpy 1D array
        Regularization parameter for crossvalidation.
        Default: None.
    method: string
        Determines the method of regularization.
        Default: cross-validation.

    Returns
    -------
    
    """
    csd_at, true_csd = KK.generate_csd(csd_profile, csd_seed)
    ele_pos, pots = KK.electrode_config(csd_profile, csd_seed, total_ele,
                                        ele_lims, KK.h, KK.sigma,
                                        noise, nr_broken_ele)
    k, est_csd = KK.do_kcsd(pots, ele_pos, method=method, Rs=Rs,
                            lambdas=lambdas)
    err = point_errors(true_csd, est_csd)
    return (k.k_pot, k.k_interp_cross, k.lambd, csd_at, true_csd, ele_pos,
            pots, est_csd, err)


def point_errors(true_csd, est_csd):
    true_csd_r = true_csd.reshape(true_csd.size, 1)
    est_csd_r = est_csd.reshape(est_csd.size, 1)
    epsilon = np.linalg.norm(true_csd_r)/np.max(abs(true_csd_r))
    err_r = abs(est_csd_r/(np.linalg.norm(est_csd_r)) -
                true_csd_r/(np.linalg.norm(true_csd_r)))
    err_r *= epsilon
    err = err_r.reshape(true_csd.shape)
    return err


if __name__ == '__main__':
    CSD_PROFILE = CSD.gauss_2d_small
    CSD_SEED = 16
    ELE_LIMS = [0.118, 0.882]  # range of electrodes space
    method = 'cross-validation'
    Rs = np.linspace(0.01, 0.15, 15)
    lambdas = None
    noise = 0
    n = 100

    KK = ValidateKCSD2D(CSD_SEED, h=50., sigma=1., n_src_init=1000,
                        est_xres=0.01, est_yres=0.01, ele_lims=ELE_LIMS)

    if PARALLEL_AVAILABLE:
        err = Parallel(n_jobs=NUM_CORES)(delayed
                                         (make_reconstruction)
                                         (KK, CSD_PROFILE, i, total_ele=9,
                                          noise=noise, Rs=Rs,
                                          lambdas=lambdas, method=method)
                                         for i in range(n))
        k_pot_n = np.array([item[0] for item in err])
        k_interp_cross_n = np.array([item[1] for item in err])
        lambd_n = np.array([item[2] for item in err])
        csd_at = np.array([item[3] for item in err])[0]
        true_csd_n = np.array([item[4] for item in err])
        ele_pos = np.array([item[5] for item in err])[0]
        pots_n = np.array([item[6] for item in err])
        est_csd_n = np.array([item[7] for item in err])
        error_n = np.array([item[8] for item in err])
    else:
        k_n = []
        true_csd_n = []
        pots_n = []
        est_csd_n = []
        error_n = []
        for i in range(n):
            print(i)
            (k_pot, k_interp_cross, lambd, csd_at, true_csd, ele_pos, pots,
             est_csd, err) = make_reconstruction(KK, CSD_PROFILE, i,
                                                 total_ele=9,
                                                 noise=noise,
                                                 Rs=Rs,
                                                 lambdas=lambdas,
                                                 method=method)
            k_pot_n.append(k_pot)
            k_interp_cross_n.append(k_interp_cross)
            lambd_n.append(lambd)
            true_csd_n.append(true_csd)
            pots_n.append(pots)
            est_csd_n.append(est_csd)
            error_n.append(err)
    data = {}
    data['k_pot_n'] = k_pot_n
    data['k_interp_cross_n'] = k_interp_cross_n
    data['lambd_n'] = lambd_n
    data['csd_at'] = csd_at
    data['true_csd_n'] = true_csd_n
    data['ele_pos'] = ele_pos
    data['pots'] = pots_n
    data['est_csd'] = est_csd_n
    data['error'] = error_n
    np.savez('data_small_100.npz', data=data)
