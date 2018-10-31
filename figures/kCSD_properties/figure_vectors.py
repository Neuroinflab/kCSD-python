"""
@author: mkowalska
"""
import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import time

from kcsd import ValidateKCSD, ValidateKCSD1D, SpectralStructure, KCSD1D
import targeted_basis as tb
import reconstruction_stability as rs

__abs_file__ = os.path.abspath(__file__)


def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

BLACK     = _html(  0,   0,   0)
ORANGE    = _html(230, 159,   0)
SKY_BLUE  = _html( 86, 180, 233)
GREEN     = _html(  0, 158, 115)
YELLOW    = _html(240, 228,  66)
BLUE      = _html(  0, 114, 178)
VERMILION = _html(213,  94,   0)
PURPLE    = _html(204, 121, 167)

#def plot_k_interp_cross_v2(k_icross, eigenvectors, save_path, n_src, title):
#    """
#    Creates plot of product of cross kernel vectors and eigenvectors for
#    different number of basis sources
#
#    Parameters
#    ----------
#    k_icross: numpy array
#        List of cross kernel matrixes for different number of basis sources.
#    eigenvectors: numpy array
#        Eigenvectors of k_pot matrix.
#    save_path: string
#        Directory.
#    n_src: list
#        List of number of basis sources.
#
#    Returns
#    -------
#    None
#    """
#    fig = plt.figure(figsize=(15, 12))
#    gridspec.GridSpec(7, 6)
#    markers = ['^', '.', '*', 'x', ',']
#    linestyles = ['-', '--', '-.', ':']
#    #    plt.suptitle('Vectors of cross kernel and eigenvectors product for '
#    #                 'different number of basis sources')
#    plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3)
#    plt.plot(N_SRC)
#    plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3)
#    plt.plot(N_SRC)
#    #    for i in range(k_icross[0].shape[1]):
#    #        plt.subplot(int(k_icross[0].shape[1]/3), 6, (i + 6 + 1, (i + 6 + 2)))
#    #        for idx, j in enumerate(n_src):
#    #            plt.plot(np.dot(k_icross[idx], eigenvectors[idx, :, i]),
#    #                     linestyle=linestyles[idx],
#    #                     marker=markers[idx], label='M='+str(j))
#    #            plt.title(r'$\tilde{K}*v_' + str(i) + '$')
#    #            plt.locator_params(axis='y', nbins=5)
#    ##        plt.ylabel(r'$\tilde{K}*v_' + str(i) + '$')
#    #    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False,
#    #               shadow=False, ncol=len(n_src))
#    plt.xlabel('Number of estimation points')
#    plt.tight_layout()
#    plt.show()
#    save_path = save_path + '/cross_kernel'
#    tb.makemydir(save_path)
#    save_as = (save_path + '/cross_kernel_eigenvector_product_for_different_M')
#    fig.savefig(os.path.join(save_path, save_as+'.png'))
#    plt.close()

#    fig = plt.figure(figsize=(15, 15))
##    fig.suptitle('Vectors of cross kernel and eigenvectors product')
#    for i in range(eigenvectors.shape[0]):
#        plt.subplot(int(k_icross.shape[1]/3), 3, i + 1)
#        plt.plot(np.dot(k_icross, eigenvectors[:, i]), '--',
#                 marker='.')
#        plt.title(r'$\tilde{K}*v_' + str(i + 1) + '$')
##        plt.ylabel('Product K~V')
#    plt.xlabel('Number of estimation points')
#    fig.tight_layout()
#    plt.show()
#    save_path = save_path + '/cross_kernel'
#    tb.makemydir(save_path)
#    save_as = (save_path + '/cross_kernel_eigenvector_product' + title)
#    fig.savefig(os.path.join(save_path, save_as+'.png'))
#    plt.close()

def stability_M(csd_profile, csd_seed, n_src, ele_lims, true_csd_xlims,
                total_ele, ele_pos, pots):
    """
    Investigates stability of reconstruction for different number of basis
    sources

    Parameters
    ----------
    csd_profile: function
        Function to produce csd profile.
    csd_seed: int
        Seed for random generator to choose random CSD profile.
    n_src: int
        Number of basis sources.
    ele_lims: list
        Boundaries for electrodes placement.
    true_csd_xlims: list
        Boundaries for ground truth space.
    total_ele: int
        Number of electrodes.
    Returns
    -------
    obj_all: class object
    rms: float
        Normalized error of reconstruction.
    point_error_all: numpy array
        Normalized error of reconstruction calculated separetly at every point
        point of estimation space.
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    eigenvectors: numpy array
        Eigen vectors of k_pot matrix.
    """
    obj_all = []
    rms = np.zeros((len(n_src)))
    point_error_all = []
    eigenvectors = np.zeros((len(n_src), total_ele, total_ele))
    eigenvalues = np.zeros((len(n_src), total_ele))
    for i, value in enumerate(n_src):
        
#        KK = ValidateKCSD1D(csd_seed, n_src_init=value, R_init=0.23,
#                            ele_lims=ele_lims, true_csd_xlims=true_csd_xlims,
#                            sigma=0.3, h=0.25, src_type='gauss', est_xres=0.01)
#        obj, rms[i], point_error = KK.make_reconstruction(csd_profile,
#                                                          csd_seed,
#                                                          total_ele=total_ele,
#                                                          noise=0,
#                                                          Rs=np.arange(0.2,
#                                                                       0.5,
#                                                                       0.1))
        pots = pots.reshape((len(ele_pos), 1))
        obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=0.3, h=0.25, gdx=0.01,
                     n_src_init=n_src[i], ext_x=0, xmin=0, xmax=1)
        obj.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
#        est_csd = obj.values('CSD')
#        test_csd = csd_profile(obj.estm_x, [R, MU])
#        rms = val.calculate_rms(test_csd, est_csd)
        ss = SpectralStructure(obj)
        eigenvectors[i], eigenvalues[i] = ss.evd()

        obj_all.append(obj)
    return obj_all, eigenvalues, eigenvectors



home = expanduser('~')
DAY = datetime.datetime.now()
DAY = DAY.strftime('%Y%m%d')
TIMESTR = time.strftime("%H%M%S")
SAVE_PATH = home + "/kCSD_results/" + DAY + '/' + TIMESTR
tb.makemydir(SAVE_PATH)
tb.save_source_code(SAVE_PATH, time.strftime("%Y%m%d-%H%M%S"))

CSD_SEED = 15
N_SRC = [2, 8, 16, 512]
ELE_LIMS = [0, 1.]  # range of electrodes space
TRUE_CSD_XLIMS = [0., 1.]
TOTAL_ELE = 12

R = 0.2
MU = 0.25
csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                        TRUE_CSD_XLIMS, R, MU,
                                                        TOTAL_ELE, ELE_LIMS)
title = 'A_basis_lims_0_1'
obj_all = []
k_all = []
k_interp_cross_list = []
eigenvectors = np.zeros([len(N_SRC), TOTAL_ELE, TOTAL_ELE])
for i in range(len(N_SRC)):
    obj, k = tb.targeted_basis(val, csd_at, true_csd, ele_pos, pots, N_SRC[i],
                               R, MU, TRUE_CSD_XLIMS, ELE_LIMS, title)
    k_interp_cross_list.append(obj.k_interp_cross)
    ss = SpectralStructure(obj)
    eigenvectors[i], eigenvalues = ss.evd()

OBJ, eigenval, eigenvec = stability_M(tb.csd_profile, CSD_SEED, N_SRC,
                                      ELE_LIMS, TRUE_CSD_XLIMS, TOTAL_ELE,
                                      ele_pos, pots)
n_src_M = [2, 4, 8, 16, 32, 64, 128, 256, 512]
OBJ_M, eigenval_M, eigenvec_M = stability_M(tb.csd_profile, CSD_SEED, n_src_M,
                                      ELE_LIMS, TRUE_CSD_XLIMS, TOTAL_ELE,
                                      ele_pos, pots)

### FIGURE
colors = [BLUE, ORANGE, GREEN, PURPLE, VERMILION, SKY_BLUE, YELLOW, BLACK]
plt_cord = [(3, 0), (3, 2), (3, 4), (4, 0), (4, 2), (4, 4), (5, 0), (5, 2),
            (5, 4), (6, 0), (6, 2), (6, 4)]
fig = plt.figure(figsize=(13, 12))
gridspec.GridSpec(7, 6)
markers = ['^', '.', '*', 'x', ',']
linestyles = [':', '--', '-.', '-']
plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3)
for indx, i in enumerate(N_SRC):
        plt.plot(np.arange(1, TOTAL_ELE + 1), eigenval[indx],
                 linestyle=linestyles[indx], color=colors[indx],
                 marker=markers[indx], label='M='+str(i), markersize=6)
plt.title('A)', fontsize=12)
plt.legend(loc='lower left')
plt.xlabel('Number of components', fontsize=12)
plt.ylabel('Eigenvalues', fontsize=12)
plt.yscale('log')
plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3)
plt.plot(n_src_M, eigenval_M[:, 0], marker='s', color='k', markersize=5,
         linestyle=' ')
plt.title('B)', fontsize=12)
plt.xlabel('Number of basis sources', fontsize=12)
plt.xscale('log')
plt.ylabel('Eigenvalues', fontsize=12)
#plt.yscale('log')

for i in range(k_interp_cross_list[0].shape[1]):
    plt.subplot2grid((7, 6), plt_cord[i], colspan=2, rowspan=1)
    for idx, j in enumerate(N_SRC):
        plt.plot(np.linspace(0, 1, 100), np.dot(k_interp_cross_list[idx],
                 eigenvectors[idx, :, i]),
                 linestyle=linestyles[idx], color=colors[idx],
                 label='M='+str(j), lw=2)
        plt.title(f'$\\tilde{{K}}*v_{{{i+1:d}}}$', fontsize=12)
        plt.locator_params(axis='y', nbins=3)
        plt.xlabel('Depth', fontsize=12)
        plt.ylabel('CSD', fontsize=12)

#        plt.ylabel(r'$\tilde{K}*v_' + str(i) + '$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False,
           shadow=False, ncol=len(N_SRC))
#plt.xlabel('Number of estimation points')
plt.tight_layout()
plt.show()
save_path = SAVE_PATH + '/cross_kernel'
tb.makemydir(save_path)
save_as = (save_path + '/cross_kernel_eigenvector_product_for_different_M')
fig.savefig(os.path.join(save_path, save_as+'.png'))
plt.close()
