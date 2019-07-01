"""
@author: mkowalska
"""
import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
from figure_properties import *
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import datetime
import time

from kcsd import SpectralStructure, KCSD2D, ValidateKCSD2D
from kcsd import csd_profile as CSD

__abs_file__ = os.path.abspath(__file__)


def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def stability_M(csd_profile, n_src, ele_lims,
                total_ele, ele_pos, pots,
                method='cross-validation', Rs=None, lambdas=None):
    """
    Investigates stability of reconstruction for different number of basis
    sources

    Parameters
    ----------
    csd_profile: function
        Function to produce csd profile.
    n_src: int
        Number of basis sources.
    ele_lims: list
        Boundaries for electrodes placement.
    total_ele: int
        Number of electrodes.
    ele_pos: numpy array
        Electrodes positions.
    pots: numpy array
        Values of potentials at ele_pos.
    method: string
        Determines the method of regularization.
        Default: cross-validation.
    Rs: numpy 1D array
        Basis source parameter for crossvalidation.
        Default: None.
    lambdas: numpy 1D array
        Regularization parameter for crossvalidation.
        Default: None.

    Returns
    -------
    obj_all: class object
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    eigenvectors: numpy array
        Eigen vectors of k_pot matrix.
    """
    obj_all = []
    eigenvectors = np.zeros((len(n_src), total_ele, total_ele))
    eigenvalues = np.zeros((len(n_src), total_ele))
    for i, value in enumerate(n_src):
        pots = pots.reshape((len(ele_pos), 1))
        obj = KCSD2D(ele_pos, pots, src_type='gauss', sigma=1., h=50.,
                     gdx=0.01, gdy=0.01, n_src_init=n_src[i], xmin=0, xmax=1,
                     ymax=1, ymin=0)
        if method == 'cross-validation':
            obj.cross_validate(Rs=Rs, lambdas=lambdas)
        elif method == 'L-curve':
            obj.L_curve(Rs=Rs, lambdas=lambdas)
        ss = SpectralStructure(obj)
        eigenvectors[i], eigenvalues[i] = ss.evd()

        obj_all.append(obj)
    np.save('obj_all.npy', obj_all)
    np.save('eigenvalues.npy', eigenvalues)
    np.save('eigenvectors.npy', eigenvectors)
    return obj_all, eigenvalues, eigenvectors


def set_axis(ax, x, y, letter=None):
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
        x,
        y,
        letter,
        fontsize=15,
        weight='bold',
        transform=ax.transAxes)
    return ax


def generate_figure(csd_profile, csd_seed, ele_pos, pots, total_ele, ele_lims,
                    save_path, method='cross-validation', Rs=None,
                    lambdas=None, noise=0):
    """
    Generates figure for spectral structure decomposition.

    Parameters
    ----------
    csd_profile: function
        Function to produce csd profile.
    csd_seed: int
        Seed for random generator to choose random CSD profile.
    ele_pos: numpy array
        Electrodes positions.
    pots: numpy array
        Values of potentials at ele_pos.
    total_ele: int
        Number of electrodes.
    ele_lims: list
        Electrodes limits.
    save_path: string
        Directory.
    method: string
        Determines the method of regularization.
        Default: cross-validation.
    Rs: numpy 1D array
        Basis source parameter for crossvalidation.
        Default: None.
    lambdas: numpy 1D array
        Regularization parameter for crossvalidation.
        Default: None.
    noise: float
        Determines the level of noise in the data.
        Default: 0.

    Returns
    -------
    None
    """
    csd_at = np.mgrid[0.:1.:100j,
                      0.:1.:100j]
    x, y = csd_at

    n_src_M = [4, 9, 16, 27, 64, 81, 128, 256, 512, 729, 1024]
    OBJ_M = np.load('obj_all.npy')
    eigenval_M = np.load('eigenvalues.npy')
    eigenvec_M = np.load('eigenvectors.npy')
#    OBJ_M, eigenval_M, eigenvec_M = stability_M(csd_profile, n_src_M,
#                                                ele_lims,
#                                                total_ele, ele_pos, pots,
#                                                method=method, Rs=Rs,
#                                                lambdas=lambdas)

    plt_cord = [(3, 0), (3, 2), (3, 4),
                (5, 0), (5, 2), (5, 4),
                (7, 0), (7, 2), (7, 4)]

    letters = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

    BLACK = _html(0, 0, 0)
    ORANGE = _html(230, 159, 0)
    SKY_BLUE = _html(86, 180, 233)
    GREEN = _html(0, 158, 115)
    YELLOW = _html(240, 228, 66)
    BLUE = _html(0, 114, 178)
    VERMILION = _html(213, 94, 0)
    PURPLE = _html(204, 121, 167)
    colors = [BLUE, ORANGE, GREEN, PURPLE, VERMILION, SKY_BLUE, YELLOW, BLACK]

    fig = plt.figure(figsize=(18, 16))
    heights = [3, 0.1, 1, 1, 1, 1, 1, 1, 1]
    markers = ['^', '.', '*', 'x', ',']
#    linestyles = [':', '--', '-.', '-', ':']
    linestyles = ['-', '-', '-', '-', '-']
    src_idx = [0, 1, 2, 4, 7]

    gs = gridspec.GridSpec(9, 6, height_ratios=heights, hspace=0.6, wspace=0.5)

    ax = fig.add_subplot(gs[0, :3])
    for indx, i in enumerate(src_idx):
        ax.plot(np.arange(1, total_ele + 1), eigenval_M[i],
                linestyle=linestyles[indx], color=colors[indx],
                marker=markers[indx], label='M='+str(n_src_M[i]),
                markersize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5,
              frameon=False)
    ht, lh = ax.get_legend_handles_labels()
    set_axis(ax, -0.05, 1.05, letter='A')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Eigenvalues')
    ax.set_yscale('log')
    ax.set_ylim([1e-6, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = fig.add_subplot(gs[0, 3:])
    ax.plot(n_src_M, eigenval_M[:, 0], marker='s', color='k', markersize=5,
            linestyle=' ')

    set_axis(ax, -0.05, 1.05, letter='B')
    ax.set_xlabel('Number of basis sources')
    ax.set_xscale('log')
    ax.set_ylabel('Eigenvalues')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    src_idx = 10
    v = np.dot(OBJ_M[src_idx].k_interp_cross, eigenvec_M[src_idx])
    for i in range(total_ele):
        ax = fig.add_subplot(gs[plt_cord[i][0]:plt_cord[i][0]+2,
                                plt_cord[i][1]:plt_cord[i][1]+2],
                             aspect='equal')

        a = OBJ_M[src_idx].process_estimate(v[:, i])
        cset = ax.contourf(x, y, a[:, :, 0], cmap=cm.bwr)

        ax.text(0.5, 1.05, r"$\tilde{K} \cdot{v_{{%(i)d}}}$" % {'i': i+1},
                horizontalalignment='center', transform=ax.transAxes,
                fontsize=15)

        set_axis(ax, -0.10, 1.1, letter=letters[i])
        if i < 6:
            ax.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.set_xlabel('X ($mm$)')
            ax.spines['bottom'].set_visible(True)
        if i % 3 == 0:
            ax.set_ylabel('Y ($mm$)')
            ax.yaxis.set_label_coords(-0.28, 0.5)

        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis="y", direction='out')
        ax.ticklabel_format(style='sci', axis='y', scilimits=((0.0, 0.0)))
        ax.ticklabel_format(style='sci', axis='x', scilimits=((0.0, 0.0)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.colorbar(cset, ax=ax)
#    fig.legend(ht, lh, loc='lower center', ncol=5, frameon=False)
    fig.savefig(os.path.join('stability_2d.png'), dpi=300)

    plt.show()


if __name__ == '__main__':
    CSD_PROFILE = CSD.gauss_2d_large
    N_SRC_INIT = 1000
    ELE_LIMS = [0.118, 0.882]  # range of electrodes space
    method = 'cross-validation'
    Rs = np.linspace(0.1, 1.5, 15)
    lambdas = np.zeros(1)
    noise = 0
    CSD_SEED = 16
    TRUE_CSD_XLIMS = [0., 1.]
    home = expanduser('~')
    DAY = datetime.datetime.now()
    DAY = DAY.strftime('%Y%m%d')
    TIMESTR = time.strftime("%H%M%S")
    SAVE_PATH = home + "/kCSD_results/" + DAY + '/' + TIMESTR
    total_ele = 9

    KK = ValidateKCSD2D(CSD_SEED, h=50., sigma=1., n_src_init=N_SRC_INIT,
                        ele_lims=ELE_LIMS, est_xres=0.01, est_yres=0.01)
    ele_pos, pots = KK.electrode_config(CSD_PROFILE, CSD_SEED, total_ele=9,
                                        ele_lims=ELE_LIMS, h=50., sigma=1.)

    generate_figure(CSD_PROFILE, CSD_SEED, ele_pos, pots, total_ele, ELE_LIMS,
                    SAVE_PATH, method='cross-validation', Rs=Rs,
                    lambdas=lambdas)
