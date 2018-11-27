"""
@author: mkowalska
"""
import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
from figure_properties import *
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter
import datetime
import time

from kcsd import SpectralStructure, KCSD1D
import targeted_basis as tb

__abs_file__ = os.path.abspath(__file__)


def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def stability_M(csd_profile, n_src, ele_lims, true_csd_xlims,
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
    true_csd_xlims: list
        Boundaries for ground truth space.
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
        obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=0.3, h=0.25,
                     gdx=0.01, n_src_init=n_src[i], ext_x=0, xmin=0, xmax=1)
        if method == 'cross-validation':
            obj.cross_validate(Rs=Rs, lambdas=lambdas)
        elif method == 'L-curve':
            obj.L_curve(Rs=Rs, lambdas=lambdas)
        ss = SpectralStructure(obj)
        eigenvectors[i], eigenvalues[i] = ss.evd()

        obj_all.append(obj)
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


def generate_figure(csd_profile, R, MU, true_csd_xlims, total_ele, ele_lims,
                    save_path, method='cross-validation', Rs=None,
                    lambdas=None, noise=0):
    """
    Generates figure for spectral structure decomposition.

    Parameters
    ----------
    csd_profile: function
        Function to produce csd profile.
    R: float
        Thickness of the groundtruth source.
        Default: 0.2.
    MU: float
        Central position of Gaussian source
        Default: 0.25.
    true_csd_xlims: list
        Boundaries for ground truth space.
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
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(csd_profile,
                                                            true_csd_xlims,
                                                            R, MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)

    n_src_M = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    OBJ_M, eigenval_M, eigenvec_M = stability_M(csd_profile, n_src_M,
                                                ele_lims, true_csd_xlims,
                                                total_ele, ele_pos, pots,
                                                method=method, Rs=Rs,
                                                lambdas=lambdas)

    plt_cord = [(2, 0), (2, 2), (2, 4),
                (3, 0), (3, 2), (3, 4),
                (4, 0), (4, 2), (4, 4),
                (5, 0), (5, 2), (5, 4)]


    letters = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O']

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
#    heights = [1, 1, 1, 0.2, 1, 1, 1, 1]
    heights = [4, 0.3, 1, 1, 1, 1]
    markers = ['^', '.', '*', 'x', ',']
#    linestyles = [':', '--', '-.', '-']
    linestyles = ['-', '-', '-', '-']
    src_idx = [0, 2, 3, 8]

    gs = gridspec.GridSpec(6, 6, height_ratios=heights, hspace=0.3, wspace=0.6)

    ax = fig.add_subplot(gs[0, :3])
    for indx, i in enumerate(src_idx):
        ax.plot(np.arange(1, total_ele + 1), eigenval_M[i],
                linestyle=linestyles[indx], color=colors[indx],
                marker=markers[indx], label='M='+str(n_src_M[i]),
                markersize=10)
#    ax.set_title(' ', fontsize=12)
    ht, lh = ax.get_legend_handles_labels()
    set_axis(ax, -0.05, 1.05, letter='A')
#    ax.legend(loc='lower left')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Eigenvalues')
    ax.set_yscale('log')
    ax.set_ylim([1e-6, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = fig.add_subplot(gs[0, 3:])
    ax.plot(n_src_M, eigenval_M[:, 0], marker='s', color='k', markersize=5,
            linestyle=' ')
    #ax.set_title(' ', fontsize=12)
    set_axis(ax, -0.05, 1.05, letter='B')
    ax.set_xlabel('Number of basis sources')
    ax.set_xscale('log')
    ax.set_ylabel('Eigenvalues')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(OBJ_M[0].k_interp_cross.shape[1]):
        ax = fig.add_subplot(gs[plt_cord[i][0],
                                plt_cord[i][1]:plt_cord[i][1]+2])
        for idx, j in enumerate(src_idx):
            ax.plot(np.linspace(0, 1, 100), np.dot(OBJ_M[j].k_interp_cross,
                    eigenvec_M[j, :, i]),
                    linestyle=linestyles[idx], color=colors[idx],
                    label='M='+str(n_src_M[j]), lw=2)
            #ax.set_title(r"$\tilde{K}*v_{{%(i)d}}$" % {'i': i+1})
            ax.text(0.5, 1., r"$\tilde{K}*v_{{%(i)d}}$" % {'i': i+1},
                    horizontalalignment='center', transform=ax.transAxes, fontsize=20)
#            ax.locator_params(axis='y', nbins=3)

#            ax.set_xlabel('Depth (mm)', fontsize=12)
#            ax.set_ylabel('CSD (mA/mm)', fontsize=12)
            set_axis(ax, -0.10, 1.1, letter=letters[i])
            if i < 9:
                ax.get_xaxis().set_visible(False)
                ax.spines['bottom'].set_visible(False)
            else:
                ax.set_xlabel('Depth ($mm$)')
            if i % 3 == 0:
                ax.set_ylabel('CSD ($mA/mm$)')
                ax.yaxis.set_label_coords(-0.18, 0.5)
#            ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
#            ax.tick_params(direction='out', pad=10)
#            ax.yaxis.get_major_formatter(FormatStrFormatter('%.2f'))
            ax.ticklabel_format(style='sci', axis='y', scilimits=((0.0, 0.0)))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
#     ht, lh = ax.get_legend_handles_labels()

#     ax = fig.add_subplot(gs[3, :])
#     ax.legend(ht, lh,  fancybox=False, shadow=False, ncol=len(src_idx),
#               loc='upper center', frameon=False, bbox_to_anchor=(0.5, 0.0))
#     ax.axis('off')

#    plt.tight_layout()
    fig.legend(ht, lh, loc='lower center', ncol=5, frameon=False)
    fig.savefig(os.path.join(save_path, 'vectors_' + method +
                             '_noise_' + str(noise) + '.png'), dpi=300)

    plt.show()


if __name__ == '__main__':
    home = expanduser('~')
    DAY = datetime.datetime.now()
    DAY = DAY.strftime('%Y%m%d')
    TIMESTR = time.strftime("%H%M%S")
    SAVE_PATH = home + "/kCSD_results/" + DAY + '/' + TIMESTR
    tb.makemydir(SAVE_PATH)
    tb.save_source_code(SAVE_PATH, time.strftime("%Y%m%d-%H%M%S"))

    ELE_LIMS = [0, 1.]
    TRUE_CSD_XLIMS = [0., 1.]
    TOTAL_ELE = 12
    CSD_PROFILE = tb.csd_profile
    R = 0.2
    MU = 0.25
    generate_figure(CSD_PROFILE, R, MU, TRUE_CSD_XLIMS, TOTAL_ELE, ELE_LIMS,
                    SAVE_PATH, method='cross-validation',
                    Rs=np.arange(0.1, 0.5, 0.05), noise=None)
