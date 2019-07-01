"""
@author: mkowalska
"""
import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
from figure_properties import *
import matplotlib.gridspec as gridspec
import datetime
import time

from kcsd import KCSD1D
import targeted_basis as tb

__abs_file__ = os.path.abspath(__file__)


def pots_scan(n_src, ele_lims, true_csd_xlims,
              total_ele, ele_pos, R_init=0.23):
    """
    Investigates kCSD reconstructions for unitary potential on different
    electrodes

    Parameters
    ----------
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

    Returns
    -------
    obj_all: class object
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    eigenvectors: numpy array
        Eigen vectors of k_pot matrix.
    """
    obj_all = []
    est_csd = []
    for i, value in enumerate(ele_pos):
        pots = np.zeros(len(ele_pos))
        pots[i] = 1
        pots = pots.reshape((len(ele_pos), 1))
        obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=0.3, h=0.25,
                     gdx=0.01, n_src_init=n_src, ext_x=0, xmin=0, xmax=1,
                     R_init=R_init)
        est_csd.append(obj.values('CSD'))

        obj_all.append(obj)
    return obj_all, est_csd


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


def generate_figure(true_csd_xlims, total_ele, ele_lims,
                    save_path, R_init=0.23):
    """
    Generates figure for potential propagation. Shows kCSD recostruction
    for unitary potential on one electrode.

    Parameters
    ----------
    true_csd_xlims: list
        Boundaries for ground truth space.
    total_ele: int
        Number of electrodes.
    ele_lims: list
        Electrodes limits.
    save_path: string
        Directory.
    R_init: float
        Initial value of R parameter.

    Returns
    -------
    None
    """
    ele_pos = np.linspace(ele_lims[0], ele_lims[1], total_ele)
    ele_pos = ele_pos.reshape((len(ele_pos), 1))
    n_src = 256
    OBJ_M, est_csd = pots_scan(n_src, ele_lims, true_csd_xlims,
                               total_ele, ele_pos, R_init=R_init)

    plt_cord = [(0, 0), (0, 2), (0, 4),
                (1, 0), (1, 2), (1, 4),
                (2, 0), (2, 2), (2, 4),
                (3, 0), (3, 2), (3, 4)]

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    fig = plt.figure(figsize=(17, 12))
    heights = [1, 1, 1, 1]
    gs = gridspec.GridSpec(4, 6, height_ratios=heights, hspace=0.3, wspace=0.8)

    for i in range(len(ele_pos)):
        ax = fig.add_subplot(gs[plt_cord[i][0],
                                plt_cord[i][1]:plt_cord[i][1]+2])
        ax.plot(np.linspace(0, 1, 100), est_csd[i], lw=2, c='red',
                label='kCSD')
        ax.scatter(ele_pos, np.zeros(len(ele_pos)), c='k', label='Electordes')
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, 1, 100), OBJ_M[i].values('POT'), c='green',
                 label='Potentials')
        ax2.set_ylim([-1.5, 1.5])
        ax.set_ylim([-150, 150])
        set_axis(ax, -0.10, 1.1, letter=letters[i])

        if i < 9:
            ax.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.set_xlabel('Depth ($mm$)')
        if i % 3 == 0:
            ax.set_ylabel('CSD ($mA/mm$)')
            ax.set_yticks((-100, 0, 100))
            ax.set_yticklabels(('-100', '0', '100'))
            ax.yaxis.set_label_coords(-0.18, 0.5)
        else:
            ax.set_yticks(())
        if (i+1) % 3 == 0:
            ax2.set_ylabel('Potentials ($mA$)')
            ax2.set_yticks((-1, 0, 1))
            ax2.set_yticklabels(('-1', '0', '1'))
        else:
            ax2.set_yticks(())

#        ax.ticklabel_format(style='sci', axis='y', scilimits=((0.0, 0.0)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
#        align_yaxis(ax, 0, ax2, 0)
    ht1, lh1 = ax.get_legend_handles_labels()
    ht2, lh2 = ax2.get_legend_handles_labels()

    fig.legend(ht1+ht2, lh1+lh2, loc='lower center', ncol=3, frameon=False)
    fig.savefig(os.path.join(save_path, 'vectors_' + 'R0_2' + '.png'), dpi=300)
    plt.show()


#def align_yaxis(ax1, v1, ax2, v2):
#    """
#    Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
#
#    Parametres
#    ----------
#    ax1
#    v1
#    ax2
#    v2
#
#    Returns
#    -------
#    None
#    """
#    _, y1 = ax1.transData.transform((0, v1))
#    _, y2 = ax2.transData.transform((0, v2))
#    inv = ax2.transData.inverted()
#    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
#    miny, maxy = ax2.get_ylim()
#    ax2.set_ylim(miny+dy, maxy+dy)


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
    R_init = 0.2
    lambdas = np.zeros([1])
    generate_figure(TRUE_CSD_XLIMS, TOTAL_ELE, ELE_LIMS,
                    SAVE_PATH, R_init=R_init)
