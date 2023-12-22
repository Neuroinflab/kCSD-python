
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from kcsd import csd_profile as CSD
from kcsd import ValidateKCSD2D
from figure_properties import *
from kCSD_with_reliability_map_2D import make_reconstruction, matrix_symmetrization


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


def make_single_subplot(ax, val_type, xs, ys, values, cax, title=None,
                        ele_pos=None, xlabel=False, ylabel=False, letter='',
                        t_max=1., mask=False, level=False):
    cmap = cm.Greys
    ax.set_aspect('equal')
    if t_max is None:
        t_max = np.max(np.abs(values))
    if level is not False:
        levels = level
    else:
        levels = np.linspace(0, 0.2, 32)
    im = ax.contourf(xs, ys, values,
                     levels=levels, cmap=cmap, alpha=1)
    CS = ax.contour(xs, ys, values, cmap='Greys')
    ax.clabel(CS,  # label every second level
              inline=1,
              fmt='%1.2f',
              colors='blue')
    if val_type == 'err':
        ax.scatter(ele_pos[:, 0], ele_pos[:, 1], s=20, marker='.', c='black',
                   zorder=3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if xlabel:
        ax.set_xlabel('X (mm)')
    if ylabel:
        ax.set_ylabel('Y (mm)')
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ticks = np.linspace(0, 0.2, 3, endpoint=True)
    plt.colorbar(im, cax=cax, orientation='horizontal', format='%.2f',
                 ticks=ticks)
    set_axis(ax, letter=letter)
    #plt.tight_layout()
    return ax, cax


def generate_reliability_map(point_error, ele_pos, title):
    csd_at = np.mgrid[0.:1.:100j,
                      0.:1.:100j]
    csd_x, csd_y = csd_at
    plt.figure(figsize=(17, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1., 0.04], left=0.415,
                           right=0.585, top=0.880, bottom=0.110)
    ax = plt.subplot(gs[0, 0])
    cax = plt.subplot(gs[1, 0])
    make_single_subplot(ax, 'err', csd_x, csd_y, point_error, cax=cax,
                        ele_pos=ele_pos, title=None, xlabel=True, ylabel=True,
                        letter=' ', t_max=0.2, level=np.linspace(0, 0.2, 16))
    plt.savefig(title + '.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    CSD_PROFILE = CSD.gauss_2d_large
    CSD_SEED = 16
    ELE_LIMS = [0.05, 0.95]  # range of electrodes space
    method = 'cross-validation'
    Rs = np.arange(0.2, 0.5, 0.1)
    lambdas = np.zeros(1)
    noise = 0

    KK = ValidateKCSD2D(CSD_SEED, h=50., sigma=1., n_src_init=400,
                        est_xres=0.01, est_yres=0.01, ele_lims=ELE_LIMS)
    k, csd_at, true_csd, ele_pos, pots = make_reconstruction(KK, CSD_PROFILE,
                                                             CSD_SEED,
                                                             total_ele=100,
                                                             noise=noise,
                                                             Rs=Rs,
                                                             lambdas=lambdas,
                                                             method=method)

    error_l = np.load('error_maps_2D/point_error_large_100_all_ele.npy')
    error_s = np.load('error_maps_2D/point_error_small_100_all_ele.npy')
    error_all = np.concatenate((error_l, error_s))
    symm_array_large = matrix_symmetrization(error_l)
    symm_array_small = matrix_symmetrization(error_s)
    symm_array_all = matrix_symmetrization(error_all)

    generate_reliability_map(np.mean(symm_array_all, axis=0), ele_pos,
                             'Reliability_map_random_newRDM_symm')
    generate_reliability_map(np.mean(symm_array_large, axis=0), ele_pos,
                             'Reliability_map_large_newRDM_symm')
    generate_reliability_map(np.mean(symm_array_small, axis=0), ele_pos,
                             'Reliability_map_small_newRDM_symm')
