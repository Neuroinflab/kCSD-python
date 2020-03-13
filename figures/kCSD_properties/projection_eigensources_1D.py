"""
@author: mbejtka
"""
import os
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from figure_properties import *
import matplotlib.gridspec as gridspec
import scipy

from kcsd import KCSD1D
import targeted_basis as tb

__abs_file__ = os.path.abspath(__file__)


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.dot(np.dot(csd, orthn), orthn.T)


def calculate_eigensources(cross_kernel, eigenvectors):
    return np.dot(cross_kernel, eigenvectors)


def calculate_projection(csd, k, eigenvectors):
    eigensources = calculate_eigensources(k.k_interp_cross, eigenvectors)
    return csd_into_eigensource_projection(csd, eigensources)


def calculate_diff(csd, projection):
    # return np.abs(csd - projection)/np.max(csd)
    return csd - projection


def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def stability_M(n_src, total_ele, ele_pos, pots, R_init=0.23):
    """
    Investigates stability of reconstruction for different number of basis
    sources

    Parameters
    ----------
    n_src: int
        Number of basis sources.
    total_ele: int
        Number of electrodes.
    ele_pos: numpy array
        Electrodes positions.
    pots: numpy array
        Values of potentials at ele_pos.
    R_init: float
        Initial value of R parameter - width of basis source
        Default: 0.23.

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
                     gdx=0.01, n_src_init=n_src[i], ext_x=0, xmin=0, xmax=1,
                     R_init=R_init)
        try:
            eigenvalue, eigenvector = np.linalg.eigh(obj.k_pot +
                                                     obj.lambd *
                                                     np.identity
                                                     (obj.k_pot.shape[0]))
        except LinAlgError:
            raise LinAlgError('EVD is failing - try moving the electrodes'
                              'slightly')
        idx = eigenvalue.argsort()[::-1]
        eigenvalues[i] = eigenvalue[idx]
        eigenvectors[i] = eigenvector[:, idx]
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
                    noise=0, R_init=0.23):
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
    noise: float
        Determines the level of noise in the data.
        Default: 0.
    R_init: float
        Initial value of R parameter - width of basis source
        Default: 0.23.

    Returns
    -------
    None
    """
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(csd_profile,
                                                            true_csd_xlims,
                                                            R, MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)
    print('csd', true_csd.shape)
    n_src_M = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    OBJ_M, eigenval_M, eigenvec_M = stability_M(n_src_M,
                                                total_ele, ele_pos, pots,
                                                R_init=R_init)
    # print('eigenvector', eigenvec_M[0].shape)
    # eigensources = calculate_eigensources(OBJ_M[0].k_interp_cross, eigenvec_M[0])
    projection_M = []
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(131)
    ax.set_title('Projection')
    for i in range(len(n_src_M)):
        projection = calculate_projection(true_csd, OBJ_M[i], eigenvec_M[i])
        projection_M.append(projection)
        ax.plot(np.linspace(0, 1, 100), projection, label='M='+str(n_src_M[i]))
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('CSD (mA/mm)')
    set_axis(ax, -0.05, 1.05, letter='A')
    
    ax = fig.add_subplot(132)
    ax.set_title('Error')
    for i in range(len(n_src_M)):
        err = calculate_diff(true_csd, projection_M[i])
        ax.plot(np.linspace(0, 1, 100), err, label='M='+str(n_src_M[i]))
        ax.set_xlabel('Depth (mm)')
    set_axis(ax, -0.05, 1.05, letter='B')
    
    ax = fig.add_subplot(133)
    ax.set_title('Anihilator')
    for i in range(len(n_src_M)):
        err = calculate_diff(true_csd.reshape(true_csd.shape[0], 1), OBJ_M[i].values('CSD'))
        ax.plot(np.linspace(0, 1, 100), err, label='M='+str(n_src_M[i]))
        ax.set_xlabel('Depth (mm)')
    set_axis(ax, -0.05, 1.05, letter='C')
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', ncol=9, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.savefig(os.path.join('projections_different_M' + '.png'), dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    print(OBJ_M[i].values('CSD').shape)

#     BLACK = _html(0, 0, 0)
#     ORANGE = _html(230, 159, 0)
#     SKY_BLUE = _html(86, 180, 233)
#     GREEN = _html(0, 158, 115)
#     YELLOW = _html(240, 228, 66)
#     BLUE = _html(0, 114, 178)
#     VERMILION = _html(213, 94, 0)
#     PURPLE = _html(204, 121, 167)
#     colors = [BLUE, ORANGE, GREEN, PURPLE, VERMILION, SKY_BLUE, YELLOW, BLACK]

#     fig = plt.figure(figsize=(18, 16))
# #    heights = [1, 1, 1, 0.2, 1, 1, 1, 1]
#     heights = [4, 0.3, 1, 1, 1, 1]
#     markers = ['^', '.', '*', 'x', ',']
# #    linestyles = [':', '--', '-.', '-']
#     linestyles = ['-', '-', '-', '-']

    return true_csd, projection_M


if __name__ == '__main__':
    ELE_LIMS = [0, 1.]
    TRUE_CSD_XLIMS = [0., 1.]
    TOTAL_ELE = 12
    CSD_PROFILE = tb.csd_profile
    R = 0.2
    MU = 0.25
    R_init = 0.2
    csd, projection_M = generate_figure(CSD_PROFILE, R, MU, TRUE_CSD_XLIMS, TOTAL_ELE, ELE_LIMS,
                    noise=None, R_init=R_init)

