"""
@author: mkowalska
"""

import os
import numpy as np
from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import targeted_basis as tb

__abs_file__ = os.path.abspath(__file__)


def set_axis(ax, letter=None):
    """
    Formats the plot's caption.

    Parameters
    ----------
    ax: Axes object.
    letter: string
        Caption of the plot.
        Default: None.

    Returns
    -------
    ax: modyfied Axes object.
    """
    ax.text(
        -0.15,
        1.05,
        letter,
        fontsize=18,
        weight='bold',
        transform=ax.transAxes)
    return ax


def make_subplot(ax, true_csd, est_csd, estm_x, title=None, ele_pos=None,
                 xlabel=False, ylabel=False, letter='', t_max=None,
                 est_csd_LC=None):

    """
    Parameters
    ----------
    ax: Axes object.
    true_csd: numpy array
        Values of generated CSD.
    est_csd: numpy array
        Reconstructed csd.
    estm_x: numpy array
        Locations at which CSD is requested.
    title: string
        Title of the plot.
        Default: None.
    ele_pos: numpy array
        Positions of electrodes.
        Default: None.
    xlabel: string
        Caption of the x axis.
        Default: False.
    ylabel: string
        Caption of the y axis.
        Default: False.
    letter: string
        Caption of the plot.
        Default: ''.
    t_max:
        Default: None.
    est_csd_LC: numpy array
        CSD reconstructed with L-curve method.
        Default: None.

    Returns
    -------
    ax: modyfied Axes object.
    """

    x = np.linspace(0, 1, 100)
    l1 = ax.plot(x, true_csd, label='True CSD', lw=2.)
    if est_csd_LC is not None:
        l2 = ax.plot(estm_x, est_csd, '--', label='kCSD Cross-validation', lw=2.)
        l3 = ax.plot(estm_x, est_csd_LC, '.', label='kCSD L-Curve', lw=2.)
    else:
        l2 = ax.plot(estm_x, est_csd, '--', label='kCSD', lw=2.)
    ax.set_xlim([0, 1])
    if xlabel:
        ax.set_xlabel('Depth ($mm$)')
    if ylabel:
        ax.set_ylabel('CSD ($mA/mm$)')
    if title is not None:
        ax.set_title(title)
    if np.max(est_csd) < 1.2:
        ax.set_ylim(-0.2, 1.2)
        s1 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-0.2, 17, 'k', label='Electrodes')
        s1.set_clip_on(False)
    elif np.max(est_csd) < 1.7:
        ax.set_ylim(-10000, 10000)
        s3 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-10000, 17, 'k', label='Electrodes')
        s3.set_clip_on(False)
        ax.set_yticks([-7000, 0, 7000])
    if np.max(est_csd) > 500:
        ax.set_ylim(-7000, 7000)
        s3 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-7000, 17, 'k', label='Electrodes')
        s3.set_clip_on(False)
        ax.set_yticks([-5000, 0, 5000])
    elif np.max(est_csd) > 50:
        ax.set_ylim(-100, 100)
        s2 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-100, 17, 'k', label='Electrodes')
        s2.set_clip_on(False)
        ax.set_yticks([-70, 0, 70])
    ax.set_xticks([0, 0.5, 1])
    set_axis(ax, letter=letter)
    # ax.legend(frameon=False, loc='upper center', ncol=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def generate_figure_CVLC(R, MU, n_src, true_csd_xlims, total_ele,
                         Rs=None, lambdas=None, noise=0):
    """
    Generates figure for targeted basis investigation including results from
    both cross-validation and L-curve.

    Parameters
    ----------
    R: float
        Thickness of the groundtruth source.
        Default: 0.2.
    MU: float
        Central position of Gaussian source
        Default: 0.25.
    nr_src: int
        Number of basis sources.
    true_csd_xlims: list
        Boundaries for ground truth space.
    total_ele: int
        Number of electrodes.
    save_path: string
        Directory.
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

    m_cv = 'cross-validation'
    m_lc = 'L-curve'
    method = 'CV_LC'
    ele_lims = [0, 1.]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            true_csd_xlims, R,
                                                            MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)

    fig = plt.figure(figsize=(15, 12))
    widths = [1, 1, 1]
    heights = [1, 1, 1]
    gs = gridspec.GridSpec(3, 3, height_ratios=heights, width_ratios=widths,
                           hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title='Basis limits = [0, 1]', xlabel=False,
                 ylabel=True, letter='A', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[0, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title='Basis limits = [0, 0.5]',
                 xlabel=False, ylabel=False, letter='B',
                 est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[0, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title='Basis limits = [0.5, 1]',
                 xlabel=False, ylabel=False, letter='C',
                 est_csd_LC=obj_LC.values('CSD'))

    ele_lims = [0, 0.5]
#    total_ele = 6
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            true_csd_xlims, R,
                                                            MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)
    ax = fig.add_subplot(gs[1, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title=None, xlabel=False, ylabel=True,
                 letter='D', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[1, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title=None, xlabel=False, ylabel=False,
                 letter='E', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[1, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title=None, xlabel=False, ylabel=False,
                 letter='F', est_csd_LC=obj_LC.values('CSD'))

    ele_lims = [0.5, 1.]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            true_csd_xlims, R,
                                                            MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)
    ax = fig.add_subplot(gs[2, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title=None, xlabel=True, ylabel=True,
                 letter='G', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[2, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                 ele_pos=ele_pos, title=None, xlabel=True, ylabel=False,
                 letter='H', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[2, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                               sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                               xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    ax = make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x,
                      ele_pos=ele_pos, title=None, xlabel=True, ylabel=False,
                      letter='I', est_csd_LC=obj_LC.values('CSD'))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

    fig.savefig(os.path.join('targeted_basis_' + method +
                             '_noise_' + str(noise) + '.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    N_SRC = 64
    TRUE_CSD_XLIMS = [0., 1.]
    TOTAL_ELE = 12
    R = 0.2
    MU = 0.25
    Rs = np.arange(0.1, 0.4, 0.05)
    generate_figure_CVLC(R, MU, N_SRC, TRUE_CSD_XLIMS, TOTAL_ELE,
                         Rs=Rs, lambdas=None, noise=10)
