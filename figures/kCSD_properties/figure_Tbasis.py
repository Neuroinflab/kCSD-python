"""
@author: mkowalska
"""

import os
from os.path import expanduser
import numpy as np
from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import time

import targeted_basis as tb

__abs_file__ = os.path.abspath(__file__)


def set_axis(ax, letter=None):
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

    x = np.linspace(0, 1, 100)
    l1 = ax.plot(x, true_csd, label='True CSD', lw=2.)
    if est_csd_LC is not None:
        l2 = ax.plot(estm_x, est_csd, label='kCSD_CV', lw=2.)
        l3 = ax.plot(estm_x, est_csd_LC, label='kCSD_LC', lw=2.)
    else:
        l2 = ax.plot(estm_x, est_csd, label='kCSD', lw=2.)
    s1 = ax.scatter(ele_pos, np.zeros(len(ele_pos)), 13, 'k', label='Electrodes')
    #ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    if xlabel:
        ax.set_xlabel('Depth ($mm$)')
    if ylabel:
        ax.set_ylabel('CSD ($mA/mm$)')
    if title is not None:
        ax.set_title(title)
    if np.max(est_csd) < 1.2:
        ax.set_ylim(-0.2, 1.2)
    elif np.max(est_csd) > 500:
        ax.set_yticks([-5000, 0, 5000])
    ax.set_xticks([0, 0.5, 1])
    set_axis(ax, letter=letter)
    # ax.legend(frameon=False, loc='upper center', ncol=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def generate_figure(R, MU, N_SRC, TRUE_CSD_XLIMS, TOTAL_ELE, SAVE_PATH,
                    method='cross-validation', Rs=None, lambdas=None,
                    noise=None):

    ELE_LIMS = [0, 1.]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            TRUE_CSD_XLIMS, R,
                                                            MU, TOTAL_ELE,
                                                            ELE_LIMS,
                                                            noise=noise)

    fig = plt.figure(figsize=(15, 12))
    widths = [1, 1, 1]
    heights = [1, 1, 1]
    gs = gridspec.GridSpec(3, 3, height_ratios=heights, width_ratios=widths, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title='Basis limits = [0, 1]', xlabel=False, ylabel=True,
                 letter='A')

    ax = fig.add_subplot(gs[0, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title='Basis limits = [0, 0.5]', xlabel=False, ylabel=False,
                 letter='B')

    ax = fig.add_subplot(gs[0, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC,  h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title='Basis limits = [0.5, 1]', xlabel=False, ylabel=False,
                 letter='C')

    ELE_LIMS = [0, 0.5]
#    TOTAL_ELE = 6
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            TRUE_CSD_XLIMS, R,
                                                            MU, TOTAL_ELE,
                                                            ELE_LIMS)
    ax = fig.add_subplot(gs[1, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=True, letter='D')

    ax = fig.add_subplot(gs[1, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='E')

    ax = fig.add_subplot(gs[1, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='F')

    ELE_LIMS = [0.5, 1.]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            TRUE_CSD_XLIMS, R,
                                                            MU, TOTAL_ELE,
                                                            ELE_LIMS)
    ax = fig.add_subplot(gs[2, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=True, letter='G')

    ax = fig.add_subplot(gs[2, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=False, letter='H')

    ax = fig.add_subplot(gs[2, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    ax = make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                      title=None, xlabel=True, ylabel=False, letter='I')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    
    #plt.tight_layout()
    fig.savefig(os.path.join(SAVE_PATH, 'targeted_basis_' + method +
                             '_noise_' + str(noise) + '.png'), dpi=300)
    plt.show()


def generate_figure_CVLC(R, MU, N_SRC, TRUE_CSD_XLIMS, TOTAL_ELE, SAVE_PATH,
                         Rs=None, lambdas=None, noise=None):

    m_cv = 'cross-validation'
    m_lc = 'L-curve'
    method = 'CV_LC'
    ELE_LIMS = [0, 1.]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            TRUE_CSD_XLIMS, R,
                                                            MU, TOTAL_ELE,
                                                            ELE_LIMS,
                                                            noise=noise)

    fig = plt.figure(figsize=(15, 12))
    widths = [1, 1, 1]
    heights = [1, 1, 1]
    gs = gridspec.GridSpec(3, 3, height_ratios=heights, width_ratios=widths, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title='Basis limits = [0, 1]', xlabel=False, ylabel=True,
                 letter='A', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[0, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title='Basis limits = [0, 0.5]', xlabel=False, ylabel=False,
                 letter='B', est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[0, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title='Basis limits = [0.5, 1]', xlabel=False, ylabel=False,
                 letter='C', est_csd_LC=obj_LC.values('CSD'))

    ELE_LIMS = [0, 0.5]
#    TOTAL_ELE = 6
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            TRUE_CSD_XLIMS, R,
                                                            MU, TOTAL_ELE,
                                                            ELE_LIMS)
    ax = fig.add_subplot(gs[1, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=True, letter='D',
                 est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[1, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='E',
                 est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[1, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='F',
                 est_csd_LC=obj_LC.values('CSD'))

    ELE_LIMS = [0.5, 1.]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            TRUE_CSD_XLIMS, R,
                                                            MU, TOTAL_ELE,
                                                            ELE_LIMS)
    ax = fig.add_subplot(gs[2, 0])
    xmin = 0
    xmax = 1
    ext_x = 0
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=True, letter='G',
                 est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[2, 1])
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=False, letter='H',
                 est_csd_LC=obj_LC.values('CSD'))

    ax = fig.add_subplot(gs[2, 2])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj_CV = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_cv, Rs=Rs, lambdas=lambdas)
    obj_LC = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=m_lc, Rs=Rs, lambdas=lambdas)
    ax = make_subplot(ax, true_csd, obj_CV.values('CSD'), obj_CV.estm_x, ele_pos=ele_pos,
                      title=None, xlabel=True, ylabel=False, letter='I',
                      est_csd_LC=obj_LC.values('CSD'))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    
    #plt.tight_layout()
    fig.savefig(os.path.join(SAVE_PATH, 'targeted_basis_' + method +
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

    N_SRC = 64
    TRUE_CSD_XLIMS = [0., 1.]
    TOTAL_ELE = 12
    R = 0.2
    MU = 0.25
#    method = 'cross-validation'  # L-curve
    method = 'L-curve'
    Rs = np.arange(0.1, 0.4, 0.05)
#    Rs = np.array([0.2])
    lambdas = np.zeros(1)
#    generate_figure(R, MU, N_SRC, TRUE_CSD_XLIMS, TOTAL_ELE, SAVE_PATH, method,
#                    Rs, lambdas=None, noise=10)
    generate_figure_CVLC(R, MU, N_SRC, TRUE_CSD_XLIMS, TOTAL_ELE, SAVE_PATH,
                         Rs=Rs, lambdas=None, noise=10)
