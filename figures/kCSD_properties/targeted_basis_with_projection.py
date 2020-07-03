import figure_Tbasis as fb
import targeted_basis as tb
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from figure_properties import *
import os
from sklearn import preprocessing


def calculate_eigensources(obj):        
    try:
        eigenvalue, eigenvector = np.linalg.eigh(obj.k_pot +
                                                 obj.lambd *
                                                 np.identity
                                                 (obj.k_pot.shape[0]))
        print('lambd: ', obj.lambd)
    except LinAlgError:
        raise LinAlgError('EVD is failing - try moving the electrodes'
                          'slightly')
    idx = eigenvalue.argsort()[::-1]
    eigenvalues = eigenvalue[idx]
    eigenvectors = eigenvector[:, idx]
    eigensources = np.dot(obj.k_interp_cross, eigenvectors)
    eigensources = preprocessing.normalize(eigensources, axis=0)
    return eigensources


def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def make_subplot(ax, eigensources, estm_x, title=None, ele_pos=None,
                 xlabel=False, ylabel=False, letter='', t_max=None):
    BLACK = _html(0, 0, 0)
    ORANGE = _html(230, 159, 0)
    SKY_BLUE = _html(86, 180, 233)
    GREEN = _html(0, 158, 115)
    YELLOW = _html(240, 228, 66)
    BLUE = _html(0, 114, 178)
    VERMILION = _html(213, 94, 0)
    PURPLE = _html(204, 121, 167)
    colors = [GREEN, PURPLE, VERMILION, ORANGE, SKY_BLUE, YELLOW, BLUE, BLACK]
    for i in range(6):
        l1 = ax.plot(estm_x, eigensources[:, i], color=colors[i], lw=2., label='Eigensource ' + str(i+1))
    ax.set_xlim([0, 1])
    if xlabel:
        ax.set_xlabel('Depth ($mm$)')
    if ylabel:
        ax.set_ylabel('CSD ($mA/mm$)')
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    fb.set_axis(ax, letter=letter)
    # ax.legend(frameon=False, loc='upper center', ncol=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.matmul(np.matmul(csd, orthn), orthn.T)


def plot_projection(ax, projection, estm_x, title=None, ele_pos=None,
                 xlabel=False, ylabel=False, letter='', t_max=None,
                 est_csd_LC=None, c='b', label='Projection'):

    x = np.linspace(0, 1, 100)
    l1 = ax.plot(estm_x, projection, ':', color=c, label=label, lw=2.)
    est_csd = projection
    ax.set_xlim([0, 1])
    if xlabel:
        ax.set_xlabel('Depth ($mm$)')
    if ylabel:
        ax.set_ylabel('CSD ($mA/mm$)')
    if title is not None:
        ax.set_title(title)
    if np.max(est_csd) < 1.2:
        ax.set_ylim(-0.2, 1.2)
        s1 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-0.2, 17, 'k')
        s1.set_clip_on(False)
    elif np.max(est_csd) < 1.7:
        ax.set_ylim(-10000, 10000)
        s3 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-10000, 17, 'k')
        s3.set_clip_on(False)
        ax.set_yticks([-7000, 0, 7000])
    if np.max(est_csd) > 500:
        ax.set_ylim(-7000, 7000)
        s3 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-7000, 17, 'k')
        s3.set_clip_on(False)
        ax.set_yticks([-5000, 0, 5000])
    elif np.max(est_csd) > 50:
        ax.set_ylim(-100, 100)
        s2 = ax.scatter(ele_pos, np.zeros(len(ele_pos))-100, 17, 'k')
        s2.set_clip_on(False)
        ax.set_yticks([-70, 0, 70])
    ax.set_xticks([0, 0.5, 1])
    fb.set_axis(ax, letter=letter)
    # ax.legend(frameon=False, loc='upper center', ncol=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def generate_figure_projection(R, MU, n_src, true_csd_xlims, total_ele,
                    method='cross-validation', Rs=None, lambdas=None,
                    noise=0):
    """
    Generates figure for targeted basis investigation.

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
    true_csd_projection = tb.csd_profile(np.linspace(-0.5, 1, 150), [R, MU])
    ele_lims = [0, 0.5]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            true_csd_xlims, R,
                                                            MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)

    fig = plt.figure(figsize=(15, 12))
    widths = [1, 1, 1, 1]
    heights = [1, 1, 1]
    gs = gridspec.GridSpec(3, 4, height_ratios=heights, width_ratios=widths,
                           hspace=0.45, wspace=0.3)


    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    
    ax = fig.add_subplot(gs[0, 0])
    fb.make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title="CSD", xlabel=False, ylabel=True, letter='A')
    
    eigensources = calculate_eigensources(obj)
    projection = csd_into_eigensource_projection(true_csd_projection, eigensources)
    anihilator = true_csd_projection - projection#obj.values('CSD')[:, 0]
    
    ax = fig.add_subplot(gs[0, 1])
    plot_projection(ax, projection, obj.estm_x, ele_pos=ele_pos,
                 title='Projection', xlabel=False, ylabel=False, letter='B')
    
    ax = fig.add_subplot(gs[0, 2])
    plot_projection(ax, anihilator, obj.estm_x, ele_pos=ele_pos,
                 title='Annihilated', xlabel=False, ylabel=False, letter='C', c='r', label='Annihilated')
    
    ax = fig.add_subplot(gs[0, 3])
    make_subplot(ax, eigensources, obj.estm_x, ele_pos=ele_pos,
                 title='Eigensources', xlabel=False, ylabel=False, letter='D')


    ele_lims = [0.5, 1]
    csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                            true_csd_xlims, R,
                                                            MU, total_ele,
                                                            ele_lims,
                                                            noise=noise)
    
    xmin = -0.5
    xmax = 1
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)
    
    ax = fig.add_subplot(gs[1, 0])
    fb.make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=True, letter='E')
    
    eigensources = calculate_eigensources(obj)
    projection = csd_into_eigensource_projection(true_csd_projection, eigensources)
    anihilator = true_csd_projection - projection#obj.values('CSD')[:, 0]
    
    ax = fig.add_subplot(gs[1, 1])
    plot_projection(ax, projection, obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='F')
    
    ax = fig.add_subplot(gs[1, 2])
    plot_projection(ax, anihilator, obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='G', c='r')
    
    ax = fig.add_subplot(gs[1, 3])
    make_subplot(ax, eigensources, obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=False, ylabel=False, letter='H')

    
    true_csd_projection2 = tb.csd_profile(np.linspace(0, 1.5, 150), [R, MU])
    xmin = 0
    xmax = 1.5
    ext_x = -0.5
    obj = tb.modified_bases(val, pots, ele_pos, n_src, h=0.25,
                            sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                            xmax=xmax, method=method, Rs=Rs, lambdas=lambdas)

    ax1 = fig.add_subplot(gs[2, 0])
    ax1 = fb.make_subplot(ax1, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=True, letter='I')
    
    eigensources = calculate_eigensources(obj)
    projection = csd_into_eigensource_projection(true_csd_projection2, eigensources)
    anihilator = true_csd_projection2 - projection#obj.values('CSD')[:, 0]
    
    ax2 = fig.add_subplot(gs[2, 1])
    ax2 = plot_projection(ax2, projection, obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=False, letter='J')
    
    ax3 = fig.add_subplot(gs[2, 2])
    ax3 = plot_projection(ax3, anihilator, obj.estm_x, ele_pos=ele_pos,
                 title=None, xlabel=True, ylabel=False, letter='K', c='r', label='Annihilated')
    
    ax4 = fig.add_subplot(gs[2, 3])
    ax4 = make_subplot(ax4, eigensources, obj.estm_x,
                      ele_pos=ele_pos, title=None, xlabel=True, ylabel=False,
                      letter='L')
    
    handles, labels = [(a + b +c + d) for a, b, c, d in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels(),
                                                            ax3.get_legend_handles_labels(), ax4.get_legend_handles_labels())]
    
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False)

    fig.savefig(os.path.join('Projection_targeted_basis_noise_' + str(noise) + 'normalized2.png'), dpi=300)
    plt.show()


N_SRC = 64
TRUE_CSD_XLIMS = [0., 1.]
TOTAL_ELE = 12
R = 0.2
MU = 0.25
method = 'cross-validation'  # L-curve
#method = 'L-curve'
Rs = np.array([0.2])
lambdas = np.zeros(1)
generate_figure_projection(R, MU, N_SRC, TRUE_CSD_XLIMS, TOTAL_ELE,
                method=method, Rs=Rs, lambdas=lambdas, noise=0)