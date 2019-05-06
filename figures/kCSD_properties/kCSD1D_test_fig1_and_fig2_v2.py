'''
This script is used to test Current Source Density Estimates, 
using the kCSD method Jan et.al (2012)

This script is in alpha phase.

This was written by :
Chaitanya Chintaluri, 
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
'''
import os
import scipy.stats as st
import numpy as np
from scipy.integrate import simps 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from kcsd.corelib.KCSDgit import KCSD1D
from kcsd import KCSD1D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter
from figure_properties import *


def set_axis(ax, x, y, letter=None):
    ax.text(
        x,
        y,
        letter,
        fontsize=15,
        weight='bold',
        transform=ax.transAxes)
    return ax

def make_plot_fig2(sim_results):
    rms_lc = sim_results[0, 2]
    lam_lc = sim_results[0, 0]
    rms_cv = sim_results[1, 2]
    lam_cv = sim_results[1, 0]
    
    fig = plt.figure(figsize = (12,12), dpi = 300)
    widths = [10]
    heights = [1, 1]
    gs = gridspec.GridSpec(2, 1, height_ratios=heights, width_ratios=widths,
                           hspace=0.45, wspace=0.3)
    ax1 = plt.subplot(gs[0])
    if np.min(rms_cv) < np.min(rms_lc):
        trans = np.min(np.mean(rms_cv, axis=0))
    else:
        trans = np.min(np.mean(rms_lc, axis=0))
    mn_rms = np.mean(rms_lc, axis=0) - trans
    st_rms = st.sem(rms_lc, axis=0)
    plt.plot(noise_lvl, mn_rms, marker = 'o', color = 'blue', label = 'kCSD L-Curve')
    plt.fill_between(noise_lvl, mn_rms - st_rms, 
                     mn_rms + st_rms, alpha = 0.3, color = 'blue')
    mn_rms = np.mean(rms_cv, axis=0) - trans
    st_rms = st.sem(rms_cv, axis=0)
    plt.plot(noise_lvl, mn_rms, marker = 'o', color = 'green', label = 'kCSD Cross-Validation')
    plt.fill_between(noise_lvl, mn_rms - st_rms, 
                    mn_rms + st_rms, alpha = 0.3, color = 'green')
    plt.ylabel('Estimation error', labelpad = 30)
    plt.xlabel('Relative noise level', labelpad = 15)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    set_axis(ax1, -0.05, 1.05, letter='A')
    plt.title('Comparison of the regularization methods performance')

    '''second plot'''
    ax2 = plt.subplot(gs[1])
    mn_lam = np.mean(lam_lc, axis=0)
    st_lam = st.sem(lam_lc, axis=0)
    plt.plot(noise_lvl, mn_lam, marker = 'o', color = 'blue', label = 'kCSD L-Curve')
    plt.fill_between(noise_lvl, mn_lam - st_lam,
                    mn_lam + st_lam, alpha = 0.3, color = 'blue')
    mn_lam = np.mean(lam_cv, axis=0)
    st_lam = st.sem(lam_cv, axis=0)
    plt.plot(noise_lvl, mn_lam, marker = 'o', color = 'green', label = 'kCSD Cross-Validation')
    plt.fill_between(noise_lvl, mn_lam - st_lam,
                    mn_lam + st_lam, alpha = 0.3, color = 'green')
#    ax2.set_yscale('log')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=((0.0, 0.0)))
    plt.ylabel('Lambda', labelpad = 20)
    plt.xlabel('Relative noise level', labelpad = 15)
    set_axis(ax2, -0.05, 1.05, letter='B')
    ht, lh = ax2.get_legend_handles_labels()
    fig.legend(ht, lh, loc='lower center', ncol=2, frameon=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.savefig('stats.jpg')

def do_kcsd(ele_pos, pots, **params):
    """
    Function that calls the KCSD1D module
    """
    num_ele = len(ele_pos)
    pots = pots.reshape(num_ele, 1)
    ele_pos = ele_pos.reshape(num_ele, 1)
    Lamb = [-7, -3]
    lambdas = np.logspace(Lamb[0], Lamb[1], 50, base=10)
    k = KCSD1D(ele_pos, pots, **params)
    noreg_csd = k.values('CSD')
    if name == 'lc':
        k.L_curve(lambdas=lambdas, Rs=ery)
    else:
        k.cross_validate(Rs=ery, lambdas=lambdas)
    est_csd = k.values('CSD')
    est_pot = k.values('POT')
    return k, est_csd, est_pot, noreg_csd

def make_plots(title, m_norm, m_resi, true_csd, curveseq, ele_y,
               pots, k_csd_x, k_csd_y, est_pot, est_csd, noreg_csd, save_as):
    """
    Shows 4 plots
    1_ LFP measured (with added noise) and estimated LFP with kCSD method
    2_ true CSD and reconstructed CSD with kCSD
    3_ L-curve of the model
    4_ Surface of parameters R and Lambda with scores for optimal paramater selection with L-curve or cross-validation
    """
    #True CSD
    fig = plt.figure(figsize=(12, 12), dpi=300)
    widths = [1, 1]
    heights = [1, 1]
    xpad=5
    ypad = 10
    gs = gridspec.GridSpec(2, 2, height_ratios=heights, width_ratios=widths,
                           hspace=0.5, wspace=0.5)
    xrange = np.linspace(0, 1, len(true_csd))
    ax1 = plt.subplot(gs[0])
    ax1.plot(ele_y, pots*1e3, 'r', marker='o', linewidth=0, label='Measured potential')
    ax1.scatter(ele_y, np.zeros(len(ele_y)), 8, color='black', label = "Electrode position")
    ax1.plot(xrange, est_pot*1e3, label='Reconstructed potential', color='blue')
    ax1.set_ylabel('Potential ($mV$)', labelpad = ypad)
    ax1.set_xlabel('Distance', labelpad = xpad)
    ax1.tick_params(axis='both', which='major')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    set_axis(ax1, -0.05, 1.05, letter='A')
    ax1.legend(bbox_to_anchor=(1.5, -0.16), ncol=2, frameon=False)
    
    ax_L = plt.subplot(gs[1])
    Lamb = [-7, -3]
    if name == 'lc':
        imax = np.argmax(curveseq[np.argmax(np.max(curveseq, axis=-1))])
        plt.ylabel("Norm of the model", labelpad = ypad)
        plt.xlabel("Norm of the prediction error", labelpad = xpad)
        ax_L.plot(m_resi, m_norm, marker=".", c="green", label = 'L-Curve')
    else:
        imax = np.argmin(m_norm)
        plt.xlabel("Lambda", labelpad = xpad)
        plt.ylabel("CV error", labelpad = ypad)
        ax_L.plot(m_resi, m_norm, marker=".", c="green", label = 'CV curve')
    #print(m_resi, m_norm)
    ax_L.plot([m_resi[imax]], [m_norm[imax]], marker="o", c="red")
    x = [m_resi[0], m_resi[imax], m_resi[-1]]
    y = [m_norm[0], m_norm[imax], m_norm[-1]]
    ax_L.fill(x, y, alpha=0.2)
    ax_L.set_xscale('log')
    ax_L.set_yscale('log')
    ax_L.tick_params(axis='both', which='major')
    ax_L.spines['right'].set_visible(False)
    ax_L.spines['top'].set_visible(False)
    set_axis(ax_L, -0.05, 1.05, letter='B')
    ax_L.legend(bbox_to_anchor=(0.7, -0.16), ncol=1, frameon=False)

    ax2 = plt.subplot(gs[2])
    plt.plot(xrange, true_csd, label='True CSD', color='red', linestyle = '--')
    plt.plot(xrange, est_csd, label='kCSD + regularization', color='blue')
    plt.plot(xrange, noreg_csd, label='kCSD', color='darkgreen', alpha = 0.6)
    plt.ylim(-1, 1)
    plt.scatter(ele_y, np.zeros(len(ele_y)), 8, color='black', label = "Electrode position")
    ax2.set_ylabel('CSD ($mA/mm$)', labelpad = ypad)
    ax2.set_xlabel('Distance', labelpad = xpad)
    ax2.tick_params(axis='both', which='major')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(bbox_to_anchor=(-.25, -0.25), ncol=2, frameon=False, loc = 'center left')
    set_axis(ax2, -0.05, 1.05, letter='C')
    
    ax4 = plt.subplot(gs[3])
    if name == 'lc':
        Lamb = [-7, -3]
        lambdas = np.logspace(Lamb[0], Lamb[1], 50, base=10)
        ax4.plot(lambdas, curveseq[0], marker=".", label = 'Curvature evaluation')
        ax4.plot([lambdas[imax]], [curveseq[0][imax]], marker="o", c="red")
        ax4.set_ylabel('Curvature', labelpad = ypad)
        ax4.set_xlabel('Lambda', labelpad = xpad)
        ax4.set_xscale('log')
        ax4.tick_params(axis='both', which='major')
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.legend(bbox_to_anchor=(1, -0.16), ncol=2, frameon=False)
    else:
        imax = np.argmin(m_norm)
        plt.ylabel("CV error", labelpad = ypad)
        plt.xlabel("Lambda", labelpad = xpad)
        ax4.plot(m_resi, m_norm, marker=".", c="green",  label = 'CV curve')
        ax4.plot([m_resi[imax]], [m_norm[imax]], marker="o", c="red")
        x = [m_resi[0], m_resi[imax], m_resi[-1]]
        y = [m_norm[0], m_norm[imax], m_norm[-1]]
        ax4.fill(x, y, alpha=0.2)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.tick_params(axis='both', which='major')
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.legend(bbox_to_anchor=(1, -0.16), ncol=2, frameon=False)
    set_axis(ax4, -0.05, 1.05, letter='D')
    fig.savefig(save_as+'.jpg')
    true_csd_error = np.linalg.norm(true_csd/np.max(abs(true_csd)) - est_csd/np.max(abs(est_csd)))
    print('true_csd_error:', true_csd_error)
    return true_csd_error

def generate_csd_1D(src_width, name, srcs=np.array([0]),
                    start_x=0., end_x=1.,
                    start_y=0., end_y=1.,
                    res_x=50, res_y=50):
    """
    Gives CSD profile at the requested spatial location, at 'res' resolution
    """
    csd_y = np.linspace(start_x, end_x, res_x)
    csd_x = np.linspace(start_x, end_x, res_x)
#    f = prof.gauss_2d_small(csd_x, csd_y, src_width, tpy=name, srcs=srcs)
    src_peak = np.exp(-((csd_x-0.25)**2)/(2.*src_width))
    src_thr1 = .5*np.exp(-((csd_x-0.5)**2)/(2.*src_width))
    src_thr2 = .5*np.exp(-((csd_x-0.6)**2)/(2.*src_width))
    f = src_peak - src_thr1 - src_thr2
    csd_y = np.linspace(start_x, end_x, res_x)
    return csd_x, csd_y, f

def integrate_1D(x0, csd_x, csd, h):
    m = np.sqrt((csd_x-x0)**2 + h**2) - abs(csd_x-x0)
    y = csd * m
    I = simps(y, csd_x)
    return I

def calculate_potential_1D(csd, measure_locations, csd_space_x, h):
    sigma = 1
    print(measure_locations.shape, csd_space_x.shape, csd.shape)
    pots = np.zeros(len(measure_locations))
    for ii in range(len(measure_locations)):
        pots[ii] = integrate_1D(measure_locations[ii], csd_space_x, csd, h)
    pots *= 1/(2.*sigma) #eq.: 26 from Potworowski et al
    return pots

def transformAndNormalizeLongGrid(D):
    #D is a dictionary
    locations = np.zeros((len(D), 2))
    for i in range(len(D)):
        locations[i, 0] = D[i]['x']
        locations[i, 1] = D[i]['y']
    return locations

def generate_electrodes(inpos, lpos, b):
    loc = {}
    cordx = -(inpos[0] - lpos[0])/b
    cordy = -(inpos[1] - lpos[1])/b
    def _addLocToDict(D, chnum, x, y):
        d = {}
        d['x'] = x
        d['y'] = y
        D[chnum] = d
    #first add locations just as they suppose to be
    for i in range(b):
        _addLocToDict(loc, i, inpos[0] + cordx*i, inpos[1] + cordy*i)
    loc_mtrx = transformAndNormalizeLongGrid(loc)
    loc_mtrx = loc_mtrx.T
    return loc_mtrx[0], loc_mtrx[1]

def electrode_config(ele_res, true_csd, csd_x, csd_y, inpos, lpos):
    """
    What is the configuration of electrode positions, between what and what positions
    """
    ele_x, ele_y = generate_electrodes(inpos, lpos, ele_res)
    pots = calculate_potential_1D(true_csd, ele_y.T, csd_y, h=1)
    ele_pos = np.vstack((ele_x, ele_y)).T     #Electrode configs
    return ele_pos, pots

def main_loop(src_width, total_ele, inpos, lpos, nm, noise=0, srcs=1):
    """
    Loop that decides the random number seed for the CSD profile,
    electrode configurations and etc.
    """
    #TrueCSD
    t_csd_x, t_csd_y, true_csd = generate_csd_1D(src_width, nm, srcs=srcs,
                                                 start_x=0, end_x=1.,
                                                 start_y=0, end_y=1,
                                                 res_x=100, res_y=100)
    if type(noise) ==  float:
        n_spec = [noise]
    else:
        n_spec = noise
    RMS_wek = np.zeros(len(n_spec))
    LandR = np.zeros((2, len(n_spec)))
    for i, noise in enumerate(n_spec):
        plt.close('all')
        noise = np.round(noise, 5)
        print('numer rekonstrukcji: ', i, 'noise level: ', noise)
        #Electrodes
        ele_pos, pots = electrode_config(total_ele, true_csd, t_csd_x, t_csd_y, inpos, lpos)
        ele_y = ele_pos[:, 1]
        gdX = 0.01
        x_lims = [0, 1] #CSD estimation place
        np.random.seed(srcs)
        pots += (np.random.rand(total_ele)*np.max(abs(pots))-np.max(abs(pots))/2)*noise

        k, est_csd, est_pot, noreg_csd = do_kcsd(ele_y, pots, h=1., gdx=gdX, sigma = 1,
                                                 xmin=x_lims[0], xmax=x_lims[1], n_src_init=1e4)

        save_as = nm + '_noise' + str(np.round(noise*100, 1))
        if name == 'lc':
            m_norm = k.m_norm
            m_resi = k.m_resi
            curve_surf = k.curve_surf
        else:
            m_norm = k.errs[0]
            m_resi = np.arange(len(k.errs[0]))
            curve_surf = k.errs
        title = [str(k.lambd), str(k.R), str(noise), nm]
        RMS_wek[i] = make_plots(title, m_norm, m_resi, true_csd, curve_surf,
                                ele_y, pots, k.estm_x, k.estm_x, est_pot, est_csd, noreg_csd, save_as)
        LandR[0, i] = k.lambd
        LandR[1, i] = k.R
    return RMS_wek, LandR

if __name__=='__main__':
    saveDir = "./LCurve/"
    try:
        os.chdir(saveDir)
    except FileNotFoundError:
        os.mkdir(saveDir)
        os.chdir(saveDir)
    figs1_and_fig2 = True
    total_ele = 32
    names = ['lc', 'cv']
    src_width = 0.001
    noises = 3
    seeds = 3
    inpos = [0.5, 0.1]#od dolu
    lpos = [0.5, 0.9]
    noise_lvl = np.linspace(0, 0.5, noises)
    ery = np.linspace(3*0.025, 0.025*16, 1)
    sim_results = np.zeros((2, 4, seeds, noises))
    sim_results[:, 3, :, :] = noise_lvl
    if figs1_and_fig2:
        for iname, name in enumerate(names): 
            for src in range(seeds):
                print('noise seed:', src)
                mypath = name + str(src) + '/'
                if not os.path.isdir(mypath):
                    os.makedirs(mypath)
                os.chdir(mypath)
                RMS_wek, LandR = main_loop(src_width, total_ele, inpos,
                                           lpos, name, noise=noise_lvl, srcs=src)
                sim_results[iname,:2, src] = LandR
                sim_results[iname, 2, src] = RMS_wek
                os.chdir('..')
        np.save('sim_results', sim_results)
        sim_results = np.load('sim_results.npy')
        make_plot_fig2(sim_results)
    else:
        name = 'lc'
        main_loop(src_width, total_ele, inpos,
                  lpos, name, noise=noise_lvl[:1], srcs=0)
