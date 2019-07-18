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
from kcsd import KCSD1D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as colors
from figure_properties import *

def set_axis(ax, x, y, letter=None):
    ax.text(x,
            y,
            letter,
            fontsize=15,
            weight='bold',
            transform=ax.transAxes)
    return ax

def plot_surface(obj, errsy, save_as):
    fsize = 15
    fig = plt.figure(figsize = (20,9), dpi = 300)
    gs = gridspec.GridSpec(16, 12, hspace=2, wspace=2)
    ax = plt.subplot(gs[0:16, 0:6])
    set_axis(ax, -0.05, 1.05, letter='A')
    plt.pcolormesh(lambdas, np.arange(9), obj.curve_surf, 
                   cmap = 'BrBG', vmin = -2, vmax=2)
    plt.colorbar()
    for i,m in enumerate(obj.curve_surf.argmax(axis=1)):
        plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', alpha = 0.7)
        if i==7:
            plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', 
                        label = 'Maximum Curvature', alpha = 0.7)
    plt.xlim(lambdas[1],lambdas[-1])
    plt.title('L-curve regularization',  fontsize = fsize)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.12), ncol=1, 
               frameon = False, fontsize = fsize)
    plt.yticks(np.arange(8)+0.5, [str(x)+'x' for x in range(1,9)])
    plt.xscale('log')
    plt.ylabel('Parameter $R$ in electrode distance', fontsize=fsize, labelpad = 15)
    plt.xlabel('$\lambda$',fontsize=fsize)
    ax = plt.subplot(gs[0:16, 6:12])
    set_axis(ax, -0.05, 1.05, letter='B')
    plt.pcolormesh(lambdas, np.arange(9), errsy, 
                   cmap = 'Greys_r', vmin= 0, vmax = np.max(errsy))
    plt.colorbar()
    for i,m in enumerate(errsy.argmin(axis=1)):
        plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', alpha = 0.7)
        if i==7:
            plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', 
                        label = 'Minimum Error', alpha = 0.7)
    plt.xlim(lambdas[1],lambdas[-1])
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.12), ncol=1, 
               frameon = False, fontsize = 15)
    plt.title('Cross-validation regularization', fontsize = fsize)
    plt.yticks(np.arange(8)+0.5, [str(x)+'x' for x in range(1,9)])
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    fig.savefig(save_as+'.jpg')
    
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
    plt.title('Performance of regularization methods')

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
    plt.ylabel('$\lambda$', labelpad = 20)
    plt.xlabel('Relative noise level', labelpad = 15)
    set_axis(ax2, -0.05, 1.05, letter='B')
    ht, lh = ax2.get_legend_handles_labels()
    fig.legend(ht, lh, loc='lower center', ncol=2, frameon=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.savefig('stats.jpg')

def make_plots(title, m_norm, m_resi, true_csd, curveseq, ele_y,
               pots_n, pots, k_csd_x, k_csd_y, est_pot, est_csd, noreg_csd, save_as):
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
    xpad= 5
    ypad = 10
    gs = gridspec.GridSpec(2, 2, height_ratios=heights, width_ratios=widths,
                           hspace=0.6, wspace=0.6)
    xrange = np.linspace(0, 1, len(true_csd))
    ax1 = plt.subplot(gs[0])
    ax1.plot(ele_y, pots_n*1e3, 'r', marker='o', linewidth=0, label='Measured potential with noise')
    ax1.plot(ele_y, pots*1e3, 'b', marker='o', linewidth=0, label='Measured potential')
    ax1.scatter(ele_y, np.zeros(len(ele_y))-11, 8, color='black', 
                clip_on=False, label = "Electrode position")
    ax1.plot(xrange, est_pot*1e3, label='Reconstructed potential', color='blue')
    plt.ylim(-11,11)
    ax1.set_ylabel('Potential ($mV$)', labelpad = ypad+2)
    ax1.set_xlabel('Distance ($mm$)', labelpad = xpad)
    ax1.tick_params(axis='both', which='major')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    set_axis(ax1, -0.05, 1.05, letter='A')
    ax1.legend(bbox_to_anchor=(1.7, -0.2), ncol=2, frameon=False)
    
    ax_L = plt.subplot(gs[1])
    m_resi = np.log(m_resi)
    m_norm = np.log(m_norm)
    imax = np.argmax(curveseq[np.argmax(np.max(curveseq, axis=-1))])
    imax2 = np.argmax(np.max(curveseq, axis=-1))
    plt.ylabel("Log norm of the model", labelpad = ypad)
    plt.xlabel("Log norm of the prediction error", labelpad = xpad)
    ax_L.plot(m_resi, m_norm, marker=".", c="green", label = 'L-Curve')
    ax_L.plot([m_resi[imax]], [m_norm[imax]], marker="o", c="red")
    x = [m_resi[0], m_resi[imax], m_resi[-1]]
    y = [m_norm[0], m_norm[imax], m_norm[-1]]
    ax_L.fill(x, y, alpha=0.2)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax_L.spines['right'].set_visible(False)
    ax_L.spines['top'].set_visible(False)
    set_axis(ax_L, -0.05, 1.05, letter='B')
    ax_L.legend(bbox_to_anchor=(0.7, -0.2), ncol=1, frameon=False)

    ax2 = plt.subplot(gs[2])
    plt.plot(xrange, true_csd, label='True CSD', color='red', linestyle = '--')
    plt.plot(xrange, est_csd, label='kCSD + regularization', color='blue')
    plt.plot(xrange, noreg_csd, label='kCSD', color='darkgreen', alpha = 0.6)
    plt.scatter(ele_y, np.zeros(len(ele_y))-1, 8, color='black', 
                label = "Electrode position", clip_on=False)
    ax2.set_ylabel('CSD ($mA/mm$)', labelpad = ypad)
    ax2.set_xlabel('Distance ($mm$)', labelpad = xpad)
    plt.ylim(-1, 1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(bbox_to_anchor=(-.25, -0.27), ncol=2, frameon=False, loc = 'center left')
    set_axis(ax2, -0.05, 1.05, letter='C')
    
    ax4 = plt.subplot(gs[3])
    ax4.plot(lambdas, curveseq[imax2], marker=".", label = 'Curvature evaluation')
    ax4.plot([lambdas[imax]], [curveseq[imax2][imax]], marker="o", c="red")
    ax4.set_ylabel('Curvature', labelpad = ypad)
    ax4.set_xlabel('$\lambda$', labelpad = xpad)
    ax4.set_xscale('log')
    ax4.tick_params(axis='both', which='major')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.legend(bbox_to_anchor=(1, -0.2), ncol=2, frameon=False)
    set_axis(ax4, -0.05, 1.05, letter='D')
    fig.savefig(save_as+'.jpg')

def do_kcsd(i, ele_pos, pots, **params):
    """
    Function that calls the KCSD1D module
    """
    num_ele = len(ele_pos)
    pots = pots.reshape(num_ele, 1)
    ele_pos = ele_pos.reshape(num_ele, 1)
    k = KCSD1D(ele_pos, pots, **params)
    noreg_csd = k.values('CSD')
    k.cross_validate(Rs=ery, lambdas=lambdas)
    errsy = k.errs
    LandR[1,0,i] = k.lambd
    LandR[1,1,i] = k.R
    est_csd_cv = k.values('CSD') 
    k.L_curve(lambdas=lambdas, Rs=ery)
    LandR[0,0,i] = k.lambd
    LandR[0,1,i] = k.R
    est_csd = k.values('CSD')
    est_pot = k.values('POT')
    return k, est_csd, est_pot, noreg_csd, errsy, est_csd_cv

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
    src_peak = .7*np.exp(-((csd_x-0.25)**2)/(2.*src_width))
    src_thr1 = .35*np.exp(-((csd_x-0.45)**2)/(2.*src_width))
    src_thr2 = .35*np.exp(-((csd_x-0.65)**2)/(2.*src_width))
#    src_peak2 = .35*np.exp(-((csd_x-0.55)**2)/(2.*src_width))
    f = src_peak - src_thr1-src_thr2#+src_peak2
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
    if type(noise) ==  float: n_spec = [noise]
    else: n_spec = noise
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
        pots_n = pots +(np.random.rand(total_ele)*np.max(abs(pots))-np.max(abs(pots))/2)*noise
        k, est_csd, est_pot, noreg_csd, errsy, est_csd_cv = do_kcsd(i, ele_y, pots_n, h=1., gdx=gdX, sigma = 1,
                                                                    xmin=x_lims[0], xmax=x_lims[1], n_src_init=1e3)
        save_as = nm + '_noise' + str(np.round(noise*100, 1))
        m_norm = k.m_norm
        m_resi = k.m_resi
        curve_surf = k.curve_surf
        title = [str(k.lambd), str(k.R), str(noise), nm]
        make_plots(title, m_norm, m_resi, true_csd, curve_surf, ele_y, pots_n, 
                   pots, k.estm_x, k.estm_x, est_pot, est_csd, noreg_csd, save_as)
        plot_surface(k, errsy, save_as+'surf')
        RMS_wek[0, i] = np.linalg.norm(true_csd/np.linalg.norm(true_csd) - est_csd[:,0]/np.linalg.norm(est_csd[:,0]))
        RMS_wek[1, i] = np.linalg.norm(true_csd/np.linalg.norm(true_csd) - est_csd_cv[:,0]/np.linalg.norm(est_csd_cv[:,0]))

if __name__=='__main__':
    saveDir = "./LCurve_pselection/"
    try:
        os.chdir(saveDir)
    except FileNotFoundError:
        os.mkdir(saveDir)
        os.chdir(saveDir)
    figs1_and_fig2 = True
    total_ele = 32
    src_width = 0.001
    noises = 3
    seeds = 3
    inpos = [0.5, 0.1]#od dolu
    lpos = [0.5, 0.9]
    noise_lvl = np.linspace(0, 0.5, noises)
    ery = np.linspace(1*0.025, 8*0.025, 8)
    lambdas = np.logspace(-7, -3, 50)
    sim_results = np.zeros((2, 4, seeds, noises))
    sim_results[:, 3, :, :] = noise_lvl
    for src in range(seeds):
        print('noise seed:', src)
        RMS_wek = np.zeros((2,len(noise_lvl)))
        LandR = np.zeros((2,2, len(noise_lvl)))
        mypath = 'lc' + str(src) + '/'
        if not os.path.isdir(mypath): os.makedirs(mypath)
        os.chdir(mypath)
        main_loop(src_width, total_ele, inpos,
                  lpos, 'lc', noise=noise_lvl, srcs=src)
        sim_results[:,:2, src] = LandR
        sim_results[:, 2, src] = RMS_wek
        os.chdir('..')
    np.save('sim_results', sim_results)
    sim_results = np.load('sim_results.npy')
    make_plot_fig2(sim_results)

