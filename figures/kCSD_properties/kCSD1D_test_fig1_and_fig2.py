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
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_plot_fig2(sim_results):
    rms_lc = sim_results[0, 2]
    lam_lc = sim_results[0, 0]
    rms_cv = sim_results[1, 2]
    lam_cv = sim_results[1, 0]
    
    fig = plt.figure(figsize=(12, 10), dpi=100)
    plt.subplot(211)
    plt.xlabel('',fontsize = 25)
    n_spec = np.linspace(1e-2,1,19)
    mn_rms = np.mean(rms_lc, axis=0)
    st_rms = np.std(rms_lc, axis=0)
    plt.plot(n_spec, mn_rms, marker = 'o', color = 'blue', label = 'l-curve')
    plt.fill_between(n_spec, mn_rms - st_rms, 
                     mn_rms + st_rms, alpha = 0.3, color = 'blue')
    mn_rms = np.mean(rms_cv, axis=0)
    st_rms = np.std(rms_cv, axis=0)
    plt.plot(n_spec, mn_rms, marker = 'o', color = 'green', label = 'cross-validation')
    plt.fill_between(n_spec, mn_rms - st_rms, 
                    mn_rms + st_rms, alpha = 0.3, color = 'green')
    plt.legend(fontsize = 15)
    #py.xlabel('Noise',fontsize = 25)
    plt.ylabel('Estimation error',fontsize = 18)
    plt.xlabel('Relative noise level',fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    #py.xlim(0, 0.9)
    plt.ylim(0, 100)
    
    '''second plot'''
    plt.subplot(212)
    mn_lam = np.mean(lam_lc, axis=0)
    st_lam = np.std(lam_lc, axis=0)
    plt.plot(n_spec, mn_lam, marker = 'o', color = 'blue', label = 'l-curve')
    plt.fill_between(n_spec, mn_lam - st_lam,
                    mn_lam + st_lam, alpha = 0.3, color = 'blue')
    mn_lam = np.mean(lam_cv, axis=0)
    st_lam = np.std(lam_cv, axis=0)
    plt.plot(n_spec, mn_lam, marker = 'o', color = 'green', label = 'cross-validation')
    plt.fill_between(n_spec, mn_lam - st_lam,
                    mn_lam + st_lam, alpha = 0.3, color = 'green')
    plt.legend(fontsize = 15, loc =2)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    #py.xlim(0, 0.9)
    plt.ticklabel_format(style='sci', axis='y',  scilimits=(0,0))
    plt.ylabel('Lambda',fontsize = 18)
    plt.xlabel('Relative noise level',fontsize = 15)

def do_kcsd(ele_pos, pots, **params):
    """
    Function that calls the KCSD1D module
    """
    num_ele = len(ele_pos)
    pots = pots.reshape(num_ele, 1)
    ele_pos = ele_pos.reshape(num_ele, 1)
    Lamb = [-9, -3]
    lambdas = np.logspace(Lamb[0], Lamb[1], 50, base=10)
    k = KCSD1D(ele_pos, pots, **params)
    if name == 'lc':
        k.L_curve(lambdas=lambdas, Rs=ery)
    else:
        k.cross_validate(Rs=ery, lambdas=lambdas)
    est_csd = k.values('CSD')
    est_pot = k.values('POT')
    return k, est_csd, est_pot

def make_plots(title, m_norm, m_resi, true_csd, curveseq, ele_y,
               pots, k_csd_x, k_csd_y, est_pot, est_csd, save_as):
    """
    Shows 4 plots
    1_ LFP measured (with added noise) and estimated LFP with kCSD method
    2_ true CSD and reconstructed CSD with kCSD
    3_ L-curve of the model
    3_ Surface of parameters R and Lambda with scores for optimal paramater selection with L-curve or cross-validation
    """
    #True CSD
    fig = plt.figure(figsize=(16, 12), dpi=120)
    xrange = np.linspace(0, 1, len(true_csd))
    ax1 = plt.subplot(221)
    ax1.plot(ele_y, pots*1e3, 'r', marker='o', linewidth=0, label='measured potential')
    ax1.scatter(ele_y, np.zeros(len(ele_y)), 8, color='black')
    ax1.plot(xrange, est_pot*1e3, label='recon. potential', color='blue')
    ax1.set_ylabel('Potential [mV]', fontsize=15)
    ax1.set_xlabel('Distance', fontsize=15)
    plt.legend()
    
    ax2 = plt.subplot(222)
    plt.plot(xrange, true_csd, label='true csd', color='red')
    plt.plot(xrange, est_csd, label='recon. csd', color='blue')
    plt.scatter(ele_y, np.zeros(len(ele_y)), 8, color='black')
    ax2.set_ylabel('CSD [mA/mm]', fontsize=15)
    ax2.set_xlabel('Distance', fontsize=15)
    plt.legend(loc=1)
    Lamb = [-9, -3]
    imax = np.argmax(curveseq[np.argmax(np.max(curveseq, axis=-1))])

    ax_L = fig.add_subplot(223)
    plt.ylabel("Norm of the model", fontsize=15)
    plt.xlabel("Norm of the prediction error", fontsize=15)
    ax_L.plot(m_resi, m_norm, marker=".", c="green")
    #print(m_resi, m_norm)
    ax_L.plot([m_resi[imax]], [m_norm[imax]], marker="o", c="red")
    x = [m_resi[0], m_resi[imax], m_resi[-1]]
    y = [m_norm[0], m_norm[imax], m_norm[-1]]
    ax_L.fill(x, y, alpha=0.2)
    ax_L.set_xscale('log')
    ax_L.set_yscale('log')

    ax4 = fig.add_subplot(224)
    im = plt.imshow(curveseq, extent=[Lamb[0], Lamb[1], ery[-1], ery[0]],
                    interpolation='none', aspect='auto',
                    cmap='BrBG_r', vmax=np.max(curveseq), vmin=-np.max(curveseq))
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax = cax4)
    ax4.set_ylabel('R parameter', fontsize=15)
    ax4.set_xlabel('log(Lambda)', fontsize=15)
#    fig.suptitle("Lambda =" + title[0] + " I R = " + title[1] + 
#                 " I Noise_lvl =" + title[2])
    fig.savefig(save_as+'.png')

    return np.linalg.norm(true_csd - est_csd)

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
    if type(noise) ==  'numpy.float64':
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

        k, est_csd, est_pot = do_kcsd(ele_y, pots, h=1., gdx=gdX,
                                      xmin=x_lims[0], xmax=x_lims[1], n_src_init=1e4)

        save_as = nm + '_noise' + str(np.round(noise*100, 1))
        if name == 'lc':
            m_norm = k.m_norm
            m_resi = k.m_resi
            curve_surf = k.curve_surf
        else:
            m_norm = k.errs[0]
            m_resi = np.arange(len(k.errs[1]))
            curve_surf = k.errs
        title = [str(k.lambd), str(k.R), str(noise), nm]
        RMS_wek[i] = make_plots(title, m_norm, m_resi, true_csd, curve_surf,
                               ele_y, pots, k.estm_x, k.estm_x, est_pot, est_csd, save_as)
        LandR[0, i] = k.lambd
        LandR[1, i] = k.R
    return RMS_wek, LandR

if __name__=='__main__':
    saveDir = "./LCurve/"
    figs1_and_fig2 = False
    total_ele = 32
    names = ['lc', 'cv']
    src_width = 0.001
    noise_lvl = np.linspace(1e-2, 1, 19)
    inpos = [0.5, 0.1]#od dolu
    lpos = [0.5, 0.8]
    ery = np.linspace(0.01, 0.1, 10)
    sim_results = np.zeros((2, 4, 10, 19))
    sim_results[:, 3, :, :] = noise_lvl
    if figs1_and_fig2:
        for iname, name in enumerate(names): 
            for src in range(10):
                print('noise seed:', src)
                mypath = saveDir + name + str(src) + '/'
                if not os.path.isdir(mypath):
                    os.makedirs(mypath)
                os.chdir(mypath)
                RMS_wek, LandR = main_loop(src_width, total_ele, inpos,
                                           lpos, name, noise=noise_lvl, srcs=src)
                sim_results[iname,:2] = LandR
                sim_results[iname, 2] = RMS_wek
        make_plot_fig2(sim_results)
    else:
        name = 'lc'
        main_loop(src_width, total_ele, inpos,
                  lpos, name, noise=noise_lvl[:1], srcs=0)