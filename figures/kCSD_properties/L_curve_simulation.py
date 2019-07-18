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
from kcsd import KCSD1D
from figure4_and_13 import plot_surface, make_plots
from figure5_perf import make_plot_perf

def do_kcsd(i, ele_pos, pots, **params):
    """
    Function that calls the KCSD1D module
    """
    num_ele = len(ele_pos)
    pots = pots.reshape(num_ele, 1)
    ele_pos = ele_pos.reshape(num_ele, 1)
    k = KCSD1D(ele_pos, pots, **params)
    noreg_csd = k.values('CSD')
    k.cross_validate(Rs=Rs, lambdas=lambdas)
    errsy = k.errs
    LandR[1,0,i] = k.lambd
    LandR[1,1,i] = k.R
    est_csd_cv = k.values('CSD')
    k.L_curve(lambdas=lambdas, Rs=Rs)
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
    src_peak = .7*np.exp(-((csd_x-0.25)**2)/(2.*src_width))
    src_thr1 = .35*np.exp(-((csd_x-0.45)**2)/(2.*src_width))
    src_thr2 = .35*np.exp(-((csd_x-0.65)**2)/(2.*src_width))
    f = src_peak-src_thr1-src_thr2
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
        pots_n = pots + (np.random.rand(total_ele)*np.max(abs(pots))-np.max(abs(pots))/2)*noise
        k, est_csd, est_pot, noreg_csd, errsy, est_csd_cv = do_kcsd(i, ele_y, pots_n, h=1., gdx=gdX, sigma = 1,
                                                                    xmin=x_lims[0], xmax=x_lims[1], n_src_init=1e3)
        save_as = nm + '_noise' + str(np.round(noise*100, 1))
        m_norm = k.m_norm
        m_resi = k.m_resi
        curve_surf = k.curve_surf
        title = [str(k.lambd), str(k.R), str(noise), nm]
        make_plots(title, m_norm, m_resi, true_csd, curve_surf, ele_y, pots_n,
                   pots, k.estm_x, est_pot, est_csd, noreg_csd, save_as)
        plot_surface(curve_surf, errsy, save_as+'surf')
        vals_to_save = {'m_norm':m_norm, 'm_resi':m_resi, 'true_csd':true_csd, 
                        'curve_surf':curve_surf, 'ele_y':ele_y, 'pots_n':pots_n,
                        'pots':pots, 'estm_x':k.estm_x, 'est_pot':est_pot, 
                        'est_csd':est_csd, 'noreg_csd':noreg_csd, 'errsy':errsy}
        np.savez('data_fig4_and_fig13_'+save_as, **vals_to_save)
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
    noises = 9
    seeds = 9
    inpos = [0.5, 0.1]#od dolu
    lpos = [0.5, 0.9]
    noise_lvl = np.linspace(0, 0.5, noises)
    Rs = np.linspace(0.025, 8*0.025, 8)
    lambdas = np.logspace(-7, -3, 50)
    sim_results = np.zeros((2, 4, seeds, noises))
    sim_results[:, 3] = noise_lvl
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
    make_plot_perf(sim_results)

