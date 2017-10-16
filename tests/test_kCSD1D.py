'''
This script is used to test the Current Source Density Estimates,
using the kCSD method Potworowski et.al (2012) for 1D case

This script is in alpha phase.

This was written by:
Michal Czerwinski, Chaitanya Chintaluri,
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
'''
import time
import os
import sys
sys.path.append('..')

import numpy as np

import matplotlib.pyplot as plt

import csd_profile as CSD 
from KCSD1D import KCSD1D




def electrode_config(ele_lims, ele_res, true_csd, t_csd_x, h):
    """
    creates electrodes positions, and potentials on them
    electrode lims, electrode resolution, profile, states
    """
    ele_x = generate_electrodes(ele_lims, ele_res)
    pots = calculate_potential_1D(true_csd, ele_x, t_csd_x, h)
    ele_pos = ele_x.reshape((len(ele_x), 1))
    return ele_pos, pots

def do_kcsd(ele_pos, pots, **params):
    """
    Function that calls the KCSD2D module
    """
    num_ele = len(ele_pos)
    pots = pots.reshape(num_ele, 1)
    k = KCSD1D(ele_pos, pots, **params)
    #k.cross_validate(Rs=np.arange(0.01,0.2,0.01), lambdas= np.logspace(15,-25,25))
    k.cross_validate(Rs=np.array([0.275]), lambdas=np.logspace(15,-25, 35))
    est_csd = k.values()
    est_pot =  k.values('POT')
    return k, est_csd, est_pot

def main_loop(csd_profile, csd_seed, total_ele):
    """
    Loop that decides the random number seed for the CSD profile,
    electrode configurations and etc.
    """
    csd_name = csd_profile.func_name
    print 'Using sources %s - Seed: %d ' % (csd_name, csd_seed)
    h = 10.

    #TrueCSD
    start_x, end_x, csd_res = [0.,1.,100]    
    t_csd_x, true_csd = utils.generate_csd_1D(csd_profile, csd_seed, 
                                              start_x=start_x, 
                                              end_x=end_x, 
                                              res_x=csd_res)
    
    #Electrodes 
    ele_res = int(total_ele) 
    ele_lims = [0.10, 0.9]
    ele_pos, pots = utils.electrode_config_1D(ele_lims, ele_res, true_csd, t_csd_x, h)
    num_ele = ele_pos.shape[0]
    print 'Number of electrodes:', num_ele
    x_array_pots, true_pots = utils.electrode_config_1D(ele_lims, 100, true_csd, t_csd_x, h)

    #kCSD estimation
    gdX = 0.01
    x_lims = [0.,1.] #CSD estimation place
    k, est_csd, est_pot = do_kcsd(ele_pos, pots, h=h, gdx=gdX,
                                  xmin=x_lims[0], xmax=x_lims[1], n_src_init=300)

    #RMS of estimation - gives estimate of how good the reconstruction was
    chr_x, test_csd = generate_csd_1D(csd_profile, csd_seed,
                                      start_x=x_lims[0], end_x=x_lims[1], 
                                      res_x=int((x_lims[1]-x_lims[0])/gdX))
    rms = np.linalg.norm(abs(test_csd - est_csd[:,0]))
    rms /= np.linalg.norm(test_csd)

    #Plots
    title ="Lambda: %0.2E; R: %0.2f; CV_Error: %0.2E; RMS_Error: %0.2E; Time: %0.2f" %(k.lambd, k.R, k.cv_error, rms)
    make_plots(title, t_csd_x, true_csd, ele_pos, pots, k.estm_x, est_csd, est_pot, true_pots)
    return

def test_calculating_potentials(csd_seed):
    csd_profile = CSD.gauss_1d_mono
    plt.figure('csd')
    _csdx = np.linspace(0,1,100)
    plt.plot(_csdx, CSD.gauss_1d_mono(_csdx, csd_seed))
    #constant measure_locations, constant csd_space_lims, different csd resolution
    plt.figure('changing csd res')
    for i in xrange(5):    
        csdres = 10+i*10
        (x, V) = electrode_config([0.,1.], 1000, csd_profile, [0.,1.], csd_seed, CSDres=csdres)
        #ans = calculate_potential_1D(csd_profile, measure_locations, states, csd_space_lims=[0.,1.], CSDres=csdres)    
        plt.plot(x, V, label=str(csdres))
    plt.legend()
    #changing olny measure resolution
    #measure_locations = np.arange(0,1,1000)
    plt.figure('changing measure res')
    for i in xrange(5):    
        measure_res= 20+i*50
        (x, V) = electrode_config([0.,1.], measure_res, csd_profile, [0.,1.], csd_seed, CSDres=1000)
        plt.plot(x,V, ms = 0.5, label =str(measure_res))
    plt.legend()
    plt.show()
    return

if __name__=='__main__':
    total_ele = 30
    csd_seed = 11
    csd_profile = CSD.gauss_1d_mono
    #test_calculating_potentials(csd_seed)
    a = main_loop(csd_profile, csd_seed, total_ele)
    #fig.savefig(os.path.join(plots_folder, save_as+'.png'))
    #plt.clf()
    #plt.close()
