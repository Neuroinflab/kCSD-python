from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, sKCSDcell
from kcsd import sKCSD_utils as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
import numpy.random

n_src = 512
l = 1e-1
R = 16e-6/2**.5
dt = 2**(-1)
noise_levels = [0, 16, 4 , 1]
lambd = l/(2*(2*np.pi)**3*R**2*n_src)
if __name__ == '__main__':
    fname_base = "simulation/Figure_9_noise_%f"
    tstop = 7
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    data_dir = []
    colnb = 4
    rownb = 8
    xmin, xmax = 0, 500
    ymin, ymax = -80, 80
    t1 = int(5.5/dt)
    fname = "Figure_9"
    c = sKCSD_utils.simulate(fname,
                             morphology=2,
                             simulate_what="symmetric",
                             colnb=colnb,
                             rownb=rownb,
                             xmin=xmin,
                             xmax=xmax,
                             ymin=ymin,
                             ymax=ymax,
                             tstop=tstop,
                             seed=1988,
                             weight=0.04,
                             n_syn=100,
                             electrode_distribution=1,
                             dt=dt)
    
    data_dir = c.return_paths_skCSD_python()

    seglen = np.loadtxt(os.path.join(data_dir,
                                     'seglength'))

    ground_truth = np.loadtxt(os.path.join(data_dir,
                                           'membcurr'))
    
    ground_truth = ground_truth/seglen[:, None]*1e-3
    
    fig, ax = plt.subplots(2, 3, figsize=(8,20))
    fname = "Figure_9.png"
    fig_name = sKCSD_utils.make_fig_names(fname)
    
    data = utils.LoadData(data_dir)
    ele_pos = data.ele_pos/scaling_factor
    data.LFP = data.LFP/scaling_factor_LFP
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
    std = data.LFP.var()**.5
    shape = data.LFP.shape
    cell =  sKCSDcell(morphology, ele_pos, n_src,
                      tolerance=2e-6,
                      xmin=-120e-6, xmax=120e-6,
                      zmin=-50e-6, zmax=550e-6)
    ground_truth_grid = cell.transform_to_3D(ground_truth, what="morpho")
    vmax, vmin = pl.get_min_max(ground_truth_grid[:, :, :, t1].sum(axis=1))
    gdt1 = ground_truth_grid[:,:,:,t1].sum(axis=1)
    morpho, extent = cell.draw_cell2D(axis=1)
    extent = [extent[-2], extent[-1], extent[0], extent[1]]
    new_ele_pos = np.array([ele_pos[:, 2], ele_pos[:, 0]]).T
    pl.make_map_plot(ax[0, 0],
                     gdt1,
                     vmin=vmin,
                     vmax=vmax,
                     extent=extent,
                     alpha=.9,
                     morphology=morpho,
                     ele_pos=new_ele_pos)
    snrs = []
    L1 = []
    
    for i, nl in enumerate(noise_levels):
        new_LFP = data.LFP
        if nl:
            noise = numpy.random.normal(scale=std/nl, size=shape)
            snr = np.round((new_LFP.var()**.5/noise.var()**.5))
            
        else:
            noise = 0
            snr = 0
        new_LFP += noise    
        snrs.append(snr)
            

        

        
        k = sKCSD(ele_pos,
                  new_LFP,
                  morphology,
                  n_src_init=n_src,
                  src_type='gauss',
                  lambd=lambd,
                  R_init=R,
                  exact=True,
                  sigma=0.3)
        
        if sys.version_info < (3, 0):
            path = os.path.join(fname_base % nl, "preprocessed_data/Python_2")
        else:
            path = os.path.join(fname_base % nl, "preprocessed_data/Python_3")
            
        if not os.path.exists(path):
            print("Creating", path)
            os.makedirs(path)
        try:
            utils.save_sim(path, k)
        except NameError:
            pass
      
        
        est_skcsd, est_pot, morphology, ele_pos, n_src = utils.load_sim(path)
        cell_object = sKCSDcell(morphology, ele_pos, n_src)
        est_skcsd = cell.transform_to_3D(est_skcsd)
        L1.append(sKCSD_utils.L1_error(ground_truth_grid, est_skcsd))
        if nl == 0:
            pl.make_map_plot(ax[0, 1],
                             est_skcsd[:,:,:,t1].sum(axis=1),
                             vmin=vmin,
                             vmax=vmax,
                             extent=extent,
                             title="No noise",
                             alpha=.9,
                             morphology=morpho,
                             ele_pos=new_ele_pos)
        else:
            pl.make_map_plot(ax[1, i-1],
                             est_skcsd[:,:,:,t1].sum(axis=1),
                             vmin=vmin,
                             vmax=vmax,
                             extent=extent,
                             title='SNR %f' % snr,
                             alpha=.9,
                             morphology=morpho,
                             ele_pos=new_ele_pos)

    ax[0, 2].plot([i for i in range(len(snrs))], L1, 'dk')
    ax[0, 2].set_xticks([i for i in range(len(snrs))])
    ax[0, 2].set_xticklabels([ 'No noise','16', '4', '1', ], rotation = 90)
    ax[0, 0].set_title('Ground truth')
    ax[0, 2].set_ylabel('L1 error', fontsize=12)
    ax[0, 2].set_xlabel('SNR', fontsize=12)
    fig.subplots_adjust(hspace=.5, wspace=.35)
    fig.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    
