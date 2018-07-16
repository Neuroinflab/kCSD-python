from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from kcsd import sKCSD
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
sys.path.insert(1, os.path.join(sys.path[0], '../sKCSD_paper'))
import sKCSD_utils
import matplotlib.gridspec as gridspec


n_src = 512
lambd = 0.01
R = 16e-6/2**.5
if __name__ == '__main__':
    fname_base = "Figure_3"
    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [2**i*1e-6/np.sqrt(2) for i in range(3,8)]
    lambdas = [10**(-i) for i in range(-3, 6, 1)]
    electrode_number = [8, 32 , 128]
    data_dir = []
    xmin, xmax = -100, 600
    ymin, ymax = 0, 200
    orientation = 1
    for rownb in electrode_number:
        fname = "Figure_3"
        c = sKCSD_utils.simulate(fname,
                                 morphology=1,
                                 simulate_what="random",
                                 colnb=1,
                                 rownb=rownb,
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax,
                                 tstop=tstop,
                                 seed=1988,
                                 weight=0.05,
                                 n_syn=100,
                                 dt=0.125)
        data_dir.append(c.return_paths_skCSD_python())
    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    gvmax, gvmin = pl.get_min_max(ground_truth)
    data_paths = []
    
    fig =   plt.figure(figsize=(20, 6))
    
    gs = gridspec.GridSpec(2, 5, figure=fig)

    ax_gt = plt.subplot(gs[0,0])
    ax = []
    for i in range(2):
        for j in range(2, 5):
            ax.append(plt.subplot(gs[i, j]))
    cax = ax_gt.imshow(ground_truth,
                          extent=[0, tstop, 1, 52],
                          origin='lower',
                          aspect='auto',
                          cmap='seismic_r',
                          vmax=gvmax,
                          vmin=gvmin)
    ax_gt.set_title('Ground truth')
    ax_gt.set_xlabel('time (s)')
    ax_gt.set_ylabel('#segment')
    new_fname = fname_base + '.png'
    fig_name = sKCSD_utils.make_fig_names(new_fname)
    vmax, vmin = pl.get_min_max(ground_truth)
    for i, datd in enumerate(data_dir):
        
        data = utils.LoadData(datd)
        ele_pos = data.ele_pos/scaling_factor
        data.LFP = data.LFP/scaling_factor_LFP
        morphology = data.morphology
        morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
        k = sKCSD(ele_pos,
                  data.LFP,
                  morphology,
                  n_src_init=n_src,
                  src_type='gauss',
                  lambd=lambd,
                  R_init=R,
                  skmonaco_available=False,
                  dist_table_density=100)
        csd = k.values(transformation='segments')
        
        cax = ax[i].imshow(csd,
                           extent=[0, tstop, 1, 52],
                           origin='lower',
                           aspect='auto',
                           cmap='seismic_r',
                           vmax=gvmax,
                           vmin=gvmin)
        ax[3+i].set_title(electrode_number[i])
        k.L_curve(lambdas=np.array(lambdas), Rs=np.array(R_inits))
        csd_Lcurve = k.values(transformation='segments')
        
        #Rcv, lambdacv = k.cross_validate(lambdas=np.array(lambdas), Rs=np.array(R_inits))
       
        cax = ax[3+i].imshow(csd_Lcurve,
                              extent=[0, tstop, 1, 52],
                              origin='lower',
                              aspect='auto',
                              cmap='seismic_r',
                              vmax=gvmax,
                              vmin=gvmin)
        ax[i].set_title('%d electrodes' % electrode_number[i])
        ax[i].set_xticklabels([])
        ax[i+3].set_xlabel('time (s)')
  
        if i:
            ax[i].set_yticklabels([])
            ax[i+3].set_yticklabels([])
        else:
            ax[i].set_ylabel('#segment')
            ax[i+3].set_ylabel('#segment')
            
    
    plt.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    plt.savefig(fig_name[:-4]+'.svg',
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    plt.show()
