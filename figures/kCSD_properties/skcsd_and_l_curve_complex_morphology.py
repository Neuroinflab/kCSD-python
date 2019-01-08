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
R = 16e-6
lambd = 0.1#/(16*np.pi**3*R**2*n_src)


if __name__ == '__main__':
    fname_base = "Figure_complex"
    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [2**(i-0.5)*1e-6 for i in range(3, 8)]
    lambdas = [10**(-i) for i in range(5)]
    electrode_number = [10, 20, 30]
    colnb = 20
    data_dir = []
    for rownb in electrode_number:
        fname = "Figure_3_complex"
        c = sKCSD_utils.simulate(fname_base,
                             morphology=6,
                             tstop=tstop,
                             seed=1988,
                             weight=0.04,
                             n_syn=1000,
                             simulate_what='oscillatory',
                             electrode_distribution=1,
                             electrode_orientation=3,
                             xmin=-400,
                             xmax=400,
                             ymin=-400,
                             ymax=400,
                             colnb=colnb,
                             rownb=rownb,
                                 dt=0.5)
                                 
        data_dir.append(c.return_paths_skCSD_python())
    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    n_seg = len(seglen)
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    gvmax, gvmin = pl.get_min_max(ground_truth)
    data_paths = []
    print(ground_truth.min(), ground_truth.max())
    fig =   plt.figure(figsize=(20, 6))
    
    gs = gridspec.GridSpec(2, 5, figure=fig)

    ax_gt = plt.subplot(gs[0,0])
    ax = []
    for i in range(2):
        for j in range(2, 5):
            ax.append(plt.subplot(gs[i, j]))

            
    cax = ax_gt.imshow(ground_truth,
                          extent=[0, tstop, 1, ground_truth.shape[0]],
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
    for i, datd in enumerate(data_dir):
        print(datd)
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
                  exact=True,
                  R_init=R,
                  sigma=0.3)
        csd = k.values(transformation='segments')
        print(csd.shape)
        print(csd.max(), csd.min())
        cax = ax[i].imshow(csd,
                           extent=[0, tstop, 1, csd.shape[0]],
                           origin='lower',
                           aspect='auto',
                           cmap='seismic_r',
                           vmax=gvmax,
                           vmin=gvmin)
        ax[3+i].set_title(electrode_number[i])
        #k.L_curve(lambdas=np.array(lambdas), Rs=np.array(R_inits))
        #csd_Lcurve = k.values(transformation=None)
        
        # #Rcv, lambdacv = k.cross_validate(lambdas=np.array(lambdas), Rs=np.array(R_inits))
       
        # cax = ax[3+i].imshow(csd_Lcurve,
        #                       extent=[0, tstop, 1, csd_Lcurve.shape[0]],
        #                       origin='lower',
        #                       aspect='auto',
        #                       cmap='seismic_r',
        #                       vmax=gvmax,
        #                       vmin=gvmin)
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
