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


n_src = 1024
R = 32e-6
lambd = 0.0001#/(16*np.pi**3*R**2*n_src)


if __name__ == '__main__':
    fname_base = "Figure_complex"
    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [2**(i-0.5)*1e-6 for i in range(3, 8)]
    lambdas = [10**(-i) for i in range(5)]
    rownb = 10
    colnb = 10

    fname = "Figure_3_complex"
    c = sKCSD_utils.simulate(fname_base,
                             morphology=7,
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
                                 
    data_dir = c.return_paths_skCSD_python()
    seglen = np.loadtxt(os.path.join(data_dir, 'seglength'))
    n_seg = len(seglen)
    ground_truth = np.loadtxt(os.path.join(data_dir, 'membcurr'))
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
    data = utils.LoadData(data_dir)
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
    path = os.path.join(data_dir, 'lambda_%f_R_%f_n_src_%d' % (lambd, R, n_src))
    if sys.version_info < (3, 0):
        path = os.path.join(path, "preprocessed_data/Python_2")
    else:
        path = os.path.join(path, "preprocessed_data/Python_3")
    if not os.path.exists(path):
        print("Creating", path)
        os.makedirs(path)
    try:
        utils.save_sim(path, k)
    except NameError:
        pass
    skcsd, pot, cell_obj = utils.load_sim(path)
    csd = cell_obj.transform_to_segments(skcsd)
    print(csd.shape)
    print(csd.max(), csd.min())
    cax = ax[1].imshow(csd,
                       extent=[0, tstop, 1, csd.shape[0]],
                       origin='lower',
                       aspect='auto',
                       cmap='seismic_r',
                       vmax=gvmax,
                       vmin=gvmin)
    ax[1].set_title('10 x 10')
    ax[1].set_title('%d electrodes' % colnb*rownb)
    ax[1].set_xticklabels([])
    ax[1].set_xlabel('time (s)')
  
    
    plt.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    plt.savefig(fig_name[:-4]+'.svg',
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    plt.show()
