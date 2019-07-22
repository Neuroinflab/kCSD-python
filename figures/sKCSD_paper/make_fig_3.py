from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD
from kcsd import sKCSD_utils as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
n_src = 512
lambd = 1
R = 8e-6/2**.5
if __name__ == '__main__':
    fname_base = "Figure_3"
    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [2**i for i in range(3, 8)]
    lambdas = [10**(-i) for i in range(6)]
    electrode_number = [8, 32, 128]
    data_dir = []
    xmin, xmax = -100, 600
    ymin, ymax = 0, 200
    orientation = 1
    for rownb in electrode_number:
        fname = '%s_rows_%d' % (fname_base, rownb)
        if sys.version_info < (3, 0):
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
                                     weight=0.1,
                                     n_syn=100)
            new_dir = c.return_paths_skCSD_python()
        else:
            new_dir = os.path.join('simulation', fname)
        data_dir.append(new_dir)
    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    gvmax, gvmin = pl.get_min_max(ground_truth)
    data_paths = []
    fig, ax = plt.subplots(2, 3)
    cax = ax[0, 0].imshow(ground_truth,
                          extent=[0, tstop, 1, 52],
                          origin='lower',
                          aspect='auto',
                          cmap='seismic_r',
                          vmax=gvmax,
                          vmin=gvmin)
    new_fname = fname_base + '.png'
    fig_name = sKCSD_utils.make_fig_names(new_fname)
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
                  exact=True,
                  sigma=0.3)
        csd = k.values(transformation='segments')
        
        cax = ax[1, i].imshow(csd,
                              extent=[0, tstop, 1, 52],
                              origin='lower',
                              aspect='auto',
                              cmap='seismic_r',
                              vmax=gvmax,
                              vmin=gvmin)
        ax[1, i].set_title(electrode_number[i])
    fig.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
