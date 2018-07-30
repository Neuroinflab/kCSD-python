from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
if __name__ == '__main__':
    fname_base = "Figure_7"
    fig_name = sKCSD_utils.make_fig_names(fname_base + '.png')
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R_inits = np.array([(2**(i - .5))/scale_factor for i in range(4, 7)])
    lambdas = np.array([(10**(-i))for i in range(4, -4, -1)])
    n_srcs = np.array([32, 64, 128, 512, 1024])
    x_ticklabels = [2**i for i in range(3, 9)]
    y_ticklabels = [str(lambd) for lambd in lambdas]
    colnb = 4
    rownb = 4
    fname = '%s_rows_%d' % (fname_base, rownb)
    if sys.version_info < (3, 0):
        c = sKCSD_utils.simulate(fname,
                                 morphology=2,
                                 colnb=colnb,
                                 rownb=rownb,
                                 xmin=-200,
                                 xmax=600,
                                 ymin=-200,
                                 ymax=200,
                                 tstop=tstop,
                                 seed=1988,
                                 weight=0.04,
                                 n_syn=100,
                                 simulate_what='symmetric')
        new_path = c.return_paths_skCSD_python()
    else:
        new_path = os.path.join('simulation', fname)
    data = utils.LoadData(new_path)
    ele_pos = data.ele_pos/scale_factor
    pots = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scale_factor
    
    ground_truth = np.loadtxt(os.path.join(new_path,
                                           'membcurr'))
    seglen = np.loadtxt(os.path.join(new_path,
                                     'seglength'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    outs = np.zeros((len(n_srcs), len(lambdas), len(R_inits)))
    for i, n_src in enumerate(n_srcs):
        for j, l in enumerate(lambdas):
            for k, R in enumerate(R_inits):
                lambd = l
                ker = sKCSD(ele_pos,
                            pots,
                            morphology,
                            n_src_init=n_src,
                            src_type='gauss_lim',
                            lambd=lambd,
                            R_init=R,
                            skmonaco_available=False)
                est_skcsd = ker.values(estimate='CSD',
                                       transformation='segments')
                outs[i, j, k] = sKCSD_utils.L1_error(ground_truth,
                                                     est_skcsd)
                print(outs[i, j, k])
    fig, ax = plt.subplots(1, 4, sharey=True)
    vmax = outs.max()
    vmin = outs.min()
    for i, ax_i in enumerate(ax):
        title = "M = %d" % n_srcs[i]
        if not i:
            pl.make_map_plot(ax_i,
                    outs[i],
                    yticklabels=y_ticklabels,
                    xticklabels=x_ticklabels,
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    cmap='gray')
        elif i < 3:
            pl.make_map_plot(ax_i,
                    outs[i],
                    xticklabels=x_ticklabels,
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    cmap='gray')
        else:
            pl.make_map_plot(ax_i,
                    outs[i],
                    xticklabels=x_ticklabels,
                    fig=fig,
                    vmin=vmin,
                    vmax=vmax,
                    sinksource=False,
                    title=title,
                    cmap='gray')
    fig.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
