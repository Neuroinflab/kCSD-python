from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
dt = 0.5
if __name__ == '__main__':
    fname_base = "Figure_7"
    fig_name = sKCSD_utils.make_fig_names(fname_base + '.png')
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R_inits = np.array([(2**(i - .5))/scale_factor for i in range(3, 7)])
    lambdas = np.array([(10**(-i))for i in range(5, 0, -1)])
    n_srcs = np.array([32, 64, 128, 512, 1024])
    x_ticklabels = [2**i for i in range(3, 7)]
    y_ticklabels = [str(lambd) for lambd in lambdas]
    colnb = 4
    rownb = 4
    c = sKCSD_utils.simulate(fname_base,
                             morphology=2,
                             colnb=colnb,
                             rownb=rownb,
                             xmin=0,
                             xmax=500,
                             ymin=-100,
                             ymax=100,
                             tstop=tstop,
                             seed=1988,
                             weight=0.04,
                             n_syn=100,
                             electrode_orientation=2,
                             simulate_what='symmetric',
                             dt=dt)
    data = utils.LoadData(c.return_paths_skCSD_python())
    ele_pos = data.ele_pos/scale_factor
    pots = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scale_factor
    new_path = c.return_paths_skCSD_python()
    ground_truth = np.loadtxt(os.path.join(new_path,
                                           'membcurr'))
    seglen = np.loadtxt(os.path.join(new_path,
                                     'seglength'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    outs = np.zeros((len(n_srcs), len(lambdas), len(R_inits)))
    for i, n_src in enumerate(n_srcs):
        for j, l in enumerate(lambdas):
            for k, R in enumerate(R_inits):
                lambd = l/(16*np.pi**3*R**2*n_src)
                ker = sKCSD(ele_pos,
                            pots,
                            morphology,
                            n_src_init=n_src,
                            src_type='gauss',
                            lambd=lambd,
                            R_init=R,
                            exact=True,
                            sigma=0.3)
                est_skcsd = ker.values(estimate='CSD',
                                       transformation='segments')
          
                outs[i, j, k] = sKCSD_utils.L1_error(ground_truth,
                                                     est_skcsd)
                print(outs[i, j, k], est_skcsd.min(), est_skcsd.max(), ground_truth.min(), ground_truth.max(), n_src, l, R)
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
