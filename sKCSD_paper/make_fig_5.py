from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, KCSD3D
import kcsd.utility_functions as utils
import loadData as ld
import validation.plotting_functions as pl
import sKCSD_utils
n_src = 512
R = 8e-6/2**.5
lambd = 1

if __name__ == '__main__':
    fname_base = "Figure_5"
    fig_name = sKCSD_utils.make_fig_names(fname_base)
    tstop = 70
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    electrode_number = [1, 4]
    data_dir = []
    colnb = 16
    lfps = []
    extent = [-200, 200, -200, 600]
    xmin = [50, -200]
    for i, rownb in enumerate(electrode_number):
        fname = fname_base
        c = sKCSD_utils.simulate(fname,
                                 morphology=2,
                                 colnb=colnb,
                                 rownb=rownb,
                                 xmin=-200,
                                 xmax=600,
                                 ymin=xmin[i],
                                 ymax=200,
                                 tstop=tstop,
                                 seed=1988,
                                 weight=0.01,
                                 n_syn=100,
                                 simulate_what="symmetric",
                                 dt=.2)
        data_dir.append(c.return_paths_skCSD_python())
    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    ground_truth_grid = []
    ground_truth_t1 = None
    ground_truth_t2 = None
    dt = c.cell_parameters['dt']
    t1 = int(45.5/dt)
    t2 = int(5.5/dt)
    R_inits = [2**i for i in range(3, 8)]
    lambdas = [10**(-i) for i in range(6)]
    n = 100
    fname = fname_base+'.png'
    fig_name = sKCSD_utils.make_fig_names(fname)
    fig, ax = plt.subplots(3, 4)
    for i, datd in enumerate(data_dir):
        data = ld.Data(datd)
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
                  dist_table_density=n,
                  skmonaco_available=False)
        xmin = -200/scaling_factor
        xmax = 200/scaling_factor
        ymin = -100/scaling_factor
        ymax = 200/scaling_factor
        zmin = -200/scaling_factor
        zmax = 600/scaling_factor
        gdx = (xmax-xmin)/100
        gdy = (ymax-ymin)/2
        gdz = (zmax-zmin)/200

        kcsd = KCSD3D(ele_pos,
                      data.LFP,
                      n_src_init=n_src,
                      src_type='gauss',
                      lambd=lambd,
                      R_init=R,
                      dist_table_density=n,
                      xmin=xmin,
                      xmax=xmax,
                      ymin=ymin,
                      ymax=ymax,
                      zmin=zmin,
                      zmax=zmax,
                      gdx=gdx,
                      gdy=gdy,
                      gdz=gdz)
        if not len(ground_truth_grid):
            ground_truth_grid = k.cell.transform_to_3D(ground_truth,
                                                       what="morpho")
            ground_truth_t1 = ground_truth_grid[:, :, :, t1].sum(axis=1).T
            ground_truth_t2 = ground_truth_grid[:, :, :, t2].sum(axis=1).T
            est_skcsd = k.values(estimate='CSD')
            est_skcsd_t1 = est_skcsd[:, :, :, t1].sum(axis=1).T
            est_skcsd_t2 = est_skcsd[:, :, :, t2].sum(axis=1).T
            est_kcsd = kcsd.values(estimate='CSD')
            est_kcsd_pot = kcsd.values(estimate='POT')
        if i == 0:
            cax = pl.plot(ax[1, 2], ground_truth_t1)
            cax = pl.plot(ax[2, 2], ground_truth_t2)
            cax = pl.plot(ax[1, 3], est_skcsd_t1)
            cax = pl.plot(ax[2, 3], est_skcsd_t2)
        else:
            cax = pl.plot(ax[0, 0], est_kcsd_pot[:, :, :, t1].sum(axis=1))
            cax = pl.plot(ax[0, 1], est_kcsd[:, :, :, t1].sum(axis=1))
            cax = pl.plot(ax[0, 2], ground_truth_t1)
            cax = pl.plot(ax[0, 3], est_skcsd_t1)
            cax = pl.plot(ax[1, 0], est_kcsd_pot[:, :, :, t1].sum(axis=1))
            cax = pl.plot(ax[1, 1], est_kcsd[:, :, :, t1].sum(axis=1))
            cax = pl.plot(ax[2, 0],
                          est_kcsd_pot[:, :, :, t2].sum(axis=1),
                          extent=[-200, 200, -200, 600])
            cax = pl.plot(ax[2, 1], est_kcsd[:, :, :, t2].sum(axis=1))
    fig.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
