from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, KCSD3D, sKCSDcell
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
n_src = 512
R = 16e-6/2**.5
lambd = .1/((2*(2*np.pi)**3*R**2*n_src))
dt = 0.5
if __name__ == '__main__':
    fname_base = "Figure_5"
    fig_name = sKCSD_utils.make_fig_names("Figure_5.png")
    tstop = 70
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    electrode_number = [1, 4]
    data_dir = []
    colnb = 16
    lfps = []
    
    xmin = [33, -100]
    for i, rownb in enumerate(electrode_number):
        fname = fname_base
        c = sKCSD_utils.simulate(fname,
                                 morphology=2,
                                 colnb=rownb,
                                 rownb=colnb,
                                 xmin=0,
                                 xmax=500,
                                 ymin=xmin[i],
                                 ymax=100,
                                 tstop=tstop,
                                 seed=1988,
                                 weight=0.01,
                                 n_syn=100,
                                 simulate_what="symmetric",
                                 electrode_orientation=2,
                                 dt=dt)
        data_dir.append(c.return_paths_skCSD_python())
    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    dt = c.cell_parameters['dt']
    t1 = int(45.5/dt)
    t2 = int(5.5/dt)
    R_inits = [2**i for i in range(3, 8)]
    lambdas = [10**(-i) for i in range(6)]
    n = 100
    fname = fname_base+'.png'
    fig_name = sKCSD_utils.make_fig_names(fname)
    
    fig, ax = plt.subplots(3, 4)
    fname_base = "simuation/Figure_5_%d"
    data = utils.LoadData(data_dir[0])
    ele_pos = data.ele_pos/scaling_factor
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
    cell_itself =  sKCSDcell(morphology, ele_pos, n_src)
    cell_itself.distribute_srcs_3D_morph()
    morpho, extent = cell_itself.draw_cell2D(axis=1)
    ground_truth_grid = cell_itself.transform_to_3D(ground_truth, what="morpho")
    ground_truth_t1 = ground_truth_grid[:, :, :, t1].sum(axis=1)
    ground_truth_t2 = ground_truth_grid[:, :, :, t2].sum(axis=1)
    for i, x in enumerate(extent):
        if i%2:
            extent[i] = x + 50
        else:
            extent[i] = x - 50

    for i, datd in enumerate(data_dir):
       
        data = utils.LoadData(datd)
        ele_pos = data.ele_pos/scaling_factor
       
        data.LFP = data.LFP/scaling_factor_LFP
        morphology = data.morphology
        morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
        ker = sKCSD(ele_pos,
                  data.LFP,
                  morphology,
                  n_src_init=n_src,
                  src_type='gauss',
                  lambd=lambd,
                  R_init=R,
                  exact=True)
        
        xmin = cell_itself.xmin
        xmax = cell_itself.xmax
        ymin = cell_itself.ymin
        ymax = cell_itself.ymax + 0.00001
        zmin = cell_itself.zmin
        zmax = cell_itself.zmax
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

        if sys.version_info < (3, 0):
            path = os.path.join(fname_base % i, "preprocessed_data/Python_2")
        else:
            path = os.path.join(fname_base % i, "preprocessed_data/Python_3")

        if not os.path.exists(path):
            print("Creating", path)
            os.makedirs(path)
        utils.save_sim(path, ker)
        try:
            est_skcsd = ker.values(estimate='CSD')
        except NameError:
            skcsd, pot, cell_obj = utils.load_sim(path)
            est_skcsd = cell_obj.transform_to_3D(skcsd)


        est_skcsd_t1 = est_skcsd[:, :, :, t1].sum(axis=1)
        est_skcsd_t2 = est_skcsd[:, :, :, t2].sum(axis=1)
        est_kcsd = kcsd.values(estimate='CSD')
        est_kcsd_pot = kcsd.values(estimate='POT')

        if i == 0:
            for j in [1, 2]:
                for k in [2, 3]:
                    ax[j, k].imshow(morpho, extent=extent, origin='lower', aspect="auto")
                    for z in ele_pos:
                        pos_x, pos_y = 1e6*z[2], 1e6*z[0]
                        ax[j, k].text(pos_x, pos_y, '*',
                                      ha="center", va="center", color="k")
            cax = pl.make_map_plot(ax[1, 2], ground_truth_t1, extent=extent)
            cax = pl.make_map_plot(ax[2, 2], ground_truth_t2, extent=extent)
            cax = pl.make_map_plot(ax[1, 3], est_skcsd_t1, extent=extent)
            cax = pl.make_map_plot(ax[2, 3], est_skcsd_t2, extent=extent)

        else:
            for j in [1, 2]:
                for k in [0, 1]:
                    ax[j, k].imshow(morpho, extent=extent, origin='lower', aspect="auto")
                    for z in ele_pos:
                        pos_x, pos_y = 1e6*z[2], 1e6*z[0]
                        print(z)
                        ax[j, k].text(pos_x, pos_y, '*',
                                      ha="center", va="center", color="k")
            for j in [0, 1, 2, 3]:
                    ax[0, j].imshow(morpho, extent=extent, origin='lower', aspect="auto")
                    for z in ele_pos:
                        print(z)
                        pos_x, pos_y = 1e6*z[2], 1e6*z[0]
                        ax[0, j].text(pos_x, pos_y, '*',
                                      ha="center", va="center", color="k")
            cax = pl.make_map_plot(ax[0, 0], est_kcsd_pot[:, :, :, t1].sum(axis=1),cmap=plt.cm.viridis, extent=extent)
            cax = pl.make_map_plot(ax[0, 1], est_kcsd[:, :, :, t1].sum(axis=1), extent=extent)
            cax = pl.make_map_plot(ax[0, 2], ground_truth_t1, extent=extent)
            cax = pl.make_map_plot(ax[0, 3], est_skcsd_t1, extent=extent)
            cax = pl.make_map_plot(ax[1, 0], est_kcsd_pot[:, :, :, t1].sum(axis=1),cmap=plt.cm.viridis, extent=extent)
            cax = pl.make_map_plot(ax[1, 1], est_kcsd[:, :, :, t1].sum(axis=1), extent=extent)
            cax = pl.make_map_plot(ax[2, 0],
                          est_kcsd_pot[:, :, :, t2].sum(axis=1),
                                   extent=extent, cmap=plt.cm.viridis)
            cax = pl.make_map_plot(ax[2, 1], est_kcsd[:, :, :, t2].sum(axis=1), extent=extent)
            
        fig.savefig(fig_name,
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0.1)
