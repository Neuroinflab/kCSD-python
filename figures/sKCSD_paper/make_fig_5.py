import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, KCSD3D, sKCSDcell
from  kcsd import sKCSD_utils as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils

n_src = 512
R = 16e-6/2**.5
lambd = .1/((2*(2*np.pi)**3*R**2*n_src))
dt = 0.5
n = 100  # dist_table_density for kCSD
if __name__ == '__main__':
    fname_base = "Figure_5"
    fig_name = sKCSD_utils.make_fig_names("Figure_5.png")
    tstop = 70
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    rownb =  [16, 4]
    data_dir = []
    colnb = [4, 16]
    lfps = []
    xmax = [100, 500]
    ymax = [500, 100]
    cell_itself = []
    
    for i, electrode_orientation in enumerate([1, 2]):
        fname = fname_base
        c = sKCSD_utils.simulate(fname,
                                 morphology=2,
                                 colnb=rownb[i],
                                 rownb=colnb[i],
                                 xmin=-100,
                                 xmax=xmax[i],
                                 ymin=-100,
                                 ymax=ymax[i],
                                 tstop=tstop,
                                 seed=1988,
                                 weight=0.01,
                                 n_syn=100,
                                 simulate_what="symmetric",
                                 electrode_orientation=electrode_orientation,
                                 dt=dt)
        data_dir.append(c.return_paths_skCSD_python())
        data = utils.LoadData(data_dir[i])
        ele_pos = data.ele_pos/scaling_factor
        morphology = data.morphology
        morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
        if i == 0:
            cell_itself.append(sKCSDcell(morphology,
                                         ele_pos,
                                         n_src,
                                         xmin=-120e-6,
                                         xmax=120e-6,
                                         ymin=-200e-6,
                                         ymax=200e-6,
                                         zmin=-150e-6,
                                         zmax=550e-6))
        else:
            cell_itself.append(sKCSDcell(morphology,
                                         ele_pos,
                                         n_src,
                                         xmin=-120e-6,
                                         xmax=120e-6,
                                         ymin=-200e-6,
                                         ymax=200e-6,
                                         zmin=-150e-6,
                                         zmax=550e-6))
        morpho, extent = cell_itself[i].draw_cell2D(axis=1)
        extent = [extent[-2], extent[-1], extent[0], extent[1]]
        if i == 0:
            morpho_kcsd, extent_kcsd = cell_itself[i].draw_cell2D(axis=0, resolution=[50, 50, 50])
            extent_kcsd = [extent_kcsd[-2], extent_kcsd[-1],
                           extent_kcsd[0], extent_kcsd[1]]

    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    dt = c.cell_parameters['dt']
    t1 = int(45.5/dt)
    t2 = int(5.5/dt)
    ground_truth_grid = cell_itself[1].transform_to_3D(ground_truth, what="morpho")
    ground_truth_t1 = ground_truth_grid[:, :, :, t1].sum(axis=1)
    ground_truth_t2 = ground_truth_grid[:, :, :, t2].sum(axis=1)
    
    vmax, vmin = pl.get_min_max(ground_truth_grid)
    
    fname = fname_base+'.png'
    fig_name = sKCSD_utils.make_fig_names(fname)
    fig, ax = plt.subplots(3, 4)
    fname_base = "simulation/Figure_5_%d"

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
                    exact=True,
                    sigma=0.3)
        if sys.version_info < (3, 0):
            path = os.path.join(fname_base % i, "preprocessed_data/Python_2")
        else:
            path = os.path.join(fname_base % i, "preprocessed_data/Python_3")

        if not os.path.exists(path):
            print("Creating", path)
            os.makedirs(path)
            
        try:
            utils.save_sim(path, ker)
            est_skcsd = ker.values(estimate='CSD')
        except NameError:
            skcsd, pot, morphology, ele_pos, n_src = utils.load_sim(path)
            cell_obj =  sKCSDcell(morphology, ele_pos, n_src)
            est_skcsd = cell_itself[i].transform_to_3D(skcsd)

        xmin = cell_itself[i].xmin
        xmax = cell_itself[i].xmax
        ymin = -200e-6
        ymax = 200e-6
        zmin = cell_itself[i].zmin
        zmax = cell_itself[i].zmax
        gdx = (xmax-xmin)/50
        gdy = (ymax-ymin)/50
        gdz = (zmax-zmin)/50
        
        kcsd = KCSD3D(ele_pos,
                      data.LFP,
                      n_src_init=n_src,
                      src_type='gauss',
                      lambd=lambd*((2*(2*np.pi)**3*R**2*n_src))*.00001,
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
                      gdz=gdz,
                      sigma=0.3)
        est_kcsd = kcsd.values(estimate='CSD')
        est_kcsd_pot = kcsd.values(estimate='POT')
        if sys.version_info < (3, 0):
            path = os.path.join(fname_base % i, "preprocessed_data/Python_2")
        else:
            path = os.path.join(fname_base % i, "preprocessed_data/Python_3")

        if not os.path.exists(path):
            print("Creating", path)
            os.makedirs(path)
        
        try:
            est_skcsd = ker.values(estimate='CSD')
        except NameError:
            skcsd, pot, morphology, ele_pos, n_src = utils.load_sim(path)
            cell_object = sKCSDcell(morphology, ele_pos, n_src)
            est_skcsd = cell_itself[i].transform_to_3D(skcsd)


        est_skcsd_t1 = est_skcsd[:, :, :, t1].sum(axis=1)
        est_skcsd_t2 = est_skcsd[:, :, :, t2].sum(axis=1)
        
        if i == 0:  # wrong plane
            for j in [1, 2]:
                for k in [2, 3]:
                    ax[j, k].imshow(morpho, extent=extent, origin='lower', aspect="auto", alpha=1.)
                    for z in ele_pos:
                        pos_x, pos_y = z[2], z[0]
                        ax[j, k].text(pos_x, pos_y, '*',
                                      ha="center", va="center", color="k", fontsize=3)
            cax = pl.make_map_plot(ax[1, 2], ground_truth_t1, extent=extent, alpha=.95, vmin=vmin, vmax=vmax)
            cax = pl.make_map_plot(ax[2, 2], ground_truth_t2, extent=extent, alpha=.95, vmin=vmin, vmax=vmax)
            cax = pl.make_map_plot(ax[1, 3], est_skcsd_t1, extent=extent, vmin=vmin, vmax=vmax)
            cax = pl.make_map_plot(ax[2, 3], est_skcsd_t2, extent=extent, vmin=vmin, vmax=vmax)  # ok
            for j in [1, 2]:
                for k in [0, 1]:
                    ax[j, k].imshow(morpho_kcsd, extent=extent_kcsd, origin='lower', aspect="auto", alpha=1)
                    for z in ele_pos:
                        pos_x, pos_y = z[2], z[1]
                        ax[j, k].text(pos_x, pos_y, '*',
                                      ha="center", va="center", color="k", fontsize=3)
           
            ax[1, 0].imshow(est_kcsd_pot[:, :, :, t1].sum(axis=0),
                            cmap=plt.cm.viridis,
                            extent=extent_kcsd,
                            origin='lower',
                            aspect='auto',
                            interpolation="none",
                            alpha=0.5)
            ax[1, 0].set_xticklabels([])
            ax[1, 0].set_yticklabels([])
                        
            ax[1, 1].imshow(est_kcsd[:, :, :, t1].sum(axis=0),
                            extent=extent_kcsd,
                            cmap=plt.cm.bwr_r,
                            origin='lower',
                            interpolation="none",
                            aspect='auto', vmin=vmin, vmax=vmax,
                            alpha=0.5)
            ax[1, 1].set_xticklabels([])
            ax[1, 1].set_yticklabels([])
            
            ax[2, 0].imshow(est_kcsd_pot[:, :, :, t2].sum(axis=0),
                            extent=extent_kcsd,
                            cmap=plt.cm.viridis,
                            origin='lower',
                            aspect='auto',
                            interpolation="none",
                            alpha=0.5)
            ax[2, 0].set_xticklabels([])
            ax[2, 0].set_yticklabels([])

            ax[2, 1].imshow(est_kcsd[:, :, :, t2].sum(axis=0),
                            extent=extent_kcsd,
                            cmap=plt.cm.bwr_r,
                            origin='lower',
                            aspect='auto', vmin=vmin, vmax=vmax,
                            interpolation="none",
                            alpha=0.5)
            ax[2, 1].set_xticklabels([])
            ax[2, 1].set_yticklabels([])

        else:
            
            for j in [0, 1, 2, 3]:
                    ax[0, j].imshow(morpho, extent=extent, origin='lower', aspect="auto", alpha=1)
                    for z in ele_pos:
                        pos_x, pos_y = z[2], z[0]
                        ax[0, j].text(pos_x, pos_y, '*',
                                      ha="center", va="center", color="k", fontsize=3)
            
            cax = pl.make_map_plot(ax[0, 2], ground_truth_t1, extent=extent, alpha=.95, vmin=vmin, vmax=vmax)
            cax = pl.make_map_plot(ax[0, 3], est_skcsd_t1, extent=extent, vmin=vmin, vmax=vmax)

            ax[0, 0].imshow(est_kcsd_pot[:, :, :, t1].sum(axis=1),
                            origin='lower',
                            aspect='auto',
                            cmap=plt.cm.viridis,
                            interpolation="none",
                            extent=extent,
                            alpha=0.5)
            
            ax[0, 0].set_xticklabels([])
            ax[0, 0].set_yticklabels([])
            ax[0, 1].imshow(est_kcsd[:, :, :, t1].sum(axis=1),
                            origin='lower',
                            aspect='auto',
                            interpolation="none",
                            cmap=plt.cm.bwr_r,
                            extent=extent, vmin=vmin, vmax=vmax,
                            alpha=0.5)
            ax[0, 1].set_xticklabels([])
            ax[0, 1].set_yticklabels([])
            
            
            
    for i in range(3):
        for j in range(4):
            if not j:
                ax[i, j].set_title('Potential')
            elif j == 1:
                ax[i, j].set_title('KCSD')
            elif j == 2:
                ax[i, j].set_title('Ground truth')
            elif j == 3:
                ax[i, j].set_title('sKCSD')
                
    fig.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)

