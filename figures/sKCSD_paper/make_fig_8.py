import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, sKCSDcell, KCSD3D
from kcsd import sKCSD_utils as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
n_src = 1024
dt = 0.5
tolerance = 5e-6
cmap = plt.cm.bwr_r
n = 100
scale_factor = 1000**2
R = 32e-6
lambd = 0.0001
if __name__ == '__main__':
    fname_base = "Figure_8"
    fig_name = sKCSD_utils.make_fig_names(fname_base)
    tstop = 850
    scale_factor_LFP = 1000
    xmin = -650/scale_factor
    xmax = 650/scale_factor
    ymin = -650/scale_factor
    ymax = 650/scale_factor
    colnb = 8
    rownb = 16
    c = sKCSD_utils.simulate(fname_base,
                             morphology=7,
                             tstop=tstop,
                             seed=1988,
                             weight=0.01,
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
                             dt=dt)
    c.save_for_R_kernel()
    data_dir = c.return_paths_skCSD_python()
    data = utils.LoadData(data_dir)
    ele_pos = data.ele_pos/scale_factor
    data.LFP = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scale_factor
    seglen = np.loadtxt(os.path.join(data_dir,
                                     'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir,
                                           'membcurr'))/seglen[:, None]*1e-3
    dt = c.cell_parameters['dt']
   
    somav = np.loadtxt(os.path.join(data_dir,
                                           'somav.txt'))
    time = np.linspace(0, tstop, len(somav))
    t0 = np.argmax(somav[int(400./dt):int(600./dt)])+int(400./dt)
 
    cell_itself = sKCSDcell(morphology, ele_pos, n_src, tolerance=tolerance, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    ground_truth_3D = cell_itself.transform_to_3D(ground_truth[:, t0],
                                                  what="morpho")
    vmax, vmin = pl.get_min_max(ground_truth_3D[:, :, :].sum(axis=(2)))
    morpho, extent = cell_itself.draw_cell2D(axis=2)
    extent = [extent[-2], extent[-1], extent[0], extent[1]]
    gvmax, gvmin = pl.get_min_max(ground_truth)
    fig = plt.figure(figsize=(8, 20))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(time, somav)
    lims = ax1.get_ylim()
    ax1.plot(time[t0]*np.ones_like(time), np.linspace(lims[0], lims[-1], len(time)), 'r')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('Vm (mV)')
    ax2 = plt.subplot2grid((4, 2), (2, 0))
    ax3 = plt.subplot2grid((4, 2), (2, 1))
    ax4 = plt.subplot2grid((4, 2), (3, 0))
    ax5 = plt.subplot2grid((4, 2), (3, 1))
    cell_itself = sKCSDcell(morphology,
                            ele_pos,
                            n_src,
                            tolerance=tolerance,
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax)
    ker = sKCSD(ele_pos,
                data.LFP,
                morphology,
                n_src_init=n_src,
                src_type='gauss',
                lambd=lambd,
                R_init=R,
                tolerance=tolerance,
                exact=True,
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
        utils.save_sim(path, ker)
    except NameError:
        pass
    skcsd, pot, morphology, ele_pos, n_src = utils.load_sim(path)
    cell_object = sKCSDcell(morphology, ele_pos, n_src)
    est_skcsd = cell_itself.transform_to_3D(skcsd)
    skcsd_seg = cell_itself.transform_to_segments(skcsd)
    skcsd_snapshot = est_skcsd.sum(axis=(2))[:, :, 0]
    gt_snapshot = ground_truth_3D.sum(axis=(2))[:, :, 0]

    cax = pl.make_map_plot(ax4,
                           gt_snapshot,
                           vmin=vmin,
                           vmax=vmax,
                           extent=extent,
                           cmap=cmap,
                           alpha=.95,
                           ele_pos=ele_pos,
                           morphology=morpho)
            
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    
    
  
    cax2 = pl.make_map_plot(ax5,
                            skcsd_snapshot,
                            vmin=vmin,
                            vmax=vmax,
                            extent=extent,
                            alpha=.95,
                            ele_pos=ele_pos,
                            morphology=morpho)
    
    fig.suptitle('lambda %f, R %f' % (lambd, R))
    xmin = cell_itself.xmin
    xmax = cell_itself.xmax
    ymin = cell_itself.ymin
    ymax = cell_itself.ymax
    zmin = cell_itself.zmin
    zmax = cell_itself.zmax
    gdx = (xmax-xmin)/50
    gdy = (ymax-ymin)/50
    gdz = (zmax-zmin)/5

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

    kcsd_csd = kcsd.values()
    kcsd_pot = kcsd.values("POT")

    pl.make_map_plot(ax2, kcsd_pot[:, :, :,
                                   t0].sum(axis=(2)),
                     extent=extent,
                     cmap=plt.cm.viridis,
                     alpha=.9,
                     ele_pos=ele_pos,
                     morphology=morpho)
    pl.make_map_plot(ax3, kcsd_csd[:, :, :,
                                   t0].sum(axis=(2)),
                     vmin=vmin,
                     vmax=vmax,
                     extent=extent,
                     cmap=cmap,
                     alpha=.9,
                     ele_pos=ele_pos,
                     morphology=morpho)

    ax1.set_title('Time %d' % t0)
    ax2.set_title('Potential')
    ax3.set_title('kCSD')
    ax4.set_title('Ground truth')
    ax5.set_title('skCSD')
    fig.savefig(fig_name+'.png',
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1) 
    plt.show()
