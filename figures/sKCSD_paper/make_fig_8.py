from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, sKCSDcell, KCSD3D
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
import run_LFP
n_src = 512
dt = 0.5
tolerance = 5e-6
cmap = plt.cm.bwr_r
if __name__ == '__main__':
    fname_base = "Figure_8"
    fig_name = sKCSD_utils.make_fig_names(fname_base)
    print(fig_name)
    tstop = 85
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R = 16e-6/np.sqrt(2)
    
    colnb = 8
    rownb = 16
    c = sKCSD_utils.simulate(fname_base,
                             morphology=7,
                             tstop=tstop,
                             seed=1988,
                             weight=0.01,
                             n_syn=1000,
                             simulate_what='oscillatory',
                             electrode_distribution=3,
                             electrode_orientation=3,
                             xmin=-300,
                             xmax=400,
                             ymin=-600,
                             ymax=400,
                             triside=45,
                             colnb=colnb,
                             rownb=rownb,
                             dt=dt)
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
    t0 = np.argmax(somav)
 
    vmax, vmin = pl.get_min_max(ground_truth)
    
    
    cell_itself = sKCSDcell(morphology, ele_pos, n_src, tolerance=tolerance)
    cell_itself.distribute_srcs_3D_morph()
    ground_truth_3D = cell_itself.transform_to_3D(ground_truth,
                                                  what="morpho")
    vmax, vmin = pl.get_min_max(ground_truth_3D[:, :, :, t0])
    morpho, extent = cell_itself.draw_cell2D(axis=2)
    extent = [-600, 400, -400, 400]
    for l in [1]:#, 1e-1, 1e-2, 1e-3, 1e-4]:
        for R in [16e-6/np.sqrt(2), 8e-6/np.sqrt(2)]:#, 32e-6/np.sqrt(2)]:
            print(l/(2*(2*np.pi)**3*R**2*n_src))
            lambd = l/(2*(2*np.pi)**3*R**2*n_src)
            fig = plt.figure()
            ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        
            ax1.plot(time, somav)
            ax1.set_xlabel('time (ms)')
            ax1.set_ylabel('Vm (mV)')
            ax2 = plt.subplot2grid((3, 2), (1, 0))
            ax3 = plt.subplot2grid((3, 2), (1, 1))
            ax4 = plt.subplot2grid((3, 2), (2, 0))
            ax5 = plt.subplot2grid((3, 2), (2, 1))
            
            ker = sKCSD(ele_pos,
                        data.LFP,
                        morphology,
                        n_src_init=n_src,
                        src_type='gauss',
                        lambd=lambd,
                        R_init=R,
                        dist_table_density=250,
                        tolerance=tolerance,
                        exact=True)
 
    
    
            skcsd = ker.values()
            path = '%s_lambda_%f_R_%f' % (fname_base, l, R)
            if sys.version_info < (3, 0):
                path = os.path.join(path, "preprocessed_data/Python_2")
            else:
                path = os.path.join(path, "preprocessed_data/Python_3")
            if not os.path.exists(path):
                print("Creating", path)
                os.makedirs(path)
            # skcsd, pot, cell_obj = utils.load_sim(path)
            utils.save_sim(path, ker)
            est_skcsd = cell_itself.transform_to_3D(skcsd)
            ax4.imshow(morpho,
                       origin='lower',
                       aspect='auto',
                       interpolation='none',
                       extent=extent)
            cax = ax4.imshow(ground_truth_3D[:, :, :, t0].sum(axis=2),
                             origin='lower',
                             aspect='auto',
                             interpolation='none',
                             vmin=vmin,
                             vmax=vmax,
                             extent=extent,
                             cmap=cmap,alpha=.5)
            

            ax5.imshow(morpho,
                       origin='lower',
                       aspect='auto',
                       interpolation='none',
                       extent=extent)
            cax2 = ax5.imshow(est_skcsd[:, :, :, t0].sum(axis=2),
                             origin='lower',
                             aspect='auto',
                             interpolation='none',
                             vmin=vmin,
                             vmax=vmax,
                             extent=extent,
                             cmap=cmap,alpha=.5)

            fig.suptitle('lambda %f, R %f' % (l, R))
            xmin = cell_itself.xmin
            xmax = cell_itself.xmax
            ymin = cell_itself.ymin
            ymax = cell_itself.ymax
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

            kcsd_csd = kcsd.values()
            kcsd_pot = kcsd.values("POT")
            ax2.imshow(morpho,
                       origin='lower',
                       aspect='auto',
                       interpolation='none',
                       extent=extent)
            ax3.imshow(morpho,
                       origin='lower',
                       aspect='auto',
                       interpolation='none',
                       extent=extent)

            cax = ax2.imshow(kcsd_pot[:, :, :, t0].sum(axis=2),
                             origin='lower',
                             aspect='auto',
                             interpolation='none',
                             vmin=vmin,
                             vmax=vmax,
                             extent=extent,
                             cmap=plt.cm.viridis,alpha=.5)
            cax = ax3.imshow(kcsd_csd[:, :, :, t0].sum(axis=2),
                             origin='lower',
                             aspect='auto',
                             interpolation='none',
                             vmin=vmin,
                             vmax=vmax,
                             extent=extent,
                             cmap=cmap,alpha=.5)

            for i in range(ele_pos.shape[0]):
                pos_x, pos_y = 1e6*ele_pos[i, 0], 1e6*ele_pos[i, 1]
                text = ax2.text(pos_x, pos_y, '*',
                                ha="center", va="center", color="k")
                text = ax3.text(pos_x, pos_y, '*',
                                ha="center", va="center", color="k")
                text = ax4.text(pos_x, pos_y, '*',
                                ha="center", va="center", color="k")
                text = ax5.text(pos_x, pos_y, '*',
                                ha="center", va="center", color="k")

            ax1.set_title('Time %d', int(t0))
            ax2.set_title('Potential')
            ax1.set_title('kCSD')
            ax1.set_title('Ground truth')
            ax1.set_title('skCSD')
            fig.savefig(fig_name+'%_f_%f.png'%(R, l),
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
                
    plt.show()
 
