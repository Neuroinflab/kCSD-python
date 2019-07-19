from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD, sKCSDcell, KCSD3D
import kcsd.sKCSD_utils as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
import run_LFP
n_src = 1024
dt = 0.5
tolerance = 10e-6
cmap = plt.cm.bwr_r
n = 100
xmin_fig = -1000e-6
xmax_fig = 1000e-6
ymin_fig = -1000e-6
ymax_fig = 1000e-6

different_trials_parameters = {0:{'colnb':5, 'rownb': 5, 'xmin':-100, 'xmax':100, 'ymin':-100, 'ymax':100},
                               1:{'colnb':5, 'rownb': 5, 'xmin':-200, 'xmax':200, 'ymin':-200, 'ymax':200},
                               2:{'colnb':5, 'rownb': 5, 'xmin':-400, 'xmax':400, 'ymin':-400, 'ymax':400},
                               3:{'colnb':5, 'rownb': 5, 'xmin':-800, 'xmax':800, 'ymin':-800, 'ymax':800},
                               4:{'colnb':9, 'rownb': 9, 'xmin':-800, 'xmax':800, 'ymin':-800, 'ymax':800},
                               5:{'colnb':9, 'rownb': 9, 'xmin':-400, 'xmax':400, 'ymin':-400, 'ymax':400},
                               6:{'colnb':21, 'rownb': 21, 'xmin':-400, 'xmax':400, 'ymin':-400, 'ymax':400},
}
titles = ['IED 50 um',
          'IED 100 um',
          'IED 200 um',
          'IED 400 um',
          'IED 200 um',
          'IED 100 um',
          'IED 40 um']
if __name__ == '__main__':
    fname_base = "Figure_10"
    fig_name = sKCSD_utils.make_fig_names(fname_base)
    tstop = 250
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R = 64e-6/np.sqrt(2)
    l = .1
    data_dir = []
    for i in range(7):
        colnb = different_trials_parameters[i]['colnb']
        rownb = different_trials_parameters[i]['rownb']
        xmin = different_trials_parameters[i]['xmin']
        xmax = different_trials_parameters[i]['xmax']
        ymin = different_trials_parameters[i]['ymin']
        ymax = different_trials_parameters[i]['ymax']
        
        c = sKCSD_utils.simulate(fname_base,
                                 morphology=7,
                                 tstop=tstop,
                                 seed=1988,
                                 weight=0.01,
                                 n_syn=1000,
                                 simulate_what='oscillatory',
                                 electrode_distribution=1,
                                 electrode_orientation=3,
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax,
                                 colnb=colnb,
                                 rownb=rownb,
                                 dt=dt)
        c.save_for_R_kernel()
        data_dir.append(c.return_paths_skCSD_python())
    
    data = utils.LoadData(data_dir[0])
    ele_pos = data.ele_pos/scale_factor
    data.LFP = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scale_factor
    seglen = np.loadtxt(os.path.join(data_dir[0],
                                     'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0],
                                           'membcurr'))/seglen[:, None]*1e-3
    somav = np.loadtxt(os.path.join(data_dir[0],
                                           'somav.txt'))
    time = np.linspace(0, tstop, len(somav))
    plt.figure()
    plt.plot(time, somav)
    
    dt = c.cell_parameters['dt']
    t0 = int(247.5/dt)#np.argmax(somav)
    print(t0*dt)
    L1 = []
    cell_itself = sKCSDcell(morphology,
                            ele_pos,
                            n_src,
                            tolerance=tolerance,
                            xmin=-1000e-6,
                            xmax=1000e-6,
                            ymin=-1000e-6,
                            ymax=1000e-6)
    ground_truth_3D = cell_itself.transform_to_3D(ground_truth,
                                                  what="morpho")
    ground_truth_t0 = ground_truth_3D[:, :, :, t0].sum(axis=2)
    vmax, vmin = pl.get_min_max(ground_truth_t0)
    morpho, extent = cell_itself.draw_cell2D(axis=2)
    extent = [extent[-2], extent[-1], extent[0], extent[1]]
    lambd = l/(2*(2*np.pi)**3*R**2*n_src)
    fig, ax = plt.subplots(3, 3, figsize=(8, 20))
    pl.make_map_plot(ax[0, 0], morpho, extent=extent, circles=False)
    pl.make_map_plot(ax[0, 0], ground_truth_t0, extent=extent, title="Ground truth", vmin=vmin, vmax=vmax, alpha=.75)
    vmax, vmin = pl.get_min_max(ground_truth_t0)
    
    for i, di in enumerate(data_dir):
        data = utils.LoadData(di)
        ele_pos = data.ele_pos/scale_factor
        data.LFP = data.LFP/scale_factor_LFP
        morphology = data.morphology
        morphology[:, 2:6] = morphology[:, 2:6]/scale_factor
        ker = sKCSD(ele_pos,
                    data.LFP,
                    morphology,
                    n_src_init=n_src,
                    src_type='gauss',
                    lambd=lambd,
                    R_init=R,
                    tolerance=tolerance,
                    dist_table_density=20,
                    exact=True,
                    sigma=0.3)
    
            
        path = os.path.join('simulation', '%s_lambda_%f_R_%f' % (fname_base, l, R))
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

    
        skcsd, pot, cell_obj = utils.load_sim(path)
        skcsd = cell_itself.transform_to_3D(skcsd)
            
        ax[(i+1)//3, (i+1)%3].imshow(morpho,
                                     origin='lower',
                                     aspect='auto',
                                     interpolation='none',
                                     extent=extent)
        pl.make_map_plot(ax[(i+1)//3, (i+1)%3],
                         skcsd[:,:,:,t0].sum(axis=2),
                         vmin=vmin,
                         vmax=vmax,
                         extent=extent,
                         cmap=cmap,
                         alpha=.75,
                         title=titles[i])
            
        for j in range(ele_pos.shape[0]):
            pos_x, pos_y = 1e6*ele_pos[j, 0], 1e6*ele_pos[j, 1]
            text = ax[(i+1)//3, (i+1)%3].text(pos_x, pos_y, '*',
                            ha="center", va="center", color="k",
                            fontsize=4)
        fig.suptitle('lambda %f, R %f' % (l, R))
    fig.savefig(fig_name+'.png',
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
 
