from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD, KCSD
sKCSD.skmonaco_available = False
import corelib.utility_functions as utils
import loadData as ld
import validation.plotting_functions as pl
import sKCSD_utils

n_src = 512
lambd = 1e-1
R = 8e-6/2**.5
if __name__ == '__main__':

    #fname_base = "Figure_6.png"
    #fig_name = sKCSD_utils.make_fig_names(fname_base)
    fname_base = "Figure_6"
    tstop = 70
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    data_dir = []
    colnb = 4
    rows = [2,4,8,16]
    xmin, xmax = -200, 600
    ymin, ymax = -200, 200
    sim_type = {'1':"grid",'2':"random"}
    for i, rownb in enumerate(rows):
        for orientation in [1,2]:
            fname = "Figure_6_"+sim_type[str(orientation)]
            c = sKCSD_utils.simulate(fname,morphology=2,simulate_what="symmetric",colnb=colnb,rownb=rownb,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,tstop=tstop,seed=1988,weight=0.04,n_syn=100,electrode_distribution=orientation,dt=2**(-3))
            data_dir.append(c.return_paths_skCSD_python())
            
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]*1e-3
    dt = c.cell_parameters['dt']
    print(dt)
    t1 = int(42/dt)
    t2 = int(5/dt)
    atstart = t2
    atstop = t2+int(20/dt)

    simulation_paths = []
    data_paths = []
    
    skcsd_grid = []
    skcsd_random = []

    fig, ax = plt.subplots(1,3)
    fname = fname_base+'.png'
    fig_name = sKCSD_utils.make_fig_names(fname)
    vmax, vmin = pl.get_min_max(ground_truth[:,atstart:atstop])
    pl.plot(ax[0],ground_truth[:,atstart:atstop],yticklabels=[x for x in range(0,86,15)],fig=fig,title="Ground truth",ylabel='#segment')

    for i, datd in enumerate(data_dir):
        data = ld.Data(datd)
        ele_pos = data.ele_pos/scaling_factor
        data.LFP = data.LFP/scaling_factor_LFP
        morphology = data.morphology
        morphology[:,2:6] = morphology[:,2:6]/scaling_factor
        k = sKCSD.sKCSD(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R,dist_table_density=100)
        
        est_skcsd = k.values(estimate='CSD',transformation='segments')
        
        est_skcsd /= seglen[:,None]
                
        if i%2:
            skcsd_random.append(est_skcsd)
        else:
            skcsd_grid.append(est_skcsd)
        
        if sys.version_info < (3,0):
            path = os.path.join(datd, "preprocessed_data/Python_2")
        else:
            path = os.path.join(datd, "preprocessed_data/Python_3")

        if not os.path.exists(path):
            print("Creating",path)
            os.makedirs(path)
            
        utils.save_sim(path,k)
        
    skcsd_maps_grid = sKCSD_utils.merge_maps(skcsd_grid,tstart=atstart,tstop=atstop,merge=2)
    pl.plot(ax[1],skcsd_maps_grid,xticklabels=['8','16','32','64'],title="Grid",xlabel='electrodes',vmin=vmin,vmax=vmax)
        
    skcsd_maps_random = sKCSD_utils.merge_maps(skcsd_random,tstart=atstart,tstop=atstop,merge=2)
    pl.plot(ax[2],skcsd_maps_random,xticklabels=['8','16','32','64'],title="Random",xlabel='electrodes',vmin=vmin,vmax=vmax)
    
    fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
