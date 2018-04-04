from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD3D, KCSD
import corelib.utility_functions as utils
import corelib.loadData as ld
import functions as fun

n_src = 512
lambd = 1e-5
R = 16e-6/2**.5

if __name__ == '__main__':

    fname_base = "Figure_6.png"
    fig_name = fun.make_fig_names(fname_base)

    atstart = 50
    atstop = 65
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    data_dir = []
    colnb = 4
    rows = [2,4,8,16]
    xmin, xmax = -200, 600
    ymin, ymax = -200, 200
    sim_type = {'1':"grid",'2':"random"}
    for i, rownb in enumerate(rows):
        for orientation in [1,2]:
            fname = "Figure_6_"+sim_type[str(orientation)]
            c = fun.simulate(fname,morphology=2,simulate_what="symmetric",colnb=colnb,rownb=rownb,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,tstop=tstop,seed=1988,weight=0.04,n_syn=100,electrode_distribution=orientation)
            data_dir.append(c.return_paths_skCSD_python())
            
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]

    simulation_paths = []
    data_paths = []
    fig = plt.figure()
    fig, ax = plt.subplots(1,3)
    print(ground_truth.shape)
    fun.plot(ax[0],ground_truth[:,atstart:atstop],yticklabels=[x for x in range(0,86,15)],fig=fig,title="Ground truth",vmin=-0.05,vmax=0.05)
    
    skcsd_grid = []
    skcsd_random = []
            
    for i, datd in enumerate(data_dir):
        data = ld.Data(datd)
        ele_pos = data.ele_pos/scale_factor
        pots = data.LFP/scale_factor_LFP
        morphology = data.morphology
        morphology[:,2:6] = morphology[:,2:6]/scale_factor
        k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
        
        est_skcsd = k.values(estimate='CSD',segments=True)#/seglen[:,None]
      
        if i%2:
           skcsd_random.append(est_skcsd)
        else:
           skcsd_grid.append(est_skcsd)
        #skcsd_grid.append(est_skcsd)
    skcsd_maps_grid = fun.merge_maps(skcsd_grid,tstart=atstart,tstop=atstop,merge=2)
    fun.plot(ax[1],skcsd_maps_grid,xticklabels=['8','16','32','64'],title="Grid")

    skcsd_maps_random = fun.merge_maps(skcsd_random,tstart=atstart,tstop=atstop,merge=2)
    fun.plot(ax[2],skcsd_maps_random,xticklabels=['8','16','32','64'],title="Random")
    
    fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
