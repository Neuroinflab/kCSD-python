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
import corelib.loadData as ld
import functions as fun

n_src = 512

if __name__ == '__main__':

    #fname_base = "Figure_6.png"
    #fig_name = fun.make_fig_names(fname_base)
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
            c = fun.simulate(fname,morphology=2,simulate_what="symmetric",colnb=colnb,rownb=rownb,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,tstop=tstop,seed=1988,weight=0.04,n_syn=100,electrode_distribution=orientation,dt=2**(-2))
            data_dir.append(c.return_paths_skCSD_python())
            
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]
    dt = c.cell_parameters['dt']
    
    t1 = int(42/dt)
    t2 = int(5/dt)
    atstart = t2
    atstop = int(15/dt)

    R_inits = [2**i for i in range(3,9)]
    lambdas = [10**(-i) for i in range(6)]
    for R_init in R_inits:
        for la in lambdas:
            simulation_paths = []
            data_paths = []
    
            skcsd_grid = []
            skcsd_random = []

            R = R_init/np.sqrt(2)/scaling_factor
            lambd = la#*2*(2*np.pi)**3*R**2*n_src
            fig = plt.figure()
            fig, ax = plt.subplots(1,3)
            fname = fname_base+'_R_%d_lambda_%f.png'%(R_init,la)
            fig_name = fun.make_fig_names(fname)

            fun.plot(ax[0],ground_truth[:,atstart:atstop],yticklabels=[x for x in range(0,86,15)],fig=fig,title="Ground truth",vmin=-0.05,vmax=0.05)

            for i, datd in enumerate(data_dir):
                data = ld.Data(datd)
                ele_pos = data.ele_pos/scaling_factor
                pots = data.LFP/scaling_factor_LFP
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scaling_factor
                k = sKCSD.sKCSD(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R,dist_table_density=100)
        
                est_skcsd = k.values(estimate='CSD',transformation='segments')
                
                est_skcsd /= seglen[:,None]
                
                if i%2:
                    skcsd_random.append(est_skcsd)
                else:
                    print(i)
                    skcsd_grid.append(est_skcsd)
                    #skcsd_grid.append(est_skcsd)
            skcsd_maps_grid = fun.merge_maps(skcsd_grid,tstart=atstart,tstop=atstop,merge=1)
            fun.plot(ax[1],skcsd_maps_grid,xticklabels=['8','16','32','64'],title="Grid")

            skcsd_maps_random = fun.merge_maps(skcsd_random,tstart=atstart,tstop=atstop,merge=1)
            fun.plot(ax[2],skcsd_maps_random,xticklabels=['8','16','32','64'],title="Random")
    
            fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
