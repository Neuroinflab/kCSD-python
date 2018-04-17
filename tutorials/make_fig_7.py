from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD, KCSD
import corelib.utility_functions as utils
import corelib.loadData as ld
import functions as fun

if __name__ == '__main__':
    
    fname_base = "Figure_7.png"
    fig_name = fun.make_fig_names(fname_base)
    
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1
    
    R_inits = np.array([(2**(i-.5))/scale_factor for i in range(3,9)])
    lambdas = np.array([(10**(-i))for i in range(6)])
    n_srcs = np.array([32,64,128,512,1024])
    x_ticklabels = [2**i for i in range(3,9)]
    y_ticklabels = [str(lambd) for lambd in lambdas]

    colnb = 4
    rownb = 4
    c = fun.simulate(fname_base,morphology=2,colnb=colnb,rownb=rownb,xmin=-200,xmax=600,ymin=-200,ymax=200,tstop=tstop,seed=1988,weight=0.04,n_syn=100,simulate_what='symmetric')
    
    data = ld.Data(c.return_paths_skCSD_python())
    ele_pos = data.ele_pos/scale_factor
    pots = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:,2:6] = morphology[:,2:6]/scale_factor
    

    ground_truth = np.loadtxt(os.path.join(c.return_paths_skCSD_python(),'membcurr'))
    
    outs = np.zeros((len(n_srcs),len(lambdas),len(R_inits)))

    for i, n_src in enumerate(n_srcs):
        for j, l in enumerate(lambdas):
            for k, R in enumerate(R_inits):
                lambd = l*2*(2*np.pi)**3*R**2*n_src
                ker = sKCSD.sKCSD(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
                est_skcsd = ker.values(estimate='CSD',segments=True)
                print(est_skcsd.max(),est_skcsd.min())
                outs[i,j,k] = fun.L1_error(ground_truth, est_skcsd)
                print(outs[i,j,k])

    fig, ax = plt.subplots(1, 4, sharey=True)
    vmax = outs.max()
    vmin = outs.min()
    print(vmax,vmin)
    for i, ax_i in enumerate(ax):
        title = "M = %d"%n_srcs[i]
        if not i:
            fun.plot(ax_i,outs[i],yticklabels=y_ticklabels, xticklabels=x_ticklabels,vmin=vmin,vmax=vmax,title=title)
        elif i<3:
            fun.plot(ax_i,outs[i], xticklabels=x_ticklabels,vmin=vmin,vmax=vmax, title=title)

        else:
            fun.plot(ax_i,outs[i], xticklabels=x_ticklabels,fig=fig,vmin=vmin,vmax=vmax,sinksource=False,title=title)
        
    plt.show()
    fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
