from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD3D
import corelib.utility_functions as utils
import corelib.loadData as ld
import functions as fun

n_src = 512
lambd = 1e-1
R = 64e-6/2**.5

if __name__ == '__main__':
    fname_base = "Figure_3"

    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [2**i for i in range(3,8)]
    lambdas = [10**(-i) for i in range(6)]
    electrode_number = [8,32,128]
    data_dir = []
    xmin, xmax = -100, 600
    ymin, ymax = 0, 200
    orientation = 1
    for rownb in electrode_number:
        fname = "Figure_3"
        c = fun.simulate(fname,morphology=1,simulate_what="random",colnb=1,rownb=rownb,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,tstop=tstop,seed=1988,weight=0.1,n_syn=100)
        data_dir.append(c.return_paths_skCSD_python())
        
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))#/scaling_factor
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]
    gvmax, gvmin = fun.get_min_max(ground_truth)
    for R_init in R_inits:
        for l in lambdas:
            
            
            R = R_init/np.sqrt(2)/scaling_factor
            lambd = l*2*(2*np.pi)**3*R**2*n_src
            print(R,lambd,l)
            data_paths = []
            fig = plt.figure()
            ax = []
            for i in range(6):
                ax.append(fig.add_subplot(2,3,i+1))
    
            cax = ax[0].imshow(ground_truth,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r',vmax=gvmax,vmin=gvmin)
            #cbar = fig.colorbar(cax,ticks=[gvmin,gvmax])
            #cbar.ax.set_yticklabels(['source','sink'])
            new_fname = fname_base+'_R_'+str(R_init)+'_lambda_'+str(l)+'.png'
            fig_name = fun.make_fig_names(new_fname)
            print(fig_name)
            for i, datd in enumerate(data_dir):
                data = ld.Data(datd)
                ele_pos = data.ele_pos/scaling_factor
                pots = data.LFP/scaling_factor_LFP
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scaling_factor
                k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
                csd = k.values(segments=True)
                
        
                vmax, vmin = fun.get_min_max(csd)
                cax = ax[i+3].imshow(csd,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r',vmax=vmax,vmin=vmin)
                #cbar = fig.colorbar(cax,ticks=[vmin,vmax])
                ax[i+3].set_title(electrode_number[i])
                #cbar.ax.set_yticklabels(['source','sink'])
        
            fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)
