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

if __name__ == '__main__':
    fname_base = "ball_stick_random"
    fig_dir = 'Figures'
    if not os.path.exists(fig_dir):
        print("Creating",fig_dir)
        os.makedirs(fig_dir)
        
    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [(2**i)/np.sqrt(2) for i in [3,4,5,6,7]]
    electrode_number = [8,16,32,64,128]
    data_dir = []
    
    for rownb in electrode_number:
        fname = fname_base+str(rownb)
        c = run_LFP.CellModel(morphology=1,cell_name=fname,colnb=1,rownb=rownb,xmin=-100,xmax=600,ymin=0,ymax=200,tstop=tstop,seed=1988,weight=0.01,n_syn=100)
        c.simulate()
        c.save_skCSD_python()
        c.save_memb_curr()
        c.save_seg_length()
        data_dir.append(c.return_paths_skCSD_python())
        
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))#/scaling_factor
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]
    gvmin, gvmax = fun.get_min_max(ground_truth)
    
    for lambd in [1e-5]:
        for R_init in [64/2**0.5]:
            simulation_paths = []
            data_paths = []
            fig = plt.figure()
            ax = []
            for i, datd in enumerate(data_dir):
                l = 0
                data = ld.Data(datd)
                ele_pos = data.ele_pos/scaling_factor
                pots = data.LFP/scaling_factor_LFP
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scaling_factor
                R = R_init/scaling_factor
               
                k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
                est_csd = k.values()
                dir_name = "ball_stick_random_R_"+str(R_init)+'_lambda_'+str(lambd)+'_src_'+str(n_src)
                fig_name = os.path.join(fig_dir,dir_name+'.png')
                if sys.version_info >= (3, 0):
                    new_path = os.path.join(datd,"preprocessed_data/Python_3", dir_name)
                else:
                    new_path = os.path.join(datd,"preprocessed_data/Python_2",dir_name)
                
                if not os.path.exists(new_path):
                    print("Creating",new_path)
                    os.makedirs(new_path)
        
                utils.save_sim(new_path,k)
                simulation_paths.append(new_path)
                ax.append(fig.add_subplot(2,3,i+1))
                if not i:
                    cax = ax[0].imshow(ground_truth,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r',vmax=gvmax,vmin=gvmin)
                    cbar = fig.colorbar(cax,ticks=[gvmin,gvmax])
                    cbar.ax.set_yticklabels(['source','sink'])
                else:
                    new_csd = np.zeros(est_csd.shape[2:])
                    zdim = est_csd.shape[2]
                    tot_len = sum(seglen)
                    for k in range(new_csd.shape[0]):
                        for j in range(new_csd.shape[1]):
                            new_csd[k,j] = est_csd[:,:,k,j].sum()/(est_csd.shape[0]*est_csd.shape[1])*(np.pi)**0.5*R_inits[i-1]
                 
                    csd = np.zeros(ground_truth.shape)
                    where_am_I = 0
                    for j,seg in enumerate(seglen):
                        if j == len(seglen)-1:
                            csd[j,:] = new_csd[where_am_I:,:].sum(axis=0)/seg
                        else:
                            no_vox = int(round(seg*zdim/tot_len))
                            csd[j,:] = new_csd[where_am_I:where_am_I+no_vox,:].sum(axis=0)/seg
                            where_am_I += no_vox
            
                    
                    
                        
                    vmin, vmax = fun.get_min_max(csd)
                    cax = ax[i].imshow(csd,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r',vmax=vmax,vmin=vmin)
                    cbar = fig.colorbar(cax,ticks=[vmin,vmax])
                    ax[i].set_title(electrode_number[i-1])
                    cbar.ax.set_yticklabels(['source','sink'])
    
            fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)

    
            #plt.show()
