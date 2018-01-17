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

R_init = 10
lambd = 0.0001
n_src = 512
if __name__ == '__main__':
    fname_base = "ball_stick_"
    
    tstop = 75
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    R_inits = [2**i/np.sqrt(2)*3 for i in [3,4,5,6,7]]
    electrode_number = [8,16,32,64,128]
    nx,ny,nz = 10,10,52*4
    
    for lambd in [1e-5,1e-4,1e-3,1e-2,1e-1,1,0]:
        for R_init in R_inits:
            simulation_paths = []
            data_paths = []
            fig_lfp = plt.figure()
            ax_lfp = []
            l = 0
            for rownb in electrode_number:
                fname = fname_base+str(rownb)
                c = run_LFP.CellModel(morphology=1,cell_name=fname,colnb=1,rownb=rownb,xmin=-100,xmax=600,ymin=0,ymax=200,tstop=tstop,seed=1988,weight=0.01,n_syn=100)
                c.simulate()
                c.save_skCSD_python()
                c.save_memb_curr()
                c.save_seg_length()
                data_dir = c.return_paths_skCSD_python()
                print(data_dir)
                data = ld.Data(data_dir)
 
                ele_pos = data.ele_pos/scaling_factor

                pots = data.LFP#/scaling_factor_LFP
                ax_lfp.append(fig_lfp.add_subplot(1,5,l+1))
                x = ax_lfp[l].imshow(data.LFP,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='bwr')
                
                l = l+1
  
                params = {}
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scaling_factor
                R = R_init/scaling_factor
                xmin = morphology[:,2].min()-morphology[:,5].max()*2
                xmax = morphology[:,2].max()+morphology[:,5].max()*2
                ymin = morphology[:,3].min()-morphology[:,5].max()*2
                ymax = morphology[:,3].max()+morphology[:,5].max()*2
                zmin = morphology[:,4].min()-morphology[:,5].max()*2
                zmax = morphology[:,4].max()+morphology[:,5].max()*2
                #print(ele_pos,'\n',R,'\n',morphology[:,2:5])
                gdx = (xmax-xmin)/nx
                gdy = (ymax-ymin)/ny
                gdz = (zmax-zmin)/nz
                k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, gdx=gdx, gdy=gdy, gdz=gdz, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, n_src_init=n_src, src_type='gauss_lim',lambd=lambd,R_init=R)
                #k.cross_validate()
                dir_name = "ball_stick_R_"+str(R_init)+'_lambda_'+str(lambd)+'_src_'+str(n_src)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)
                if sys.version_info >= (3, 0):
                    path = os.path.join(data_dir,"preprocessed_data/Python_3", dir_name)
                else:
                    path = os.path.join(data_dir,"preprocessed_data/Python_2",dir_name)
                
                if not os.path.exists(path):
                    print("Creating",path)
                    os.makedirs(path)
        
                utils.save_sim(path,k)
                simulation_paths.append(path)
                data_paths.append(data_dir)
               
            #data_paths = ['simulation/ball_stick_8','simulation/ball_stick_32','simulation/ball_stick_128']
            #simulation_paths = ['simulation/ball_stick_8/preprocessed_data/Python_2','simulation/ball_stick_32/preprocessed_data/Python_2','simulation/ball_stick_128/preprocessed_data/Python_2']

            seglen = np.loadtxt(os.path.join(data_paths[0],'seglength'))#/scaling_factor
            ground_truth = np.loadtxt(os.path.join(data_paths[0],'membcurr'))
            LFP = pots
   
            ground_truth = ground_truth/seglen[:,None]
            
                
            fig = plt.figure()
            ax = []
            fig2 = plt.figure()
            ax2 = []
            for i in range(6):
                ax.append(fig.add_subplot(2,3,i+1))
                if i < 5:
                    ax2.append(fig2.add_subplot(1,5,i+1))
            
                if not i:
                    vmin,vmax = ground_truth.min(),ground_truth.max()
                    if vmin < 0 and vmax > 0:
                        if abs(vmax) > abs(vmin):
                            vmin = -vmax
                        else:
                            vmax = -vmin
                    cax = ax[0].imshow(ground_truth,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r',vmax=vmax,vmin=vmin)
                    cbar = fig.colorbar(cax,ticks=[vmin,vmax])
                    cbar.ax.set_yticklabels(['source','sink'])
          

                else:
                    path = simulation_paths[i-1]
                    data_dir = data_paths[i-1]
                    data = ld.Data(data_dir)
                    est_csd, est_pot, cell_obj = utils.load_sim(path)
                    new_csd = np.zeros(est_csd.shape[2:])
                    new_pot = np.zeros(est_pot.shape[2:])
                    zdim = est_csd.shape[2]
                    tot_len = sum(seglen)
                    for k in range(new_csd.shape[0]):
                        for j in range(new_csd.shape[1]):
                            new_csd[k,j] = est_csd[:,:,k,j].sum()/(est_csd.shape[0]*est_csd.shape[1])
                            new_pot[k,j] = est_pot[:,:,k,j].sum()/(est_pot.shape[0]*est_pot.shape[1])
                 
                    csd = np.zeros(ground_truth.shape)
                    where_am_I = 0
                    for j,seg in enumerate(seglen):
                        if j == len(seglen)-1:
                            csd[j,:] = new_csd[where_am_I:,:].sum(axis=0)/seg
                        else:
                            no_vox = round(seg*zdim/tot_len)
                            csd[j,:] = new_csd[where_am_I:where_am_I+no_vox,:].sum(axis=0)/seg
                            where_am_I += no_vox
            
                    
                    vmin,vmax = csd.min(),csd.max()
                    print(vmin,vmax)
                    if vmin*vmax < 0:
                        if abs(vmax) > abs(vmin):
                            vmin = -vmax
                        else:
                            vmax = -vmin
                        
                        
                    cax = ax[i].imshow(csd,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r',vmax=vmax,vmin=vmin)
                    cbar = fig.colorbar(cax,ticks=[vmin,vmax])
                    
                    
                    cax2 = ax2[i-1].imshow(new_pot,extent=[0, tstop,1, 52,],origin='lower',aspect='auto',cmap='seismic_r')#,vmax=vmax,vmin=vmin)

                    ax[i].set_title(electrode_number[i-1])
                    cbar.ax.set_yticklabels(['source','sink'])
    
            fig.savefig(dir_name+'.png', bbox_inches='tight', transparent=True, pad_inches=0.1)

    
            #plt.show()
