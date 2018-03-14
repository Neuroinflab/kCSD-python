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
def merge_maps(maps,tstart=40,tstop=60,merge=4):
    single_width = (tstop-tstart)//merge
    outs = np.zeros((maps[0].shape[0],single_width*len(maps)))
    for i,mappe in enumerate(maps):
        outs[:,i*single_width:(i+1)*single_width] = make_output(mappe,tstart=tstart,tstop=tstop,merge=merge)

    return outs

def make_output(what,tstart=40,tstop=60,merge=4):
    plotage = what[:,tstart:tstop]
    out = np.zeros((what.shape[0],(tstop-tstart)//merge))
    
    for k in range((tstop-tstart)//merge):
        out[:,k] = plotage[:,k*merge:(k+1)*merge].sum(axis=1)/merge
    return out

def plot(ax,i,what,xticks=[],yticks=[],fig=None,title=None):
    xmax, xmin = 0.05, -0.05#fun.get_min_max(what)
    cax = ax[i].imshow(what,origin='lower',aspect='auto',cmap='seismic_r',interpolation='none',vmin=xmin,vmax=xmax)
    ax[i].set_xticks(xticks)
   
    ax[i].set_yticks(yticks)
    if xticks != []:
        ax[i].set_xticklabels(['8','16','32','64'])
    
    if not i:
        cbar = fig.colorbar(cax, ticks=[-0.05, 0, 0.05])

    if title:
        ax[i].set_title(title)
    return cax

if __name__ == '__main__':
    fname_base = "y_shaped_symmetric_different_electrode_positions"
    atstart = 41
    atstop = 51
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R_inits = np.array([(2**i)/np.sqrt(2)/scale_factor for i in range(9)])
    electrode_number = [8,16,32,64]
    data_dir_grid = []
    data_dir_random = []
    colnb = 4
    rows = [2,4,8,16]
    lfps_grid = []
    lfps_random = []
    #extent = [-200, 200, -200, 600]
    lambd = 1e-5
    lambdas = np.array([(10**(-i))for i in range(10)])
    R = 0.02
    for i, rownb in enumerate(rows):
        fname = fname_base+str(rownb)+'_grid_'
        c = run_LFP.CellModel(morphology=2,cell_name=fname,colnb=colnb,rownb=rownb,xmin=-200,xmax=600,ymin=-200,ymax=200,tstop=tstop,seed=1988,weight=0.04,n_syn=100)
        c.simulate('symmetric')
        c.save_skCSD_python()
        c.save_memb_curr()
        c.save_seg_length()
        lfps_grid.append(c.simulation_parameters['electrode'].LFP)
        data_dir_grid.append(c.return_paths_skCSD_python())
        data = ld.Data(c.return_paths_skCSD_python())
        ele_pos = data.ele_pos/scale_factor
        pots = data.LFP/scale_factor_LFP
        morphology = data.morphology
        morphology[:,2:6] = morphology[:,2:6]/scale_factor
        
    for i, rownb in enumerate(rows):
        fname = fname_base+str(rownb)+'_random_'
        c = run_LFP.CellModel(morphology=2,cell_name=fname,electrode_distribution=2,colnb=colnb,rownb=rownb,xmin=-200,xmax=600,ymin=-200,ymax=200,tstop=tstop,seed=1988,weight=0.01,n_syn=100)
        c.simulate('symmetric')
        c.save_skCSD_python()
        c.save_memb_curr()
        c.save_seg_length()
        lfps_random.append(c.simulation_parameters['electrode'].LFP)
        data_dir_random.append(c.return_paths_skCSD_python())
        data = ld.Data(c.return_paths_skCSD_python())
        ele_pos = data.ele_pos/scale_factor
        pots = data.LFP/scale_factor_LFP
        morphology = data.morphology
        morphology[:,2:6] = morphology[:,2:6]/scale_factor
       
    seglen = np.loadtxt(os.path.join(data_dir_random[0],'seglength'))#/scale_factor
    ground_truth = np.loadtxt(os.path.join(data_dir_random[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]

    for lambd in [1e-5]:
        for R_init in [16/scale_factor/2**.5]:
            simulation_paths = []
            data_paths = []
            fig = plt.figure()
            ax = []
            for j in range(3):
                ax.append(fig.add_subplot(1,3,j+1))
            plot(ax,0,ground_truth[:,atstart:atstop],yticks=[x for x in range(0,86,5)],fig=fig,title="Ground truth")
            R = R_init
            skcsd_grid = []
            skcsd_random = []
            
            for i, datd in enumerate(data_dir_grid):
                data = ld.Data(datd)
                ele_pos = data.ele_pos/scale_factor
                pots = data.LFP/scale_factor_LFP
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scale_factor
                k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
                                  
                est_skcsd = k.values(estimate='CSD',segments=True)/seglen[:,None]
                dir_name = fname_base+"_grid_R_"+str(R_init)+'_lambda_'+str(lambd)+'_src_'+str(n_src)

                if sys.version_info >= (3, 0):
                    new_path = os.path.join(datd,"preprocessed_data/Python_3", dir_name)
                else:
                    new_path = os.path.join(datd,"preprocessed_data/Python_2",dir_name)
                
                if not os.path.exists(new_path):
                    print("Creating",new_path)
                    os.makedirs(new_path)

                utils.save_sim(new_path,k)
                simulation_paths.append(new_path)
                skcsd_grid.append(est_skcsd)
            skcsd_maps = merge_maps(skcsd_grid,tstart=atstart,tstop=atstop,merge=2)
            step = skcsd_maps.shape[1]/4
            xticks = [i+2 for i in range(0, skcsd_maps.shape[1],5)]
                      
            plot(ax,1,skcsd_maps,xticks=xticks,title="Grid")
            for i, datd in enumerate(data_dir_random):
                data = ld.Data(datd)
                ele_pos = data.ele_pos/scale_factor
                pots = data.LFP/scale_factor_LFP
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scale_factor
               
               
                k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
                                  
                est_skcsd = k.values(estimate='CSD',segments=True)/seglen[:,None]
                dir_name = fname_base+"_random_R_"+str(R_init)+'_lambda_'+str(lambd)+'_src_'+str(n_src)

                if sys.version_info >= (3, 0):
                    new_path = os.path.join(datd,"preprocessed_data/Python_3", dir_name)
                else:
                    new_path = os.path.join(datd,"preprocessed_data/Python_2",dir_name)
                
                if not os.path.exists(new_path):
                    print("Creating",new_path)
                    os.makedirs(new_path)
        
                utils.save_sim(new_path,k)
                simulation_paths.append(new_path)
                skcsd_random.append(est_skcsd)
            skcsd_maps = merge_maps(skcsd_random,tstart=atstart,tstop=atstop,merge=2)
            plot(ax,2,skcsd_maps,xticks=xticks,title="Random")
    
            fig.savefig(fname_base+'_'+str(R)+'_'+str(lambd)+'.png', bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
