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

def plot(ax,i,what):
    xmax, xmin = fun.get_min_max(what)
    im = ax[i].imshow(what,extent=extent,origin='lower',aspect='auto',cmap='seismic_r',vmin=xmin,vmax=xmax)
    return im

if __name__ == '__main__':
    fname_base = "y_shaped_symmetric_different_electrode_positions"
    
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R_inits = [(2**i)/np.sqrt(2) for i in [2,3,4,5,6,7,8]]
    electrode_number = [8,16,32,64]
    data_dir_grid = []
    data_dir_random = []
    colnb = 4
    rows = [2,4,8,16]
    lfps_grid = []
    lfps_random = []
    extent = [-200, 200, -200, 600]

    for i, rownb in enumerate(rows):
        fname = fname_base+str(rownb)
        c = run_LFP.CellModel(morphology=2,cell_name=fname,colnb=colnb,rownb=rownb,xmin=-200,xmax=200,zmin=-200,zmax=600,tstop=tstop,seed=1988,weight=0.01,n_syn=100)
        c.simulate('symmetric')
        c.save_skCSD_python()
        c.save_memb_curr()
        c.save_seg_length()
        lfps_grid.append(c.simulation_parameters['electrode'].LFP)
        data_dir_grid.append(c.return_paths_skCSD_python())
        
    for i, e_no in enumerate(electrode_number):
        fname = fname_base+str(rownb)
        c = run_LFP.CellModel(morphology=2,cell_name=fname,electrode_distribution=2,xmin=-200,xmax=200,zmin=-200,zmax=600,tstop=tstop,seed=1988,weight=0.01,n_syn=100)
        c.simulate('symmetric')
        c.save_skCSD_python()
        c.save_memb_curr()
        c.save_seg_length()
        lfps_random.append(c.simulation_parameters['electrode'].LFP)
        data_dir_random.append(c.return_paths_skCSD_python())
        
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))#/scale_factor
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]
    ground_truth_grid = []
    ground_truth_t1 = None
    ground_truth_t2 = None
    t1 = 364
    t2 = 44
    
    for lambd in [1e-5,1e-4,1e-3,1e-2,1e-1]:
        for R_init in R_inits:
            simulation_paths = []
            data_paths = []
            fig = plt.figure()
            ax = []
            for j in range(12):
                ax.append(fig.add_subplot(3,4,j+1))
                
            for i, datd in enumerate(data_dir):
                data = ld.Data(datd)
                ele_pos = data.ele_pos/scale_factor
                pots = data.LFP/scale_factor_LFP
                morphology = data.morphology
                morphology[:,2:6] = morphology[:,2:6]/scale_factor
                R = R_init/scale_factor
               
                k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
                                  
                est_skcsd = k.values(estimate='CSD',segments=True)
                dir_name = fname_base+"_R_"+str(R_init)+'_lambda_'+str(lambd)+'_src_'+str(n_src)

                if sys.version_info >= (3, 0):
                    new_path = os.path.join(datd,"preprocessed_data/Python_3", dir_name)
                else:
                    new_path = os.path.join(datd,"preprocessed_data/Python_2",dir_name)
                
                if not os.path.exists(new_path):
                    print("Creating",new_path)
                    os.makedirs(new_path)
        
                utils.save_sim(new_path,k)
                simulation_paths.append(new_path)

                if i == 0:
                    cax = plot(ax,6,ground_truth_t1)
                    cax = plot(ax,10,ground_truth_t2)
                    cax = plot(ax,7,est_skcsd_t1)
                    cax = plot(ax,11,est_skcsd_t2)
                else:
                    cax = plot(ax,0,est_kcsd_pot[:,:,:,t1].sum(axis=1))
                    cax = plot(ax,1,est_kcsd[:,:,:,t1].sum(axis=1))
                    cax = plot(ax,2, ground_truth_t1)
                    cax = plot(ax,3,est_skcsd_t1)
                    cax = plot(ax,4,est_kcsd_pot[:,:,:,t1].sum(axis=1))
                    cax = plot(ax,5,est_kcsd[:,:,:,t1].sum(axis=1))
                    cax = plot(ax,8,est_kcsd_pot[:,:,:,t2].sum(axis=1))
                    cax = plot(ax,9,est_kcsd[:,:,:,t2].sum(axis=1))
            fig.savefig(dir_name+'.png', bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
