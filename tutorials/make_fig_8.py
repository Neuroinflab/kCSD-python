from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD, KCSD
import corelib.utility_functions as utils
import loadData as ld
import corelib.plotting_functions as pl
import functions as fun
import run_LFP
n_src = 512
sKCSD.skmonaco_available = False

if __name__ == '__main__':
    
    fname_base = "Figure_8"
    fig_dir = 'Figures'
    fig_name = fun.make_fig_names(fname_base)
    tstop = 850
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    
    R_inits = np.array([(2**(i-.5))/scale_factor for i in range(1,9)])
    lambdas = np.array([(10**(-i))for i in range(5)])
 
    colnb = 10
    rownb = 10
    c = fun.simulate(fname_base,morphology=6,tstop=tstop,seed=1988,weight=0.04,n_syn=1000,simulate_what='oscillatory',electrode_distribution=1,electrode_orientation=3,xmin=-400,xmax=400,ymin=-400,ymax=400,colnb=colnb,rownb=rownb,dt=0.5)
    data_dir = c.return_paths_skCSD_python()
    data = ld.Data(data_dir)
    ele_pos = data.ele_pos/scale_factor
    data.LFP = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:,2:6] = morphology[:,2:6]/scale_factor
    seglen = np.loadtxt(os.path.join(data_dir,'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir,'membcurr'))/seglen[:,None]*1e-3

    for i,R in enumerate(R_inits):
        
        for j,l in enumerate(lambdas):
            lambd = l*2*(2*np.pi)**3*R**2*n_src
            ker = sKCSD.sKCSD(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R,dist_table_density=250)
            if not i and not j:
               
               ground_truth_3D = ker.cell.transform_to_3D(ground_truth,what="morpho")
               print(ground_truth_3D.max(),ground_truth_3D.min())
               vmax, vmin = pl.get_min_max(ground_truth_3D)
            ker_dir = data_dir+'_R_%f_lambda_%f'%(R,lambd)
            c.new_path = ker_dir
            c.save_skCSD_python()

            morpho,extent = ker.cell.draw_cell2D(axis=2)
            est_skcsd = ker.values()
            fig, ax = plt.subplots(1,2)
            if sys.version_info < (3,0):
                path = os.path.join(ker_dir, "preprocessed_data/Python_2")
            else:
                path = os.path.join(ker_dir, "preprocessed_data/Python_3")

            if not os.path.exists(path):
                print("Creating",path)
                os.makedirs(path)
            utils.save_sim(path,ker)
            
            print(R,l,lambd,est_skcsd.max(),est_skcsd.min(),ground_truth.max(),ground_truth.min(),fun.L1_error(ground_truth_3D,est_skcsd))
  
            pl.plot(ax[1],morpho,extent=extent)
            pl.plot(ax[1],est_skcsd[:,:,:,t0].sum(axis=(2)),extent=extent,vmin=vmin,vmax=vmax)
            pl.plot(ax[0],morpho,extent=extent)
            pl.plot(ax[0],ground_truth_3D[:,:,:,t0].sum(axis=(2)),extent=extent)
            
    plt.show()
            
    
