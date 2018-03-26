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

if __name__ == '__main__':
    
    fname_base = "Figure_8"
    fig_dir = 'Figures'
    fig_name = fun.make_fig_names(fname_base)
    tstop = 850
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    
    R_inits = np.array([(2**i)/scale_factor/2**0.5 for i in range(3,8)])
    lambdas = np.array([(10**(-i))for i in range(5)])
    #x_ticklabels = [2**i for i in range(1,7)]
    #y_ticklabels = [str(lambd) for lambd in lambdas]

    colnb = 4
    rownb = 4
    lfp_dir,data_dir = fun.simulate(fname_base,morphology=6,tstop=tstop,seed=1988,weight=0.04,n_syn=100,simulate_what='oscillatory',electrode_distribution=4)
    data = ld.Data(data_dir)
    ele_pos = data.ele_pos/scale_factor
    pots = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:,2:6] = morphology[:,2:6]/scale_factor
    
    ground_truth = np.loadtxt(os.path.join(data_dir,'membcurr'))
    ground_truth = ground_truth
    
   

    for i,R in enumerate(R_inits):
        for j,lambd in enumerate(lambdas):
            ker = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
            # f not i and not j:
               
            #    ground_truth_3D = ker.cell.transform_to_3D(ground_truth,what="morpho")
            #    vmax, vmin = fun.get_min_max(ground_truth_3D)
            ker_dir = data_dir+'_R_%f_lambda_%f'%(R,lambd)
            if sys.version_info < (3,0):
                path = os.path.join(ker_dir, "preprocessed_data/Python_2")
            else:
                path = os.path.join(ker_dir, "preprocessed_data/Python_3")

            if not os.path.exists(path):
                print("Creating",path)
                os.makedirs(path)

            morpho,extent = ker.cell.draw_cell2D(axis=2)
            est_skcsd = ker.values(estimate='CSD')
            fig, ax = plt.subplots(1,2)
            utils.save_sim(path,ker)
            fun.plot(ax[1],morpho,extent=extent)
            fun.plot(ax[1],est_skcsd[:,:,:,550].sum(axis=(2)),extent=extent)
            plt.show()
            
    
