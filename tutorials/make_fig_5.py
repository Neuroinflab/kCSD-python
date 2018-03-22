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
R = 32e-6/2**.5
lambd = 1e-5
if __name__ == '__main__':
    fname_base = "Figure_5"
    fig_name = fun.make_fig_names(fname_base)
    tstop = 70
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R_inits = [(2**i)/np.sqrt(2) for i in [2,3,4,5,6,7,8]]
    electrode_number = [1,4]
    data_dir = []
    colnb = 16
    lfps = []
    extent = [-200, 200, -200, 600]
    xmin = [50,-200]
    for i, rownb in enumerate(electrode_number):
        fname = fname_base+str(rownb)
        c = run_LFP.CellModel(morphology=2,cell_name=fname,colnb=colnb,rownb=rownb,xmin=-200,xmax=600,ymin=xmin[i],ymax=200,tstop=tstop,seed=1988,weight=0.01,n_syn=100)
        c.simulate('symmetric')
        c.save_skCSD_python()
        c.save_memb_curr()
        c.save_seg_length()
        lfps.append(c.simulation_parameters['electrode'].LFP)
        data_dir.append(c.return_paths_skCSD_python())
        
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))#/scale_factor
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]
    ground_truth_grid = []
    ground_truth_t1 = None
    ground_truth_t2 = None
    t1 = 405
    t2 = 81
 
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
        
        k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
        xmin = -200/scale_factor
        xmax = 200/scale_factor
        ymin = -100/scale_factor
        ymax = 200/scale_factor
        zmin = -200/scale_factor
        zmax = 600/scale_factor
        gdx = (xmax-xmin)/100
        gdy = (ymax-ymin)/2
        gdz = (zmax-zmin)/200
        
        kcsd = KCSD.KCSD3D(ele_pos,data.LFP,n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R,dist_table_density=100,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zmin=zmin,zmax=zmax,gdx=gdx,gdy=gdy,gdz=gdz)
                
        if not len(ground_truth_grid):
            ground_truth_grid = k.cell.transform_to_3D(ground_truth,what="morpho")
            ground_truth_t1 = ground_truth_grid[:,:,:,t1].sum(axis=1).T
            ground_truth_t2 = ground_truth_grid[:,:,:,t2].sum(axis=1).T
                    
        est_skcsd = k.values(estimate='CSD')
        est_skcsd_t1 = est_skcsd[:,:,:,t1].sum(axis=1).T
        est_skcsd_t2 = est_skcsd[:,:,:,t2].sum(axis=1).T
        est_kcsd = kcsd.values(estimate='CSD')
        est_kcsd_pot = kcsd.values(estimate='POT')

        if i == 0:
            cax = fun.plot(ax[6],ground_truth_t1)
            cax = fun.plot(ax[10],ground_truth_t2)
            cax = fun.plot(ax[7],est_skcsd_t1)
            cax = fun.plot(ax[11],est_skcsd_t2)
        else:
            cax = fun.plot(ax[0],est_kcsd_pot[:,:,:,t1].sum(axis=1))
            cax = fun.plot(ax[1],est_kcsd[:,:,:,t1].sum(axis=1))
            cax = fun.plot(ax[2], ground_truth_t1)
            cax = fun.plot(ax[3],est_skcsd_t1)
            cax = fun.plot(ax[4],est_kcsd_pot[:,:,:,t1].sum(axis=1))
            cax = fun.plot(ax[5],est_kcsd[:,:,:,t1].sum(axis=1))
            cax = fun.plot(ax[8],est_kcsd_pot[:,:,:,t2].sum(axis=1),extent=[-200,200,-200,600])
            cax = fun.plot(ax[9],est_kcsd[:,:,:,t2].sum(axis=1))
    fig.savefig(fig_name+'.png', bbox_inches='tight', transparent=True, pad_inches=0.1)

    
           
