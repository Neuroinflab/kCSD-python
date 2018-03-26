from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from distutils.spawn import find_executable, spawn
import shutil
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD3D
import corelib.utility_functions as utils
import corelib.loadData as ld
import functions as fun

n_src = 512
lambd = 1e-5
R = 64e-6/2**.5

if find_executable('nrnivmodl') is not None:
    for path in ['x86_64', 'i686', 'powerpc']:
        if os.path.isdir(path):
            shutil.rmtree(path)
    spawn([find_executable('nrnivmodl')])
    subprocess.call(["nrnivmodl", "sinsyn.mod"])
else:
    print("nrnivmodl script not found in PATH, thus NEURON .mod files could" +
"not be compiled, and LFPy.test() functions will fail")
    
if __name__ == '__main__':
    fname_base = "Figure_2.png"
    fig_name = fun.make_fig_names(fname_base)
    tstop = 850
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    electrode_number = [8,16,128]
    data_dir = []
    xmin, xmax = -100, 600
    ymin, ymax = 0, 200
    orientation = 1
    for rownb in electrode_number:
        fname = "Figure_1"
        c = fun.simulate(fname,morphology=1,simulate_what="sine",colnb=1,rownb=rownb,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,tstop=tstop,seed=1988,weight=0.1,n_syn=100)
        
        data_dir.append(c.return_paths_skCSD_python())
        
    seglen = np.loadtxt(os.path.join(data_dir[0],'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0],'membcurr'))
    ground_truth = ground_truth/seglen[:,None]
    gvmin, gvmax = fun.get_min_max(ground_truth)
    
    data_paths = []
    fig, ax = plt.subplots(4,1)
    xticklabels = list(np.linspace(0,800,5))
    yticklabels = list(np.linspace(0,52,5))
    fun.plot(ax[0],ground_truth)
    for i, datd in enumerate(data_dir):
        l = 0
        data = ld.Data(datd)
        ele_pos = data.ele_pos/scaling_factor
        pots = data.LFP/scaling_factor_LFP
        morphology = data.morphology
        morphology[:,2:6] = morphology[:,2:6]/scaling_factor
        k = sKCSD3D.sKCSD3D(ele_pos,data.LFP,morphology, n_src_init=n_src, src_type='gauss',lambd=lambd,R_init=R)
        est_csd = k.values(segments=True)/seglen[:,None]
        if i == 2:
            fun.plot(ax[i+1],est_csd,xticklabels=xticklabels,yticklabels=yticklabels)
        else:
            fun.plot(ax[i+1],est_csd)
        fig.savefig(fig_name, bbox_inches='tight', transparent=True, pad_inches=0.1)
