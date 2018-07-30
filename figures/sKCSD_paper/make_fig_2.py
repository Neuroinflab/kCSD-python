from __future__ import division, print_function
import sys
import os
from distutils.spawn import find_executable, spawn
import shutil
import subprocess
if find_executable('nrnivmodl') is not None:
    for path in ['x86_64', 'i686', 'powerpc']:
        if os.path.isdir(path):
            shutil.rmtree(path)
    spawn([find_executable('nrnivmodl')])
    subprocess.call(["nrnivmodl", "sinsyn.mod"])
else:
    print("""nrnivmodl script not found in PATH, thus NEURON .mod files could
    not be compiled, and LFPy.test() functions will fail""")
import numpy as np
import matplotlib.pyplot as plt
from kcsd import sKCSD
import kcsd.utility_functions as utils
import sKCSD_utils
from kcsd.validation import plotting_functions as pl

n_src = 512
lambd = 1
R = 8e-6/2**.5
if __name__ == '__main__':
    fname_base = "Figure_2"
    fig_name = sKCSD_utils.make_fig_names(fname_base)
    tstop = 850
    scaling_factor = 1000**2
    scaling_factor_LFP = 1000
    electrode_number = [8, 16, 128]
    data_dir = []
    xmin, xmax = -100, 600
    ymin, ymax = 0, 200
    orientation = 1
    
    for rownb in electrode_number:
        fname = '%s_rows_%d' % (fname_base, rownb)
        if sys.version_info < (3, 0):
            c = sKCSD_utils.simulate(fname,
                                     morphology=1,
                                     simulate_what="sine",
                                     colnb=1,
                                     rownb=rownb,
                                     xmin=xmin,
                                     xmax=xmax,
                                     ymin=ymin,
                                     ymax=ymax,
                                     tstop=tstop,
                                     seed=1988,
                                     weight=0.1,
                                     n_syn=100)
            data_dir.append(c.return_paths_skCSD_python())
        else:
            new_dir = os.path.join('simulation', fname)
            data_dir.append(new_dir)
            
    seglen = np.loadtxt(os.path.join(data_dir[0], 'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir[0], 'membcurr'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    gvmin, gvmax = pl.get_min_max(ground_truth)
    fname = fname_base + '.png'
    fig_name = sKCSD_utils.make_fig_names(fname)
    data_paths = []
    fig, ax = plt.subplots(4, 1)
    xticklabels = list(np.linspace(0, 800, 5))
    yticklabels = list(np.linspace(0, 52, 5))
    pl.make_map_plot(ax[0], ground_truth)
    for i, datd in enumerate(data_dir):
        l = 0
        data = utils.LoadData(datd)
        ele_pos = data.ele_pos/scaling_factor
        data.LFP = data.LFP/scaling_factor_LFP
        morphology = data.morphology
        morphology[:, 2:6] = morphology[:, 2:6]/scaling_factor
        sKCSD.skmonaco_available = False
        k = sKCSD(ele_pos,
                  data.LFP,
                  morphology,
                  n_src_init=n_src,
                  src_type='gauss',
                  lambd=lambd,
                  R_init=R,
                  skmonaco_available=False)
        est_csd = k.values(transformation='segments')/seglen[:, None]
        if i == 2:
            pl.make_map_plot(ax[i+1],
                    est_csd,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    xlabel='Time [ms]',
                    ylabel='#segment')
        else:
            pl.make_map_plot(ax[i+1], est_csd)

        if sys.version_info < (3, 0):
            path = os.path.join(datd, "preprocessed_data/Python_2")
        else:
            path = os.path.join(datd, "preprocessed_data/Python_3")

        if not os.path.exists(path):
            print("Creating", path)
            os.makedirs(path)
        utils.save_sim(path, k)
    fig.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
