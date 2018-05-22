from __future__ import division, print_function
import run_LFP
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from kcsd import sKCSD
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
import sKCSD_utils
import run_LFP
n_src = 512
if __name__ == '__main__':
    fname_base = "Figure_8"
    fig_dir = 'Figures'
    fig_name = sKCSD_utils.make_fig_names(fname_base)
    tstop = 850
    scale_factor = 1000**2
    scale_factor_LFP = 1000
    R_inits = np.array([(2**(i - .5))/scale_factor for i in range(3, 9)])
    lambdas = np.array([(10**(-i))for i in range(5)])
    colnb = 10
    rownb = 10
    c = sKCSD_utils.simulate(fname_base,
                             morphology=6,
                             tstop=tstop,
                             seed=1988,
                             weight=0.04,
                             n_syn=1000,
                             simulate_what='oscillatory',
                             electrode_distribution=1,
                             electrode_orientation=3,
                             xmin=-400,
                             xmax=400,
                             ymin=-400,
                             ymax=400,
                             colnb=colnb,
                             rownb=rownb,
                             dt=0.5)
    data_dir = c.return_paths_skCSD_python()
    data = utils.LoadData(data_dir)
    ele_pos = data.ele_pos/scale_factor
    data.LFP = data.LFP/scale_factor_LFP
    morphology = data.morphology
    morphology[:, 2:6] = morphology[:, 2:6]/scale_factor
    seglen = np.loadtxt(os.path.join(data_dir,
                                     'seglength'))
    ground_truth = np.loadtxt(os.path.join(data_dir,
                                           'membcurr'))/seglen[:, None]*1e-3
    dt = c.cell_parameters['dt']
    t0 = 500/dt
    for i, R in enumerate(R_inits):
        for j, l in enumerate(lambdas):
            lambd = l*2*(2*np.pi)**3*R**2*n_src
            ker = sKCSD(ele_pos,
                        data.LFP,
                        morphology,
                        n_src_init=n_src,
                        src_type='gauss',
                        lambd=lambd,
                        R_init=R,
                        dist_table_density=250,
                        skmonaco_available=False)
            if not i and not j:
                ground_truth_3D = ker.cell.transform_to_3D(ground_truth,
                                                          what="morpho")
                vmax, vmin = pl.get_min_max(ground_truth_3D)
            ker_dir = data_dir + '_R_%f_lambda_%f' % (R, lambd)
            c.new_path = ker_dir
            c.save_skCSD_python()
            morpho, extent = ker.cell.draw_cell2D(axis=2)
            est_skcsd = ker.values()
            fig, ax = plt.subplots(1, 2)
            if sys.version_info < (3, 0):
                path = os.path.join(ker_dir, "preprocessed_data/Python_2")
            else:
                path = os.path.join(ker_dir, "preprocessed_data/Python_3")
            if not os.path.exists(path):
                print("Creating", path)
                os.makedirs(path)
            utils.save_sim(path, ker)
            pl.plot(ax[1],
                    morpho,
                    extent=extent)
            pl.plot(ax[1],
                    est_skcsd[:, :, :, t0].sum(axis=(2)),
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax)
            pl.plot(ax[0],
                    morpho,
                    extent=extent)
            pl.plot(ax[0],
                    ground_truth_3D[:, :, :, t0].sum(axis=(2)),
                    extent=extent)
    plt.show()
