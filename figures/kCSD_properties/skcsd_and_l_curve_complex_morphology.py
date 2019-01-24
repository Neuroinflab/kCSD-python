from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from kcsd import sKCSD, sKCSDcell
import kcsd.utility_functions as utils
import kcsd.validation.plotting_functions as pl
sys.path.insert(1, os.path.join(sys.path[0], '../sKCSD_paper'))
import sKCSD_utils
import matplotlib.gridspec as gridspec


n_src = 1024
R = 32e-6
lambd = 0.1
fname = "Figure_complex"
scaling_factor = 1000**2
scaling_factor_LFP = 1000


def make_larger_cell(data, n_sources=n_src):
    if data.ele_pos[:, 0].max() > data.morphology[:, 2].max():
        xmax = data.ele_pos[:, 0].max() + 50e-6
    else:
        xmax = data.morphology[:, 2].max() + 50e-6
    if data.ele_pos[:, 0].min() < data.morphology[:, 2].min():
        xmin = data.ele_pos[:, 0].min() - 50e-6
    else:
        xmin = data.morphology[:, 2].min() - 50e-6
    
    if data.ele_pos[:, 1].max() > data.morphology[:, 3].max():
        ymax = data.ele_pos[:, 1].max() + 50e-6
    else:
        ymax = data.morphology[:, 3].max() + 50e-6
    if data.ele_pos[:, 1].min() < data.morphology[:, 3].min():
        ymin = data.ele_pos[:, 1].min() - 50e-6
    else:
        ymin = data.morphology[:, 3].min() - 50e-6

    if data.ele_pos[:, 2].max() > data.morphology[:, 4].max():
        zmax = data.ele_pos[:, 2].max() + 50e-6
    else:
        zmax = data.morphology[:, 4].max() + 50e-6
    if data.ele_pos[:, 2].min() < data.morphology[:, 4].min():
        zmin = data.ele_pos[:, 2].min() - 50e-6
    else:
        zmin = data.morphology[:, 4].min() - 50e-6
   
    return sKCSDcell(data.morphology, data.ele_pos, n_sources, xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, ymin=ymin, ymax=ymax, tolerance=2e-6)

def make_figure():
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 5, figure=fig)
    ax_morpho = plt.subplot(gs[0,0])
    ax_loops = plt.subplot(gs[0,1])
    ax_somav = plt.subplot2grid((2, 5), (1, 0), rowspan=2)
    ax = []
    for i in range(2):
        for j in range(2, 4):
            ax.append(plt.subplot(gs[i, j]))
    return fig, ax_morpho, ax_loops, ax_somav, ax

def simulate():
    tstop = 75
    rownb = 10
    colnb = 10
    c = sKCSD_utils.simulate(fname_base,
                             morphology=7,
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
    return c.return_paths_skCSD_python()

def read_in_data(ddir):
    seglen = np.loadtxt(os.path.join(ddir, 'seglength'))
    n_seg = len(seglen)
    ground_truth = np.loadtxt(os.path.join(ddir, 'membcurr'))
    time = np.loadtxt(os.path.join(ddir, 'tvec.txt'))
    somav = np.loadtxt(os.path.join(ddir, 'somav.txt'))
    ground_truth = ground_truth/seglen[:, None]*1e-3
    Data = utils.LoadData(ddir)
    Data.ele_pos = Data.ele_pos/scaling_factor
    Data.LFP = Data.LFP/scaling_factor_LFP
    Data.morphology[:, 2:6] = Data.morphology[:, 2:6]/scaling_factor
    
    return ground_truth, Data, time, somav

if __name__ == '__main__':
    fname_base = "Figure_complex"
    data_dir = simulate()
    fig, ax_morpho, ax_loops, ax_somav, ax = make_figure()
    ground_truth, data, time, somav = read_in_data(data_dir)
    gvmax, gvmin = pl.get_min_max(ground_truth)        
    cax = ax[0].imshow(ground_truth,
                       extent=[0, time[-1], 1, ground_truth.shape[0]],
                       origin='lower',
                       aspect='auto',
                       cmap='seismic_r',
                       vmax=gvmax,
                       vmin=gvmin)
    ax_morpho.set_title('Cell morphology and morphology loop')
    ax_morpho.set_xlabel('')
    ax_morpho.set_ylabel('')
    ax_somav.plot(time, somav)
    ax_somav.set_title('Voltage in the soma')
    ax_somav.set_xlabel('time (ms)')
    ax_somav.set_ylabel('Vm (mV)')
                       
    
    new_fname = fname + '.png'
    fig_name = sKCSD_utils.make_fig_names(new_fname)
    cell_itself = make_larger_cell(data, n_src)
    morphology, extent = cell_itself.draw_cell2D()
    extent = [ex*1e6 for ex in extent]
    ax_morpho.imshow(morphology,
                     origin="lower",
                     interpolation="spline36",
                     aspect="auto",
                     extent=extent)
    # k = sKCSD(data.ele_pos,
    #           data.LFP,
    #           data.morphology,
    #           n_src_init=n_src,
    #           src_type='gauss',
    #           lambd=lambd,
    #           exact=True,
    #           R_init=R,
    #           sigma=0.3)
    path = os.path.join(data_dir, 'lambda_%f_R_%f_n_src_%d' % (lambd, R, n_src))
    if sys.version_info < (3, 0):
        path = os.path.join(path, "preprocessed_data/Python_2")
    else:
        path = os.path.join(path, "preprocessed_data/Python_3")
    if not os.path.exists(path):
        print("Creating", path)
        os.makedirs(path)
    try:
        utils.save_sim(path, k)
    except NameError:
        pass
    skcsd, pot, cell_obj = utils.load_sim(path)
    ax_loops.imshow(skcsd, origin="lower",
                    interpolation="spline36",
                    extent=[0, time[-1], 1, skcsd.shape[0]],
                    aspect='auto',
                    cmap='seismic_r',
                    vmax=gvmax,
                    vmin=gvmin)
    ax_loops.set_title('KCSD in morphology loop space')
    csd = cell_obj.transform_to_segments(skcsd)
 
    cax = ax[1].imshow(csd,
                       extent=[0, time[-1], 1, csd.shape[0]],
                       origin='lower',
                       aspect='auto',
                       cmap='seismic_r',
                       vmax=gvmax,
                       vmin=gvmin)
    ax[0].set_title('Ground truth')
    ax[1].set_title('sKCSD in segments')
    ax[1].set_xticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[3].set_xlabel('time (s)')
    ax[2].set_xlabel('time (s)')
    ax[0].set_ylabel('# segment')

    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
        
    plt.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1,
                dpi=600)
    plt.savefig(fig_name[:-4]+'.svg',
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1,
                dpi=600)
    plt.show()
