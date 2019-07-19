#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from kcsd import sKCSD, sKCSDcell
import kcsd.sKCSD_utils as utils
import kcsd.validation.plotting_functions as pl
sys.path.insert(1, os.path.join(sys.path[0], '../sKCSD_paper'))
import sKCSD_utils
import matplotlib.gridspec as gridspec


n_src = 2048
R = 32e-6
lambd = 0.01
fname = "Figure_complex"
scaling_factor = 1000**2
scaling_factor_LFP = 1000

def add_figure_labels(ax, ax_somav):
    limx = ax_somav.get_xlim()
    limy = ax_somav.get_ylim()
    ax_somav.text(limx[0]-(limx[-1] - limx[0])/20, limy[-1]+(limy[-1]-limy[0])/20, 'A', fontsize=16)
    titles = ['B', 'C', 'D', 'E']
    for i, x in enumerate(ax):
        limx = x.get_xlim()
        limy = x.get_ylim()
        x.text(limx[0]-(limx[-1] - limx[0])/20,
               limy[-1]+(limy[-1]-limy[0])/20,
               titles[i], fontsize=16)
        
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
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    ax_somav = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax = []
    for i in range(1,3):
        for j in range(2):
            ax.append(plt.subplot(gs[i, j]))

    return fig,  ax_somav, ax

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

def draw_morpho(ax, morpho, extent, electrode_positions, title=False):
    if title is not False:
        ax.set_title('Cell morphology')
    ax.set_ylabel(u'x (μm)')
    ax.set_xlabel(u'y (μm)')
    ax.imshow(morpho,
              origin="lower",
              interpolation="spline36",
              aspect="auto",
              extent=extent)
    
    for epos in electrode_positions:
        pos_x, pos_y = 1e6*epos[0], 1e6*epos[1]
        text = ax.text(pos_x, pos_y, '*',
                       ha="center", va="center", color="k",
                       fontsize=4)

def draw_somav(ax, t, V):
    ax.plot(t, V)
    ax.set_title('Voltage in the soma')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Vm (mV)')
    toplot = np.argmax(somav)
    lim = ax.get_ylim()
    ys = np.linspace(lim[0], lim[-1], len(t))
    xs = t[toplot]*np.ones_like(t)
    ax.plot(xs, ys, 'r')
    ax.text(t[toplot]+10, 0, 't = %1.1f ms'%(t[toplot]),
            ha="center", va="center", color="k",
            fontsize=10)

def draw_loops(ax_loops, skcsd, gvmin, gvmax):
    ax_loops.imshow(skcsd,
                    origin="lower",
                    interpolation="spline36",
                    extent=[0, time[-1], 1, skcsd.shape[0]],
                    aspect='auto',
                    cmap='seismic_r',
                    vmax=gvmax,
                    vmin=gvmin)
    ax_loops.set_title('CSD in morphology loops')

def draw_ground_truth_skcsd_segments(ax1, gtruth, skcsd, time, gvmin, gvmax):
    cax = ax1[0].imshow(gtruth,
                        extent=[0, time[-1], 1, ground_truth.shape[0]],
                        origin='lower',
                        aspect='auto',
                        cmap='seismic_r',
                        vmax=gvmax,
                        vmin=gvmin,
                        alpha=0.5)
    cax = ax1[1].imshow(skcsd,
                        extent=[0, time[-1], 1, skcsd.shape[0]],
                        origin='lower',
                        aspect='auto',
                        cmap='seismic_r',
                        vmax=gvmax,
                        vmin=gvmin,
                        alpha=0.5)
    ax[0].set_title('Ground truth in segments')
    ax[1].set_title('CSD in segments')
    ax[1].set_yticklabels([])
    ax[0].set_xlabel('time (s)')
    ax[1].set_xlabel('time (s)')
    ax[0].set_ylabel('# segment')

def draw_ground_truth_skcsd_3D(ax1, gtruth, skcsd, data,
                                gvmin, gvmax, t_plot):
    ax1[2].set_title('Ground truth at t = %1.1f ms' %t_plot)
    ax1[3].set_title('CSD at t = %1.1f ms' %t_plot)
    for i in [2, 3]:
        ax1[i].set_ylabel('x (um)')
        ax1[i].set_xlabel('y (um)')
        draw_morpho(ax1[i], morphology, extent, data.ele_pos)
   
    pl.make_map_plot(ax1[2],
                     gtruth.sum(axis=(2, 3)), 
                     extent=extent,
                     cmap='seismic_r',
                     vmax=gvmax,
                     vmin=gvmin,
                     alpha=0.5,
                     circles=True)
    pl.make_map_plot(ax1[3],
                     skcsd.sum(axis=(2, 3)),
                     extent=extent,
                     cmap='seismic_r',
                     vmax=gvmax,
                     vmin=gvmin, alpha=0.5)
    
if __name__ == '__main__':

    fname_base = "Figure_complex"
    data_dir = simulate()
    fig,  ax_somav, ax = make_figure()
    ground_truth, data, time, somav = read_in_data(data_dir)
    toplot = np.argmax(somav)
    gvmax, gvmin = pl.get_min_max(ground_truth[:, toplot])        
    new_fname = fname + '.png'
    fig_name = sKCSD_utils.make_fig_names(new_fname)
    cell_itself = make_larger_cell(data, n_src)
    morphology, extent = cell_itself.draw_cell2D()
    extent = [ex*1e6 for ex in extent]
    draw_somav(ax_somav, time, somav)
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

    skcsd, pot, morphology_file, ele_pos, n_src  = utils.load_sim(path)
    cell_obj = sKCSDcell(morphology_file, ele_pos, n_src)
    csd = cell_obj.transform_to_segments(skcsd)
    draw_ground_truth_skcsd_segments(ax, ground_truth, csd, time, gvmin, gvmax)
    csd_3D = cell_itself.transform_to_3D(skcsd[:, toplot])
    gt_3D = cell_itself.transform_to_3D(ground_truth[:, toplot],
                                        what="morpho")
    draw_ground_truth_skcsd_3D(ax, gt_3D, csd_3D, data,
                                gvmin, gvmax, time[toplot])
    
   

    fig.subplots_adjust(wspace=0.5)
    fig.subplots_adjust(hspace=0.5)
    add_figure_labels(ax, ax_somav)
    plt.savefig(fig_name,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.05,
                dpi=600)
    plt.savefig(fig_name[:-4]+'.svg',
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.05,
                dpi=600)
    plt.show()
