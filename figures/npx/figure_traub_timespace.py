#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mbejtka
"""
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import os
from traub_data_kcsd_figure import (prepare_electrodes, prepare_pots, do_kcsd,
                                    set_axis)


def make_plot_spacetime(ax, val, cut=9, title='True CSD',
                        cmap=plt.cm.bwr, letter='A'):
    yy = np.linspace(-3500, 500, val.shape[1])
    xx = np.linspace(-50, 250, val[cut, :, :].shape[1])
    max_val = np.max(np.abs(val[cut, :, :]))
    levels = np.linspace(-max_val, max_val, 200)
    im = ax.contourf(xx, yy, val[cut, :, :], levels=levels, cmap=cmap)
    if 'CSD' in title:
        name = ['', 'II/III', 'IV', 'V', 'VI']
        layer_level = [0, -400, -700, -1200, -1700]
        for i, value in enumerate(layer_level):
            plt.axhline(y=value, xmin=xx.min(), xmax=xx.max(), linewidth=1,
                        color='k', ls='--')
            plt.text(110, value+165, name[i], fontsize=10, va='top', ha='center')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title, fontsize=20, pad=30)
    ticks = np.linspace(-max_val, max_val, 3, endpoint=True)
    plt.colorbar(im, orientation='vertical', format='%.3f', ticks=ticks)
    set_axis(ax, letter=letter)
    plt.tight_layout()


def make_figure_spacetime(val_pots, val_csd, cut=13, titl1='POT', titl2='CSD',
                          fig_title='Pot and CSD in time'):
    fig = plt.figure(figsize=(12, 8))
    ax2 = plt.subplot(121)
    make_plot_spacetime(ax2, val_pots, cut=cut,
              title=titl1, cmap=plt.cm.PRGn, letter='A')
    ax1 = plt.subplot(122)
    make_plot_spacetime(ax1, val_csd, cut=cut, 
              title=titl2, cmap=plt.cm.bwr, letter='B')
    fig.savefig(os.path.join(fig_title + '.png'), dpi=300)


if __name__ == '__main__':
    time_pt_interest = 3000
    time_pts = 6000  # number of all time frames
    num_cmpts = [74, 74, 59, 59, 59, 59, 61, 61, 50, 59, 59, 59]
    cell_range = [0, 1000, 1050, 1140, 1230, 1320,
                  1560, 2360, 2560, 3060, 3160, 3260, 3360]
    num_cells = np.diff(cell_range) / 10  # 10% MODEL
    total_cmpts = list(num_cmpts * num_cells)
    pop_names = ['pyrRS23', 'pyrFRB23', 'bask23', 'axax23', 'LTS23',
                 'spinstel4', 'tuftIB5', 'tuftRS5', 'nontuftRS6',
                 'bask56', 'axax56', 'LTS56']

    h = h5.File('pulsestimulus10model.h5', 'r')
    elec_pos_list, names_list = prepare_electrodes()

    pot_np = prepare_pots(elec_pos_list[1], names_list[1], h, pop_names, time_pts)
    kcsd, est_pot, x, y = do_kcsd(pot_np, elec_pos_list[1][:, :2], -40, 40, -3500, 500)
    
    time_pts_ds = int(time_pts/4)
    cut = 9
    start_pt = 625  # 50 ms before the stimulus
    end_pt = 1375  # 250 ms after the stimulus
    # for cut in range(kcsd.shape[0]):
    make_figure_spacetime(est_pot[:, :, start_pt:end_pt],
                          kcsd[:, :, start_pt:end_pt], cut=cut,
                          titl1='Estimated LFP', titl2='Estimated CSD',
                          fig_title=('Estimated POT and CSD in time 1stim '+ str(cut)))
