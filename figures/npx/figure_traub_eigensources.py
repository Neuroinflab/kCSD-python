#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mbejtka
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import os
from kcsd import KCSD2D
from numpy.linalg import LinAlgError
from traub_data_kcsd_figure import (prepare_electrodes, prepare_pots,
                                    set_axis)


def do_kcsd_evd(pot, ele_pos, xmin, xmax, ymin, ymax, n_src_init=1000,
                R_init=30):
    k = KCSD2D(ele_pos, pot,
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, h=1, sigma=1,
               n_src_init=n_src_init, gdx=4, gdy=4, R_init=R_init)
    try:
        eigenvalue, eigenvector = np.linalg.eigh(k.k_pot +
                                                 k.lambd *
                                                 np.identity(k.k_pot.shape[0]))
    except LinAlgError:
            raise LinAlgError('EVD is failing - try moving the electrodes'
                              'slightly')
    idx = eigenvalue.argsort()[::-1]
    eigenvalues = eigenvalue[idx]
    eigenvectors = eigenvector[:, idx]
    return k, eigenvalues, eigenvectors


def plot_eigensources(k, v, start=0, stop=6):
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    fig = plt.figure(figsize=(15, 16))
    for idx, i in enumerate(list(range(start, stop))):
        ax = plt.subplot(231+idx, aspect='equal')
        a = v[:,  i].reshape(k.estm_x.shape)
        max_a = np.max(np.abs(a))
        levels = np.linspace(-max_a, max_a, 200)
        ax.contourf(k.estm_x, k.estm_y, a[:, :], levels=levels, cmap=plt.cm.bwr)
        ax.set_xlabel('X ($\mu$m)')
        ax.set_ylabel('Y ($\mu$m)')
        ax.set_title(r"$\tilde{K} \cdot{v_{{%(i)d}}}$" % {'i': i+1}, fontsize=20)
        set_axis(ax, letter=letters[idx])
    fig.savefig(os.path.join('Eigensources_' + str(start) + '_' + str(stop)
                             + '.png'), dpi=300)


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
    k, eigenvalues, eigenvectors = do_kcsd_evd(pot_np, elec_pos_list[1][:, :2],
                                                -400, 400, -2000, 500,
                                                R_init=30, n_src_init=5000)
    v = np.dot(k.k_interp_cross, eigenvectors)
    plot_eigensources(k, v, start=0, stop=6)
