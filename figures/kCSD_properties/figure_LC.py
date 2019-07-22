# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:39:53 2019

@author: Wladek
"""
import scipy.stats as st
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from figure_properties import *

def set_axis(ax, x, y, letter=None):
    ax.text(x,
            y,
            letter,
            fontsize=15,
            weight='bold',
            transform=ax.transAxes)
    return ax

def make_plots(title, m_norm, m_resi, true_csd, curveseq, ele_y,
               pots_n, pots, k_csd_x, est_pot, est_csd, noreg_csd, save_as):
    """
    Shows 4 plots
    1_ LFP measured (with added noise) and estimated LFP with kCSD method
    2_ true CSD and reconstructed CSD with kCSD
    3_ L-curve of the model
    4_ Surface of parameters R and Lambda with scores for optimal paramater selection with L-curve or cross-validation
    """
    #True CSD
    lambdas = np.logspace(-7, -3, 50)
    fig = plt.figure(figsize=(12, 12), dpi=300)
    widths = [1, 1]
    heights = [1, 1]
    xpad= 5
    ypad = 10
    gs = gridspec.GridSpec(2, 2, height_ratios=heights, width_ratios=widths,
                           hspace=0.6, wspace=0.6)
    xrange = np.linspace(0, 1, len(true_csd))
    ax1 = plt.subplot(gs[0])
    ax1.plot(ele_y, pots_n*1e3, 'r', marker='o', linewidth=0, label='Measured potential with noise')
    ax1.plot(ele_y, pots*1e3, 'b', marker='o', linewidth=0, label='Measured potential')
    ax1.scatter(ele_y, np.zeros(len(ele_y))-11, 8, color='black', 
                clip_on=False, label = "Electrode position")
    ax1.plot(xrange, est_pot*1e3, label='Reconstructed potential', color='blue')
    plt.ylim(-11,11)
    ax1.set_ylabel('Potential ($mV$)', labelpad = ypad+2)
    ax1.set_xlabel('Distance ($mm$)', labelpad = xpad)
    ax1.tick_params(axis='both', which='major')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    set_axis(ax1, -0.05, 1.05, letter='A')
    ax1.legend(bbox_to_anchor=(1.7, -0.2), ncol=2, frameon=False)

    ax_L = plt.subplot(gs[1])
    m_resi = np.log(m_resi)
    m_norm = np.log(m_norm)
    imax = np.argmax(curveseq[np.argmax(np.max(curveseq, axis=-1))])
    imax2 = np.argmax(np.max(curveseq, axis=-1))
    plt.ylabel("Log norm of the model", labelpad = ypad)
    plt.xlabel("Log norm of the prediction error", labelpad = xpad)
    ax_L.plot(m_resi, m_norm, marker=".", c="green", label = 'L-Curve')
    ax_L.plot([m_resi[imax]], [m_norm[imax]], marker="o", c="red")
    x = [m_resi[0], m_resi[imax], m_resi[-1]]
    y = [m_norm[0], m_norm[imax], m_norm[-1]]
    ax_L.fill(x, y, alpha=0.2)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax_L.spines['right'].set_visible(False)
    ax_L.spines['top'].set_visible(False)
    set_axis(ax_L, -0.05, 1.05, letter='B')
    ax_L.legend(bbox_to_anchor=(0.7, -0.2), ncol=1, frameon=False)

    ax2 = plt.subplot(gs[2])
    plt.plot(xrange, true_csd, label='True CSD', color='red', linestyle = '--')
    plt.plot(xrange, est_csd, label='kCSD + regularization', color='blue')
    plt.plot(xrange, noreg_csd, label='kCSD', color='darkgreen', alpha = 0.6)
    plt.scatter(ele_y, np.zeros(len(ele_y))-1, 8, color='black', 
                label = "Electrode position", clip_on=False)
    ax2.set_ylabel('CSD ($mA/mm$)', labelpad = ypad)
    ax2.set_xlabel('Distance ($mm$)', labelpad = xpad)
    plt.ylim(-1, 1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(bbox_to_anchor=(-.25, -0.27), ncol=2, frameon=False, loc = 'center left')
    set_axis(ax2, -0.05, 1.05, letter='C')

    ax4 = plt.subplot(gs[3])
    ax4.plot(lambdas, curveseq[imax2], marker=".", label = 'Curvature evaluation')
    ax4.plot([lambdas[imax]], [curveseq[imax2][imax]], marker="o", c="red")
    ax4.set_ylabel('Curvature', labelpad = ypad)
    ax4.set_xlabel('$\lambda$', labelpad = xpad)
    ax4.set_xscale('log')
    ax4.tick_params(axis='both', which='major')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.legend(bbox_to_anchor=(1, -0.2), ncol=2, frameon=False)
    set_axis(ax4, -0.05, 1.05, letter='D')
    fig.savefig(save_as+'.png')

if __name__=='__main__':
#    os.chdir("./LCurve/LC2")
    noises = 3
    noise_lvl = np.linspace(0, 0.5, noises)
#    df = np.load('data_fig4_and_fig13_lc_noise25.0.npz')
    Rs = np.linspace(0.025, 8*0.025, 8)
    title = ['nazwa_pliku']
    save_as = 'noise'
#    make_plots(title, df['m_norm'], df['m_resi'], df['true_csd'], 
#               df['curve_surf'], df['ele_y'], df['pots_n'],
#               df['pots'], df['estm_x'], df['est_pot'], df['est_csd'], 
#               df['noreg_csd'], save_as)