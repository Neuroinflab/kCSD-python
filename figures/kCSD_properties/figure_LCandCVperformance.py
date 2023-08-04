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

def make_plot_perf(sim_results):
    rms_lc = sim_results[0, 2]
    lam_lc = sim_results[0, 0]
    rms_cv = sim_results[1, 2]
    lam_cv = sim_results[1, 0]
    fig = plt.figure(figsize = (12,7), dpi = 300)
    widths = [1, 1]
    heights = [1]
    gs = gridspec.GridSpec(1, 2, height_ratios=heights, width_ratios=widths,
                           hspace=0.45, wspace=0.3)
    ax1 = plt.subplot(gs[0])
    if np.min(rms_cv) < np.min(rms_lc):
        trans = np.min(np.mean(rms_cv, axis=0))
    else:
        trans = np.min(np.mean(rms_lc, axis=0))
    mn_rms = np.mean(rms_lc, axis=0) - trans
    st_rms = st.sem(rms_lc, axis=0)
    plt.plot(noise_lvl, mn_rms, marker = 'o', color = 'blue', label = 'kCSD L-Curve')
    plt.fill_between(noise_lvl, mn_rms - st_rms, 
                     mn_rms + st_rms, alpha = 0.3, color = 'blue')
    mn_rms = np.mean(rms_cv, axis=0) - trans
    st_rms = st.sem(rms_cv, axis=0)
    plt.plot(noise_lvl, mn_rms, marker = 'o', color = 'green', label = 'kCSD Cross-Validation')
    plt.fill_between(noise_lvl, mn_rms - st_rms, 
                     mn_rms + st_rms, alpha = 0.3, color = 'green')
    plt.ylabel('Estimation Error', labelpad = 15)
    plt.xlabel('Relative Noise Level', labelpad = 15)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    set_axis(ax1, -0.05, 1.05, letter='A')
    ax1.legend(loc='upper left', frameon=False)
    # plt.title('Performance of regularization methods')

    '''second plot'''
    ax2 = plt.subplot(gs[1])
    mn_lam = np.mean(lam_lc, axis=0)
    st_lam = st.sem(lam_lc, axis=0)
    plt.plot(noise_lvl, mn_lam, marker = 'o', color = 'blue', label = 'kCSD L-Curve')
    plt.fill_between(noise_lvl, mn_lam - st_lam,
                    mn_lam + st_lam, alpha = 0.3, color = 'blue')
    mn_lam = np.mean(lam_cv, axis=0)
    st_lam = st.sem(lam_cv, axis=0)
    plt.plot(noise_lvl, mn_lam, marker = 'o', color = 'green', label = 'kCSD Cross-Validation')
    plt.fill_between(noise_lvl, mn_lam - st_lam,
                    mn_lam + st_lam, alpha = 0.3, color = 'green')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=((0.0, 0.0)))
    plt.ylabel('$\lambda$', labelpad = 30, rotation = 0)
    plt.xlabel('Relative Noise Level', labelpad = 15)
    set_axis(ax2, -0.05, 1.05, letter='B')
    ht, lh = ax2.get_legend_handles_labels()
    #fig.legend(ht, lh, loc='upper center', ncol=2, frameon=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(loc='upper left', frameon=False)
    fig.savefig('stats.png')

if __name__=='__main__':
    os.chdir("./LCurve/")
    noises = 9
    noise_lvl = np.linspace(0, 0.5, noises)
    sim_results = np.load('sim_results.npy')
    make_plot_perf(sim_results)
