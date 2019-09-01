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

def plot_surface(curve_surf, errsy, save_as):
    fsize = 18
    lambdas = np.logspace(-7, -3, 50)
    fig = plt.figure(figsize = (20,9), dpi = 300)
    gs = gridspec.GridSpec(16, 12, hspace=2, wspace=2)
    ax = plt.subplot(gs[0:16, 0:6])
    set_axis(ax, -0.05, 1.05, letter='A')
    plt.pcolormesh(lambdas, np.arange(9), curve_surf, 
                   cmap = 'BrBG', vmin = -2, vmax=2)
    plt.colorbar()
    for i,m in enumerate(curve_surf.argmax(axis=1)):
        plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', alpha = 0.7)
        if i==7:
            plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', 
                        label = 'Maximum Curvature', alpha = 0.7)
    plt.xlim(lambdas[1],lambdas[-1])
    plt.title('L-curve regularization', fontsize = fsize)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.12), ncol=1, 
               frameon = False, fontsize = fsize)
    plt.yticks(np.arange(8)+0.5, [str(x)+'x' for x in range(1,9)])
    plt.xscale('log')
    plt.ylabel('Parameter $R$ in electrode distance', fontsize=fsize, labelpad = 15)
    plt.xlabel('$\lambda$',fontsize=fsize)
    ax = plt.subplot(gs[0:16, 6:12])
    set_axis(ax, -0.05, 1.05, letter='B')
    plt.pcolormesh(lambdas, np.arange(9), errsy, cmap = 'Greys')
    plt.colorbar()
    for i,m in enumerate(errsy.argmin(axis=1)):
        plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', alpha = 0.7)
        if i==7:
            plt.scatter([lambdas[m]], [i+0.5], s=50, color='red', 
                        label = 'Minimum Error', alpha = 0.7)
    plt.xlim(lambdas[1],lambdas[-1])
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.12), ncol=1, 
               frameon = False, fontsize = fsize)
    plt.title('Cross-validation regularization', fontsize = fsize)
    plt.yticks(np.arange(8)+0.5, [str(x)+'x' for x in range(1,9)])
    plt.xscale('log')
    plt.xlabel('$\lambda$', fontsize=fsize)
    fig.savefig(save_as+'.png')

if __name__=='__main__':
#    os.chdir("./LCurve/")
    noises = 3
    noise_lvl = np.linspace(0, 0.5, noises)
#    df = np.load('LC2/data_fig4_and_fig13_lc_noise25.0.npz')
    Rs = np.linspace(0.025, 8*0.025, 8)
    title = ['nazwa_pliku']
    save_as = 'noise'
#    plot_surface(df['curve_surf'], df['errsy'], save_as+'surf')
    plt.close('all')