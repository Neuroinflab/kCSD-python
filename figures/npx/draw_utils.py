# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:27:39 2020

@author: Wladek
"""
import matplotlib.pyplot as plt
import numpy as np

def make_plot_spacetime(ax, xx, yy, zz, Fs, title='True CSD', cmap='bwr', ymin=0, ymax=10000):
    im = ax.imshow(zz,extent=[0, zz.shape[1]/Fs,4000,0], aspect='auto',
                   vmax = 1*zz.max(),vmin = -1*zz.max(), cmap=cmap)
    ax.set_xlabel('Time sec')
    ax.set_ylabel('')
    ax.set_title(title)
    plt.colorbar(im, orientation='vertical', format='%.2f')
    # plt.gca().invert_yaxis()

def make_plot(ax, xx, yy, zz, ele_pos, title='True CSD', cmap='bwr'):
    ax.set_aspect('auto')
    levels = np.linspace(zz.min(), -zz.min(), 64)
    # if 'CSD' in title: levels = np.linspace(-.001, .001, 64)
    # if 'POT' in title: levels = np.linspace(-3, 3, 64)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    # ticks = np.linspace(-100,100, 7, endpoint=True)
    plt.colorbar(im, orientation='vertical', format='%.2f')#, ticks=ticks)
    plt.scatter(ele_pos[:, 0], ele_pos[:, 1], s=0.8, color='black')
    plt.gca().invert_yaxis()
    return ax

def plot_1D_pics(k, est_csd, est_pots, savedir, save=1, cut=9):
    plt.figure(figsize=(12, 8))
    plt.suptitle('plane: '+str(k.estm_x[cut,0])+' $\mu$m '+' $\lambda$ : '+str(k.lambd)+
                 '  R: '+ str(k.R))
    ax1 = plt.subplot(122)
    make_plot_spacetime(ax1, k.estm_x, k.estm_y, est_csd[cut,:,:], 
              title='Estimated CSD', cmap='bwr')
    for lvl, name in zip([500,700,1200,1600,2000], ['I', 'II/III', 'IV', 'V', 'VI']):
        plt.axhline(lvl, ls='--', color='grey')
        plt.text(340, lvl-20, name)
    plt.xlim(.25,.35)
    ax2 = plt.subplot(121)
    make_plot_spacetime(ax2, k.estm_x, k.estm_y, est_pots[cut,:,:],
              title='Estimated POT', cmap='PRGn')
    plt.xlim(.25,.35)
    if save:
        if cut<10: plt.savefig(savedir +'csd_spacetime_reg_zoom_0'+str(cut))
        elif cut>=10: plt.savefig(savedir + 'csd_spacetime_reg_zoom_'+str(cut))

def plot_2D_pics(tp, k, est_csd, est_pots, Fs, cut, ele_pos, save=0):
    for i in range(1):
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(121)
        make_plot(ax1, k.estm_x, k.estm_y, est_csd[:,:,tp], ele_pos,
                  title='Estimated CSD', cmap='bwr')
        for i in range(len(ele_pos)): plt.text(ele_pos[i,0], ele_pos[i,1]+8, str(i), fontsize=8)
        plt.axvline(k.estm_x[cut][0], ls='--')
        ax2 = plt.subplot(122)
        make_plot(ax2, k.estm_x, k.estm_y, est_pots[:,:,tp], ele_pos,
                  title='Estimated POT', cmap='PRGn')
        plt.suptitle('lambda = %f, R = %f, snap = %f' % (k.lambd, k.R, i*2/Fs))
        if save:
            if i<10: plt.savefig('csd_00'+str(i))
            elif i>10: plt.savefig('csd_0'+str(i))
            elif i>100: plt.savefig('csd_'+str(i))

def plot_electrodes_position(electrodes):
    plt.figure()
    plt.scatter(electrodes[:, 0], electrodes[:, 1], s=0.8, color='black')