import numpy as np
from kcsd import KCSD2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import filtfilt, butter
from figure_properties import *
plt.close('all')
#%%
def make_plot_spacetime(ax, xx, yy, zz, title='True CSD', cmap=cm.bwr_r, ymin=0, ymax=10000):
    im = ax.imshow(zz,extent=[0, zz.shape[1]/Fs*1000,-3500, 500], aspect='auto',
                   vmax = 1*zz.max(),vmin = -1*zz.max(), cmap=cmap)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Y ($\mu$m)')
    if 'Pot' in title: ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title)
    if 'CSD' in title:
        plt.colorbar(im, orientation='vertical', format='%.2f', ticks = [-0.01,0,0.01])
    else:
        plt.colorbar(im, orientation='vertical', format='%.1f', ticks = [-0.6,0,0.6]) 
    # plt.gca().invert_yaxis()

def make_plot(ax, xx, yy, zz, title='True CSD', cmap=cm.bwr):
    ax.set_aspect('auto')
    levels = np.linspace(zz.min(), -zz.min(), 61)
    im = ax.contourf(xx, -(yy-500), zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X ($\mu$m)')
    ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title)
    if 'CSD' in title: 
        plt.colorbar(im, orientation='vertical',  format='%.2f', ticks=[-0.02,0,0.02])
    else: plt.colorbar(im, orientation='vertical',  format='%.1f', ticks=[-0.6,0,0.6])
    plt.scatter(ele_pos[:, 0], 
                -(ele_pos[:, 1]-500),
                s=0.8, color='black')
    # plt.gca().invert_yaxis()
    return ax
    
def eles_to_ycoord(eles):
    y_coords = []
    for ii in range(192):
        y_coords.append(ii*20)
        y_coords.append(ii*20)
    return y_coords[::-1]

def eles_to_xcoord(eles):
    x_coords = []
    for ele in eles:
        off = ele%4
        if off == 1: x_coords.append(-24)
        elif off == 2: x_coords.append(8)
        elif off == 3: x_coords.append(-8)
        elif off==0: x_coords.append(24)
    return x_coords

def eles_to_coords(eles):
    xs = eles_to_xcoord(eles)
    ys = eles_to_ycoord(eles)
    return np.array((xs, ys)).T

def plot_1D_pics(k, est_csd, est_pots, tp, cut=9):
    plt.figure(figsize=(12, 8))
    # plt.suptitle('plane: '+str(k.estm_x[cut,0])+' $\mu$m '+' $\lambda$ : '+str(k.lambd)+
                 # '  R: '+ str(k.R))
    ax1 = plt.subplot(122)
    set_axis(ax1, -0.05, 1.05, letter= 'B')
    make_plot_spacetime(ax1, k.estm_x, k.estm_y, est_csd[cut,:,:], 
              title='Estimated CSD', cmap='bwr')
    for lvl, name in zip([-500,-850,-2000], ['II/III', 'IV', 'V/VI']):
        plt.axhline(lvl, ls='--', color='grey')
        plt.text(340, lvl+20, name)
    plt.xlim(250, 400)
    ax2 = plt.subplot(121)
    set_axis(ax2, -0.05, 1.05, letter= 'A')
    make_plot_spacetime(ax2, k.estm_x, k.estm_y, est_pots[cut,:,:],
              title='Estimated LFP', cmap='PRGn')
    plt.axvline(tp/Fs*1000, ls='--', color ='grey', lw=2)
    plt.xlim(250, 400)
    plt.tight_layout()

def plot_2D_pics(k, est_csd, est_pots, tp, cut, save=0):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(122)
    set_axis(ax1, -0.05, 1.05, letter= 'B')
    make_plot(ax1, k.estm_x, k.estm_y, est_csd[:,:,tp], 
              title='Estimated CSD', cmap='bwr')
    # for i in range(383): plt.text(ele_pos_for_csd[i,0], ele_pos_for_csd[i,1]+8, str(i+1))
    plt.axvline(k.estm_x[cut][0], ls='--', color ='grey', lw=2)
    ax2 = plt.subplot(121)
    set_axis(ax2, -0.05, 1.05, letter= 'A')
    make_plot(ax2, k.estm_x, k.estm_y, est_pots[:,:,tp],
              title='Estimated LFP', cmap='PRGn')
    # plt.suptitle(' $\lambda$ : '+str(k.lambd)+ '  R: '+ str(k.R))
    plt.tight_layout()

def do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit):
    ele_position = ele_pos_for_csd[:ele_limit[1]][0::1]
    csd_pots = pots_for_csd[:ele_limit[1]][0::1]
    k = KCSD2D(ele_position, csd_pots,
               h=1, sigma=1, R_init=32, lambd=1e-9,
               xmin= -42, xmax=42, gdx=4,
               ymin=0, ymax=4000, gdy=4)
    # k.L_curve(Rs=np.linspace(16, 48, 3), lambdas=np.logspace(-9, -3, 20))
    return k, k.values('CSD'), k.values('POT'), ele_position
#%%
if __name__ == '__main__':
    lowpass = 0.5
    highpass = 300
    Fs = 30000
    resamp = 12
    tp= 760
    
    forfilt=np.load('npx_data.npy')
    
    [b,a] = butter(3, [lowpass/(Fs/2.0), highpass/(Fs/2.0)] ,btype = 'bandpass')
    filtData = filtfilt(b,a, forfilt)
    pots_resamp = filtData[:,::resamp]
    pots = pots_resamp[:, :]
    Fs=int(Fs/resamp)
    
    pots_for_csd = np.delete(pots, 191, axis=0)
    ele_pos_def = eles_to_coords(np.arange(384,0,-1))
    ele_pos_for_csd = np.delete(ele_pos_def, 191, axis=0)
    
    k, est_csd, est_pots, ele_pos = do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit = (0,320))
    
    plot_1D_pics(k, est_csd, est_pots, tp, 15) 
    plot_2D_pics(k, est_csd, est_pots, tp=tp, cut=15)
