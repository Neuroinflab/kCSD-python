import numpy as np
from kcsd import KCSD2D
from pathlib import Path
import DemoReadSGLXData.readSGLX as readSGLX
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import filtfilt, butter
from figure_properties import *
# from matplotlib import gridspec
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
    # if 'CSD' in title: levels = levels = np.linspace(zz.min(), -zz.min(), 32)
    # if 'POT' in title: levels = np.linspace(zz.min(), -zz.min(), 64)
    im = ax.contourf(xx, -(yy-500), zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X ($\mu$m)')
    ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title)
    # ticks = np.linspace(-100,100, 7, endpoint=True)
    if 'CSD' in title: 
        plt.colorbar(im, orientation='vertical',  format='%.2f', ticks=[-0.02,0,0.02])
    else: plt.colorbar(im, orientation='vertical',  format='%.1f', ticks=[-0.6,0,0.6])
    plt.scatter(ele_pos[:, 0], 
                -(ele_pos[:, 1]-500),
                s=0.8, color='black')
    # plt.gca().invert_yaxis()
    return ax

def dan_fetch_electrodes(meta):
    imroList = meta['imroTbl'].split(sep=')')
    # One entry for each channel plus header entry,
    # plus a final empty entry following the last ')'
    nChan = len(imroList) - 2
    electrode = np.zeros(nChan, dtype=int)        # default type = float
    channel = np.zeros(nChan, dtype=int)
    bank = np.zeros(nChan, dtype=int)
    for i in range(0, nChan):
        currList = imroList[i+1].split(sep=' ')
        print(currList)
        channel[i] = int(currList[0][1:])
        bank[i] = int(currList[1])
        # reference_electrode[i] = currList[2]
    # Channel N => Electrode (1+N+384*A), where N = 0:383, A=0:2
    electrode = 1 + channel + 384 * bank
    return(electrode, channel)
    
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
#%%
binFullPath = Path('/Users/Wladek/Dysk Google/kCSD_lcurve/validation/'
                   '15_3800_bank0_defauld PnoFIltr_OLD_headsage_OLD_electrode_g0_t0.imec0.ap.bin')

meta = readSGLX.readMeta(binFullPath)
Fss = int(readSGLX.SampRate(meta))

path = '/Users/Wladek/Dysk Google/kCSD_lcurve/validation/'
electrodes, channels = dan_fetch_electrodes(meta)
ch_order = electrodes.argsort()

rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
selectData = rawData[channels, 30*Fss:50*Fss]
# convData is the potential in uV or mV
if meta['typeThis'] == 'imec':    
    rawData = 1e3*readSGLX.GainCorrectIM(selectData, channels, meta)
else:    
    rawData = 1e3*readSGLX.GainCorrectNI(selectData, channels, meta)

electrodes.sort()
ele_pos_def = eles_to_coords(electrodes[::-1])
#%%
ex_time = 16.6#12.7
lowpass = 0.5#20 beta
highpass = 300#50 beta
after = 0.3
forfilt = rawData[:,int((ex_time-0.3)*Fss):int((ex_time+after)*Fss)]
# forfilt = detrend(forfilt, bp=np.array([0,int(0.1*Fss)]))
# for i in range(384): forfilt[i] = forfilt[i]/np.std(forfilt[i])
[b,a] = butter(3, [lowpass/(Fss/2.0), highpass/(Fss/2.0)] ,btype = 'bandpass')
filtData = filtfilt(b,a, forfilt)
np.save('npx_data', filtData)
#%%
resamp = 12
pots_resamp = filtData[:,::resamp]
pots = pots_resamp[:, :]
Fs=int(Fss/resamp)
#%%
time = np.linspace(0, pots.shape[1]/Fs, pots.shape[1])
plt.figure()
plt.subplot(121)
for ch in range(0,384,8):#, potsy.shape[0], 8):
    plt.plot(time, pots[ch,:]+1*ch, color='grey', lw=0.3)
print('start averaging')
plt.subplot(122)
plt.imshow(pots[::-1], extent=[0,pots.shape[1]/Fs,pots.shape[0],0],
            aspect='auto', cmap = 'PRGn', 
            vmin = -pots.max(), vmax = pots.max())
# plt.xlim(280, 330)
# plt.subplot(133)
# %%
pots_for_csd = np.delete(pots, 191, axis=0)
ele_pos_for_csd = np.delete(ele_pos_def, 191, axis=0)
# pots_for_csd = pots
# ele_pos_for_csd = ele_pos_def
def do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit):
    ele_position = ele_pos_for_csd[:ele_limit[1]][0::1]
    csd_pots = pots_for_csd[:ele_limit[1]][0::1]
    k = KCSD2D(ele_position, csd_pots,
               h=1, sigma=1, 
               xmin= -42, xmax=42, gdx=4,
               ymin=0, ymax=4000, gdy=4)
    k.L_curve(Rs=np.linspace(32, 90, 1), lambdas=np.logspace(-9, -7, 1))
    # k.cross_validate(Rs=np.linspace(20, 30, 1), lambdas=np.logspace(-5, -3, 20))
    plt.figure()
    plt.imshow(k.curve_surf)#, vmin=-k.curve_surf.max(), vmax=k.curve_surf.max(), cmap='BrBG_r')
    plt.colorbar()
    return k, k.values('CSD'), k.values('POT'), ele_position

k, est_csd, est_pots, ele_pos = do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit = (0,320))
#%%
plt.close('all')
save= 1
tp= 760
def plot_1D_pics(k, est_csd, est_pots, cut=9):
    plt.figure(figsize=(12, 8))
    # plt.suptitle('plane: '+str(k.estm_x[cut,0])+' $\mu$m '+' $\lambda$ : '+str(k.lambd)+
                 # '  R: '+ str(k.R))
    ax1 = plt.subplot(122)
    set_axis(ax1, -0.05, 1.05, letter= 'B')
    make_plot_spacetime(ax1, k.estm_x, k.estm_y, est_csd[cut,:,:], 
              title='Estimated CSD', cmap=cm.bwr)
    for lvl, name in zip([-500,-850,-2000], ['II/III', 'IV', 'V/VI']):
        plt.axhline(lvl, ls='--', color='grey')
        plt.text(340, lvl+20, name)
    plt.xlim(250, 400)
    ax2 = plt.subplot(121)
    set_axis(ax2, -0.05, 1.05, letter= 'A')
    make_plot_spacetime(ax2, k.estm_x, k.estm_y, est_pots[cut,:,:],
              title='Estimated LFP', cmap=cm.PRGn)
    plt.axvline(tp/Fs*1000, ls='--', color ='grey', lw=2)
    plt.xlim(250, 400)
    plt.tight_layout()
    plt.savefig(savedir +'Figure_15', dpi=300)
savedir = '/Users/Wladek/Dysk Google/kCSD_lcurve/validation/'
for cut in range(15,16,1): plot_1D_pics(k, est_csd, est_pots, cut)
# plt.close('all')

def plot_2D_pics(tp, cut, save=0):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(122)
    set_axis(ax1, -0.05, 1.05, letter= 'B')
    make_plot(ax1, k.estm_x, k.estm_y, est_csd[:,:,tp], 
              title='Estimated CSD', cmap=cm.bwr)
    # for i in range(383): plt.text(ele_pos_for_csd[i,0], ele_pos_for_csd[i,1]+8, str(i+1))
    plt.axvline(k.estm_x[cut][0], ls='--', color ='grey', lw=2)
    ax2 = plt.subplot(121)
    set_axis(ax2, -0.05, 1.05, letter= 'A')
    make_plot(ax2, k.estm_x, k.estm_y, est_pots[:,:,tp],
              title='Estimated LFP', cmap=cm.PRGn)
    # plt.suptitle(' $\lambda$ : '+str(k.lambd)+ '  R: '+ str(k.R))
    plt.tight_layout()
    plt.savefig(savedir +'Figure_14', dpi=300)
        
plot_2D_pics(tp = tp, cut=15)