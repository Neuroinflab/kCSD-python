import numpy as np
from kcsd import KCSD2D
from pathlib import Path
import DemoReadSGLXData.readSGLX as readSGLX
import matplotlib.pyplot as plt
import draw_utils as du
from scipy.signal import filtfilt, butter, argrelmax
#%%    
def fetch_electrodes(meta):
    imroList = meta['imroTbl'].split(sep=')')
    nChan = len(imroList) - 2
    electrode = np.zeros(nChan, dtype=int)
    channel = np.zeros(nChan, dtype=int)
    bank = np.zeros(nChan, dtype=int)
    reference_electrode = np.zeros(nChan, dtype=int)
    for i in range(nChan):
        currList = imroList[i+1].split(sep=' ')
        channel[i] = int(currList[0][1:])
        bank[i] = int(currList[1])
        reference_electrode[i] = currList[2]
    # Channel N => Electrode (1+N+384*A), where N = 0:383, A=0:2
    electrode = 1 + channel + 384 * bank
    return electrode, channel, reference_electrode

def create_electrode_map(start_x, start_y):
    x_dist = 16 #um
    y_dist = 20
    ele_map = {}
    ele_list = []
    for i in range(960):
        x_pos = start_x+(i%2)*x_dist*2+int(((i/2)%2))*x_dist
        y_pos = int(i/2)*y_dist
        ele_map[i+1] = (x_pos, y_pos)
        ele_list.append((i+1, x_pos, y_pos))
    return ele_map, ele_list

def eles_to_coords(eles):
    ele_map, ele_list = create_electrode_map(-24, 0)
    coord_list = []
    for ele in eles:
        coord_list.append(ele_map[ele])
    return np.array(coord_list)

def get_npx(path, time_start, time_stop):
    binFullPath = Path(path)
    meta = readSGLX.readMeta(binFullPath)
    rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
    # get electrode positions
    electrodes, channels, reference_electrode = fetch_electrodes(meta)
    ele_pos_def = eles_to_coords(electrodes)
    # convData is the potential in uV or mV
    Fs = readSGLX.SampRate(meta)
    time_start, time_stop = int(time_start*Fs), int(time_stop*Fs)
    selectData = rawData[:, time_start:time_stop]
    if meta['typeThis'] == 'imec': rawData = 1e3*readSGLX.GainCorrectIM(selectData, channels, meta)
    else: rawData = 1e3*readSGLX.GainCorrectNI(rawData, channels, meta)
    return rawData, ele_pos_def, channels, meta, reference_electrode

def do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit):
    ele_position = ele_pos_for_csd[:ele_limit[1]][2::20]
    csd_pots = pots_for_csd[:ele_limit[1]][2::20]
    k = KCSD2D(ele_position, csd_pots,
               h=1, sigma=1, 
               # xmin= -42, xmax=42, gdx=4,
               xmin=0, xmax=4000, gdx=4) 
    k.L_curve(Rs=np.linspace(30, 90, 1), lambdas=np.logspace(-9, -7, 1))
    # k.cross_validate(Rs=np.linspace(20, 30, 1), lambdas=np.logspace(-5, -3, 20))
    plt.figure()
    plt.imshow(k.curve_surf)#, vmin=-k.curve_surf.max(), vmax=k.curve_surf.max(), cmap='BrBG_r')
    plt.colorbar()
    return k, k.values('CSD'), ele_position
#%%
if __name__ == '__main__':
    # dir_path=  '/Users/Wladek/Dysk Google/kCSD_lcurve/validation/Steinmetz_data/'
    dir_path=  '/Users/Wladek/Dysk Google/kCSD_lcurve/validation/'
    # bin_path = 'Hopkins_20160722_g0_t0.imec.lf.bin'
    # bin_path = '08_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim50V_g0_t0.imec0.lf.bin'
    bin_path = '15_3800_bank0_defauld PnoFIltr_OLD_headsage_OLD_electrode_g0_t0.imec0.ap.bin'
    time_start = 46.55
    time_stop = 46.6
    data, ele_pos, channels, meta, ref = get_npx(dir_path+bin_path, time_start, time_stop)
    Fs = int(float(meta['imSampRate']))
    downsample = 1
    Fs = int(Fs/downsample)
    data = data[:, ::downsample]
    b,a = butter(3, [10/(Fs/2),100/(Fs/2)], btype = 'bandpass')
    data = filtfilt(b,a, data)
    b,a = butter(3, 500/(Fs/2), btype = 'highpass')
    filtdata = filtfilt(b,a, data)
    filtdata = np.delete(filtdata, 191, axis=0)
    data = np.delete(data, 191, axis=0)
#%%
    k, est_csd, ele_pos = do_kcsd(ele_pos, data, ele_limit = 320)
#%%
    plt.figure()
    plt.subplot(121)
    # plt.imshow(abs(filtdata)[::-1], aspect='auto', extent=[0,data.shape[1]/Fs, 4000, 0],
               # vmin=0, vmax =0.05 , cmap='Greys')
    plt.imshow(est_csd[15][::-1], aspect='auto', extent=[0,data.shape[1]/Fs, 4000, 0],
               vmin=-est_csd[15].max(), vmax =est_csd[15].max(), cmap='bwr', alpha= 1)
    plt.subplot(122)
    plt.imshow(data[::-1], aspect='auto', extent=[0,data.shape[1]/Fs, 4000, 0],
               vmin=-2, vmax =2 , cmap='PRGn')
    plt.colorbar()
