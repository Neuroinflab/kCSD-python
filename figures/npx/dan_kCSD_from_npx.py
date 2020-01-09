import numpy as np
from kcsd import KCSD2D
from pathlib import Path
from openpyxl import load_workbook
import DemoReadSGLXData.readSGLX as readSGLX
# from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw,
#  GainCorrectIM, GainCorrectNI, ExtractDigital
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib import gridspec


def make_plot(ax, xx, yy, zz, title='True CSD', cmap=cm.bwr):
    # fig = plt.figure(figsize=(7, 7))
    # ax = plt.subplot(111)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(zz))
    levels = np.linspace(-1 * t_max, t_max, 32)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    # ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    # plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
    return ax


def dan_make_plot(k):
    fig = plt.figure(figsize=(7, 7))
    ax1 = plt.subplot(121)
    
    est_csd = k.values('CSD').squeeze()
    est_pots = k.values('POT').squeeze()
    
    make_plot(ax1, k.estm_x, k.estm_y, est_csd[:, :], 
          title='Estimated CSD', cmap=cm.bwr)
    
    ax2 = plt.subplot(122)
    make_plot(ax2, k.estm_x, k.estm_y, est_pots[:, :],
          title='Estimated POT', cmap=cm.PRGn)
    fig.suptitle('lambda = %f, R = %f' % (k.lambd, k.R))
    
    return fig



# Specific to Ewas experimental setup
def load_chann_map():
    book = load_workbook('NP_do_map.xlsx')
    sheet = book.get_sheet_by_name('sov12 sorted')
    eleid = sheet['C3':'C386']
    chanid = sheet['J3':'J386']
    chan_ele_dict = {}
    ele_chan_dict = {}
    for e,c in zip(eleid, chanid):
        chan_ele_dict[int(c[0].value)] = int(e[0].value)
        ele_chan_dict[int(e[0].value)] = int(c[0].value)
    return ele_chan_dict, chan_ele_dict


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
        channel[i] = int(currList[0][1:])
        bank[i] = int(currList[1])
        # reference_electrode[i] = currList[2] 
    
    # Channel N => Electrode (1+N+384*A), where N = 0:383, A=0:2
    electrode = 1 + channel + 384 * bank
    
    return(electrode, channel)
    

# def fetch_channels(eles):
#     chans = []
#     exist_ele = []
#     for ii in eles:
#         try:
#             chans.append(ele_chan_dict[ii])
#             exist_ele.append(ii)
#         except KeyError:
#             print('Not recording from ele', ii)
#     return chans, exist_ele

def eles_to_rows(eles):
    rows = []
    for ele in eles:
        rows.append(np.int(np.ceil(ele/2)))
    return rows

def eles_to_ycoord(eles):
    rows = eles_to_rows(eles)
    y_coords = []
    for ii in rows:
        y_coords.append(int((480 - ii)*20))
    return y_coords

def eles_to_xcoord(eles):
    x_coords = []
    for ele in eles:
        off = ele%4
        if off == 1:
            x_coords.append(-24)
        elif off == 2:
            x_coords.append(8)
        elif off == 3:
            x_coords.append(-8)
        else:
            x_coords.append(24)
    return x_coords

def eles_to_coords(eles):
    xs = eles_to_xcoord(eles)
    ys = eles_to_ycoord(eles)
    return np.array((xs, ys)).T


# File with the data
# old
# binFullPath = Path('./data/08_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim50V_g0_t0.imec0.lf.bin')
# Daniel
binFullPath = Path('/mnt/zasoby/data/neuropixel/Neuropixel data from Ewa Kublik/SOV_12/data/08_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim50V_g0_t0.imec0.lf.bin')
# binFullPath = Path('/mnt/zasoby/data/neuropixel/Neuropixel data from Ewa Kublik/SOV_12/data/09_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim25V_g0_t0.imec0.lf.bin')


# Chaitanya
# binFullPath = Path('/home/chaitanya/LFP/SOV_12/data/08_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim50V_g0_t0.imec0.lf.bin')

meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

tStart, tEnd = 0., 600.    # 0., 600.  # 0., 1.        # in seconds

firstSamp = int(sRate*tStart)
lastSamp = int(sRate*tEnd)


# Return array of original channel IDs. As an example, suppose we want the
# imec gain for the ith channel stored in the binary data. A gain array
# can be obtained using ChanGainsIM(), but we need an original channel
# index to do the lookup. Because you can selectively save channels, the
# ith channel in the file isn't necessarily the ith acquired channel.
# Use this function to convert from ith stored to original index.
# Note that the SpikeGLX channels are 0 based.
#
# chans = readSGLX.OriginalChans(meta)

electrodes, channels = dan_fetch_electrodes(meta)
# for Ewa's initial file, channel 768 is SY
# and it hould be removed - this has not been done yet
# DANIEL





# =============================================================================
# # chanList = [0, 6, 9, 383]
# # eleList = np.arange(769, 860)
# eleList = np.arange(0, 959)
# 
# ele_chan_dict, chan_ele_dict = load_chann_map()
# # print(ele_dict)
# chanList, eleList = fetch_channels(eleList)
# 
# =============================================================================


# Which digital word to read. 
# For imec, there is only 1 digital word, dw = 0.
# For NI, digital lines 0-15 are in word 0, lines 16-31 are in word 1, etc.
dw = 0
# Which lines within the digital word, zero-based
# Note that the SYNC line for PXI 3B is stored in line 6.
dLineList = [6]

rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
selectData = rawData[channels, firstSamp:lastSamp+1]
# digArray = readSGLX.ExtractDigital(rawData, firstSamp, lastSamp, dw, dLineList, meta)

# convData is the potential in uV or mV
if meta['typeThis'] == 'imec':
    # apply gain correction and convert to uV
    convData = 1e6*readSGLX.GainCorrectIM(selectData, channels, meta)
else:
    # apply gain correction and convert to mV
    convData = 1e3*readSGLX.GainCorrectNI(selectData, channels, meta)

tDat = np.arange(firstSamp, lastSamp+1)
tDat = 1000*tDat/sRate      # plot time axis in msec



ele_pos = eles_to_coords(electrodes)
print(ele_pos)
csd_at_time = 0.3
pots = []
for ii, chann in enumerate(channels):
    print(ii, chann)
    pots.append(convData[ii, int(sRate*csd_at_time)])

pots = np.array(pots)
print(pots.shape)



electrode_order = np.argsort(ele_pos[:,1])
temp_pots = convData[electrode_order, :]
ax = plt.subplot(111)
plt.imshow(temp_pots[:, 192000:194000], aspect='auto')
# ax.set_aspect(100000)

plt.imshow(convData[:, 192000:194000], aspect='auto')


plt.figure()
ax = plt.subplot(111)
for n in range(384):
    plt.plot(100*n+convData[n, 700000:799999])



pots = pots.reshape((len(channels), 1))
R = 5. # 0.3
lambd = 0.
h = 1.   # 50
sigma = 0.3

k = KCSD2D(ele_pos, pots, h=h, sigma=sigma,
               xmin=-400, xmax=400,
               # ymin=1100, ymax=2000,
               # ymin=1000, ymax=10000,
               ymin=500, ymax=3000,
               gdx=10, gdy=10, lambd=lambd,
               R_init=R, n_src_init=10000,
               src_type='gauss')   # rest of the parameters are set at default
dan_make_plot(k)


k.L_curve(Rs=np.logspace(-1., 2., 5), lambdas = None)
# k.L_curve(Rs=np.logspace(-1., 2., 11), lambdas=np.logspace(-5., 1., 11))
# k.L_curve(Rs=np.logspace(-1., 2., 11), lambdas=np.logspace(-5., 1., 31))
plt.imshow(k.curve_surf)

# k.cross_validate(Rs=np.logspace(0., 2., 21), lambdas=np.logspace(-5., 1., 11))
# k.cross_validate(Rs=np.linspace(0.1, 1.001, 2), lambdas=None)
# 2 -> 20

dan_make_plot(k)

for h in 1., 4., 16., 32., 64., 128.:
    k = KCSD2D(ele_pos, pots, h=h, sigma=sigma,
               xmin=-400, xmax=400,
               # ymin=1100, ymax=2000,
               # ymin=1000, ymax=10000,
               ymin=500, ymax=3000,
               gdx=10, gdy=10, lambd=lambd, 
               R_init=R, n_src_init=10000,
               src_type='gauss')   # rest of the parameters are set at default
    k.L_curve(Rs=np.logspace(-1., 2., 11))
    plt.imshow(k.curve_surf)
    dan_make_plot(k)


# =============================================================================
# for R in np.logspace(0., 2., 11):
#     for lambd in np.logspace(-5., 1., 11):
#         k = KCSD2D(ele_pos, pots, h=h, sigma=sigma,
#                 xmin=-35, xmax=35,
#                 ymin=1100, ymax=2000,
#                 # ymin=1000, ymax=10000,
#                 gdx=10, gdy=10, lambd=lambd,
#                 R_init=R, n_src_init=1000,
#                 src_type='gauss')   # rest of the parameters are set at default
#     
#         est_csd = k.values('CSD')
#         est_csd = est_csd.reshape(7, 90)
#         est_pots = k.values('POT')
#         est_pots = est_pots.reshape(7, 90)
#         
#         dan_make_plot(k)
# 
# =============================================================================


# make_plot(k.estm_x, k.estm_y, est_csd[:, :],
#           title='Estimated CSD without CV', cmap=cm.bwr)

# make_plot(k.estm_x, k.estm_y, est_pots[:, :],
#           title='Estimated POT without CV', cmap=cm.PRGn)


# # ax = plt.subplot(121)
# # for ii, chan in enumerate(chanList):
# #     ax.plot(tDat, convData[ii, :], label=str(chan)+' Ele'+str(chan_dict[chan]))
# # plt.legend()
# # ax = plt.subplot(122)
# # for i in range(0, len(dLineList)):
# #     ax.plot(tDat, digArray[i, :])

# rowList = eles_to_rows(eleList)
# num_rows = max(rowList) - min(rowList) + 1
# print(num_rows)
# fig = plt.figure(figsize=(4, num_rows))
# gs = gridspec.GridSpec(nrows=num_rows, ncols=4, wspace=0, hspace=0)
# all_maxy = -100
# axs = []
# for ii, chann in enumerate(chanList):
#     ee = chan_ele_dict[chann]
#     rr = eles_to_rows([ee])[0] - min(rowList) # last row first
#     rr = num_rows - rr - 1
#     print(rr, ee, num_rows-rr)
#     off = ee%4
#     if off == 0:
#         ax = fig.add_subplot(gs[rr, 3])
#     elif off == 1:
#         ax = fig.add_subplot(gs[rr, 0])
#     elif off == 2:
#         ax = fig.add_subplot(gs[rr, 2])
#     else:
#         ax = fig.add_subplot(gs[rr, 1])
#     ax.plot(tDat, convData[ii, :])
#     all_maxy = max(all_maxy, max(convData[ii, :]))
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     # ax.spines['left'].set_visible(False)
#     # ax.set_yticklabels([])
#     # ax.set_yticks([])
#     ax.set_title('E('+str(ee)+')')
#     axs.append(ax)
# print(all_maxy)
# plt.show()

