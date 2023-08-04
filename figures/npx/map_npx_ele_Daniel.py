import numpy as np
from pathlib import Path
from openpyxl import load_workbook
from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
import matplotlib.pyplot as plt
from matplotlib import gridspec

def eles_to_rows(eles):
    rows = []
    for ele in eles:
        rows.append(int(np.ceil(ele/2)))
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


ele_chan_dict, chan_ele_dict = load_chann_map()

def fetch_channels(eles):
    chans = []
    exist_ele = []
    for ii in eles:
        try:
            chans.append(ele_chan_dict[ii])
            exist_ele.append(ii)
        except KeyError:
            print('Not recording from ele', ii)
    return chans, exist_ele

# print(ele_dict)


# File with the data
# binFullPath = Path('./data/08_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim50V_g0_t0.imec0.lf.bin')
binFullPath = Path('/mnt/zasoby/data/neuropixel/Neuropixel data from Ewa Kublik/SOV_12/data/08_refGND_APx500_LFPx125_ApfiltON_corr_banks_stim50V_g0_t0.imec0.lf.bin')

tStart = 0        # in seconds
tEnd = 1
# chanList = [0, 6, 9, 383]
# eleList = np.arange(769, 860)
eleList = np.arange(0, 959)

chanList, eleList = fetch_channels(eleList)


# Which digital word to read. 
# For imec, there is only 1 digital word, dw = 0.
# For NI, digital lines 0-15 are in word 0, lines 16-31 are in word 1, etc.
dw = 0
# Which lines within the digital word, zero-based
# Note that the SYNC line for PXI 3B is stored in line 6.
dLineList = [6]

meta = readMeta(binFullPath)
sRate = SampRate(meta)
firstSamp = int(sRate*tStart)
lastSamp = int(sRate*tEnd)
rawData = makeMemMapRaw(binFullPath, meta)
selectData = rawData[chanList, firstSamp:lastSamp+1]
digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw, dLineList, meta)

if meta['typeThis'] == 'imec':
    # apply gain correction and convert to uV
    convData = 1e6*GainCorrectIM(selectData, chanList, meta)
else:
    # apply gain correction and convert to mV
    convData = 1e3*GainCorrectNI(selectData, chanList, meta)

tDat = np.arange(firstSamp, lastSamp+1)
tDat = 1000*tDat/sRate      # plot time axis in msec

# ax = plt.subplot(121)
# for ii, chan in enumerate(chanList):
#     ax.plot(tDat, convData[ii, :], label=str(chan)+' Ele'+str(chan_dict[chan]))
# plt.legend()
# ax = plt.subplot(122)
# for i in range(0, len(dLineList)):
#     ax.plot(tDat, digArray[i, :])

rowList = eles_to_rows(eleList)
num_rows = max(rowList) - min(rowList) + 1
print(num_rows)
fig = plt.figure(figsize=(4, num_rows))
gs = gridspec.GridSpec(nrows=num_rows, ncols=4, wspace=0, hspace=0)
all_maxy = -100
axs = []
for ii, chann in enumerate(chanList):
    ee = chan_ele_dict[chann]
    rr = eles_to_rows([ee])[0] - min(rowList) # last row first
    rr = num_rows - rr - 1
    print(rr, ee, num_rows-rr)
    off = ee%4
    if off == 0:
        ax = fig.add_subplot(gs[rr, 3])
    elif off == 1:
        ax = fig.add_subplot(gs[rr, 0])
    elif off == 2:
        ax = fig.add_subplot(gs[rr, 2])
    else:
        ax = fig.add_subplot(gs[rr, 1])
    ax.plot(tDat, convData[ii, :])
    all_maxy = max(all_maxy, max(convData[ii, :]))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_yticklabels([])
    # ax.set_yticks([])
    ax.set_title('E('+str(ee)+')')
    axs.append(ax)
print(all_maxy)
plt.show()

