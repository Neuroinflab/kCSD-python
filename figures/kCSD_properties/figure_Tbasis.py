"""
@author: mkowalska
"""

import os
from os.path import expanduser
import numpy as np
from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import time

import targeted_basis as tb

__abs_file__ = os.path.abspath(__file__)
home = expanduser('~')
DAY = datetime.datetime.now()
DAY = DAY.strftime('%Y%m%d')
TIMESTR = time.strftime("%H%M%S")
SAVE_PATH = home + "/kCSD_results/" + DAY + '/' + TIMESTR

tb.makemydir(SAVE_PATH)
tb.save_source_code(SAVE_PATH, time.strftime("%Y%m%d-%H%M%S"))


def set_axis(ax, letter=None):
    ax.text(
        -0.05,
        1.05,
        letter,
        fontsize=18,
        weight='bold',
        transform=ax.transAxes)
    return ax


def make_subplot(ax, true_csd, est_csd, estm_x, title=None, ele_pos=None,
                 xlabel=False, ylabel=False, letter='', t_max=None):

    x = np.linspace(0, 1, 100)
    ax.plot(x, true_csd, ':', label='TrueCSD')
    ax.plot(estm_x, est_csd, '--', label='kCSD')
    ax.plot(ele_pos, np.zeros(len(ele_pos)), 'ko', label='Electrodes')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    if xlabel:
        ax.set_xlabel('Depth (mm)', fontsize=15)
    if ylabel:
        ax.set_ylabel('CSD (mA/mm)', fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=15)
    ax.set_xticks([0, 0.5, 1])
    set_axis(ax, letter=letter)
    return ax


N_SRC = 64
ELE_LIMS = [0, 1.]  # range of electrodes space
TRUE_CSD_XLIMS = [0., 1.]
TOTAL_ELE = 12

#  A
R = 0.2
MU = 0.25
csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                        TRUE_CSD_XLIMS, R, MU,
                                                        TOTAL_ELE, ELE_LIMS)

fig = plt.figure(figsize=(12, 12))
widths = [1, 1, 1]
heights = [1, 1, 1]
gs = gridspec.GridSpec(3, 3, height_ratios=heights, width_ratios=widths)

ax = fig.add_subplot(gs[0, 0])
xmin = 0
xmax = 1
ext_x = 0
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                        xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title='Basis lims = [0, 1]', xlabel=False, ylabel=True,
             letter='A')

ax = fig.add_subplot(gs[0, 1])
xmin = -0.5
xmax = 1
ext_x = -0.5
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                        xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title='Basis lims = [0, 0.5]', xlabel=False, ylabel=False,
             letter='B')

ax = fig.add_subplot(gs[0, 2])
xmin = 0
xmax = 1.5
ext_x = -0.5
obj = tb.modified_bases(val, pots, ele_pos, N_SRC,  h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x,
                        xmin=xmin, xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title='Basis lims = [0.5, 1]', xlabel=False, ylabel=False,
             letter='C')


ELE_LIMS = [0, 0.5]
TOTAL_ELE = 6
csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                        TRUE_CSD_XLIMS, R, MU,
                                                        TOTAL_ELE, ELE_LIMS)
ax = fig.add_subplot(gs[1, 0])
xmin = 0
xmax = 1
ext_x = 0
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                        xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title=None, xlabel=False, ylabel=True, letter='D')

ax = fig.add_subplot(gs[1, 1])
xmin = -0.5
xmax = 1
ext_x = -0.5
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                        xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title=None, xlabel=False, ylabel=False, letter='E')

ax = fig.add_subplot(gs[1, 2])
xmin = 0
xmax = 1.5
ext_x = -0.5
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x,
                        xmin=xmin, xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title=None, xlabel=False, ylabel=False, letter='F')


ELE_LIMS = [0.5, 1.]
TOTAL_ELE = 6
csd_at, true_csd, ele_pos, pots, val = tb.simulate_data(tb.csd_profile,
                                                        TRUE_CSD_XLIMS, R, MU,
                                                        TOTAL_ELE, ELE_LIMS)
ax = fig.add_subplot(gs[2, 0])
xmin = 0
xmax = 1
ext_x = 0
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                        xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title=None, xlabel=True, ylabel=True, letter='G')

ax = fig.add_subplot(gs[2, 1])
xmin = -0.5
xmax = 1
ext_x = -0.5
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x, xmin=xmin,
                        xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title=None, xlabel=True, ylabel=False, letter='H')

ax = fig.add_subplot(gs[2, 2])
xmin = 0
xmax = 1.5
ext_x = -0.5
obj = tb.modified_bases(val, pots, ele_pos, N_SRC, h=0.25,
                        sigma=0.3, gdx=0.01, ext_x=ext_x,
                        xmin=xmin, xmax=xmax)
make_subplot(ax, true_csd, obj.values('CSD'), obj.estm_x, ele_pos=ele_pos,
             title=None, xlabel=True, ylabel=False, letter='I')

fig.savefig(os.path.join(SAVE_PATH, 'targeted_basis.png'), dpi=300)
plt.show()
