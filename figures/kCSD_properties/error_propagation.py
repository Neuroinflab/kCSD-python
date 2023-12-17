#!/usr/bin/python3
### BEGIN colorblind_friendly.py ###
# Based on
# Color Universal Design (CUD)
# - How to make figures and presentations that are friendly to Colorblind people
#
#
# Masataka Okabe
# Jikei Medial School (Japan)
#
# Kei Ito
# University of Tokyo, Institute for Molecular and Cellular Biosciences (Japan)
# (both are strong protanopes)
# 11.20.2002 (modified on 2.15.2008, 9.24.2008)
# http://jfly.iam.u-tokyo.ac.jp/color/#pallet

import collections
from matplotlib import colors


_Color = collections.namedtuple('_Color', ['red', 'green', 'blue'])

def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


_BLACK     = _Color(  0,   0,   0)
_ORANGE    = _Color(230, 159,   0)
_SKY_BLUE  = _Color( 86, 180, 233)
_GREEN     = _Color(  0, 158, 115)
_YELLOW    = _Color(240, 228,  66)
_BLUE      = _Color(  0, 114, 178)
_VERMILION = _Color(213,  94,   0)
_PURPLE    = _Color(204, 121, 167)

BLACK     = _html(*_BLACK)
ORANGE    = _html(*_ORANGE)
SKY_BLUE  = _html(*_SKY_BLUE)
GREEN     = _html(*_GREEN)
YELLOW    = _html(*_YELLOW)
BLUE      = _html(*_BLUE)
VERMILION = _html(*_VERMILION)
PURPLE    = _html(*_PURPLE)

def _BipolarColormap(name, negative, positive):
    return colors.LinearSegmentedColormap(
                      name,
                      {k: [(0.0,) + (getattr(negative, k) / 255.,) * 2,
                           (0.5, 1.0, 1.0),
                           (1.0,) + (getattr(positive, k) / 255.,) * 2,]
                       for k in ['red', 'green', 'blue']})

bwr = _BipolarColormap('cbf.bwr', _BLUE, _VERMILION)
PRGn = _BipolarColormap('cbf.PRGn', _PURPLE, _GREEN)
### END colorblind_friendly.py ###

import matplotlib.pyplot as plt
from matplotlib.cm import Greys, bwr
from matplotlib import gridspec

import numpy as np
from kcsd import KCSD2D

import figure_properties

plt.rcParams.update({
    'xtick.minor.size': 10 - 6.18,
    'ytick.minor.size': 10 - 6.18,
})


xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
n_src_init = 1000
R_init = 1.
ext_x = 0.0
ext_y = 0.0
h = 50. # distance between the electrode plane and the CSD plane
conductivity = 1.0 # S/m

csd_at = np.mgrid[0.:1.:101j,
                  0.:1.:101j]
csd_x, csd_y = csd_at

D = 2 - 1.618
ele_x, ele_y = np.mgrid[0.5 - D: 0.5 + D: 3j,
                        0.5 - D: 0.5 + D: 3j]
ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T

pots = np.eye(9)

kcsd = KCSD2D(ele_pos, pots, h=h, sigma=conductivity,
              xmin=xmin, xmax=xmax,
              ymin=ymin, ymax=ymax,
              n_src_init=n_src_init,
              src_type='gauss',
              R_init=R_init)

csd = kcsd.values('CSD')

E = np.array([csd[:, :, i].flatten()
              for i in range(csd.shape[-1])])
VAR = np.dot(E.T, E)[np.eye(E.shape[1], dtype=bool)].reshape(int(np.sqrt(E.shape[1])), -1)



FIG_WIDTH = 17.0
FIG_HEIGHT = 6.0
FIG_WSPACE = 2.0
FIG_TOP = 0.9
FIG_BOTTOM = 0.15
FIG_LEFT = 0.125
FIG_RIGHT = 0.875


def maxabs(values):
    t_max = np.abs(values).max()
    if t_max == 0:
        return np.finfo(type(t_max)).eps

    return t_max


fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT),
                 constrained_layout=False)

gs_fig = gridspec.GridSpec(1, 2,
                           figure=fig,
                           wspace=2 * FIG_WSPACE / ((FIG_RIGHT - FIG_LEFT) * FIG_WIDTH - FIG_WSPACE),
                           left=FIG_LEFT,
                           right=FIG_RIGHT,
                           top=FIG_TOP,
                           bottom=FIG_BOTTOM)
gs1 = gridspec.GridSpecFromSubplotSpec(3, 4,
                                       subplot_spec=gs_fig[0],
                                       width_ratios=[1, 1, 1, 0.2],
                                       **{#'left': 0.1,
                                          #'right': 0.85,
                                          #'top': FIG_TOP,
                                          #'bottom': FIG_BOTTOM,
                                          'wspace': 0.2,
                                          'hspace': 0.2})

gs2 = gridspec.GridSpecFromSubplotSpec(3, 4,
                                       subplot_spec=gs_fig[1],
                                       width_ratios=[1, 1, 1, 0.2],
                                       **{#'left': 0.1,
                                          #'right': 0.85,
                                          #'top': FIG_TOP,
                                          #'bottom': FIG_BOTTOM,
                                          'wspace': 0.2,
                                          'hspace': 0.2})

t_max = maxabs(csd)
levels = np.linspace(-t_max, t_max, 256)
colorbar_ticks = np.linspace(-t_max, t_max, 3, endpoint=True)
ticks = np.linspace(0, 1, 5)
ticklabels = ['{:g}'.format(x) if i % 2 == 0 else ''
              for i, x in enumerate(ticks)]
ticks_minor = [x for x in np.linspace(0, 1, 21) if x not in ticks]

X = list(ele_x.flatten())
Y = list(ele_y.flatten())
for y in range(3):
    for x in range(3):
        ax = fig.add_subplot(gs1[2-y, x])
        i = y + x * 3
        CSD = csd[:, :, i]
        ax.set_aspect('equal')
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_xticks(ticks_minor, minor=True)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.set_yticks(ticks_minor, minor=True)
        im = ax.contourf(csd_x, csd_y, CSD, levels=levels, cmap=bwr)
        ax.scatter(X[:i] + X[i + 1:],
                   Y[:i] + Y[i + 1:],
                   color='k',
                   marker='x')
        ax.plot(X[i:i+1], Y[i:i+1],
                markeredgecolor='k',
                markerfacecolor='none',
                marker='o')

        if x == 0:
            ax.set_ylabel('Y (mm)')
            if y == 2:
                ax.text(-0.16,
                        1.16,
                        'A',
                        fontsize=20,
                        weight='bold',
                        transform=ax.transAxes)

        else:
            for tk in ax.get_yticklabels():
                tk.set_visible(False)

        if y == 0:
            ax.set_xlabel('X (mm)')

        else:
            for tk in ax.get_xticklabels():
                tk.set_visible(False)


fig.colorbar(im,
             cax=fig.add_subplot(gs1[:, 3]),
             orientation='vertical',
             format='%.2g',
             ticks=colorbar_ticks)



t_max = maxabs(VAR)

levels = np.linspace(0, t_max, 256)
colorbar_ticks = np.linspace(0, t_max, 2, endpoint=True)
ticklabels = ['{:g}'.format(x) for i, x in enumerate(ticks)]
ax = fig.add_subplot(gs2[:3, :3])

ax.set_aspect('equal')
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
ax.set_xticks(ticks_minor, minor=True)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
ax.set_yticks(ticks_minor, minor=True)
ax.set_ylabel('Y (mm)')
ax.set_xlabel('X (mm)')
im = ax.contourf(csd_x, csd_y, VAR, levels=levels, cmap=Greys)
ax.scatter(X,
           Y,
           color=VERMILION, #'#{:02X}{:02X}{:02X}'.format(213,  94,   0),
           marker='x')
ax.text(-0.05,
        1.05,
        'B',
        fontsize=20,
        weight='bold',
        transform=ax.transAxes)
fig.colorbar(im,
             cax=fig.add_subplot(gs2[:, 3]),
             orientation='vertical',
             #format='%.2g',
             ticks=colorbar_ticks)

fig.savefig('error_propagation.pdf')
