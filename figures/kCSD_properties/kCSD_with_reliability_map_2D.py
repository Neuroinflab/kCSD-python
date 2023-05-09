"""
@author: mkowalska
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import range

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from matplotlib import gridspec
from kcsd import csd_profile as CSD
from kcsd import ValidateKCSD2D
from figure_properties import *


def set_axis(ax, letter=None):
    """
    Formats the plot's caption.

    Parameters
    ----------
    ax: Axes object.
    x: float
        X-position of caption.
    y: float
        Y-position of caption.
    letter: string
        Caption of the plot.
        Default: None.

    Returns
    -------
    ax: modyfied Axes object.
    """
    ax.text(-0.05, 1.05, letter, fontsize=20, weight="bold", transform=ax.transAxes)
    return ax


def make_reconstruction(
    KK,
    csd_profile,
    csd_seed,
    total_ele,
    ele_lims=None,
    noise=0,
    nr_broken_ele=None,
    Rs=None,
    lambdas=None,
    method="cross-validation",
):
    """
    Main method, makes the whole kCSD reconstruction.

    Parameters
    ----------
    KK: instance of the class
        Instance of class ValidateKCSD1D.
    csd_profile: function
        Function to produce csd profile.
    csd_seed: int
        Seed for random generator to choose random CSD profile.
    total_ele: int
        Number of electrodes.
    ele_lims: list
        Electrodes limits.
        Default: None.
    noise: float
        Determines the level of noise in the data.
        Default: 0.
    nr_broken_ele: int
        How many electrodes are broken (excluded from analysis)
        Default: None.
    Rs: numpy 1D array
        Basis source parameter for crossvalidation.
        Default: None.
    lambdas: numpy 1D array
        Regularization parameter for crossvalidation.
        Default: None.
    method: string
        Determines the method of regularization.
        Default: cross-validation.

    Returns
    -------
    k: instance of the class
        Instance of class KCSD1D.
    csd_at: numpy array
        Where to generate CSD.
    true_csd: numpy array
        CSD at csd_at positions.
    ele_pos: numpy array
        Electrodes positions.
    pots: numpy array
        Potentials measured (calculated) on electrodes.
    """
    csd_at, true_csd = KK.generate_csd(csd_profile, csd_seed)
    ele_pos, pots = KK.electrode_config(
        csd_profile, csd_seed, total_ele, ele_lims, KK.h, KK.sigma, noise, nr_broken_ele
    )
    k, est_csd = KK.do_kcsd(pots, ele_pos, method=method, Rs=Rs, lambdas=lambdas)
    return k, csd_at, true_csd, ele_pos, pots


def make_subplot(
    ax,
    val_type,
    xs,
    ys,
    values,
    cax,
    title=None,
    ele_pos=None,
    xlabel=False,
    ylabel=False,
    letter="",
    t_max=None,
    mask=False,
    level=False,
):
    if val_type == "csd":
        cmap = cm.bwr
    elif val_type == "pot":
        cmap = cm.PRGn
    else:
        cmap = cm.Greys
    ax.set_aspect("equal")
    if t_max is None:
        t_max = np.max(np.abs(values))
    if level is not False:
        levels = level
    else:
        levels = np.linspace(-t_max, t_max, 32)
    if val_type == "pot":
        X, Y, Z = grid(ele_pos[:, 0], ele_pos[:, 1], values)
        im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=1)
    else:
        im = ax.contourf(
            xs, ys, values, levels=levels, cmap=cmap, alpha=1, extent=(0, 0.5, 0, 0.5)
        )
    if mask is not False:
        CS = ax.contour(xs, ys, mask, cmap="Greys")
        ax.clabel(CS, inline=1, fmt="%1.2f", fontsize=9)  # label every second level
    if val_type == "pot":
        ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c="k")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if xlabel:
        ax.set_xlabel("X (mm)")
    if ylabel:
        ax.set_ylabel("Y (mm)")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ticks = np.linspace(-t_max, t_max, 3, endpoint=True)
    plt.colorbar(im, cax=cax, orientation="horizontal", format="%.2f", ticks=ticks)
    set_axis(ax, letter=letter)
    return ax, cax


def grid(x, y, z, resX=100, resY=100):
    """
    Convert 3 column data to matplotlib grid

    Parameters
    ----------
    x
    y
    z

    Returns
    -------
    xi
    yi
    zi
    """
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xi, yi = np.mgrid[
        min(x) : max(x) : complex(0, resX), min(y) : max(y) : complex(0, resY)
    ]
    zi = griddata((x, y), z, (xi, yi), method="linear")
    return xi, yi, zi


def generate_figure(k, true_csd, ele_pos, pots, mask=False):
    csd_at = np.mgrid[0.0:1.0:100j, 0.0:1.0:100j]
    csd_x, csd_y = csd_at
    plt.figure(figsize=(17, 6))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 0.04], width_ratios=[1] * 4)
    #    gs.update(top=.95, bottom=0.53)
    ax = plt.subplot(gs[0, 0])
    cax = plt.subplot(gs[1, 0])
    make_subplot(
        ax,
        "csd",
        csd_x,
        csd_y,
        true_csd,
        cax=cax,
        ele_pos=ele_pos,
        title="True CSD",
        xlabel=True,
        ylabel=True,
        letter="A",
        t_max=np.max(abs(true_csd)),
    )
    ax = plt.subplot(gs[0, 1])
    cax = plt.subplot(gs[1, 1])
    make_subplot(
        ax,
        "pot",
        ele_pos[:, 0],
        ele_pos[:, 1],
        pots,
        cax=cax,
        ele_pos=ele_pos,
        title="Potentials",
        xlabel=True,
        letter="B",
        t_max=np.max(abs(pots)),
    )
    ax = plt.subplot(gs[0, 2])
    cax = plt.subplot(gs[1, 2])
    make_subplot(
        ax,
        "csd",
        k.estm_x,
        k.estm_y,
        k.values("CSD")[:, :, 0],
        cax=cax,
        ele_pos=ele_pos,
        title="kCSD with Reliability Map",
        xlabel=True,
        letter="C",
        t_max=np.max(abs(true_csd)),
        mask=mask,
    )
    ax = plt.subplot(gs[0, 3])
    cax = plt.subplot(gs[1, 3])
    make_subplot(
        ax,
        "diff",
        csd_x,
        csd_y,
        abs(true_csd - k.values("CSD")[:, :, 0]),
        cax=cax,
        ele_pos=ele_pos,
        title="|True CSD - kCSD|",
        xlabel=True,
        letter="D",
        t_max=np.max(abs(true_csd - k.values("CSD")[:, :, 0])),
        level=np.linspace(0, np.max(abs(true_csd - k.values("CSD")[:, :, 0])), 16),
    )
    plt.savefig("kCSD_with_reliability_map_2D.png", dpi=300)
    plt.show()


def matrix_symmetrization(point_error):
    r1 = np.rot90(point_error, k=1, axes=(1, 2))
    r2 = np.rot90(point_error, k=2, axes=(1, 2))
    r3 = np.rot90(point_error, k=3, axes=(1, 2))
    arr_lr = np.zeros(point_error.shape)
    for i in range(len(point_error)):
        arr_lr[i] = np.flipud(point_error[i, :, :])
    r11 = np.rot90(arr_lr, k=1, axes=(1, 2))
    r12 = np.rot90(arr_lr, k=2, axes=(1, 2))
    r13 = np.rot90(arr_lr, k=3, axes=(1, 2))
    symm_array = np.concatenate((point_error, r1, r2, r3, arr_lr, r11, r12, r13))
    return symm_array


if __name__ == "__main__":
    CSD_PROFILE = CSD.gauss_2d_large
    CSD_SEED = 16
    ELE_LIMS = [0.05, 0.95]  # range of electrodes space
    method = "cross-validation"
    Rs = np.arange(0.2, 0.5, 0.1)
    lambdas = None
    noise = 0

    KK = ValidateKCSD2D(
        CSD_SEED,
        h=50.0,
        sigma=1.0,
        n_src_init=400,
        est_xres=0.01,
        est_yres=0.01,
        ele_lims=ELE_LIMS,
    )
    k, csd_at, true_csd, ele_pos, pots = make_reconstruction(
        KK,
        CSD_PROFILE,
        CSD_SEED,
        total_ele=100,
        noise=noise,
        Rs=Rs,
        lambdas=lambdas,
        method=method,
    )
    error_l = np.load("error_maps_2D/point_error_large_100_all_ele.npy")
    error_s = np.load("error_maps_2D/point_error_small_100_all_ele.npy")
    error_all = np.concatenate((error_l, error_s))
    symm_array_all = matrix_symmetrization(error_all)
    generate_figure(k, true_csd, ele_pos, pots, mask=np.mean(symm_array_all, axis=0))
