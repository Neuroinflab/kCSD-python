import numpy as np
import os
from kcsd import csd_profile as CSD
from kcsd import KCSD2D
from scipy.integrate import simps
from scipy.interpolate import griddata

from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def integrate_2d(csd_at, true_csd, ele_pos, h, csd_lims):
    csd_x, csd_y = csd_at
    xlin = csd_lims[0]
    ylin = csd_lims[1]
    Ny = ylin.shape[0]
    m = np.sqrt((ele_pos[0] - csd_x) ** 2 + (ele_pos[1] - csd_y) ** 2)
    m[m < 0.0000001] = 0.0000001
    y = np.arcsinh(2 * h / m) * true_csd
    integral_1D = np.zeros(Ny)
    for i in range(Ny):
        integral_1D[i] = simps(y[:, i], ylin)
    integral = simps(integral_1D, xlin)
    return integral


def grid(x, y, z):
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xi, yi = np.mgrid[
        min(x) : max(x) : complex(0, 100), min(y) : max(y) : complex(0, 100)
    ]
    zi = griddata((x, y), z, (xi, yi), method="linear")
    return xi, yi, zi


def set_axis(ax, letter=None):
    ax.text(-0.05, 1.05, letter, fontsize=20, weight="bold", transform=ax.transAxes)
    return ax


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
):
    if val_type == "csd":
        cmap = cm.bwr
    elif val_type == "pot":
        cmap = cm.PRGn
    ax.set_aspect("equal")
    if t_max is None:
        t_max = np.max(np.abs(values))
    levels = np.linspace(-1 * t_max, t_max, 32)
    im = ax.contourf(xs, ys, values, levels=levels, cmap=cmap)
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
    ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    plt.colorbar(im, cax=cax, orientation="horizontal", format="%.2f", ticks=ticks)
    set_axis(ax, letter=letter)
    return ax, cax


def do_kcsd(CSD_PROFILE, csd_seed):
    if CSD_PROFILE.__name__ == "gauss_2d_small":
        R_init = 1.0
        R_final = 0.1
    elif CSD_PROFILE.__name__ == "gauss_2d_large":
        R_init = 0.1
        R_final = 0.3
    # True CSD_PROFILE
    csd_at = np.mgrid[0.0:1.0:100j, 0.0:1.0:100j]
    csd_x, csd_y = csd_at
    # Small source
    true_csd = CSD_PROFILE(csd_at, seed=csd_seed)
    # Electrode positions
    ele_x, ele_y = np.mgrid[0.05:0.95:10j, 0.05:0.95:10j]
    ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T

    # Potentials generated
    pots = np.zeros(ele_pos.shape[0])
    xlin = csd_at[0, :, 0]
    ylin = csd_at[1, 0, :]
    h = 50.0
    sigma = 0.3
    for ii in range(ele_pos.shape[0]):
        pots[ii] = integrate_2d(
            csd_at, true_csd, [ele_pos[ii][0], ele_pos[ii][1]], h, [xlin, ylin]
        )
    pots /= 2 * np.pi * sigma
    pot_X, pot_Y, pot_Z = grid(ele_pos[:, 0], ele_pos[:, 1], pots)
    pots = pots.reshape((len(ele_pos), 1))
    # KCSD2D
    k = KCSD2D(
        ele_pos,
        pots,
        h=h,
        sigma=sigma,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        gdx=0.01,
        gdy=0.01,
        R_init=R_init,
        n_src_init=1000,
        src_type="gauss",
    )  # rest of the parameters are set at default
    est_csd_pre_cv = k.values("CSD")
    # k.cross_validate(Rs=np.linspace(0.03, 0.12, 10))
    # k.cross_validate()
    #    k.cross_validate(lambdas=None, Rs=np.array(R_final).reshape(1))
    k.cross_validate(lambdas=None, Rs=np.linspace(0.05, 1.0, 20))
    est_csd_post_cv = k.values("CSD")
    return (
        csd_x,
        csd_y,
        true_csd,
        ele_pos,
        pot_X,
        pot_Y,
        pot_Z,
        k,
        est_csd_pre_cv,
        est_csd_post_cv,
    )


def generate_figure(small_seed, large_seed):
    fig = plt.figure(figsize=(20, 12))
    (
        csd_x,
        csd_y,
        true_csd,
        ele_pos,
        pot_X,
        pot_Y,
        pot_Z,
        k,
        est_csd_pre_cv,
        est_csd_post_cv,
    ) = do_kcsd(CSD.gauss_2d_small, small_seed)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 0.04], width_ratios=[1] * 4)
    gs.update(top=0.95, bottom=0.53)
    ax = plt.subplot(gs[0, 0])
    cax = plt.subplot(gs[1, 0])
    t_max_1 = 0.50
    make_subplot(
        ax,
        "csd",
        csd_x,
        csd_y,
        true_csd,
        cax,
        "True CSD",
        xlabel=True,
        ylabel=True,
        letter="A",
        t_max=t_max_1,
    )
    ax = plt.subplot(gs[0, 1])
    cax = plt.subplot(gs[1, 1])
    make_subplot(
        ax,
        "pot",
        pot_X,
        pot_Y,
        pot_Z,
        cax,
        "Interpolated potentials",
        xlabel=True,
        ele_pos=ele_pos,
        letter="B",
    )
    ax = plt.subplot(gs[0, 2])
    cax = plt.subplot(gs[1, 2])
    make_subplot(
        ax,
        "csd",
        k.estm_x,
        k.estm_y,
        est_csd_pre_cv[:, :, 0],
        cax,
        "Estimated CSD pre CV",
        xlabel=True,
        letter="C",
        t_max=t_max_1,
    )
    ax = plt.subplot(gs[0, 3])
    cax = plt.subplot(gs[1, 3])
    make_subplot(
        ax,
        "csd",
        k.estm_x,
        k.estm_y,
        est_csd_post_cv[:, :, 0],
        cax,
        "Estimated CSD post CV",
        xlabel=True,
        letter="D",
        t_max=t_max_1,
    )

    (
        csd_x,
        csd_y,
        true_csd,
        ele_pos,
        pot_X,
        pot_Y,
        pot_Z,
        k,
        est_csd_pre_cv,
        est_csd_post_cv,
    ) = do_kcsd(CSD.gauss_2d_large, large_seed)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 0.04], width_ratios=[1] * 4)
    gs.update(top=0.47, bottom=0.05)
    ax = plt.subplot(gs[0, 0])
    cax = plt.subplot(gs[1, 0])
    t_max_2 = 0.52
    make_subplot(
        ax,
        "csd",
        csd_x,
        csd_y,
        true_csd,
        cax,
        ylabel=True,
        xlabel=True,
        letter="E",
        t_max=t_max_2,
    )
    ax = plt.subplot(gs[0, 1])
    cax = plt.subplot(gs[1, 1])
    make_subplot(
        ax, "pot", pot_X, pot_Y, pot_Z, cax, xlabel=True, ele_pos=ele_pos, letter="F"
    )
    ax = plt.subplot(gs[0, 2])
    cax = plt.subplot(gs[1, 2])
    make_subplot(
        ax,
        "csd",
        k.estm_x,
        k.estm_y,
        est_csd_pre_cv[:, :, 0],
        cax,
        xlabel=True,
        letter="G",
        t_max=t_max_2,
    )
    ax = plt.subplot(gs[0, 3])
    cax = plt.subplot(gs[1, 3])
    make_subplot(
        ax,
        "csd",
        k.estm_x,
        k.estm_y,
        est_csd_post_cv[:, :, 0],
        cax,
        xlabel=True,
        letter="H",
        t_max=t_max_2,
    )
    # plt.tight_layout()
    plt.savefig("tutorial_basic_function.png", dpi=300)
    # plt.close()
    # plt.show()
    # np.savez(os.path.join(prefix, str(csd_seed)+'.npz'),
    #         true_csd=true_csd, pots=pots, post_cv=est_csd_post_cv, R=k.R)


if __name__ == "__main__":
    small_seed = 15
    large_seed = 6
    generate_figure(small_seed, large_seed)
