import numpy as np
import os
from kcsd import csd_profile as CSD
from kcsd import KCSD2D
from scipy.integrate import simps
from scipy.interpolate import griddata
from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def integrate_2d(csd_at, true_csd, ele_pos, h, csd_lims):
    csd_x, csd_y = csd_at
    xlin = csd_lims[0]
    ylin = csd_lims[1]
    Ny = ylin.shape[0]
    m = np.sqrt((ele_pos[0] - csd_x)**2 + (ele_pos[1] - csd_y)**2)
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
    xi, yi = np.mgrid[min(x):max(x):np.complex(0, 100),
                      min(y):max(y):np.complex(0, 100)]
    zi = griddata((x, y), z, (xi, yi), method='linear')
    return xi, yi, zi


def do_kcsd(CSD_PROFILE, csd_seed, prefix, missing_ele):
    # True CSD_PROFILE
    csd_at = np.mgrid[0.:1.:100j,
                      0.:1.:100j]
    csd_x, csd_y = csd_at
    true_csd = CSD_PROFILE(csd_at, seed=csd_seed)

    # Electrode positions
    ele_x, ele_y = np.mgrid[0.05: 0.95: 10j,
                            0.05: 0.95: 10j]
    ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T

    #Remove some electrodes
    remove_num = missing_ele
    rstate = np.random.RandomState(42)  # just a random seed
    rmv = rstate.choice(ele_pos.shape[0], remove_num, replace=False)
    ele_pos = np.delete(ele_pos, rmv, 0)
    
    # Potentials generated 
    pots = np.zeros(ele_pos.shape[0])
    xlin = csd_at[0, :, 0]
    ylin = csd_at[1, 0, :]
    h = 50.
    sigma = 0.3
    for ii in range(ele_pos.shape[0]):
        pots[ii] = integrate_2d(csd_at, true_csd,
                                [ele_pos[ii][0], ele_pos[ii][1]], h,
                                [xlin, ylin])
    pots /= 2 * np.pi * sigma
    pot_X, pot_Y, pot_Z = grid(ele_pos[:, 0], ele_pos[:, 1], pots)
    pots = pots.reshape((len(ele_pos), 1))

    # KCSD2D
    k = KCSD2D(ele_pos, pots, h=h, sigma=sigma,
               xmin=0.0, xmax=1.0,
               ymin=0.0, ymax=1.0,
               gdx=0.01, gdy=0.01,
               R_init=0.1, n_src_init=1000,
               src_type='gauss')   # rest of the parameters are set at default
    est_csd_pre_cv = k.values('CSD')
    R_range = np.linspace(0.1, 1.0, 10)
    #R_range = np.linspace(0.03, 0.12, 10)
    #R_range = np.linspace(0.1, 1.0, 10)
    k.cross_validate(Rs=R_range)
    #k.cross_validate()
    #k.cross_validate(lambdas=None, Rs=np.array(0.08).reshape(1))
    est_csd_post_cv = k.values('CSD')

    fig = plt.figure(figsize=(20, 5))
    ax = plt.subplot(141)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(true_csd))
    levels = np.linspace(-1 * t_max, t_max, 16)
    im = ax.contourf(csd_x, csd_y, true_csd,
                     levels=levels, cmap=cm.bwr)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title('True CSD')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ticks = np.linspace(-1 * t_max, t_max, 5, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)

    ax = plt.subplot(142)
    ax.set_aspect('equal')
    v_max = np.max(np.abs(pots))
    levels_pot = np.linspace(-1 * v_max, v_max, 16)
    im = ax.contourf(pot_X, pot_Y, pot_Z,
                     levels=levels_pot, cmap=cm.PRGn) 
    ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_title('Interpolated potentials')
    ticks = np.linspace(-1 * v_max, v_max, 5, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)

    ax = plt.subplot(143)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(est_csd_pre_cv[:, :, 0]))
    levels_kcsd = np.linspace(-1 * t_max, t_max, 16, endpoint=True)
    im = ax.contourf(k.estm_x, k.estm_y, est_csd_pre_cv[:, :, 0],
                     levels=levels_kcsd, cmap=cm.bwr) 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_title('Estimated CSD without CV')
    ticks = np.linspace(-1 * t_max, t_max, 5, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)

    ax = plt.subplot(144)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(est_csd_post_cv[:, :, 0]))
    levels_kcsd = np.linspace(-1 * t_max, t_max, 16, endpoint=True)
    im = ax.contourf(k.estm_x, k.estm_y, est_csd_post_cv[:, :, 0],
                     levels=levels_kcsd, cmap=cm.bwr) 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_title('Estimated CSD with CV')
    ticks = np.linspace(-1 * t_max, t_max, 5, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)
    plt.savefig(os.path.join(prefix, str(csd_seed)+'.pdf'))
    plt.close()
    #plt.show()
    np.savez(os.path.join(prefix, str(csd_seed)+'.npz'),
             true_csd=true_csd, pots=pots, post_cv=est_csd_post_cv, R=k.R)

if __name__ == '__main__':
    CSD_PROFILE =  CSD.gauss_2d_large
    prefix = '/home/chaitanya/kCSD-python/figures/kCSD_properties/large_srcs_all_ele'
    for csd_seed in range(100):
        do_kcsd(CSD_PROFILE, csd_seed, prefix, missing_ele=0)
        print("Done ", csd_seed)
    CSD_PROFILE =  CSD.gauss_2d_small
    prefix = '/home/chaitanya/kCSD-python/figures/kCSD_properties/small_srcs_all_ele'
    for csd_seed in range(100):
        do_kcsd(CSD_PROFILE, csd_seed, prefix, missing_ele=0)
        print("Done ", csd_seed)
