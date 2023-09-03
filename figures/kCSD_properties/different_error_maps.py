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
    xi, yi = np.mgrid[min(x):max(x):complex(0, 100),
                      min(y):max(y):complex(0, 100)]
    zi = griddata((x, y), z, (xi, yi), method='linear')
    return xi, yi, zi


def point_errors(true_csd, est_csd):
    true_csd_r = true_csd.reshape(true_csd.size, 1)
    est_csd_r = est_csd.reshape(est_csd.size, 1)
    epsilon = np.linalg.norm(true_csd_r)/np.max(abs(true_csd_r))
    err_r = abs(est_csd_r/(np.linalg.norm(est_csd_r)) -
                true_csd_r/(np.linalg.norm(true_csd_r)))
    err_r *= epsilon
    err = err_r.reshape(true_csd.shape)
    return err

def point_errors2(true_csd, est_csd):
    epsilon = np.max(abs(true_csd.reshape(true_csd.size, 1)))
    err2 = abs(true_csd.reshape(true_csd.size, 1) -
                          est_csd.reshape(est_csd.size, 1))
    err2 /= abs(true_csd.reshape(true_csd.size, 1)) + \
    epsilon #*np.max(abs(true_csd.reshape(true_csd.size, 1)))
    err = err2.reshape(true_csd.shape)
    return err


def sigmoid_mean(error):
    sig_error = 2./(1. + np.exp(-error)) - 1.
    return sig_error


def point_errors_Ch(true_csd, est_csd):
    nrm_est = est_csd.reshape(est_csd.size, 1) / np.max(np.abs(est_csd))
    nrm_csd = true_csd.reshape(true_csd.size, 1) / np.max(np.abs(true_csd))
    err = abs(nrm_csd - nrm_est).reshape(true_csd.shape)
    return err


def calculate_rdm(true_csd, est_csd):
    rdm = abs(est_csd.reshape(est_csd.size, 1)/(np.linalg.norm(est_csd.reshape(est_csd.size, 1))) -
              true_csd.reshape(true_csd.size, 1)/(np.linalg.norm(true_csd.reshape(true_csd.size, 1))))
    rdm *= np.linalg.norm(true_csd.reshape(true_csd.size, 1))/np.max(abs(true_csd.reshape(true_csd.size, 1)))
    return rdm.reshape(true_csd.shape)


def calculate_mag(true_csd, est_csd):
    epsilon = np.max(abs(true_csd.reshape(true_csd.size, 1)))
    mag = abs(est_csd.reshape(est_csd.size, 1))/(abs(true_csd.reshape(true_csd.size, 1)) + epsilon)
    return mag.reshape(true_csd.shape)


def do_kcsd(CSD_PROFILE, data, csd_seed, prefix, missing_ele):
    # True CSD_PROFILE
    csd_at = np.mgrid[0.:1.:100j,
                      0.:1.:100j]
    csd_x, csd_y = csd_at
    true_csd = data['true_csd']

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
    pots = data['pots']
    h = 50.
    sigma = 0.3
    pot_X, pot_Y, pot_Z = grid(ele_pos[:, 0], ele_pos[:, 1], pots)

    # KCSD2D
    k = KCSD2D(ele_pos, pots, h=h, sigma=sigma,
               xmin=0.0, xmax=1.0,
               ymin=0.0, ymax=1.0,
               gdx=0.01, gdy=0.01,
               R_init=0.1, n_src_init=1000,
               src_type='gauss')   # rest of the parameters are set at default
    est_csd_pre_cv = k.values('CSD')
    est_csd_post_cv = data['post_cv']

    fig = plt.figure(figsize=(20, 12))
    ax = plt.subplot(241)
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

    ax = plt.subplot(242)
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

    ax = plt.subplot(243)
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

    ax = plt.subplot(244)
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

    ax = plt.subplot(245)
    error1 = point_errors(true_csd, est_csd_post_cv)
    print(error1.shape)
    ax.set_aspect('equal')
    t_max = np.max(abs(error1))
    levels_kcsd = np.linspace(0, t_max, 16, endpoint=True)
    im = ax.contourf(k.estm_x, k.estm_y, error1,
                     levels=levels_kcsd, cmap=cm.Greys) 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title('Sigmoid error')
    ticks = np.linspace(0, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)

    ax = plt.subplot(246)
    error2 = point_errors_Ch(true_csd, est_csd_post_cv)
    ax.set_aspect('equal')
    t_max = np.max(abs(error2))
    levels_kcsd = np.linspace(0, t_max, 16, endpoint=True)
    im = ax.contourf(k.estm_x, k.estm_y, error2,
                     levels=levels_kcsd, cmap=cm.Greys) 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_title('Normalized difference')
    ticks = np.linspace(0, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)
    
    ax = plt.subplot(247)
    error3 = calculate_rdm(true_csd, est_csd_post_cv[:, :, 0])
    print(error3.shape)
    ax.set_aspect('equal')
    t_max = np.max(abs(error3))
    levels_kcsd = np.linspace(0, t_max, 16, endpoint=True)
    im = ax.contourf(k.estm_x, k.estm_y, error3,
                     levels=levels_kcsd, cmap=cm.Greys) 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_title('Relative difference measure')
    ticks = np.linspace(0, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)

    ax = plt.subplot(248)
    error4 = calculate_mag(true_csd, est_csd_post_cv[:, :, 0])
    print(error4.shape)
    ax.set_aspect('equal')
    t_max = np.max(abs(error4))
    levels_kcsd = np.linspace(0, t_max, 16, endpoint=True)
    im = ax.contourf(k.estm_x, k.estm_y, error4,
                     levels=levels_kcsd, cmap=cm.Greys) 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('X [mm]')
    ax.set_title('Magnitude ratio')
    ticks = np.linspace(0, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks, pad=0.25)

    plt.savefig(os.path.join(prefix, str(csd_seed)+'.pdf'))
    plt.close()
    #plt.show()
    np.savez(os.path.join(prefix, str(csd_seed)+'.npz'),
             true_csd=true_csd, pots=pots, post_cv=est_csd_post_cv, R=k.R)

if __name__ == '__main__':
    CSD_PROFILE =  CSD.gauss_2d_large #CSD.gauss_2d_small #
    
    prefix = '/home/mkowalska/Marta/kCSD-python/figures/kCSD_properties/small_srcs_all_ele'
    for csd_seed in range(100):
        data = np.load(prefix + '/' + str(csd_seed) + '.npz')
        do_kcsd(CSD_PROFILE, data, csd_seed, prefix, missing_ele=0)
        print("Done ", csd_seed)
