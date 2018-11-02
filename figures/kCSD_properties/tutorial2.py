import numpy as np
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

# True CSD_PROFILE
csd_at = np.mgrid[0.:1.:100j,
                  0.:1.:100j]
csd_x, csd_y = csd_at
CSD_PROFILE = CSD.gauss_2d_small
true_csd = CSD_PROFILE(csd_at, seed=15)

# Electrode positions
ele_x, ele_y = np.mgrid[0.05: 0.95: 10j,
                        0.05: 0.95: 10j]
ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T

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
k.cross_validate()

est_csd_post_cv = k.values('CSD')

fig = plt.figure(figsize=(20, 5))
ax = plt.subplot(141)
ax.set_aspect('equal')


nrm_est = est_csd_post_cv.reshape(est_csd_post_cv.size, 1) / np.max(est_csd_post_cv)
nrm_csd = true_csd.reshape(true_csd.size, 1) / np.max(true_csd) 
err = np.linalg.norm(nrm_csd - nrm_est, axis=1).reshape(true_csd.shape)
e_max = np.max(err)
levels = np.linspace(0, e_max, 16)
im = ax.contourf(csd_x, csd_y, err,
                 levels=levels, cmap=cm.Greys)
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_title('Error CSD', pad=20)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ticks = np.linspace(0.0, e_max, 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

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
ax.set_title('Interpolated potentials', pad=20)
ticks = np.linspace(-1 * v_max, v_max, 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

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
ax.set_title('Estimated CSD without CV', pad=20)
ticks = np.linspace(-1 * t_max, t_max, 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

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
ax.set_title('Estimated CSD with CV', pad=20)
ticks = np.linspace(-1 * t_max, t_max, 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

plt.show()
