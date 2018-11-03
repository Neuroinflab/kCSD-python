import os
import numpy as np
from kcsd import csd_profile as CSD
from kcsd import KCSD2D
from scipy.integrate import simps
from scipy.interpolate import griddata
from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def fetch_folder(csd_type='small'):
    if csd_type == 'small':
        all_folder = 'small_srcs_all_ele'
        all_minus_5 = 'small_srcs_minus_5'
        all_minus_10 = 'small_srcs_minus_10'
        all_minus_20 = 'small_srcs_minus_20'
    else:
        all_folder = 'large_srcs_all_ele'
        all_minus_5 = 'large_srcs_minus_5'
        all_minus_10 = 'large_srcs_minus_10'
        all_minus_20 = 'large_srcs_minus_20'

    fldrs = [all_folder, all_minus_5, all_minus_10, all_minus_20]
    fldrs = [os.path.join(os.path.abspath('.'), fs) for fs in fldrs]
    return fldrs

def load_files(folderpaths, seeds=[15]):
    list_dict = []
    for folderpath in folderpaths:
        files = [os.path.join(folderpath, str(seed)+'.npz') for seed in seeds]
        eval_dict = {}
        for f in files:
            csd_seed = os.path.basename(f)[:-4]
            eval_dict[int(csd_seed)]  = np.load(f)
        list_dict.append(eval_dict)
    return list_dict

def grid(x, y, z):
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xi, yi = np.mgrid[min(x):max(x):np.complex(0, 100),
                      min(y):max(y):np.complex(0, 100)]
    zi = griddata((x, y), z, (xi, yi), method='linear')
    return xi, yi, zi

def electrode_positions(missing_ele=0):
    # Electrode positions
    ele_x, ele_y = np.mgrid[0.05: 0.95: 10j,
                            0.05: 0.95: 10j]
    ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T
    #Remove some electrodes
    remove_num = missing_ele
    rstate = np.random.RandomState(42)  # just a random seed
    rmv = rstate.choice(ele_pos.shape[0], remove_num, replace=False)
    ele_pos = np.delete(ele_pos, rmv, 0)
    return ele_pos


def point_errors(true_csd, est_csd):
    nrm_est = est_csd.reshape(est_csd.size, 1) / np.max(np.abs(est_csd))
    nrm_csd = true_csd.reshape(true_csd.size, 1) / np.max(np.abs(true_csd))
    err = np.linalg.norm(nrm_csd - nrm_est, axis=1).reshape(true_csd.shape)
    return err


def eval_errors(list_dicts, seed_list):
    list_errs = []
    list_levels = []
    list_emax = []
    for ii, ele_data in enumerate(list_dicts):
        errs = []
        for seed in seed_list:
            est_csd = ele_data[seed]['post_cv']
            true_csd = ele_data[seed]['true_csd']
            errs.append(point_errors(true_csd, est_csd))
        err = sum(errs) / len(errs)
        list_errs.append(err) # average error
        list_emax.append(np.max(err))
        list_levels.append(np.linspace(0, np.max(err), 16))
    return list_errs, list_emax, list_levels


fldrs = fetch_folder(csd_type='large')
seed_list = range(60)
list_dicts = load_files(fldrs, seed_list)
errs, emaxs, levels = eval_errors(list_dicts, seed_list)

csd_at = np.mgrid[0.:1.:100j,
                  0.:1.:100j]
csd_x, csd_y = csd_at


fig = plt.figure(figsize=(20, 5))
ax = plt.subplot(141)
ax.set_aspect('equal')
ele_pos = electrode_positions(missing_ele=0)
im = ax.contourf(csd_x, csd_y, errs[0],
                 levels=levels[0], cmap=cm.Greys)
ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c='k')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_title('Error CSD')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ticks = np.linspace(0.0, emaxs[0], 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

ax = plt.subplot(142)
ax.set_aspect('equal')
ele_pos = electrode_positions(missing_ele=5)
im = ax.contourf(csd_x, csd_y, errs[1],
                 levels=levels[1], cmap=cm.Greys)
ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c='k')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_title('Error CSD, 5 broken')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ticks = np.linspace(0.0, emaxs[1], 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

ax = plt.subplot(143)
ax.set_aspect('equal')
ele_pos = electrode_positions(missing_ele=10)
im = ax.contourf(csd_x, csd_y, errs[2],
                 levels=levels[2], cmap=cm.Greys)
ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c='k')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_title('Error CSD, 10 broken')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ticks = np.linspace(0.0, emaxs[2], 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

ax = plt.subplot(144)
ax.set_aspect('equal')
ele_pos = electrode_positions(missing_ele=20)
im = ax.contourf(csd_x, csd_y, errs[3],
                 levels=levels[3], cmap=cm.Greys)
ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10, c='k')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_title('Error CSD, 20 broken')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ticks = np.linspace(0.0, emaxs[3], 5, endpoint=True)
plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)

plt.show()
