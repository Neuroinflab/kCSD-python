#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mbejtka, mczerwinski, dwojcik
"""
import scipy.spatial
import numpy as np
from numpy.linalg import LinAlgError
import h5py as h5
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp2d, interp1d
from scipy.signal import butter, filtfilt
from kcsd import KCSD2D
# from figure_properties import *


def set_axis(ax, letter=None):
    ax.text(
        -0.05,
        1.07,
        letter,
        fontsize=20,
        weight='bold',
        transform=ax.transAxes)
    return ax


def fetch_mid_pts(h, pop_name):
    """gets the mid points from a file, of a particular population name"""
    all_pts = h['/data/static/morphology/' + pop_name]
    x = (all_pts['x0'] + all_pts['x1']) / 2.
    y = (all_pts['y0'] + all_pts['y1']) / 2.
    z = (all_pts['z0'] + all_pts['z1']) / 2.
    x = x.reshape(x.size, 1)
    y = y.reshape(y.size, 1)
    z = z.reshape(z.size, 1)
    return np.hstack((x, y, z))


def find_map_idx(h, pop_name, field_name):
    """Find the corresponding, locations of morphology in field"""
    mor = h['/data/static/morphology/' + pop_name]
    i_data = h['/data/uniform/' + pop_name + '/' + field_name]
    mor_names = mor.dims[0].values()[0]  # entry in /map
    i_names = i_data.dims[0].values()[0]  # entry in /map
    if np.array_equal(i_names, mor_names):
        idx = range(i_names.shape[0])
    else:
        idx = [np.where(i_names.value() == entry)[0][0] for entry in mor_names]
    return idx


def inv_distance(src_pos, ele_pos):
    """computes the inverse distance between src_pos and ele_pos"""
    dist_matrix = np.zeros((src_pos.shape[0], ele_pos.shape[0]))
    for ii, electrode in enumerate(ele_pos):
        dist_matrix[:, ii] = scipy.spatial.distance.cdist(
            src_pos, electrode.reshape(1, 3)).flatten()
    dist_matrix = 1 / dist_matrix  # inverse distance matrix
    return dist_matrix


def pot_vs_time(h, pop_name, field_name, src_pos, ele_pos):
    """returns potentials, at ele_pos, due to src_pos, over time"""
    idx = find_map_idx(h, pop_name, field_name)
    src_time = h['/data/uniform/' + pop_name + '/' + field_name][()]
    src_time = src_time[idx]  # Order according to correct indices
    ele_src = inv_distance(src_pos, ele_pos).T
    return np.dot(ele_src, src_time) * (1 / (4 * np.pi * 1))


def get_all_src_pos(h, pop_names, total_cmpts):
    """Function to compute the positions for a list of populations"""
    all_srcs = np.zeros((int(np.sum(total_cmpts)), 3))
    for jj, pop_name in enumerate(pop_names):
        all_srcs[np.sum(total_cmpts[:jj], dtype=int):
                 np.sum(total_cmpts[:jj + 1], dtype=int), :] =\
                 fetch_mid_pts(h, pop_name)
                 # total_cmpts[:jj + 1]), :] = fetch_mid_pts(h, pop_name)
    return all_srcs


def get_extracellular(h, pop_names, time_pts, ele_pos):
    """Fuction to obtain the extracellular potentials at some time points and
    electrode positions"""
    num_ele = ele_pos.shape[0]
    pot_sum = np.zeros((num_ele, time_pts))
    for pop_name in pop_names:
        src_pos = fetch_mid_pts(h, pop_name)
        pot_sum += pot_vs_time(h, pop_name, 'i', src_pos, ele_pos)
        print('Done extracellular pots for pop_name', pop_name)
    return pot_sum


def plot_morp_ele(ax1, src_pos, ele_pos, pot, time_pt):
    """Plots the morphology midpoints and the electrode positions"""
    ymin = -2000  # -2150
    ymax = 500  # 550
    xmin = -400  # -450
    xmax = 400  # 450

    ax = plt.subplot(111, aspect='equal')
    plt.scatter(src_pos[:, 0], src_pos[:, 1],
                marker='.', alpha=0.7, color='k', s=0.7)
    plt.scatter(ele_pos[:, 0], ele_pos[:, 1],
                marker='x', alpha=0.8, color='r', s=10)
    # for tx in range(len(ele_pos[:,0])):
    #    plt.text(ele_pos[tx, 0], ele_pos[tx, 1], str(tx))
    try:
        ele_1 = 152
        ele_2 = 148
        plt.scatter(ele_pos[ele_1, 0], ele_pos[ele_1, 1],
                    marker='s', color='m', s=30.)
        plt.scatter(ele_pos[ele_2, 0], ele_pos[ele_2, 1],
                    marker='s', color='b', s=30.)
    except IndexError:
        ele_1 = 10
        ele_2 = 15
        plt.scatter(ele_pos[ele_1, 0], ele_pos[ele_1, 1],
                    marker='s', color='m', s=30.)
        plt.scatter(ele_pos[ele_2, 0], ele_pos[ele_2, 1],
                    marker='s', color='b', s=30.)

    plt.xlabel('X ($\mu$m)', fontsize=fsize)
    plt.ylabel('Y ($\mu$m)', fontsize=fsize)
    plt.title('Morphology and electrodes')
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.xlim(xmin=xmin, xmax=xmax)

    cbaxes = inset_axes(ax,
                        height="17%",  # height : 50%
                        loc=4, borderpad=2.2)

    plt.plot(np.arange(6000), pot[ele_1, :], color='m', linewidth=0.7)
    plt.plot(np.arange(6000), pot[ele_2, :], color='b', linewidth=0.7)

    dummy_line = np.arange(-0.5, 0.5, 0.1)
    plt.plot(np.zeros_like(dummy_line) + time_pt,
             dummy_line, color='black', linewidth=1)
    plt.xlim((2750, 3500))  # 4250))
    plt.xticks(np.arange(2750, 3750, 250), np.arange(275, 375, 25))
    plt.ylim((-0.2, 0.12))
    plt.yticks(np.arange(-0.2, 0.1, 0.1), np.arange(-0.2, 0.1, 0.1))
    ax = plt.gca()
    ax.get_yaxis().tick_right()  # set_tick_params(direction='in')
    ax.xaxis.set_tick_params(labelsize=fsize)
    ax.yaxis.set_tick_params(labelsize=fsize)

    return


def plot_extracellular(ax, lfp, ele_pos, num_x, num_y, time_pt, name,
                       letter='', title='Estimated LFP'):
    """Plots the extracellular potentials at a given potentials"""
    ymin = -2000  # -2150
    ymax = 500  # 550
    xmin = -400  # -450
    xmax = 400  # 450

    # lfp *= 1000.

    xaxis = np.arange(np.min(ele_pos[:, 0]), np.max(ele_pos[:, 0]) + 1, 10)
    yaxis = np.arange(np.min(ele_pos[:, 1]), np.max(ele_pos[:, 1]) + 1, 10)

    xx, yy = np.meshgrid(xaxis, yaxis)

    lfp_max = np.max(np.abs(lfp))

    pcm1 = ax.scatter(ele_pos[:, 0], ele_pos[:, 1], marker='s', s=20,
                     edgecolors='face', c=lfp[:, time_pt],
                     vmin=-lfp_max/25, vmax=lfp_max/25, cmap=plt.cm.PRGn)

    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=xmin, xmax=xmax)

    # ax.set_xlabel('X ($\mu$m)', fontsize=fsize)
    ax.set_xticks([])
    # ax.set_ylabel('Y ($\mu$m)', fontsize=fsize)
    ax.set_title(title, fontsize=16)
    ax.xaxis.set_tick_params(labelsize=fsize)
    # ax.yaxis.set_tick_params(labelsize=fsize)
    ax.set_yticks([])
    set_axis(ax, letter=letter)
    cb1 = plt.colorbar(pcm1)
    cb1.set_label('mV',fontsize=fsize)
    cb1.ax.tick_params(labelsize=16)
    # cb1.set_ticks([-2e-9,-1e-9,0,1e-9,2e-9])


def extract_csd_timepoint(h, pop_names, time_pts, field_name):
    all_x, all_y, all_z, all_val = [], [], [], []
    counter = 0
    # field_name = 'i'
    for pop_name in pop_names:
        print(pop_name)
        src = fetch_mid_pts(h, pop_name)
        print()
        idx = find_map_idx(h, pop_name, field_name)
        src_time = h['/data/uniform/' + pop_name + '/' + field_name][()]
        src_time = src_time[idx]  # Order according to correct indices

    # different way:
        for i in range(src.shape[0]):
            counter += 1
            if counter % 100 == 0:
                print(counter, ' /', src.shape[0])

            x = src[i, 0]
            y = src[i, 1]
            z = src[i, 2]  # depth, here width of the column

            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_val.append(src_time[i, time_pts])

    return np.array(all_x), np.array(all_y), np.array(all_z), np.array(all_val)



def calculate_smoothed_csd(X, Y, Z, Val, nX=32, nY=100, l=3009):
    #resX, resY = 25, 25
    nX = nX #int(800/resX)
    nY = nY #int(2500/resY)

    if len(Val.shape) == 2:
        nT = Val.shape[1]
        csd_smoothed = np.zeros((nX, nY, nT))
        normalization = np.zeros((nX, nY, nT))
    else:
        csd_smoothed = np.zeros((nX, nY))
        normalization = np.zeros((nX, nY))
    csd_smoothed_Z = 0

    linX = np.linspace(-400, 400, nX)
    linY = np.linspace(-2000, 500, nY)

    # linX = np.linspace(-500, 500, nX)
    # linY = np.linspace(-2250, 750, nY)

    sigma = 10
    A = ((2*np.pi)**(3/2))*(sigma**3)
    denominator = 2*np.pi*sigma**3
    for i_source in range(Val.shape[0]):
        print('percent done: ', i_source/Val.shape[0])
        for ii in range(nX):
            for jj in range(nY):
                csd_smoothed[ii, jj] += A * Val[i_source] *\
                    np.exp(-((linX[ii]-X[i_source])**2 +
                              (linY[jj]-Y[i_source])**2 +
                              (csd_smoothed_Z-Z[i_source])**2)/denominator)
                normalization[ii, jj] += A *\
                    np.exp(-((linX[ii]-X[i_source])**2 +
                              (linY[jj]-Y[i_source])**2 +
                              (csd_smoothed_Z-Z[i_source])**2)/denominator)

    XX, YY = np.meshgrid(linX, linY)
    print('normalization', normalization)
    print('csd/normalization', csd_smoothed/normalization)
    return csd_smoothed/normalization


def plot_all_currents(ax, xmin, xmax, ymin, ymax, all_x, all_y, all_z, all_val,
                      letter='', title='All currents'):
    currents_max = np.max(abs(all_val))
    ax.scatter(all_x, all_y, s=50*abs(all_val), c=all_val, cmap=plt.cm.cool,
                vmin=-currents_max, vmax=currents_max, marker='o')
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_xticks([])
    # ax.set_xlabel('X ($\mu$m)', fontsize=fsize)
    ax.set_ylabel('Y ($\mu$m)', fontsize=fsize)
    ax.set_title(title, fontsize=16)
    ax.xaxis.set_tick_params(labelsize=fsize)
    ax.yaxis.set_tick_params(labelsize=fsize)
    set_axis(ax, letter=letter)


def plot_csd_slice(ax, xmin, xmax, ymin, ymax, all_x, all_y, all_z, all_val,
                   letter='', title='True CSD in voxels'):
    nx, ny = int(800/50), int(2500/50)
    vx, vy = np.mgrid[-400:400:complex(0, nx+1),
                      -2000:500:complex(0, ny+1)]
    voxels_csd = np.zeros((nx, ny))

    counter = 0
    indxs_close_to_0 = np.where(abs(np.array(all_z)) < 50)[0]
    indexes_z = indxs_close_to_0
    for ii in range(nx):
        indexes_lower_x = np.array(np.where(np.array(all_x) < vx[ii+1, 0])[0],
                                   dtype=int)
        indexes_higher_x = np.array(np.where(np.array(all_x) > vx[ii, 0])[0],
                                    dtype=int)
        indexes_x = np.intersect1d(indexes_higher_x, indexes_lower_x)
        indexes_xz = np.intersect1d(indexes_x, indexes_z)
        for jj in range(ny):
            indexes_lower_y = np.array(
                    np.where(np.array(all_y) < vy[0, jj+1])[0], dtype=int)
            indexes_higher_y = np.array(
                    np.where(np.array(all_y) > vy[0, jj])[0], dtype=int)
            indexes_y = np.intersect1d(indexes_higher_y, indexes_lower_y)
            indexes_xyz = np.intersect1d(indexes_xz, indexes_y)
            voxels_csd[ii, jj] += np.sum(all_val[indexes_xyz])
            counter += 1  # to check if it is correct
    maxval = np.max(abs(voxels_csd))
    pcm=ax.pcolor(vx, vy, voxels_csd, cmap=plt.cm.bwr, vmin=-maxval, vmax=maxval)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=xmin, xmax=xmax)
    # ax.set_xlabel('X ($\mu$m)', fontsize=fsize)
    ax.set_yticks([]), ax.set_xticks([])
    # ax.set_ylabel('Y ($\mu$m)', fontsize=fsize)
    ax.set_title(title, fontsize=16)
    ax.xaxis.set_tick_params(labelsize=fsize)
    # cb1 = plt.colorbar(pcm)
    # ax.yaxis.set_tick_params(labelsize=fsize)
    set_axis(ax, letter=letter)
    return voxels_csd


def plot_dense_potentials(ax, h, pop_names, time_pts, time_pt_interest,
                          letter='', title='Estimated LFP', filt=False):
    # big 2d grid
    # nx, ny = 180, 600
    nx, ny = int(900 / 10), int(2500 / 10)
    # nx, ny = 900, 2500
    iso_x, iso_y = 16, 16

    z = 0
    tot_ele = nx * ny
    zz = np.ones((tot_ele, 1)) * z
    xx, yy = np.mgrid[-450:450:complex(0, nx), -2000:500:complex(0, ny)]
    xx = xx.reshape(tot_ele, 1)
    yy = yy.reshape(tot_ele, 1)
    elec_pos = np.hstack((xx, yy, zz))
    try:
        pot=np.load('pot_extra.npy')
    except:
        pot = get_extracellular(h, pop_names, time_pts, elec_pos)
        np.save('pot_extra', pot)
    if filt:
        pot = butter_bandpass_filter(pot, 0.1, 100, 10e3, order=3)
    plot_extracellular(ax, pot, elec_pos, iso_x, iso_y, time_pt_interest,
                       'dense', letter=letter, title=title)


def prepare_electrodes():
    electrode_settings_list = []
    electrode_names = []

    if False:
        # probes, at 0, 0,
        ny = 16
        xx = np.zeros((ny, 1))
        zz = np.zeros((ny, 1))
        yy = np.arange(0,-1600,-100).reshape((ny, 1))
        elec2 = np.hstack((xx, yy, zz))

        electrode_settings_list.append(elec2)
        electrode_names.append('16 probe')
    if False:
        # probes, at 0, 0,
        ny = 32
        xx = np.zeros((ny, 1))
        zz = np.zeros((ny, 1))
        yy = np.arange(0,-1600,-50).reshape((ny, 1))
        elec3 = np.hstack((xx, yy, zz))
        electrode_settings_list.append(elec3)
        electrode_names.append('32 probe')

    # neuroseeker
    nx, ny = 4, 32
    z = 0
    tot_ele = nx * ny
    zz = np.ones((tot_ele, 1)) * z
    dist_from_center_axis = 1.5*22.5+20
    od, do = 0, -(10+22.5)*ny
    xx, yy = np.mgrid[-dist_from_center_axis:dist_from_center_axis:
                      complex(0, nx), od:do:complex(0, ny)]
    xx = xx.reshape(tot_ele, 1)
    yy = yy.reshape(tot_ele, 1)
    elec4 = np.hstack((xx, yy, zz))

    electrode_settings_list.append(elec4)
    electrode_names.append('neuroseeker')

    # neuropixel
    nx, ny = 4, int(384/2)  # int(960/2)
    z = 0
    tot_ele = nx * int(ny/2)
    zz = np.ones((tot_ele, 1)) * z
    # width = 12  # center to center distance
    dist_v = 20
    dist_h = 16
    # left_border = -1.5/2*width
    left_border = -30

    od, do = 500, -(40*ny/2 - 500)
    x1, y1 = np.mgrid[left_border:left_border+2*dist_h:complex(0, nx/2),
                      od:do:complex(0, ny/2)]
    x1 = x1.reshape(int(tot_ele/2), 1)
    y1 = y1.reshape(int(tot_ele/2), 1)

    x2, y2 = np.mgrid[left_border+dist_h:left_border +
                      3*dist_h:complex(0, nx/2),
                      od-dist_v:do-dist_v:complex(0, ny/2)]
    x2 = x2.reshape(int(tot_ele/2), 1)
    y2 = y2.reshape(int(tot_ele/2), 1)

    xx = np.vstack((x1, x2))
    yy = np.vstack((y1, y2))
    elec5 = np.hstack((xx, yy, zz))
    electrode_settings_list.append(elec5)
    electrode_names.append('neuropixel')

    return electrode_settings_list, electrode_names


def plot_csd_smooth(ax, xmin, xmax, ymin, ymax, csd, XX, YY, letter='',
                    title='', ele_pos=None):

    max_csd = np.max(np.abs(csd))
    if letter in ['G', 'H']:
        levels = np.linspace(-0.0005, 0.0005, 201, endpoint=True)
    else:
        levels = np.linspace(-max_csd, max_csd, 201, endpoint=True)
    print(max_csd)
    pcm = ax.contourf(XX, YY, csd, levels=levels, cmap=plt.cm.bwr)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=xmin, xmax=xmax)
    try:
        ax.scatter(ele_pos[:, 0], ele_pos[:, 1],
                    marker='s', alpha=0.3, color='m', s=10)
    except (RuntimeError, TypeError, NameError):
        pass
    if letter!='C':
        ax.set_xlabel('X ($\mu$m)', fontsize=18)
    else:
        ax.set_xticks([])
    if letter=='E':
        ax.set_ylabel('Y ($\mu$m)', fontsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=16, pad=-2)
    ax.xaxis.set_tick_params(labelsize=18)
    set_axis(ax, letter=letter)
    # if letter=='H':
    ticks=[-0.0005,-0.00025,0,0.00025,0.0005]
    if letter=='H':
        cb1 = plt.colorbar(pcm, ticks=ticks)
        cb1.set_label('$\mu$A/mm$^3$',fontsize=18)
        cb1.formatter.set_powerlimits((-2,2))
        cb1.ax.tick_params(labelsize=16)
        cb1.update_ticks()
    # cb1.set_ticks([-2e4,-1e4,0,1e4,2e4])
        # cb1.set_ticks([-2e-9,-1e-9,0,1e-9,2e-9])


def make_column_plot(h, pop_names, time_pts, time_pt_interest, elec_pos_list, names_list,
                     all_x, all_y, all_z, all_val, true_csd, true_csd_p,
                     fig_title='Traubs_column'):
    global voxel_csd
    fig = plt.figure(figsize=(16, 14))
    # fig, axs = plt.subplots(2,4,figsize=(16, 14))
    ymin = -2000  # -2150
    ymax = 500  # 550
    xmin = -400  # -450
    xmax = 400  # 450
    aspect = 'auto'
    ax1 = plt.subplot(241, aspect=aspect)
    plot_all_currents(ax1, xmin, xmax, ymin, ymax, all_x, all_y, all_z,
                      all_val, letter='A')

    ax2 = plt.subplot(242, aspect=aspect)
    voxel_csd = plot_csd_slice(ax2, xmin, xmax, ymin, ymax, all_x, all_y, all_z, all_val,
                               letter='B')

    ax3 = plt.subplot(244, aspect=aspect)
    plot_dense_potentials(ax3, h, pop_names, time_pts, time_pt_interest,
                          letter='D', filt=False)

    ax4 = plt.subplot(243, aspect=aspect)
    xx, yy = np.mgrid[xmin:xmax:complex(0, true_csd.shape[0]),
                      ymin:ymax:complex(0, true_csd.shape[1])]
    plot_csd_smooth(ax4, xmin, xmax, ymin, ymax, true_csd[:, :],#/abs(true_csd).max(),
                    xx, yy, letter='C', title='Smoothed true CSD')

    ax5 = plt.subplot(247, aspect=aspect)
    ele_pos,name = elec_pos_list[0],names_list[0]
    pot = prepare_pots(ele_pos, name, h, pop_names, time_pts)
    kcsd, est_pot, x, y, k = do_kcsd(pot[:,750].reshape((pot.shape[0],1)), ele_pos[:, :2], xmin, xmax, ymin, ymax, n_src_init=5000)
    plot_csd_smooth(ax5, xmin, xmax, ymin, ymax, kcsd[:, :, 0], x, y,
                    letter='G', ele_pos=ele_pos, title='Neuroseeker CSD')

    ax6 = plt.subplot(248, aspect=aspect)
    ele_pos,name = elec_pos_list[1],names_list[1]
    pot = prepare_pots(ele_pos, name, h, pop_names, time_pts)
    kcsd, est_pot, x, y, k = do_kcsd(pot[:,750].reshape((pot.shape[0],1)), ele_pos[:, :2], xmin, xmax, ymin, ymax, n_src_init=5000)
    plot_csd_smooth(ax6, xmin, xmax, ymin, ymax, kcsd[:, :, 0], x, y,
                    letter='H', ele_pos=ele_pos, title='Neuropixel CSD')

    ax7 = plt.subplot(245, aspect=aspect)
    ele_pos,name = elec_pos_list[0],names_list[0]
    pot = prepare_pots(ele_pos, name, h, pop_names, time_pts)
    kcsd, est_pot, x, y, k = do_kcsd(pot[:,750].reshape((pot.shape[0],1)), ele_pos[:, :2], xmin, xmax, ymin, ymax, n_src_init=5000)
    eigensources = calculate_eigensources(k)
    projection = csd_into_eigensource_projection(true_csd_p.flatten(), eigensources)
    plot_csd_smooth(ax7, xmin, xmax, ymin, ymax, projection.reshape(true_csd_p.shape), x, y,
                    letter='E', title='Neuroseeker proj.')

    ax8 = plt.subplot(246, aspect=aspect)
    ele_pos,name = elec_pos_list[1],names_list[1]
    pot = prepare_pots(ele_pos, name, h, pop_names, time_pts)
    kcsd, est_pot, x, y, k = do_kcsd(pot[:,750].reshape((pot.shape[0],1)), ele_pos[:, :2], xmin, xmax, ymin, ymax, n_src_init=5000)
    eigensources = calculate_eigensources(k)
    projection = csd_into_eigensource_projection(true_csd_p.flatten(), eigensources)
    plot_csd_smooth(ax8, xmin, xmax, ymin, ymax, projection.reshape(true_csd_p.shape), x, y,
                        letter='F', title='Neuropixel proj.')


    fig.savefig('Fig2_h=1_final.png', dpi=300)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def do_kcsd(pot, ele_pos, xmin, xmax, ymin, ymax, n_src_init=1000):
    k = KCSD2D(ele_pos, pot,
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, h=1, sigma=1,
               n_src_init=n_src_init, gdx=8, gdy=8, R_init=32, lambd=1e-9)
    # k.L_curve(lambdas=np.linspace(1e-08,1e-3,20), Rs=np.array([32]))
    # k.cross_validate(lambdas=np.array(6.455e-10),
    #                   Rs=np.array([80]))
    kcsd = k.values('CSD')
    est_pot = k.values('POT')
    from scipy.spatial import distance
    ele_dists = distance.cdist(ele_pos[::4], ele_pos[::4], 'euclidean')
    ele_dist = np.zeros(ele_dists.shape[0]-1)
    for i in range(1,ele_dists.shape[0]):
        ele_dist[i-1]=ele_dists[i,i-1]
    # trad_csd = np.zeros(ele_dists.shape[0]-2)
    # trad_pot = pot[::4]
    # for i in range(1,ele_dists.shape[0]-1):
    #     print(trad_csd.shape)
    #     print((1/(4*np.pi*1)*(2*trad_pot[i]-trad_pot[i-1]-trad_pot[i+1])/ele_dist[i-1]**2).shape)
    #     trad_csd[i-1] = 1/(4*np.pi*1)*(2*trad_pot[i]-trad_pot[i-1]-trad_pot[i+1])/ele_dist[i-1]**2
    # print('kcsd max min: ', kcsd.max(), kcsd.min())
    # print('trad csd max min: ', trad_csd.max(), trad_csd.min())
    return kcsd, est_pot, k.estm_x, k.estm_y, k


def prepare_pots(ele_pos, name, h, pop_names, time_pts):
    """Filter the data and downsample (from 10kHz to 2.5kHz) """
    print(name)
    try:
        print('pot loaded from file')
        pot = np.load('pot_fig1'+name+'.npy')
    except:
        pot = get_extracellular(h, pop_names, time_pts, ele_pos)
        np.save('pot_fig1'+name, pot)

    # filtering and downsampling
    fs = 10e3
    lowcut = 0.1
    highcut = 100
    pots_filter = butter_bandpass_filter(pot, lowcut, highcut, fs, order=3)
    pot_down_samp = pots_filter[:, ::4]
    pot = pot_down_samp
    return pot


def calculate_eigensources(obj):
    try:
        eigenvalue, eigenvector = np.linalg.eigh(obj.k_pot +
                                                 obj.lambd *
                                                 np.identity
                                                 (obj.k_pot.shape[0]))
    except LinAlgError:
        raise LinAlgError('EVD is failing - try moving the electrodes'
                          'slightly')
    idx = eigenvalue.argsort()[::-1]
    eigenvalues = eigenvalue[idx]
    eigenvectors = eigenvector[:, idx]
    eigensources = np.dot(obj.k_interp_cross, eigenvectors)
    return eigensources


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.matmul(np.matmul(csd, orthn), orthn.T)


if __name__ == '__main__':
    fsize=16
    time_pt_interest = 3000
    time_pts = 6000  # number of all time frames
    num_cmpts = [74, 74, 59, 59, 59, 59, 61, 61, 50, 59, 59, 59]
    cell_range = [0, 1000, 1050, 1140, 1230, 1320,
                  1560, 2360, 2560, 3060, 3160, 3260, 3360]
    num_cells = np.diff(cell_range) / 10  # 10% MODEL
    total_cmpts = list(num_cmpts * num_cells)
    pop_names = ['pyrRS23', 'pyrFRB23', 'bask23', 'axax23', 'LTS23',
                 'spinstel4', 'tuftIB5', 'tuftRS5', 'nontuftRS6',
                 'bask56', 'axax56', 'LTS56']

    h = h5.File('pulsestimulus10model.h5', 'r')
    elec_pos_list, names_list = prepare_electrodes()

    ### calculate sources values for a single time frame (it may take a while)
    all_x, all_y, all_z, all_val = extract_csd_timepoint(h, pop_names,
                                                          time_pt_interest, 'i')

    true_csd = calculate_smoothed_csd(all_x, all_y, all_z, all_val)
    true_csd_p = calculate_smoothed_csd(all_x, all_y, all_z, all_val, nX=101, nY=313)
    np.save('true_csd_fig2', true_csd)
    np.save('true_csd_p_fig2',true_csd_p)
    # # true_csd, true_csd_p = np.load('true_csd_fig2_v2.npy'), np.load('true_csd_p_fig2_v2.npy')[::1,::1]
    true_csd, true_csd_p = np.load('true_csd_fig2.npy'), np.load('true_csd_p_fig2.npy')[::2,::2]
    # sigma=1
    # A = ((2*np.pi)**(3/2))*(sigma**3)
    # denominator = 2*np.pi*sigma**3
    # norm = 0
    # for linZ in np.linspace(-400,400,true_csd_p.shape[0]):
    #     # print('percent done',(linZ+400)/800)
    #     for linX in np.linspace(-400,400,true_csd_p.shape[0]):
    #         for linY in np.linspace(-2000,500,true_csd_p.shape[1]):
    #             # print(np.exp(-((linX)**2 +(linY)**2)/denominator))
    #             norm +=  A * np.exp(-((linX)**2 +(linY)**2+(linZ)**2)/denominator)*8**3
    # true_csd/=norm
    # true_csd_p/=norm

    make_column_plot(h, pop_names, time_pts, time_pt_interest, elec_pos_list, names_list,
                      all_x, all_y, all_z, all_val, true_csd, true_csd_p)
