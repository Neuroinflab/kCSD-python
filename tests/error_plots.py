#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:28:36 2017

@author: mkowalska
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma


def mean_error_threshold(point_error, ele_x, ele_y, path, save_as, n,
                         threshold=1.):
        """
        Creates plot of mean error calculated separately for every point of
        estimation space

        Parameters
        ----------
        rms: numpy array
        R: float
        csd_profile: function
            function to produce csd profile
        csd_seed: int
            seed for random generator

        Returns
        -------
        None
        """
        point_mask = ma.masked_array(point_error, point_error > threshold)
        mean_mask = ma.mean(point_mask, axis=0)
        mean_nr = ma.count(point_mask, axis=0)
        x, y = np.mgrid[0:1:
                        np.complex(0, 100),
                        0:1:
                        np.complex(0, 100)]
        fig = plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(121, aspect='equal')
        levels = np.linspace(0, 1., 15)
        im = ax1.contourf(x, y, mean_mask, levels=levels, cmap='Greys')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.scatter(ele_x, ele_y)
        ax1.set_xlabel('Depth x [mm]')
        ax1.set_ylabel('Depth y [mm]')
        ax1.legend()
        ax2 = plt.subplot(122, aspect='equal')
        levels = np.linspace(0, 100., 15)
        im2 = ax2.contourf(x, y, mean_nr/n, levels=levels, cmap='Greys')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.scatter(ele_x, ele_y)
        ax2.set_xlabel('Depth x [mm]')

        ax2.legend()
        fig.savefig(os.path.join(path, save_as + '.png'))
#        plt.close()
        return


def broken_electrode(seed, n, res):
    ele_x, ele_y = np.mgrid[0.1:0.9:np.complex(0, res),
                            0.1:0.9:np.complex(0, res)]
    ele_x, ele_y = ele_x.flatten(), ele_y.flatten()
    ele_grid = np.vstack((ele_x, ele_y)).T
    random_indices = np.arange(0, ele_grid.shape[0])
    np.random.seed(seed)
    np.random.shuffle(random_indices)
    ele_pos = ele_grid[random_indices[:36 - n]]
    return ele_pos[:, 0], ele_pos[:, 1]


def matrix_symmetrization(point_error):
    r1 = np.rot90(point_error, k=1, axes=(1, 2))
    r2 = np.rot90(point_error, k=2, axes=(1, 2))
    r3 = np.rot90(point_error, k=3, axes=(1, 2))
    arr_lr = np.zeros([100, 100, 100])
    for i in range(100):
        arr_lr[i] = np.flipud(point_error[i, :, :])
    r11 = np.rot90(arr_lr, k=1, axes=(1, 2))
    r12 = np.rot90(arr_lr, k=2, axes=(1, 2))
    r13 = np.rot90(arr_lr, k=3, axes=(1, 2))
    symm_array = np.concatenate((point_error, r1, r2, r3, arr_lr, r11, r12,
                                 r13))
    print(symm_array.shape)
    return symm_array


def generate_electrodes(start, end, x_res, y_res):
    ele_x, ele_y = np.mgrid[start:end:np.complex(0, x_res),
                            start:end:np.complex(0, y_res)]
    return ele_x.flatten(), ele_y.flatten()


def error_difference(error1, error2, path, save_as, n=0, threshold=1):
    mask1 = ma.masked_array(error1, error1 > threshold)
    mean_mask1 = ma.mean(mask1, axis=0)
    mask2 = ma.masked_array(error2, error2 > threshold)
    mean_mask2 = ma.mean(mask2, axis=0)
    x, y = np.mgrid[0:1:
                    np.complex(0, 100),
                    0:1:
                    np.complex(0, 100)]
    difference = mean_mask1 - mean_mask2
    fig = plt.figure(figsize=(5, 5))
    axs = plt.subplot()

    levels = np.linspace(-0.1, 0.1, 20)
    difference[np.where(difference > 0.1)] = 0.1
    difference[np.where(difference < -0.1)] = -0.1
    im = plt.contourf(x, y, difference, levels, cmap='PuOr')
#    im = plt.contourf(x, y, difference, cmap='PuOr')

    plt.axis('equal')
    axs.set_aspect('equal', 'box')
    plt.xlabel('Depth x [mm]')
    plt.ylabel('Depth y [mm]')
    ele_x, ele_y = broken_electrode(10, n, 6)
    plt.scatter(ele_x, ele_y, s=40, c='black')
    plt.colorbar(im, fraction=0.046, pad=0.04,
                 ticks=[-0.1, -0.05, 0, 0.05, 0.1])
    fig.savefig(os.path.join(path, save_as + 'with_electrodes' + '.png'))
    return difference


def sigmoid_mean(error):
    sig_error = 2*(1./(1 + np.exp((-error))) - 1/2.)
    error_mean = np.mean(sig_error, axis=0)
#    print(error_mean.shape)
    return error_mean


def error_diff(error, error1, error2, error4, error6, error8, error10, error12,
               error15, path, save_as, n=0, threshold=1):
    x, y = np.mgrid[0:1:
                    np.complex(0, 100),
                    0:1:
                    np.complex(0, 100)]
    n = [1, 2, 4, 6, 8, 10, 12, 15]
    letters = ['A) ', 'B) ', 'C) ', 'D) ', 'E) ', 'F) ', 'G) ', 'H) ']
    err_list = [error1, error2, error4, error6, error8, error10, error12,
                error15]
    fig, axs = plt.subplots(2, 4, figsize=(17, 9))
    axs = axs.ravel()
    levels = np.linspace(-0.15, 0.15, 11)
    for i in range(8):
        im = axs[i].contourf(x, y, (err_list[i] - error), levels=levels,
                             cmap='PuOr')
        axs[i].set_aspect('equal', 'box')
        axs[i].set_xlabel('Depth x [mm]')
        axs[i].set_ylabel('Depth y [mm]')
        if n[i] == 1:
            axs[i].set_title(letters[i] + str(n[i]) + ' broken electrode')
        else:
            axs[i].set_title(letters[i] + str(n[i]) + ' broken electrodes')
        ele_x, ele_y = broken_electrode(10, n[i], 6)
        axs[i].scatter(ele_x, ele_y, s=40, c='black')
    plt.colorbar(im, fraction=0.046, pad=0.04, ax=axs[i], format='%.2f')
    fig.savefig(os.path.join(path, save_as + 'with_electrodes' + '.png'))
    return


def error_sigmoid(point_error, save_as, res, n=0):
    sig_error = 2*(1./(1 + np.exp(-point_error)) - 1/2.)
    error_mean = np.mean(sig_error, axis=0)
    x, y = np.mgrid[0:1:
                    np.complex(0, 100),
                    0:1:
                    np.complex(0, 100)]
    ele_x, ele_y = np.mgrid[0.1:0.9:
                            np.complex(0, res),
                            0.1:0.9:
                            np.complex(0, res)]
    if res == 6:
        ele_x, ele_y = broken_electrode(10, n, res)
    fig = plt.figure(figsize=(5, 5))
    axs = plt.subplot()
    levels = np.linspace(0, 100, 20)
#    error_mean_rescaled = (error_mean - 0.5) * 2
    im = plt.contourf(x, y, error_mean*100, levels=levels, cmap='Greys')
#    im = plt.contourf(x, y, error_mean_rescaled*100, levels=levels,
#                      cmap='Greys')
    plt.scatter(ele_x, ele_y)
    axs.set_aspect('equal', 'box')
    plt.xlabel('Depth x [mm]')
    plt.ylabel('Depth y [mm]')
    plt.colorbar(im, label=' Error [%]', fraction=0.046, pad=0.04,
                 format='%.0f')
    fig.savefig(os.path.join(path, save_as + '_sigmoidal_percentage_scale' +
                             '.png'))
    return sig_error

path = '/home/mkowalska/Dropbox/PNI/20171125_Aspects_Warsaw/Marta/sigmoidal_representation'

#point_error = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-21/20171121-094110/point_error.npy')
#symm_array6free = matrix_symmetrization(point_error)
#sig_error = error_sigmoid(symm_array6free)
#m = 8
#save_as = 'Thresholded_Mean_point_error_'+str(m)+'x'+str(m)+'_noise_free_symmetry'
#path = '/home/mkowalska/Dropbox/PNI/20171125_Aspects_Warsaw/Marta'
point_error8free = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-20/20171120-165312/point_error.npy')
symm_array8free = matrix_symmetrization(point_error8free)
#sig_error8free = error_sigmoid(symm_array8free, '8x8_noise_free', 8, n=0)
##ele_x, ele_y = generate_electrodes(0.1, 0.9, m, m)
##mean_error_threshold(symm_array, ele_x, ele_y, path, save_as, 8, threshold=1)
#
point_error6free = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-21/20171121-094110/point_error.npy')
symm_array6free = matrix_symmetrization(point_error6free)
#error_sigmoid(point_error6free, '6x6_noise_free', 6, n=0)
#sig_error6free = error_sigmoid(symm_array6free, '6x6_noise_free', 6, n=0)
##difference = error_difference(symm_array6free, symm_array8free, path, '6x6_8x8_difference_noise_free')
#
"""
point_error6noise = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-21/20171121-162023/point_error.npy')
symm_array6noise = matrix_symmetrization(point_error6noise)
sig_error6noise = error_sigmoid(symm_array6noise, '6x6_with_noise', 6, n=0)
point_error8noise = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-21/20171121-175459/point_error.npy')
symm_array8noise = matrix_symmetrization(point_error8noise)
sig_error8noise = error_sigmoid(symm_array8noise, '8x8_with_noise', 8, n=0)
"""
##difference = error_difference(symm_array6noise, symm_array8noise, path, '6x6_8x8_difference_with_noise')
#
##difference = error_difference(symm_array6noise, symm_array6free, path, '6x6_difference_with_noise_vs_noise_free')
##difference = error_difference(symm_array8noise, symm_array8free, path, '8x8_difference_with_noise_vs_noise_free')
#
broken1 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-22/20171122-120315/point_error.npy')
#sig_error_broken1 = error_sigmoid(broken1, 'broken1', 6, n=1)
broken2 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-23/20171123-085556/point_error.npy')
#sig_error_broken2 = error_sigmoid(broken2, 'broken2', 6, n=2)
broken4 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-23/20171123-103602/point_error.npy')
#sig_error_broken4 = error_sigmoid(broken4, 'broken4', 6, n=4)
broken6 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-23/20171123-120701/point_error.npy')
#sig_error_broken6 = error_sigmoid(broken6, 'broken6', 6, n=6)
broken8 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-22/20171122-182142/point_error.npy')
#sig_error_broken8 = error_sigmoid(broken8, 'broken8', 6, n=8)
broken10 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-23/20171123-135240/point_error.npy')
#sig_error_broken10 = error_sigmoid(broken10, 'broken10', 6, n=10)
broken12 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-23/20171123-153122/point_error.npy')
#sig_error_broken12 = error_sigmoid(broken12, 'broken12', 6, n=12)
broken15 = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-23/20171123-171820/point_error.npy')
#sig_error_broken15 = error_sigmoid(broken15, 'broken15', 6, n=15)
#
"""
point_error4noise = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-22/20171122-085534/point_error.npy')
symm_array4noise = matrix_symmetrization(point_error4noise)
sig_error4noise = error_sigmoid(symm_array4noise, '4x4_with_noise', 4, n=0)
point_error4free = np.load('/home/mkowalska/Marta/kCSD_results/2017-11-21/20171121-144525/point_error.npy')
symm_array4free = matrix_symmetrization(point_error4free)
sig_error4free = error_sigmoid(symm_array4free, '4x4_noise_free', 4, n=0)
"""
##difference = error_difference(symm_array4noise, symm_array4free, path, '4x4_difference_with_noise_vs_noise_free')
#
##difference = error_difference(symm_array4noise, symm_array8noise, path, '4x4_8x8_difference_with_noise')
#
##difference = error_difference(symm_array4free, symm_array8free, path, '4x4_8x8_difference_noise_free')
#
#difference = error_difference(broken1, point_error6free, path, '6x6_difference_noise_free_broken_1', n=1)
#difference = error_difference(broken4, point_error6free, path,'6x6_difference_noise_free_broken_4',  n=4)
#difference = error_difference(broken2, point_error6free, path, '6x6_difference_noise_free_broken_2', n=2)
#difference = error_difference(broken6, point_error6free, path, '6x6_difference_noise_free_broken_6', n=6)
#difference = error_difference(broken8, point_error6free, path, '6x6_difference_noise_free_broken_8', n=8)
#difference = error_difference(broken10, point_error6free, path, '6x6_difference_noise_free_broken_10', n=10)
#difference = error_difference(broken12, point_error6free, path, '6x6_difference_noise_free_broken_12', n=12)
#difference = error_difference(broken15, point_error6free, path, '6x6_difference_noise_free_broken_15', n=15)

error_diff(sigmoid_mean(point_error6free), sigmoid_mean(broken1),
           sigmoid_mean(broken2), sigmoid_mean(broken4),
           sigmoid_mean(broken6), sigmoid_mean(broken8),
           sigmoid_mean(broken10), sigmoid_mean(broken12),
           sigmoid_mean(broken15), path, '6x6_broken_electrodes_subplot')
