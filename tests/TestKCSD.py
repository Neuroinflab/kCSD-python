#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:34:35 2017

@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import time
import os
from datetime import date

from builtins import int
from future import standard_library

import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

sys.path.append('..')
standard_library.install_aliases()


class TestKCSD(object):
    """
    Base class for tests of kCSD method
    """
    def __init__(self, dim, **kwargs):
        """Initialize TestKCSD class
        Parameters
        ----------
        k : object of the class 'KCSD1D'
        dim: int
            case dimention (1, 2 or 3 D)
        **kwargs
            configuration parameters, that may contain the following keys:
            src_type : str
                basis function type ('gauss', 'step', 'gauss_lim')
                Defaults to 'gauss'
            sigma : float
                space conductance of the medium
                Defaults to 0.3
            R_init : float
                demanded thickness of the basis element
                Defaults to 0.23
            h : float
                thickness of analyzed cylindrical slice
                Defaults to 1.
            nr_basis: int
                number of basis sources
                Defaults to 300
            total_ele: int
                number of electrodes
                Defaults to 10
            ele_placement: string
                how the electrodes are distributed ('regular', 'random')
                Defaults to 'regular'
            ele_seed: int
                seed for random electrodes position placement
                Defaults to 10
            timestr: string
                contains information about current time
                Defaults to time.strftime("%Y%m%d-%H%M%S")
            day: string
                what is the day (date) today
                Defaults: date.today()
            path: string
                where to save the data
                Defaults to os.getcwd()
            ext_x : float
                length of space extension: x_min-ext_x ... x_max+ext_x
                Defaults to 0.
            basis_xlims: list
                boundaries for basis placement space
                Defaults to [0., 1.]
            est_xres: int
                resolution of kcsd estimation
                Defaults to 100
            kcsd_xlims: list
                boundaries for kcsd estimation space
                Defaults to [0., 1.]
            true_csd_xlims: list
                boundaries for ground truth space
                Defaults to [0., 1.]
            ele_xlims: list
                boundaries for electrodes placement
                Defaults to [0., 1.]
            csd_xres: int
                resolution of ground truth
                Defaults to 100

        Returns
        -------
        None
        """
        self.dim = dim
        self.general_parameters(**kwargs)
        self.dimension_parameters(**kwargs)
        return

    def general_parameters(self, **kwargs):
        """
        Method that loads parameters or takes default values

        Parameters
        ----------
        **kwargs
            Same as those passed to initialize the Class

        Returns
        -------
        None
        """
        self.src_type = kwargs.get('src_type', 'gauss')
        self.sigma = kwargs.get('sigma', 0.3)
        self.h = kwargs.get('h', 0.25)
        self.R_init = kwargs.get('R_init', 0.23)
        self.nr_basis = kwargs.get('nr_basis', 300)
        self.total_ele = kwargs.get('total_ele', 10)
        self.ele_placement = kwargs.get('ELE_PLACEMENT', 'regular')
        self.ele_seed = kwargs.get('ele_seed', 10)
        self.timestr = kwargs.get('TIMESTR', time.strftime("%Y%m%d-%H%M%S"))
        self.day = kwargs.get('DAY', date.today())
        self.path = kwargs.get('path', os.getcwd())
        self.config = kwargs.get('config', 'regular')
        self.doc = kwargs.get('doc', ' ')
        return

    def dimension_parameters(self, **kwargs):
        """
        Method that loads parameters or takes default values

        Parameters
        ----------
        dim: int
            dimention of analyzed case (1, 2 or 3D)
        **kwargs
            Same as those passed to initialize the Class

        Returns
        -------
        None
        """
        self.ext_x = kwargs.get('ext_x', 0.0)
        self.basis_xlims = kwargs.get('basis_xlims', [0., 1.])
        self.kcsd_xlims = kwargs.get('kcsd_xlims', [0., 1.])
        self.est_xres = kwargs.get('est_xres', 100)
        self.true_csd_xlims = kwargs.get('true_csd_xlims', [0., 1.])
        self.ele_xlims = kwargs.get('ele_xlims', [0.1, 0.9])
        self.ele_xres = kwargs.get('ele_xres', (self.total_ele))
        self.csd_xres = kwargs.get('csd_xres', 100)
        if self.dim >= 2:
            self.ext_y = kwargs.get('ext_y', 0.0)
            self.ele_ylims = kwargs.get('ele_ylims', [0.1, 0.9])
            self.basis_ylims = kwargs.get('basis_ylims', [0., 1.])
            self.kcsd_ylims = kwargs.get('kcsd_ylims', [0., 1.])
            self.true_csd_ylims = kwargs.get('true_csd_ylims', [0., 1.])
            self.est_yres = kwargs.get('est_yres', 100)
            self.ele_yres = kwargs.get('ele_yres',
                                       int(np.sqrt(self.total_ele)))
            self.csd_yres = kwargs.get('csd_yres', 100)
        if self.dim == 3:
            self.ext_z = kwargs.get('ext_z', 0.0)
            self.ele_zlims = kwargs.get('ele_zlims', [0.1, 0.9])
            self.basis_zlims = kwargs.get('basis_zlims', [0., 1.])
            self.kcsd_zlims = kwargs.get('kcsd_zlims', [0., 1.])
            self.true_csd_zlims = kwargs.get('true_csd_zlims', [0., 1.])
            self.est_zres = kwargs.get('est_zres', 100)
            self.ele_zres = kwargs.get('ele_zres',
                                       int(np.cbrt(self.total_ele)))
            self.csd_zres = kwargs.get('csd_zres', 100)
        return

    def svd(self, k):
        """
        Method that calculates singular value decomposition of total kernel
        matrix

        Defines:
            self.u_svd: numpy array, shape (nr_basis, total_ele)
                left singular vectors
            self.sigma: numpy array, shape (total_ele)
                singular values
            self.v_svd: numpy array, shape (total_ele, total_ele)
                right singular vectors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        kernel = np.dot(k.k_interp_cross,
                        inv(k.k_pot + k.lambd * np.identity(k.k_pot.shape[0])))
        u_svd, sigma, v_svd = np.linalg.svd(kernel, full_matrices=False)
        self.plot_svd_sigma(sigma)
        self.plot_svd_u(u_svd)
        self.plot_svd_v(v_svd)
        np.save(os.path.join(self.path, 'kernel.npy'), kernel)
        np.save(os.path.join(self.path, 'u_svd.npy'), u_svd)
        np.save(os.path.join(self.path, 'sigma.npy'), sigma)
        np.save(os.path.join(self.path, 'v_svd.npy'), v_svd)
        np.save(os.path.join(self.path, 'k_pot.npy'), k.k_pot)
        return u_svd, sigma, v_svd

    def picard_plot(self, k, b):
        u, s, v = np.linalg.svd(k.k_pot)
        picard = np.zeros(len(s))
        picard_norm = np.zeros(len(s))
        for i in range(len(s)):
            picard[i] = abs(np.dot(u[:, i].T, b))
            picard_norm[i] = abs(np.dot(u[:, i].T, b))/s[i]
        fig = plt.figure(figsize=(10, 6))
        plt.plot(s, marker='.', label=r'$\sigma_{i}$')
        plt.plot(picard, marker='.', label='$|u(:, i)^{T}*b|$')
        plt.plot(picard_norm, marker='.',
                 label=r'$\frac{|u(:, i)^{T}*b|}{\sigma_{i}}$')
        plt.yscale('log')
        plt.legend()
        plt.title('Picard plot')
        plt.xlabel('i')
        fig.savefig(os.path.join(self.path, 'Picard_plot' + '.png'))
        self.plot_s(s)
        self.plot_u(u)
        self.plot_v(v)
        a = int(self.total_ele - int(np.sqrt(self.total_ele))**2)
        if a == 0:
            size = int(np.sqrt(self.total_ele))
        else:
            size = int(np.sqrt(self.total_ele)) + 1
        fig2, axs = plt.subplots(int(np.sqrt(self.total_ele)),
                                 size, figsize=(15, 13))
        axs = axs.ravel()
        beta = np.zeros(v.shape)
        fig2.suptitle('vectors products of k_pot matrix')
        for i in range(self.total_ele):
            beta[i] = ((np.dot(u[:, i].T, b)/s[i]) * v[i, :])
            axs[i].plot(beta[i, :], marker='.')
            axs[i].set_title(r'$vec_{'+str(i+1)+'}$')
        fig2.savefig(os.path.join(self.path, 'vectores_k_pot' +
                                  '.png'))

#        for i in range(len(b)):
#            beta[i] = ((np.dot(u[:, i].T, b)/s[i]) * v[i, :])
#        plt.plot(beta.T, marker='.')
        return

    def plot_s(self, s):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig = plt.figure()
        plt.plot(s, '.')
        plt.title('Singular values of k_pot matrix')
        plt.xlabel('Components number')
        plt.ylabel('Singular values')
        plt.yscale('log')
        fig.savefig(os.path.join(self.path, 'SingularValues_k_pot' + '.png'))
#        plt.close()
        return

    def evd(self, k):
        """
        Method that calculates eigenvalue decomposition of kernel

        Defines:
            self.eigenvectors: numpy array, shape(total_ele, total_ele)
                eigen vectors
            self.eigrnvalues: numpy array, shape(total_ele)
                eigenvalues

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        eigenvalues, eigenvectors = np.linalg.eigh(k.k_pot +
                                                   k.lambd * np.identity
                                                   (k.k_pot.shape[0]))
        idx = eigenvalues.argsort()[::-1]  # SVD&EVD in the opposite order
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.plot_evd(eigenvalues)
        return

    def plot_svd_sigma(self, sigma):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig = plt.figure()
        plt.plot(sigma, 'b.')
        plt.title('Singular values of kernels product')
        plt.xlabel('Components number')
        plt.ylabel('Singular values')
        plt.yscale('log')
        fig.savefig(os.path.join(self.path, 'SingularValues_kernels_product' +
                                 '.png'))
        plt.close()
        return

    def plot_u(self, u):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig1 = plt.figure()
        plt.plot(u.T, 'b.')
        plt.title('Left singular vectors of k_pot matrix')
        plt.ylabel('Singular vectors')
        fig1.savefig(os.path.join(self.path, 'left_SingularVectorsT_k_pot' +
                                  '.png'))
        plt.close()
        a = int(self.total_ele - int(np.sqrt(self.total_ele))**2)
        if a == 0:
            size = int(np.sqrt(self.total_ele))
        else:
            size = int(np.sqrt(self.total_ele)) + 1
        fig2, axs = plt.subplots(int(np.sqrt(self.total_ele)),
                                 size, figsize=(15, 13))
        axs = axs.ravel()
        fig2.suptitle('Left singular vectors of k_pot matrix')
        for i in range(self.total_ele):
            axs[i].plot(u[:, i], marker='.')
            axs[i].set_title(r'$u_{'+str(i+1)+'}$')
        fig2.savefig(os.path.join(self.path, 'left_SingularVectors_k_pot' +
                                  '.png'))
#        plt.close()
        return

    def plot_v(self, v):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig1 = plt.figure()
        plt.plot(v.T, 'b.')
        plt.title('Right singular vectors of k_pot matrix')
        plt.ylabel('Singular vectors')
        fig1.savefig(os.path.join(self.path, 'right_SingularVectorsT_k_pot' +
                                  '.png'))
        plt.close()
        a = int(self.total_ele - int(np.sqrt(self.total_ele))**2)
        if a == 0:
            size = int(np.sqrt(self.total_ele))
        else:
            size = int(np.sqrt(self.total_ele)) + 1
        fig2, axs = plt.subplots(int(np.sqrt(self.total_ele)),
                                 size, figsize=(15, 13))
        axs = axs.ravel()
        fig2.suptitle('right singular vectors of k_pot matrix')
        for i in range(self.total_ele):
            axs[i].plot(v[i, :], marker='.')
            axs[i].set_title(r'$v_{'+str(i+1)+'}$')
        fig2.savefig(os.path.join(self.path, 'right_SingularVectors_k_pot' +
                                  '.png'))
#        plt.close()
        return

    def plot_svd_u(self, u_svd):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig1 = plt.figure()
        plt.plot(u_svd.T, 'b.')
        plt.title('Singular vectors of kernels product')
        plt.ylabel('Singular vectors')
        fig1.savefig(os.path.join(self.path, 'SingularVectorsT' + '.png'))
        plt.close()
        a = int(self.total_ele - int(np.sqrt(self.total_ele))**2)
        if a == 0:
            size = int(np.sqrt(self.total_ele))
        else:
            size = int(np.sqrt(self.total_ele)) + 1
        fig2, axs = plt.subplots(int(np.sqrt(self.total_ele)),
                                 size, figsize=(15, 14))
        axs = axs.ravel()
        fig2.suptitle('Left singular vectors of kernels product')
        for i in range(self.total_ele):
            axs[i].plot(u_svd[:, i], '.')
            axs[i].set_title(r'$u_{'+str(i+1)+'}$')
        fig2.savefig(os.path.join(self.path, 'SingularVectors' + '.png'))
        plt.close()
        return

    def plot_svd_v(self, v_svd):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig1 = plt.figure()
        plt.plot(v_svd.T, 'b.')
        plt.title('Right singular vectors of kernels product')
        plt.ylabel('Singular vectors')
        fig1.savefig(os.path.join(self.path, 'right_SingularVectorsT' +
                                  '.png'))
        plt.close()
        a = int(self.total_ele - int(np.sqrt(self.total_ele))**2)
        if a == 0:
            size = int(np.sqrt(self.total_ele))
        else:
            size = int(np.sqrt(self.total_ele)) + 1
        fig2, axs = plt.subplots(int(np.sqrt(self.total_ele)),
                                 size, figsize=(15, 14))
        axs = axs.ravel()
        fig2.suptitle('Right singular vectors of kernels product')
        for i in range(self.total_ele):
            axs[i].plot(v_svd[i, :], marker='.')
            axs[i].set_title(r'$v_{'+str(i+1)+'}$')
        fig2.savefig(os.path.join(self.path, 'Right_SingularVectors' + '.png'))
#        plt.close()
        return

    def plot_evd(self, eigenvalues):
        """
        Creates plot of singular values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig = plt.figure()
        plt.plot(eigenvalues, 'b.')
        plt. title('Eigenvalues of kernels product')
        plt.xlabel('Components number')
        plt.ylabel('Eigenvalues')
        fig.savefig(os.path.join(self.path, 'Eigenvalues' + '.png'))
        plt.close()
        return

    def broken_electrode(self, seed, n):
        ele_x, ele_y = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                np.complex(0, self.ele_yres),
                                self.ele_ylims[0]:self.ele_ylims[1]:
                                np.complex(0, self.ele_yres)]
        ele_x, ele_y = ele_x.flatten(), ele_y.flatten()
        ele_grid = np.vstack((ele_x, ele_y)).T
        random_indices = np.arange(0, ele_grid.shape[0])
        np.random.seed(seed)
        np.random.shuffle(random_indices)
        ele_pos = ele_grid[random_indices[:self.total_ele - n]]
        return ele_pos[:, 0], ele_pos[:, 1]

    def generate_electrodes(self):
        """
        Places electrodes linearly

        Parameters
        ----------
        dim: int
            dimention of analyzed case (1, 2 or 3D)

        Returns
        -------
        linearly placed electrodes positions
        (for 1D case: ele_x, 2D case: ele_x, ele_y, and
        3D case: ele_x, ele_y, ele_z)
        """
        if self.dim == 1:
            ele_x = np.linspace(self.ele_xlims[0], self.ele_xlims[1],
                                self.total_ele)
            return ele_x
        elif self.dim == 2:
            if self.config == 'mavi':
                self.total_ele = 16
                ele_x, ele_y = self.mavi_electrodes()
#            elif self.config == 'broken':
#                ele_x, ele_y = self.
            else:
                ele_x, ele_y = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                        np.complex(0, self.ele_yres),
                                        self.ele_ylims[0]:self.ele_ylims[1]:
                                        np.complex(0, self.ele_yres)]
            return ele_x.flatten(), ele_y.flatten()
        elif self.dim == 3:
            ele_x, ele_y, ele_z = np.mgrid[self.ele_xlims[0]:self.ele_xlims[1]:
                                           np.complex(0, self.ele_zres),
                                           self.ele_ylims[0]:self.ele_ylims[1]:
                                           np.complex(0, self.ele_zres),
                                           self.ele_zlims[0]:self.ele_zlims[1]:
                                           np.complex(0, self.ele_zres)]
            return ele_x.flatten(), ele_y.flatten(), ele_z.flatten()

    def mavi_electrodes(self):
        """
        Electrodes locations for Mavi's arrays

        Parameters
        ----------
        None

        Returns
        -------
        ele_x
        ele_y
        """
        ele_x = [0.87419355, 0.51290323, 0.48709677, 0.1, 0.12580645, 0.9,
                 0.8483871, 0.46129032, 0.46129032, 0.8483871, 0.9,
                 0.12580645, 0.1, 0.48709677, 0.51290323, 0.87419355]
        ele_y = [0.35789378, 0.3131957, 0.35789378, 0.35789378, 0.3131957,
                 0.3131957, 0.3131957, 0.3131957, 0.5263914, 0.5263914,
                 0.5263914, 0.5263914, 0.47477849, 0.47477849, 0.5263914,
                 0.47477849]
        return np.array(ele_x), np.array(ele_y)

    def do_kcsd(self, ele_pos, pots, k, Rs=np.arange(0.19, 0.3, 0.04)):
        """
        Function that calls the KCSD2D module

        Parameters
        ----------
        ele_pos: numpy array, shape (total_ele)
            electrodes locations/positions
        pots: numpy array, shape (total_ele)
            values of potentials at ele_pos
        k: instance of the class
            instance of TestKCSD1D, TestKCSD2D or TestKCSD3D class

        Returns
        -------
        est_csd: numpy array, shape (est_xres)
            estimated csd (with kCSD method)
        est_pot: numpy array, shape (est_xres)
            estimated potentials
        """
#        num_ele = len(ele_pos)
#        pots = pots.reshape(num_ele, 1)
#        k.cross_validate(Rs=Rs, lambdas=np.array([3.0435374446012946e-13]))
        np.save(os.path.join(self.path, 'pots.npy'), pots)
        k.cross_validate(Rs=Rs)
        est_csd = k.values('CSD')
        est_pot = k.values('POT')
        return est_csd, est_pot

    def calculate_rms(self, test_csd, est_csd):
        """
        Calculates error of reconstruction

        Parameters
        ----------
        test_csd: numpy array
            values of true csd at points of kcsd estimation
        est_csd: numpy array
            csd estimated with kcsd method

        Returns
        -------
        rms: float
            error of reconstruction
        """
        rms = np.linalg.norm((test_csd - est_csd))
        epsilon = 0.0000000001
        rms /= np.linalg.norm(test_csd) + epsilon
        return rms

    def calculate_point_error(self, test_csd, est_csd):
        """
        Calculates error of reconstruction at every point of estimation space

        Parameters
        ----------
        test_csd: numpy array
            values of true csd at points of kcsd estimation
        est_csd: numpy array
            csd estimated with kcsd method

        Returns
        -------
        point_error: numpy array, shape: test_csd.shape
            point error of reconstruction
        """
        epsilon = 0.0000000001
        point_error = np.linalg.norm(test_csd.reshape(test_csd.size, 1) -
                                     est_csd.reshape(est_csd.size, 1),
                                     axis=1)
        point_error /= np.linalg.norm(test_csd.reshape(test_csd.size, 1),
                                      axis=1) + epsilon
        if self.dim != 1:
            point_error = point_error.reshape(test_csd.shape)
        return point_error

    def add_noise(self, seed, pots, level=0.001):
        """
        Adds noise to potentials

        Parameters
        ----------
        pots: numpy array, shape (total_ele)

        Returns
        -------
        pots_noise: numpy array, shape (total_ele)
        """
        rstate = np.random.RandomState(seed)
        noise = level*rstate.normal(np.mean(pots), np.std(pots), len(pots))
        pots_noise = pots + noise
        return pots_noise

    def standard_csd_1d(self, ele_pos, pots):
        """
        Function using standard csd method to estimate current sources
        (calculations according to Eq. 3 (Łęski et al 2007))

        Parameters
        ----------
        ele_pos: numpy array, shape (total_ele)
            electrodes locations/positions
        pots: numpy array, shape (total_ele)
            values of potentials at ele_pos

        Returns
        -------
        tcsd: numpy array, shape (total_ele)
            values of current source density estimated with standard CSD method
        """
        distance_x = abs(ele_pos[0] - ele_pos[1])
        tcsd = np.zeros(len(ele_pos))
        for i in range(1, len(ele_pos) - 1):
            tcsd[i] = -self.sigma * (pots[i - 1] - 2 * pots[i] +
                                     pots[i + 1]) / distance_x**2
        tcsd[0] = -self.sigma * (pots[0] - 2 * pots[0] +
                                 pots[1]) / distance_x**2
        tcsd[len(ele_pos) - 1] = -self.sigma \
            * (pots[len(ele_pos) - 2] - 2 *
               pots[len(ele_pos) - 1] +
               pots[len(ele_pos) - 1]) / distance_x**2
        return tcsd


if __name__ == '__main__':
    print('Invalid usage, use this an inheritable class only')
