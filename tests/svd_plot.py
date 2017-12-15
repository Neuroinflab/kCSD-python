#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:43:48 2017

@author: mkowalska
"""

import numpy as np
import matplotlib.pyplot as plt

u = np.load('/home/mkowalska/Marta/kCSD_results/2017-12-12/20171212-100003/u_svd.npy')
v = np.load('/home/mkowalska/Marta/kCSD_results/2017-12-12/20171212-100003/v_svd.npy')
sigma = np.load('/home/mkowalska/Marta/kCSD_results/2017-12-12/20171212-100003/sigma.npy')
kernel = np.load('/home/mkowalska/Marta/kCSD_results/2017-12-12/20171212-100003/kernel.npy')
k_pot = np.load('/home/mkowalska/Marta/kCSD_results/2017-12-12/20171212-100003/k_pot.npy')
pots = np.load('/home/mkowalska/Marta/kCSD_results/2017-12-12/20171212-100003/pots.npy')


def picard_plot(kernel, pots, sigma, u, v):
    b = pots
    print(b.shape)
    picard = np.zeros(len(sigma))
    picard_norm = np.zeros(len(sigma))
    beta = np.zeros(v.shape)
    for i in range(len(sigma)):
        picard[i] = abs(np.dot(u[:, i].T, b))
#        picard[i] = np.linalg.norm(np.dot(u[:, i].reshape(len(u[:, i]), 1),
#                                          b.reshape((len(pots), 1)).T), ord=1)
#        picard_norm[i] = np.linalg.norm(u[:, i].T*b, ord=1)/sigma[i]
        picard_norm[i] = abs(np.dot(u[:, i].T, b))/sigma[i]
        beta[i] = ((np.dot(u[:, i].T, b))/sigma[i]) * v[i, :]
    return picard, picard_norm, beta
picard_plot(kernel, pots, sigma, u, v)

def plot_svd_u(u, total_ele):
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
    plt.title('Singular vectors of kernels product')
    plt.ylabel('Singular vectors')
#    fig1.savefig(os.path.join(self.path, 'SingularVectorsT' + '.png'))
#    plt.close()
    a = int(total_ele - int(np.sqrt(total_ele))**2)
    if a == 0:
        size = int(np.sqrt(total_ele))
    else:
        size = int(np.sqrt(total_ele)) + 1
    fig2, axs = plt.subplots(int(np.sqrt(total_ele)),
                             size, figsize=(12, 9))
    axs = axs.ravel()
    for i in range(total_ele):
        axs[i].plot(u[:, i], '.')
        axs[i].set_title(str(i+1))
#    fig2.savefig(os.path.join(self.path, 'SingularVectors' + '.png'))
#    plt.close()
    return


def plot_svd_v(v, total_ele):
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
    plt.title('Singular vectors of kernels product')
    plt.ylabel('Singular vectors')
    plt.show()
#    fig1.savefig(os.path.join(self.path, 'SingularVectorsT' + '.png'))
#    plt.close()
    a = int(total_ele - int(np.sqrt(total_ele))**2)
    if a == 0:
        size = int(np.sqrt(total_ele))
    else:
        size = int(np.sqrt(total_ele)) + 1
    fig2, axs = plt.subplots(int(np.sqrt(total_ele)),
                             size, figsize=(12, 9))
    axs = axs.ravel()
    for i in range(total_ele):
        axs[i].plot(v[:, i])
        axs[i].set_title(str(i+1))
#    fig2.savefig(os.path.join(self.path, 'SingularVectors' + '.png'))
#    plt.close()
    return


U, s, V = np.linalg.svd(k_pot)
picard, picard_norm, beta = picard_plot(k_pot, pots, s, U, V)
plt.figure()
plt.plot(s, marker='.', label=r'$\sigma_{i}$')
plt.plot(picard, marker='.', label='$|u(:, i)^{T}*b|$')
plt.plot(picard_norm, marker='.',
         label=r'$\frac{|u(:, i)^{T}*b|}{\sigma_{i}}$')
plt.yscale('log')
plt.legend()
plt.title('Picard plot')
plt.xlabel('i')
plt.show()


plt.figure()
plt.plot(beta.T, marker='.')

plt.figure()
plt.plot(k_pot, marker='.')