# -*- coding: utf-8 -*-
"""
These are some useful functions used in CSD methods,
They include CSD source profiles to be used as ground truths,
placement of electrodes in 1D, 2D and 3D., etc

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.spatial import distance
import os
import pickle
from scipy import interpolate
import json
import sys

try:
    from joblib.parallel import Parallel, delayed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

    
raise_errror = """Unknown electrode position file format.
Load either one column file (or a one row file) with x positions,
y positions, z positions, or a 3 column file with x and y and z positions.
"""

def check_for_duplicated_electrodes(elec_pos):
    """Checks for duplicate electrodes

    Parameters
    ----------
    elec_pos : np.array

    Returns
    -------
    has_duplicated_elec : Boolean

    """
    unique_elec_pos = np.vstack(list({tuple(row) for row in elec_pos}))
    has_duplicated_elec = unique_elec_pos.shape == elec_pos.shape
    return has_duplicated_elec


def distribute_srcs_1D(X, n_src, ext_x, R_init):
    """Distribute sources in 1D equally spaced

    Parameters
    ----------
    X : np.arrays
        points at which CSD will be estimated
    n_src : int
        number of sources to be included in the model
    ext_x : floats
        how much should the sources extend the area X
    R_init : float
        Same as R in 1D case

    Returns
    -------
    X_src : np.arrays
        positions of the sources
    R : float
        effective radius of the basis element

    """
    X_src = np.mgrid[(np.min(X) - ext_x):(np.max(X) + ext_x):
                     np.complex(0, n_src)]
    R = R_init
    return X_src, R


def distribute_srcs_2D(X, Y, n_src, ext_x, ext_y, R_init):
    """Distribute n_src's in the given area evenly

    Parameters
    ----------
    X, Y : np.arrays
        points at which CSD will be estimated
    n_src : int
        demanded number of sources to be included in the model
    ext_x, ext_y : floats
        how should the sources extend the area X, Y
    R_init : float
        demanded radius of the basis element

    Returns
    -------
    X_src, Y_src : np.arrays
        positions of the sources
    nx, ny : ints
        number of sources in directions x,y
        new n_src = nx * ny may not be equal to the demanded number of sources
    R : float
        effective radius of the basis element

    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lx_n = Lx + (2 * ext_x)
    Ly_n = Ly + (2 * ext_y)
    [nx, ny, Lx_nn, Ly_nn, ds] = get_src_params_2D(Lx_n, Ly_n, n_src)
    ext_x_n = (Lx_nn - Lx) / 2
    ext_y_n = (Ly_nn - Ly) / 2
    X_src, Y_src = np.mgrid[(np.min(X) - ext_x_n):(np.max(X) + ext_x_n):
                            np.complex(0, nx),
                            (np.min(Y) - ext_y_n):(np.max(Y) + ext_y_n):
                            np.complex(0, ny)]
    # d = round(R_init / ds)
    R = R_init  # R = d * ds
    return X_src, Y_src, R


def get_src_params_2D(Lx, Ly, n_src):
    """Distribute n_src sources evenly in a rectangle of size Lx * Ly

    Parameters
    ----------
    Lx, Ly : floats
        lengths in the directions x, y of the area,
        the sources should be placed
    n_src : int
        demanded number of sources

    Returns
    -------
    nx, ny : ints
        number of sources in directions x, y
        new n_src = nx * ny may not be equal to the demanded number of sources
    Lx_n, Ly_n : floats
        updated lengths in the directions x, y
    ds : float
        spacing between the sources

    """
    S = Lx * Ly
    S_unit = S / n_src
    L_unit = S_unit**(1. / 2.)
    nx = np.ceil(Lx / L_unit)
    ny = np.ceil(Ly / L_unit)
    ds = Lx / (nx - 1)
    Lx_n = (nx - 1) * ds
    Ly_n = (ny - 1) * ds
    return (nx, ny, Lx_n, Ly_n, ds)


def distribute_srcs_3D(X, Y, Z, n_src, ext_x, ext_y, ext_z, R_init):
    """Distribute n_src sources evenly in a rectangle of size Lx * Ly * Lz

    Parameters
    ----------
    X, Y, Z : np.arrays
        points at which CSD will be estimated
    n_src : int
        desired number of sources we want to include in the model
    ext_x, ext_y, ext_z : floats
        how should the sources extend over the area X,Y,Z
    R_init : float
        demanded radius of the basis element

    Returns
    -------
    X_src, Y_src, Z_src : np.arrays
        positions of the sources in 3D space
    nx, ny, nz : ints
        number of sources in directions x,y,z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources

    R : float
        updated radius of the basis element

    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lz = np.max(Z) - np.min(Z)
    Lx_n = Lx + 2 * ext_x
    Ly_n = Ly + 2 * ext_y
    Lz_n = Lz + 2 * ext_z
    (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds) = get_src_params_3D(Lx_n,
                                                              Ly_n,
                                                              Lz_n,
                                                              n_src)
    ext_x_n = (Lx_nn - Lx) / 2
    ext_y_n = (Ly_nn - Ly) / 2
    ext_z_n = (Lz_nn - Lz) / 2
    X_src, Y_src, Z_src = np.mgrid[(np.min(X) - ext_x_n):(np.max(X) + ext_x_n):
                                   np.complex(0, nx),
                                   (np.min(Y) - ext_y_n):(np.max(Y) + ext_y_n):
                                   np.complex(0, ny),
                                   (np.min(Z) - ext_z_n):(np.max(Z) + ext_z_n):
                                   np.complex(0, nz)]
    # d = np.round(R_init / ds)
    R = R_init
    return (X_src, Y_src, Z_src, R)


def get_src_params_3D(Lx, Ly, Lz, n_src):
    """Helps to evenly distribute n_src sources in a cuboid of size Lx * Ly * Lz

    Parameters
    ----------
    Lx, Ly, Lz : floats
        lengths in the directions x, y, z of the area,
        the sources should be placed
    n_src : int
        demanded number of sources to be included in the model

    Returns
    -------
    nx, ny, nz : ints
        number of sources in directions x, y, z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources
    Lx_n, Ly_n, Lz_n : floats
        updated lengths in the directions x, y, z
    ds : float
        spacing between the sources (grid nodes)

    """
    V = Lx * Ly * Lz
    V_unit = V / n_src
    L_unit = V_unit**(1. / 3.)
    nx = np.ceil(Lx / L_unit)
    ny = np.ceil(Ly / L_unit)
    nz = np.ceil(Lz / L_unit)
    ds = Lx / (nx - 1)
    Lx_n = (nx - 1) * ds
    Ly_n = (ny - 1) * ds
    Lz_n = (nz - 1) * ds
    return (nx, ny, nz, Lx_n, Ly_n, Lz_n, ds)


def get_estm_places(wsp_plot, gdx, gdy, gdz):
    """Distribute sources under virtual electrode surface
    default method for electrode surface interpolation is "nearest"
    form scipy.interpolate.griddata function

    Parameters
    ----------
    wsp_plot : np.arrays
        electrode XYZ coordinates
    gdx, gdy, gdz : ints
        distance beetwen estimation/source points 

    Returns
    -------
    est_xyz : np.array
        coordinates of points where we want to estimate CSD (under electordes)

    """
    xmin = np.min(wsp_plot[0])
    xmax = np.max(wsp_plot[0])
    ymin = np.min(wsp_plot[1])
    ymax = np.max(wsp_plot[1])
    zmin = np.min(wsp_plot[2])
    zmax = np.max(wsp_plot[2])

    lnx = int((xmax - xmin)/gdx)
    lny = int((ymax - ymin)/gdy)
    lnz = int((zmax - zmin)/gdz)

    grid_x, grid_y = np.mgrid[xmin:xmax:lnx*1j, ymin:ymax:lny*1j]
    points = np.array([wsp_plot[0], wsp_plot[1]]).T
    values = wsp_plot[2]
    grid_z = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
    estm_x, estm_y, estm_z = np.mgrid[xmin:xmax:np.complex(0,int(lnx)), 
                                      ymin:ymax:np.complex(0,int(lny)),
                                      zmin:zmax:np.complex(0,int(lnz))]
    mask_mtrx = np.zeros(estm_x.shape)
    for z in range(lnz):
        mask_mtrx[:,:,z] = estm_z[:,:,z]<grid_z
    estm_z_new = mask_mtrx * estm_z

    xpos = estm_x.ravel()
    ypos = estm_y.ravel()
    zpos = estm_z_new.ravel()

    idx_to_remove = np.where(zpos == 0)
    xpos = np.delete(xpos, idx_to_remove)
    ypos = np.delete(ypos, idx_to_remove)
    zpos = np.delete(zpos, idx_to_remove)

    est_xyz = np.array([xpos,ypos,zpos])
    return est_xyz

def L_model_fast(k_pot, pots, lamb, i):
    """Method for Fast L-curve computation

    Parameters
    ----------
    k_pot : np.array
    pots : list
    lambd : list
    i : int

    Returns
    -------
    modelnorm : float
    residual : float

    """
    k_inv = np.linalg.inv(k_pot + lamb*np.identity(k_pot.shape[0]))
    beta_new = np.dot(k_inv, pots)
    V_est = np.dot(k_pot, beta_new)
    modelnorm = np.einsum('ij,ji->i', beta_new.T, V_est)
    residual = np.linalg.norm(V_est - pots)
    modelnorm = np.max(modelnorm)
    return modelnorm, residual


def parallel_search(k_pot, pots, lambdas, n_jobs=4):
    """Method for Parallel L-curve computation

    Parameters
    ----------
    k_pot : np.array
    pots : list
    lambdas : list

    Returns
    -------
    modelnormseq : list
    residualseq : list

    """
    if PARALLEL_AVAILABLE:
        jobs = (delayed(L_model_fast)(k_pot, pots, lamb, i)
                for i, lamb in enumerate(lambdas))
        modelvsres = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    else:
        # Please verify this!
        modelvsres = []
        for i, lamb in enumerate(lambdas):
            modelvsres.append(L_model_fast(k_pot, pots, lamb, i))
    modelnormseq, residualseq = zip(*modelvsres)
    return modelnormseq, residualseq


