# -*- coding: utf-8 -*-
"""
These are some useful functions used in CSD methods,
They include CSD source profiles to be used as ground truths,
placement of electrodes in 1D, 2D and 3D., etc
These scripts are based on Grzegorz Parka's,
Google Summer of Code 2014, INFC/pykCSD
This was written by :
Michal Czerwinski, Chaitanya Chintaluri
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import os
import pickle
from scipy import interpolate
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
raise_errror = """Unknown electrode position file format.
Load either one column file (or a one row file) with x positions,
y positions, z positions, or a 3 column file with x and y and z positions.
"""


def load_swc(path):
    """Load swc file"""
    morphology = np.loadtxt(path)
    return morphology


def save_sim(path, k):

    est_csd = k.values('CSD', transformation=None)
    est_pot = k.values("POT", transformation=None)
    np.save(os.path.join(path, "csd.npy"), est_csd)
    print("Save csd, ", os.path.join(path, "csd.npy"))
    np.save(os.path.join(path, "pot.npy"), est_pot)
    print("Save pot, ", os.path.join(path, "pot.npy"))

    cell_data = {
        'morphology': k.cell.morphology.tolist(),
        'ele_pos': k.cell.ele_pos.tolist(),
        'n_src': k.cell.n_src
    }
    with open(os.path.join(path, "cell_data"), 'w') as handle:
        json.dump(cell_data, handle)


def load_sim(path):
    est_csd = np.load(os.path.join(path, "csd.npy"))
    est_pot = np.load(os.path.join(path, "pot.npy"))

    try:
        with open(os.path.join(path, "cell_data"), 'r') as handle:
            cell_data = json.load(handle)
    except Exception as error:
        print('Could not load', os.path.join(path, "cell_data"))
        return est_csd, est_pot, None

    from . import sKCSDcell

    morphology = np.array(cell_data['morphology'])
    ele_pos = np.array(cell_data['ele_pos'])
    cell_obj = sKCSDcell(morphology, ele_pos, cell_data['n_src'])
    return est_csd, est_pot, cell_obj


def load_elpos(path):
    """Load electrode postions.

    File format: text file, one column, x of all the electrodes, y of
    all the electrodes, z of all the electrodes

    """
    raw_ele_pos = np.loadtxt(path)
    if len(raw_ele_pos.shape) == 1:
        if raw_ele_pos.shape[0] % 3:
            raise Exception(raise_error)
        else:
            n_el = raw_ele_pos.shape[0]//3
            ele_pos = np.zeros(shape=(n_el, 3))
            ele_pos[:, 0] = raw_ele_pos[:n_el]
            ele_pos[:, 1] = raw_ele_pos[n_el:2*n_el]
            ele_pos[:, 2] = raw_ele_pos[2*n_el:]
    elif len(raw_ele_pos.shape) == 2:
        if raw_ele_pos.shape[1] == 1:
            if raw_ele_pos.shape[0] % 3:
                raise Exception(raise_error)
            else:
                n_el = raw_ele_pos.shape[0]/3
                ele_pos = np.zeros(shape=(n_el, 3))
                ele_pos[:, 0] = raw_ele_pos[:n_el]
                ele_pos[:, 1] = raw_ele_pos[n_el:2*n_el]
                ele_pos[:, 2] = raw_ele_pos[2*n_el:]
        elif raw_ele_pos.shape[0] == 1:
            if raw_ele_pos.shape[1] % 3:
                raise Exception(raise_error)
            else:
                n_el = raw_ele_pos.shape[1]/3
                ele_pos = np.zeros(shape=(n_el, 3))
                ele_pos[:, 0] = raw_ele_pos[:n_el]
                ele_pos[:, 1] = raw_ele_pos[n_el:2*n_el]
                ele_pos[:, 2] = raw_ele_pos[2*n_el:]
        elif raw_ele_pos.shape[1] == 3:
            ele_pos = raw_ele_pos
        else:
            raise Exception(raise_error)
    else:
        raise Exception(raise_error)
    return ele_pos


def check_for_duplicated_electrodes(elec_pos):
    """Checks for duplicate electrodes
    Parameters
    ----------
    elec_pos : np.array
    Returns
    -------
    has_duplicated_elec : Boolean
    """
    unique_elec_pos = np.vstack({tuple(row) for row in elec_pos})
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
    coeff = [Ly, Lx - Ly, -Lx * n_src]
    rts = np.roots(coeff)
    r = [r for r in rts if type(r) is not complex and r > 0]
    nx = r[0]
    ny = n_src / nx
    ds = Lx / (nx - 1)
    nx = np.floor(nx) + 1
    ny = np.floor(ny) + 1
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
