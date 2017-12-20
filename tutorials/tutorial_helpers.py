
import numpy as np
import config
from csd_profile import csd_available_dict

from scipy.integrate import simps

def generate_csd(csd_profile, csd_at=None, seed=0):
    """
    csd_at is like an np.mgrid type data
    """
    if csd_at is None:
        if config.dim == 1:
            csd_at = np.mgrid[0:1:100j]
        elif config.dim == 2:
            csd_at = np.mgrid[0:1:100j,
                              0:1:100j]
        else:
            csd_at = np.mgrid[0:1:100j,
                              0:1:100j,
                              0:1:100j]
    else:
        if config.dim == 1:
            if not csd_at.ndim == config.dim:
                print('Invalid csd_at and dim')
        if config.dim > 1:
            if not csd_at.shape[0] == config.dim:
                print('Invalid csd_at and dim')
    if csd_profile not in csd_available_dict[config.dim]:
        print('Incorrect csd_profile selection')
    return csd_at, csd_profile(csd_at, seed=seed)


def generate_electrodes(ele_lim=None, ele_res=None):
    if ele_lim is None:
        ele_lim = [0.1, 0.9]
    if ele_res is None:         # reduce electrode resolution
        if config.dim == 1:
            ele_res = 30
        elif config.dim == 2:
            ele_res = 10
        else:
            ele_res = 5
    if config.dim == 1:
        x = np.mgrid[ele_lim[0]:ele_lim[1]:np.complex(0, ele_res)]
        ele_pos = x.flatten().reshape(ele_res, 1)
    elif config.dim == 2:
        x, y = np.mgrid[ele_lim[0]:ele_lim[1]:np.complex(0, ele_res),
                        ele_lim[0]:ele_lim[1]:np.complex(0, ele_res)]
        ele_pos = np.vstack((x.flatten(),
                             y.flatten())).T
    else:
        x, y, z = np.mgrid[ele_lim[0]:ele_lim[1]:np.complex(0, ele_res),
                           ele_lim[0]:ele_lim[1]:np.complex(0, ele_res),
                           ele_lim[0]:ele_lim[1]:np.complex(0, ele_res)]
        ele_pos = np.vstack((x.flatten(),
                             y.flatten(),
                             z.flatten())).T
    return ele_pos.shape[0], ele_pos


def integrate_1D(x0, csd_x, csd, h):
    m = np.sqrt((csd_x-x0)**2 + h**2) - abs(csd_x-x0)
    y = csd * m
    I = simps(y, csd_x)
    return I


def integrate_2D(x, y, xlim, ylim, csd, h, xlin, ylin, X, Y):
    """
    X,Y - parts of meshgrid - Mihav's implementation
    """
    Ny = ylin.shape[0]
    m = np.sqrt((x - X)**2 + (y - Y)**2)     # construct 2-D integrand
    m[m < 0.0000001] = 0.0000001             # I increased acuracy
    y = np.arcsinh(2*h / m) * csd            # corrected
    I = np.zeros(Ny)                         # do a 1-D integral over every row
    for i in xrange(Ny):
        I[i] = simps(y[:, i], ylin)          # I changed the integral
    F = simps(I, xlin)                       # then an integral over the result
    return F


def integrate_3D(x, y, z, xlim, ylim, zlim, csd, xlin, ylin, zlin, X, Y, Z):
    """
    X,Y - parts of meshgrid - Mihav's implementation
    """
    Nz = zlin.shape[0]
    Ny = ylin.shape[0]
    m = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    m[m < 0.0000001] = 0.0000001
    z = csd / m
    Iy = np.zeros(Ny)
    for j in xrange(Ny):
        Iz = np.zeros(Nz)
        for i in xrange(Nz):
            Iz[i] = simps(z[:, j, i], zlin)
        Iy[j] = simps(Iz, ylin)
    F = simps(Iy, xlin)
    return F


def calculate_potential(csd_at, csd, measure_locations, h, sigma=1.):
    if config.dim == 1:
        pots = np.zeros(len(measure_locations))
        for ii in range(len(measure_locations)):
            pots[ii] = integrate_1D(measure_locations[ii], csd_at, csd, h)
        pots *= 1/(2.*sigma)  # eq.: 26 from Potworowski et al
        pots = pots.reshape((len(measure_locations), 1))
    elif config.dim == 2:
        csd_x = csd_at[0, :, :]
        csd_y = csd_at[1, :, :]
        xlin = csd_x[:, 0]
        ylin = csd_y[0, :]
        xlims = [xlin[0], xlin[-1]]
        ylims = [ylin[0], ylin[-1]]
        num_ele = measure_locations.shape[0]
        ele_xx = measure_locations[:, 0]
        ele_yy = measure_locations[:, 1]
        pots = np.zeros(num_ele)
        for ii in range(num_ele):
            pots[ii] = integrate_2D(ele_xx[ii], ele_yy[ii],
                                    xlims, ylims, csd, h,
                                    xlin, ylin, csd_x, csd_y)
        pots /= 2*np.pi*sigma
        pots = pots.reshape(num_ele, 1)
    else:
        csd_x = csd_at[0, :, :, :]
        csd_y = csd_at[1, :, :, :]
        csd_z = csd_at[2, :, :, :]
        xlin = csd_x[:, 0, 0]
        ylin = csd_y[0, :, 0]
        zlin = csd_z[0, 0, :]
        xlims = [xlin[0], xlin[-1]]
        ylims = [ylin[0], ylin[-1]]
        zlims = [zlin[0], zlin[-1]]
        sigma = 1.0
        ele_xx = measure_locations[:, 0]
        ele_yy = measure_locations[:, 1]
        ele_zz = measure_locations[:, 2]
        num_ele = measure_locations.shape[0]
        pots = np.zeros(num_ele)
        for ii in range(num_ele):
            pots[ii] = integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                    xlims, ylims, zlims, csd,
                                    xlin, ylin, zlin,
                                    csd_x, csd_y, csd_z)
        pots /= 4*np.pi*sigma
        pots = pots.reshape(num_ele, 1)
    return pots
