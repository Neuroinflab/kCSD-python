"""
@author: mkowalska
"""
import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from kcsd import csd_profile as CSD
from kcsd import ValidateKCSD, ValidateKCSD1D, SpectralStructure, KCSD1D

__abs_file__ = os.path.abspath(__file__)


def makemydir(directory):
    """
    Creates a new folder if it doesn't exist

    Parameters
    ----------
    directory: string
        directory

    Returns
    -------
    None
    """
    try:
        os.makedirs(directory)
    except OSError:
        pass
    os.chdir(directory)


def save_source_code(save_path, timestr):
    """
    Saves the source code.

    Parameters
    ----------
    save_path: string
        directory
    timestr: float

    Returns
    -------
    None
    """
    with open(save_path + '/source_code_' + str(timestr), 'w') as sf:
        sf.write(open(__file__).read())


def csd_profile(x, seed):
    '''Function used for adding multiple 1D gaussians.

    Parameters
    ----------
    x: numpy array
        x coordinates of true source profile.
    seed: list [r, mu]

    Returns
    -------
    gauss: numpy array
        Gaussian profile for given R and M.
    '''
    r = seed[0]
    mu = seed[1]
    STDDEV = r/3.0
    gauss = (np.exp(-((x - mu)**2)/(2 * STDDEV**2)) /
             (np.sqrt(2 * np.pi) * STDDEV)**1)
    gauss /= np.max(gauss)
    return gauss


def targeted_basis(val, csd_at, true_csd, ele_pos, pots, n_src, R, MU,
                   true_csd_xlims, ele_lims, h=0.25, sigma=0.3,
                   csd_res=100):
    '''
    Function investigating kCSD analysis for targeted bases.

    Parameters
    ----------
    val: object of the class ValidateKCSD.
    csd_at: numpy array
        Coordinates of ground truth data.
    true_csd: numpy array
        Values of ground truth data (true_csd).
    ele_pos: numpy array
        Locations of electrodes.
    pots: numpy array
        Potentials measured (calculated) on electrodes.
    n_src: int
        Number of basis sources.
    R: float
        Thickness of the groundtruth source.
    MU: float
        x coordinate of maximum ampliude of groundtruth source.
    true_csd_xlims: list
        Boundaries for ground truth space.
    ele_lims: list
        Boundaries for electrodes placement.
    h: float
        Thickness of analyzed cylindrical slice.
        Default: 0.25.
    sigma: float
        Space conductance of the medium.
        Default: 0.3.
    csd_res: int
        Resolution of ground truth.
        Default: 100.

    Returns
    -------
    obj: object of the class KCSD1D
    k: object of the class ValidateKCSD1D
    '''
    k = ValidateKCSD1D(1, n_src_init=n_src, R_init=0.23,
                       ele_lims=ele_lims,
                       true_csd_xlims=true_csd_xlims, sigma=sigma, h=h,
                       src_type='gauss')
    obj, est_csd = k.recon(pots, ele_pos, method='cross-validation',
                           Rs=np.arange(0.2, 0.5, 0.1))
    test_csd = csd_profile(obj.estm_x, [R, MU])
    rms = val.calculate_rms(test_csd, est_csd)
    title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R,
                                                            rms)
    fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
    save_as = (SAVE_PATH + '/A_basis_on_[0_1]')
    fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
    plt.close()
    return obj, k


def simulate_data(csd_profile, true_csd_xlims, R, MU, total_ele, ele_lims,
                  h=0.25, sigma=0.3, csd_res=100):
    '''
    Generates groundtruth profiles and interpolates potentials.

    Parameters
    ----------
    csd_profile: function
            Function to produce csd profile.
    true_csd_xlims: list
        Boundaries for ground truth space.
    R: float
        Thickness of the groundtruth source.
    MU: float
        x coordinate of maximum ampliude of groundtruth source.
    total_ele: int
        Number of electrodes.
    ele_lims: list
        Boundaries for electrodes placement.
    h: float
        Thickness of analyzed cylindrical slice.
        Default: 0.25.
    sigma: float
        Space conductance of the medium.
        Default: 0.3.
    csd_res: int
        Resolution of ground truth.
        Default: 100.

    Returns
    -------
    csd_at: numpy array
        Coordinates of ground truth data.
    true_csd: numpy array
        Values of ground truth data (true_csd).
    ele_pos: numpy array
        Locations of electrodes.
    pots: numpy array
        Potentials measured (calculated) on electrodes.
    val: object of the class ValidateKCSD
    '''
    val = ValidateKCSD(1)
    csd_at = np.linspace(true_csd_xlims[0], true_csd_xlims[1], csd_res)
    true_csd = csd_profile(csd_at, [R, MU])
    ele_pos = val.generate_electrodes(total_ele=total_ele, ele_lims=ele_lims)
    pots = val.calculate_potential(true_csd, csd_at, ele_pos, h, sigma)
    return csd_at, true_csd, ele_pos, pots, val


def structure_investigation(csd_profile, true_csd_xlims, n_src, R, MU,
                            total_ele, ele_lims, h=0.25, sigma=0.3,
                            csd_res=100):
    '''
    .

    Parameters
    ----------
    csd_profile: function
        Function to produce csd profile.
    true_csd_xlims: list
        Boundaries for ground truth space.
    n_src: int
        Number of basis sources.
    R: float
        Thickness of the groundtruth source.
    MU: float
        x coordinate of maximum ampliude of groundtruth source.
    total_ele: int
        Number of electrodes.
    ele_lims: list
        Boundaries for electrodes placement.
    h: float
        Thickness of analyzed cylindrical slice.
        Default: 0.25.
    sigma: float
        Space conductance of the medium.
        Default: 0.3.
    csd_res: int
        Resolution of ground truth.
        Default: 100.

    Returns
    -------
    obj: object of the class KCSD1D
    '''
    val = ValidateKCSD(1)
    csd_at, true_csd, ele_pos, pots, val = simulate_data(csd_profile,
                                                         true_csd_xlims, R, MU,
                                                         total_ele, ele_lims,
                                                         h=h, sigma=sigma)
    obj, k = targeted_basis(val, csd_at, true_csd, ele_pos, pots, n_src, R, MU,
                            true_csd_xlims, ele_lims, h=0.25,
                            sigma=0.3, csd_res=100)
    return obj


def plot_eigenvalues(eigenvalues, save_path, title):
    '''
    Creates plot of eigenvalues of kernel matrix (k_pot).

    Parameters
    ----------
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    save_path: string
        Directory.
    title: string
        Title of the plot.

    Returns
    -------
    None
    '''
    fig = plt.figure()
    plt.plot(eigenvalues, '--', marker='.')
    plt.title('Eigenvalue decomposition of kernel matrix. ele_lims=basis_lims')
    plt.xlabel('Number of components')
    plt.ylabel('Eigenvalues')
    plt.show()
    save_as = (save_path + '/eigenvalues_for_' + title)
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_eigenvectors(eigenvectors, save_path, title):
    """
    Creates plot of eigenvectors of kernel matrix (k_pot).

    Parameters
    ----------
    eigenvectors: numpy array
        Eigenvectors of k_pot matrix.
    save_path: string
        Directory.
    title: string
        Title of the plot.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('Eigenvalue decomposition of kernel matrix for different '
                 'number of basis sources')
    for i in range(eigenvectors.shape[1]):
        plt.subplot(int(eigenvectors.shape[1]/2) + 1, 2, i + 1)
        plt.plot(eigenvectors[:, i].T, '--', marker='.')
        plt.ylabel('Eigenvectors')
        plt.title('v_' + str(i + 1))
    plt.xlabel('Number of components')
    plt.tight_layout()
    plt.show()
    save_as = (save_path + '/eigenvectors_for_' + title)
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def modified_bases(k, pots, ele_pos, n_src, title, h=0.25, sigma=0.3,
                   gdx=0.035, ext_x=0, xmin=0, xmax=1):
    '''
    Parameters
    ----------
    k: object of the class ValidateKCSD1D
    pots: numpy array
        Potentials measured (calculated) on electrodes.
    ele_pos: numpy array
        Locations of electrodes.
    n_src: int
        Number of basis sources.
    title: string
        Title of the plot.
    h: float
        Thickness of analyzed cylindrical slice.
        Default: 0.25.
    sigma: float
        Space conductance of the medium.
        Default: 0.3.
    gdx: float
        Space increments in the estimation space.
        Default: 0.035.
    ext_x: float
        Length of space extension: xmin-ext_x ... xmax+ext_x.
        Default: 0.
    xmin: float
        Boundaries for CSD estimation space.
    xmax: float
        boundaries for CSD estimation space.

    Returns
    -------
    None
    '''
    pots = pots.reshape((len(ele_pos), 1))
    obj_m = KCSD1D(ele_pos, pots, src_type='gauss', sigma=sigma, h=h, gdx=gdx,
                   n_src_init=n_src, ext_x=ext_x, xmin=xmin, xmax=xmax)
    obj_m.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
    est_csd = obj_m.values('CSD')
    test_csd = csd_profile(obj_m.estm_x, [R, MU])
    rms = val.calculate_rms(test_csd, est_csd)
    title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj_m.lambd,
                                                            obj_m.R, rms)
    fig = k.make_plot(csd_at, true_csd, obj_m, est_csd, ele_pos, pots, title)
    save_as = (SAVE_PATH + '/basis_on_[0_0_5]')
    fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
    plt.close()
    ss = SpectralStructure(obj_m)
    eigenvectors, eigenvalues = ss.evd()
    title = 'A_basis_lims_[0_0_5]'
    plot_eigenvalues(eigenvalues, SAVE_PATH, title)
    plot_eigenvectors(eigenvectors, SAVE_PATH, title)


home = expanduser('~')
DAY = datetime.datetime.now()
DAY = DAY.strftime('%Y%m%d')
TIMESTR = time.strftime("%H%M%S")
SAVE_PATH = home + "/kCSD_results/" + DAY + '/' + TIMESTR
makemydir(SAVE_PATH)
save_source_code(SAVE_PATH, time.strftime("%Y%m%d-%H%M%S"))

CSD_PROFILE = CSD.gauss_1d_mono
CSD_SEED = 15
N_SRC = 64
ELE_LIMS = [0, 1.]  # range of electrodes space
TRUE_CSD_XLIMS = [0., 1.]
TOTAL_ELE = 12

#  A
R = 0.2
MU = 0.25
csd_at, true_csd, ele_pos, pots, val = simulate_data(csd_profile,
                                                     TRUE_CSD_XLIMS, R, MU,
                                                     TOTAL_ELE, ELE_LIMS)
obj, k = targeted_basis(val, csd_at, true_csd, ele_pos, pots, N_SRC, R, MU,
                        TRUE_CSD_XLIMS, ELE_LIMS)
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'A_basis_lims_[0_1]'
plot_eigenvalues(eigenvalues, SAVE_PATH, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, title)

#  A.2
title = 'A_basis_lims_[0_0_5]'
modified_bases(k, pots, ele_pos, N_SRC, title, h=0.25, sigma=0.3, gdx=0.035,
               ext_x=0, xmin=0, xmax=0.5)

#  A.2.b
title = 'A_basis_lims_[0_0_5]_less_sources'
modified_bases(k, pots, ele_pos, N_SRC/2, title, h=0.25, sigma=0.3, gdx=0.035,
               ext_x=0, xmin=0, xmax=0.5)

#  B
TRUE_CSD_XLIMS = [0., 1.5]
R = 0.2
MU = 1.25
csd_at, true_csd, ele_pos, pots, val = simulate_data(csd_profile,
                                                     TRUE_CSD_XLIMS, R, MU,
                                                     TOTAL_ELE, ELE_LIMS)
obj, k = targeted_basis(val, csd_at, true_csd, ele_pos, pots, N_SRC, R, MU,
                        TRUE_CSD_XLIMS, ELE_LIMS)
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'B_basis_lims_[0_1]'
plot_eigenvalues(eigenvalues, SAVE_PATH, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, title)

#  B.2
title = 'B_basis_lims_[1_1_5]'
modified_bases(k, pots, ele_pos, N_SRC, title, h=0.25, sigma=0.3, gdx=0.035,
               ext_x=0, xmin=1, xmax=1.5)

#  B.2.b
title = 'B_basis_lims_[1_1_5]_less_sources'
modified_bases(k, pots, ele_pos, N_SRC/2, title, h=0.25, sigma=0.3, gdx=0.035,
               ext_x=0, xmin=1, xmax=1.5)

#  B.3
title = 'B_basis_lims_[0_1_5]'
modified_bases(k, pots, ele_pos, N_SRC, title, h=0.25, sigma=0.3, gdx=0.035,
               ext_x=0, xmin=0, xmax=1.5)
