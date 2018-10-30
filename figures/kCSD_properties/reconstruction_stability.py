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
from kcsd import ValidateKCSD1D, SpectralStructure

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


def stability_M(csd_profile, csd_seed, n_src, ele_lims, true_csd_xlims,
                total_ele):
    """
    Investigates stability of reconstruction for different number of basis
    sources

    Parameters
    ----------
    csd_profile: function
        Function to produce csd profile.
    csd_seed: int
        Seed for random generator to choose random CSD profile.
    n_src: int
        Number of basis sources.
    ele_lims: list
        Boundaries for electrodes placement.
    true_csd_xlims: list
        Boundaries for ground truth space.
    total_ele: int
        Number of electrodes.
    Returns
    -------
    obj_all: class object
    rms: float
        Normalized error of reconstruction.
    point_error_all: numpy array
        Normalized error of reconstruction calculated separetly at every point
        point of estimation space.
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    eigenvectors: numpy array
        Eigen vectors of k_pot matrix.
    """
    obj_all = []
    rms = np.zeros((len(n_src)))
    point_error_all = []
    eigenvectors = np.zeros((len(n_src), total_ele, total_ele))
    eigenvalues = np.zeros((len(n_src), total_ele))
    for i, value in enumerate(n_src):
        KK = ValidateKCSD1D(csd_seed, n_src_init=value, R_init=0.23,
                            ele_lims=ele_lims, true_csd_xlims=true_csd_xlims,
                            sigma=0.3, h=0.25, src_type='gauss', est_xres=0.01)
        obj, rms[i], point_error = KK.make_reconstruction(csd_profile,
                                                          csd_seed,
                                                          total_ele=total_ele,
                                                          noise=0,
                                                          Rs=np.arange(0.2,
                                                                       0.5,
                                                                       0.1))
        ss = SpectralStructure(obj)
        eigenvectors[i], eigenvalues[i] = ss.evd()
        point_error_all.append(point_error)
        obj_all.append(obj)
    return obj_all, rms, point_error_all, eigenvalues, eigenvectors


def plot_M(n_src_init, rms, save_path):
    """
    Creates plot of relationship between RMS error and different number of
    basis sources

    Parameters
    ----------
    n_src_init: list
        List of number of basis sources.
    rms: numpy array
        Error of reconstruction.
    save_path: string
        Directory.
    timestr: float
        Time.

    Returns
    -------
    None
    """
    fig = plt.figure()
    plt.plot(n_src_init, rms, '--', marker='.')
    plt.xscale('log')
    plt.title('Stability of reconstruction for different number of basis '
              'sources')
    plt.xlabel('Number of basis sources')
    plt.ylabel('RMS')
    plt.show()
    save_as = (save_path + '/RMS_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_eigenvalues(eigenvalues, save_path, n_src):
    """
    Creates plot of eigenvalues of kernel matrix (k_pot) for different number
    of basis sources

    Parameters
    ----------
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure()
    for indx, i in enumerate(n_src):
        plt.plot(eigenvalues[indx], '--', marker='.', label='M='+str(i))
    plt.legend()
#    plt.title('Eigenvalue decomposition of kernel matrix for different number '
#              'of basis sources')
    plt.xlabel('Number of components')
    plt.ylabel('Eigenvalues')
    plt.yscale('log')
    save_as = (save_path + '/eigenvalues_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_max_eigenvalue_M(eigenvalues, save_path, n_src):
    """
    Creates plot of eigenvalues of kernel matrix (k_pot) for different number
    of basis sources

    Parameters
    ----------
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(8, 6))
    plt.plot(n_src, eigenvalues[:, 0], '--', marker='.', label=r'$\mu_1$')
    plt.legend()
    plt.title('First eigenvalue in the function of different number of basis'
              'sources')
    plt.xlabel('Number of basis sources')
    plt.xscale('log')
    plt.ylabel('Eigenvalues')
    plt.yscale('log')
    save_as = (save_path + '/max_eigenvalue_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_eigenvectors(eigenvectors, save_path, n_src):
    """
    Creates plot of eigenvectors of kernel matrix (k_pot) for different number
    of basis sources

    Parameters
    ----------
    eigenvectors: numpy array
        Eigenvectors of k_pot matrix.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(15, 15))
#    plt.suptitle('Eigenvalue decomposition of kernel matrix for different '
#                 'number of basis sources')
    for i in range(eigenvectors.shape[2]):
        plt.subplot(int(eigenvectors.shape[2]/2) + 1, 2, i + 1)
        for idx, j in enumerate(n_src):
            plt.plot(eigenvectors[idx, :, i].T, '--', marker='.',
                     label='M='+str(j))
        plt.ylabel('Eigenvectors')
        plt.title(r'$v_' + str(i + 1) + '$')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Number of components')
    plt.tight_layout()
    plt.show()
    save_as = (save_path + '/eigenvectors_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_k_interp_cross(k_icross, save_path, n_src):
    """
    Creates plot of vectors of cross kernel matrix (k_interp_cross) for
    different number of basis sources

    Parameters
    ----------
    k_icross: numpy array
        List of cross kernel matrixes for different number of basis sources.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(k_icross[0].shape[1] + 5,
                              k_icross[0].shape[1] + 5))
#    plt.suptitle('Vectors of cross kernel matrix for different number '
#                 'of basis sources')
    for i in range(k_icross[0].shape[1]):
        plt.subplot(int(k_icross[0].shape[1]/2) + 1, 2, i + 1)
        for idx, j in enumerate(n_src):
            plt.plot(k_icross[idx][:, i], '--', marker='.',
                     label='M='+str(j))
            plt.title(r'$\tilde{K}_' + str(i + 1) + '$')
        plt.ylabel('Cross kernel')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Number of estimation points')
    plt.tight_layout()
    plt.show()
    save_path = save_path + '/cross_kernel'
    makemydir(save_path)
    save_as = (save_path + '/cross_kernel_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_eigenvalue_lambda(eigenvalues, lambd, save_path, n_src):
    """
    Creates plot of eigenvalues of kernel matrix (k_pot) with lambda for
    different number of basis sources

    Parameters
    ----------
    eigenvalues: numpy array
        Eigenvalues of k_pot matrix.
    lambd: list
        Regularization parameter.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(7, 7))
    x = np.arange(1, eigenvalues.shape[1] + 1)
    for indx, i in enumerate(n_src):
        plt.plot(x, 1/(eigenvalues[indx] + lambd[indx]), '--', marker='.',
                 label='M='+str(i))
    plt.legend()
    plt.title(r'$\frac{1}{(\mu_j + \lambda)}$')
    plt.xlabel('Components number j')
    plt.ylabel(r'1/($\mu_j + \lambda)$')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    save_path = save_path + '/cross_kernel'
    makemydir(save_path)
    save_as = (save_path + '/eigenvalues_coefficients_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_k_interp_cross_v(k_icross, eigenvectors, save_path, n_src):
    """
    Creates plot of product of cross kernel vectors and eigenvectors for
    different number of basis sources

    Parameters
    ----------
    k_icross: numpy array
        List of cross kernel matrixes for different number of basis sources.
    eigenvectors: numpy array
        Eigenvectors of k_pot matrix.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(k_icross[0].shape[1] + 5,
                              k_icross[0].shape[1] + 5))
#    plt.suptitle('Vectors of cross kernel and eigenvectors product for '
#                 'different number of basis sources')
    for i in range(k_icross[0].shape[1]):
        plt.subplot(int(k_icross[0].shape[1]/2) + 1, 2, i + 1)
        for idx, j in enumerate(n_src):
            plt.plot(np.dot(k_icross[idx], eigenvectors[idx, :, i]), '--',
                     marker='.', label='M='+str(j))
            plt.title(r'$\tilde{K}*v_' + str(i) + '$')
#        plt.ylabel(r'$\tilde{K}*v_' + str(i) + '$')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Number of estimation points')
    plt.tight_layout()
    plt.show()
    save_path = save_path + '/cross_kernel'
    makemydir(save_path)
    save_as = (save_path + '/cross_kernel_eigenvector_product_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_k_pot(k_pot, save_path, n_src):
    """
    Creates plot of vectors of kernel matrix (k_pot) for
    different number of basis sources

    Parameters
    ----------
    k_pot: numpy array
        List of kernel matrixes for different number of basis sources.
    save_path: string
        Directory.
    n_src: list
        List of number of basis sources.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(k_pot[0].shape[1] + 5,
                              k_pot[0].shape[1] + 5))
#    plt.suptitle('Vectors of kernel matrix for different number '
#                 'of basis sources')
    for i in range(k_pot[0].shape[1]):
        plt.subplot(int(k_pot[0].shape[1]/2) + 1, 2, i + 1)
        for idx, j in enumerate(n_src):
            plt.plot(k_pot[idx][:, i], '--', marker='.',
                     label='M='+str(j))
            plt.title(r'$K_' + str(i + 1) + '$')
        plt.ylabel('Kernel')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Number of components')
    plt.tight_layout()
    plt.show()
    save_path = save_path + '/kernel'
    makemydir(save_path)
    save_as = (save_path + '/kernel_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


if __name__ == '__main__':
    HOME = expanduser('~')
    DAY = datetime.datetime.now()
    DAY = DAY.strftime('%Y%m%d')
    TIMESTR = time.strftime("%H%M%S")
    SAVE_PATH = HOME + "/kCSD_results/" + DAY + '/' + TIMESTR
    makemydir(SAVE_PATH)
    save_source_code(SAVE_PATH, time.strftime("%Y%m%d-%H%M%S"))

    CSD_PROFILE = CSD.gauss_1d_mono
    CSD_SEED = 15
#    N_SRC = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    N_SRC = [2, 8, 16, 512]
    ELE_LIMS = [0.1, 0.9]  # range of electrodes space
    TRUE_CSD_XLIMS = [0., 1.]
    TOTAL_ELE = 10
    OBJ, RMS, POINT_ERROR, eigenval, eigenvec = stability_M(CSD_PROFILE,
                                                            CSD_SEED,
                                                            N_SRC, ELE_LIMS,
                                                            TRUE_CSD_XLIMS,
                                                            TOTAL_ELE)
    k_pot_list = []
    k_interp_cross_list = []
    lambdas = []
    for index in range(len(OBJ)):
        k_pot_list.append(OBJ[index].k_pot)
        k_interp_cross_list.append(OBJ[index].k_interp_cross)
        lambdas.append(OBJ[index].lambd)

    plot_M(N_SRC, RMS, SAVE_PATH)
    plot_eigenvalues(eigenval, SAVE_PATH, N_SRC)
    plot_max_eigenvalue_M(eigenval, SAVE_PATH, N_SRC)
    plot_eigenvectors(eigenvec, SAVE_PATH, N_SRC)
    plot_k_interp_cross(k_interp_cross_list, SAVE_PATH, N_SRC)
    plot_k_interp_cross_v(k_interp_cross_list, eigenvec, SAVE_PATH, N_SRC)
    plot_eigenvalue_lambda(eigenval, lambdas, SAVE_PATH, N_SRC)
    plot_k_pot(k_pot_list, SAVE_PATH, N_SRC)
