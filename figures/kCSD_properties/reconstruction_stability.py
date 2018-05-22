"""
@author: mkowalska
"""
import os
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


DAY = datetime.datetime.now()
DAY = DAY.strftime('%Y%m%d')
TIMESTR = time.strftime("%H%M%S")
SAVE_PATH = "/home/mkowalska/Marta/kCSD_results/" + DAY + '/' + TIMESTR
makemydir(SAVE_PATH)
save_source_code(SAVE_PATH, time.strftime("%Y%m%d-%H%M%S"))

CSD_PROFILE = CSD.gauss_1d_mono
CSD_SEED = 15
N_SRC = [2, 4, 8, 16, 32, 64, 128, 256, 512]
ELE_LIMS = [0.1, 0.9]  # range of electrodes space
TRUE_CSD_XLIMS = [0., 1.]
TOTAL_ELE = 10


def stability_M(csd_profile, csd_seed, n_src, ele_lims, true_csd_xlims,
                total_ele):
    obj_all = []
    rms = np.zeros((len(n_src)))
    point_error_all = []
    eigenvectors = np.zeros((len(n_src), total_ele, total_ele))
    eigenvalues = np.zeros((len(n_src), total_ele))
    for indx, i in enumerate(n_src):
        KK = ValidateKCSD1D(csd_seed, n_src_init=i, R_init=0.23,
                            ele_lims=ele_lims, true_csd_xlims=true_csd_xlims,
                            sigma=0.3, h=0.25, src_type='gauss')
        obj, rms[indx], point_error = KK.make_reconstruction(csd_profile,
                                                             csd_seed,
                                                             total_ele=total_ele,
                                                             noise=0,
                                                             Rs=np.arange(0.2, 0.5, 0.1))
        ss = SpectralStructure(obj)
        eigenvectors[indx], eigenvalues[indx] = ss.evd()
        point_error_all.append(point_error)
        obj_all.append(obj)
    return obj_all, rms, point_error_all, eigenvalues, eigenvectors


OBJ, RMS, POINT_ERROR, eigenval, eigenvec = stability_M(CSD_PROFILE, CSD_SEED,
                                                        N_SRC, ELE_LIMS,
                                                        TRUE_CSD_XLIMS,
                                                        TOTAL_ELE)
k_pot_list = []
k_interp_cross_list = []
for i in range(len(OBJ)):
    k_pot_list.append(OBJ[i].k_pot)
    k_interp_cross_list.append(OBJ[i].k_interp_cross)


def plot_M(n_src_init, rms, save_path):
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
    fig = plt.figure()
    for indx, i in enumerate(n_src):
        plt.plot(eigenvalues[indx], '--', marker='.', label='M='+str(i))
    plt.legend()
    plt.title('Eigenvalue decomposition of kernel matrix for different number '
              'of basis sources')
    plt.xlabel('Number of components')
    plt.ylabel('Eigenvalues')
    plt.show()
    save_as = (save_path + '/eigenvalues_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_eigenvectors(eigenvectors, save_path, n_src):
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('Eigenvalue decomposition of kernel matrix for different '
                 'number of basis sources')
    for i in range(eigenvectors.shape[2]):
        plt.subplot(int(eigenvectors.shape[2]/2) + 1, 2, i + 1)
        for idx, j in enumerate(n_src):
            plt.plot(eigenvectors[idx, :, i].T, '--', marker='.',
                     label='M='+str(j))
        plt.ylabel('Eigenvectors')
        plt.title('v_' + str(i + 1))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Number of components')
    plt.show()
    save_as = (save_path + '/eigenvectors_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_k_interp_cross(k_icross, save_path, n_src):
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('Vectors of cross kernel matrix for different number '
                 'of basis sources')
    for i in range(k_icross[0].shape[1]):
        plt.subplot(int(len(n_src)/2) + 1, 2, i + 1)
        for idx, j in enumerate(n_src):
            plt.plot(k_icross[idx][:, i], '--', marker='.',
                     label='k='+str(i + 1))
            plt.title('n_src = ' + str(j))
        plt.ylabel('Cross kernel')
    plt.legend()
    plt.xlabel('Number of components')
    plt.ylabel('Cross kernel')
    plt.show()
    save_as = (save_path + '/cross_kernel_for_different_M')
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


#plot_M(N_SRC, RMS, SAVE_PATH)
#plot_eigenvalues(eigenval, SAVE_PATH, N_SRC)
#plot_eigenvectors(eigenvec, SAVE_PATH, N_SRC)
plot_k_interp_cross(k_interp_cross_list, SAVE_PATH, N_SRC)
