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


def csd_profile(x, seed):
    '''Function used for adding multiple 1D gaussians'''
    R = seed[0]
    MU = seed[1]
    STDDEV = R/3.0
    gauss = (np.exp(-((x - MU)**2)/(2 * STDDEV**2)) /
             (np.sqrt(2 * np.pi) * STDDEV)**1)
    gauss /= np.max(gauss)
    return gauss


x = np.linspace(0, 1., 100)
gauss = csd_profile(x, [0.2, 0.25])
plt.plot(gauss)

#### A ####
val = ValidateKCSD(1)
R = 0.2
MU = 0.25
csd_at = np.linspace(TRUE_CSD_XLIMS[0], TRUE_CSD_XLIMS[1], 100)
true_csd = csd_profile(csd_at, [R, MU])
ele_pos = val.generate_electrodes(total_ele=TOTAL_ELE, ele_lims=ELE_LIMS)
h = 0.25
sigma = 0.3
pots = val.calculate_potential(true_csd, csd_at, ele_pos, h, sigma)
k = ValidateKCSD1D(CSD_SEED, n_src_init=N_SRC, R_init=0.23, ele_lims=ELE_LIMS,
                   true_csd_xlims=TRUE_CSD_XLIMS, sigma=sigma, h=h,
                   src_type='gauss')
obj, est_csd = k.recon(pots, ele_pos, method='cross-validation',
                       Rs=np.arange(0.2, 0.5, 0.1))
rms = val.calculate_rms(true_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/A_basis_on_[0_1]')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()


def plot_eigenvalues(eigenvalues, save_path, n_src, title):
    fig = plt.figure()
    plt.plot(eigenvalues, '--', marker='.')
    plt.title('Eigenvalue decomposition of kernel matrix. ele_lims=basis_lims')
    plt.xlabel('Number of components')
    plt.ylabel('Eigenvalues')
    plt.show()
    save_as = (save_path + '/eigenvalues_for_' + title)
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()


def plot_eigenvectors(eigenvectors, save_path, n_src, title):
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('Eigenvalue decomposition of kernel matrix. ele_lims=basis_lims')
    for i in range(eigenvectors.shape[1]):
        plt.subplot(int(eigenvectors.shape[1]/2) + 1, 2, i + 1)
        plt.plot(eigenvectors[:, i].T, '--', marker='.')
        plt.ylabel('Eigenvectors')
        plt.title('v_' + str(i + 1))
#    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Number of components')
    plt.show()
    save_as = (save_path + '/eigenvectors_for_' + title)
    fig.savefig(os.path.join(save_path, save_as+'.png'))
    plt.close()

ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'A_basis_lims_[0_1]'
plot_eigenvalues(eigenvalues, SAVE_PATH, N_SRC, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, N_SRC, title)

############## A.2 ####################
pots = pots.reshape((len(ele_pos), 1))
obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=sigma,
             h=h, n_src_init=N_SRC, ext_x=0,
             gdx=0.035, xmin=0,
             xmax=0.5)
obj.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
est_csd = obj.values('CSD')
test_csd = csd_profile(obj.estm_x, [R, MU])
rms = val.calculate_rms(test_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/basis_on_[0_0_5]')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'A_basis_lims_[0_0_5]'
plot_eigenvalues(eigenvalues, SAVE_PATH, N_SRC, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, N_SRC, title)

############## A.2.b ####################
pots = pots.reshape((len(ele_pos), 1))
obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=sigma,
             h=h, n_src_init=int(N_SRC/2), ext_x=0,
             gdx=0.035, xmin=0,
             xmax=0.5)
obj.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
est_csd = obj.values('CSD')
test_csd = csd_profile(obj.estm_x, [R, MU])
rms = val.calculate_rms(test_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/basis_on_[0_0_5]_less_sources')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'A_basis_lims_[0_0_5]_less_sources'
plot_eigenvalues(eigenvalues, SAVE_PATH, N_SRC, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, N_SRC, title)


#### B ####
TRUE_CSD_XLIMS = [0., 1.5]
val = ValidateKCSD(1)
R = 0.2
MU = 1.25
csd_at = np.linspace(TRUE_CSD_XLIMS[0], TRUE_CSD_XLIMS[1], 150)
true_csd = csd_profile(csd_at, [R, MU])
ele_pos = val.generate_electrodes(total_ele=TOTAL_ELE, ele_lims=ELE_LIMS)
h = 0.25
sigma = 0.3
pots = val.calculate_potential(true_csd, csd_at, ele_pos, h, sigma)
k = ValidateKCSD1D(CSD_SEED, n_src_init=N_SRC, R_init=0.23, ele_lims=ELE_LIMS,
                   true_csd_xlims=TRUE_CSD_XLIMS, sigma=sigma, h=h,
                   src_type='gauss', kcsd_xlims=[0, 1])
obj, est_csd = k.recon(pots, ele_pos, method='cross-validation',
                       Rs=np.arange(0.2, 0.5, 0.1))
rms = val.calculate_rms(true_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/B_basis_on_[0_1]')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()

############## B.2 ####################
pots = pots.reshape((len(ele_pos), 1))
obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=sigma,
             h=h, n_src_init=N_SRC, ext_x=0,
             gdx=0.035, xmin=1,
             xmax=1.5)
obj.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
est_csd = obj.values('CSD')
test_csd = csd_profile(obj.estm_x, [R, MU])
rms = val.calculate_rms(test_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/basis_on_[1_1_5]')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'B_basis_lims_[1_1_5]'
plot_eigenvalues(eigenvalues, SAVE_PATH, N_SRC, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, N_SRC, title)

############## B.2.b ####################
pots = pots.reshape((len(ele_pos), 1))
obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=sigma,
             h=h, n_src_init=int(N_SRC/2), ext_x=0,
             gdx=0.035, xmin=1,
             xmax=1.5)
obj.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
est_csd = obj.values('CSD')
test_csd = csd_profile(obj.estm_x, [R, MU])
rms = val.calculate_rms(test_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/B_basis_on_[1_1_5]_less_sources')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'B_basis_lims_[1_1_5]_less_sources'
plot_eigenvalues(eigenvalues, SAVE_PATH, N_SRC, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, N_SRC, title)

############## B.3 ####################
pots = pots.reshape((len(ele_pos), 1))
obj = KCSD1D(ele_pos, pots, src_type='gauss', sigma=sigma,
             h=h, n_src_init=int(N_SRC), ext_x=0,
             gdx=0.035, xmin=0,
             xmax=1.5)
obj.cross_validate(Rs=np.arange(0.2, 0.5, 0.1))
est_csd = obj.values('CSD')
test_csd = csd_profile(obj.estm_x, [R, MU])
rms = val.calculate_rms(test_csd, est_csd)
title = "Lambda: %0.2E; R: %0.2f; RMS_Error: %0.2E;" % (obj.lambd, obj.R, rms)
fig = k.make_plot(csd_at, true_csd, obj, est_csd, ele_pos, pots, title)
save_as = (SAVE_PATH + '/B_basis_on_[0_1_5]')
fig.savefig(os.path.join(SAVE_PATH, save_as+'.png'))
plt.close()
ss = SpectralStructure(obj)
eigenvectors, eigenvalues = ss.evd()
title = 'B_basis_lims_[0_1_5]'
plot_eigenvalues(eigenvalues, SAVE_PATH, N_SRC, title)
plot_eigenvectors(eigenvectors, SAVE_PATH, N_SRC, title)
