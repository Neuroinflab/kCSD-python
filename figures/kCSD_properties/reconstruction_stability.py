"""
@author: mkowalska
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

from kcsd import csd_profile as CSD
from kcsd import ValidateKCSD1D

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
    print('here')
    try:
        os.makedirs(directory)
    except OSError:
        pass
    os.chdir(directory)
    print('bye')


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
SAVE_PATH = "/home/mkowalska/Marta/kCSD_results/" + DAY
makemydir(SAVE_PATH)
TIMESTR = time.strftime("%Y%m%d-%H%M%S")
save_source_code(SAVE_PATH, TIMESTR)

CSD_PROFILE = CSD.gauss_1d_mono
CSD_SEED = 15
N_SRC_INIT = [2, 4, 8, 16, 32, 64, 128, 256, 512]
ELE_LIMS = [0.1, 0.9]  # range of electrodes space
TRUE_CSD_XLIMS = [0., 1.]
TOTAL_ELE = 8

OBJ = []
RMS = []
POINT_ERROR = []
for value in N_SRC_INIT:
    KK = ValidateKCSD1D(CSD_SEED, n_src_init=value, R_init=0.23,
                        ele_lims=ELE_LIMS, true_csd_xlims=TRUE_CSD_XLIMS,
                        sigma=0.3, h=0.25, src_type='gauss')
    obj, rms, point_error = KK.make_reconstruction(CSD_PROFILE, CSD_SEED,
                                                   total_ele=TOTAL_ELE,
                                                   noise=0,
                                                   Rs=np.arange(0.2, 0.5, 0.1))
    OBJ.append(obj)
    RMS.append(rms)
    POINT_ERROR.append(point_error)


plt.figure()
plt.plot(N_SRC_INIT, RMS, '.')
plt.xscale('log')
plt.title('Stability of reconstruction for different number of basis sources')
plt.xlabel('Number of basis sources')
plt.ylabel('RMS')
plt.show()
