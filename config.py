import csd_profile as CSD
import sys
from ValidationClassKCSD import ValidationClassKCSD1D, ValidationClassKCSD2D, ValidationClassKCSD3D, SpectralStructure
from ErrorMap import ErrorMap1D, ErrorMap2D, ErrorMap3D
from KCSD import MoIKCSD
sys.path.append('../corelib/')
#from KCSD import KCSD1D, KCSD2D, KCSD3D, MoIKCSD

# Function to initialize default parameters
def initialize(value):
    if value == '1D':
        dim = 1
        csd_profile = CSD.gauss_1d_mono
        kCSD = ValidationClassKCSD1D
        error_map = ErrorMap1D
    elif value == '2D':
        dim = 2
        csd_profile = CSD.gauss_2d_small
        kCSD = ValidationClassKCSD2D
        error_map = ErrorMap2D
    else:
        dim = 3
        csd_profile = CSD.gauss_3d_small
        kCSD = ValidationClassKCSD3D
        error_map = ErrorMap3D
    return dim, csd_profile, kCSD, error_map

dim, csd_profile, kCSD, error_map = initialize('1D')


kcsd_options = {1: {'ValidationClassKCSD1D': ValidationClassKCSD1D},
                2: {'ValidationClassKCSD2D': ValidationClassKCSD2D, 'MoIKCSD': MoIKCSD},
                3: {'ValidationClassKCSD3D': ValidationClassKCSD3D}}

csd_options = {1: {'monopole gauss': CSD.gauss_1d_mono,
                   'dipole gauss': CSD.gauss_1d_dipole},
               2: {'quadpole small': CSD.gauss_2d_small,
                   'dipole large': CSD.gauss_2d_large},
               3: {'gaussian small': CSD.gauss_3d_small}}

defaults = {'ValidationClassKCSD1D': {'R_init': 0.27,
                                      'n_src_init': 300,
                                      'gdx': 0.01,
                                      'xmin': 0.0,
                                      'xmax': 1.0,
                                      'ext_x': 0.0,
                                      'sigma': 1.0,
                                      'h': 1.0,
                                      'lambd': 0.0},
            'ValidationClassKCSD2D': {'R_init': 0.08,
                                      'n_src_init': 1000,
                                      'gdx': 0.01, 'gdy': 0.01,
                                      'xmin': 0.0, 'xmax': 1.0,
                                      'ymin': 0.0, 'ymax': 1.0,
                                      'ext_x': 0.0, 'ext_y': 0.0,
                                      'sigma': 1.0,
                                      'h': 1.0,
                                      'lambd': 0.0},
            'ValidationClassMoIKCSD': {'R_init': 0.08,
                                       'n_src_init': 1000,
                                       'gdx': 0.01, 'gdy': 0.01,
                                       'xmin': 0.0, 'xmax': 1.0,
                                       'ymin': 0.0, 'ymax': 1.0,
                                       'ext_x': 0.0, 'ext_y': 0.0,
                                       'sigma': 1.0,
                                       'h': 1.0,
                                       'lambd': 0.0,
                                       'MoI_iters': 20,
                                       'sigma_S': 5.0},
            'ValidationClassKCSD3D': {'R_init': 0.31,
                                      'n_src_init': 300,
                                      'gdx': 0.05, 'gdy': 0.05, 'gdz': 0.05,
                                      'xmin': 0.0, 'xmax': 1.0,
                                      'ymin': 0.0, 'ymax': 1.0,
                                      'zmin': 0.0, 'zmax': 1.0,
                                      'ext_x': 0.0, 'ext_y': 0.0, 'ext_z': 0.0,
                                      'sigma': 1.0,
                                      'lambd': 0.0,
                                      'h': 1.0 }}
