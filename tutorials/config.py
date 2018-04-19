from kcsd import csd_profile as csd
from kcsd import MoIKCSD
from kcsd import ValidateKCSD, ValidateKCSD1D, ValidateKCSD2D, ValidateKCSD3D, SpectralStructure
from kcsd import ErrorMap1D, ErrorMap2D, ErrorMap3D


# Function to initialize default parameters
def initialize(value):
    if value == '1D':
        dim = 1
        csd_profile = csd.gauss_1d_mono
        kCSD = ValidateKCSD1D
        error_map = ErrorMap1D
    elif value == '2D':
        dim = 2
        csd_profile = csd.gauss_2d_small
        kCSD = ValidateKCSD2D
        error_map = ErrorMap2D
    else:
        dim = 3
        csd_profile = csd.gauss_3d_small
        kCSD = ValidateKCSD3D
        error_map = ErrorMap3D
    return dim, csd_profile, kCSD, error_map


dim, csd_profile, kCSD, error_map = initialize('1D')


kcsd_options = {1: {'ValidateKCSD1D': ValidateKCSD1D},
                2: {'ValidateKCSD2D': ValidateKCSD2D,
                    'MoIKCSD': MoIKCSD},
                3: {'ValidateKCSD3D': ValidateKCSD3D}}

csd_options = {1: {'monopole gauss': csd.gauss_1d_mono,
                   'dipole gauss': csd.gauss_1d_dipole},
               2: {'quadpole small': csd.gauss_2d_small,
                   'dipole large': csd.gauss_2d_large},
               3: {'gaussian small': csd.gauss_3d_small}}

defaults = {'ValidateKCSD1D': {'R_init': 0.23,
                               'n_src_init': 300,
                               'true_csd_xlims': [0., 1.],
                               'ele_lims': [0.1, 0.9],
                               'kcsd_xlims': [0.1, 0.9],
                               'est_xres': 0.01,
                               'ext_x': 0.0,
                               'sigma': 1.0,
                               'h': 1.0},
            'ValidateKCSD2D': {'R_init': 0.08,
                               'n_src_init': 1000,
                               'est_xres': 0.01, 'est_yres': 0.01,
                               'ext_x': 0.0, 'ext_y': 0.0,
                               'sigma': 1.0,
                               'h': 1.0},
            'MoIKCSD': {'R_init': 0.08,
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
            'ValidateKCSD3D': {'R_init': 0.31,
                               'n_src_init': 300,
                               'est_xres': 0.05, 'est_yres': 0.05, 'est_zres': 0.05,
                               'ext_x': 0.0, 'ext_y': 0.0, 'ext_z': 0.0,
                               'sigma': 1.0,
                               'h': 1.0}}
