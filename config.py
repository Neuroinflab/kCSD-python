import csd_profile as CSD
from KCSD import KCSD1D, KCSD2D, KCSD3D, MoIKCSD

# These are the default setting  -can be changed to what ever after UI
# interaction.
dim = 1
csd_profile = CSD.gauss_1d_mono
kCSD = KCSD1D


kcsd_options = {1: {'KCSD1D': KCSD1D},
                2: {'KCSD2D': KCSD2D, 'MoIKCSD': MoIKCSD},
                3: {'KCSD3D': KCSD3D}}

csd_options = {1: {'monopole gauss': CSD.gauss_1d_mono,
                   'dipole gauss': CSD.gauss_1d_dipole},
               2: {'quadpole small': CSD.gauss_2d_small,
                   'dipole large': CSD.gauss_2d_large},
               3: {'gaussian small': CSD.gauss_3d_small}}

defaults = {'KCSD1D': {'R_init': 0.27,
                       'n_src_init': 300,
                       'gdx': 0.01,
                       'xmin': 0.0,
                       'xmax': 1.0,
                       'ext_x': 0.0,
                       'sigma': 1.0,
                       'h': 1.0,
                       'lambd': 0.0},
            'KCSD2D': {'R_init': 0.08,
                       'n_src_init': 1000,
                       'gdx': 0.01, 'gdy': 0.01,
                       'xmin': 0.0, 'xmax': 1.0,
                       'ymin': 0.0, 'ymax': 1.0,
                       'ext_x': 0.0, 'ext_y': 0.0,
                       'sigma': 1.0,
                       'h': 1.0,
                       'lambd': 0.0},
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
            'KCSD3D': {'R_init': 0.31,
                       'n_src_init': 300,
                       'gdx': 0.05, 'gdy': 0.05, 'gdz': 0.05,
                       'xmin': 0.0, 'xmax': 1.0,
                       'ymin': 0.0, 'ymax': 1.0,
                       'zmin': 0.0, 'zmax': 1.0,
                       'ext_x': 0.0, 'ext_y': 0.0, 'ext_z': 0.0,
                       'sigma': 1.0,
                       'lambd': 0.0}}
