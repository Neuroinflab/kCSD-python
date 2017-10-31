import csd_profile as CSD
import KCSD

# These are the default setting  -can be changed to what ever after UI
# interaction.
dim = 1
csd_profile = CSD.gauss_1d_mono
kCSD = KCSD.KCSD1D


kcsd_options = {1: {'KCSD1D': KCSD.KCSD1D},
                2: {'KCSD2D': KCSD.KCSD2D, 'MoiKCSD': KCSD.MoIKCSD},
                3: {'KCSD3D': KCSD.KCSD3D}}

csd_options = {1: {'monopole gauss': CSD.gauss_1d_mono,
                   'dipole gauss': CSD.gauss_1d_dipole},
               2: {'quadpole small': CSD.gauss_2d_small,
                   'dipole large': CSD.gauss_2d_large},
               3: {'gaussian small': CSD.gauss_3d_small}}

defaults = {'KCSD1D': {},
            'KCSD2D': {},
            'MoIKCSD': {},
            'KCSD3D': {}}
