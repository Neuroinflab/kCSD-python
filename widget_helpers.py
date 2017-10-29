import ipywidgets as widgets
import config
import csd_profile as CSD

def change_dim(value):
    if dim_select.value == '1D':
        config.dim = 1
    elif dim_select.value == '2D':
        config.dim = 2
    else:
        config.dim = 3
    update_csd_types()

    
def change_csd(value):
    if config.dim == 1:
        if csd_select.value == 'monopole gauss':
            csd_profile = CSD.gauss_1d_mono
        elif csd_select.value == 'dipole gauss':
            csd_profile = CSD.gauss_1d_dipole
    elif config.dim == 2:
        if csd_select.value == 'quadpole small':
            csd_profile = CSD.gauss_2d_small
        elif csd_select.value == 'dipole large':
            csd_profile = CSD.gauss_2d_large
    elif config.dim == 3:
        if csd_select.value == 'gaussian small':
            csd_profile = CSD.gauss_3d_small
    config.csd_profile = csd_profile
    

def update_csd_types():
    if config.dim == 1:
        csd_select.options = ['monopole gauss', 'dipole gauss']
    elif config.dim == 2:
        csd_select.options = ['quadpole small', 'dipole large']
    else:
        csd_select.options = ['gaussian small']


csd_select = widgets.ToggleButtons(options=['monopole gauss', 'dipole gauss'],
                                   description='True source type',
                                   button_style='')
                                   

dim_select = widgets.ToggleButtons(options=['1D', '2D', '3D'],
                                   description='Dimensions of the setup:',
                                   disabled=False,
                                   button_style='', tooltips=['Laminar probes',
                                                              'MEA like flat electrodes',
                                                              'Utah array or SEEG'])
dim_select.observe(change_dim, 'value')
csd_select.observe(change_csd, 'value')
