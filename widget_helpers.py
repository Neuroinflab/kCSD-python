import ipywidgets as widgets
import config


def change_dim(value):
    if dim_select.value == '1D':
        config.dim = 1
    elif dim_select.value == '2D':
        config.dim = 2
    else:
        config.dim = 3
    update_csd_types()
    update_kcsd_types()


def change_csd(value):
    config.csd_profile = config.csd_options[config.dim][csd_select.value]


def change_kcsd(value):
    config.kCSD = config.kcsd_options[config.dim][kcsd_select.value]


def update_csd_types():
    csd_select.options = config.csd_options[config.dim].keys()


def update_kcsd_types():
    kcsd_select.options = config.kcsd_options[config.dim].keys()


dim_select = widgets.ToggleButtons(options=['1D', '2D', '3D'],
                                   description='Dimensions of the setup:',
                                   disabled=False,
                                   button_style='',
                                   tooltips=['Laminar probes',
                                             'MEA like flat electrodes',
                                             'Utah array or SEEG'])

csd_select = widgets.ToggleButtons(options=config.csd_options[1].keys(),
                                   description='True source type',
                                   button_style='')

kcsd_select = widgets.ToggleButtons(options=config.kcsd_options[1].keys(),
                                    description='KCSD method',
                                    button_style='')





dim_select.observe(change_dim, 'value')
csd_select.observe(change_csd, 'value')
kcsd_select.observe(change_kcsd, 'value')
