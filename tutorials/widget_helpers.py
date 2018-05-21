import ipywidgets as widgets
import config


def change_dim(value):
    if dim_select.value == '1D':
        config.dim, config.csd_profile, config.kCSD, config.visibility_map = \
            config.initialize(dim_select.value)
    elif dim_select.value == '2D':
        config.dim, config.csd_profile, config.kCSD, config.visibility_map = \
            config.initialize(dim_select.value)
    else:
        config.dim, config.csd_profile, config.kCSD, config.visibility_map = \
            config.initialize(dim_select.value)
    update_csd_types()
    update_kcsd_types()
    update_accordion()


def change_csd(value):
    config.csd_profile = config.csd_options[config.dim][csd_select.value]


def change_kcsd(value):
    config.kCSD = config.kcsd_options[config.dim][kcsd_select.value]


def update_csd_types():
    csd_select.options = list(config.csd_options[config.dim].keys())


def update_kcsd_types():
    kcsd_select.options = list(config.kcsd_options[config.dim].keys())


dim_select = widgets.ToggleButtons(options=['1D', '2D', '3D'],
                                   description='Dimensions of the setup:',
                                   disabled=False,
                                   button_style='',
                                   tooltips=['Laminar probes',
                                             'MEA like flat electrodes',
                                             'Utah array or SEEG'])

csd_select = widgets.ToggleButtons(options=list(config.csd_options[1].keys()),
                                   description='True source type',
                                   button_style='')

kcsd_select = widgets.ToggleButtons(options=list(config.kcsd_options[1].keys()),
                                    description='KCSD method',
                                    button_style='')

nr_ele_select = widgets.BoundedIntText(value=10,
                                       min=1,
                                       max=200,
                                       step=1,
                                       description='Select nr of electrodes:',
                                       disabled=False)

nr_broken_ele = widgets.BoundedIntText(value=5,
                                       min=1,
                                       max=nr_ele_select.value - 1,
                                       step=1,
                                       description='Select number of broken electrodes:',
                                       disabled=False)

noise_select = widgets.FloatSlider(value=0.,
                                   min=0,
                                   max=100,
                                   step=0.1,
                                   description='Noise level [%]:',
                                   disabled=False,
                                   continuous_update=False,
                                   orientation='horizontal',
                                   readout=True,
                                   readout_format='.1f')

regularization_select = widgets.Select(options=['cross-validation', 'L-curve'],
                                       value='cross-validation',
                                       description='Regularization method:',
                                       disabled=False)

def create_text_wid(txt, val):
    wid = widgets.FloatText(value=val,
                            description=txt + ':',
                            disabled=False)
    return wid


def wid_lists(var_list):
    def_dict = config.defaults[config.kCSD.__name__]
    wid_list = []
    for var in var_list:
        try:
            wid_list.append(create_text_wid(var, def_dict[var]))
            big_wid = widgets.VBox(wid_list)
        except KeyError:
            pass
    return big_wid


def refresh_accordion_wids():
    src_ass = wid_lists(['R_init', 'n_src_init', 'lambd'])
    est_pos = wid_lists(['xmin', 'xmax',
                         'ymin', 'ymax',
                         'zmin', 'zmax',
                         'ext_x', 'ext_y', 'ext_z',
                         'gdx', 'gdy', 'gdz'])
    med_ass = wid_lists(['simga', 'h', 'sigma_S', 'MoI_iters'])
    return [src_ass, est_pos, med_ass]


accordion = widgets.Accordion(children=refresh_accordion_wids())
accordion.set_title(0, 'Source assumptions')
accordion.set_title(1, 'Estimate positions')
accordion.set_title(2, 'Medium assumptions')


def update_accordion():
    accordion.children = refresh_accordion_wids()
    accordion.set_title(0, 'Source assumptions')
    accordion.set_title(1, 'Estimate positions')
    accordion.set_title(2, 'Medium assumptions')


dim_select.observe(change_dim, 'value')
csd_select.observe(change_csd, 'value')
kcsd_select.observe(change_kcsd, 'value')
