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
