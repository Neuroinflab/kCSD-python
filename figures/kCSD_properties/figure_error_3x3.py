"""
@author: mkowalska
"""

import os
import numpy as np
from figure_properties import *
from kCSD_with_reliability_map_2D import matrix_symmetrization
from reliability_map_2D import generate_reliability_map


if __name__ == '__main__':
    path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'kCSDrev-pics')
    large = np.load(path + '/error_maps_2D/data_large_100_3x3.npz')
    small = np.load(path + '/error_maps_2D/data_small_100_3x3.npz')
    data_l = {key: large[key].item() for key in large}
    data_s = {key: small[key].item() for key in small}
    csd_at = data_l['data']['csd_at']
    error_l = data_l['data']['error']
    error_s = data_s['data']['error']
    ele_pos = data_l['data']['ele_pos']

    error_all = np.concatenate((error_l, error_s))
    symm_array_large = matrix_symmetrization(error_l)
    symm_array_small = matrix_symmetrization(error_s)
    symm_array_all = matrix_symmetrization(error_all)
    generate_reliability_map(np.mean(symm_array_all, axis=0), ele_pos,
                             'Reliability_map_random_newRDM_symm_3x3')
    generate_reliability_map(np.mean(symm_array_large, axis=0), ele_pos,
                             'Reliability_map_large_newRDM_symm_3x3')
    generate_reliability_map(np.mean(symm_array_small, axis=0), ele_pos,
                             'Reliability_map_small_newRDM_symm_3x3')
