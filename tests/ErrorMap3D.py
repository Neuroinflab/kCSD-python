#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:09:20 2017

@author: mkowalska
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super

from builtins import range
from future import standard_library

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.ma as ma

from TestKCSD3D import TestKCSD3D
import csd_profile as CSD
sys.path.append('../tests')
from KCSD import KCSD2D
from save_paths import where_to_save_results, where_to_save_source_code, \
    TIMESTR

standard_library.install_aliases()
__abs_file__ = os.path.abspath(__file__)

try:
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count() - 1
    parallel_available = True
except ImportError:
    parallel_available = False


class ErrorMap3D(TestKCSD3D):
    """
    Class that produces error map for 3D CSD reconstruction
    """
    def __init__(self, csd_profile, csd_seed, **kwargs):
        self.test_res = 256
        self.n = kwargs.get('n', 1)
        super(ErrorMap3D, self).__init__(csd_profile, csd_seed, **kwargs)
#        csd_profile = CSD.gauss_2d_error_map
        self.calculate_error_map_r(csd_profile, **kwargs)
        return


def save_source_code(save_path, TIMESTR):
    """
    Wilson G., et al. (2014) Best Practices for Scientific Computing,
    PLoS Biol 12(1): e1001745
    """
    with open(save_path + 'source_code_' + str(TIMESTR), 'w') as sf:
        sf.write(open(__abs_file__).read())
    return


def makemydir(directory):
    """
    Creates directory if it doesn't exist
    """
    try:
        os.makedirs(directory)
    except OSError:
        pass
    os.chdir(directory)


if __name__ == '__main__':
    makemydir(where_to_save_source_code)
    save_source_code(where_to_save_source_code, TIMESTR)
    csd_profile = CSD.gauss_3d_small
    csd_seed = 10
    total_ele = 36
    a = ErrorMap3D(csd_profile, csd_seed, total_ele=total_ele, h=50.,
                   sigma=1., nr_basis=400, config='regular', n=15)
