from __future__ import print_function, division

import os
import sys
import numpy as np

from kcsd import sKCSD, sample_data_path
from kcsd import utility_functions as utils

data_dir = os.path.join(sample_data_path, 'ball_and_stick_8')

scaling_factor = 1000000
data = utils.LoadData(data_dir)
ele_pos = data.ele_pos/scaling_factor
pots = data.LFP
params = {}
morphology = data.morphology 
morphology[:,2:6] = morphology[:,2:6].copy()/scaling_factor



k = sKCSD(ele_pos, pots,morphology, n_src_init=1000, src_type='gauss_lim')
k.cross_validate()

if sys.version_info >= (3, 0):
    path = os.path.join(data_dir,"preprocessed_data/Python_3")
else:
    path = os.path.join(data_dir,"preprocessed_data/Python_2")
    
if not os.path.exists(path):
    print("Creating",path)
    os.makedirs(path)
        
utils.save_sim(path,k)
