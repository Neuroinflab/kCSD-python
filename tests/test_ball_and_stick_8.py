from __future__ import print_function, division
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from corelib import sKCSD
from sKCSD_paper import loadData as ld
from corelib import utility_functions as utils
try:
    data_dir = sys.argv[1]
except IndexError:
    data_dir = "Data/ball_and_stick_8"




scaling_factor = 1000000
data = ld.Data(data_dir)
ele_pos = data.ele_pos/scaling_factor
pots = data.LFP
params = {}
morphology = data.morphology 
morphology[:,2:6] = morphology[:,2:6].copy()/scaling_factor



k = sKCSD.sKCSD(ele_pos, pots,morphology, n_src_init=1000, src_type='gauss_lim')
k.cross_validate()

if sys.version_info >= (3, 0):
    path = os.path.join(data_dir,"preprocessed_data/Python_3")
else:
    path = os.path.join(data_dir,"preprocessed_data/Python_2")
    
if not os.path.exists(path):
    print("Creating",path)
    os.makedirs(path)
        
utils.save_sim(path,k)
