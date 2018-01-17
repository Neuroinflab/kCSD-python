from __future__ import print_function, division
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from corelib import sKCSD3D
from corelib import loadData as ld
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

xmin, ymin, zmin, xmax,ymax,zmax = -20/scaling_factor,-20/scaling_factor,-100/scaling_factor,20/scaling_factor,20/scaling_factor,600/scaling_factor
print(xmin, ymin, zmin, xmax,ymax,zmax)
gdx = (xmax-xmin)/100.
gdy = (ymax-ymin)/100.
gdz = (zmax-zmin)/10.

k = sKCSD3D.sKCSD3D(ele_pos, pots,morphology,
            gdx=gdx, gdy=gdx, gdz=gdz,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
            n_src_init=1000, src_type='gauss_lim')
k.cross_validate()

if sys.version_info >= (3, 0):
    path = os.path.join(data_dir,"preprocessed_data/Python_3")
else:
    path = os.path.join(data_dir,"preprocessed_data/Python_2")
    
if not os.path.exists(path):
    print("Creating",path)
    os.makedirs(path)
        
utils.save_sim(path,k)
