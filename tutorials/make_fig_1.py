from __future__ import division, print_function
import run_LFP
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from distutils.spawn import find_executable, spawn
import shutil
import subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corelib import sKCSD3D
import corelib.utility_functions as utils
import corelib.loadData as ld

if find_executable('nrnivmodl') is not None:
    for path in ['x86_64', 'i686', 'powerpc']:
        if os.path.isdir(path):
            shutil.rmtree(path)
    spawn([find_executable('nrnivmodl')])
    subprocess.call(["nrnivmodl", "sinsyn.mod"])
else:
    print("nrnivmodl script not found in PATH, thus NEURON .mod files could" +
"not be compiled, and LFPy.test() functions will fail")
    
