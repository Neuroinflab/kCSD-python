# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:12:08 2018

@author: Wladek
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:01:35 2018

@author: Wladek
"""
import numpy as np
import pylab as py
import pandas as pd
import os


py.close('all')
npDir = "./LCurve/"
os.chdir(npDir)

nss = 10
rms_lc = np.zeros((nss,19))
lam_lc = np.zeros((nss,19))
rms_cv = np.zeros((nss,19))
lam_cv = np.zeros((nss,19))
for i in range(nss):
    #os.chdir(npDir+'example_figs/lc_all/lc'+str(i)+'/')
    os.chdir(npDir+'/lc'+str(i)+'/')
    dip = pd.read_excel('lc' + str(i) +'_.xlsx')
    rms_lc[i] = np.array([dip['RMS'].values])
    lam_lc[i] = np.array([dip['Lambda'].values])
    os.chdir(npDir+'example_figs/cv_all/cv'+str(i)+'/')
    dip = pd.read_excel('cv' + str(i) +'_.xlsx')
    rms_cv[i] = np.array([dip['RMS'].values])
    lam_cv[i] = np.array([dip['Lambda'].values])

fig = py.figure(figsize=(12, 10), dpi=100)
ax1 = fig.add_subplot(211)
py.xlabel('',fontsize = 25)
n_spec = np.linspace(1e-2,1,19)

lw = 0.5
mn_rms = np.mean(rms_lc, axis=0)
st_rms = np.std(rms_lc, axis=0)
py.plot(n_spec, mn_rms, marker = 'o', color = 'blue', label = 'l-curve')
py.fill_between(n_spec, mn_rms - st_rms, 
                mn_rms + st_rms, alpha = 0.3, color = 'blue')
mn_rms = np.mean(rms_cv, axis=0)
st_rms = np.std(rms_cv, axis=0)
py.plot(n_spec, mn_rms, marker = 'o', color = 'green', label = 'cross-validation')
py.fill_between(n_spec, mn_rms - st_rms, 
                mn_rms + st_rms, alpha = 0.3, color = 'green')
py.legend(fontsize = 15)
#py.xlabel('Noise',fontsize = 25)
py.ylabel('Estimation error',fontsize = 18)
py.xlabel('Relative noise level',fontsize = 15)
py.xticks(fontsize = 15)
py.yticks(fontsize = 15)
#py.xlim(0, 0.9)
py.ylim(0, 100)

'''second plot'''
ax2 = fig.add_subplot(212)
mn_lam = np.mean(lam_lc, axis=0)
st_lam = np.std(lam_lc, axis=0)
py.plot(n_spec, mn_lam, marker = 'o', color = 'blue', label = 'l-curve')
py.fill_between(n_spec, mn_lam - st_lam,
                mn_lam + st_lam, alpha = 0.3, color = 'blue')
mn_lam = np.mean(lam_cv, axis=0)
st_lam = np.std(lam_cv, axis=0)
py.plot(n_spec, mn_lam, marker = 'o', color = 'green', label = 'cross-validation')
py.fill_between(n_spec, mn_lam - st_lam,
                mn_lam + st_lam, alpha = 0.3, color = 'green')
py.legend(fontsize = 15, loc =2)
py.xticks(fontsize = 15)
py.yticks(fontsize = 15)
#py.xlim(0, 0.9)
py.ticklabel_format(style='sci', axis='y',  scilimits=(0,0))
py.ylabel('Lambda',fontsize = 18)
py.xlabel('Relative noise level',fontsize = 15)
