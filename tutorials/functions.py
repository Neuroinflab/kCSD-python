from __future__ import division

import numpy as np
import os
import run_LFP

def make_fig_names(fname_base):
    
    if not os.path.exists('Figures'):
        print("Creating",'Figures')
        os.makedirs('Figures')
        
    return os.path.join('Figures',fname_base)

def simulate(fname_base,**kwargs):
    
    morphology = kwargs.pop("morphology",1)
    simulate_what = kwargs.pop("simulate_what",1)
    electrode_orientation = kwargs.pop("electrode_orientation",2)
    electrode_distribution = kwargs.pop("electrode_distribution",1)

    colnb = kwargs.pop("colnb",4)
    rownb = kwargs.pop("rownb",4)
    xmin = kwargs.pop("xmin",-200)
    xmax = kwargs.pop("xmax",200)
    ymin = kwargs.pop("ymin",-200)
    ymax = kwargs.pop("ymax",200)
    tstop = kwargs.pop("tstop",100)
    seed = kwargs.pop("seed",1988)
    weight = kwargs.pop("weight",.01)
    n_syn = kwargs.pop("n_syn",100)
    fname = fname_base+'_rows_%s'%rownb
    
    if kwargs:
            raise TypeError('Invalid keyword arguments:', kwargs.keys())
        
    c = run_LFP.CellModel(morphology=morphology,cell_name=fname,colnb=colnb,rownb=rownb,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,tstop=tstop,seed=seed,weight=weight,n_syn=n_syn,electrode_distribution=electrode_distribution)
    c.simulate(stimulus=simulate_what)
    c.save_skCSD_python()
    c.save_memb_curr()
    c.save_seg_length()
    return c.simulation_parameters['electrode'].LFP, c.return_paths_skCSD_python()


def L1_error(csd,est_csd):
    return (abs(csd-est_csd)).sum()/abs(csd).sum()


def make_output(what,tstart,tstop,merge):
    plotage = what[:,tstart:tstop]
    out = np.zeros((what.shape[0],(tstop-tstart)//merge))
    
    for k in range((tstop-tstart)//merge):
        out[:,k] = plotage[:,k*merge:(k+1)*merge].sum(axis=1)/merge
    return out

def merge_maps(maps,tstart,tstop,merge):
    single_width = (tstop-tstart)//merge
    outs = np.zeros((maps[0].shape[0],single_width*len(maps)))
    for i,mappe in enumerate(maps):
        outs[:,i*single_width:(i+1)*single_width] = make_output(mappe,tstart=tstart,tstop=tstop,merge=merge)

    return outs

def get_min_max(csd):
    vmin,vmax = csd.min(),csd.max()
    
    if vmin*vmax < 0:
        if abs(vmax) > abs(vmin):
            vmin = -vmax
        else:
            vmax = -vmin
            
    return vmax,vmin

def calculate_ticks(ticklabels,length):
    n = len(ticklabels)
    step = length//n
    if not step:
        step = 1
    
    return [i for i in range(0, length,step)]

def plot(ax_i,what,xticklabels=None,yticklabels=None,fig=None,title=None,vmin=None,vmax=None,sinksource=True):
    if not vmin or not vmax:
        xmax, xmin = get_min_max(what)
    else:
        xmax = vmax
        xmin = vmin
    cax = ax_i.imshow(what,origin='lower',aspect='auto',cmap='seismic_r',interpolation='none',vmin=xmin,vmax=xmax)

    if xticklabels:
        xticks = calculate_ticks(xticklabels,what.shape[1])
        ax_i.set_xticks(xticks)
        ax_i.set_xticklabels(xticklabels)
    else:
        ax_i.set_xticks([])
        
    if yticklabels:

        yticks = calculate_ticks(yticklabels,what.shape[0])
        
        ax_i.set_yticks(yticks)
        ax_i.set_yticklabels(yticklabels)
    else:
        ax_i.set_yticks([])
    
    if fig:
        cbar = fig.colorbar(cax, ticks=[xmin, 0, xmax])
        if sinksource:
            cbar.ax.set_yticklabels(['source','sink'])
    if title:
        ax_i.set_title(title)
    return cax
