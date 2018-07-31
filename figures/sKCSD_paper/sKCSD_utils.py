from __future__ import division
import numpy as np
import os
import sys
if sys.version_info < (3, 0):
    import run_LFP


def make_fig_names(fname_base):
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    return os.path.join('Figures', fname_base)


def simulate(fname, **kwargs):
    morphology = kwargs.pop("morphology", 1)
    simulate_what = kwargs.pop("simulate_what", 1)
    electrode_orientation = kwargs.pop("electrode_orientation", 2)
    electrode_distribution = kwargs.pop("electrode_distribution", 1)
    colnb = kwargs.pop("colnb", 4)
    rownb = kwargs.pop("rownb", 4)
    xmin = kwargs.pop("xmin", -200)
    xmax = kwargs.pop("xmax", 200)
    ymin = kwargs.pop("ymin", -200)
    ymax = kwargs.pop("ymax", 200)
    tstop = kwargs.pop("tstop", 100)
    seed = kwargs.pop("seed", 1988)
    weight = kwargs.pop("weight", .01)
    n_syn = kwargs.pop("n_syn", 1000)
    triside = kwargs.pop("triside", 60)
    dt = kwargs.pop("dt", 0.5)
    if kwargs:
            raise TypeError('Invalid keyword arguments:', kwargs.keys())
    c = run_LFP.CellModel(morphology=morphology,
                          cell_name=fname,
                          colnb=colnb,
                          rownb=rownb,
                          xmin=xmin,
                          xmax=xmax,
                          ymin=ymin,
                          ymax=ymax,
                          tstop=tstop,
                          seed=seed,
                          weight=weight,
                          n_syn=n_syn,
                          electrode_distribution=electrode_distribution,
                          electrode_orientation=electrode_orientation,
                          triside=triside,
                          dt=dt)
    c.simulate(stimulus=simulate_what)
    c.save_skCSD_python()
    c.save_memb_curr()
    c.save_seg_length()
    return c


def L1_error(csd, est_csd):
    return (abs(csd-est_csd)).sum()/abs(csd).sum()


def make_output(what, tstart, tstop, merge):
    plotage = what[:, tstart:tstop]
    out = np.zeros((what.shape[0], (tstop-tstart)//merge))
    for k in range((tstop-tstart)//merge):
        out[:, k] = plotage[:, k*merge:(k+1)*merge].sum(axis=1)/merge
    return out


def merge_maps(maps, tstart, tstop, merge=1):
    single_width = (tstop-tstart)//merge
    print(single_width, tstop-tstart)
    outs = np.zeros((maps[0].shape[0], single_width*len(maps)))
    for i, mappe in enumerate(maps):
        outs[:, i*single_width:(i+1)*single_width] = make_output(mappe,
                                                                 tstart=tstart,
                                                                 tstop=tstop,
                                                                 merge=merge)
    return outs
