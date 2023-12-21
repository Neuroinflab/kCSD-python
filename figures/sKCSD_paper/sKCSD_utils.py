import numpy as np
import os
import sys
import run_LFP


def make_fig_names(fname_base):
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    return os.path.join('Figures', fname_base)


def simulate(fname_base, **kwargs):
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
    fname = fname_base+'_rows_%s_cols_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s_orientation_%s' % (rownb, colnb, xmin, xmax, ymin, ymax,  electrode_orientation)
    triside = kwargs.pop("triside", 60)
    electrode_distance = kwargs.pop("electrode_distance", 50)
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
                          electrode_distance=electrode_distance,
                          triside=triside,
                          dt=dt)
    c.simulate(stimulus=simulate_what)
    c.save_skCSD_python()
    c.save_memb_curr()
    c.save_seg_length()
    c.save_somav()
    c.save_tvec()
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
    outs = np.zeros((maps[0].shape[0], single_width*len(maps)))
    for i, mappe in enumerate(maps):
        outs[:, i*single_width:(i+1)*single_width] = make_output(mappe,
                                                                 tstart=tstart,
                                                                 tstop=tstop,
                                                                 merge=merge)
    return outs


if __name__ == '__main__':
    fname_base = "gang_7x7_200_distal_osc"
    args = {
        'morphology': 9,
        'xmin': -40,
        'xmax': 40,
        'ymin': -20,
        'ymax': 20,
        'electrode_distribution': 1,
        'electrode_orientation': 3,
        'colnb': 5,
        'rownb': 4,
        'tstop': 100,
        'simulate_what': 'distal_oscillatory',
    }
    simulate(fname_base, **args)
    fname_base = "gang_7x7_200"
    args = {
        'morphology': 9,
        'xmin': -200,
        'xmax': 200,
        'ymin': -200,
        'ymax': 200,
        'electrode_distribution': 1,
        'electrode_orientation': 3,
        'simulate_what':'oscillatory',
        'colnb': 10,
        'rownb': 10,
        'tstop': 100,
    }
    simulate(fname_base, **args)

