# -*- coding: utf-8 -*-
"""
These are some useful functions used in CSD methods,
They include CSD source profiles to be used as ground truths,
placement of electrodes in 1D, 2D and 3D., etc
These scripts are based on Grzegorz Parka's,
Google Summer of Code 2014, INFC/pykCSD
This was written by :
Michal Czerwinski, Chaitanya Chintaluri, Joanna Jędrzejewska-Szmek
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.


N-D Bresenham line algo Copyright 2012 Vikas Dhiman

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.spatial import distance
import os
import pickle
from scipy import interpolate
import json
import sys


def load_swc(path):
    """Load swc morphology from file

    Used for sKCSD

    Parameters
    ----------
    path : str

    Returns
    -------
    morphology : np.array
    """
    morphology = np.loadtxt(path)
    return morphology


def save_sim(path, k):
    """
    Save estimated CSD, potential and cell morphology to file.
    
    Used for saving sKCSD results
    
    Parameters:
    -----------
    path : str
    k : sKCSD object
      
    """
    est_csd = k.values('CSD', transformation=None)
    est_pot = k.values("POT", transformation=None)
    np.save(os.path.join(path, "csd.npy"), est_csd)
    print("Save csd, ", os.path.join(path, "csd.npy"))
    np.save(os.path.join(path, "pot.npy"), est_pot)
    print("Save pot, ", os.path.join(path, "pot.npy"))
    cell_data = {'morphology':k.cell.morphology.tolist(),
                 'ele_pos':k.cell.ele_pos.tolist(),
                 'n_src':k.cell.n_src}
    with open(os.path.join(path, "cell_data"), 'w') as handle:
        json.dump(cell_data, handle)


def load_sim(path):
    """
    Load sKCSD estimation results (CSD, potential and cell specifics). Return 
    CSD, potential, and three objects necessary for initialization 
    of a sKCSDcell object: morphology, positions of electrodes, 
    and number of sources
    
    Parameters
    ----------
    path: str

    Returns
    -------
    est_csd : np.array
    est_pot : np.array
    morphology : np.array
    ele_pos : np.array
    n_src : int
    """
    est_csd = np.load(os.path.join(path, "csd.npy"))
    est_pot = np.load(os.path.join(path, "pot.npy"))
    try:
        with open(os.path.join(path, "cell_data"), 'r') as handle:
            cell_data = json.load(handle)
    except Exception as error:
        print('Could not load', os.path.join(path, "cell_data"))
        return est_csd, est_pot, None
    morphology = np.array(cell_data['morphology'])
    ele_pos = np.array(cell_data['ele_pos'])
    n_src = cell_data['n_src']
    return est_csd, est_pot, cell_obj, morphology, ele_pos, n_src


def load_elpos(path):
    """Load electrode postions.

    File format: text file, one column, x of all the electrodes, y of
    all the electrodes, z of all the electrodes, or three columns 
    with cartesian coordinates of the electrodes


    Parameters
    ----------
    path : str
    Returns
    -------
    ele_pos : np.array
    """
    raw_ele_pos = np.loadtxt(path)
    if len(raw_ele_pos.shape) == 1:
        if raw_ele_pos.shape[0]%3:
            raise Exception('Unknown electrode position file format.')
        else:
            n_el = raw_ele_pos.shape[0]//3
            ele_pos = np.zeros(shape=(n_el, 3))
            ele_pos[:, 0] = raw_ele_pos[:n_el]
            ele_pos[:, 1] = raw_ele_pos[n_el:2 * n_el]
            ele_pos[:, 2] = raw_ele_pos[2 * n_el:]
    elif len(raw_ele_pos.shape) == 2:
        if raw_ele_pos.shape[1] == 1:
            if raw_ele_pos.shape[0]%3:
                raise Exception('Unknown electrode position file format.')
            else:
                n_el = raw_ele_pos.shape[0]/3
                ele_pos = np.zeros(shape=(n_el, 3))
                ele_pos[:, 0] = raw_ele_pos[:n_el]
                ele_pos[:, 1] = raw_ele_pos[n_el:2 * n_el]
                ele_pos[:, 2] = raw_ele_pos[2 * n_el:]
        elif raw_ele_pos.shape[0] == 1:
            if raw_ele_pos.shape[1]%3:
                raise Exception('Unknown electrode position file format.')
            else:
                n_el = raw_ele_pos.shape[1]/3
                ele_pos = np.zeros(shape=(n_el,3))
                ele_pos[:, 0] = raw_ele_pos[:n_el]
                ele_pos[:, 1] = raw_ele_pos[n_el:2 * n_el]
                ele_pos[:, 2] = raw_ele_pos[2 * n_el:]
        elif raw_ele_pos.shape[1] == 3:
            ele_pos = raw_ele_pos
        else:
            raise Exception('Unknown electrode position file format.')
    else:
        raise Exception('Unknown electrode position file format.')
    return ele_pos

def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.
    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])
    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])
    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 
    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)

def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed
    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.
    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def calculate_distance(xp_coor, x_coor):
    """
    Calculate euclidean distance of two points. 
    If this distance is smaller that 9 nm set it to 9 nm.

    Parameters
    ----------
    xp_coor, x_coor : np.array like

    Returns
    -------
    float
    """
    dist = distance.euclidean(xp_coor, x_coor)
    if dist < 1e-9:
        return 1e-9
    return dist
    
class LoadData(object):
    """
    Class for loading data for sKCSD calculations.
    Data should be divided into three subdirectories: morphology,
    electrode_positions and LFP, each containing one file with morphology,
    electrode_positions and LFP. LoadData currently supports only swc morphology
    format. LoadData can read in electrode positions as a text file either with 
    1 column with x postions for each electrode followed by y postions 
    for each electrodes followed by z positions of each electrode; 
    or a textfile with 3 columns with x, y, z electrode postions. 
    LFPs should be a text file with appropriate numbers.

    LoadData allows for initialization of an empty object and reading 
    in arbitrary data files from specific location using specified function.
    """
    def __init__(self, path):
        """
        Initialize LoadData object.

        Parameteres
        -----------
        path : path to directory with 3 subdirectories containing morphology,
        electrode positions and LFP.
        """
        self.Func = {}
        self.Path = {}
        self.path = path
        self.get_paths()
        self.load('morphology')
        self.load('electrode_positions')
        self.load('LFP')

    def assign(self, what, value):
        """
        Assign values to specified fields, if the field value is either
        morphology, electrode_postions or LFP.
        
        Parameters
        ----------
        what : string
        value 
        
        Returns
        -------
        None
        """
        if what == 'morphology':
            self.morphology = value
        elif what == 'electrode_positions':
            self.ele_pos = value
        elif what == 'LFP':
            self.LFP = value
            
    def sub_dir_path(self, d):
        """Find all the directories inside d
        Parameters
        ----------
        d : string

        Returns
        -------
        list of strings
        """
        return filter(os.path.isdir,
                      [os.path.join(d, f) for f in os.listdir(d)])

    def get_fname(self, d, fnames):
        """
        Find all the files in directory d. 

        Parameters
        ----------
        d : string
        fnames : string or list of strings

        Returns
        -------
        string or list of strings
        """
        if len(fnames) == 1:
            return os.path.join(d, fnames[0])
        else:
            paths = []
            for fname in fnames:
                paths.append(os.path.join(d, fname))
            return paths
        
    def get_paths(self):
        """
        Look for morphology file, electrode positions and LFP in the 
        subdirectories of path. Assign correct functions for loadig 
        data.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        dir_list = self.sub_dir_path(self.path)
        for drc in dir_list:
            files = os.listdir(drc)
            if drc.endswith("morphology"):
                self.path_morphology = self.get_fname(drc, files)
                self.Path['morphology'] = self.path_morphology
                self.Func['morphology'] = load_swc
            if drc.endswith("positions"):
                self.path_ele_pos = self.get_fname(drc, files)
                self.Path["electrode_positions"] = self.path_ele_pos
                self.Func["electrode_positions"] = load_elpos
            if drc.endswith("LFP"):
                self.path_LFP = self.get_fname(drc, files)
                self.Path["LFP"] = self.path_LFP
                self.Func["LFP"] = np.loadtxt
                   
                
    def load(self, what, func=None, path=None):
        """
        Load file with morphology, electrode positions or LFP. 
        what is specifying what is going to be loaded. Both function
        for loading and path can be overwritten, which should make
        it possible to read in arbitrary file using arbitrary function.

        Parameters
        ----------
        what : string
        func : object for reading data
               Defaults to None
        path : string
               Alternative path to a data file
               Defaults to None
        """
        if not func:
            func = self.Func[what]

        if not path:
            
            path = self.Path[what]

        if isinstance(path,list):
            for p in path:
                if p.endswith('swc'):
                    path = p
                    break
        try:
            f = open(path)
        except IOError:
            print('Could not open file',path)
            self.assign(what,None)
            return

        try:
            data = func(f)
            self.assign(what, data)
        except ValueError:
            print('Could not load file',path)
            self.assign(what, None)
            f.close()
            return
        print('Load', path)
        f.close()
