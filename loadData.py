#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division
import sys
import os
import utility_functions as utils
import numpy as np



class Data(object):
    
    
    def assign(self,what, value):
        if what == 'morphology':
            self.morphology = value
        elif what == 'electrode_positions':
            self.ele_pos = value
        elif what == 'LFP':
            self.LFP = value
            
    def sub_dir_path(self,d):
        return  filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)])

    def get_fname(self,d,fnames):

        if len(fnames) == 1:
            return os.path.join(d,fnames[0])
        else:
            paths = []
            for fname in fnames:
                paths.append(os.path.join(d,fname))
            return paths
        
    def get_paths(self):

        dir_list = self.sub_dir_path(self.path)
        for drc in dir_list:
            files = os.listdir(drc)
            if drc.endswith("morphology"):
                self.path_morphology = self.get_fname(drc,files)
                self.Path['morphology'] = self.path_morphology
                self.Func['morphology'] = utils.load_swc
            if drc.endswith("positions"):
                self.path_ele_pos = self.get_fname(drc,files)
                self.Path["electrode_positions"] = self.path_ele_pos
                self.Func["electrode_positions"] = utils.load_elpos
            if drc.endswith("LFP"):
                self.path_LFP = self.get_fname(drc,files)
                self.Path["LFP"] = self.path_LFP
                self.Func["LFP"] = np.loadtxt
                   
                
    def load(self,what, func=None, path=None,):
        
        if not func:
            func = self.Func[what]

        if not path:
            
            path = self.Path[what]
            #print(what,'unknown file type. Currently recognized file types are morphology, electrode_positions, LFP')
            #return
                
        try:
            f = open(path)
        except IOError:
            print('Could not open file',path)
            self.assign(what,None)
            return

        try:
            data = func(f)
            self.assign(what,data)
        except ValueError:
            print('Could not load file',path)
            self.assign(what,None)
            return
        print('Load',path)
    
            
    def __init__(self,path):
        self.Func = {}
        self.Path = {}
        self.path = path
        self.get_paths()
        self.load('morphology')
        self.load('electrode_positions')
        self.load('LFP')


if __name__ == '__main__':
    obj = Data("Data/gang_7x7_200")
