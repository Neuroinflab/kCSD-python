#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division
import sys
import os
import utility_functions as utils



class Data(object):
    
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
            if drc.endswith("positions"):
                self.path_ele_pos = self.get_fname(drc,files)
            if drc.endswith("LFP"):
                self.path_LFP = self.get_fname(drc,files)
                   
        
    def load_morpho(self, morpho_path=None):
        
        if not morpho_path:
            morpho_path = self.path_morphology
            
        try:
            f = open(morpho_path)
        except IOError:
            print('Could not open morphology file',morpho_path)
            self.morphology = None
            return

        try:
            self.morphology = utils.load_swc(f)
        except ValueError:
            print('Could not load morphology file',morpho_path)
            self.morphology = None
            return
        print('Load morphology file',morpho_path)
        
    def load_ele_pos(self, ele_path=None):
        
        if not ele_path:
            ele_path = self.path_ele_pos
            
        try:
            f = open(ele_path)
        except IOError:
            print('Could not open electrode positions file',ele_path)
            self.ele_pos = None
            return

        try:
            self.ele_pos = utils.load_elpos(f)
        except ValueError:
            print('Could not load electrode positions file',ele_path)
            self.ele_pos = None
            return
        print('Load electrode positions file', ele_path)
            
    def __init__(self,path):

        self.path = path
        self.get_paths()
        self.load_morpho()
        self.load_ele_pos()


if __name__ == '__main__':
    obj = Data("Data/gang_7x7_200")
