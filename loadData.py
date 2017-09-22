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
                   
        
        
    def __init__(self,path):

        self.path = path
        self.get_paths()


if __name__ == '__main__':
    obj = Data("Data/gang_7x7_200")
