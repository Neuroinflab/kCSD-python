#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division
import sys
import os
import utility_functions as utils



class Data(object):
    
    def sub_dir_path(self,d):
        return  filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)])

    def get_paths(self):
        dir_list = self.sub_dir_path(self.path)
    def __init__(self,path):

        self.path = path
        self.get_paths()
