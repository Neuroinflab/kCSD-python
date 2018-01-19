#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:26:18 2017

@author: mkowalska
"""
from builtins import str
from datetime import date
import time

DAY = date.today()
TIMESTR = time.strftime("%Y%m%d-%H%M%S")

where_to_save_source_code = '/home/mkowalska/Marta/kCSD_results/' +\
    str(DAY) + '/' + TIMESTR + '/'

where_to_save_results = '/home/mkowalska/Marta/kCSD_results/' +\
    DAY.isoformat() + '/' + TIMESTR + '/'
