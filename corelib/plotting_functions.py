 # -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:11:07 2017

@author: Joanna Jędrzejewska-Szmek, Jan Maka
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import corelib.utility_functions as utils
import glob

import corelib.loadData as ld
sliders = []

def skCSD_reconstruction_plot(pots,est_csd,est_pot,cell_obj,t_min=0,electrode=5):
    """Displays interactive skCSD reconstruction plot
            Parameters
            ----------
            pots - potentials recorded with electrodes
            est_csd - csd estimated with sKCSD class
            est_pot - potentials estimated with sKCSD class
            image - image of morphology constructed with sKCSDcell class
            t_min - starting time of the simulation

            Returns
            -------
            None
    """
    cell_obj.distribute_srcs_3D_morph()
    image, extent = cell_obj.draw_cell2D(axis=2)
    n_x,n_y,n_z,n_t = est_pot.shape
    y_max = np.max(pots[electrode,t_min:t_min+n_t])
    y_min = np.min(pots[electrode,t_min:t_min+n_t])
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.25, bottom=0.25, hspace=1,wspace=1)
    gridspec.GridSpec(5, 4)
    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=1)
    t_plot, = ax1.plot(pots[electrode,t_min:t_min+n_t], color='black')
    l, v = ax1.plot(t_min, y_min, t_min+n_t, y_min, linewidth=2, color='red')
    ax1.set_yticks(ax1.get_yticks()[::2])
    est_csd_sub = plt.subplot2grid((5, 4), (1, 0), colspan=2, rowspan=2)
    est_csd_sub.set_title("skCSD")
    est_csd_plot = est_csd_sub.imshow(est_csd[:,:,n_z//2,0],cmap=plt.cm.bwr_r,vmin=np.min(est_csd),vmax=np.max(est_csd), aspect="auto",extent=extent,origin='lower')
    est_csd_morph = est_csd_sub.imshow(image, aspect="auto",extent=extent,origin='lower')
    
    est_pot_sub = plt.subplot2grid((5, 4), (1, 2), colspan=2, rowspan=2)
    est_pot_sub.set_title("Potential")
    est_pot_plot = est_pot_sub.imshow(est_pot[:,:,n_z//2,0].T, cmap=plt.cm.PRGn,vmin=np.min(est_pot),vmax=np.max(est_pot),aspect="auto",extent=extent,origin='lower')
    est_pot_morph = est_pot_sub.imshow(image, aspect="auto",extent=extent,origin='lower')

    axcolor = 'lightgoldenrodyellow'
    axt = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axslice = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    tcut = Slider(axt, 'Time', t_min, t_min+n_t-1, valinit=0,valfmt='%0.0f')
    
    slicecut = Slider(axslice, 'Z slice', 0, n_z-1, valinit=int(n_z/2),valfmt='%0.0f')
 

    def update(val):
        z = int(slicecut.val)
        t = int(tcut.val)
        l.set_data([t, t], [y_min,y_max])
        est_csd_plot.set_data(est_csd[:,:,z,t])
        est_pot_plot.set_data(est_pot[:,:,z,t])
        
        fig.canvas.draw_idle()
        
    tcut.on_changed(update)
    slicecut.on_changed(update)

    morphax = fig.add_axes([0.01, 0.75-0.08, 0.15, 0.04]) 
    projectionaxxy = fig.add_axes([0.01, 0.25-0.00, 0.15, 0.02])
    projectionaxxz = fig.add_axes([0.01, 0.25-0.03, 0.15, 0.02])
    projectionaxyz = fig.add_axes([0.01, 0.25-0.06, 0.15, 0.02])
   
    morphbutton = Button(morphax, 'Hide Morphology', color=axcolor, hovercolor='0.975')
    projectionbuttonxy = Button(projectionaxxy, 'XY', color=axcolor, hovercolor='0.975')
    projectionbuttonxz = Button(projectionaxxz, 'switch to XZ', color=axcolor, hovercolor='0.975')
    projectionbuttonyz = Button(projectionaxyz, 'switch to YZ', color=axcolor, hovercolor='0.975')


    def switch(event):
        #print(morphbutton.label.get_text())
        if morphbutton.label.get_text()=='Hide Morphology':
            est_csd_morph.set_data(np.zeros(shape=image.shape))
            est_pot_morph.set_data(np.zeros(shape=image.shape))
            morphbutton.label.set_text('Show Morphology')
        else:
            #print()
            est_csd_morph.set_data(image)
            est_pot_morph.set_data(image)
            morphbutton.label.set_text('Hide Morphology')
        fig.canvas.draw_idle()

    def switchyz(event):
        #print(morphbutton.label.get_text())
        if morphbutton.label.get_text()=='Hide Morphology':
            est_csd_morph.set_data(np.zeros(shape=image.shape))
            est_pot_morph.set_data(np.zeros(shape=image.shape))
            morphbutton.label.set_text('Show Morphology')
        else:
            #print()
            est_csd_morph.set_data(image)
            est_pot_morph.set_data(image)
            morphbutton.label.set_text('Hide Morphology')
        fig.canvas.draw_idle()

    morphbutton.on_clicked(switch)


    #rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
    #radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    #radio.on_clicked(colorfunc)

    plt.show()

def load_data(data_dir):

    data = ld.Data(data_dir)
    pots = data.LFP
    if sys.version_info < (3,0):
        path = os.path.join(data_dir, "preprocessed_data/Python_2")
    else:
        path = os.path.join(data_dir, "preprocessed_data/Python_3")

    est_csd, est_pot, cell_obj = utils.load_sim(path)
    return (pots, est_csd, est_pot, cell_obj)

def make_transformation(est_csd,est_pot,cell_object,transformation):
    if transformation == '3D':
        new_csd = cell_object.transform_to_3D(est_csd)
        new_pot = cell_object.transform_to_3D(est_pot)
    elif transformation == 'segments':
        new_csd = cell_object.transform_to_segments(est_csd)
        new_pot = cell_object.transform_to_segments(est_pot)
    elif transformation == 'loops':
        new_csd = est_csd
        new_pot = est_pot
    else:
        raise Exception("Unknown transformation %s"%transformation)

    return new_csd, new_pot

def calculate_ticks(ticklabels,length):
    n = len(ticklabels)
    step = length//n
    if not step:
        step = 1
    
    return [i for i in range(0, length,step)]

def get_min_max(csd):
    vmin,vmax = csd.min(),csd.max()
    
    if vmin*vmax < 0:
        if abs(vmax) > abs(vmin):
            vmin = -vmax
        else:
            vmax = -vmin
            
    return vmax,vmin

def make_fig(est_csd,est_pot,transformation,tstop=None):
    fig, ax = plt.subplots(1,2,figsize=(10, 8))
    if tstop:
        xlabel = 'Time [ms]'
        extent = [0, tstop, 0, est_csd.shape[0]]
    else:
        xlabel = None
        extent = None
    if transformation == 'loops':
        ylabel = '#Loop'
    elif transformation == 'segments':
        ylabel = '#segment'
    plot(ax[0],est_pot,fig=fig,title='Potential',cmap=plt.cm.PRGn,xlabel=xlabel,ylabel=ylabel,extent=extent)
    plot(ax[1],est_csd,fig=fig,title='CSD',xlabel=xlabel,extent=extent)
    
    
    

def plot(ax_i,what,xticklabels=None,yticklabels=None,fig=None,title=None,vmin=None,vmax=None,sinksource=True,extent=None,cmap=plt.cm.bwr_r,xlabel=None,ylabel=None):
    if not vmin or not vmax:
        xmax, xmin = get_min_max(what)
    else:
        xmax = vmax
        xmin = vmin
    if extent:
         cax = ax_i.imshow(what,origin='lower',aspect='auto',interpolation='none',vmin=xmin,vmax=xmax,extent=extent,cmap=cmap)
         for tick in ax_i.get_xticklabels():
             tick.set_rotation(90)
         
    else:
        cax = ax_i.imshow(what,origin='lower',aspect='auto',interpolation='none',vmin=xmin,vmax=xmax,cmap=cmap)

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
        cbar = fig.colorbar(cax, ax=ax_i, ticks=[xmin, 0, xmax])
        if sinksource:
            cbar.ax.set_yticklabels(['source','0','sink'])
    if title:
        ax_i.set_title(title)
    if xlabel:
        ax_i.set_xlabel(xlabel)
    if ylabel:
        ax_i.set_ylabel(ylabel)
    return cax


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Plot sKCSD current/potential estimation')
    parser.add_argument('path', metavar='path', type=str, 
                    help='Path to sKCSD results')
    parser.add_argument('--transformation',choices=set(('3D','segments','loops')), default='3D',
                    help='Space of CSD/current visualization: 3D, segments, loops')
    parser.add_argument('--tstop',type=float, default=None,
                    help='Length of the measurement/simulation')
    args = parser.parse_args()
    
    data_dir = args.path
    
    pots, est_csd, est_pot, cell_obj = load_data(data_dir)
    est_csd, est_pot = make_transformation(est_csd,est_pot,cell_obj,args.transformation)
    
    if args.transformation == '3D':
        skCSD_reconstruction_plot(pots,est_csd,est_pot,cell_obj)
    elif args.transformation == 'loops' or args.transformation == 'segments':
        make_fig(est_csd,est_pot,args.transformation,tstop=args.tstop)
        
    plt.show()
