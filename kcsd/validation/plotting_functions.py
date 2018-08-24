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
import kcsd.utility_functions as utils
import glob

def skCSD_reconstruction_plot_z(pots, est_csd, est_pot, cell_obj,
                                t_min=0, electrode=5):
    """Displays interactive skCSD reconstruction plot in z plane
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
    axis = 2

    image, extent = cell_obj.draw_cell2D(axis=axis)
    n_x, n_y, n_z, n_t = est_pot.shape

    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.25, bottom=0.25, hspace=1, wspace=1)
    gridspec.GridSpec(5, 4)
    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=1)

    if not isinstance(pots, type(None)):
        y_max = np.max(pots[electrode, t_min:t_min+n_t])
        y_min = np.min(pots[electrode, t_min:t_min+n_t])
        t_plot, = ax1.plot(pots[electrode, t_min:t_min+n_t], color='black')
        l, v = ax1.plot(t_min, y_min, t_min + n_t, y_min, linewidth=2,
                        color='red')
        ax1.set_yticks(ax1.get_yticks()[::2])
    else:
        y_max = 1
        y_min = 0
        l, v = ax1.plot(0, 0, n_t, 1, linewidth=2, color='red')
        ax1.set_yticks(ax1.get_yticks()[::2])

    est_csd_sub = plt.subplot2grid((5, 4), (1, 0), colspan=2, rowspan=2)
    est_csd_sub.set_title("skCSD")
    est_pot_sub = plt.subplot2grid((5, 4), (1, 2), colspan=2, rowspan=2)
    est_pot_sub.set_title("Potential")

    est_csd_plot = est_csd_sub.imshow(est_csd[:, :, n_z//2, 0],
                                      cmap=plt.cm.bwr_r,
                                      vmin=np.min(est_csd),
                                      vmax=np.max(est_csd),
                                      aspect="auto",
                                      extent=extent,
                                      origin='lower')
    est_csd_morph = est_csd_sub.imshow(image,
                                       aspect="auto",
                                       extent=extent,
                                       origin='lower')

    est_pot_plot = est_pot_sub.imshow(est_pot[:, :, n_z//2, 0].T,
                                      cmap=plt.cm.PRGn,
                                      vmin=np.min(est_pot),
                                      vmax=np.max(est_pot),
                                      aspect="auto",
                                      extent=extent,
                                      origin='lower')
    est_pot_morph = est_pot_sub.imshow(image,
                                       aspect="auto",
                                       extent=extent,
                                       origin='lower')
    axcolor = 'lightgoldenrodyellow'
    axt = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axslice = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    tcut = Slider(axt, 'Time', t_min, t_min+n_t-1, valinit=0, valfmt='%0.0f')
    slicecut = Slider(axslice, 'Z slice', 0, n_z-1,
                      valinit=int(n_z/2), valfmt='%0.0f')

    def update(val):
        t = int(tcut.val)
        l.set_data([t, t], [y_min, y_max])
        z = int(slicecut.val)
        est_csd_plot.set_data(est_csd[:, :, z, t])
        est_pot_plot.set_data(est_pot[:, :, z, t])
        fig.canvas.draw_idle()

    tcut.on_changed(update)
    slicecut.on_changed(update)

    morphax = fig.add_axes([0.01, 0.75-0.08, 0.15, 0.04])
    morphbutton = Button(morphax, 'Hide Morphology',
                         color=axcolor, hovercolor='0.975')

    def switch(event):
        if morphbutton.label.get_text() == 'Hide Morphology':
            est_csd_morph.set_data(np.zeros(shape=image.shape))
            est_pot_morph.set_data(np.zeros(shape=image.shape))
            morphbutton.label.set_text('Show Morphology')
        else:
            est_csd_morph.set_data(image)
            est_pot_morph.set_data(image)
            morphbutton.label.set_text('Hide Morphology')
        fig.canvas.draw_idle()

    morphbutton.on_clicked(switch)

    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()

    plt.show()


def load_data(data_dir):
    """
    Load sKCSD estimation data and measured LFPs

    pots are the measured LFPs

    Parameters
    ----------
    data_dir : str

    Returns
    -------
    pots : np.array
    est_csd : np.array
    est_pot : np.array
    cell_obj : sKCSDcell object
    """
    try:
        data = utils.LoadData(data_dir)
    except KeyError:
        print('Could not load %s LFP from' % data_dir)
        data = None
    if data:
        pots = data.LFP
    else:
        pots = None

    try:
        est_csd, est_pot, cell_obj = utils.load_sim(data_dir)
    except IOError:
        if sys.version_info < (3, 0):
            path = os.path.join(data_dir, "preprocessed_data/Python_2")
        else:
            path = os.path.join(data_dir, "preprocessed_data/Python_3")

        est_csd, est_pot, cell_obj = utils.load_sim(path)

    return (pots, est_csd, est_pot, cell_obj)


def make_transformation(est_csd, est_pot, cell_object, transformation):
    """
    Transform both estimated csd and potential from the loop space
    to 3D or cell segments. Possible transformations 
    (values of parameter transformation): 3D, segments, loops

    Parameters
    ----------
    est_csd : np.array
    est_pot : np.array
    cell_object : sKCSDcell object
    transformation : str
        3D, segments, loops
    Returns
    -------
    new_csd : np.array
    new_pot : np.array
    """
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
        raise Exception("Unknown transformation %s" % transformation)

    return new_csd, new_pot


def calculate_ticks(ticklabels, length):
    """
    Calculate ticklabel positions for make_map_plot
    make_map_plot uses imshow from matplotlib.pyplot
    Take list of labels and axis length

    Parameters
    ----------
    ticklabels : list
    length : int
    """
    n = len(ticklabels)
    step = length//n
    if not step:
        step = 1
    return [i for i in range(0, length, step)]


def get_min_max(csd):
    """
    Return minimum and maximum value of a np.array. 
    If min and max are of a different sign, make sure
    that min = -max

    Parameters
    ----------
    csd : np.array

    Returns
    -------
    vmax : csd dtype instance
    vmin : csd dtype instance
    """
    vmin, vmax = csd.min(), csd.max()
    if vmin*vmax <= 0:
        if abs(vmax) > abs(vmin):
            vmin = -vmax
        else:
            vmax = -vmin

    return vmax, vmin


def make_fig(est_csd, est_pot, transformation, tstop=None):
    """
    2D figure of estimated csd and potential
    evolution in time
    in a chosen space: loops, segements

    Parameters:
    est_csd :  np.array
    est_pot : np.array
    transformation : str
       loops or segments
    tstop : double
    
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
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
    make_map_plot(ax[0],
         est_pot,
         fig=fig,
         title='Potential',
         cmap=plt.cm.PRGn,
         xlabel=xlabel,
         ylabel=ylabel,
         extent=extent)
    make_map_plot(ax[1],
         est_csd,
         fig=fig,
         title='CSD',
         xlabel=xlabel,
         extent=extent)


def make_map_plot(ax_i, what, **kwargs):
    """
    Make a figure of a map using imshow

    Parameters:
    ax_i : matplotlib.axes
    what : np.array
    xticklabels : list_like, optional
    yticklabels : list_like, optional
    fig : matplotlib.figure, optional
         fig allows to add a colorbar to axes
    title : str, optional
    vmax : double, optional
    vmin : double, optional
    sinksource: list_like, optional
         Labels of the colorbar
    extent: list_like, optional
         Adds extent (xlim and ylim) to figure
    cmap: plt.cm, optional
         default set to plt.cm.bwr_r
    xlabel: str, optional
    ylabel: str, optional
    """
    xticklabels = kwargs.pop('xticklabels', None)
    yticklabels = kwargs.pop('yticklabels', None)
    fig = kwargs.pop('fig', None)
    title = kwargs.pop('title', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    sinksource = kwargs.pop('sinksource', None)
    extent = kwargs.pop('extent', None)
    cmap = kwargs.pop('cmap', plt.cm.bwr_r)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    alpha = kwargs.pop('alpha', 0.5)  # transparency
    if kwargs:
        raise TypeError('Invalid keyword arguments:', kwargs.keys())

    if not vmin or not vmax:
        xmax, xmin = get_min_max(what)
    else:
        xmax = vmax
        xmin = vmin
    if extent:
        cax = ax_i.imshow(what,
                          origin='lower',
                          aspect='auto',
                          interpolation='none',
                          vmin=xmin,
                          vmax=xmax,
                          extent=extent,
                          cmap=cmap,
                          alpha=alpha)
        for tick in ax_i.get_xticklabels():
            tick.set_rotation(90)
    else:
        cax = ax_i.imshow(what,
                          origin='lower',
                          aspect='auto',
                          interpolation='none',
                          vmin=xmin,
                          vmax=xmax,
                          cmap=cmap,
                          alpha=alpha)
    if xticklabels:
        xticks = calculate_ticks(xticklabels, what.shape[1])
        ax_i.set_xticks(xticks)
        ax_i.set_xticklabels(xticklabels)
    else:
        ax_i.set_xticks([])

    if yticklabels:
        yticks = calculate_ticks(yticklabels, what.shape[0])
        ax_i.set_yticks(yticks)
        ax_i.set_yticklabels(yticklabels)
    else:
        ax_i.set_yticks([])

    if fig:
        cbar = fig.colorbar(cax, ax=ax_i, ticks=[xmin, 0, xmax])
        if sinksource:
            cbar.ax.set_yticklabels(['source', '0', 'sink'])
    if title:
        ax_i.set_title(title)
    if xlabel:
        ax_i.set_xlabel(xlabel)
    if ylabel:
        ax_i.set_ylabel(ylabel)
    return cax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot sKCSD current/potential estimation')
    parser.add_argument('path',
                        metavar='path',
                        type=str,
                        help='Path to sKCSD results')
    parser.add_argument('--transformation',
                        choices=set(('3D', 'segments', 'loops')),
                        default='3D',
                        help='''Space of CSD/current visualization:
                        3D, segments, loops''')
    parser.add_argument('--tstop',
                        type=float,
                        default=None,
                        help='Length of the measurement/simulation')
    args = parser.parse_args()
    data_dir = args.path
    pots, est_csd, est_pot, cell_obj = load_data(data_dir)
    est_csd, est_pot = make_transformation(est_csd,
                                           est_pot,
                                           cell_obj,
                                           args.transformation)
    if args.transformation == '3D':
        skCSD_reconstruction_plot_z(pots,
                                    est_csd,
                                    est_pot,
                                    cell_obj)
    elif args.transformation == 'loops' or args.transformation == 'segments':
        make_fig(est_csd,
                 est_pot,
                 args.transformation,
                 tstop=args.tstop)
    plt.show()
