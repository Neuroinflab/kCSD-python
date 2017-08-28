# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:11:07 2017

@author: Jan Maka
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
import os

def skCSD_reconstruction_plot(pots,est_csd,est_pot,image,t_min=0):
    """Displays interactive skCSD reconstruction plot
            Parameters
            ----------
            pots - potentials recorded with elecdrodes
            est_csd - csd estimated with sKCSD3D class
            est_pot - potentials estimated with sKCSD class
            image - image of morphology constructed with sKCSDcell class
            t_min - starting time of the simulation

            Returns
            -------
            None
    """
    n_x,n_y,n_z,n_t = est_pot.shape
    y_max = np.max(pots[10,t_min:t_min+n_t])
    y_min = np.min(pots[10,t_min:t_min+n_t])
    fig = plt.figure(figsize=(10, 8))
    #plt.subplots_adjust(left=0.25, bottom=0.25)
    gridspec.GridSpec(5, 4)
    plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=1)
    t_plot, = plt.plot(pots[10,t_min:t_min+n_t], color='black')
    l, v = plt.plot(t_min, y_min, t_min+n_t, y_min, linewidth=2, color='red')
    est_csd_sub = plt.subplot2grid((5, 4), (1, 0), colspan=2, rowspan=2)
    est_csd_sub.set_title("skCSD")
    est_csd_plot = est_csd_sub.imshow(est_csd[:,:,n_z/2,0],cmap=plt.cm.bwr_r,vmin=np.min(est_csd),vmax=np.max(est_csd))
    est_csd_morph = est_csd_sub.imshow(image)
    est_pot_sub = plt.subplot2grid((5, 4), (1, 2), colspan=2, rowspan=2)
    est_pot_sub.set_title("Potential")
    est_pot_plot = est_pot_sub.imshow(est_pot[:,:,n_z/2,0], cmap=plt.cm.PRGn,vmin=np.min(est_pot),vmax=np.max(est_pot))
    est_pot_morph = est_pot_sub.imshow(image)

    axcolor = 'lightgoldenrodyellow'
    axt = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axslice = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

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

    morphax = plt.axes([0.01, 0.125, 0.15, 0.04])
    morphbutton = Button(morphax, 'Hide Morphology', color=axcolor, hovercolor='0.975')


    def switch(event):
        print morphbutton.label.get_text()
        if morphbutton.label.get_text()=='Hide Morphology':
            est_csd_morph.set_data(np.zeros(shape=image.shape))
            est_pot_morph.set_data(np.zeros(shape=image.shape))
            morphbutton.label.set_text('Show Morphology')
        else:
            print
            est_csd_morph.set_data(image)
            est_pot_morph.set_data(image)
            morphbutton.label.set_text('Hide Morphology')
        fig.canvas.draw_idle()
    morphbutton.on_clicked(switch)

    #rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    #radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    #radio.on_clicked(colorfunc)

    plt.show()

if __name__ == '__main__':
    data_dir = "examples"
    pots = np.loadtxt(os.path.join(data_dir, "raw_data\simData_skCSD\gang_7x7_200\myLFP"))
    est_csd = np.load(os.path.join(data_dir,"preprocessed_data/test_csd_100.npy"))
    image = np.load(os.path.join(data_dir, "preprocessed_data/image.npy"))
    print est_csd.shape
    est_pot = np.load(os.path.join(data_dir,"preprocessed_data/test_pot_100.npy"))
    skCSD_reconstruction_plot(pots,est_csd,est_pot,image)