
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib import gridspec
import matplotlib.cm as cm
import config


def show_csd(csd_at, csd, show_ele=None, show_kcsd=False):
    if config.dim == 1:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        if show_kcsd is False:
            ax.plot(csd_at, csd, 'g', label='CSD', linestyle='-', linewidth=3)
        else:
            ax.plot(csd_at, csd, 'g', label='kCSD', linestyle='--', linewidth=3)
        if show_ele is not None:
            ax.plot(show_ele, np.zeros_like(show_ele), 'ko', label='Electrodes', markersize=2.)
        max_csd = max(np.abs(csd))
        max_csd += max_csd*0.2
        ax.set_ylim([-max_csd, max_csd])
        ax.set_xlabel('Position mm')
        ax.set_ylabel('CSD mA/mm')
        ax.set_xlim([0., 1.])
        #ax.set_ylim([0., 1.])
        plt.legend()
    elif config.dim == 2:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, aspect='equal')
        t_max = np.max(np.abs(csd))
        levels = np.linspace(-1*t_max, t_max, 12)
        im = ax.contourf(csd_at[0], csd_at[1], csd, levels=levels, cmap=cm.bwr_r)
        if show_kcsd is False:
            ax.set_title('TrueCSD')
        else:
            ax.set_title('kCSD')
        ax.set_xlabel('Position mm')
        ax.set_ylabel('Position mm')
        cbar = plt.colorbar(im, orientation='vertical')
        cbar.set_ticks(levels[::2])
        cbar.set_ticklabels(np.around(levels[::2], decimals=2))
        if show_ele is not None:
            plt.scatter(show_ele[:, 0], show_ele[:, 1], 5, 'k')
        ax.set_xlim([0., 1.])
        ax.set_ylim([0., 1.])
    else:
        fig = plt.figure(figsize=(15, 5))
        z_steps = 5
        height_ratios = [1 for i in range(z_steps)]
        # height_ratios.append(0.1)
        width_ratios = [1, 0.05]
        gs = gridspec.GridSpec(z_steps, 2, height_ratios=height_ratios, width_ratios=width_ratios)
        t_max = np.max(np.abs(csd))
        levels = np.linspace(-1*t_max, t_max, 12)
        ind_interest = np.mgrid[0:csd_at[2].shape[2]:np.complex(0, z_steps+2)]
        ind_interest = np.array(ind_interest, dtype=np.int)[1:-1]
        for ii, idx in enumerate(ind_interest):
            ax = plt.subplot(gs[ii, 0])
            im = plt.contourf(csd_at[0][:, :, idx], csd_at[1][:, :, idx],
                              csd[:, :, idx], levels=levels, cmap=cm.bwr_r)
            if show_ele is not None:
                plt.scatter(show_ele[:, 0], show_ele[:, 1], 5, 'k')  # needs fix
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            title = str(csd_at[2][:, :, idx][0][0])[:4]
            ax.set_title(label=title, fontdict={'x': 0.8, 'y': 0.8})
            ax.set_aspect('equal')
            ax.set_xlim([0., 1.])
            ax.set_ylim([0., 1.])
        cax = plt.subplot(gs[:, -1])
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_ticks(levels[::2])
        cbar.set_ticklabels(np.around(levels[::2], decimals=2))
    return


def show_pot(ele_pos, pot, no_ele=False):
    if config.dim == 1:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, aspect='equal')
        ax.plot(ele_pos, pot, 'orange', label='Potential', linestyle='-', linewidth=3)
        if not no_ele:
            ax.plot(ele_pos, np.zeros_like(ele_pos), 'ko', label='Electrodes', markersize=2.)
        max_pot = max(np.abs(pot))
        max_pot += max_pot*0.2
        ax.set_ylim([-max_pot, max_pot])
        ax.set_xlabel('Position mm')
        ax.set_ylabel('Potential mV')
        plt.legend()
    elif config.dim == 2:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, aspect='equal')
        ele_x = ele_pos[:, 0]
        scale_x = max(ele_x) - min(ele_x)
        ele_y = ele_pos[:, 1]
        scale_y = max(ele_y) - min(ele_y)
        v_max = np.max(np.abs(pot))
        levels_pot = np.linspace(-1*v_max, v_max, 12)
        X, Y, Z = grid(ele_x, ele_y, pot)
        im = plt.contourf(X, Y, Z, levels=levels_pot, cmap=cm.PRGn)
        ax.set_xlim([min(ele_x) - (0.1*scale_x),
                     max(ele_x) + (0.1*scale_x)])
        ax.set_ylim([min(ele_y) - (0.1*scale_y),
                     max(ele_y) + (0.1*scale_y)])
        ax.set_title('Potentials')
        cbar2 = plt.colorbar(im, orientation='vertical')
        if not no_ele:
            im2 = plt.scatter(ele_x, ele_y, 5, c='k')
    else:
        fig = plt.figure(figsize=(15, 5))
        z_steps = 5
        height_ratios = [1 for i in range(z_steps)]
        # height_ratios.append(0.1)
        width_ratios = [1, 0.05]
        gs = gridspec.GridSpec(z_steps, 2, height_ratios=height_ratios, width_ratios=width_ratios)
        v_max = np.max(np.abs(pot))
        levels_pot = np.linspace(-1*v_max, v_max, 12)
        ele_res = int(np.ceil(len(pot)**(3**-1)))
        ele_x = ele_pos[:, 0]
        ele_y = ele_pos[:, 1]
        ele_z = ele_pos[:, 2]
        ele_x = ele_x.reshape(ele_res, ele_res, ele_res)
        ele_y = ele_y.reshape(ele_res, ele_res, ele_res)
        ele_z = ele_z.reshape(ele_res, ele_res, ele_res)
        pot = pot.reshape(ele_res, ele_res, ele_res)
        for idx in range(min(5, ele_res)):
            X, Y, Z = grid(ele_x[:, :, idx].flatten(),
                           ele_y[:, :, idx].flatten(),
                           pot[:, :, idx])
            ax = plt.subplot(gs[idx, 0])
            im = plt.contourf(X, Y, Z, levels=levels_pot, cmap=cm.PRGn)
            plt.scatter(ele_x[:, :, idx], ele_y[:, :, idx], 5)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            title = str(ele_z[:, :, idx][0][0])[:4]
            ax.set_title(label=title, fontdict={'x': 0.8, 'y': 0.8})
            ax.set_aspect('equal')
            ax.set_xlim([0., 1.])
            ax.set_ylim([0., 1.])
        cax = plt.subplot(gs[:, -1])
        cbar2 = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar2.set_ticks(levels_pot[::2])
        cbar2.set_ticklabels(np.around(levels_pot[::2], decimals=2))
    return


def grid(x, y, z, resX=100, resY=100):
    """
    Convert 3 column data to matplotlib grid
    """
    z = z.flatten()
    xx = np.linspace(min(x), max(x), resX)
    yy = np.linspace(min(y), max(y), resY)
    zz = griddata(x, y, z, xx, yy, interp='linear')
    return xx, yy, zz
