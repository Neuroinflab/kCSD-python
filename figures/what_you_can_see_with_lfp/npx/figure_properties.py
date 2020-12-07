import matplotlib.pyplot as plt

def set_axis(ax, x, y, letter=None):
    ax.text(
        x,
        y,
        letter,
        fontsize=25,
        weight='bold',
        transform=ax.transAxes)
    return ax

plt.rcParams.update({
    'xtick.labelsize': 15,
    'xtick.major.size': 10,
    'ytick.labelsize': 15,
    'ytick.major.size': 10,
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 20,
    'axes.titlepad' : 30,
    'legend.fontsize': 15,
    # 'figure.subplot.wspace': 0.4,
    # 'figure.subplot.hspace': 0.4,
    # 'figure.subplot.left': 0.1,
})


