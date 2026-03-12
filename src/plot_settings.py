import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def set_plot_style():
    mpl.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titlepad': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.frameon': False,
        'legend.fontsize': 10,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0,
    })
    plt.rcParams['image.cmap'] = 'inferno'
    inferno = cm.get_cmap('inferno')
    colors = [inferno(i) for i in [0.2, 0.4, 0.6, 0.8]]
    return colors