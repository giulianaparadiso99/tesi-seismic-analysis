"""
plot_settings.py
----------------
Global matplotlib style configuration for the project. Defines a
consistent visual theme applied across all notebooks and figures,
ensuring uniformity in font sizes, colours, grid style, and spine
visibility throughout the analysis.

The module exposes a single public function:

    set_plot_style()
        Applies the project-wide rcParams settings to the current
        matplotlib session and returns a list of four colours sampled
        from the 'inferno' colormap at evenly spaced intervals.
        The returned colour list is intended to be used directly in
        plotting functions (e.g. for distinguishing components HNE,
        HNN, HNZ, or distance groups Near, Mid, Far).

Usage:
    from src.plot_settings import set_plot_style
    colors = set_plot_style()

    # colors[0] -> HNE / Near
    # colors[1] -> HNN / Mid
    # colors[2] -> HNZ / Far
    # colors[3] -> auxiliary / highlight
"""

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
    colors = [inferno(i) for i in [0.3, 0.4, 0.6, 0.8]]
    colors1 = [inferno(i) for i in [0.15, 0.35, 0.55, 0.75]]
    return colors, colors1