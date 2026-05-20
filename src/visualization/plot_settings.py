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
        matplotlib session and returns two lists of four colours 
        sampled from the 'inferno' colormap at evenly spaced intervals.
        
        Returns
        -------
        colors : list
            Primary color palette [inferno(0.3), inferno(0.4), 
            inferno(0.6), inferno(0.8)]
        colors1 : list
            Alternative color palette [inferno(0.15), inferno(0.35), 
            inferno(0.55), inferno(0.75)]

Usage:
    from src.visualization.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    # colors[0] -> HNE / Near
    # colors[1] -> HNN / Mid
    # colors[2] -> HNZ / Far
    # colors[3] -> auxiliary / highlight
"""

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import List, Tuple

def set_plot_style() -> Tuple[List[Tuple[float, float, float, float]], 
                               List[Tuple[float, float, float, float]]]:
    """
    Apply project-wide matplotlib style and return color palettes.
    
    Configures matplotlib rcParams for consistent figure appearance across
    the project and generates two color palettes from the 'inferno' colormap.
    
    Returns
    -------
    colors : list of tuple
        Primary color palette with 4 colors sampled at [0.3, 0.4, 0.6, 0.8]
    colors1 : list of tuple
        Alternative color palette with 4 colors sampled at [0.15, 0.35, 0.55, 0.75]
    
    Examples
    --------
    >>> colors, colors1 = set_plot_style()
    >>> plt.plot(x, y, color=colors[0])  # Use first color from primary palette
    """
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