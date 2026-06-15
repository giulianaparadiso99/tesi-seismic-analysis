"""
plot_moment_scaling.py
----------------------
Visualization functions for moment scaling analysis results.

This module provides functions to visualize the scaling behavior of
seismic signals through moment scaling analysis. It creates plots showing:
- Scaling curves: log(M_q) vs log(τ) for different moment orders q
- Scaling exponents: ζ(q) vs q, revealing multifractal properties

All functions operate on ensemble-averaged results from multiple signals,
producing multi-panel figures (2×2 subplots) comparing different seismic
phases: pre-event noise, P-wave, S-wave, and coda.

Functions
---------
plot_scaling_curves : Plot log(M_q) vs log(τ) with power-law fits
plot_scaling_exponents : Plot scaling exponents ζ(q) with error bars

Notes
-----
Moment scaling analysis computes:
    M_q(τ) = ⟨|Δx(τ)|^q⟩
    
where Δx(τ) are signal increments at timescale τ, q is the moment order,
and ⟨·⟩ denotes ensemble averaging.

The scaling exponent ζ(q) is extracted from:
    M_q(τ) ∝ τ^ζ(q)
    
For normal diffusion: ζ(q) = q/2 (linear)
For anomalous diffusion: ζ(q) deviates from linearity, indicating
multifractal behavior common in complex out-of-equilibrium systems.

References
----------
Vollmer et al. (2024), "Moment scaling functions of multifractal signals
    with high regularity"
Rondoni et al. (2024), "Detecting phase transitions through nonequilibrium
    work fluctuations"

Examples
--------
>>> from src.analysis.moment_scaling import analyze_all_windows
>>> from src.visualization.plot_moment_scaling import (
...     plot_scaling_curves,
...     plot_scaling_exponents
... )
>>>
>>> # Analyze ensemble
>>> results = analyze_all_windows(windowed_signals, sampling_rate=200)
>>>
>>> # Plot scaling curves
>>> fig1 = plot_scaling_curves(
...     results,
...     output_dir='../figures/moment_scaling/',
...     q_subset=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
... )
>>>
>>> # Plot scaling exponents
>>> fig2 = plot_scaling_exponents(
...     results,
...     output_dir='../figures/moment_scaling/'
... )
>>> plt.show()
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from src.visualization.plot_settings import set_plot_style
colors, colors1 = set_plot_style()

def plot_scaling_curves(
    results: Dict,
    output_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    q_subset: Optional[np.ndarray] = None,
    q_colors: Optional[List] = colors1 
) -> plt.Figure:
    """
    Plot log(M_q) vs log(tau) for all windows (2x2 subplots).
    
    Parameters
    ----------
    results : dict
        Output from analyze_all_windows()
    output_dir : str or Path, optional
        Directory to save figure. If None, figure is displayed but not saved.
    figsize : tuple of float, optional
        Figure size in inches (default: (16, 12))
    q_subset : np.ndarray, optional
        Subset of q values to plot. If None, uses default subset:
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    colors : list, optional
        List of colors for different q values. If None, uses 'inferno' colormap.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex='col')
    axes = axes.flatten()
    
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_titles = {
        'pre_event': 'Pre-event (noise)',
        'p_wave': 'P-wave',
        's_wave': 'S-wave',
        'coda': 'Coda'
    }
    
    if q_subset is None:
        q_subset = np.array([0.5, 1.0, 2.0, 3.0])
    else:
        q_subset = np.asarray(q_subset)
    n_q = len(q_subset) 
    
    for idx, window_name in enumerate(windows):
        ax = axes[idx]
        
        if window_name not in results or results[window_name] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(window_titles[window_name], fontsize=13, fontweight='bold')
            continue
        
        ensemble = results[window_name]['ensemble']
        scaling = results[window_name]['scaling']
        
        tau = ensemble['tau']
        q_values = ensemble['q']
        moments_mean = ensemble['moments_mean']
        zeta = scaling['zeta']
        intercepts = scaling['intercepts']
        
        q_mask = np.isin(q_values, q_subset)
        q_plot = q_values[q_mask]
        moments_plot = moments_mean[:, q_mask]
        zeta_plot = zeta[q_mask]
        intercepts_plot = intercepts[q_mask]
        
        if q_colors is None:
            q_colors = colors1

        plot_colors = q_colors * (n_q // len(q_colors) + 1)
        plot_colors = plot_colors[:n_q]
        
        for i, (q, color) in enumerate(zip(q_plot, plot_colors)):
            M_q = moments_plot[:, i]
            valid = (M_q > 0) & np.isfinite(M_q)
            
            if valid.sum() < 2:
                continue
            
            tau_valid = tau[valid]
            M_q_valid = M_q[valid]
            
            label = f'q={q:.1f}, ζ={zeta_plot[i]:.2f}'
            
            ax.loglog(tau_valid, M_q_valid, 'o', color=color, markersize=6,
                     alpha=0.7, label=label, markeredgewidth=0.5, 
                     markeredgecolor='white')
            
            if not np.isnan(zeta_plot[i]):
                tau_fit = tau_valid
                log_M_fit = zeta_plot[i] * np.log10(tau_fit) + intercepts_plot[i]
                M_fit = 10 ** log_M_fit
                ax.loglog(tau_fit, M_fit, '--', color=color, linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('τ (s)', fontsize=12)
        ax.set_ylabel('⟨|Δx(τ)|^q⟩', fontsize=12)
        ax.set_title(window_titles[window_name], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linewidth=0.5)
        
        ax.legend(fontsize=8, ncol=2, loc='lower right', framealpha=0.95,
                 edgecolor='gray', fancybox=False, columnspacing=1.0,
                 handletextpad=0.5, borderpad=0.4)
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        q_min = q_subset.min()
        q_max = q_subset.max()
        output_file = output_dir / f'ensemble_scaling_curves_q{q_min:.2f}-{q_max:.2f}_n{n_q}.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig

def plot_scaling_exponents(
    results: Dict,
    output_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    point_color: Optional[str] = None
) -> plt.Figure:
    """
    Plot scaling exponents ζ(q) vs q for all windows (2x2 subplots).
    
    Parameters
    ----------
    results : dict
        Output from analyze_all_windows()
    output_dir : str or Path, optional
        Directory to save figure. If None, figure is displayed but not saved.
    figsize : tuple of float, optional
        Figure size in inches (default: (16, 12))
    point_color : str or tuple, optional
        Color for data points. If None, uses 'navy' (default).
        Can be any matplotlib color (name, hex, RGB tuple).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_titles = {
        'pre_event': 'Pre-event (noise)',
        'p_wave': 'P-wave',
        's_wave': 'S-wave',
        'coda': 'Coda'
    }
    
    if point_color is None:
        point_color = 'black'
    
    for idx, window_name in enumerate(windows):
        ax = axes[idx]
        
        if window_name not in results or results[window_name] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(window_titles[window_name], fontsize=13, fontweight='bold')
            continue
        
        q_values = results[window_name]['ensemble']['q']
        zeta = results[window_name]['scaling']['zeta']
        zeta_err = results[window_name]['scaling']['zeta_err']
        r_squared = results[window_name]['scaling']['r_squared']
        
        valid = np.isfinite(zeta)
        
        ax.errorbar(q_values[valid], zeta[valid], yerr=zeta_err[valid],
                   fmt='o', markersize=7, capsize=4, capthick=1.5,
                   color=point_color, ecolor=point_color, alpha=0.8,
                   label='Measured ζ(q)', zorder=3)
        
        q_ref = np.linspace(q_values.min(), q_values.max(), 100)
        zeta_normal = q_ref / 2
        ax.plot(q_ref, zeta_normal, '--', color='red', linewidth=2.5,
               label='Normal diffusion (ζ=q/2)', alpha=0.7, zorder=2)
        
        ax.set_xlabel('q', fontsize=12)
        ax.set_ylabel('ζ(q)', fontsize=12)
        ax.set_title(window_titles[window_name], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=10, loc='upper left', framealpha=0.95,
                 edgecolor='gray', fancybox=False)
        
        
        ax.set_xlim(q_values.min() - 0.2, q_values.max() + 0.2)
        y_max_data = (zeta[valid] + zeta_err[valid]).max() if valid.any() else 1.0
        y_max_ref = (q_values.max() / 2) * 1.05
        ax.set_ylim(bottom=0, top=max(y_max_data * 1.1, y_max_ref))
            
    plt.tight_layout()

    row_labels = ['Acceleration', 'Velocity', 'Displacement']
    row_label_x = 0.01  # coordinata x in unità figura
    row_label_y_positions = []
    for row in range(3):
        # calcola il centro verticale della riga dall'oggetto axes
        bbox = axes[row][0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        row_label_y_positions.append(y_center)

    for row, (label, y_pos) in enumerate(zip(row_labels, row_label_y_positions)):
        fig.text(
            row_label_x, y_pos, label,
            fontsize=cfg['font_axis_label'],
            ha='left', va='center',
            rotation=90,
            fontweight='bold',
        )
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'ensemble_scaling_exponents.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig

def plot_scaling_curves_v2(
    results_by_signal: Dict[str, Dict],
    coda_method: str = 'rautian',
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'thesis',
) -> plt.Figure:
    """
    Plot log(M_q) vs log(tau) for P-wave, S-wave, and coda windows,
    across acceleration, velocity, and displacement signals (3x3 grid).

    Parameters
    ----------
    results_by_signal : dict
        Dictionary mapping signal type to analyze_all_windows() output:
        {'acceleration': results_acc, 'velocity': results_vel,
         'displacement': results_disp}.
    coda_method : str, optional
        Coda onset method label, used only for the output filename
        (default: 'rautian').
    output_path : str or Path, optional
        If provided, save the figure to this path. File extension is set
        by the mode (.pdf for thesis/interactive, .png for paper/poster).
    mode : str, optional
        Output mode controlling figure size and font sizes.
        One of 'thesis', 'paper', 'poster', 'interactive'
        (default: 'thesis').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _VALID_MODES = ('interactive', 'paper', 'poster', 'thesis')
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}"
        )

    _mode_settings = {
        'interactive': dict(
            figsize=(16, 12),
            dpi_save=150,
            font_title=13,
            font_axis_label=11,
            font_tick=10,
            font_legend=10,
            linewidth_fit=1.5,
            markersize=5,
            output_suffix='.pdf',
        ),
        'paper': dict(
            figsize=(6.89, 5.5),
            dpi_save=600,
            font_title=8,
            font_axis_label=7,
            font_tick=6,
            font_legend=6,
            linewidth_fit=1.0,
            markersize=3,
            output_suffix='.png',
        ),
        'poster': dict(
            figsize=(14.85, 11.0),
            dpi_save=300,
            font_title=18,
            font_axis_label=15,
            font_tick=13,
            font_legend=13,
            linewidth_fit=2.0,
            markersize=6,
            output_suffix='.png',
        ),
        'thesis': dict(
            figsize=(12, 9),
            dpi_save=300,
            font_title=10,
            font_axis_label=9,
            font_tick=8,
            font_legend=8,
            linewidth_fit=1.5,
            markersize=4,
            output_suffix='.pdf',
        ),
    }
    cfg = _mode_settings[mode]

    signal_types = ['acceleration', 'velocity', 'displacement']
    windows = ['p_wave', 's_wave', 'coda']
    window_titles = {'p_wave': 'P-wave', 's_wave': 'S-wave', 'coda': 'Coda'}

    q_subset = np.array([0.5, 1.0, 2.0, 3.0])
    q_colors = {
        0.5: '#00807F',
        1.0: '#C8861D',
        2.0: '#729EC1',
        3.0: '#8B6BAE',
    }

    fig, axes = plt.subplots(
        3, 3,
        figsize=cfg['figsize'],
        sharex='col',
    )
    fig.subplots_adjust(
        top=0.88, bottom=0.08, left=0.10, right=0.97,
        hspace=0.10, wspace=0.12,
    )

    legend_elements = []
    legend_built = False

    for row, signal_type in enumerate(signal_types):
        results = results_by_signal.get(signal_type)

        for col, window_name in enumerate(windows):
            ax = axes[row][col]

            if col == 0:
                ax.set_ylabel(
                    r'$\langle|\Delta x(\tau)|^q\rangle$',
                    fontsize=cfg['font_axis_label'],
                )
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

            if row == 0:
                ax.set_title(
                    window_titles[window_name],
                    fontsize=cfg['font_title'],
                    fontweight='bold',
                )

            if row < 2:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$\tau$ (s)', fontsize=cfg['font_axis_label'])

            ax.tick_params(labelsize=cfg['font_tick'])

            ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=10))
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=10))
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

            ax.grid(True, alpha=0.3, which='major', linewidth=0.5)

            if results is None or window_name not in results or results[window_name] is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=cfg['font_axis_label'], color='gray')
                continue

            ensemble = results[window_name]['ensemble']
            scaling = results[window_name]['scaling']
            tau = ensemble['tau']
            q_values = ensemble['q']
            moments_mean = ensemble['moments_mean']
            zeta = scaling['zeta']
            intercepts = scaling['intercepts']

            for q_val, color in q_colors.items():
                q_idx = np.where(np.isclose(q_values, q_val))[0]
                if len(q_idx) == 0:
                    continue
                q_idx = q_idx[0]

                M_q = moments_mean[:, q_idx]
                valid = (M_q > 0) & np.isfinite(M_q)
                if valid.sum() < 2:
                    continue

                tau_valid = tau[valid]
                M_q_valid = M_q[valid]
                zeta_val = zeta[q_idx]

                ax.loglog(
                    tau_valid, M_q_valid, 'o',
                    color=color, markersize=cfg['markersize'],
                    alpha=0.7, markeredgewidth=0.3,
                    markeredgecolor='white',
                )
                if not np.isnan(zeta_val):
                    log_M_fit = (
                        zeta_val * np.log10(tau_valid)
                        + intercepts[q_idx]
                    )
                    ax.loglog(
                        tau_valid, 10 ** log_M_fit, '--',
                        color=color, linewidth=cfg['linewidth_fit'],
                        alpha=0.6,
                    )

                if not legend_built:
                    legend_elements.append(plt.Line2D(
                        [0], [0], color=color, marker='o',
                        markersize=cfg['markersize'],
                        markeredgewidth=0.3, markeredgecolor='white',
                        linestyle='--', linewidth=cfg['linewidth_fit'],
                        alpha=0.8,
                        label=f'$q = {q_val:.1f}$',
                    ))

            if not legend_built:
                legend_built = True

    row_labels = ['Acceleration', 'Velocity', 'Displacement']
    for row, label in enumerate(row_labels):
        bbox = axes[row][0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.01, y_center, label,
            fontsize=cfg['font_axis_label'],
            ha='left', va='center',
            rotation=90,
            fontweight='bold',
        )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(legend_elements),
        fontsize=cfg['font_legend'],
        framealpha=0.9,
        handlelength=2.0,
        columnspacing=1.5,
    )

    if output_path is not None:
        suffix = cfg['output_suffix']
        output_path = Path(output_path).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig


def plot_scaling_exponents_v2(
    results_by_signal: Dict[str, Dict],
    coda_method: str = 'rautian',
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'thesis',
) -> plt.Figure:
    """
    Plot scaling exponents zeta(q) vs q for P-wave, S-wave, and coda
    windows, across acceleration, velocity, and displacement signals
    (3x3 grid).

    Parameters
    ----------
    results_by_signal : dict
        Dictionary mapping signal type to analyze_all_windows() output:
        {'acceleration': results_acc, 'velocity': results_vel,
         'displacement': results_disp}.
    coda_method : str, optional
        Coda onset method label, used only for the output filename
        (default: 'rautian').
    output_path : str or Path, optional
        If provided, save the figure to this path. File extension is set
        by the mode (.pdf for thesis/interactive, .png for paper/poster).
    mode : str, optional
        Output mode controlling figure size and font sizes.
        One of 'thesis', 'paper', 'poster', 'interactive'
        (default: 'thesis').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _VALID_MODES = ('interactive', 'paper', 'poster', 'thesis')
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}"
        )

    _mode_settings = {
        'interactive': dict(
            figsize=(16, 12),
            dpi_save=150,
            font_title=13,
            font_axis_label=11,
            font_tick=10,
            font_legend=10,
            linewidth_ref=2.0,
            markersize=6,
            capsize=4,
            output_suffix='.pdf',
        ),
        'paper': dict(
            figsize=(6.89, 5.5),
            dpi_save=600,
            font_title=8,
            font_axis_label=7,
            font_tick=6,
            font_legend=6,
            linewidth_ref=1.2,
            markersize=3,
            capsize=2,
            output_suffix='.png',
        ),
        'poster': dict(
            figsize=(14.85, 11.0),
            dpi_save=300,
            font_title=18,
            font_axis_label=15,
            font_tick=13,
            font_legend=13,
            linewidth_ref=2.5,
            markersize=6,
            capsize=4,
            output_suffix='.png',
        ),
        'thesis': dict(
            figsize=(12, 9),
            dpi_save=300,
            font_title=10,
            font_axis_label=9,
            font_tick=8,
            font_legend=10,
            linewidth_ref=1.8,
            markersize=5,
            capsize=3,
            output_suffix='.pdf',
        ),
    }
    cfg = _mode_settings[mode]

    point_color = "#010A0A"
    ref_color = '#C0392B'

    signal_types = ['acceleration', 'velocity', 'displacement']
    windows = ['p_wave', 's_wave', 'coda']
    window_titles = {'p_wave': 'P-wave', 's_wave': 'S-wave', 'coda': 'Coda'}

    fig, axes = plt.subplots(
        3, 3,
        figsize=cfg['figsize'],
    )
    fig.subplots_adjust(
        top=0.88, bottom=0.08, left=0.10, right=0.97,
        hspace=0.10, wspace=0.12,
    )

    legend_elements = [
        plt.Line2D(
            [0], [0], color=point_color, marker='o',
            markersize=cfg['markersize'], linestyle='none',
            label=r'Measured $\zeta(q)$',
        ),
        plt.Line2D(
            [0], [0], color=ref_color, linestyle='--',
            linewidth=cfg['linewidth_ref'], alpha=0.7,
            label=r'Normal diffusion ($\zeta = q/2$)',
        ),
    ]

    for row, signal_type in enumerate(signal_types):
        results = results_by_signal.get(signal_type)

        for col, window_name in enumerate(windows):
            ax = axes[row][col]

            if col == 0:
                ax.set_ylabel(
                    r'$\zeta(q)$',
                    fontsize=cfg['font_axis_label'],
                )
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

            if row == 0:
                ax.set_title(
                    window_titles[window_name],
                    fontsize=cfg['font_title'],
                    fontweight='bold',
                )

            if row < 2:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$q$', fontsize=cfg['font_axis_label'])

            ax.tick_params(labelsize=cfg['font_tick'])
            ax.grid(True, alpha=0.3, linewidth=0.5)

            if results is None or window_name not in results or results[window_name] is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=cfg['font_axis_label'], color='gray')
                continue

            q_values = results[window_name]['ensemble']['q']
            zeta = results[window_name]['scaling']['zeta']
            zeta_err = results[window_name]['scaling']['zeta_err']
            r_squared = results[window_name]['scaling']['r_squared']

            valid = np.isfinite(zeta)

            ax.errorbar(
                q_values[valid], zeta[valid], yerr=zeta_err[valid],
                fmt='o', markersize=cfg['markersize'],
                capsize=cfg['capsize'], capthick=1.2,
                color=point_color, ecolor=point_color,
                alpha=0.85, zorder=3,
            )

            q_ref = np.linspace(q_values.min(), q_values.max(), 100)
            ax.plot(
                q_ref, q_ref / 2, '--',
                color=ref_color, linewidth=cfg['linewidth_ref'],
                alpha=0.7, zorder=2,
            )

            mean_r2 = np.nanmean(r_squared[valid])
            ax.text(
                0.97, 0.05,
                f'$\\bar{{R}}^2 = {mean_r2:.2f}$',
                transform=ax.transAxes,
                fontsize=cfg['font_tick'],
                ha='right', va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='lightgray',
                ),
            )

            ax.set_xlim(q_values.min() - 0.2, q_values.max() + 0.2)
            if valid.any():
                y_min_data = (zeta[valid] - zeta_err[valid]).min()
                y_max_data = (zeta[valid] + zeta_err[valid]).max()
                y_bottom = min(0.0, y_min_data * 1.1)
                y_top = max(y_max_data * 1.1, (q_values.max() / 2) * 1.05)
            else:
                y_bottom = 0.0
                y_top = (q_values.max() / 2) * 1.05
            ax.set_ylim(bottom=y_bottom, top=y_top)

    row_labels = ['Acceleration', 'Velocity', 'Displacement']
    for row, label in enumerate(row_labels):
        bbox = axes[row][0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.01, y_center, label,
            fontsize=cfg['font_axis_label'],
            ha='left', va='center',
            rotation=90,
            fontweight='bold',
        )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(legend_elements),
        fontsize=cfg['font_legend'],
        framealpha=0.9,
        handlelength=2.0,
        columnspacing=1.5,
    )

    if output_path is not None:
        suffix = cfg['output_suffix']
        output_path = Path(output_path).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig