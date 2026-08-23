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

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from src.visualization.plot_settings import set_plot_style
colors, colors1 = set_plot_style()

# Shared figure-mode presets, used by plot_scaling_exponents_v2 and
# plot_scaling_exponents_comparison.
_MODE_SETTINGS = {
    'interactive': dict(
        figsize=(16, 12), dpi_save=150,
        font_title=13, font_axis_label=11, font_tick=10, font_legend=10,
        linewidth_ref=2.0, linewidth_fit=2.0, markersize=6, capsize=4,
        output_suffix='.pdf',
    ),
    'paper': dict(
        figsize=(6.89, 5.5), dpi_save=600,
        font_title=10, font_axis_label=9, font_tick=8, font_legend=8,
        linewidth_fit=1.0, linewidth_ref=0.8, markersize=3, capsize=4,
        output_suffix='.png',
    ),
    'poster': dict(
        figsize=(14.85, 11.0), dpi_save=600,
        font_title=18, font_axis_label=15, font_tick=13, font_legend=13,
        linewidth_ref=2.5, linewidth_fit=2.5, markersize=6, capsize=4,
        output_suffix='.png',
    ),
    'thesis': dict(
        figsize=(14.85, 6.86), dpi_save=600,
        font_title=10, font_axis_label=9, font_tick=8, font_legend=10,
        linewidth_ref=1.8, linewidth_fit=1.8, markersize=5, capsize=3,
        output_suffix='.png',
    ),
}

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
        y_max_ref = (q_values.max() / 2) * 1.05
        if valid.any():
            y_min_data = (zeta[valid] - zeta_err[valid]).min()
            y_max_data = (zeta[valid] + zeta_err[valid]).max()
            y_bottom = min(0.0, y_min_data * 1.1)
            y_top = max(y_max_data * 1.1, y_max_ref)
        else:
            y_bottom = 0.0
            y_top = y_max_ref
        ax.set_ylim(bottom=y_bottom, top=y_top)
            
    plt.tight_layout()

    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'ensemble_scaling_exponents.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig

def clean_log_formatter(x, pos):
    exp = int(np.floor(np.log10(x)))
    coeff = round(x / 10**exp)
    if coeff == 1:
        return r'$10^{%d}$' % exp
    else:
        return r'$%d \times 10^{%d}$' % (coeff, exp)

def nice_log_ticks(lo, hi, n=4):
    """Return n approximately evenly spaced round values in log scale."""
    ticks = np.logspace(np.log10(lo), np.log10(hi), n)
    # Round each tick to 1 significant figure
    rounded = []
    for t in ticks:
        exp = np.floor(np.log10(t))
        mantissa = t / 10**exp
        rounded.append(round(mantissa) * 10**exp)
    return sorted(set(rounded))

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

    if mode not in _MODE_SETTINGS:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {tuple(_MODE_SETTINGS)}"
        )
    cfg = _MODE_SETTINGS[mode]

    signal_types = ['acceleration', 'velocity', 'displacement']
    windows = ['p_wave', 's_wave', 'coda']
    window_titles = {'p_wave': 'P-wave', 's_wave': 'S-wave', 'coda': 'Coda'}

    q_colors = {
        0.5: '#00807F',
        1.0: '#C8861D',
        2.0: '#729EC1',
        3.0: '#8B6BAE',
    }

    fig, axes = plt.subplots(
        3, 3,
        figsize=cfg['figsize'],
    )
    fig.subplots_adjust(
        top=0.88, bottom=0.08, left=0.10, right=0.97,
        hspace=0.35, wspace=0.35,
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

            if row == 0:
                ax.set_title(
                    window_titles[window_name],
                    fontsize=cfg['font_title'],
                    fontweight='bold',
                )

            if row == 2:
                ax.set_xlabel(r'$\tau$ (s)', fontsize=cfg['font_axis_label'])

            ax.tick_params(labelsize=cfg['font_tick'])
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

            # Raccogliere tau e M_q validi da tutti i q che verranno plottati
            tau_valid_all = []
            M_q_valid_all = []
            for q_val in q_colors.keys():
                q_idx = np.where(np.isclose(q_values, q_val))[0]
                if len(q_idx) == 0:
                    continue
                q_idx = q_idx[0]
                M_q = moments_mean[:, q_idx]
                valid = (M_q > 0) & np.isfinite(M_q)
                if valid.sum() < 2:
                    continue
                tau_valid_all.extend(tau[valid])
                M_q_valid_all.extend(M_q[valid])

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

            # Imposta locator DOPO il plot, così non vengono sovrascritti da loglog
            if tau_valid_all and M_q_valid_all:
                tau_lo = np.min(tau_valid_all)
                tau_hi = np.max(tau_valid_all)
                x_ticks = np.logspace(np.log10(tau_lo), np.log10(tau_hi), num=4)
                ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(x_ticks))
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_log_formatter))
                ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

                M_lo = np.min(M_q_valid_all)
                M_hi = np.max(M_q_valid_all)
                y_ticks = np.logspace(np.log10(M_lo), np.log10(M_hi), num=4)
                ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(y_ticks))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_log_formatter))
                ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())


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
    if mode not in _MODE_SETTINGS:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {tuple(_MODE_SETTINGS)}"
        )
    cfg = _MODE_SETTINGS[mode]

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
        hspace=0.35, wspace=0.35,
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


            if row == 0:
                ax.set_title(
                    window_titles[window_name],
                    fontsize=cfg['font_title'],
                    fontweight='bold',
                )

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

def plot_scaling_exponents_comparison(
    results_by_method: Dict[str, Dict[str, Dict]],
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'thesis',
    show_uncertainty: bool = True,
    uncertainty_style: str = 'band',
) -> plt.Figure:
    """
    Plot scaling exponents zeta(q) vs q for P-wave, S-wave, and coda
    windows, across acceleration, velocity, and displacement signals
    (3x3 grid), comparing all four coda onset methods in each panel.

    The P-wave column shows a single curve, since P-wave window
    boundaries do not depend on the coda onset method and results are
    therefore identical across methods. The S-wave and coda columns
    overlay one curve per coda method.

    Parameters
    ----------
    results_by_method : dict
        Dictionary mapping coda method name to the output of
        load_scaling_results_by_signal(): {method_name:
        results_by_signal}. Expected keys: 'rautian', 'arias',
        'envelope', 'median'.
    output_path : str or Path, optional
        If provided, save the figure to this path. File extension is
        set by the mode (.pdf for interactive, .png otherwise).
    mode : str, optional
        Output mode controlling figure size and font sizes. One of
        'thesis', 'paper', 'poster', 'interactive' (default: 'thesis').
    show_uncertainty : bool, default=True
        If False, only the central zeta(q) curves are drawn, with no
        uncertainty representation.
    uncertainty_style : str, default='band'
        Only used if show_uncertainty is True. One of 'band' (shaded
        semi-transparent region) or 'errorbar' (classic error bars,
        horizontally dodged per method to remain distinguishable).

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If mode or uncertainty_style is not a recognized option.
    """
    if mode not in _MODE_SETTINGS:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {tuple(_MODE_SETTINGS)}"
        )
    if uncertainty_style not in ('band', 'errorbar'):
        raise ValueError(
            f"Invalid uncertainty_style '{uncertainty_style}'. "
            f"Must be one of: ('band', 'errorbar')"
        )
    cfg = _MODE_SETTINGS[mode]

    method_order = ['rautian', 'arias', 'envelope', 'median']
    method_colors = {
        'rautian': '#00807F',
        'arias': '#C8861D',
        'envelope': '#729EC1',
        'median': '#8B6BAE',
    }
    method_labels = {
        'rautian': 'Rautian', 'arias': 'Arias',
        'envelope': 'Envelope', 'median': 'Median',
    }
    p_wave_color = "#010A0A"
    ref_color = '#C0392B'

    signal_types = ['acceleration', 'velocity', 'displacement']
    windows = ['p_wave', 's_wave', 'coda']
    window_titles = {'p_wave': 'P-wave', 's_wave': 'S-wave', 'coda': 'Coda'}

    fig, axes = plt.subplots(3, 3, figsize=cfg['figsize'])
    fig.subplots_adjust(
        top=0.86, bottom=0.10, left=0.10, right=0.97,
        hspace=0.35, wspace=0.35,
    )

    dodge_fractions = np.linspace(-0.4, 0.4, len(method_order))

    for row, signal_type in enumerate(signal_types):
        for col, window_name in enumerate(windows):
            ax = axes[row][col]

            if col == 0:
                ax.set_ylabel(r'$\zeta(q)$', fontsize=cfg['font_axis_label'])
            if row == 0:
                ax.set_title(
                    window_titles[window_name],
                    fontsize=cfg['font_title'], fontweight='bold',
                )
            ax.set_xlabel(r'$q$', fontsize=cfg['font_axis_label'])
            ax.tick_params(labelsize=cfg['font_tick'])
            ax.grid(True, alpha=0.3, linewidth=0.5)

            methods_to_plot = [method_order[0]] if window_name == 'p_wave' else method_order

            q_min, q_max = None, None
            y_bottom, y_top = 0.0, None
            any_data = False

            for m_idx, method in enumerate(methods_to_plot):
                results = results_by_method.get(method, {}).get(signal_type)
                if results is None or window_name not in results or results[window_name] is None:
                    continue

                q_values = results[window_name]['ensemble']['q']
                zeta = results[window_name]['scaling']['zeta']
                zeta_err = results[window_name]['scaling']['zeta_err']
                valid = np.isfinite(zeta)
                if not valid.any():
                    continue
                any_data = True

                color = p_wave_color if window_name == 'p_wave' else method_colors[method]
                q_step = np.median(np.diff(np.sort(q_values[valid]))) if valid.sum() > 1 else 0.25
                q_dodge = q_values[valid] + dodge_fractions[m_idx] * q_step * 0.5

                if show_uncertainty and uncertainty_style == 'band':
                    ax.fill_between(
                        q_values[valid],
                        zeta[valid] - zeta_err[valid],
                        zeta[valid] + zeta_err[valid],
                        color=color, alpha=0.15, zorder=1,
                    )
                    ax.plot(
                        q_values[valid], zeta[valid],
                        color=color, linewidth=cfg['linewidth_fit'],
                        zorder=3,
                    )
                elif show_uncertainty and uncertainty_style == 'errorbar':
                    ax.errorbar(
                        q_dodge, zeta[valid], yerr=zeta_err[valid],
                        fmt='o', markersize=cfg['markersize'] * 0.7,
                        capsize=cfg['capsize'] * 0.7, capthick=1.0,
                        color=color, ecolor=color, alpha=0.85, zorder=3,
                    )
                else:
                    ax.plot(
                        q_values[valid], zeta[valid],
                        color=color, linewidth=cfg['linewidth_fit'],
                        marker='o', markersize=cfg['markersize'] * 0.7,
                        zorder=3,
                    )

                q_min = q_values.min() if q_min is None else min(q_min, q_values.min())
                q_max = q_values.max() if q_max is None else max(q_max, q_values.max())
                y_lo = (zeta[valid] - zeta_err[valid]).min() if show_uncertainty else zeta[valid].min()
                y_hi = (zeta[valid] + zeta_err[valid]).max() if show_uncertainty else zeta[valid].max()
                y_bottom = min(y_bottom, y_lo * 1.1)
                y_top = y_hi * 1.1 if y_top is None else max(y_top, y_hi * 1.1)

            if not any_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=cfg['font_axis_label'], color='gray')
                continue

            q_ref = np.linspace(q_min, q_max, 100)
            ax.plot(
                q_ref, q_ref / 2, '--',
                color=ref_color, linewidth=cfg['linewidth_ref'],
                alpha=0.7, zorder=2,
            )
            y_top = max(y_top, (q_max / 2) * 1.05)

            ax.set_xlim(q_min - 0.2, q_max + 0.2)
            ax.set_ylim(bottom=y_bottom, top=y_top)

    row_labels = ['Acceleration', 'Velocity', 'Displacement']
    for row, label in enumerate(row_labels):
        bbox = axes[row][0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.01, y_center, label,
            fontsize=cfg['font_axis_label'], ha='left', va='center',
            rotation=90, fontweight='bold',
        )

    legend_elements = [
        plt.Line2D([0], [0], color=p_wave_color, marker='o',
                   markersize=cfg['markersize'] * 0.7,
                   label='P-wave (method-independent)'),
    ]
    legend_elements += [
        plt.Line2D([0], [0], color=method_colors[m], marker='o',
                   markersize=cfg['markersize'] * 0.7,
                   label=method_labels[m])
        for m in method_order
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color=ref_color, linestyle='--',
                   linewidth=cfg['linewidth_ref'], alpha=0.7,
                   label=r'Normal diffusion ($\zeta = q/2$)')
    )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_elements),
        fontsize=cfg['font_legend'],
        framealpha=0.9,
        handlelength=1.5,
        columnspacing=1.2,
    )

    if output_path is not None:
        suffix = cfg['output_suffix']
        output_path = Path(output_path).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig

def plot_tau_subinterval_sensitivity(
    df_by_signal: Dict[str, pd.DataFrame],
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'thesis',
) -> plt.Figure:
    """
    Plot zeta(q) vs q for the P-wave window, comparing the baseline fit
    (full tau range) against fits restricted to the lower and upper
    tau sub-intervals, across acceleration, velocity, and displacement
    signals (1x3 grid).

    Parameters
    ----------
    df_by_signal : dict
        Dictionary mapping signal type to the output of
        compute_tau_subinterval_sensitivity(): {'acceleration': df_acc,
        'velocity': df_vel, 'displacement': df_disp}.
    output_path : str or Path, optional
        If provided, save the figure to this path. File extension is
        set by the mode (.pdf for interactive, .png otherwise).
    mode : str, optional
        Output mode controlling figure size and font sizes. One of
        'thesis', 'paper', 'poster', 'interactive' (default: 'thesis').

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If mode is not a recognized option.
    """
    if mode not in _MODE_SETTINGS:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {tuple(_MODE_SETTINGS)}"
        )
    cfg = _MODE_SETTINGS[mode]

    signal_types = ['acceleration', 'velocity', 'displacement']
    signal_titles = {
        'acceleration': 'Acceleration',
        'velocity': 'Velocity',
        'displacement': 'Displacement',
    }
    series_colors = {
        'baseline': '#010A0A',
        'lower_half': '#3B7EA1',
        'upper_half': '#D17A22',
    }
    series_labels = {
        'baseline': 'Baseline (full range)',
        'lower_half': 'Lower half of $\\tau$',
        'upper_half': 'Upper half of $\\tau$',
    }

    fig, axes = plt.subplots(1, 3, figsize=(cfg['figsize'][0], cfg['figsize'][0] / 3))
    fig.subplots_adjust(top=0.72, bottom=0.18, left=0.08, right=0.98, wspace=0.35)

    for col, signal_type in enumerate(signal_types):
        ax = axes[col]
        ax.set_title(
            signal_titles[signal_type],
            fontsize=cfg['font_title'], fontweight='bold',
        )
        ax.set_xlabel(r'$q$', fontsize=cfg['font_axis_label'])
        if col == 0:
            ax.set_ylabel(r'$\zeta(q)$', fontsize=cfg['font_axis_label'])
        ax.tick_params(labelsize=cfg['font_tick'])
        ax.grid(True, alpha=0.3, linewidth=0.5)

        df = df_by_signal.get(signal_type)
        if df is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=cfg['font_axis_label'], color='gray')
            continue

        for series_key, zeta_col, err_col in [
            ('baseline', 'zeta_baseline', 'zeta_err_baseline'),
            ('lower_half', 'zeta_lower_half', 'zeta_err_lower_half'),
            ('upper_half', 'zeta_upper_half', 'zeta_err_upper_half'),
        ]:
            valid = df[zeta_col].notna()
            color = series_colors[series_key]
            ax.fill_between(
                df.loc[valid, 'q'],
                df.loc[valid, zeta_col] - df.loc[valid, err_col],
                df.loc[valid, zeta_col] + df.loc[valid, err_col],
                color=color, alpha=0.15, zorder=1,
            )
            ax.plot(
                df.loc[valid, 'q'], df.loc[valid, zeta_col],
                color=color, linewidth=cfg['linewidth_fit'], zorder=3,
            )

        q_ref = np.linspace(df['q'].min(), df['q'].max(), 100)
        ax.plot(
            q_ref, q_ref / 2, '--',
            color='#C0392B', linewidth=cfg['linewidth_ref'],
            alpha=0.7, zorder=2,
        )

    legend_elements = [
        plt.Line2D([0], [0], color=series_colors[k], linewidth=cfg['linewidth_fit'],
                   label=series_labels[k])
        for k in ('baseline', 'lower_half', 'upper_half')
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color='#C0392B', linestyle='--',
                   linewidth=cfg['linewidth_ref'], alpha=0.7,
                   label=r'Normal diffusion ($\zeta = q/2$)')
    )
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_elements),
        fontsize=cfg['font_legend'],
        framealpha=0.9,
        handlelength=1.8,
        columnspacing=1.3,
    )

    if output_path is not None:
        suffix = cfg['output_suffix']
        output_path = Path(output_path).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig