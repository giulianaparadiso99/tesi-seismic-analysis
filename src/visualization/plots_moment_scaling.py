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
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_titles = {
        'pre_event': 'Pre-event (noise)',
        'p_wave': 'P-wave',
        's_wave': 'S-wave',
        'coda': 'Coda'
    }
    
    if q_subset is None:
        q_subset = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
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
        
        mean_r2 = np.nanmean(r_squared[valid])
        ax.text(0.98, 0.05, f'Mean R² = {mean_r2:.3f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                        alpha=0.7, edgecolor='gray'))
        
        ax.set_xlim(q_values.min() - 0.2, q_values.max() + 0.2)
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'ensemble_scaling_exponents.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig