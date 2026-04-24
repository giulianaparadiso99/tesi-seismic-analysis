import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from src.visualization.plot_settings import set_plot_style
colors, colors1 = set_plot_style()

def plot_scaling_curves(
    results: Dict,
    output_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    q_subset: Optional[np.ndarray] = None,
    colors: Optional[List] = colors1 
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
        
        n_q = len(q_plot)
        
        plot_colors = colors * (n_q // len(colors) + 1)
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
        output_file = output_dir / f'ensemble_scaling_curves_q{q_subset.min():.2f}-{q_subset.max():.2f}_n{n_q}.pdf'
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

def plot_single_scaling_curves(
    results: Dict,
    station: str,
    component: str,
    window_name: str,
    output_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
    q_subset: Optional[np.ndarray] = None,
    colors: Optional[List] = None
) -> plt.Figure:
    """
    Plot log(M_q) vs log(tau) for a single signal.
    
    Parameters
    ----------
    results : dict
        Output from analyze_single_signal()
    station : str
        Station code (for title)
    component : str
        Component code (for title)
    window_name : str
        Window name: 'pre_event', 'p_wave', 's_wave', 'coda'
    output_dir : str or Path, optional
        Directory to save figure. If None, figure is displayed but not saved.
    figsize : tuple of float, optional
        Figure size in inches (default: (10, 7))
    q_subset : np.ndarray, optional
        Subset of q values to plot. If None, plots all q values.
    colors : list, optional
        List of colors for different q values. If None, uses 'viridis' colormap.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    tau = results['tau']
    q_values = results['q']
    moments = results['moments']
    zeta = results['zeta']
    intercepts = results['intercepts']
    r_squared = results['r_squared']
    
    # Subset di q se specificato
    if q_subset is not None:
        q_mask = np.isin(q_values, q_subset)
        q_plot = q_values[q_mask]
        moments_plot = moments[:, q_mask]
        zeta_plot = zeta[q_mask]
        intercepts_plot = intercepts[q_mask]
        r_squared_plot = r_squared[q_mask]
    else:
        q_plot = q_values
        moments_plot = moments
        zeta_plot = zeta
        intercepts_plot = intercepts
        r_squared_plot = r_squared
    
    if colors is None:
        colors_plot = plt.cm.viridis(np.linspace(0.2, 0.9, len(q_plot)))
    else:
        colors_plot = colors * (len(q_plot) // len(colors) + 1)
        colors_plot = colors_plot[:len(q_plot)]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (q, color) in enumerate(zip(q_plot, colors_plot)):
        M_q = moments_plot[:, i]
        valid = (M_q > 0) & np.isfinite(M_q)
        
        if valid.sum() < 2:
            continue
        
        tau_valid = tau[valid]
        M_q_valid = M_q[valid]
        
        label = f'q={q:.1f}, ζ={zeta_plot[i]:.2f}'
        
        # Dati
        ax.loglog(tau_valid, M_q_valid, 'o', color=color, markersize=6,
                 alpha=0.7, label=label, markeredgewidth=0.5, 
                 markeredgecolor='white')
        
        # Fit line
        if not np.isnan(zeta_plot[i]):
            tau_fit = tau_valid
            log_M_fit = zeta_plot[i] * np.log10(tau_fit) + intercepts_plot[i]
            M_fit = 10 ** log_M_fit
            ax.loglog(tau_fit, M_fit, '--', color=color, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('τ (s)', fontsize=13)
    ax.set_ylabel('M_q(τ) = |Δx(τ)|^q', fontsize=13)
    
    window_titles = {
        'pre_event': 'Pre-event (noise)',
        'p_wave': 'P-wave',
        's_wave': 'S-wave',
        'coda': 'Coda'
    }
    title = f'{station}-{component}: {window_titles.get(window_name, window_name)}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=9, ncol=2, loc='lower right', framealpha=0.95,
             edgecolor='gray', fancybox=False, columnspacing=1.0,
             handletextpad=0.5, borderpad=0.4)
    ax.grid(True, alpha=0.3, which='both', linewidth=0.5)
    
    mean_r2 = np.nanmean(r_squared_plot)
    textstr = f'Mean R² = {mean_r2:.3f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'single_scaling_curves_{station}_{component}_{window_name}.pdf'
        output_file = output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def plot_single_scaling_exponents(
    results: Dict,
    station: str,
    component: str,
    window_name: str,
    output_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 6),
    point_color: Optional[str] = None
) -> plt.Figure:
    """
    Plot scaling exponents ζ(q) vs q for a single signal.
    
    Parameters
    ----------
    results : dict
        Output from analyze_single_signal()
    station : str
        Station code (for title)
    component : str
        Component code (for title)
    window_name : str
        Window name: 'pre_event', 'p_wave', 's_wave', 'coda'
    output_dir : str or Path, optional
        Directory to save figure. If None, figure is displayed but not saved.
    figsize : tuple of float, optional
        Figure size in inches (default: (9, 6))
    point_color : str or tuple, optional
        Color for data points. If None, uses 'steelblue' (default).
        Can be any matplotlib color (name, hex, RGB tuple).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    q_values = results['q']
    zeta = results['zeta']
    zeta_err = results['zeta_err']
    r_squared = results['r_squared']
    
    if point_color is None:
        point_color = 'steelblue'
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filtra valori validi
    valid = np.isfinite(zeta)
    
    # Plot ζ(q)
    ax.errorbar(q_values[valid], zeta[valid], yerr=zeta_err[valid],
               fmt='o', markersize=9, capsize=5, capthick=2,
               color=point_color, ecolor=point_color, alpha=0.8,
               label='Measured ζ(q)', linewidth=2, zorder=3)
    
    q_ref = np.linspace(q_values.min(), q_values.max(), 100)
    zeta_normal = q_ref / 2
    ax.plot(q_ref, zeta_normal, '--', color='red', linewidth=2.5,
           label='Normal diffusion (ζ=q/2)', alpha=0.7, zorder=2)
    
    if valid.sum() > 0:
        zeta_mean = np.nanmean(zeta[valid])
        ax.axhline(zeta_mean, color='orange', linestyle=':', linewidth=2,
                  label=f'Monofractal (ζ={zeta_mean:.2f})', alpha=0.7, zorder=1)
    
    ax.set_xlabel('Moment order q', fontsize=13)
    ax.set_ylabel('Scaling exponent ζ(q)', fontsize=13)
    
    window_titles = {
        'pre_event': 'Pre-event (noise)',
        'p_wave': 'P-wave',
        's_wave': 'S-wave',
        'coda': 'Coda'
    }
    title = f'{station}-{component}: {window_titles.get(window_name, window_name)}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95,
             edgecolor='gray', fancybox=False)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    mean_r2 = np.nanmean(r_squared[valid])
    h1_idx = np.argmin(np.abs(q_values - 1.0))
    h2_idx = np.argmin(np.abs(q_values - 2.0))
    
    textstr = f'Mean R²: {mean_r2:.3f}\n'
    textstr += f'ζ(q=1): {zeta[h1_idx]:.3f}\n'
    textstr += f'ζ(q=2): {zeta[h2_idx]:.3f}'
    
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7, edgecolor='gray')
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax.set_xlim(q_values.min() - 0.2, q_values.max() + 0.2)
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'single_scaling_exponents_{station}_{component}_{window_name}.pdf'
        output_file = output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig