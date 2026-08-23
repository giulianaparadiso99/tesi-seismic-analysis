"""
Visualization functions for sensitivity analysis results.

This module provides comparative plots for sensitivity analysis:
- Type A: Compare coda detection methods (within one data type)
- Type B: Compare picking methods (AR-AIC vs PhaseNet)
- Type C: Compare data types (acceleration, velocity, displacement)

Each function works with results from run_sensitivity_analysis().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _get_window_labels() -> List[str]:
    """Return standardized window labels for plots."""
    return ['Pre-event', 'P-wave', 'S-wave', 'Coda']


def _get_scenario_labels() -> Dict[str, str]:
    """Return mapping from scenario codes to display labels."""
    return {
        'noise_small': 'Noise σ=0.2s',
        'noise_medium': 'Noise σ=0.5s',
        'noise_large': 'Noise σ=1.0s',
        'bias_early': 'Bias -0.5s',
        'bias_late': 'Bias +0.5s',
        'monte_carlo': 'Monte Carlo'
    }


def _extract_metrics_for_scenario(
    results: Dict,
    coda_method: str,
    scenario: str,
    metric: str = 'rmse'
) -> Dict[str, float]:
    """
    Extract metric values for all windows in a given scenario.
    
    Parameters
    ----------
    results : dict
        Results from run_sensitivity_analysis()
        Structure: {coda_method: {scenario: {window: data}}}
    coda_method : str
        Coda detection method
    scenario : str
        Perturbation scenario name
    metric : str
        Metric to extract (rmse, mae, correlation, max_deviation)
        
    Returns
    -------
    window_metrics : dict
        Dictionary {window_name: metric_value}
    """
    
    window_metrics = {}
    
    if coda_method not in results:
        return window_metrics
    
    if scenario not in results[coda_method]:
        return window_metrics
    
    scenario_data = results[coda_method][scenario]
    
    for window in ['pre_event', 'p_wave', 's_wave', 'coda']:
        if window in scenario_data:
            window_data = scenario_data[window]
            
            # Handle regular scenarios (direct metrics dict)
            if isinstance(window_data, dict):
                if 'metrics' in window_data:
                    # Monte Carlo case
                    metrics = window_data['metrics']
                    if metric in metrics:
                        window_metrics[window] = metrics[metric]
                elif metric in window_data:
                    # Regular scenario case
                    window_metrics[window] = window_data[metric]
                else:
                    window_metrics[window] = np.nan
            else:
                window_metrics[window] = np.nan
        else:
            window_metrics[window] = np.nan
    
    return window_metrics


# ==============================================================================
# TYPE A: COMPARE CODA METHODS (WITHIN ONE DATA TYPE)
# ==============================================================================

def plot_rmse_heatmap_by_coda(
    results: Dict,
    data_type: str,
    picking_method: str,
    save_dir: Path,
    scenarios: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 10)
) -> None:
    """
    Plot RMSE heatmap comparing coda detection methods.
    
    Creates a 2×2 figure with one subplot per coda method.
    Each heatmap shows scenarios (rows) vs windows (columns).
    
    Parameters
    ----------
    results : dict
        Results from run_sensitivity_analysis()
        Structure: {coda_method: {scenario: {window: metrics}}}
    data_type : str
        Data type name (for title and filename)
    picking_method : str
        Picking method name (for title and filename)
    save_dir : Path
        Output directory for figure
    scenarios : list of str, optional
        List of scenarios to include. If None, use all except monte_carlo
    figsize : tuple, optional
        Figure size (width, height)
    """
    
    if scenarios is None:
        scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                    'bias_early', 'bias_late']
    
    coda_methods = list(results.keys())
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    scenario_labels = _get_scenario_labels()
    window_labels = _get_window_labels()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'RMSE Sensitivity - {data_type.capitalize()} ({picking_method})', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, coda_method in enumerate(coda_methods):
        ax = axes[idx]
        
        # Build heatmap data matrix
        heatmap_data = np.zeros((len(scenarios), len(windows)))
        
        for i, scenario in enumerate(scenarios):
            metrics = _extract_metrics_for_scenario(
                results, coda_method, scenario, metric='rmse'
            )
            for j, window in enumerate(windows):
                heatmap_data[i, j] = metrics.get(window, np.nan)
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'RMSE'},
            xticklabels=window_labels,
            yticklabels=[scenario_labels.get(s, s) for s in scenarios],
            vmin=0,
            vmax=0.5,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(f'{coda_method.capitalize()}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'rmse_heatmap_{data_type}_{picking_method}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


def plot_zeta_confidence_by_coda(
    results: Dict,
    baseline_results: Dict,
    data_type: str,
    picking_method: str,
    q_values: np.ndarray,
    save_dir: Path,
    figsize: Tuple[float, float] = (14, 10)
) -> None:
    """
    Plot ζ(q) curves with Monte Carlo confidence bands for coda methods.
    
    Creates a 2×2 figure with one subplot per coda method.
    Shows baseline ζ(q) with shaded confidence interval from Monte Carlo.
    
    Parameters
    ----------
    results : dict
        Results from run_sensitivity_analysis() (must include monte_carlo)
        Structure: {coda_method: {scenario: {window: {statistics: {...}, metrics: {...}}}}}
    baseline_results : dict
        Baseline ζ(q) DataFrames {coda_method: df_summary}
    data_type : str
        Data type name
    picking_method : str
        Picking method name
    q_values : np.ndarray
        Moment orders
    save_dir : Path
        Output directory
    figsize : tuple, optional
        Figure size
    """
    
    coda_methods = list(results.keys())
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_labels = _get_window_labels()
    
    # Define colors for each window
    window_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'ζ(q) with Monte Carlo Confidence - {data_type.capitalize()} ({picking_method})',
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, coda_method in enumerate(coda_methods):
        ax = axes[idx]
        
        # Check if we have baseline results for this coda method
        if coda_method not in baseline_results:
            ax.text(0.5, 0.5, 'No baseline data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{coda_method.capitalize()}', fontweight='bold')
            continue
        
        df_baseline = baseline_results[coda_method]
        
        for window, label, color in zip(windows, window_labels, window_colors):
            
            # Get baseline ζ(q)
            baseline_window = df_baseline[df_baseline['window'] == window].sort_values('q')
            
            if len(baseline_window) == 0:
                continue
            
            zeta_baseline = baseline_window['zeta'].values
            q_baseline = baseline_window['q'].values
            
            # Get Monte Carlo statistics
            if 'monte_carlo' not in results[coda_method]:
                continue
            
            mc_data = results[coda_method]['monte_carlo']
            
            if window not in mc_data:
                continue
            
            window_data = mc_data[window]
            
            if 'statistics' not in window_data:
                continue
            
            stats = window_data['statistics']
            
            # Plot baseline
            ax.plot(q_baseline, zeta_baseline, '-o', color=color, label=label, 
                   linewidth=2, markersize=4, zorder=3)
            
            # Plot confidence band
            if 'p05' in stats and 'p95' in stats:
                # Ensure same length
                n_points = min(len(q_baseline), len(stats['p05']))
                ax.fill_between(
                    q_baseline[:n_points], 
                    stats['p05'][:n_points], 
                    stats['p95'][:n_points],
                    color=color, 
                    alpha=0.2,
                    zorder=1
                )
        
        ax.set_xlabel('Moment order q', fontsize=11)
        ax.set_ylabel('Scaling exponent ζ(q)', fontsize=11)
        ax.set_title(f'{coda_method.capitalize()}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'zeta_confidence_{data_type}_{picking_method}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# TYPE B: COMPARE PICKING METHODS
# ==============================================================================

def plot_rmse_heatmap_by_picking(
    results_dict: Dict[str, Dict],
    data_type: str,
    coda_method: str,
    save_dir: Path,
    scenarios: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 6)
) -> None:
    """
    Plot RMSE heatmap comparing picking methods (AR-AIC vs PhaseNet).
    
    Creates a 1×2 figure with one subplot per picking method.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with structure: {picking_method: results}
        where results = output from run_sensitivity_analysis()
    data_type : str
        Data type name
    coda_method : str
        Coda detection method to compare
    save_dir : Path
        Output directory
    scenarios : list of str, optional
        Scenarios to include
    figsize : tuple, optional
        Figure size
    """
    
    if scenarios is None:
        scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                    'bias_early', 'bias_late']
    
    picking_methods = list(results_dict.keys())
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    scenario_labels = _get_scenario_labels()
    window_labels = _get_window_labels()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'RMSE Sensitivity - {data_type.capitalize()} ({coda_method})',
                 fontsize=14, fontweight='bold')
    
    for idx, picking_method in enumerate(picking_methods):
        ax = axes[idx]
        
        results = results_dict[picking_method]
        
        # Build heatmap data
        heatmap_data = np.zeros((len(scenarios), len(windows)))
        
        for i, scenario in enumerate(scenarios):
            metrics = _extract_metrics_for_scenario(
                results, coda_method, scenario, metric='rmse'
            )
            for j, window in enumerate(windows):
                heatmap_data[i, j] = metrics.get(window, np.nan)
        
        # Plot
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'RMSE'},
            xticklabels=window_labels,
            yticklabels=[scenario_labels.get(s, s) for s in scenarios],
            vmin=0,
            vmax=0.5,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(f'{picking_method.replace("_", "-").upper()}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'rmse_heatmap_{data_type}_{coda_method}_picking_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# TYPE C: COMPARE DATA TYPES
# ==============================================================================

def plot_rmse_heatmap_by_datatype(
    results_dict: Dict[str, Dict],
    picking_method: str,
    coda_method: str,
    save_dir: Path,
    scenarios: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 6)
) -> None:
    """
    Plot RMSE heatmap comparing data types (acceleration, velocity, displacement).
    
    Creates a 1×3 figure with one subplot per data type.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with structure: {data_type: results}
        where results = output from run_sensitivity_analysis()
    picking_method : str
        Picking method name (for title)
    coda_method : str
        Coda detection method to compare
    save_dir : Path
        Output directory
    scenarios : list of str, optional
        Scenarios to include
    figsize : tuple, optional
        Figure size
    """
    
    if scenarios is None:
        scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                    'bias_early', 'bias_late']
    
    data_types = list(results_dict.keys())
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    scenario_labels = _get_scenario_labels()
    window_labels = _get_window_labels()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'RMSE Sensitivity - {picking_method.replace("_", "-").upper()} ({coda_method})',
                 fontsize=14, fontweight='bold')
    
    for idx, data_type in enumerate(data_types):
        ax = axes[idx]
        
        results = results_dict[data_type]
        
        # Build heatmap data
        heatmap_data = np.zeros((len(scenarios), len(windows)))
        
        for i, scenario in enumerate(scenarios):
            metrics = _extract_metrics_for_scenario(
                results, coda_method, scenario, metric='rmse'
            )
            for j, window in enumerate(windows):
                heatmap_data[i, j] = metrics.get(window, np.nan)
        
        # Plot
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'RMSE'},
            xticklabels=window_labels,
            yticklabels=[scenario_labels.get(s, s) for s in scenarios],
            vmin=0,
            vmax=0.5,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(f'{data_type.capitalize()}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'rmse_heatmap_{picking_method}_{coda_method}_datatype_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# HELPER FUNCTIONS FOR LOADING RESULTS
# ==============================================================================

def load_sensitivity_results(
    base_dir: Path,
    data_type: str,
    picking_method: str
) -> Dict:
    """
    Load sensitivity results from pickle file.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing processed data
    data_type : str
        Data type (acceleration, velocity, displacement)
    picking_method : str
        Picking method (ar_pick, phasenet)
        
    Returns
    -------
    results : dict
        Results dictionary with structure from run_sensitivity_analysis()
    """
    import pickle
    
    results_dir = base_dir / '05_sensitivity_analysis' / picking_method / data_type
    
    # Try to load complete results
    results = {}
    
    # Look for method-specific pickle files
    for pkl_file in results_dir.glob(f'{data_type}_*_complete.pkl'):
        coda_method = pkl_file.stem.split('_')[1]
        
        with open(pkl_file, 'rb') as f:
            results[coda_method] = pickle.load(f)
    
    if len(results) == 0:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    return results


def load_baseline_results(
    base_dir: Path,
    data_type: str,
    picking_method: str
) -> Dict[str, pd.DataFrame]:
    """
    Load baseline moment scaling results.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing processed data
    data_type : str
        Data type
    picking_method : str
        Picking method
        
    Returns
    -------
    baseline_results : dict
        {coda_method: df_summary}
    """
    
    baseline_dir = base_dir / '04a_moment_scaling_spatial' / picking_method / data_type
    
    baseline_results = {}
    
    for coda_method in ['rautian', 'arias', 'envelope', 'median']:
        summary_file = baseline_dir / coda_method / 'ensemble_spatial_summary.parquet'
        
        if summary_file.exists():
            baseline_results[coda_method] = pd.read_parquet(summary_file)
    
    return baseline_results