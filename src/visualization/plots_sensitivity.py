"""
Visualization functions for sensitivity analysis results.

This module provides three types of comparative plots:
- Type A: Compare coda detection methods
- Type B: Compare picking methods (AR-AIC vs PhaseNet)
- Type C: Compare data types (acceleration, velocity, displacement)

Each plot type has multiple visualization functions (heatmaps, scatter plots, etc.)
organized in a 2×2 or 1×N subplot layout.
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

def _extract_metrics_for_scenario(
    results: Dict,
    data_type: str,
    coda_method: str,
    scenario: str,
    metric: str = 'rmse'
) -> Dict[str, float]:
    """
    Extract metric values for all windows in a given scenario.
    
    Parameters
    ----------
    results : dict
        Sensitivity results dictionary
    data_type : str
        Data type (acceleration, velocity, displacement)
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
    
    for window in ['pre_event', 'p_wave', 's_wave', 'coda']:
        scenario_data = results[data_type][coda_method][scenario]
        
        if window in scenario_data and scenario_data[window] is not None:
            if 'metrics' in scenario_data[window]:
                # Monte Carlo case
                metrics = scenario_data[window]['metrics']
            else:
                # Regular scenario case
                metrics = scenario_data[window]
            
            if metric in metrics:
                window_metrics[window] = metrics[metric]
            else:
                window_metrics[window] = np.nan
        else:
            window_metrics[window] = np.nan
    
    return window_metrics


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
        'distance_dependent': 'Distance-dep.',
        'monte_carlo': 'Monte Carlo'
    }


# ==============================================================================
# TYPE A: COMPARE CODA METHODS
# ==============================================================================

def plot_rmse_heatmap_by_coda(
    results: Dict,
    data_type: str,
    picking_method: str,
    save_dir: Path,
    scenarios: Optional[List[str]] = None
) -> None:
    """
    Plot RMSE heatmap comparing coda detection methods.
    
    Creates a 2×2 figure with one subplot per coda method.
    Each heatmap shows scenarios (rows) vs windows (columns).
    
    Parameters
    ----------
    results : dict
        Sensitivity results
    data_type : str
        Data type (acceleration, velocity, displacement)
    picking_method : str
        Picking method (ar_pick or phasenet)
    save_dir : Path
        Output directory for figure
    scenarios : list of str, optional
        List of scenarios to include. If None, use all except monte_carlo
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    if scenarios is None:
        scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                    'bias_early', 'bias_late', 'distance_dependent']
    
    coda_methods = ['rautian', 'arias', 'envelope', 'median']
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    scenario_labels = _get_scenario_labels()
    window_labels = _get_window_labels()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'RMSE Sensitivity - {data_type.capitalize()} ({picking_method})', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, coda_method in enumerate(coda_methods):
        ax = axes[idx]
        
        # Build heatmap data matrix
        heatmap_data = np.zeros((len(scenarios), len(windows)))
        
        for i, scenario in enumerate(scenarios):
            metrics = _extract_metrics_for_scenario(
                results, data_type, coda_method, scenario, metric='rmse'
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


def plot_scatter_by_coda(
    results: Dict,
    baseline_results: Dict,
    data_type: str,
    picking_method: str,
    scenario: str,
    save_dir: Path
) -> None:
    """
    Plot scatter comparison of baseline vs perturbed ζ(q) for coda methods.
    
    Creates a 2×2 figure with one subplot per coda method.
    Each scatter shows baseline ζ (x) vs perturbed ζ (y) with ideal y=x line.
    
    Parameters
    ----------
    results : dict
        Sensitivity results
    baseline_results : dict
        Baseline ζ(q) values
    data_type : str
        Data type
    picking_method : str
        Picking method
    scenario : str
        Perturbation scenario to plot
    save_dir : Path
        Output directory
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    coda_methods = ['rautian', 'arias', 'envelope', 'median']
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_labels = _get_window_labels()
    window_colors = colors[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'ζ(q) Comparison - {data_type.capitalize()} ({picking_method}) - {scenario}',
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, coda_method in enumerate(coda_methods):
        ax = axes[idx]
        
        # Plot each window
        for window, label, color in zip(windows, window_labels, window_colors):
            
            # Get baseline ζ
            baseline_key = f"{data_type}_{coda_method}"
            if baseline_key in baseline_results:
                df_baseline = baseline_results[baseline_key]
                zeta_baseline = df_baseline[df_baseline['window'] == window]['zeta'].values
            else:
                continue
            
            # Get perturbed ζ
            scenario_data = results[data_type][coda_method][scenario]
            if window not in scenario_data or scenario_data[window] is None:
                continue
            
            # Extract zeta from results (stored during analysis)
            # Note: this requires storing zeta arrays in results, not just metrics
            # For now, skip if not available
            # TODO: store zeta arrays in sensitivity results
            
            pass  # Placeholder - needs zeta arrays in results
        
        # Ideal line y=x
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Ideal')
        
        ax.set_xlabel('ζ baseline')
        ax.set_ylabel('ζ perturbed')
        ax.set_title(f'{coda_method.capitalize()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'scatter_{data_type}_{picking_method}_{scenario}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


def plot_zeta_confidence_by_coda(
    results: Dict,
    baseline_results: Dict,
    data_type: str,
    picking_method: str,
    q_values: np.ndarray,
    save_dir: Path
) -> None:
    """
    Plot ζ(q) curves with Monte Carlo confidence bands for coda methods.
    
    Creates a 2×2 figure with one subplot per coda method.
    Shows baseline ζ(q) with shaded confidence interval from Monte Carlo.
    
    Parameters
    ----------
    results : dict
        Sensitivity results (must include monte_carlo scenario)
    baseline_results : dict
        Baseline ζ(q) values
    data_type : str
        Data type
    picking_method : str
        Picking method
    q_values : np.ndarray
        Moment orders
    save_dir : Path
        Output directory
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    coda_methods = ['rautian', 'arias', 'envelope', 'median']
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_labels = _get_window_labels()
    window_colors = colors[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ζ(q) with Monte Carlo Confidence - {data_type.capitalize()} ({picking_method})',
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, coda_method in enumerate(coda_methods):
        ax = axes[idx]
        
        for window, label, color in zip(windows, window_labels, window_colors):
            
            # Get baseline
            baseline_key = f"{data_type}_{coda_method}"
            if baseline_key not in baseline_results:
                continue
            
            df_baseline = baseline_results[baseline_key]
            zeta_baseline = df_baseline[df_baseline['window'] == window]['zeta'].values
            
            if len(zeta_baseline) == 0:
                continue
            
            # Get Monte Carlo statistics
            mc_data = results[data_type][coda_method]['monte_carlo']
            if window not in mc_data or mc_data[window] is None:
                continue
            
            if 'statistics' not in mc_data[window]:
                continue
            
            stats = mc_data[window]['statistics']
            
            # Plot baseline
            ax.plot(q_values, zeta_baseline, '-o', color=color, label=label, 
                   linewidth=2, markersize=4)
            
            # Plot confidence band
            if 'p05' in stats and 'p95' in stats:
                ax.fill_between(q_values, stats['p05'], stats['p95'],
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Moment order q')
        ax.set_ylabel('Scaling exponent ζ(q)')
        ax.set_title(f'{coda_method.capitalize()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'zeta_confidence_{data_type}_{picking_method}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


def plot_error_distributions_by_coda(
    results: Dict,
    data_type: str,
    picking_method: str,
    save_dir: Path
) -> None:
    """
    Plot distributions of Δζ errors from Monte Carlo for coda methods.
    
    Creates a 2×2 figure with violin plots showing error distributions
    for each window across all q values.
    
    Parameters
    ----------
    results : dict
        Sensitivity results (must include monte_carlo scenario)
    data_type : str
        Data type
    picking_method : str
        Picking method
    save_dir : Path
        Output directory
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    # TODO: Implement violin/box plot of Δζ distributions
    # Requires storing individual MC run results, not just statistics
    
    logger.warning("plot_error_distributions_by_coda: Not yet implemented")
    pass


def plot_metrics_barplot_by_coda(
    results: Dict,
    data_type: str,
    picking_method: str,
    metric: str,
    save_dir: Path
) -> None:
    """
    Plot bar chart comparing metric values across scenarios for coda methods.
    
    Creates a 2×2 figure with grouped bar charts.
    
    Parameters
    ----------
    results : dict
        Sensitivity results
    data_type : str
        Data type
    picking_method : str
        Picking method
    metric : str
        Metric to plot (rmse, mae, correlation)
    save_dir : Path
        Output directory
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    coda_methods = ['rautian', 'arias', 'envelope', 'median']
    scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                'bias_early', 'bias_late', 'distance_dependent']
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    window_labels = _get_window_labels()
    scenario_labels = _get_scenario_labels()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{metric.upper()} Comparison - {data_type.capitalize()} ({picking_method})',
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for idx, coda_method in enumerate(coda_methods):
        ax = axes[idx]
        
        for i, window in enumerate(windows):
            values = []
            for scenario in scenarios:
                metrics = _extract_metrics_for_scenario(
                    results, data_type, coda_method, scenario, metric=metric
                )
                values.append(metrics.get(window, 0))
            
            offset = width * (i - 1.5)
            ax.bar(x + offset, values, width, label=window_labels[i])
        
        ax.set_xlabel('Perturbation scenario')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{coda_method.capitalize()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([scenario_labels.get(s, s) for s in scenarios], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'{metric}_barplot_{data_type}_{picking_method}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# TYPE B: COMPARE PICKING METHODS
# ==============================================================================

def plot_rmse_heatmap_by_picking(
    results: Dict,
    data_type: str,
    coda_method: str,
    save_dir: Path,
    scenarios: Optional[List[str]] = None
) -> None:
    """
    Plot RMSE heatmap comparing picking methods (AR-AIC vs PhaseNet).
    
    Creates a 1×2 figure with one subplot per picking method.
    
    Parameters
    ----------
    results : dict
        Sensitivity results
    data_type : str
        Data type
    coda_method : str
        Coda detection method
    save_dir : Path
        Output directory
    scenarios : list of str, optional
        Scenarios to include
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    if scenarios is None:
        scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                    'bias_early', 'bias_late', 'distance_dependent']
    
    picking_methods = ['ar_pick', 'phasenet']
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    scenario_labels = _get_scenario_labels()
    window_labels = _get_window_labels()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'RMSE Sensitivity - {data_type.capitalize()} ({coda_method})',
                 fontsize=14, fontweight='bold')
    
    for idx, picking_method in enumerate(picking_methods):
        ax = axes[idx]
        
        # Build heatmap data
        heatmap_data = np.zeros((len(scenarios), len(windows)))
        
        for i, scenario in enumerate(scenarios):
            metrics = _extract_metrics_for_scenario(
                results, data_type, coda_method, scenario, metric='rmse'
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
    output_file = save_dir / f'rmse_heatmap_{data_type}_{coda_method}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# TYPE C: COMPARE DATA TYPES
# ==============================================================================

def plot_rmse_heatmap_by_datatype(
    results: Dict,
    picking_method: str,
    coda_method: str,
    save_dir: Path,
    scenarios: Optional[List[str]] = None
) -> None:
    """
    Plot RMSE heatmap comparing data types (acceleration, velocity, displacement).
    
    Creates a 1×3 figure with one subplot per data type.
    
    Parameters
    ----------
    results : dict
        Sensitivity results
    picking_method : str
        Picking method
    coda_method : str
        Coda detection method
    save_dir : Path
        Output directory
    scenarios : list of str, optional
        Scenarios to include
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    if scenarios is None:
        scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                    'bias_early', 'bias_late', 'distance_dependent']
    
    data_types = ['acceleration', 'velocity', 'displacement']
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    scenario_labels = _get_scenario_labels()
    window_labels = _get_window_labels()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f'RMSE Sensitivity - {picking_method.replace("_", "-").upper()} ({coda_method})',
                 fontsize=14, fontweight='bold')
    
    for idx, data_type in enumerate(data_types):
        ax = axes[idx]
        
        # Build heatmap data
        heatmap_data = np.zeros((len(scenarios), len(windows)))
        
        for i, scenario in enumerate(scenarios):
            metrics = _extract_metrics_for_scenario(
                results, data_type, coda_method, scenario, metric='rmse'
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
    output_file = save_dir / f'rmse_heatmap_{picking_method}_{coda_method}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# AGGREGATED PLOTS (NOT TYPE-SPECIFIC)
# ==============================================================================

def plot_aggregated_sensitivity_matrix(
    results: Dict,
    metric: str,
    save_dir: Path
) -> None:
    """
    Plot aggregated sensitivity matrix across all combinations.
    
    Rows: data_type × coda_method combinations
    Columns: perturbation scenarios
    Color: metric value averaged across all windows
    
    Parameters
    ----------
    results : dict
        Sensitivity results
    metric : str
        Metric to aggregate (rmse, mae, correlation)
    save_dir : Path
        Output directory
    """
    
    from src.plot_settings import set_plot_style
    colors, colors1 = set_plot_style()
    
    data_types = list(results.keys())
    scenarios = ['noise_small', 'noise_medium', 'noise_large', 
                'bias_early', 'bias_late', 'distance_dependent']
    
    # Get all coda methods from first data type
    coda_methods = list(results[data_types[0]].keys())
    
    # Build row labels
    row_labels = []
    for dt in data_types:
        for cm in coda_methods:
            row_labels.append(f"{dt[:3]}-{cm[:3]}")
    
    # Build matrix
    n_rows = len(data_types) * len(coda_methods)
    n_cols = len(scenarios)
    matrix = np.zeros((n_rows, n_cols))
    
    row_idx = 0
    for data_type in data_types:
        for coda_method in coda_methods:
            for col_idx, scenario in enumerate(scenarios):
                # Average metric across all windows
                metrics = _extract_metrics_for_scenario(
                    results, data_type, coda_method, scenario, metric=metric
                )
                values = [v for v in metrics.values() if not np.isnan(v)]
                matrix[row_idx, col_idx] = np.mean(values) if len(values) > 0 else np.nan
            
            row_idx += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    
    scenario_labels = _get_scenario_labels()
    
    sns.heatmap(
        matrix,
        ax=ax,
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': f'{metric.upper()} (mean across windows)'},
        xticklabels=[scenario_labels.get(s, s) for s in scenarios],
        yticklabels=row_labels,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f'Aggregated {metric.upper()} Sensitivity Matrix', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Perturbation Scenario', fontsize=12)
    ax.set_ylabel('Data Type - Coda Method', fontsize=12)
    
    plt.tight_layout()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f'aggregated_{metric}_matrix.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_file}")


# ==============================================================================
# WRAPPER FUNCTION
# ==============================================================================

def generate_all_sensitivity_plots(
    results: Dict,
    baseline_results: Dict,
    q_values: np.ndarray,
    save_dir: Path,
    plot_types: List[str] = ['A', 'B', 'C'],
    picking_methods: List[str] = ['ar_pick', 'phasenet']
) -> None:
    """
    Generate all sensitivity analysis plots.
    
    Parameters
    ----------
    results : dict
        Sensitivity results with structure:
        {data_type: {coda_method: {scenario: {window: metrics}}}}
    baseline_results : dict
        Baseline ζ(q) results
    q_values : np.ndarray
        Moment orders
    save_dir : Path
        Output directory for figures
    plot_types : list of str
        Which plot types to generate: 'A', 'B', 'C', 'aggregated'
    picking_methods : list of str
        Picking methods available in results
    """
    
    logger.info("Starting sensitivity plot generation...")
    
    data_types = ['acceleration', 'velocity', 'displacement']
    coda_methods = ['rautian', 'arias', 'envelope', 'median']
    
    # Type A: Compare coda methods
    if 'A' in plot_types:
        logger.info("Generating Type A plots (compare coda methods)...")
        type_a_dir = save_dir / 'type_A'
        
        for data_type in data_types:
            for picking_method in picking_methods:
                plot_rmse_heatmap_by_coda(results, data_type, picking_method, type_a_dir)
                plot_zeta_confidence_by_coda(results, baseline_results, data_type, 
                                            picking_method, q_values, type_a_dir)
                plot_metrics_barplot_by_coda(results, data_type, picking_method, 
                                            'rmse', type_a_dir)
    
    # Type B: Compare picking methods
    if 'B' in plot_types:
        logger.info("Generating Type B plots (compare picking methods)...")
        type_b_dir = save_dir / 'type_B'
        
        for data_type in data_types:
            for coda_method in coda_methods:
                plot_rmse_heatmap_by_picking(results, data_type, coda_method, type_b_dir)
    
    # Type C: Compare data types
    if 'C' in plot_types:
        logger.info("Generating Type C plots (compare data types)...")
        type_c_dir = save_dir / 'type_C'
        
        for picking_method in picking_methods:
            for coda_method in coda_methods:
                plot_rmse_heatmap_by_datatype(results, picking_method, coda_method, type_c_dir)
    
    # Aggregated plots
    if 'aggregated' in plot_types:
        logger.info("Generating aggregated plots...")
        agg_dir = save_dir / 'aggregated'
        
        plot_aggregated_sensitivity_matrix(results, 'rmse', agg_dir)
        plot_aggregated_sensitivity_matrix(results, 'mae', agg_dir)
        plot_aggregated_sensitivity_matrix(results, 'correlation', agg_dir)
    
    logger.info("All sensitivity plots generated successfully")