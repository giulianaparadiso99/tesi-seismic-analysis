"""
plot_segmentation.py
--------------------
Visualization functions for seismic event segmentation and phase detection.

This module provides comprehensive plotting utilities for visualizing and
validating seismic phase identification results, including:
- Theoretical P/S arrival time predictions from crustal velocity models
- AR-AIC and PhaseNet onset detection performance
- Four-window segmentation (pre-event, P-wave, S-wave, coda)
- Coda onset detection method comparison (Rautian, Arias, Envelope)
- Statistical validation and residual analysis

The module supports multiple phase detection methods and enables direct
comparison of their performance through various diagnostic plots.

Functions
---------
Theoretical arrivals and crustal velocities:
    display_theoretical_arrivals_table : Display table of predicted arrivals
    plot_apparent_vs_crustal_velocities : Compare observed vs CRUST1.0 velocities
    plot_crustal_velocities_vs_distance : Show velocity variation with distance
    plot_theoretical_arrivals : Plot P/S arrival times vs epicentral distance

Onset detection validation:
    plot_onset_detection_results : Visualize AR-AIC detection on waveforms
    plot_onset_detection_results_v2 : Generic plotter supporting multiple methods
    plot_coda_onset_results : Visualize coda detection on three-component data

Coda method comparison:
    plot_coda_scatter_comparison : Scatter plots comparing method pairs
    plot_bland_altman_comparison : Bland-Altman agreement analysis
    plot_residuals_vs_distance : Method differences vs epicentral distance
    plot_pairwise_difference_histograms : Distribution of method differences
    plot_correlation_matrix_heatmap : Correlation matrix between methods

Window segmentation visualization:
    get_station_components : Auto-detect component naming convention
    plot_station_windows : Show four-window segmentation for one station
    plot_multiple_stations : Batch plotting for multiple stations
    plot_window_comparison : Compare different windowing strategies

Notes
-----
Onset detection methods supported:
- AR-AIC: Autoregressive model with Akaike Information Criterion
- PhaseNet: Deep learning phase picker (Zhu & Beroza, 2019)

Coda onset detection methods:
- Rautian (1978): t_coda = 2×t_S from origin time
- Arias (dt_PS): Empirical rule based on S-P time difference
- Envelope: Energy-based detection using signal envelope

Component naming conventions automatically detected:
- Standard ITACA: HNE, HNN, HNZ (East, North, Vertical)
- Alternative: HGE, HGN, HGZ or HN1, HN2, HNZ

References
----------
Rautian, T. G., & Khalturin, V. I. (1978). "The use of the coda for
    determination of the earthquake source spectrum." Bulletin of the
    Seismological Society of America, 68(4), 923-948.
Zhu, W., & Beroza, G. C. (2019). "PhaseNet: a deep-neural-network-based
    seismic arrival-time picking method." Geophysical Journal International,
    216(1), 261-273.
Lazo, G., et al. (2022). CRUST1.0 velocity model implementation for arrival
    time prediction.

Examples
--------
>>> from src.visualization.plots_segmentation import (
...     plot_theoretical_arrivals,
...     plot_onset_detection_results_v2,
...     plot_station_windows
... )
>>>
>>> # Plot theoretical arrivals
>>> fig = plot_theoretical_arrivals(
...     df_meta_stations,
...     output_path='../figures/theoretical_arrivals.pdf'
... )
>>>
>>> # Compare AR-AIC vs PhaseNet detection
>>> figs_ar = plot_onset_detection_results_v2(
...     signals_dict, df_results_ar,
...     method='ar_pick',
...     stations=['ACER', 'CLFR', 'SURF'],
...     output_dir='../figures/onset_detection/ar_aic/'
... )
>>>
>>> figs_pn = plot_onset_detection_results_v2(
...     signals_dict, df_results_pn,
...     method='phasenet',
...     stations=['ACER', 'CLFR', 'SURF'],
...     output_dir='../figures/onset_detection/phasenet/'
... )
>>>
>>> # Visualize four-window segmentation
>>> fig = plot_station_windows(
...     station='BRZ',
...     signals_dict=signals_dict,
...     windowed_signals=windowed_dict,
...     coda_method='rautian',
...     output_path='../figures/windows/BRZ_windows.pdf'
... )
>>> plt.show()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, linregress
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from IPython.display import display
from src.visualization.plot_settings import set_plot_style
colors, colors1 = set_plot_style()

def display_theoretical_arrivals_table(df_stations, n_rows=10):
    """
    Display table of theoretical arrival times and crustal velocities.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with theoretical arrivals
    n_rows : int, optional
        Number of rows to display (default: 10)
    
    Returns
    -------
    pd.DataFrame
        Formatted table subset
    """
    table = df_stations[[
        'STATION_CODE',
        'EPICENTRAL_DISTANCE_KM',
        'vp_crust',
        'vs_crust',
        'origin_time',
        't_p_theo_seconds',
        't_s_theo_seconds'
    ]].copy()

    table = table.round({
        'vp_crust': 2,
        'vs_crust': 2,
        'origin_time': 2,
        't_p_theo_seconds': 2,
        't_s_theo_seconds': 2
    })

    table = table.sort_values('EPICENTRAL_DISTANCE_KM').reset_index(drop=True)
    table.index += 1  # 1-based index

    print(f"First {min(n_rows, len(table))} stations sorted by epicentral distance:")
    display(table.head(n_rows))
    return table


def plot_apparent_vs_crustal_velocities(
    df_meta_stations: pd.DataFrame, 
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """
    Compare apparent velocities (from detected arrivals) with CRUST1.0 velocities.
    
    Apparent velocity = distance / (t_detected - origin_time)
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station metadata
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Calculate apparent velocities
    df = df_meta_stations.copy()
    df['v_p_apparent'] = df['EPICENTRAL_DISTANCE_KM'] / (df['t_p_detected_seconds'] - df['origin_time'])
    df['v_s_apparent'] = df['EPICENTRAL_DISTANCE_KM'] / (df['t_s_detected_seconds'] - df['origin_time'])
    
    # Remove outliers (negative or unrealistic velocities)
    df = df[(df['v_p_apparent'] > 0) & (df['v_p_apparent'] < 15)]
    df = df[(df['v_s_apparent'] > 0) & (df['v_s_apparent'] < 10)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # P-wave
    axes[0].scatter(df['vp_crust'], df['v_p_apparent'], 
                    s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 1:1 line
    lim = [min(df['vp_crust'].min(), df['v_p_apparent'].min()),
           max(df['vp_crust'].max(), df['v_p_apparent'].max())]
    axes[0].plot(lim, lim, 'r--', linewidth=2, label='1:1 line', zorder=0)
    
    axes[0].set_xlabel('v_P CRUST1.0 (km/s)', fontsize=12)
    axes[0].set_ylabel('v_P apparent (km/s)', fontsize=12)
    axes[0].set_title('P-wave: CRUST1.0 vs Apparent velocity', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # S-wave
    axes[1].scatter(df['vs_crust'], df['v_s_apparent'],
                    s=80, alpha=0.7, edgecolor='black', linewidth=0.5, color='coral')
    
    lim = [min(df['vs_crust'].min(), df['v_s_apparent'].min()),
           max(df['vs_crust'].max(), df['v_s_apparent'].max())]
    axes[1].plot(lim, lim, 'r--', linewidth=2, label='1:1 line', zorder=0)
    
    axes[1].set_xlabel('v_S CRUST1.0 (km/s)', fontsize=12)
    axes[1].set_ylabel('v_S apparent (km/s)', fontsize=12)
    axes[1].set_title('S-wave: CRUST1.0 vs Apparent velocity', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_crustal_velocities_vs_distance(
    df_meta_stations: pd.DataFrame, 
    figsize: Tuple[int, int] = (16, 6), 
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot crustal velocities (v_P and v_S) vs epicentral distance.
    
    Shows how CRUST1.0-derived velocities vary across stations.
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata with columns:
        'STATION_CODE', 'EPICENTRAL_DISTANCE_KM', 'vp_crust', 'vs_crust'
    figsize : tuple
        Figure size
    output_path : str or Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    df_sorted = df_meta_stations.sort_values('EPICENTRAL_DISTANCE_KM')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # v_P plot
    axes[0].scatter(df_sorted['EPICENTRAL_DISTANCE_KM'],
                    df_sorted['vp_crust'],
                    s=80, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    vp_mean = df_sorted['vp_crust'].mean()
    vp_std = df_sorted['vp_crust'].std()
    
    axes[0].axhline(vp_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {vp_mean:.2f} km/s')
    axes[0].axhline(vp_mean + vp_std, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    axes[0].axhline(vp_mean - vp_std, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    axes[0].set_xlabel('Epicentral Distance (km)', fontsize=12)
    axes[0].set_ylabel('v_P (km/s)', fontsize=12)
    axes[0].set_title('P-wave velocity vs Distance (CRUST1.0)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(vp_mean - 3*vp_std, vp_mean + 3*vp_std)
    
    # v_S plot
    axes[1].scatter(df_sorted['EPICENTRAL_DISTANCE_KM'],
                    df_sorted['vs_crust'],
                    s=80, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    
    vs_mean = df_sorted['vs_crust'].mean()
    vs_std = df_sorted['vs_crust'].std()
    
    axes[1].axhline(vs_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {vs_mean:.2f} km/s')
    axes[1].axhline(vs_mean + vs_std, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    axes[1].axhline(vs_mean - vs_std, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    axes[1].set_xlabel('Epicentral Distance (km)', fontsize=12)
    axes[1].set_ylabel('v_S (km/s)', fontsize=12)
    axes[1].set_title('S-wave velocity vs Distance (CRUST1.0)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(vs_mean - 3*vs_std, vs_mean + 3*vs_std)
    
    # Add statistics text
    stats_text = (f"v_P: {vp_mean:.2f} ± {vp_std:.2f} km/s (range: {df_sorted['vp_crust'].min():.2f}-{df_sorted['vp_crust'].max():.2f})\n"
                  f"v_S: {vs_mean:.2f} ± {vs_std:.2f} km/s (range: {df_sorted['vs_crust'].min():.2f}-{df_sorted['vs_crust'].max():.2f})\n"
                  f"v_P/v_S: {vp_mean/vs_mean:.2f}")
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_theoretical_arrivals(
    df_stations: pd.DataFrame, 
    figsize: Tuple[int, int] = (12, 6), 
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot theoretical P and S arrival times vs epicentral distance.
    
    Creates scatter plot showing the linear relationship between
    distance and arrival time. Slope reflects crustal velocities.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with columns:
        - EPICENTRAL_DISTANCE_KM
        - t_p_theo_seconds
        - t_s_theo_seconds
        - vp_crust
        - vs_crust
    figsize : tuple, optional
        Figure size (width, height) in inches
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    
    Examples
    --------
    >>> fig, ax = plot_theoretical_arrivals(df_meta_stations)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    distance = df_stations['EPICENTRAL_DISTANCE_KM']
    t_p = df_stations['t_p_theo_seconds']
    t_s = df_stations['t_s_theo_seconds']
    
    # Plot P-wave
    ax.scatter(distance, t_p, 
               s=80, alpha=0.6, 
               color='#2E86AB', 
               edgecolors='black', 
               linewidth=0.5,
               label='P-wave', 
               zorder=3)
    
    # Plot S-wave
    ax.scatter(distance, t_s, 
               s=80, alpha=0.6, 
               color='#A23B72', 
               edgecolors='black', 
               linewidth=0.5,
               label='S-wave', 
               zorder=3)
    
    # Add theoretical lines (origin through data)
    distance_range = np.array([0, distance.max() * 1.05])
    
    # Median velocities for lines
    vp_median = df_stations['vp_crust'].median()
    vs_median = df_stations['vs_crust'].median()
    
    ax.plot(distance_range, distance_range / vp_median, 
            '--', color='#2E86AB', alpha=0.5, linewidth=1.5,
            label=f'v_P = {vp_median:.2f} km/s')
    
    ax.plot(distance_range, distance_range / vs_median, 
            '--', color='#A23B72', alpha=0.5, linewidth=1.5,
            label=f'v_S = {vs_median:.2f} km/s')
    
    # Formatting
    ax.set_xlabel('Epicentral Distance (km)', fontsize=12)
    ax.set_ylabel('Theoretical Arrival Time (s)', fontsize=12)
    ax.set_title('Theoretical P and S Wave Arrival Times', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Add text box with info
    textstr = (
        f'Stations: {len(df_stations)}\n'
        f'Distance: {distance.min():.1f}–{distance.max():.1f} km\n'
        f'v_P/v_S: {vp_median/vs_median:.2f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.pdf')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_onset_detection_results(
    signals_dict, 
    df_results,
    method='ar_pick',
    signal_unit='cm/s²',
    stations=None, 
    figsize_per_station=(10, 8),
    output_dir=None,
    show_windows=True
): 
    """
    Plot acceleration time series with detected and theoretical onsets.
    
    Creates one figure per station showing Z, N, E components with:
    - Black lines: acceleration signals
    - Blue solid lines: detected P onset
    - Red solid lines: detected S onset
    - Blue dashed lines: theoretical P onset
    - Red dashed lines: theoretical S onset
    - Shaded rectangles: search windows (if show_windows=True)
    
    Parameters
    ----------
    signals_dict : dict
        Nested dictionary from convert_signals_to_dict()
        Structure: {station: {component: array, 'time': array}}
    df_results : pd.DataFrame
        Results from detect_onsets_ar_windowed() with columns:
        - STATION_CODE
        - t_p_theo_seconds, t_s_theo_seconds
        - t_p_detected_seconds, t_s_detected_seconds
        - p_residual, s_residual
        - p_detection_success, s_detection_success
        - p_window_start, p_window_end, s_window_start, s_window_end
    signal_unit : str, optional
        Unit label for y-axis (default: 'cm/s²')
        Use 'cm/s' for velocity, 'cm' for displacement
    stations : list of str, optional
        Which stations to plot (default: all stations in df_results)
    figsize_per_station : tuple, optional
        Figure size (width, height) for each station (default: (10, 8))
    output_dir : str or Path, optional
        Directory to save figures. If None, figures are not saved.
    show_windows : bool, optional
        If True, show search windows as shaded rectangles (default: True)
    
    Returns
    -------
    dict
        Dictionary {station: fig} of created figures
    
    Examples
    --------
    >>> figs = plot_onset_detection_results(
    ...     signals_dict, 
    ...     df_meta_stations,
    ...     stations=['ACER', 'CLFR', 'SURF'],
    ...     show_windows=True,
    ...     output_dir='../figures/onset_detection'
    ... )
    >>> plt.show()
    """
    
    if stations is None:
        stations = df_results['STATION_CODE'].tolist()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    for station in stations:
        station_result = df_results[df_results['STATION_CODE'] == station]
        
        if len(station_result) == 0:
            print(f"Warning: No results for station {station}")
            continue
        
        station_result = station_result.iloc[0]
        
        # Get signals
        if station not in signals_dict:
            print(f"Warning: Station {station} not in signals_dict")
            continue
        
        data = signals_dict[station]
        time = data['time']
        
        # Identify components
        components = [k for k in data.keys() if k != 'time']
        
        comp_z = comp_n = comp_e = None
        for comp in components:
            if comp.endswith('Z'): comp_z = comp
            elif comp.endswith('N') or comp.endswith('2'): comp_n = comp
            elif comp.endswith('E') or comp.endswith('1'): comp_e = comp
        
        if not all([comp_z, comp_n, comp_e]):
            missing = [x for x, c in [('Z', comp_z), ('N', comp_n), ('E', comp_e)] if c is None]
            print(f"Warning: {station} missing components: {missing}")
            continue
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize_per_station, sharex=True)
        components_list = [(comp_z, 'Vertical'), (comp_n, 'North'), (comp_e, 'East')]
        
        for ax, (comp, label) in zip(axes, components_list):
            signal = data[comp]
            
            # Plot signal
            ax.plot(time, signal, 'k-', linewidth=0.5, alpha=0.7, zorder=1)
            ax.set_ylabel(f'{label}\n({signal_unit})', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # ========================================
            # METHOD-SPECIFIC: Detection success flags
            # ========================================
            if method == 'ar_pick':
                # AR-AIC has explicit success flags
                p_success = station_result.get('p_detection_success', False)
                s_success = station_result.get('s_detection_success', False)
                
                # Search windows (only AR-AIC)
                if show_windows:
                    if 'p_window_start_seconds' in station_result.index:
                        p_win_start = station_result['p_window_start_seconds']
                        p_win_end = station_result['p_window_end_seconds']
                        if not pd.isna(p_win_start) and not pd.isna(p_win_end):
                            ax.axvspan(p_win_start, p_win_end, alpha=0.15, color='blue',
                                      label='P window' if ax == axes[0] else '', zorder=0)
                    
                    if 's_window_start_seconds' in station_result.index:
                        s_win_start = station_result['s_window_start_seconds']
                        s_win_end = station_result['s_window_end_seconds']
                        if not pd.isna(s_win_start) and not pd.isna(s_win_end):
                            ax.axvspan(s_win_start, s_win_end, alpha=0.15, color='red',
                                      label='S window' if ax == axes[0] else '', zorder=0)
            
            elif method == 'phasenet':
                # PhaseNet: if station is in df_results, detection was successful
                p_success = not pd.isna(station_result.get('t_p_detected_seconds', np.nan))
                s_success = not pd.isna(station_result.get('t_s_detected_seconds', np.nan))
                # PhaseNet doesn't have search windows (full signal processing)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # ========================================
            # COMMON: Plot theoretical arrivals (dashed)
            # ========================================
            if 't_p_theo_seconds' in station_result.index and not pd.isna(station_result['t_p_theo_seconds']):
                ax.axvline(station_result['t_p_theo_seconds'], color='blue',
                          linestyle='--', linewidth=1.5, alpha=0.6,
                          label='P theo' if ax == axes[0] else '', zorder=2)
            
            if 't_s_theo_seconds' in station_result.index and not pd.isna(station_result['t_s_theo_seconds']):
                ax.axvline(station_result['t_s_theo_seconds'], color='red',
                          linestyle='--', linewidth=1.5, alpha=0.6,
                          label='S theo' if ax == axes[0] else '', zorder=2)
            
            # ========================================
            # METHOD-SPECIFIC: Detected onset column names
            # ========================================
            if method == 'ar_pick':
                t_p_col = 't_p_detected_seconds'
                t_s_col = 't_s_detected_seconds'
            elif method == 'phasenet':
                t_p_col = 't_p_detected_seconds'
                t_s_col = 't_s_detected_seconds'
            
            # Plot detected arrivals
            if p_success and t_p_col in station_result.index and not pd.isna(station_result[t_p_col]):
                ax.axvline(station_result[t_p_col], color='blue',
                          linestyle='-', linewidth=2.5,
                          label='P detected' if ax == axes[0] else '', zorder=3)
            
            if s_success and t_s_col in station_result.index and not pd.isna(station_result[t_s_col]):
                ax.axvline(station_result[t_s_col], color='red',
                          linestyle='-', linewidth=2.5,
                          label='S detected' if ax == axes[0] else '', zorder=3)
        
        # Set xlabel
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        # ========================================
        # METHOD-SPECIFIC: Title with residuals
        # ========================================
        method_name = 'AR-AIC' if method == 'ar_pick' else 'PhaseNet'
        
        if 'p_residual' in station_result.index and 's_residual' in station_result.index:
            # Both methods can have residuals if theoretical times are calculated
            p_res = station_result['p_residual']
            s_res = station_result['s_residual']
            
            if p_success and s_success:
                title = (f"Station {station} - {method_name} Onset Detection\n"
                        f"P residual: {p_res:+.2f} s  |  S residual: {s_res:+.2f} s")
            elif p_success:
                title = (f"Station {station} - {method_name} Onset Detection\n"
                        f"P residual: {p_res:+.2f} s  |  S detection FAILED")
            elif s_success:
                title = (f"Station {station} - {method_name} Onset Detection\n"
                        f"P detection FAILED  |  S residual: {s_res:+.2f} s")
            else:
                title = f"Station {station} - {method_name} Detection FAILED"
        else:
            # No residuals calculated
            if p_success and s_success:
                title = f"Station {station} - {method_name} Onset Detection"
            else:
                title = f"Station {station} - {method_name} Detection FAILED"
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Legend
        axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.tight_layout()
        
        # Save
        if output_dir is not None:
            output_path = output_dir / f'onset_detection_{station}_{method}.pdf'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        figures[station] = fig
    
    return figures

def plot_coda_onset_results(signals_dict, df_onsets_full,
                            signal_unit='cm/s²',
                            stations=None,
                            figsize_per_station=(10, 8),
                            output_dir=None):
    """
    Plot acceleration time series with S-wave and coda onsets for all three methods.
    
    Creates one figure per station showing Z, N, E components with:
    - Black lines: acceleration signals
    - Red solid line: detected S onset
    - Green solid line: coda onset (Rautian)
    - Orange solid line: coda onset (Arias)
    - Purple solid line: coda onset (Envelope)
    
    Parameters
    ----------
    signals_dict : dict
        Nested dictionary from convert_signals_to_dict()
        Structure: {station: {component: array, 'time': array}}
    df_onsets_full : pd.DataFrame
        Results from add_coda_onsets_to_dataframe() with columns:
        - STATION_CODE, COMPONENT
        - t_s_detected
        - t_coda_rautian, t_coda_arias, t_coda_envelope
        - s_duration_rautian, s_duration_arias, s_duration_envelope
    signal_unit : str, optional
        Unit label for y-axis (default: 'cm/s²')
        Use 'cm/s' for velocity, 'cm' for displacement
    stations : list of str, optional
        Which stations to plot (default: all stations in df_onsets_full)
    figsize_per_station : tuple, optional
        Figure size (width, height) for each station (default: (10, 8))
    output_dir : str or Path, optional
        Directory to save figures. If None, figures are not saved.
    
    Returns
    -------
    dict
        Dictionary {station: fig} of created figures
    
    Examples
    --------
    >>> figs = plot_coda_onset_results(signals_dict, df_onsets_full,
    ...                                stations=['ACER', 'CLFR'],
    ...                                output_dir='../figures/coda_detection')
    >>> plt.show()
    """
    
    if stations is None:
        stations = df_onsets_full['STATION_CODE'].unique().tolist()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    for station in stations:
        # Get signals
        if station not in signals_dict:
            print(f"Warning: Station {station} not in signals_dict")
            continue
        
        data = signals_dict[station]
        time = data['time']
        
        # Identify components
        components = [k for k in data.keys() if k != 'time']
        
        comp_z = None
        comp_n = None
        comp_e = None
        
        for comp in components:
            if comp.endswith('Z'):
                comp_z = comp
            elif comp.endswith('N') or comp.endswith('2'):
                comp_n = comp
            elif comp.endswith('E') or comp.endswith('1'):
                comp_e = comp
        
        if comp_z is None or comp_n is None or comp_e is None:
            print(f"Warning: Incomplete components for {station}")
            continue
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize_per_station, sharex=True)
        
        components_list = [(comp_z, 'Vertical'), (comp_n, 'North'), (comp_e, 'East')]
        
        for ax, (comp, label) in zip(axes, components_list):
            # Get results for this component
            comp_result = df_onsets_full[
                (df_onsets_full['STATION_CODE'] == station) & 
                (df_onsets_full['COMPONENT'] == comp)
            ]
            
            if len(comp_result) == 0:
                print(f"Warning: No results for {station}-{comp}")
                continue
            
            comp_result = comp_result.iloc[0]
            
            # Get signal
            signal = data[comp]
            
            # Plot acceleration
            ax.plot(time, signal, 'k-', linewidth=0.5, alpha=0.7, zorder=1)
            ax.set_ylabel(f'{label}\n({signal_unit})', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Plot S-wave onset
            if not np.isnan(comp_result['t_s_detected_seconds']):
                ax.axvline(comp_result['t_s_detected_seconds'], color='red',
                          linestyle='-', linewidth=2.5,
                          label='S onset' if ax == axes[0] else '', zorder=2)
            
            # Plot coda onsets (3 methods)
            if not np.isnan(comp_result['t_coda_rautian_seconds']):
                ax.axvline(comp_result['t_coda_rautian_seconds'], color='green',
                          linestyle='-', linewidth=2,
                          label='Coda (Rautian)' if ax == axes[0] else '', zorder=3)
            
            if not np.isnan(comp_result['t_coda_arias_seconds']):
                ax.axvline(comp_result['t_coda_arias_seconds'], color='orange',
                          linestyle='-', linewidth=2,
                          label='Coda (Arias)' if ax == axes[0] else '', zorder=3)
            
            if not np.isnan(comp_result['t_coda_envelope_seconds']):
                ax.axvline(comp_result['t_coda_envelope_seconds'], color='purple',
                          linestyle='-', linewidth=2,
                          label='Coda (Envelope)' if ax == axes[0] else '', zorder=3)
        
        # Set xlabel only on bottom plot
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        # Title with S-wave duration info (average across methods and components)
        durations_rautian = []
        durations_arias = []
        durations_envelope = []
        
        for comp in [comp_z, comp_n, comp_e]:
            comp_result = df_onsets_full[
                (df_onsets_full['STATION_CODE'] == station) & 
                (df_onsets_full['COMPONENT'] == comp)
            ].iloc[0]
            
            if not np.isnan(comp_result['s_duration_rautian_seconds']):
                durations_rautian.append(comp_result['s_duration_rautian_seconds'])
            if not np.isnan(comp_result['s_duration_arias_seconds']):
                durations_arias.append(comp_result['s_duration_arias_seconds'])
            if not np.isnan(comp_result['s_duration_envelope_seconds']):
                durations_envelope.append(comp_result['s_duration_envelope_seconds'])
        
        # Calculate mean durations
        mean_rautian = np.mean(durations_rautian) if durations_rautian else np.nan
        mean_arias = np.mean(durations_arias) if durations_arias else np.nan
        mean_envelope = np.mean(durations_envelope) if durations_envelope else np.nan
        
        title = f"Station {station} - Coda Onset Detection\n"
        
        if not np.isnan(mean_rautian):
            title += f"S-wave duration: Rautian={mean_rautian:.2f}s"
        if not np.isnan(mean_arias):
            title += f"  |  Arias={mean_arias:.2f}s"
        if not np.isnan(mean_envelope):
            title += f"  |  Envelope={mean_envelope:.2f}s"
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Legend on top plot
        axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save if requested
        if output_dir is not None:
            output_path = output_dir / f'coda_detection_{station}.pdf'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        figures[station] = fig
    
    return figures

def plot_coda_scatter_comparison(
    stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create scatter plots comparing coda detection methods.
    
    Generates 1x3 subplot figure with:
    - Rautian vs Arias
    - Rautian vs Envelope
    - Arias vs Envelope
    
    Each subplot shows correlation, RMSE, MAE, and linear fit.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    
    Examples
    --------
    >>> stats = compute_coda_method_statistics(df_onsets_full)
    >>> fig = plot_coda_scatter_comparison(stats)
    >>> plt.show()
    """
    
    # Method pairs and labels
    pairs = [
        ('rautian', 'arias', 'Rautian vs Arias'),
        ('rautian', 'envelope', 'Rautian vs Envelope'),
        ('arias', 'envelope', 'Arias vs Envelope')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (method1, method2, title) in zip(axes, pairs):
        pair_name = f'{method1}_{method2}'
        pair_stats = stats['pairwise'][pair_name]
        
        x = stats['data'][method1]
        y = stats['data'][method2]
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # Perfect agreement line (y = x)
        lim_min = min(x.min(), y.min())
        lim_max = max(x.max(), y.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                'k--', alpha=0.3, linewidth=1, label='y = x (perfect agreement)')
        
        # Linear fit
        x_fit = np.linspace(lim_min, lim_max, 100)
        y_fit = pair_stats['slope'] * x_fit + pair_stats['intercept']
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Linear fit (slope={pair_stats["slope"]:.3f})')
        
        # Statistics box
        textstr = '\n'.join([
            f'$r = {pair_stats["correlation"]:.3f}$ ($p < {pair_stats["p_value"]:.3f}$)',
            f'RMSE = {pair_stats["rmse"]:.2f} s',
            f'MAE = {pair_stats["mae"]:.2f} s',
            f'$n = {pair_stats["n"]}$'
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Formatting
        ax.set_xlabel(f'{method1.capitalize()} $t_{{coda}}$ (s)', fontsize=11)
        ax.set_ylabel(f'{method2.capitalize()} $t_{{coda}}$ (s)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    
    return fig


def plot_bland_altman_comparison(
    stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create Bland-Altman plots for method comparison.
    
    Generates 1x3 subplot figure with Bland-Altman plots for:
    - Rautian vs Arias
    - Rautian vs Envelope
    - Arias vs Envelope
    
    Each subplot shows mean difference (bias), limits of agreement (±1.96 SD),
    and identifies potential outliers.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    
    Examples
    --------
    >>> stats = compute_coda_method_statistics(df_onsets_full)
    >>> fig = plot_bland_altman_comparison(stats)
    >>> plt.show()
    """
    
    pairs = [
        ('rautian', 'arias', 'Rautian - Arias'),
        ('rautian', 'envelope', 'Rautian - Envelope'),
        ('arias', 'envelope', 'Arias - Envelope')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (method1, method2, title) in zip(axes, pairs):
        pair_name = f'{method1}_{method2}'
        pair_stats = stats['pairwise'][pair_name]
        
        mean_vals = pair_stats['mean']
        diff_vals = pair_stats['diff']
        
        # Scatter plot
        ax.scatter(mean_vals, diff_vals, alpha=0.6, s=50, 
                  edgecolors='k', linewidth=0.5)
        
        # Mean difference line
        ax.axhline(pair_stats['mean_diff'], color='red', linewidth=2,
                  label=f'Mean diff: {pair_stats["mean_diff"]:.2f} s')
        
        # Limits of agreement
        loa_lower, loa_upper = pair_stats['limits_of_agreement']
        ax.axhline(loa_lower, color='gray', linestyle='--', linewidth=1.5,
                  label=f'LoA: [{loa_lower:.2f}, {loa_upper:.2f}] s')
        ax.axhline(loa_upper, color='gray', linestyle='--', linewidth=1.5)
        
        # Zero line
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.3)
        
        # Identify outliers (beyond LoA)
        outliers = (diff_vals < loa_lower) | (diff_vals > loa_upper)
        if outliers.any():
            ax.scatter(mean_vals[outliers], diff_vals[outliers],
                      color='red', s=100, marker='o', facecolors='none',
                      linewidth=2, label=f'Outliers: {outliers.sum()}')
        
        # Statistics box
        textstr = '\n'.join([
            f'Mean diff: {pair_stats["mean_diff"]:.2f} s',
            f'SD: {pair_stats["std_diff"]:.2f} s',
            f'LoA: ±{1.96 * pair_stats["std_diff"]:.2f} s',
            f'$n = {pair_stats["n"]}$'
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Formatting
        ax.set_xlabel(f'Mean of {method1.capitalize()} and {method2.capitalize()} (s)', 
                     fontsize=10)
        ax.set_ylabel(f'Difference ({method1.capitalize()} - {method2.capitalize()}) (s)', 
                     fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    
    return fig


def plot_residuals_vs_distance(
    stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot method differences vs epicentral distance.
    
    Shows how agreement between methods varies with distance, revealing
    whether bias increases for distant stations (SNR degradation effect).
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    
    Examples
    --------
    >>> stats = compute_coda_method_statistics(df_onsets_full)
    >>> fig = plot_residuals_vs_distance(stats)
    >>> plt.show()
    """
    
    pairs = [
        ('rautian', 'arias', 'Rautian - Arias'),
        ('rautian', 'envelope', 'Rautian - Envelope'),
        ('arias', 'envelope', 'Arias - Envelope')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    distance = stats['data']['distance']
    
    for ax, (method1, method2, title) in zip(axes, pairs):
        pair_name = f'{method1}_{method2}'
        pair_stats = stats['pairwise'][pair_name]
        
        diff_vals = pair_stats['diff']
        
        # Scatter plot
        ax.scatter(distance, diff_vals, alpha=0.6, s=50,
                  edgecolors='k', linewidth=0.5)
        
        # Zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, 
                  label='Zero difference')
        
        # Mean difference line
        ax.axhline(pair_stats['mean_diff'], color='red', linestyle='-',
                  linewidth=1.5, label=f'Mean: {pair_stats["mean_diff"]:.2f} s')
        
        # Polynomial fit (trend)
        if len(distance) > 5:
            z = np.polyfit(distance, diff_vals, deg=1)
            p = np.poly1d(z)
            x_fit = np.linspace(distance.min(), distance.max(), 100)
            ax.plot(x_fit, p(x_fit), 'g--', linewidth=2, alpha=0.7,
                   label=f'Linear trend (slope={z[0]:.3f})')
        
        # Distance bins overlay
        for bin_stat in stats['by_distance'][pair_name]:
            bin_min, bin_max = bin_stat['bin']
            if bin_stat['n'] > 0:
                bin_center = (bin_min + bin_max) / 2
                ax.errorbar(bin_center, bin_stat['mean_diff'],
                           yerr=bin_stat['std_diff'],
                           fmt='rs', markersize=8, capsize=5, capthick=2,
                           linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Epicentral Distance (km)', fontsize=11)
        ax.set_ylabel(f'Difference ({method1.capitalize()} - {method2.capitalize()}) (s)', 
                     fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
    
    return fig


def plot_pairwise_difference_histograms(
    stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot histograms of pairwise differences between methods.
    
    Shows distribution of differences for each method pair, revealing
    whether bias is symmetric and normally distributed.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    
    Examples
    --------
    >>> stats = compute_coda_method_statistics(df_onsets_full)
    >>> fig = plot_pairwise_difference_histograms(stats)
    >>> plt.show()
    """
    
    pairs = [
        ('rautian', 'arias', 'Rautian - Arias'),
        ('rautian', 'envelope', 'Rautian - Envelope'),
        ('arias', 'envelope', 'Arias - Envelope')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (method1, method2, title) in zip(axes, pairs):
        pair_name = f'{method1}_{method2}'
        pair_stats = stats['pairwise'][pair_name]
        
        diff_vals = pair_stats['diff']
        
        # Histogram
        n, bins, patches = ax.hist(diff_vals, bins=15, alpha=0.7, 
                                    edgecolor='black', linewidth=1.2,
                                    color='steelblue', density=True)
        
        # Normal distribution overlay
        mu = pair_stats['mean_diff']
        sigma = pair_stats['std_diff']
        x = np.linspace(diff_vals.min(), diff_vals.max(), 100)
        ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma), 
               'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        # Mean line
        ax.axvline(mu, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mu:.2f} s')
        
        # Zero line
        ax.axvline(0, color='black', linestyle=':', linewidth=1.5,
                  label='Zero difference')
        
        # Statistics box
        textstr = '\n'.join([
            f'Mean: {mu:.2f} s',
            f'SD: {sigma:.2f} s',
            f'Median: {np.median(diff_vals):.2f} s',
            f'Range: [{diff_vals.min():.1f}, {diff_vals.max():.1f}] s'
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Formatting
        ax.set_xlabel(f'Difference ({method1.capitalize()} - {method2.capitalize()}) (s)', 
                     fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
    
    return fig


def plot_correlation_matrix_heatmap(
    stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap for all methods.
    
    Displays pairwise correlations between all three coda detection methods
    in a visually intuitive heatmap format.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    
    Examples
    --------
    >>> stats = compute_coda_method_statistics(df_onsets_full)
    >>> fig = plot_correlation_matrix_heatmap(stats)
    >>> plt.show()
    """
    
    methods = ['Rautian', 'Arias', 'Envelope']
    n_methods = len(methods)
    
    # Build correlation matrix
    corr_matrix = np.ones((n_methods, n_methods))
    
    # Fill upper triangle
    corr_matrix[0, 1] = stats['pairwise']['rautian_arias']['correlation']
    corr_matrix[0, 2] = stats['pairwise']['rautian_envelope']['correlation']
    corr_matrix[1, 2] = stats['pairwise']['arias_envelope']['correlation']
    
    # Mirror to lower triangle
    corr_matrix[1, 0] = corr_matrix[0, 1]
    corr_matrix[2, 0] = corr_matrix[0, 2]
    corr_matrix[2, 1] = corr_matrix[1, 2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0,
                   aspect='auto', interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=11)
    
    # Ticks
    ax.set_xticks(np.arange(n_methods))
    ax.set_yticks(np.arange(n_methods))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_yticklabels(methods, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Annotate cells with correlation values
    for i in range(n_methods):
        for j in range(n_methods):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', 
                          fontsize=13, fontweight='bold')
    
    # Title
    ax.set_title('Coda Onset Method Correlation Matrix', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add grid
    ax.set_xticks(np.arange(n_methods) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_methods) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def get_station_components(
    station: str,
    signals_dict: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, str]:
    """
    Get available components for a station with automatic detection.
    
    Returns mapping of standardized orientations to actual component codes.
    
    Parameters
    ----------
    station : str
        Station code
    signals_dict : dict
        Signals dictionary
        
    Returns
    -------
    component_map : dict
        Mapping: {'Z': 'HGZ', 'N': 'HGN', 'E': 'HGE'} or similar
        Keys are always 'Z', 'N' or '1', 'E' or '2'
        
    Examples
    --------
    >>> comp_map = get_station_components('BRZ', signals_dict)
    >>> comp_map
    {'Z': 'HGZ', 'N': 'HGN', 'E': 'HGE'}
    
    >>> comp_map = get_station_components('CAGN', signals_dict)
    >>> comp_map
    {'Z': 'HNZ', '1': 'HN1', '2': 'HN2'}
    """
    
    if station not in signals_dict:
        return {}
    
    # Get all components (exclude 'time')
    available = [k for k in signals_dict[station].keys() if k != 'time']
    
    if len(available) == 0:
        return {}
    
    component_map = {}
    
    # Detect vertical component
    for comp in available:
        if comp.endswith('Z'):
            component_map['Z'] = comp
            break
    
    # Detect horizontal components (N-E vs 1-2 system)
    has_ne = any(c.endswith('N') or c.endswith('E') for c in available)
    has_12 = any(c.endswith('1') or c.endswith('2') for c in available)
    
    if has_ne:
        # N-E system
        for comp in available:
            if comp.endswith('N'):
                component_map['N'] = comp
            elif comp.endswith('E'):
                component_map['E'] = comp
    elif has_12:
        # 1-2 system
        for comp in available:
            if comp.endswith('1'):
                component_map['1'] = comp
            elif comp.endswith('2'):
                component_map['2'] = comp
    
    return component_map


def plot_station_windows(
    station: str,
    signals_dict: Dict[str, Dict[str, np.ndarray]],
    windowed_signals: Dict[str, Dict[str, Dict[str, Dict]]],
    df_onsets: Optional[pd.DataFrame] = None,
    signal_unit: str = 'cm/s²',
    signal_type: Optional[str] = None,
    coda_method: str = 'rautian',
    output_path: Optional[Union[str, Path]] = None,
    show_onset_lines: bool = True,
    show_window_backgrounds: bool = True,
    title_suffix: str = '',
    mode: str = 'interactive',
) -> plt.Figure:
    """
    Plot three-component signal with onset times and window boundaries.

    Only plots components that have at least one window in windowed_signals.
    The figure title shows the station name and epicentral distance (if available).

    Parameters
    ----------
    station : str
        Station code (e.g., 'BRZ')
    signals_dict : dict
        Full signals dictionary from convert_signals_to_dict()
    windowed_signals : dict
        Windowed signals from segment_all_signals()
    df_onsets : pd.DataFrame, optional
        DataFrame with onset times; used to retrieve EPICENTRAL_DISTANCE_KM
    signal_unit : str, optional
        Y-axis unit label (default: 'cm/s²')
    signal_type : str, optional
        Type of signal (default: 'acceleration')
    coda_method : str, optional
        Coda method name shown in legend (default: 'rautian')
    output_path : str or Path, optional
        If provided, save figure to this path
    show_onset_lines : bool, optional
        Show vertical lines at onset times (default: True)
    show_window_backgrounds : bool, optional
        Show colored backgrounds for each window (default: True)
    title_suffix : str, optional
        Extra text appended to the figure title (default: '')
    mode : str, optional
        Output mode. One of:
        - 'interactive': screen display, PDF output, 150 dpi (default)
        - 'paper': publication quality, PNG output, 600 dpi, 17.5cm width
        - 'poster': conference poster, PNG output, 300 dpi, 15cm width
        - 'thesis': thesis figure, PDF output, 5.5 inch width

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If station not found in signals_dict or windowed_signals, if no
        component has any window, or if mode is not one of the accepted values.
    """
    _VALID_MODES = ('interactive', 'paper', 'poster', 'thesis')
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}"
        )

    if station not in signals_dict:
        raise ValueError(f"Station '{station}' not found in signals_dict")
    if station not in windowed_signals:
        raise ValueError(f"Station '{station}' not found in windowed_signals")

    # ── Mode-dependent settings ───────────────────────────────────────────────
    _mode_settings = {
        'interactive': dict(
            figsize          = (14, 10),
            dpi_save         = 150,
            font_title       = 13,
            font_axis_label  = 12,
            font_tick        = 10,
            font_legend      = 9,
            linewidth_signal = 0.6,
            linewidth_onset  = 2.0,
            window_alpha     = 0.3,
            external_legend  = False,
            show_durations   = True,
            output_suffix    = '.pdf',
        ),
        'paper': dict(
            figsize          = (6.89, 5.0),   # 17.5cm × 12.7cm
            dpi_save         = 600,
            font_title       = 11,
            font_axis_label  = 9,
            font_tick        = 8,
            font_legend      = 8,
            linewidth_signal = 0.5,
            linewidth_onset  = 1.2,
            window_alpha     = 0.25,
            external_legend  = True,
            show_durations   = False,
            output_suffix    = '.png',
        ),
        'poster': dict(
            figsize          = (5.91, 4.72),  # 15cm × 12cm
            dpi_save         = 300,
            font_title       = 14,
            font_axis_label  = 12,
            font_tick        = 10,
            font_legend      = 10,
            linewidth_signal = 0.6,
            linewidth_onset  = 1.5,
            window_alpha     = 0.25,
            external_legend  = True,
            show_durations   = False,
            output_suffix    = '.png',
        ),
        'thesis': dict(
        figsize          = (5.5, 6.0),
        dpi_save         = 300,
        font_title       = 11,
        font_axis_label  = 9,
        font_tick        = 8,
        font_legend      = 8,
        linewidth_signal = 0.5,
        linewidth_onset  = 1.2,
        window_alpha     = 0.25,
        external_legend  = True,
        show_durations   = False,
        output_suffix    = '.pdf',
    ),
    }
    _gridspec_settings = {
        'paper':  dict(hspace=0.05, top=0.82, bottom=0.10, left=0.20, right=0.97),
        'poster': dict(hspace=0.05, top=0.82, bottom=0.10, left=0.20, right=0.97),
        'thesis': dict(hspace=0.05, top=0.82, bottom=0.10, left=0.20, right=0.97),
    }

    cfg = _mode_settings[mode]

    # ── Component ordering ────────────────────────────────────────────────────
    comp_map = get_station_components(station, signals_dict)

    if 'Z' in comp_map and 'N' in comp_map and 'E' in comp_map:
        canonical_order = [('Z', 'Vertical'), ('N', 'North'), ('E', 'East')]
    elif 'Z' in comp_map and '1' in comp_map and '2' in comp_map:
        canonical_order = [('Z', 'Vertical'), ('1', 'Horizontal-1'), ('2', 'Horizontal-2')]
    else:
        canonical_order = [(k, f'Component {v}') for k, v in comp_map.items()]

    plot_order = [
        (key, label)
        for key, label in canonical_order
        if key in comp_map
        and comp_map[key] in windowed_signals[station]
        and len(windowed_signals[station][comp_map[key]]) > 0
    ]

    if not plot_order:
        raise ValueError(
            f"Station '{station}': no component has any window in windowed_signals"
        )

    # ── Figure title ──────────────────────────────────────────────────────────
    title_parts = [f"Station {station}"]
    if df_onsets is not None and 'EPICENTRAL_DISTANCE_KM' in df_onsets.columns:
        mask = df_onsets['STATION_CODE'] == station
        if mask.any():
            dist = df_onsets.loc[mask, 'EPICENTRAL_DISTANCE_KM'].iloc[0]
            if not pd.isna(dist):
                title_parts.append(f"$d_{{\\mathrm{{epi}}}}$ = {dist:.1f} km")
    meta_parts = []
    if title_suffix:
        meta_parts.append(title_suffix)
    if signal_type is not None:
        meta_parts.append(signal_type)
    if meta_parts:
        title_parts.append(', '.join(meta_parts))
    suptitle = ' — '.join(title_parts)

    # ── Colors ────────────────────────────────────────────────────────────────
    if mode == 'interactive':
        window_colors = {
            'pre_event':  '#E8E8E8',
            'p_wave':     '#AED6F1',
            's_wave':     '#F9E79F',
            'coda':       '#D5F4E6',
            'post_event': '#F8C8DC',
        }
    else:
        window_colors = {
            'pre_event':  '#D6D6D6',
            'p_wave':     '#B8D8E8',
            's_wave':     '#F5DFA0',
            'coda':       '#A8D4C8',
            'post_event': '#E8C8A0',
        }

    onset_colors = {
        'p':        '#C0392B',
        's':        '#00807F',
        'coda':     '#C8861D',
        'coda_end': '#729EC1',
    } if mode != 'interactive' else {
        'p':        '#C0392B',
        's':        '#2471A3',
        'coda':     '#1E8449',
        'coda_end': '#7D3C98',
    }

    # ── Create figure ─────────────────────────────────────────────────────────
    n_subplots = len(plot_order)

    if cfg['external_legend']:
        fig = plt.figure(figsize=cfg['figsize'])
        gs_cfg = _gridspec_settings.get(mode, dict(
            hspace=0.05, top=0.82, bottom=0.10, left=0.20, right=0.97
        ))
        gs = fig.add_gridspec(n_subplots, 1, **gs_cfg)
        axes = [fig.add_subplot(gs[i]) for i in range(n_subplots)]
        for i in range(n_subplots - 1):
            axes[i].sharex(axes[-1])
    else:
        fig, axes = plt.subplots(n_subplots, 1, figsize=cfg['figsize'], sharex=True)
        if n_subplots == 1:
            axes = [axes]

    time_full = signals_dict[station]['time']

    # ── Legend handles (collected on first component pass) ────────────────────
    legend_elements = []
    legend_built = False

    for idx, (comp_key, comp_label) in enumerate(plot_order):
        ax = axes[idx]
        component = comp_map[comp_key]
        signal_full = signals_dict[station][component]
        windows = windowed_signals[station][component]

        has_pre_event  = 'pre_event'  in windows
        has_p_wave     = 'p_wave'     in windows
        has_s_wave     = 's_wave'     in windows
        has_coda       = 'coda'       in windows
        has_post_event = 'post_event' in windows

        t_p        = windows['p_wave']['start_seconds']  if has_p_wave     else None
        t_s        = windows['s_wave']['start_seconds']  if has_s_wave     else None
        t_coda     = windows['coda']['start_seconds']    if has_coda       else None
        t_coda_end = windows['coda']['end_seconds']      if has_post_event else None

        # ── Window backgrounds ────────────────────────────────────────────────
        if show_window_backgrounds:
            for window_name in ('pre_event', 'p_wave', 's_wave', 'coda', 'post_event'):
                if window_name in windows:
                    w = windows[window_name]
                    ax.axvspan(
                        w['start_seconds'], w['end_seconds'],
                        color=window_colors[window_name],
                        alpha=cfg['window_alpha'],
                        zorder=0,
                    )

        # ── Signal ────────────────────────────────────────────────────────────
        ax.plot(
            time_full, signal_full, 'k-',
            linewidth=cfg['linewidth_signal'], zorder=2,
        )

        # ── Onset lines ───────────────────────────────────────────────────────
        if show_onset_lines:
            if t_p is not None:
                ax.axvline(t_p, color=onset_colors['p'],
                           linewidth=cfg['linewidth_onset'],
                           linestyle='-', zorder=3)
            if t_s is not None:
                ax.axvline(t_s, color=onset_colors['s'],
                           linewidth=cfg['linewidth_onset'],
                           linestyle='-', zorder=3)
            if t_coda is not None:
                ax.axvline(t_coda, color=onset_colors['coda'],
                           linewidth=cfg['linewidth_onset'],
                           linestyle='-', zorder=3)
            if t_coda_end is not None:
                ax.axvline(t_coda_end, color=onset_colors['coda_end'],
                           linewidth=cfg['linewidth_onset'],
                           linestyle=':', zorder=3, alpha=0.8)

        # ── Axis formatting ───────────────────────────────────────────────────
        ax.set_ylabel(
            f'{comp_label} {component}\n({signal_unit})',
            fontsize=cfg['font_axis_label'],
            rotation=0,
            ha='right',
            va='center',
            labelpad=40,
        )
        if idx < n_subplots - 1:
            ax.tick_params(axis='both', labelsize=cfg['font_tick'], labelbottom=False)
        else:
            ax.tick_params(axis='both', labelsize=cfg['font_tick'])
        ax.grid(True, alpha=0.3, zorder=1)

        # ── Duration annotations (interactive mode only) ──────────────────────
        if cfg['show_durations']:
            duration_parts = []
            for window_name, short_label in [
                ('pre_event', 'Pre'), ('p_wave', 'P'), ('s_wave', 'S'),
                ('coda', 'Coda'), ('post_event', 'Post'),
            ]:
                if window_name in windows:
                    dur = windows[window_name]['duration_seconds']
                    duration_parts.append(f"{short_label}: {dur:.2f}s")
            if duration_parts:
                ax.set_title(
                    '  |  '.join(duration_parts),
                    fontsize=9, loc='left', pad=3,
                )

        # ── Build legend elements once ────────────────────────────────────────
        if not legend_built:
            if show_onset_lines:
                if has_p_wave:
                    legend_elements.append(
                        plt.Line2D([0], [0], color=onset_colors['p'],
                                   linewidth=cfg['linewidth_onset'],
                                   label='P onset')
                    )
                if has_s_wave:
                    legend_elements.append(
                        plt.Line2D([0], [0], color=onset_colors['s'],
                                   linewidth=cfg['linewidth_onset'],
                                   label='S onset')
                    )
                if has_coda:
                    legend_elements.append(
                        plt.Line2D([0], [0], color=onset_colors['coda'],
                                   linewidth=cfg['linewidth_onset'],
                                   label=f'Coda onset ({coda_method})')
                    )
                if has_post_event:
                    legend_elements.append(
                        plt.Line2D([0], [0], color=onset_colors['coda_end'],
                                   linewidth=cfg['linewidth_onset'],
                                   linestyle=':', alpha=0.8,
                                   label='Coda end')
                    )
            if show_window_backgrounds:
                for window_name, label in [
                    ('pre_event', 'Pre-event'), ('p_wave', 'P-wave'),
                    ('s_wave', 'S-wave'), ('coda', 'Coda'),
                    ('post_event', 'Post-event'),
                ]:
                    if window_name in windows:
                        legend_elements.append(
                            mpatches.Patch(
                                color=window_colors[window_name],
                                alpha=max(cfg['window_alpha'], 0.5),
                                label=label,
                            )
                        )
            legend_built = True

        # ── Inline legend (interactive mode only, first subplot) ──────────────
        if not cfg['external_legend'] and idx == 0 and legend_elements:
            ax.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=cfg['font_legend'],
                framealpha=0.9,
            )

    # ── X-axis label ─────────────────────────────────────────────────────────
    axes[-1].set_xlabel('Time (s)', fontsize=cfg['font_axis_label'])
    axes[-1].tick_params(axis='x', labelsize=cfg['font_tick'])

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(suptitle, fontsize=cfg['font_title'], fontweight='bold', y=0.97)

    # ── External legend (paper and poster modes) ──────────────────────────────
    if cfg['external_legend'] and legend_elements:
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.90),
            ncol=len(legend_elements),
            fontsize=cfg['font_legend'],
            framealpha=0.9,
            handlelength=1.5,
            columnspacing=1.0,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    if not cfg['external_legend']:
        plt.tight_layout(rect=[0, 0, 1, 0.98])

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path).with_suffix(cfg['output_suffix'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig


def plot_multiple_stations(
    stations: List[str],
    signals_dict: Dict,
    windowed_signals: Dict,
    df_onsets: Optional[pd.DataFrame] = None,
    signal_unit: str = 'cm/s²',
    signal_type: Optional[str] = None,
    title_suffix: str = 'AR-AIC',
    coda_method: str = 'rautian',
    output_dir: Optional[Union[str, Path]] = None,
    close_after_save: bool = True,
    mode: str = 'interactive',
    **kwargs
) -> Dict[str, plt.Figure]:
    """
    Plot window segmentation for multiple stations.
    
    Parameters
    ----------
    stations : list of str
        Station codes to plot (e.g., ['BRZ', 'CAGN', 'SURF'])
    signals_dict : dict
        Full signals dictionary
    windowed_signals : dict
        Windowed signals dictionary
    df_onsets : pd.DataFrame, optional
        Onset times DataFrame
    signal_unit : str, optional
        Y-axis unit label (default: 'cm/s²')
    signal_type : str, optional
        Type of signal (default: None)
    title_suffix : str
        Suffix for the plot title (default: 'AR-AIC')
    coda_method : str, optional
        Coda detection method name (default: 'rautian')
    output_dir : str, optional
        Directory to save plots (e.g., 'plots/windows/')
        If None, plots are not saved automatically
    close_after_save : bool, optional
        If True, close figures after saving to free memory (default: True)
    mode : str, optional
        Output mode. One of:
        - 'interactive': screen display, PDF output, 150 dpi (default)
        - 'paper': publication quality, PNG output, 600 dpi, 17.5cm width
        - 'poster': conference poster, PNG output, 300 dpi, 15cm width
        - 'thesis': thesis figure, PDF output, 5.5 inch width
    **kwargs
        Additional arguments passed to plot_station_windows()
        
    Returns
    -------
    figures : dict
        Dictionary mapping station codes to Figure objects
        (only if close_after_save=False)
        
    Examples
    --------
    >>> # Plot and display 3 stations
    >>> figs = plot_multiple_stations(['BRZ', 'CAGN', 'SURF'],
    ...                               signals_dict, windowed_signals)
    >>> plt.show()
    
    >>> # Plot all stations and save to directory
    >>> stations = list(windowed_signals.keys())
    >>> plot_multiple_stations(stations, signals_dict, windowed_signals,
    ...                        output_dir='plots/windows/', 
    ...                        close_after_save=True)
    """
    
    figures = {}
    
    print(f"\nPlotting {len(stations)} stations...")
    stations = [s for s in stations if s in windowed_signals]
    
    for i, station in enumerate(stations, 1):
        try:
            # Determine save path if directory provided
            if output_dir:
                output_path = Path(output_dir) / f'{station}_windows'
            else:
                output_path = None
            
            # Create plot
            fig = plot_station_windows(
                station=station,
                signals_dict=signals_dict,
                windowed_signals=windowed_signals,
                df_onsets=df_onsets,
                signal_unit=signal_unit,
                signal_type=signal_type,
                title_suffix=title_suffix,
                coda_method=coda_method,
                output_path=output_path,
                mode=mode,
                **kwargs
            )
            
            # Store or close
            if close_after_save and output_path:
                plt.close(fig)
            else:
                figures[station] = fig
            
            # Progress
            if i % 10 == 0 or i == len(stations):
                print(f"  Progress: {i}/{len(stations)} stations")
                
        except Exception as e:
            print(f"  Error plotting {station}: {e}")
            continue
    
    print(f"Done! Plotted {len(figures) if not close_after_save else len(stations)} stations")
    
    return figures if not close_after_save else {}


def plot_window_comparison(
    station: str,
    component: str,
    signals_dict: Dict,
    windowed_dict_list: List[Dict],
    method_labels: List[str],
    signal_unit: str = 'cm/s²',
    figsize: tuple = (14, 6),
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Compare different windowing methods for the same signal.
    
    Useful for comparing different coda detection methods or 
    different pre-event window strategies.
    
    Parameters
    ----------
    station : str
        Station code
    component : str
        Component code (e.g., 'HGE')
    signals_dict : dict
        Full signals dictionary
    windowed_dict_list : list of dict
        List of windowed_signals dictionaries to compare
        (e.g., [windowed_rautian, windowed_arias, windowed_envelope])
    method_labels : list of str
        Labels for each method (e.g., ['Rautian', 'Arias', 'Envelope'])
    signal_unit : str, optional
    figsize : tuple, optional
        Figure size (default: (14, 6))
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        
    Examples
    --------
    >>> # Compare coda methods
    >>> fig = plot_window_comparison(
    ...     'BRZ', 'HGE',
    ...     signals_dict,
    ...     [windowed_rautian, windowed_arias, windowed_envelope],
    ...     ['Rautian', 'Arias', 'Envelope']
    ... )
    
    >>> # Compare pre-event strategies
    >>> fig = plot_window_comparison(
    ...     'BRZ', 'HGE',
    ...     signals_dict,
    ...     [windowed_5s, windowed_full],
    ...     ['Fixed 5s', 'Full window']
    ... )
    """
    
    # Colors for different methods
    method_colors = ['#27AE60', '#E67E22', '#9B59B6', '#E74C3C', '#3498DB']
    
    # Get signal
    signal = signals_dict[station][component]
    time = signals_dict[station]['time']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot signal
    ax.plot(time, signal, 'k-', linewidth=0.6, alpha=0.7, label='Signal')
    
    # Plot onset lines for each method
    for idx, (windowed, label) in enumerate(zip(windowed_dict_list, method_labels)):
        windows = windowed[station][component]
        
        color = method_colors[idx % len(method_colors)]
        
        # P onset (should be same for all)
        if idx == 0 and 'p_wave' in windows:
            t_p = windows['p_wave']['start_seconds']
            ax.axvline(t_p, color='red', linewidth=2, linestyle='-',
                    label='P onset', zorder=10)

        # S onset (should be same for all)
        if idx == 0 and 's_wave' in windows:
            t_s = windows['s_wave']['start_seconds']
            ax.axvline(t_s, color='blue', linewidth=2, linestyle='-',
                    label='S onset', zorder=10)

        # Coda onset (different for each method)
        if 'coda' in windows:
            t_coda = windows['coda']['start_seconds']
            ax.axvline(t_coda, color=color, linewidth=2.5, linestyle='--',
                    label=f'Coda ({label}): {t_coda:.1f}s', zorder=9)

        # Pre-event start (if different)
        if idx > 0 and 'pre_event' in windows:
            prev_windows = windowed_dict_list[0][station][component]
            if 'pre_event' in prev_windows:
                t_pre = windows['pre_event']['start_seconds']
                prev_t_pre = prev_windows['pre_event']['start_seconds']
                if abs(t_pre - prev_t_pre) > 0.1:
                    ax.axvline(t_pre, color=color, linewidth=1.5, linestyle=':',
                            alpha=0.6, label=f'Pre-event start ({label})')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(f'{component} Signal ({signal_unit})', fontsize=12)
    ax.set_title(f'Station {station} - {component}: Method Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.pdf')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_station_windows_multitype(
    station: str,
    signals_dicts: Dict[str, Dict],
    windowed_signals_dicts: Dict[str, Dict],
    signal_units: Dict[str, str],
    df_onsets: Optional[pd.DataFrame] = None,
    coda_method: str = 'rautian',
    title_suffix: str = '',
    output_path: Optional[Union[str, Path]] = None,
    show_onset_lines: bool = True,
    show_window_backgrounds: bool = True,
    mode: str = 'thesis',
) -> plt.Figure:
    """
    Plot three-component windowed signals for a single station across
    three signal types (acceleration, velocity, displacement) in a
    single figure with shared axes.

    Parameters
    ----------
    station : str
        Station code (e.g., 'OGDI')
    signals_dicts : dict
        Dictionary mapping signal type to signals_dict, e.g.:
        {'acceleration': signals_dict_acc, 'velocity': signals_dict_vel,
         'displacement': signals_dict_disp}
    windowed_signals_dicts : dict
        Dictionary mapping signal type to windowed_signals, e.g.:
        {'acceleration': windowed_acc, 'velocity': windowed_vel,
         'displacement': windowed_disp}
    signal_units : dict
        Dictionary mapping signal type to unit string, e.g.:
        {'acceleration': 'cm/s²', 'velocity': 'cm/s', 'displacement': 'cm'}
    df_onsets : pd.DataFrame, optional
        DataFrame with onset times; used to retrieve EPICENTRAL_DISTANCE_KM
    coda_method : str, optional
        Coda method name shown in legend (default: 'rautian')
    title_suffix : str, optional
        Extra text appended to the figure title (default: '')
    output_path : str or Path, optional
        If provided, save figure to this path
    show_onset_lines : bool, optional
        Show vertical lines at onset times (default: True)
    show_window_backgrounds : bool, optional
        Show colored backgrounds for each window (default: True)
    mode : str, optional
        Output mode: 'thesis', 'paper', 'poster', 'interactive'

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _VALID_MODES = ('interactive', 'paper', 'poster', 'thesis')
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}")

    signal_types = ['acceleration', 'velocity', 'displacement']
    for st in signal_types:
        if st not in signals_dicts:
            raise ValueError(f"Missing signals_dict for signal type '{st}'")
        if st not in windowed_signals_dicts:
            raise ValueError(f"Missing windowed_signals for signal type '{st}'")

    # ── Mode-dependent settings ───────────────────────────────────────────────
    _mode_settings = {
        'interactive': dict(
            figsize          = (18, 10),
            dpi_save         = 150,
            font_title       = 13,
            font_col_label   = 11,
            font_axis_label  = 10,
            font_tick        = 9,
            font_legend      = 9,
            linewidth_signal = 0.6,
            linewidth_onset  = 1.5,
            window_alpha     = 0.3,
            output_suffix    = '.pdf',
        ),
        'paper': dict(
            figsize          = (6.89, 5.5),
            dpi_save         = 600,
            font_title       = 10,
            font_col_label   = 9,
            font_axis_label  = 8,
            font_tick        = 7,
            font_legend      = 7,
            linewidth_signal = 0.4,
            linewidth_onset  = 1.0,
            window_alpha     = 0.25,
            output_suffix    = '.png',
        ),
        'poster': dict(
            figsize          = (14.85, 10.35),
            dpi_save         = 300,
            font_title       = 22,
            font_col_label   = 18,
            font_axis_label  = 16,
            font_tick        = 14,
            font_legend      = 13,
            linewidth_signal = 0.8,
            linewidth_onset  = 1.8,
            window_alpha     = 0.25,
            output_suffix    = '.png',
        ),
        'thesis': dict(
            figsize          = (10.0, 6.5),
            dpi_save         = 300,
            font_title       = 10,
            font_col_label   = 9,
            font_axis_label  = 8,
            font_tick        = 7,
            font_legend      = 7,
            linewidth_signal = 0.4,
            linewidth_onset  = 1.0,
            window_alpha     = 0.25,
            output_suffix    = '.pdf',
        ),
    }
    cfg = _mode_settings[mode]

    # ── Colors ────────────────────────────────────────────────────────────────
    if mode == 'interactive':
        window_colors = {
            'pre_event':  '#E8E8E8',
            'p_wave':     '#AED6F1',
            's_wave':     '#F9E79F',
            'coda':       '#D5F4E6',
            'post_event': '#F8C8DC',
        }
        onset_colors = {
            'p':        '#C0392B',
            's':        '#2471A3',
            'coda':     '#1E8449',
            'coda_end': '#7D3C98',
        }
    else:
        window_colors = {
            'pre_event':  '#D6D6D6',
            'p_wave':     '#B8D8E8',
            's_wave':     '#F5DFA0',
            'coda':       '#A8D4C8',
            'post_event': '#E8C8A0',
        }
        onset_colors = {
            'p':        '#C0392B',
            's':        '#00807F',
            'coda':     '#C8861D',
            'coda_end': '#729EC1',
        }

    # ── Component ordering ────────────────────────────────────────────────────
    comp_map = get_station_components(station, signals_dicts['acceleration'])
    if 'Z' in comp_map and 'N' in comp_map and 'E' in comp_map:
        canonical_order = [('Z', 'HNZ'), ('N', 'HNN'), ('E', 'HNE')]
    elif 'Z' in comp_map and '1' in comp_map and '2' in comp_map:
        canonical_order = [('Z', 'HNZ'), ('1', 'HN1'), ('2', 'HN2')]
    else:
        canonical_order = [(k, v) for k, v in comp_map.items()]

    n_rows = len(canonical_order)
    n_cols = len(signal_types)

    # ── Figure title ──────────────────────────────────────────────────────────
    title_parts = [f"Station {station}"]
    if df_onsets is not None and 'EPICENTRAL_DISTANCE_KM' in df_onsets.columns:
        mask = df_onsets['STATION_CODE'] == station
        if mask.any():
            dist = df_onsets.loc[mask, 'EPICENTRAL_DISTANCE_KM'].iloc[0]
            if not pd.isna(dist):
                title_parts.append(f"$d_{{\\mathrm{{epi}}}}$ = {dist:.1f} km")
    if title_suffix:
        title_parts.append(title_suffix)
    suptitle = ' — '.join(title_parts)

    # ── Create figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=cfg['figsize'])
    gs = fig.add_gridspec(
        n_rows, n_cols,
        hspace=0.05,
        wspace=0.08,
        top=0.82,
        bottom=0.10,
        left=0.12,
        right=0.97,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    # Share x axis within each column
    for c in range(n_cols):
        for r in range(1, n_rows):
            axes[r][c].sharex(axes[0][c])

    # ── Column labels ─────────────────────────────────────────────────────────
    col_labels = {
        'acceleration': 'Acceleration',
        'velocity':     'Velocity',
        'displacement': 'Displacement',
    }
    for c, st in enumerate(signal_types):
        axes[0][c].set_title(
            f"{col_labels[st]} ({signal_units[st]})",
            fontsize=cfg['font_col_label'],
            fontweight='bold',
            pad=10,
        )

    # ── Legend handles ────────────────────────────────────────────────────────
    legend_elements = []
    legend_built = False

    # ── Plot ──────────────────────────────────────────────────────────────────
    for c, st in enumerate(signal_types):
        signals_dict    = signals_dicts[st]
        windowed_signals = windowed_signals_dicts[st]
        signal_unit     = signal_units[st]

        if station not in signals_dict or station not in windowed_signals:
            continue

        time_full = signals_dict[station]['time']

        for r, (comp_key, comp_code) in enumerate(canonical_order):
            ax = axes[r][c]

            if comp_key not in comp_map:
                ax.axis('off')
                continue

            component = comp_map[comp_key]

            if (station not in windowed_signals or
                    component not in windowed_signals[station] or
                    len(windowed_signals[station][component]) == 0):
                ax.axis('off')
                continue

            signal_full = signals_dict[station][component]
            windows     = windowed_signals[station][component]

            has_pre_event  = 'pre_event'  in windows
            has_p_wave     = 'p_wave'     in windows
            has_s_wave     = 's_wave'     in windows
            has_coda       = 'coda'       in windows
            has_post_event = 'post_event' in windows

            t_p        = windows['p_wave']['start_seconds']  if has_p_wave     else None
            t_s        = windows['s_wave']['start_seconds']  if has_s_wave     else None
            t_coda     = windows['coda']['start_seconds']    if has_coda       else None
            t_coda_end = windows['coda']['end_seconds']      if has_post_event else None

            # Window backgrounds
            if show_window_backgrounds:
                for window_name in ('pre_event', 'p_wave', 's_wave', 'coda', 'post_event'):
                    if window_name in windows:
                        w = windows[window_name]
                        ax.axvspan(
                            w['start_seconds'], w['end_seconds'],
                            color=window_colors[window_name],
                            alpha=cfg['window_alpha'],
                            zorder=0,
                        )

            # Signal
            ax.plot(
                time_full, signal_full, 'k-',
                linewidth=cfg['linewidth_signal'], zorder=2,
            )

            # Onset lines
            if show_onset_lines:
                if t_p is not None:
                    ax.axvline(t_p, color=onset_colors['p'],
                               linewidth=cfg['linewidth_onset'],
                               linestyle='-', zorder=3)
                if t_s is not None:
                    ax.axvline(t_s, color=onset_colors['s'],
                               linewidth=cfg['linewidth_onset'],
                               linestyle='-', zorder=3)
                if t_coda is not None:
                    ax.axvline(t_coda, color=onset_colors['coda'],
                               linewidth=cfg['linewidth_onset'],
                               linestyle='-', zorder=3)
                if t_coda_end is not None:
                    ax.axvline(t_coda_end, color=onset_colors['coda_end'],
                               linewidth=cfg['linewidth_onset'],
                               linestyle=':', zorder=3, alpha=0.8)

            # Y-axis label (only on leftmost column)
            if c == 0:
                ax.set_ylabel(
                    f'{comp_code}',
                    fontsize=cfg['font_axis_label'],
                    rotation=0,
                    ha='right',
                    va='center',
                    labelpad=25,
                )
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', left=False, labelleft=False)

            ax.tick_params(axis='y', labelsize=cfg['font_tick'])
            ax.grid(True, alpha=0.3, zorder=1)
            # X-axis ticks (only on bottom row)
            if r < n_rows - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.tick_params(axis='x', labelsize=cfg['font_tick'])
                ax.set_xlabel('Time (s)', fontsize=cfg['font_axis_label'])

            # Build legend once
            if not legend_built and c == 0 and r == 0:
                if show_onset_lines:
                    if has_p_wave:
                        legend_elements.append(plt.Line2D(
                            [0], [0], color=onset_colors['p'],
                            linewidth=cfg['linewidth_onset'], label='P onset'))
                    if has_s_wave:
                        legend_elements.append(plt.Line2D(
                            [0], [0], color=onset_colors['s'],
                            linewidth=cfg['linewidth_onset'], label='S onset'))
                    if has_coda:
                        legend_elements.append(plt.Line2D(
                            [0], [0], color=onset_colors['coda'],
                            linewidth=cfg['linewidth_onset'],
                            label=f'Coda onset ({coda_method})'))
                    if has_post_event:
                        legend_elements.append(plt.Line2D(
                            [0], [0], color=onset_colors['coda_end'],
                            linewidth=cfg['linewidth_onset'],
                            linestyle=':', alpha=0.8, label='Coda end'))
                if show_window_backgrounds:
                    for window_name, label in [
                        ('pre_event', 'Pre-event'), ('p_wave', 'P-wave'),
                        ('s_wave', 'S-wave'), ('coda', 'Coda'),
                        ('post_event', 'Post-event'),
                    ]:
                        if window_name in windows:
                            legend_elements.append(mpatches.Patch(
                                color=window_colors[window_name],
                                alpha=max(cfg['window_alpha'], 0.5),
                                label=label))
                legend_built = True

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(suptitle, fontsize=cfg['font_title'], fontweight='bold', y=0.97)

    # ── External legend ───────────────────────────────────────────────────────
    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.90),
            ncol=len(legend_elements),
            fontsize=cfg['font_legend'],
            framealpha=0.9,
            handlelength=1.5,
            columnspacing=1.0,
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path is not None:
        suffix = _mode_settings[mode]['output_suffix']
        output_path = Path(output_path).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig

def plot_ar_aic_onset_detection(
    station: str,
    signals_dicts: Dict[str, Dict],
    df_results: pd.DataFrame,
    signal_type: str = 'acceleration',
    component: str = 'HNZ',
    signal_units: Optional[Dict[str, str]] = None,
    show_windows: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'thesis',
) -> plt.Figure:
    """
    Plot a single-component signal with AR-AIC detected and theoretical
    P/S onset times and optional search windows.

    Parameters
    ----------
    station : str
        Station code (e.g. 'OGDI').
    signals_dicts : dict
        Dictionary mapping signal type to signals_dict, structured as
        {'acceleration': {...}, 'velocity': {...}, 'displacement': {...}}.
        Each inner dict: {station: {component: array, 'time': array}}.
    df_results : pd.DataFrame
        AR-AIC onset detection results, one row per station. Expected columns:
        STATION_CODE, EPICENTRAL_DISTANCE_KM,
        t_p_theo_seconds, t_s_theo_seconds,
        t_p_detected_seconds, t_s_detected_seconds,
        p_residual_seconds, s_residual_seconds,
        p_detection_success, s_detection_success,
        p_window_start_seconds, p_window_end_seconds (if show_windows=True),
        s_window_start_seconds, s_window_end_seconds (if show_windows=True).
    signal_type : str, optional
        Which signal type to plot (default: 'acceleration').
    component : str, optional
        Which component to plot (default: 'HNZ').
    signal_units : dict, optional
        Mapping from signal type to unit string, e.g.
        {'acceleration': 'cm/s²', 'velocity': 'cm/s', 'displacement': 'cm'}.
        If None, the y-axis label shows only the component name.
    show_windows : bool, optional
        If True, shade the P and S search windows (default: True).
    output_path : str or Path, optional
        If provided, save the figure to this path. The file extension is
        set by the mode (.pdf for thesis/interactive, .png for paper/poster).
    mode : str, optional
        Output mode controlling figure size and font sizes.
        One of 'thesis', 'paper', 'poster', 'interactive' (default: 'thesis').

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If mode is not one of the accepted values, or if signal_type or
        component are not found in signals_dicts.
    KeyError
        If station is not found in signals_dicts or df_results.
    """
    _VALID_MODES = ('interactive', 'paper', 'poster', 'thesis')
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}"
        )
    if signal_type not in signals_dicts:
        raise ValueError(
            f"Signal type '{signal_type}' not found in signals_dicts. "
            f"Available: {list(signals_dicts)}"
        )

    _mode_settings = {
        'interactive': dict(
            figsize=(10, 4),
            dpi_save=150,
            font_title=12,
            font_axis_label=10,
            font_tick=9,
            font_legend=9,
            linewidth_signal=0.6,
            linewidth_onset=1.8,
            window_alpha=0.15,
            output_suffix='.pdf',
        ),
        'paper': dict(
            figsize=(3.5, 2.5),
            dpi_save=600,
            font_title=8,
            font_axis_label=7,
            font_tick=6,
            font_legend=6,
            linewidth_signal=0.4,
            linewidth_onset=1.0,
            window_alpha=0.12,
            output_suffix='.png',
        ),
        'poster': dict(
            figsize=(7.0, 4.0),
            dpi_save=300,
            font_title=18,
            font_axis_label=15,
            font_tick=13,
            font_legend=12,
            linewidth_signal=0.8,
            linewidth_onset=2.0,
            window_alpha=0.15,
            output_suffix='.png',
        ),
        'thesis': dict(
            figsize=(6.5, 3.0),
            dpi_save=300,
            font_title=11,        # era 9
            font_axis_label=10,   # era 8
            font_tick=9,          # era 7
            font_legend=10,        # era 7
            linewidth_signal=0.4,
            linewidth_onset=1.2,
            window_alpha=0.12,
            output_suffix='.pdf',
        ),
    }
    cfg = _mode_settings[mode]

    onset_colors = {
        'p': '#C0392B',
        's': '#00807F',
    }

    signals_dict = signals_dicts[signal_type]
    if station not in signals_dict:
        raise KeyError(
            f"Station '{station}' not found in signals_dicts['{signal_type}']."
        )

    station_rows = df_results[df_results['STATION_CODE'] == station]
    if station_rows.empty:
        raise KeyError(
            f"Station '{station}' not found in df_results."
        )
    row = station_rows.iloc[0]

    data = signals_dict[station]
    if component not in data:
        raise ValueError(
            f"Component '{component}' not found for station '{station}'. "
            f"Available: {[k for k in data if k != 'time']}"
        )

    time = data['time']
    signal = data[component]

    t_p_theo = row.get('t_p_theo_seconds', np.nan)
    t_s_theo = row.get('t_s_theo_seconds', np.nan)
    t_p_det = row.get('t_p_detected_seconds', np.nan)
    t_s_det = row.get('t_s_detected_seconds', np.nan)
    p_res = row.get('p_residual_seconds', np.nan)
    s_res = row.get('s_residual_seconds', np.nan)
    p_success = bool(row.get('p_detection_success', False))
    s_success = bool(row.get('s_detection_success', False))
    p_win_start = row.get('p_window_start_seconds', np.nan)
    p_win_end = row.get('p_window_end_seconds', np.nan)
    s_win_start = row.get('s_window_start_seconds', np.nan)
    s_win_end = row.get('s_window_end_seconds', np.nan)

    fig, ax = plt.subplots(1, 1, figsize=cfg['figsize'])
    fig.subplots_adjust(top=0.83, bottom=0.20, left=0.12, right=0.97)

    ax.plot(time, signal, 'k-', linewidth=cfg['linewidth_signal'],
            alpha=0.8, zorder=1)

    if show_windows:
        if not (pd.isna(p_win_start) or pd.isna(p_win_end)):
            ax.axvspan(p_win_start, p_win_end,
                       alpha=cfg['window_alpha'],
                       color=onset_colors['p'], zorder=0)
        if not (pd.isna(s_win_start) or pd.isna(s_win_end)):
            ax.axvspan(s_win_start, s_win_end,
                       alpha=cfg['window_alpha'],
                       color=onset_colors['s'], zorder=0)

    if not pd.isna(t_p_theo):
        ax.axvline(t_p_theo, color=onset_colors['p'], linestyle='--',
                   linewidth=cfg['linewidth_onset'], alpha=0.6, zorder=2,
                   label='P theoretical')
    if not pd.isna(t_s_theo):
        ax.axvline(t_s_theo, color=onset_colors['s'], linestyle='--',
                   linewidth=cfg['linewidth_onset'], alpha=0.6, zorder=2,
                   label='S theoretical')
    if p_success and not pd.isna(t_p_det):
        ax.axvline(t_p_det, color=onset_colors['p'], linestyle='-',
                   linewidth=cfg['linewidth_onset'], zorder=3,
                   label='P detected')
    if s_success and not pd.isna(t_s_det):
        ax.axvline(t_s_det, color=onset_colors['s'], linestyle='-',
                   linewidth=cfg['linewidth_onset'], zorder=3,
                   label='S detected')

    unit_str = ''
    if signal_units is not None and signal_type in signal_units:
        unit_str = f' ({signal_units[signal_type]})'
    ax.set_xlabel('Time (s)', fontsize=cfg['font_axis_label'])
    ax.set_ylabel(f'{component}{unit_str}', fontsize=cfg['font_axis_label'])
    ax.tick_params(labelsize=cfg['font_tick'])
    ax.grid(True, alpha=0.3, zorder=0)

    legend_handles = []
    if show_windows:
        legend_handles.append(mpatches.Patch(
            color=onset_colors['p'],
            alpha=max(cfg['window_alpha'], 0.4),
            label='P search window'))
        legend_handles.append(mpatches.Patch(
            color=onset_colors['s'],
            alpha=max(cfg['window_alpha'], 0.4),
            label='S search window'))
    legend_handles += [
        plt.Line2D([0], [0], color=onset_colors['p'], linestyle='--',
                   linewidth=cfg['linewidth_onset'], alpha=0.6,
                   label='P theoretical'),
        plt.Line2D([0], [0], color=onset_colors['s'], linestyle='--',
                   linewidth=cfg['linewidth_onset'], alpha=0.6,
                   label='S theoretical'),
        plt.Line2D([0], [0], color=onset_colors['p'], linestyle='-',
                   linewidth=cfg['linewidth_onset'], label='P detected'),
        plt.Line2D([0], [0], color=onset_colors['s'], linestyle='-',
                   linewidth=cfg['linewidth_onset'], label='S detected'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(legend_handles),
        fontsize=cfg['font_legend'],
        framealpha=0.9,
        handlelength=1.5,
        columnspacing=1.0,
    )

    if p_success and s_success and not (pd.isna(p_res) or pd.isna(s_res)):
        residual_str = f'P residual: {p_res:+.2f} s  |  S residual: {s_res:+.2f} s'
    elif p_success and not pd.isna(p_res):
        residual_str = f'P residual: {p_res:+.2f} s  |  S detection failed'
    elif s_success and not pd.isna(s_res):
        residual_str = f'P detection failed  |  S residual: {s_res:+.2f} s'
    else:
        residual_str = 'Detection failed'

    dist = row.get('EPICENTRAL_DISTANCE_KM', np.nan)
    dist_str = f', $d_{{\\mathrm{{epi}}}}$ = {dist:.1f} km' if not pd.isna(dist) else ''
    fig.suptitle(
        f'Station {station}{dist_str} — AR-AIC onset detection',
        fontsize=cfg['font_title'],
        fontweight='bold',
        y=0.98,
    )
    fig.text(
        0.5, 0.90,
        residual_str,
        fontsize=cfg['font_title'] - 1,
        ha='center',
        va='top',
    )

    if output_path is not None:
        suffix = cfg['output_suffix']
        output_path = Path(output_path).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=cfg['dpi_save'], bbox_inches='tight')

    return fig