"""
plot_segmentation.py
--------------------
Visualization functions for event segmentation results.

Includes plots for:
- Theoretical arrival times
- Detected vs theoretical onsets comparison
- Window boundaries
- Residuals analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, linregress
from pathlib import Path
import re
from typing import Dict, List, Optional, Literal
import os
import matplotlib.patches as mpatches
from IPython.display import display


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
        't_p_theo', 
        't_s_theo'
    ]].copy()
    
    # Round for readability
    table['vp_crust'] = table['vp_crust'].round(2)
    table['vs_crust'] = table['vs_crust'].round(2)
    table['t_p_theo'] = table['t_p_theo'].round(2)
    table['t_s_theo'] = table['t_s_theo'].round(2)
    
    # Sort by distance
    table = table.sort_values('EPICENTRAL_DISTANCE_KM')
    
    # Display statistics
    print("Theoretical Arrival Times Summary")
    print("=" * 70)
    print(f"\nNumber of stations: {len(table)}")
    print(f"\nDistance range: {table['EPICENTRAL_DISTANCE_KM'].min():.1f} - {table['EPICENTRAL_DISTANCE_KM'].max():.1f} km")
    print(f"P-wave arrival range: {table['t_p_theo'].min():.1f} - {table['t_p_theo'].max():.1f} s")
    print(f"S-wave arrival range: {table['t_s_theo'].min():.1f} - {table['t_s_theo'].max():.1f} s")
    print(f"\nMedian crustal velocities:")
    print(f"  v_P = {table['vp_crust'].median():.2f} km/s")
    print(f"  v_S = {table['vs_crust'].median():.2f} km/s")
    print(f"\nFirst {n_rows} stations (sorted by distance):")
    print("=" * 70)
    
    display(table.head(n_rows))
    
    return table

def plot_apparent_vs_crustal_velocities(df_meta_stations, figsize=(16, 6)):
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
    import matplotlib.pyplot as plt
    
    # Calculate apparent velocities
    df = df_meta_stations.copy()
    df['v_p_apparent'] = df['EPICENTRAL_DISTANCE_KM'] / (df['t_p_detected'] - df['origin_time'])
    df['v_s_apparent'] = df['EPICENTRAL_DISTANCE_KM'] / (df['t_s_detected'] - df['origin_time'])
    
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

def plot_crustal_velocities_vs_distance(df_meta_stations, figsize=(16, 6), output_path=None):
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
    import matplotlib.pyplot as plt
    
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
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig

def plot_theoretical_arrivals(df_stations, figsize=(12, 6), save_path=None):
    """
    Plot theoretical P and S arrival times vs epicentral distance.
    
    Creates scatter plot showing the linear relationship between
    distance and arrival time. Slope reflects crustal velocities.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with columns:
        - EPICENTRAL_DISTANCE_KM
        - t_p_theo
        - t_s_theo
        - vp_crust
        - vs_crust
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str or Path, optional
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
    t_p = df_stations['t_p_theo']
    t_s = df_stations['t_s_theo']
    
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
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax

def plot_onset_detection_results(signals_dict, df_results, 
                                 stations=None, 
                                 figsize_per_station=(10, 8),
                                 save_dir=None,
                                 show_windows=True):
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
        - t_p_theo, t_s_theo
        - t_p_detected, t_s_detected
        - p_residual, s_residual
        - p_detection_success, s_detection_success
        - p_window_start, p_window_end, s_window_start, s_window_end
    stations : list of str, optional
        Which stations to plot (default: all stations in df_results)
    figsize_per_station : tuple, optional
        Figure size (width, height) for each station (default: (10, 8))
    save_dir : str or Path, optional
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
    ...     save_dir='../figures/onset_detection'
    ... )
    >>> plt.show()
    """
    
    if stations is None:
        stations = df_results['STATION_CODE'].tolist()
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    for station in stations:
        # Get results for this station
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
        
        # Check for incomplete components
        if comp_z is None or comp_n is None or comp_e is None:
            missing = []
            if comp_z is None: missing.append('Z')
            if comp_n is None: missing.append('N')
            if comp_e is None: missing.append('E')
            print(f"Warning: {station} missing components: {missing}")
            continue
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize_per_station, sharex=True)
        
        components_list = [(comp_z, 'Vertical'), (comp_n, 'North'), (comp_e, 'East')]
        
        for ax, (comp, label) in zip(axes, components_list):
            signal = data[comp]
            
            # Plot acceleration
            ax.plot(time, signal, 'k-', linewidth=0.5, alpha=0.7, zorder=1)
            ax.set_ylabel(f'{label}\n(cm/s²)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Plot search windows if available and requested
            if show_windows:
                # P window
                if 'p_window_start' in station_result.index:
                    p_win_start = station_result['p_window_start']
                    p_win_end = station_result['p_window_end']
                    
                    if not pd.isna(p_win_start) and not pd.isna(p_win_end):
                        ax.axvspan(p_win_start, p_win_end, alpha=0.15, color='blue', 
                                  label='P window' if ax == axes[0] else '', zorder=0)
                
                # S window
                if 's_window_start' in station_result.index:
                    s_win_start = station_result['s_window_start']
                    s_win_end = station_result['s_window_end']
                    
                    if not pd.isna(s_win_start) and not pd.isna(s_win_end):
                        ax.axvspan(s_win_start, s_win_end, alpha=0.15, color='red', 
                                  label='S window' if ax == axes[0] else '', zorder=0)
            
            # Plot theoretical arrivals (dashed)
            if not pd.isna(station_result['t_p_theo']):
                ax.axvline(station_result['t_p_theo'], color='blue', 
                          linestyle='--', linewidth=1.5, alpha=0.6, 
                          label='P theo' if ax == axes[0] else '', zorder=2)
            
            if not pd.isna(station_result['t_s_theo']):
                ax.axvline(station_result['t_s_theo'], color='red', 
                          linestyle='--', linewidth=1.5, alpha=0.6,
                          label='S theo' if ax == axes[0] else '', zorder=2)
            
            # Plot detected arrivals (solid) if successful
            p_success = station_result.get('p_detection_success', False)
            s_success = station_result.get('s_detection_success', False)
            
            if p_success and not pd.isna(station_result['t_p_detected']):
                ax.axvline(station_result['t_p_detected'], color='blue', 
                          linestyle='-', linewidth=2.5,
                          label='P detected' if ax == axes[0] else '', zorder=3)
            
            if s_success and not pd.isna(station_result['t_s_detected']):
                ax.axvline(station_result['t_s_detected'], color='red', 
                          linestyle='-', linewidth=2.5,
                          label='S detected' if ax == axes[0] else '', zorder=3)
        
        # Set xlabel only on bottom plot
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        # Title with residual info
        if p_success and s_success:
            title = (f"Station {station} - AR-AIC Onset Detection\n"
                    f"P residual: {station_result['p_residual']:+.2f} s  |  "
                    f"S residual: {station_result['s_residual']:+.2f} s")
        elif p_success:
            title = (f"Station {station} - AR-AIC Onset Detection\n"
                    f"P residual: {station_result['p_residual']:+.2f} s  |  S detection FAILED")
        elif s_success:
            title = (f"Station {station} - AR-AIC Onset Detection\n"
                    f"P detection FAILED  |  S residual: {station_result['s_residual']:+.2f} s")
        else:
            error_msg = station_result.get('error_message', 'Unknown error')
            title = f"Station {station} - Detection FAILED\n{error_msg}"
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Legend on top plot
        axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save if requested
        if save_dir is not None:
            save_path = save_dir / f'onset_detection_{station}.pdf'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        figures[station] = fig
    
    return figures

def plot_coda_onset_results(signals_dict, df_onsets_full,
                            stations=None,
                            figsize_per_station=(10, 8),
                            save_dir=None):
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
    stations : list of str, optional
        Which stations to plot (default: all stations in df_onsets_full)
    figsize_per_station : tuple, optional
        Figure size (width, height) for each station (default: (10, 8))
    save_dir : str or Path, optional
        Directory to save figures. If None, figures are not saved.
    
    Returns
    -------
    dict
        Dictionary {station: fig} of created figures
    
    Examples
    --------
    >>> figs = plot_coda_onset_results(signals_dict, df_onsets_full,
    ...                                stations=['ACER', 'CLFR'],
    ...                                save_dir='../figures/coda_detection')
    >>> plt.show()
    """
    
    if stations is None:
        stations = df_onsets_full['STATION_CODE'].unique().tolist()
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
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
            ax.set_ylabel(f'{label}\n(cm/s²)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Plot S-wave onset
            if not np.isnan(comp_result['t_s_detected']):
                ax.axvline(comp_result['t_s_detected'], color='red',
                          linestyle='-', linewidth=2.5,
                          label='S onset' if ax == axes[0] else '', zorder=2)
            
            # Plot coda onsets (3 methods)
            if not np.isnan(comp_result['t_coda_rautian']):
                ax.axvline(comp_result['t_coda_rautian'], color='green',
                          linestyle='-', linewidth=2,
                          label='Coda (Rautian)' if ax == axes[0] else '', zorder=3)
            
            if not np.isnan(comp_result['t_coda_arias']):
                ax.axvline(comp_result['t_coda_arias'], color='orange',
                          linestyle='-', linewidth=2,
                          label='Coda (Arias)' if ax == axes[0] else '', zorder=3)
            
            if not np.isnan(comp_result['t_coda_envelope']):
                ax.axvline(comp_result['t_coda_envelope'], color='purple',
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
            
            if not np.isnan(comp_result['s_duration_rautian']):
                durations_rautian.append(comp_result['s_duration_rautian'])
            if not np.isnan(comp_result['s_duration_arias']):
                durations_arias.append(comp_result['s_duration_arias'])
            if not np.isnan(comp_result['s_duration_envelope']):
                durations_envelope.append(comp_result['s_duration_envelope'])
        
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
        if save_dir is not None:
            save_path = save_dir / f'coda_detection_{station}.pdf'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        figures[station] = fig
    
    return figures

def plot_coda_scatter_comparison(stats, save_path=None):
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
    save_path : str or Path, optional
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_bland_altman_comparison(stats, save_path=None):
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
    save_path : str or Path, optional
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_residuals_vs_distance(stats, save_path=None):
    """
    Plot method differences vs epicentral distance.
    
    Shows how agreement between methods varies with distance, revealing
    whether bias increases for distant stations (SNR degradation effect).
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    save_path : str or Path, optional
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_pairwise_difference_histograms(stats, save_path=None):
    """
    Plot histograms of pairwise differences between methods.
    
    Shows distribution of differences for each method pair, revealing
    whether bias is symmetric and normally distributed.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    save_path : str or Path, optional
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_correlation_matrix_heatmap(stats, save_path=None):
    """
    Plot correlation matrix heatmap for all methods.
    
    Displays pairwise correlations between all three coda detection methods
    in a visually intuitive heatmap format.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_coda_method_statistics()
    save_path : str or Path, optional
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
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
    coda_method: str = 'rautian',
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
    show_onset_lines: bool = True,
    show_window_backgrounds: bool = True,
    title_suffix: str = ''
) -> plt.Figure:
    """
    Plot three-component signal with onset times and window boundaries.
    
    Automatically detects component naming convention (HNE/HNN/HNZ vs 
    HGE/HGN/HGZ vs HN1/HN2/HNZ).
    
    Parameters
    ----------
    station : str
        Station code (e.g., 'BRZ')
    signals_dict : dict
        Full signals dictionary from convert_signals_to_dict()
    windowed_signals : dict
        Windowed signals from segment_all_signals()
    df_onsets : pd.DataFrame, optional
        DataFrame with onset times for displaying in legend
    coda_method : str, optional
        Which coda method was used ('rautian', 'arias', 'envelope')
        Only used if df_onsets is provided (default: 'rautian')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (14, 10))
    save_path : str, optional
        If provided, save figure to this path
    show_onset_lines : bool, optional
        Show vertical lines at onset times (default: True)
    show_window_backgrounds : bool, optional
        Show colored backgrounds for each window (default: True)
    title_suffix : str, optional
        Additional text for title (default: '')
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        
    Raises
    ------
    ValueError
        If station not found or no components available
    """
    
    if station not in signals_dict:
        raise ValueError(f"Station '{station}' not found in signals_dict")
    
    if station not in windowed_signals:
        raise ValueError(f"Station '{station}' not found in windowed_signals")
    
    # Get component mapping
    comp_map = get_station_components(station, signals_dict)
    
    if len(comp_map) == 0:
        raise ValueError(f"No components found for station {station}")
    
    # Determine plot order and labels based on available components
    if 'Z' in comp_map and 'N' in comp_map and 'E' in comp_map:
        # N-E system: plot Z, N, E
        plot_order = [('Z', 'Vertical'), ('N', 'North'), ('E', 'East')]
    elif 'Z' in comp_map and '1' in comp_map and '2' in comp_map:
        # 1-2 system: plot Z, 1, 2
        plot_order = [('Z', 'Vertical'), ('1', 'Horizontal-1'), ('2', 'Horizontal-2')]
    else:
        # Fallback: plot whatever is available
        plot_order = [(k, f'Component {v}') for k, v in comp_map.items()]
    
    # Filter to actually available components
    plot_order = [(key, label) for key, label in plot_order if key in comp_map]
    
    # Window colors
    window_colors = {
        'pre_event': '#E8E8E8',
        'p_wave': '#AED6F1',
        's_wave': '#F9E79F',
        'coda': '#D5F4E6'
    }
    
    # Onset line colors
    onset_colors = {
        'p': '#E74C3C',
        's': '#3498DB',
        'coda': '#27AE60'
    }
    
    # Create figure
    n_subplots = len(plot_order)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    
    # Handle single subplot case
    if n_subplots == 1:
        axes = [axes]
    
    # Get full time array
    time_full = signals_dict[station]['time']
    
    # Plot each component
    for idx, (comp_key, comp_label) in enumerate(plot_order):
        ax = axes[idx]
        
        # Get actual component code
        component = comp_map[comp_key]
        
        # Get full signal
        signal_full = signals_dict[station][component]
        
        # Get windowed data
        windows = windowed_signals[station][component]
        
        # Extract onset times
        t_p = windows['p_wave']['t_start']
        t_s = windows['s_wave']['t_start']
        t_coda = windows['coda']['t_start']
        
        # Plot window backgrounds
        if show_window_backgrounds:
            for window_name in ['pre_event', 'p_wave', 's_wave', 'coda']:
                w = windows[window_name]
                ax.axvspan(w['t_start'], w['t_end'], 
                          color=window_colors[window_name],
                          alpha=0.3, zorder=0)
        
        # Plot signal
        ax.plot(time_full, signal_full, 'k-', linewidth=0.6, zorder=2)
        
        # Plot onset lines
        if show_onset_lines:
            ax.axvline(t_p, color=onset_colors['p'], linewidth=2, 
                      linestyle='-', zorder=3, label='P onset')
            ax.axvline(t_s, color=onset_colors['s'], linewidth=2,
                      linestyle='-', zorder=3, label='S onset')
            ax.axvline(t_coda, color=onset_colors['coda'], linewidth=2,
                      linestyle='-', zorder=3, label=f'Coda ({coda_method})')
        
        # Labels and formatting
        ax.set_ylabel(f'{comp_label}\n{component}\n(cm/s²)', fontsize=10)
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Legend only on first subplot
        if idx == 0:
            legend_elements = []
            
            if show_onset_lines:
                legend_elements.extend([
                    plt.Line2D([0], [0], color=onset_colors['p'], linewidth=2, 
                              label='P onset'),
                    plt.Line2D([0], [0], color=onset_colors['s'], linewidth=2,
                              label='S onset'),
                    plt.Line2D([0], [0], color=onset_colors['coda'], linewidth=2,
                              label=f'Coda onset ({coda_method})')
                ])
            
            if show_window_backgrounds:
                legend_elements.extend([
                    mpatches.Patch(color=window_colors['pre_event'], alpha=0.3,
                                  label='Pre-event'),
                    mpatches.Patch(color=window_colors['p_wave'], alpha=0.3,
                                  label='P-wave'),
                    mpatches.Patch(color=window_colors['s_wave'], alpha=0.3,
                                  label='S-wave'),
                    mpatches.Patch(color=window_colors['coda'], alpha=0.3,
                                  label='Coda')
                ])
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     fontsize=9, framealpha=0.9)
    
    # X-axis label
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    # Title with component info
    comp_str = ', '.join([comp_map[k] for k, _ in plot_order])
    dur_s = windows['s_wave']['duration']
    
    title = f"Station {station} ({comp_str}){title_suffix}\n"
    title += f"S-wave duration: {dur_s:.2f}s"
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multiple_stations(
    stations: List[str],
    signals_dict: Dict,
    windowed_signals: Dict,
    df_onsets: Optional[pd.DataFrame] = None,
    coda_method: str = 'rautian',
    save_dir: Optional[str] = None,
    close_after_save: bool = True,
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
    coda_method : str, optional
        Coda detection method name (default: 'rautian')
    save_dir : str, optional
        Directory to save plots (e.g., 'plots/windows/')
        If None, plots are not saved automatically
    close_after_save : bool, optional
        If True, close figures after saving to free memory (default: True)
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
    ...                        save_dir='plots/windows/', 
    ...                        close_after_save=True)
    """
    
    figures = {}
    
    print(f"\nPlotting {len(stations)} stations...")
    
    for i, station in enumerate(stations, 1):
        try:
            # Determine save path if directory provided
            if save_dir:
                save_path = os.path.join(save_dir, f'{station}_windows.pdf')
            else:
                save_path = None
            
            # Create plot
            fig = plot_station_windows(
                station=station,
                signals_dict=signals_dict,
                windowed_signals=windowed_signals,
                df_onsets=df_onsets,
                coda_method=coda_method,
                save_path=save_path,
                **kwargs
            )
            
            # Store or close
            if close_after_save and save_path:
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
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None
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
    figsize : tuple, optional
        Figure size (default: (14, 6))
    save_path : str, optional
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
        if idx == 0:
            t_p = windows['p_wave']['t_start']
            ax.axvline(t_p, color='red', linewidth=2, linestyle='-',
                      label='P onset', zorder=10)
        
        # S onset (should be same for all)
        if idx == 0:
            t_s = windows['s_wave']['t_start']
            ax.axvline(t_s, color='blue', linewidth=2, linestyle='-',
                      label='S onset', zorder=10)
        
        # Coda onset (different for each method)
        t_coda = windows['coda']['t_start']
        ax.axvline(t_coda, color=color, linewidth=2.5, linestyle='--',
                  label=f'Coda ({label}): {t_coda:.1f}s', zorder=9)
        
        # Pre-event start (if different)
        if idx > 0:
            t_pre = windows['pre_event']['t_start']
            prev_t_pre = windowed_dict_list[0][station][component]['pre_event']['t_start']
            if abs(t_pre - prev_t_pre) > 0.1:
                ax.axvline(t_pre, color=color, linewidth=1.5, linestyle=':',
                          alpha=0.6, label=f'Pre-event start ({label})')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(f'{component} Acceleration (cm/s²)', fontsize=12)
    ax.set_title(f'Station {station} - {component}: Method Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig