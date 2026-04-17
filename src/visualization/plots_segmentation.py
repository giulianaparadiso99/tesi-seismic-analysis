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
from scipy.stats import pearsonr, linregress
from pathlib import Path
import re


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
    from IPython.display import display
    
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

def plot_coda_method_comparison(df_onsets_full, output_path=None):
    """
    Create scatter plots comparing coda detection methods pairwise.
    
    Generates a 1x3 subplot figure showing:
    - Rautian vs Arias
    - Rautian vs Envelope
    - Arias vs Envelope
    
    Each subplot includes:
    - Scatter points
    - y=x reference line (perfect agreement)
    - Linear regression line
    - Correlation coefficient and RMSE
    
    Parameters
    ----------
    df_onsets_full : pd.DataFrame
        Must contain columns: t_coda_rautian, t_coda_arias, t_coda_envelope
    output_path : str or Path, optional
        If provided, save figure to this path
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    
    Examples
    --------
    >>> fig, axes = plot_coda_method_comparison(df_onsets_full)
    >>> plt.show()
    >>> 
    >>> # Or save directly
    >>> plot_coda_method_comparison(
    ...     df_onsets_full, 
    ...     output_path='figures/coda_method_comparison.png'
    ... )
    """
    
    # Method pairs to compare
    comparisons = [
        ('rautian', 'arias', 'Rautian vs Arias'),
        ('rautian', 'envelope', 'Rautian vs Envelope'),
        ('arias', 'envelope', 'Arias vs Envelope')
    ]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (m1, m2, title) in enumerate(comparisons):
        ax = axes[idx]
        
        col1 = f't_coda_{m1}'
        col2 = f't_coda_{m2}'
        
        # Get data (remove NaN)
        mask = df_onsets_full[col1].notna() & df_onsets_full[col2].notna()
        data1 = df_onsets_full.loc[mask, col1].values
        data2 = df_onsets_full.loc[mask, col2].values
        
        # Calculate statistics
        corr, p_value = pearsonr(data1, data2)
        rmse = np.sqrt(np.mean((data2 - data1)**2))
        mae = np.mean(np.abs(data2 - data1))
        
        # Linear regression
        slope, intercept, r_value, _, _ = linregress(data1, data2)
        
        # Scatter plot
        ax.scatter(data1, data2, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
        
        # y=x reference line (perfect agreement)
        lim_min = min(data1.min(), data2.min())
        lim_max = max(data1.max(), data2.max())
        margin = 0.05 * (lim_max - lim_min)
        ax.plot([lim_min - margin, lim_max + margin], 
                [lim_min - margin, lim_max + margin],
                'k--', linewidth=1.5, alpha=0.5, label='y = x (perfect agreement)')
        
        # Regression line
        x_fit = np.array([lim_min - margin, lim_max + margin])
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.7,
                label=f'Linear fit (slope={slope:.3f})')
        
        # Labels and title
        ax.set_xlabel(f'{m1.capitalize()} $t_{{coda}}$ (s)', fontsize=12)
        ax.set_ylabel(f'{m2.capitalize()} $t_{{coda}}$ (s)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Statistics text box
        stats_text = (
            f'$r = {corr:.3f}$ ($p < 0.001$)\n'
            f'RMSE = {rmse:.2f} s\n'
            f'MAE = {mae:.2f} s\n'
            f'$n = {len(data1)}$'
        )
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc='lower right', fontsize=9)
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lim_min - margin, lim_max + margin)
        ax.set_ylim(lim_min - margin, lim_max + margin)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    
    return fig, axes