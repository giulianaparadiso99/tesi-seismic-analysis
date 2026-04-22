"""
Quality control functions for seismic signal windowing.

This module provides validation functions to check:
- PGA location (should occur in S-wave window)
- Arrival time monotonicity with epicentral distance
- Signal-to-noise ratio (SNR) for onset picks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_pga_in_s_wave(df_metadata, station, component, coda_method='rautian'):
    """
    Check if PGA occurs in S-wave window using only metadata.
    
    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata with onset times and PGA info
    station : str
        Station code
    component : str
        Component name
    coda_method : str
        Which coda method to use: 'rautian', 'arias', 'envelope', 'median'
    
    Returns
    -------
    dict
        {
            'passed': bool,
            'pga_window': str,
            'pga_value': float,
            'pga_time': float
        }
    """
    mask = (df_metadata['STATION_CODE'] == station) & \
           (df_metadata['COMPONENT'] == component)
    
    if not mask.any():
        return {
            'passed': False,
            'pga_window': 'UNKNOWN',
            'pga_value': np.nan,
            'pga_time': np.nan
        }
    
    row = df_metadata[mask].iloc[0]
    
    pga_time = row['TIME_PGA_S']
    pga_value = row['PGA_CM/S^2']
    t_p = row['t_p_detected']
    t_s = row['t_s_detected']
    t_coda = row[f't_coda_{coda_method}']
    
    # Determine window
    if pga_time < t_p:
        pga_window = 'pre_event'
    elif t_p <= pga_time < t_s:
        pga_window = 'p_wave'
    elif t_s <= pga_time < t_coda:
        pga_window = 's_wave'
    else:  # pga_time >= t_coda
        pga_window = 'coda'
    
    return {
        'passed': pga_window == 's_wave',
        'pga_window': pga_window,
        'pga_value': pga_value,
        'pga_time': pga_time
    }


def check_monotonicity_station(df_meta_stations, station, phase='p'):
    """
    Check if arrival time is monotonic with distance for this station.
    
    Logic:
    - Stations are sorted by epicentral distance
    - For station i: check t_p[i-1] < t_p[i] < t_p[i+1]
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata (one row per station, not per component)
        Must contain: 'STATION_CODE', 'EPICENTRAL_DISTANCE_KM', 
                      't_p_detected', 't_s_detected'
    station : str
        Station code
    phase : str
        'p' or 's'
    
    Returns
    -------
    dict
        {
            'passed': bool,
            'position': int,  # position in sorted list (0=closest)
            'n_stations': int,
            't_prev': float or None,
            't_this': float,
            't_next': float or None,
            'violation_side': str or None  # 'prev', 'next', or None
        }
    """
    # Sort by distance (no groupby needed, already aggregated)
    df_sorted = df_meta_stations.sort_values('EPICENTRAL_DISTANCE_KM').reset_index(drop=True)
    
    # Find position of this station
    idx = df_sorted[df_sorted['STATION_CODE'] == station].index[0]
    n_stations = len(df_sorted)
    
    t_this = df_sorted.loc[idx, f't_{phase}_detected']
    
    # Get neighbors
    t_prev = df_sorted.loc[idx - 1, f't_{phase}_detected'] if idx > 0 else None
    t_next = df_sorted.loc[idx + 1, f't_{phase}_detected'] if idx < n_stations - 1 else None
    
    # Check monotonicity
    violation_prev = (t_prev is not None) and (t_prev >= t_this)
    violation_next = (t_next is not None) and (t_next <= t_this)
    
    passed = not (violation_prev or violation_next)
    
    violation_side = None
    if violation_prev:
        violation_side = 'prev'
    elif violation_next:
        violation_side = 'next'
    
    return {
        'passed': passed,
        'position': idx,
        'n_stations': n_stations,
        't_prev': t_prev,
        't_this': t_this,
        't_next': t_next,
        'violation_side': violation_side
    }

def check_snr(windowed_signals, station, component, 
              phase='p', threshold=3.0, signal_duration=5.0, dt=0.005):
    """
    Check SNR for P or S pick using windowed signals.
    
    Uses pre_event window as noise reference and beginning of phase window as signal.
    
    Parameters
    ----------
    windowed_signals : dict
        Output from segment_all_signals()
    station : str
        Station code
    component : str
        Component name
    phase : str
        'p' or 's'
    threshold : float
        SNR threshold (default: 3.0)
    signal_duration : float
        Duration of signal window in seconds (default: 5.0)
    dt : float
        Sampling interval in seconds (default: 0.005)
    
    Returns
    -------
    dict
        {
            'passed': bool,
            'snr': float,
            'rms_signal': float,
            'rms_noise': float,
            'threshold': float
        }
    """
    windows = windowed_signals[station][component]
    
    # Noise window: use entire pre_event window
    noise_signal = windows['pre_event']['signal']
    
    if len(noise_signal) == 0:
        return {
            'passed': False,
            'snr': np.nan,
            'rms_signal': np.nan,
            'rms_noise': np.nan,
            'threshold': threshold,
            'error': 'Empty pre_event window'
        }
    
    # Signal window: first N seconds of phase window
    phase_window_name = 'p_wave' if phase == 'p' else 's_wave'
    phase_signal = windows[phase_window_name]['signal']
    
    # Extract first signal_duration seconds
    n_samples = int(signal_duration / dt)
    signal_window = phase_signal[:n_samples]
    
    if len(signal_window) == 0:
        return {
            'passed': False,
            'snr': np.nan,
            'rms_signal': np.nan,
            'rms_noise': np.nan,
            'threshold': threshold,
            'error': f'Empty {phase_window_name} window'
        }
    
    # Compute RMS
    rms_noise = np.sqrt(np.mean(noise_signal**2))
    rms_signal = np.sqrt(np.mean(signal_window**2))
    
    # SNR
    if rms_noise == 0:
        snr = np.inf if rms_signal > 0 else np.nan
    else:
        snr = rms_signal / rms_noise
    
    return {
        'passed': snr >= threshold,
        'snr': snr,
        'rms_signal': rms_signal,
        'rms_noise': rms_noise,
        'threshold': threshold
    }

def quality_control_all_stations(windowed_signals, df_full, df_meta_stations,
                                 snr_threshold=3.0, coda_method='rautian'):
    """
    Run all quality checks for all stations and components.
    
    Parameters
    ----------
    windowed_signals : dict
        Segmented signals from segment_all_signals()
    df_full : pd.DataFrame
        Component-level metadata (used for PGA check)
        Must contain: 'STATION_CODE', 'COMPONENT', 'PGA_CM/S^2', 'TIME_PGA_S',
                      't_p_detected', 't_s_detected', 't_coda_{method}'
    df_meta_stations : pd.DataFrame
        Station-level metadata (used for monotonicity check)
        Must contain: 'STATION_CODE', 'EPICENTRAL_DISTANCE_KM',
                      't_p_detected', 't_s_detected'
    snr_threshold : float
        SNR threshold for quality check (default: 3.0)
    coda_method : str
        Coda detection method: 'rautian', 'arias', 'envelope', 'median'
    
    Returns
    -------
    dict
        Nested structure:
        {
            station: {
                component: {
                    'pga_check': dict,
                    'monotonicity_p': dict,
                    'monotonicity_s': dict,
                    'snr_p': dict,
                    'snr_s': dict,
                    'all_passed': bool
                }
            }
        }
    """
    
    results = {}
    
    for station in windowed_signals.keys():
        results[station] = {}
        
        for component in windowed_signals[station].keys():
            
            # PGA check (uses df_full, component-specific)
            pga_check = check_pga_in_s_wave(
                df_full, station, component, coda_method=coda_method
            )
            
            # Monotonicity checks (uses df_meta_stations, station-level)
            mono_p = check_monotonicity_station(
                df_meta_stations, station, phase='p'
            )
            mono_s = check_monotonicity_station(
                df_meta_stations, station, phase='s'
            )
            
            # SNR checks (uses windowed_signals)
            snr_p = check_snr(
                windowed_signals, station, component, 
                phase='p', threshold=snr_threshold
            )
            snr_s = check_snr(
                windowed_signals, station, component,
                phase='s', threshold=snr_threshold
            )
            
            # Aggregate results
            all_passed = (
                pga_check['passed'] and
                mono_p['passed'] and
                mono_s['passed'] and
                snr_p['passed'] and
                snr_s['passed']
            )
            
            results[station][component] = {
                'pga_check': pga_check,
                'monotonicity_p': mono_p,
                'monotonicity_s': mono_s,
                'snr_p': snr_p,
                'snr_s': snr_s,
                'all_passed': all_passed
            }
    
    return results

def print_quality_control_summary(qc_results):
    """
    Print quality control results in hierarchical table format.
    
    Format:
    Station A
      ├─ HGZ: [✓ PGA] [✓ MonoP] [✓ MonoS] [✓ SNRP] [✓ SNRS]
      ├─ HGN: [✗ PGA] [✓ MonoP] [✓ MonoS] [✓ SNRP] [✓ SNRS]
      └─ HGE: [✓ PGA] [✓ MonoP] [✓ MonoS] [✗ SNRP] [✓ SNRS]
    """
    
    def status_symbol(passed):
        return '✓' if passed else '✗'
    
    print("\n" + "="*80)
    print("QUALITY CONTROL SUMMARY")
    print("="*80)
    
    # Statistics
    total_components = 0
    all_passed_count = 0
    check_failures = {
        'pga': 0,
        'mono_p': 0,
        'mono_s': 0,
        'snr_p': 0,
        'snr_s': 0
    }
    
    # Print per station
    for station in sorted(qc_results.keys()):
        print(f"\n{station}")
        
        components = qc_results[station]
        component_names = sorted(components.keys())
        
        for i, component in enumerate(component_names):
            total_components += 1
            
            checks = components[component]
            
            # Choose prefix (tree structure)
            if i == len(component_names) - 1:
                prefix = "  └─"
            else:
                prefix = "  ├─"
            
            # Build status string
            pga_status = status_symbol(checks['pga_check']['passed'])
            mono_p_status = status_symbol(checks['monotonicity_p']['passed'])
            mono_s_status = status_symbol(checks['monotonicity_s']['passed'])
            snr_p_status = status_symbol(checks['snr_p']['passed'])
            snr_s_status = status_symbol(checks['snr_s']['passed'])
            
            status_line = (
                f"{prefix} {component}: "
                f"[{pga_status} PGA] "
                f"[{mono_p_status} MonoP] "
                f"[{mono_s_status} MonoS] "
                f"[{snr_p_status} SNRP] "
                f"[{snr_s_status} SNRS]"
            )
            
            print(status_line)
            
            # Count statistics
            if checks['all_passed']:
                all_passed_count += 1
            
            if not checks['pga_check']['passed']:
                check_failures['pga'] += 1
            if not checks['monotonicity_p']['passed']:
                check_failures['mono_p'] += 1
            if not checks['monotonicity_s']['passed']:
                check_failures['mono_s'] += 1
            if not checks['snr_p']['passed']:
                check_failures['snr_p'] += 1
            if not checks['snr_s']['passed']:
                check_failures['snr_s'] += 1
    
    # Print summary statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total components:     {total_components}")
    print(f"All checks passed:    {all_passed_count} ({100*all_passed_count/total_components:.1f}%)")
    print(f"\nFailures by check:")
    print(f"  PGA in S-wave:      {check_failures['pga']} ({100*check_failures['pga']/total_components:.1f}%)")
    print(f"  Monotonicity P:     {check_failures['mono_p']} ({100*check_failures['mono_p']/total_components:.1f}%)")
    print(f"  Monotonicity S:     {check_failures['mono_s']} ({100*check_failures['mono_s']/total_components:.1f}%)")
    print(f"  SNR P ≥ 3:          {check_failures['snr_p']} ({100*check_failures['snr_p']/total_components:.1f}%)")
    print(f"  SNR S ≥ 3:          {check_failures['snr_s']} ({100*check_failures['snr_s']/total_components:.1f}%)")
    print("="*80 + "\n")

def print_detailed_failures(qc_results):
    """
    Print detailed information for failed checks.
    """
    
    print("\n" + "="*80)
    print("DETAILED FAILURE REPORT")
    print("="*80)
    
    for station in sorted(qc_results.keys()):
        for component in sorted(qc_results[station].keys()):
            checks = qc_results[station][component]
            
            if checks['all_passed']:
                continue  # Skip if all passed
            
            print(f"\n{station}-{component}:")
            
            # PGA failure
            if not checks['pga_check']['passed']:
                pga = checks['pga_check']
                print(f"  ✗ PGA: Found in '{pga['pga_window']}' window (value={pga['pga_value']:.4f} cm/s²)")
            
            # Monotonicity P failure
            if not checks['monotonicity_p']['passed']:
                mono = checks['monotonicity_p']
                if mono['violation_side'] == 'prev':
                    print(f"  ✗ MonoP: t_P({mono['t_this']:.2f}s) ≤ t_P_prev({mono['t_prev']:.2f}s)")
                elif mono['violation_side'] == 'next':
                    print(f"  ✗ MonoP: t_P({mono['t_this']:.2f}s) ≥ t_P_next({mono['t_next']:.2f}s)")
            
            # Monotonicity S failure
            if not checks['monotonicity_s']['passed']:
                mono = checks['monotonicity_s']
                if mono['violation_side'] == 'prev':
                    print(f"  ✗ MonoS: t_S({mono['t_this']:.2f}s) ≤ t_S_prev({mono['t_prev']:.2f}s)")
                elif mono['violation_side'] == 'next':
                    print(f"  ✗ MonoS: t_S({mono['t_this']:.2f}s) ≥ t_S_next({mono['t_next']:.2f}s)")
            
            # SNR P failure
            if not checks['snr_p']['passed']:
                snr = checks['snr_p']
                print(f"  ✗ SNRP: {snr['snr']:.2f} < {snr['threshold']} (RMS_signal={snr['rms_signal']:.4f}, RMS_noise={snr['rms_noise']:.4f})")
            
            # SNR S failure
            if not checks['snr_s']['passed']:
                snr = checks['snr_s']
                print(f"  ✗ SNRS: {snr['snr']:.2f} < {snr['threshold']} (RMS_signal={snr['rms_signal']:.4f}, RMS_noise={snr['rms_noise']:.4f})")
    
    print("\n" + "="*80)

# ===============================================================================================
# ========================== Monotonicity Violation Analysis ===================================
# ===============================================================================================

def analyze_monotonicity_violations(df_meta_stations, phase='p'):
    """
    Analyze all monotonicity violations for a given phase.
    
    Creates a detailed report showing:
    - Stations with violations
    - Their neighbors in the distance-sorted list
    - Distances and arrival times
    - Theoretical vs detected times
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata with columns:
        'STATION_CODE', 'EPICENTRAL_DISTANCE_KM', 
        't_p_detected', 't_s_detected', 't_p_theo', 't_s_theo',
        'p_residual', 's_residual'
    phase : str
        'p' or 's'
    
    Returns
    -------
    pd.DataFrame
        Detailed violation report with columns:
        - station: station code
        - distance_km: epicentral distance
        - t_detected: detected arrival time
        - t_theo: theoretical arrival time
        - residual: t_detected - t_theo
        - prev_station: previous station code
        - prev_distance: previous station distance
        - prev_t_detected: previous station time
        - prev_residual: previous station residual
        - next_station: next station code
        - next_distance: next station distance
        - next_t_detected: next station time
        - next_residual: next station residual
        - violation_type: 'prev', 'next', 'prev+next'
    
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, phase='p')
    >>> print(f"Found {len(violations_p)} P-wave violations")
    >>> 
    >>> # Most problematic station
    >>> worst = violations_p.iloc[violations_p['residual'].abs().argmax()]
    >>> print(f"Worst: {worst['station']} with residual {worst['residual']:.3f}s")
    """
    
    df_sorted = df_meta_stations.sort_values('EPICENTRAL_DISTANCE_KM').reset_index(drop=True)
    
    violations = []
    
    for idx in range(len(df_sorted)):
        station = df_sorted.loc[idx, 'STATION_CODE']
        distance = df_sorted.loc[idx, 'EPICENTRAL_DISTANCE_KM']
        t_detected = df_sorted.loc[idx, f't_{phase}_detected']
        t_theo = df_sorted.loc[idx, f't_{phase}_theo']
        residual = df_sorted.loc[idx, f'{phase}_residual']
        
        prev_station = df_sorted.loc[idx - 1, 'STATION_CODE'] if idx > 0 else None
        prev_distance = df_sorted.loc[idx - 1, 'EPICENTRAL_DISTANCE_KM'] if idx > 0 else None
        prev_t = df_sorted.loc[idx - 1, f't_{phase}_detected'] if idx > 0 else None
        prev_t_theo = df_sorted.loc[idx - 1, f't_{phase}_theo'] if idx > 0 else None
        prev_residual = df_sorted.loc[idx - 1, f'{phase}_residual'] if idx > 0 else None
        
        next_station = df_sorted.loc[idx + 1, 'STATION_CODE'] if idx < len(df_sorted) - 1 else None
        next_distance = df_sorted.loc[idx + 1, 'EPICENTRAL_DISTANCE_KM'] if idx < len(df_sorted) - 1 else None
        next_t = df_sorted.loc[idx + 1, f't_{phase}_detected'] if idx < len(df_sorted) - 1 else None
        next_t_theo = df_sorted.loc[idx + 1, f't_{phase}_theo'] if idx < len(df_sorted) - 1 else None
        next_residual = df_sorted.loc[idx + 1, f'{phase}_residual'] if idx < len(df_sorted) - 1 else None
        
        violation_prev = (prev_t is not None) and (prev_t >= t_detected)
        violation_next = (next_t is not None) and (next_t <= t_detected)
        
        if violation_prev or violation_next:
            violation_type = []
            if violation_prev:
                violation_type.append('prev')
            if violation_next:
                violation_type.append('next')
            
            violations.append({
                'station': station,
                'distance_km': distance,
                't_detected': t_detected,
                't_theo': t_theo,
                'residual': residual,
                'prev_station': prev_station,
                'prev_distance': prev_distance,
                'prev_t_detected': prev_t,
                'prev_residual': prev_residual,
                'next_station': next_station,
                'next_distance': next_distance,
                'next_t_detected': next_t,
                'next_residual': next_residual,
                'violation_type': '+'.join(violation_type)
            })
    
    return pd.DataFrame(violations)


def print_violation_summary(df_violations, phase='p'):
    """
    Print human-readable summary of monotonicity violations.
    
    Parameters
    ----------
    df_violations : pd.DataFrame
        Output from analyze_monotonicity_violations()
    phase : str
        'p' or 's'
        
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, 'p')
    >>> print_violation_summary(violations_p, 'p')
    """
    
    n_violations = len(df_violations)
    
    if n_violations == 0:
        print("="*80)
        print(f"MONOTONICITY VIOLATIONS - {phase.upper()}-WAVE")
        print("="*80)
        print("No violations found!")
        print("="*80)
        return
    
    n_prev_only = len(df_violations[df_violations['violation_type'] == 'prev'])
    n_next_only = len(df_violations[df_violations['violation_type'] == 'next'])
    n_both = len(df_violations[df_violations['violation_type'] == 'prev+next'])
    
    print("="*80)
    print(f"MONOTONICITY VIOLATIONS - {phase.upper()}-WAVE")
    print("="*80)
    print(f"Total violations: {n_violations}")
    print(f"  Violations with previous station: {n_prev_only + n_both}")
    print(f"  Violations with next station: {n_next_only + n_both}")
    print(f"  Violations with both neighbors: {n_both}")
    print("="*80)
    
    print("\nDETAILED VIOLATIONS:")
    print("-"*80)
    
    for idx, row in df_violations.iterrows():
        print(f"\n[{idx+1}] {row['station']} (distance: {row['distance_km']:.2f} km)")
        print(f"    Detected: {row['t_detected']:.3f}s | Theoretical: {row['t_theo']:.3f}s | Residual: {row['residual']:.3f}s")
        
        if 'prev' in row['violation_type']:
            print(f"    ⚠ PREV VIOLATION:")
            print(f"       {row['prev_station']} (d={row['prev_distance']:.2f}km) arrived at {row['prev_t_detected']:.3f}s")
            print(f"       → Closer station arrived LATER or same time (Δt = {row['prev_t_detected'] - row['t_detected']:+.3f}s)")
            if row['prev_residual'] is not None:
                print(f"       → Residuals: prev={row['prev_residual']:+.3f}s, this={row['residual']:+.3f}s")
        
        if 'next' in row['violation_type']:
            print(f"    ⚠ NEXT VIOLATION:")
            print(f"       {row['next_station']} (d={row['next_distance']:.2f}km) arrived at {row['next_t_detected']:.3f}s")
            print(f"       → Farther station arrived EARLIER or same time (Δt = {row['next_t_detected'] - row['t_detected']:+.3f}s)")
            if row['next_residual'] is not None:
                print(f"       → Residuals: this={row['residual']:+.3f}s, next={row['next_residual']:+.3f}s")
    
    print("\n" + "="*80)


def plot_monotonicity_analysis(df_meta_stations, df_violations_p=None, df_violations_s=None,
                               figsize=(16, 6), output_path=None):
    """
    Plot distance vs arrival time showing monotonicity violations.
    
    Creates a 2-panel plot:
    - Left: P-wave arrivals vs distance
    - Right: S-wave arrivals vs distance
    
    Shows detected times, theoretical times, and highlights violations.
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata
    df_violations_p : pd.DataFrame, optional
        P-wave violations from analyze_monotonicity_violations()
    df_violations_s : pd.DataFrame, optional
        S-wave violations from analyze_monotonicity_violations()
    figsize : tuple
        Figure size (default: (16, 6))
    output_path : str or Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, 'p')
    >>> violations_s = analyze_monotonicity_violations(df_meta_stations, 's')
    >>> fig = plot_monotonicity_analysis(df_meta_stations, violations_p, violations_s)
    >>> plt.show()
    """
    
    import matplotlib.pyplot as plt
    
    df_sorted = df_meta_stations.sort_values('EPICENTRAL_DISTANCE_KM')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for idx, (phase, df_viol, ax) in enumerate([
        ('p', df_violations_p, axes[0]),
        ('s', df_violations_s, axes[1])
    ]):
        ax.scatter(df_sorted['EPICENTRAL_DISTANCE_KM'],
                   df_sorted[f't_{phase}_detected'],
                   label='Detected', alpha=0.7, s=50, color='steelblue')
        
        ax.plot(df_sorted['EPICENTRAL_DISTANCE_KM'],
                df_sorted[f't_{phase}_theo'],
                'r--', label='Theoretical', linewidth=2, alpha=0.7)
        
        if df_viol is not None and len(df_viol) > 0:
            violation_stations = df_viol['station'].values
            df_viol_plot = df_sorted[df_sorted['STATION_CODE'].isin(violation_stations)]
            
            ax.scatter(df_viol_plot['EPICENTRAL_DISTANCE_KM'],
                       df_viol_plot[f't_{phase}_detected'],
                       color='red', s=150, marker='x', linewidth=3,
                       label=f'Violations ({len(df_viol)})', zorder=5)
        
        ax.set_xlabel('Epicentral Distance (km)', fontsize=12)
        ax.set_ylabel(f'{phase.upper()}-wave arrival time (s)', fontsize=12)
        ax.set_title(f'{phase.upper()}-wave Monotonicity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def analyze_residuals_vs_violations(df_meta_stations, df_violations_p, df_violations_s,
                                    figsize=(16, 10)):
    """
    Analyze relationship between residuals and violations.
    
    Creates plots showing:
    - Residual distributions for stations with/without violations
    - Residuals vs distance
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata
    df_violations_p : pd.DataFrame
        P-wave violations
    df_violations_s : pd.DataFrame
        S-wave violations
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, 'p')
    >>> violations_s = analyze_monotonicity_violations(df_meta_stations, 's')
    >>> fig = analyze_residuals_vs_violations(df_meta_stations, violations_p, violations_s)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for col_idx, (phase, df_viol) in enumerate([('p', df_violations_p), ('s', df_violations_s)]):
        
        violation_stations = set(df_viol['station'].values) if len(df_viol) > 0 else set()
        df_meta_stations_copy = df_meta_stations.copy()
        df_meta_stations_copy['has_violation'] = df_meta_stations_copy['STATION_CODE'].isin(violation_stations)
        
        residuals_ok = df_meta_stations_copy[~df_meta_stations_copy['has_violation']][f'{phase}_residual']
        residuals_viol = df_meta_stations_copy[df_meta_stations_copy['has_violation']][f'{phase}_residual']
        
        ax_hist = axes[0, col_idx]
        ax_hist.hist(residuals_ok, bins=15, alpha=0.6, label='No violation',
                    edgecolor='black', color='steelblue')
        ax_hist.hist(residuals_viol, bins=15, alpha=0.6, label='Violation',
                    edgecolor='black', color='red')
        ax_hist.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax_hist.set_xlabel(f'{phase.upper()}-wave residual (s)', fontsize=11)
        ax_hist.set_ylabel('Count', fontsize=11)
        ax_hist.set_title(f'{phase.upper()}-wave: Residual distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)
        
        ax_scatter = axes[1, col_idx]
        ax_scatter.scatter(df_meta_stations_copy[~df_meta_stations_copy['has_violation']]['EPICENTRAL_DISTANCE_KM'],
                          df_meta_stations_copy[~df_meta_stations_copy['has_violation']][f'{phase}_residual'],
                          alpha=0.6, s=50, label='No violation', color='steelblue')
        ax_scatter.scatter(df_meta_stations_copy[df_meta_stations_copy['has_violation']]['EPICENTRAL_DISTANCE_KM'],
                          df_meta_stations_copy[df_meta_stations_copy['has_violation']][f'{phase}_residual'],
                          alpha=0.8, s=100, marker='x', linewidth=2, label='Violation', color='red')
        ax_scatter.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax_scatter.set_xlabel('Epicentral Distance (km)', fontsize=11)
        ax_scatter.set_ylabel(f'{phase.upper()}-wave residual (s)', fontsize=11)
        ax_scatter.set_title(f'{phase.upper()}-wave: Residuals vs Distance', fontsize=12, fontweight='bold')
        ax_scatter.legend(fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig