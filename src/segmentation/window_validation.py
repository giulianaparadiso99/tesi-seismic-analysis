"""
Quality control functions for seismic signal windowing.

This module provides validation functions to check:
- PGA location (should occur in S-wave window)
- Arrival time monotonicity with epicentral distance
- Signal-to-noise ratio (SNR) for onset picks
"""

import numpy as np
import pandas as pd

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