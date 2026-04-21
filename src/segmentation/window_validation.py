"""
Quality control functions for seismic signal windowing.

This module provides validation functions to check:
- PGA location (should occur in S-wave window)
- Arrival time monotonicity with epicentral distance
- Signal-to-noise ratio (SNR) for onset picks
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple

def check_pga_in_s_wave(windowed_signals, station, component):
    """
    Check if PGA occurs in S-wave window.
    
    Returns
    -------
    dict
        {
            'passed': bool,
            'pga_window': str,
            'pga_value': float,
            'pga_values_all': dict  # for debugging
        }
    """
    
    windows = windowed_signals[station][component]
    
    # Find PGA in each window
    pga_values = {}
    for window_name in ['pre_event', 'p_wave', 's_wave', 'coda']:
        signal = windows[window_name]['signal']
        pga_values[window_name] = np.max(np.abs(signal))
    
    # Which window has global PGA?
    pga_window = max(pga_values, key=pga_values.get)
    
    return {
        'passed': pga_window == 's_wave',
        'pga_window': pga_window,
        'pga_value': pga_values[pga_window],
        'pga_values_all': pga_values
    }


def check_monotonicity_station(df_onsets, station, component, phase='p'):
    """
    Check if arrival time is monotonic with distance for this station.
    
    Logic:
    - Sort stations by epicentral distance
    - For station i: check t_p[i-1] < t_p[i] < t_p[i+1]
    
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
    
    # Group by station (average t_p across components)
    df_stations = df_onsets.groupby('STATION_CODE').agg({
        'EPICENTRAL_DISTANCE_KM': 'first',
        f't_{phase}_detected': 'mean'
    }).reset_index()
    
    # Sort by distance
    df_stations = df_stations.sort_values('EPICENTRAL_DISTANCE_KM').reset_index(drop=True)
    
    # Find position of this station
    idx = df_stations[df_stations['STATION_CODE'] == station].index[0]
    n_stations = len(df_stations)
    
    t_this = df_stations.loc[idx, f't_{phase}_detected']
    
    # Get neighbors
    t_prev = df_stations.loc[idx - 1, f't_{phase}_detected'] if idx > 0 else None
    t_next = df_stations.loc[idx + 1, f't_{phase}_detected'] if idx < n_stations - 1 else None
    
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


def check_snr(signals_dict, df_onsets, station, component, 
              phase='p', threshold=3.0,
              noise_duration=5.0, signal_duration=5.0):
    """
    Check SNR for P or S pick.
    
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
    
    # Get signal and time
    signal = signals_dict[station][component]
    time = signals_dict[station]['time']
    
    # Get onset time
    row = df_onsets[(df_onsets['STATION_CODE'] == station) & 
                    (df_onsets['COMPONENT'] == component)].iloc[0]
    onset_time = row[f't_{phase}_detected']
    
    # Define windows
    noise_start = onset_time - noise_duration
    noise_end = onset_time
    signal_start = onset_time
    signal_end = onset_time + signal_duration
    
    # Extract windows
    mask_noise = (time >= noise_start) & (time < noise_end)
    mask_signal = (time >= signal_start) & (time < signal_end)
    
    noise_window = signal[mask_noise]
    signal_window = signal[mask_signal]
    
    # Check validity
    if len(noise_window) == 0 or len(signal_window) == 0:
        return {
            'passed': False,
            'snr': np.nan,
            'rms_signal': np.nan,
            'rms_noise': np.nan,
            'threshold': threshold
        }
    
    # Compute RMS
    rms_noise = np.sqrt(np.mean(noise_window**2))
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

def quality_control_all_stations(signals_dict, windowed_signals, df_onsets,
                                 snr_threshold=3.0):
    """
    Run all quality checks for all stations and components.
    
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
            
            # Run all checks
            pga_check = check_pga_in_s_wave(windowed_signals, station, component)
            
            mono_p = check_monotonicity_station(df_onsets, station, component, phase='p')
            mono_s = check_monotonicity_station(df_onsets, station, component, phase='s')
            
            snr_p = check_snr(signals_dict, df_onsets, station, component, 
                             phase='p', threshold=snr_threshold)
            snr_s = check_snr(signals_dict, df_onsets, station, component,
                             phase='s', threshold=snr_threshold)
            
            # Aggregate
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