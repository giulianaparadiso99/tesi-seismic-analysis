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

def check_pga_in_s_wave(df_metadata, station, component, coda_method='rautian', 
                        sampling_rate=200):
    """
    Check if PGA occurs in S-wave window using only metadata.
    
    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata with onset times and PGA info
        Expected columns (dual representation with auto-detection):
        - t_p_detected_seconds, t_p_detected_samples (or legacy t_p_detected)
        - t_s_detected_seconds, t_s_detected_samples (or legacy t_s_detected)
        - t_coda_<method>_seconds, t_coda_<method>_samples (or legacy t_coda_<method>)
        - TIME_PGA_S, PGA_CM/S^2, STATION_CODE, COMPONENT
    station : str
        Station code
    component : str
        Component name
    coda_method : str, optional
        Which coda method to use: 'rautian', 'arias', 'envelope', 'median'
        (default: 'rautian')
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
        Only used if converting samples to seconds
    
    Returns
    -------
    dict
        {
            'passed': bool,
            'pga_window': str,
            'pga_value': float,
            'pga_time': float
        }
    
    Notes
    -----
    All times are compared in seconds for physical interpretation.
    Auto-detects whether input columns use _seconds, _samples, or legacy naming.
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
    
    # PGA time is always in seconds (from raw data)
    pga_time = row['TIME_PGA_S']
    pga_value = row['PGA_CM/S^2']
    
    # ===== AUTO-DETECT t_p (prefer seconds, fallback samples, then legacy) =====
    if 't_p_detected_seconds' in df_metadata.columns:
        t_p = row['t_p_detected_seconds']
    elif 't_p_detected_samples' in df_metadata.columns:
        t_p = row['t_p_detected_samples'] / sampling_rate
    elif 't_p_detected' in df_metadata.columns:
        t_p = row['t_p_detected']
    else:
        raise ValueError(
            "No t_p_detected column found. Expected 't_p_detected_seconds', "
            "'t_p_detected_samples', or 't_p_detected'"
        )
    
    # ===== AUTO-DETECT t_s (prefer seconds, fallback samples, then legacy) =====
    if 't_s_detected_seconds' in df_metadata.columns:
        t_s = row['t_s_detected_seconds']
    elif 't_s_detected_samples' in df_metadata.columns:
        t_s = row['t_s_detected_samples'] / sampling_rate
    elif 't_s_detected' in df_metadata.columns:
        t_s = row['t_s_detected']
    else:
        raise ValueError(
            "No t_s_detected column found. Expected 't_s_detected_seconds', "
            "'t_s_detected_samples', or 't_s_detected'"
        )
    
    # ===== AUTO-DETECT t_coda (prefer seconds, fallback samples, then legacy) =====
    coda_col_seconds = f't_coda_{coda_method}_seconds'
    coda_col_samples = f't_coda_{coda_method}_samples'
    coda_col_legacy = f't_coda_{coda_method}'
    
    if coda_col_seconds in df_metadata.columns:
        t_coda = row[coda_col_seconds]
    elif coda_col_samples in df_metadata.columns:
        t_coda = row[coda_col_samples] / sampling_rate
    elif coda_col_legacy in df_metadata.columns:
        t_coda = row[coda_col_legacy]
    else:
        raise ValueError(
            f"No t_coda column found for method '{coda_method}'. "
            f"Expected '{coda_col_seconds}', '{coda_col_samples}', or '{coda_col_legacy}'"
        )
    
    # Determine window (all times now in seconds)
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

def check_monotonicity_station(df_meta_stations, station, phase='p', sampling_rate=200):
    """
    Check if arrival time is monotonic with distance for this station.
    
    Validates that detected arrival times follow the expected ordering based
    on hypocentral distance: stations closer to the hypocenter should have
    earlier arrival times.
    
    Logic:
    - Stations are sorted by hypocentral distance (3D distance from hypocenter)
    - For station i: verify t[i-1] < t[i] < t[i+1]
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata (one row per station, not per component)
        Must contain: 
        - 'STATION_CODE': Station identifier
        - 'hypocentral_distance_km': 3D distance from hypocenter to station
        - Detected arrivals (dual representation with auto-detection):
          * t_p_detected_seconds, t_p_detected_samples (or legacy t_p_detected)
          * t_s_detected_seconds, t_s_detected_samples (or legacy t_s_detected)
    station : str
        Station code to validate
    phase : str, optional
        Phase to validate: 'p' or 's' (default: 'p')
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
        Only used if converting samples to seconds
    
    Returns
    -------
    dict
        Validation results containing:
        - 'passed': bool - True if monotonicity check passed
        - 'position': int - Position in distance-sorted list (0 = closest)
        - 'n_stations': int - Total number of stations
        - 't_prev': float or None - Arrival time at previous (closer) station (seconds)
        - 't_this': float - Arrival time at this station (seconds)
        - 't_next': float or None - Arrival time at next (farther) station (seconds)
        - 'violation_side': str or None - 'prev', 'next', or None
        - 'd_prev': float or None - Distance of previous station (km)
        - 'd_this': float - Distance of this station (km)
        - 'd_next': float or None - Distance of next station (km)
    
    Notes
    -----
    Monotonicity validation assumes:
    - Seismic waves propagate outward from hypocenter
    - Farther stations receive arrivals later than closer ones
    - Hypocentral distance (not epicentral) determines arrival order
    
    All times are compared in seconds for physical interpretation.
    Auto-detects whether input columns use _seconds, _samples, or legacy naming.
    
    Violations may indicate:
    - Picking errors (misidentified phases)
    - Strong lateral velocity heterogeneities
    - Complex wave propagation paths (refraction, scattering)
    
    Examples
    --------
    >>> result = check_monotonicity_station(df_meta_stations, 'ABC', phase='p')
    >>> if not result['passed']:
    ...     print(f"Monotonicity violation on {result['violation_side']} side")
    ...     print(f"This station: t={result['t_this']:.2f}s, d={result['d_this']:.2f}km")
    """
    # Validate phase input
    if phase not in ['p', 's']:
        raise ValueError(f"phase must be 'p' or 's', got '{phase}'")
    
    # ===== AUTO-DETECT COLUMN NAMES =====
    # Determine which detected arrival column to use
    col_seconds = f't_{phase}_detected_seconds'
    col_samples = f't_{phase}_detected_samples'
    col_legacy = f't_{phase}_detected'
    
    if col_seconds in df_meta_stations.columns:
        time_col = col_seconds
        needs_conversion = False
    elif col_samples in df_meta_stations.columns:
        time_col = col_samples
        needs_conversion = True
    elif col_legacy in df_meta_stations.columns:
        time_col = col_legacy
        needs_conversion = False  # Assume legacy is seconds
    else:
        raise ValueError(
            f"No {phase}-wave detected arrival column found. "
            f"Expected '{col_seconds}', '{col_samples}', or '{col_legacy}'"
        )
    
    # Validate required columns
    required_cols = ['STATION_CODE', 'hypocentral_distance_km', time_col]
    missing = [col for col in required_cols if col not in df_meta_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Ensure add_theoretical_arrivals() and phase detection have been run."
        )
    
    if station not in df_meta_stations['STATION_CODE'].values:
        raise ValueError(f"Station '{station}' not found in DataFrame")
    
    # Sort by hypocentral distance (physical arrival order)
    df_sorted = df_meta_stations.sort_values('hypocentral_distance_km').reset_index(drop=True)
    
    # Find position of this station
    idx = df_sorted[df_sorted['STATION_CODE'] == station].index[0]
    n_stations = len(df_sorted)
    
    # ===== GET ARRIVAL TIMES (convert to seconds if needed) =====
    def get_time_seconds(row_idx):
        """Extract time in seconds from row, converting if necessary."""
        if row_idx is None:
            return None
        t = df_sorted.loc[row_idx, time_col]
        if pd.isna(t):
            return None
        if needs_conversion:
            return float(t) / sampling_rate
        else:
            return float(t)
    
    # Get arrival time and distance for this station
    t_this = get_time_seconds(idx)
    d_this = df_sorted.loc[idx, 'hypocentral_distance_km']
    
    # Get neighbors (previous = closer, next = farther)
    if idx > 0:
        t_prev = get_time_seconds(idx - 1)
        d_prev = df_sorted.loc[idx - 1, 'hypocentral_distance_km']
    else:
        t_prev = None
        d_prev = None
    
    if idx < n_stations - 1:
        t_next = get_time_seconds(idx + 1)
        d_next = df_sorted.loc[idx + 1, 'hypocentral_distance_km']
    else:
        t_next = None
        d_next = None
    
    # Check monotonicity: arrival times should increase with distance
    # t_prev < t_this < t_next
    violation_prev = (t_prev is not None) and (t_prev >= t_this)
    violation_next = (t_next is not None) and (t_next <= t_this)
    
    passed = not (violation_prev or violation_next)
    
    # Determine which side has violation
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
        'd_prev': d_prev,
        'd_this': d_this,
        'd_next': d_next,
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
    
    Performs comprehensive validation including PGA timing, monotonicity
    with distance, and signal-to-noise ratio checks.
    
    Parameters
    ----------
    windowed_signals : dict
        Segmented signals from segment_all_signals()
        Structure: {station: {component: {'pre_arrival': array, 'p_wave': array, ...}}}
    df_full : pd.DataFrame
        Component-level metadata (used for PGA check)
        Must contain: 'STATION_CODE', 'COMPONENT', 'PGA_CM/S^2', 'TIME_PGA_S',
                      't_p_detected', 't_s_detected', 't_coda_{method}'
    df_meta_stations : pd.DataFrame
        Station-level metadata (used for monotonicity check)
        Must contain: 'STATION_CODE', 'hypocentral_distance_km',
                      't_p_detected', 't_s_detected'
    snr_threshold : float, optional
        SNR threshold for quality check (default: 3.0)
    coda_method : str, optional
        Coda detection method: 'rautian', 'arias', 'envelope', 'median'
        (default: 'rautian')
    
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
    
    Notes
    -----
    Monotonicity checks validate arrival time ordering based on hypocentral
    distance (3D distance from hypocenter), not epicentral distance. This
    accounts for the actual wave propagation path from the source.
    
    The monotonicity check is station-level (same result for all components
    of a station), while PGA and SNR checks are component-specific.
    
    Examples
    --------
    >>> qc_results = quality_control_all_stations(
    ...     windowed_signals, 
    ...     df_full, 
    ...     df_meta_stations,
    ...     snr_threshold=3.0
    ... )
    >>> # Count passing stations
    >>> n_pass = sum(
    ...     results[sta][comp]['all_passed']
    ...     for sta in results for comp in results[sta]
    ... )
    >>> print(f"{n_pass} components passed all checks")
    """
    # Validate inputs
    if 'hypocentral_distance_km' not in df_meta_stations.columns:
        raise ValueError(
            "df_meta_stations must contain 'hypocentral_distance_km'. "
            "Run add_theoretical_arrivals() first."
        )
    
    results = {}
    
    # Cache monotonicity results to avoid redundant computation
    # (same for all components of a station)
    monotonicity_cache = {}
    
    for station in windowed_signals.keys():
        results[station] = {}
        
        # Compute monotonicity once per station (not per component)
        if station not in monotonicity_cache:
            monotonicity_cache[station] = {
                'p': check_monotonicity_station(df_meta_stations, station, phase='p'),
                's': check_monotonicity_station(df_meta_stations, station, phase='s')
            }
        
        for component in windowed_signals[station].keys():
            # PGA check (component-specific, uses df_full)
            pga_check = check_pga_in_s_wave(
                df_full, station, component, coda_method=coda_method
            )
            
            # Monotonicity checks (station-level, cached)
            mono_p = monotonicity_cache[station]['p']
            mono_s = monotonicity_cache[station]['s']
            
            # SNR checks (component-specific, uses windowed_signals)
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

def analyze_monotonicity_violations(df_meta_stations, phase='p', sampling_rate=200):
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
        - 'STATION_CODE', 'hypocentral_distance_km'
        - Detected arrivals (dual representation with auto-detection):
          * t_p_detected_seconds, t_p_detected_samples (or legacy t_p_detected)
          * t_s_detected_seconds, t_s_detected_samples (or legacy t_s_detected)
        - Theoretical arrivals:
          * t_p_theo_seconds, t_p_theo_samples (or legacy t_p_theo)
          * t_s_theo_seconds, t_s_theo_samples (or legacy t_s_theo)
        - Residuals:
          * p_residual_seconds, s_residual_seconds (or legacy p_residual, s_residual)
    phase : str
        'p' or 's'
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
        Only used if converting samples to seconds
    
    Returns
    -------
    pd.DataFrame
        Detailed violation report with columns (all times in seconds):
        - station: station code
        - distance_km: hypocentral distance
        - t_detected: detected arrival time (s)
        - t_theo: theoretical arrival time (s)
        - residual: t_detected - t_theo (s)
        - prev_station: previous station code
        - prev_distance: previous station hypocentral distance
        - prev_t_detected: previous station time (s)
        - prev_residual: previous station residual (s)
        - next_station: next station code
        - next_distance: next station hypocentral distance
        - next_t_detected: next station time (s)
        - next_residual: next station residual (s)
        - violation_type: 'prev', 'next', 'prev+next'
    
    Notes
    -----
    All times are reported in seconds for physical interpretation.
    Auto-detects whether input columns use _seconds, _samples, or legacy naming.
    
    Stations are sorted by hypocentral distance (3D distance from hypocenter)
    to reflect the actual wave propagation path. Monotonicity violations may
    indicate picking errors, strong lateral velocity heterogeneities, or
    complex wave propagation effects (refraction, scattering).
    
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, phase='p')
    >>> print(f"Found {len(violations_p)} P-wave violations")
    >>> 
    >>> # Most problematic station
    >>> if len(violations_p) > 0:
    ...     worst = violations_p.iloc[violations_p['residual'].abs().argmax()]
    ...     print(f"Worst: {worst['station']} with residual {worst['residual']:.3f}s")
    >>> 
    >>> # Violations by type
    >>> print(violations_p['violation_type'].value_counts())
    """
    # Validate phase
    if phase not in ['p', 's']:
        raise ValueError(f"phase must be 'p' or 's', got '{phase}'")
    
    # ===== AUTO-DETECT COLUMN NAMES =====
    # Detected arrival time
    det_col_seconds = f't_{phase}_detected_seconds'
    det_col_samples = f't_{phase}_detected_samples'
    det_col_legacy = f't_{phase}_detected'
    
    if det_col_seconds in df_meta_stations.columns:
        det_col = det_col_seconds
        det_needs_conversion = False
    elif det_col_samples in df_meta_stations.columns:
        det_col = det_col_samples
        det_needs_conversion = True
    elif det_col_legacy in df_meta_stations.columns:
        det_col = det_col_legacy
        det_needs_conversion = False
    else:
        raise ValueError(
            f"No {phase}-wave detected arrival column found. "
            f"Expected '{det_col_seconds}', '{det_col_samples}', or '{det_col_legacy}'"
        )
    
    # Theoretical arrival time
    theo_col_seconds = f't_{phase}_theo_seconds'
    theo_col_samples = f't_{phase}_theo_samples'
    theo_col_legacy = f't_{phase}_theo'
    
    if theo_col_seconds in df_meta_stations.columns:
        theo_col = theo_col_seconds
        theo_needs_conversion = False
    elif theo_col_samples in df_meta_stations.columns:
        theo_col = theo_col_samples
        theo_needs_conversion = True
    elif theo_col_legacy in df_meta_stations.columns:
        theo_col = theo_col_legacy
        theo_needs_conversion = False
    else:
        raise ValueError(
            f"No {phase}-wave theoretical arrival column found. "
            f"Expected '{theo_col_seconds}', '{theo_col_samples}', or '{theo_col_legacy}'"
        )
    
    # Residual
    res_col_seconds = f'{phase}_residual_seconds'
    res_col_legacy = f'{phase}_residual'
    
    if res_col_seconds in df_meta_stations.columns:
        res_col = res_col_seconds
    elif res_col_legacy in df_meta_stations.columns:
        res_col = res_col_legacy
    else:
        raise ValueError(
            f"No {phase}-wave residual column found. "
            f"Expected '{res_col_seconds}' or '{res_col_legacy}'"
        )
    
    # Validate required columns
    required_cols = ['STATION_CODE', 'hypocentral_distance_km', det_col, theo_col, res_col]
    missing = [col for col in required_cols if col not in df_meta_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Ensure add_theoretical_arrivals() and phase detection have been run."
        )
    
    # Sort by hypocentral distance (physical arrival order)
    df_sorted = df_meta_stations.sort_values('hypocentral_distance_km').reset_index(drop=True)
    
    # Helper function to get time in seconds
    def get_time_seconds(value, needs_conversion):
        """Convert time to seconds if necessary."""
        if pd.isna(value):
            return None
        if needs_conversion:
            return float(value) / sampling_rate
        else:
            return float(value)
    
    violations = []
    
    for idx in range(len(df_sorted)):
        station = df_sorted.loc[idx, 'STATION_CODE']
        distance = df_sorted.loc[idx, 'hypocentral_distance_km']
        
        t_detected = get_time_seconds(df_sorted.loc[idx, det_col], det_needs_conversion)
        t_theo = get_time_seconds(df_sorted.loc[idx, theo_col], theo_needs_conversion)
        residual = df_sorted.loc[idx, res_col]  # Residuals always in seconds
        
        # Previous station (closer to hypocenter)
        if idx > 0:
            prev_station = df_sorted.loc[idx - 1, 'STATION_CODE']
            prev_distance = df_sorted.loc[idx - 1, 'hypocentral_distance_km']
            prev_t = get_time_seconds(df_sorted.loc[idx - 1, det_col], det_needs_conversion)
            prev_t_theo = get_time_seconds(df_sorted.loc[idx - 1, theo_col], theo_needs_conversion)
            prev_residual = df_sorted.loc[idx - 1, res_col]
        else:
            prev_station = None
            prev_distance = None
            prev_t = None
            prev_t_theo = None
            prev_residual = None
        
        # Next station (farther from hypocenter)
        if idx < len(df_sorted) - 1:
            next_station = df_sorted.loc[idx + 1, 'STATION_CODE']
            next_distance = df_sorted.loc[idx + 1, 'hypocentral_distance_km']
            next_t = get_time_seconds(df_sorted.loc[idx + 1, det_col], det_needs_conversion)
            next_t_theo = get_time_seconds(df_sorted.loc[idx + 1, theo_col], theo_needs_conversion)
            next_residual = df_sorted.loc[idx + 1, res_col]
        else:
            next_station = None
            next_distance = None
            next_t = None
            next_t_theo = None
            next_residual = None
        
        # Check monotonicity: t_prev < t_this < t_next
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
    
    df_violations = pd.DataFrame(violations)
    
    # Print summary
    if len(df_violations) > 0:
        print(f"\nMonotonicity violations for {phase.upper()}-wave:")
        print(f"  Total violations: {len(df_violations)}/{len(df_sorted)} stations")
        print(f"  Violation types:")
        print(df_violations['violation_type'].value_counts().to_string())
        print(f"\n  Most problematic stations (by |residual|):")
        
        # Sort by absolute residual and show top 3
        top3 = df_violations.nlargest(3, df_violations['residual'].abs().values)[
            ['station', 'distance_km', 't_detected', 't_theo', 'residual']
        ]
        print(top3.to_string(index=False))
    else:
        print(f"\nNo monotonicity violations found for {phase.upper()}-wave")
    
    return df_violations

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
                               figsize=(16, 6), output_path=None, sampling_rate=200):
    """
    Plot distance vs arrival time showing monotonicity violations.
    
    Creates a 2-panel plot:
    - Left: P-wave arrivals vs hypocentral distance
    - Right: S-wave arrivals vs hypocentral distance
    
    Shows detected times, theoretical times, and highlights violations.
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata with columns:
        - 'hypocentral_distance_km', 'STATION_CODE'
        - Detected arrivals (dual representation with auto-detection):
          * t_p_detected_seconds, t_p_detected_samples (or legacy t_p_detected)
          * t_s_detected_seconds, t_s_detected_samples (or legacy t_s_detected)
        - Theoretical arrivals (optional):
          * t_p_theo_seconds, t_p_theo_samples (or legacy t_p_theo)
          * t_s_theo_seconds, t_s_theo_samples (or legacy t_s_theo)
    df_violations_p : pd.DataFrame, optional
        P-wave violations from analyze_monotonicity_violations()
    df_violations_s : pd.DataFrame, optional
        S-wave violations from analyze_monotonicity_violations()
    figsize : tuple, optional
        Figure size (default: (16, 6))
    output_path : str or Path, optional
        If provided, save figure to this path
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
        Only used if converting samples to seconds for plotting
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    
    Notes
    -----
    All times are plotted in seconds for physical interpretation.
    Auto-detects whether input columns use _seconds, _samples, or legacy naming.
    
    Plots use hypocentral distance (3D distance from hypocenter to station)
    on the x-axis, reflecting the actual wave propagation path. This provides
    a more physically meaningful representation than epicentral distance,
    especially for near-field stations.
    
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, 'p')
    >>> violations_s = analyze_monotonicity_violations(df_meta_stations, 's')
    >>> fig = plot_monotonicity_analysis(df_meta_stations, violations_p, violations_s)
    >>> plt.show()
    >>> 
    >>> # Save to file
    >>> fig = plot_monotonicity_analysis(
    ...     df_meta_stations, 
    ...     violations_p, 
    ...     violations_s,
    ...     output_path='figures/monotonicity_analysis.png'
    ... )
    """
    import matplotlib.pyplot as plt
    
    # Validate input
    required_cols = ['hypocentral_distance_km', 'STATION_CODE']
    missing = [col for col in required_cols if col not in df_meta_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run add_theoretical_arrivals() first."
        )
    
    # Sort by hypocentral distance
    df_sorted = df_meta_stations.sort_values('hypocentral_distance_km')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for idx, (phase, df_viol, ax) in enumerate([
        ('p', df_violations_p, axes[0]),
        ('s', df_violations_s, axes[1])
    ]):
        # ===== AUTO-DETECT DETECTED ARRIVAL COLUMN =====
        det_col_seconds = f't_{phase}_detected_seconds'
        det_col_samples = f't_{phase}_detected_samples'
        det_col_legacy = f't_{phase}_detected'
        
        detected_col = None
        detected_needs_conversion = False
        
        if det_col_seconds in df_sorted.columns:
            detected_col = det_col_seconds
        elif det_col_samples in df_sorted.columns:
            detected_col = det_col_samples
            detected_needs_conversion = True
        elif det_col_legacy in df_sorted.columns:
            detected_col = det_col_legacy
        
        # Check if detected data exists
        if detected_col is None:
            ax.text(0.5, 0.5, f'No {phase.upper()}-wave data available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_xlabel('Hypocentral Distance (km)', fontsize=12)
            ax.set_ylabel(f'{phase.upper()}-wave arrival time (s)', fontsize=12)
            ax.set_title(f'{phase.upper()}-wave Monotonicity', fontsize=13, fontweight='bold')
            continue
        
        # Get detected times in seconds
        if detected_needs_conversion:
            detected_times = df_sorted[detected_col].values / sampling_rate
        else:
            detected_times = df_sorted[detected_col].values
        
        # Plot detected arrivals
        ax.scatter(df_sorted['hypocentral_distance_km'],
                  detected_times,
                  label='Detected', alpha=0.7, s=50, color='steelblue')
        
        # ===== AUTO-DETECT THEORETICAL ARRIVAL COLUMN (OPTIONAL) =====
        theo_col_seconds = f't_{phase}_theo_seconds'
        theo_col_samples = f't_{phase}_theo_samples'
        theo_col_legacy = f't_{phase}_theo'
        
        theo_col = None
        theo_needs_conversion = False
        
        if theo_col_seconds in df_sorted.columns:
            theo_col = theo_col_seconds
        elif theo_col_samples in df_sorted.columns:
            theo_col = theo_col_samples
            theo_needs_conversion = True
        elif theo_col_legacy in df_sorted.columns:
            theo_col = theo_col_legacy
        
        # Plot theoretical arrivals if available
        if theo_col is not None:
            if theo_needs_conversion:
                theo_times = df_sorted[theo_col].values / sampling_rate
            else:
                theo_times = df_sorted[theo_col].values
            
            ax.plot(df_sorted['hypocentral_distance_km'],
                   theo_times,
                   'r--', label='Theoretical', linewidth=2, alpha=0.7)
        
        # Highlight violations
        if df_viol is not None and len(df_viol) > 0:
            violation_stations = df_viol['station'].values
            df_viol_plot = df_sorted[df_sorted['STATION_CODE'].isin(violation_stations)]
            
            # Get violation times in seconds
            if detected_needs_conversion:
                viol_times = df_viol_plot[detected_col].values / sampling_rate
            else:
                viol_times = df_viol_plot[detected_col].values
            
            ax.scatter(df_viol_plot['hypocentral_distance_km'],
                      viol_times,
                      color='red', s=150, marker='x', linewidth=3,
                      label=f'Violations ({len(df_viol)})', zorder=5)
        
        ax.set_xlabel('Hypocentral Distance (km)', fontsize=12)
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
    
    Creates a 2×2 panel plot showing:
    - Top row: Residual distributions for stations with/without violations (P and S)
    - Bottom row: Residuals vs hypocentral distance (P and S)
    
    Parameters
    ----------
    df_meta_stations : pd.DataFrame
        Station-level metadata with columns:
        - 'STATION_CODE', 'hypocentral_distance_km'
        - Residuals (dual representation with auto-detection):
          * p_residual_seconds, s_residual_seconds (or legacy p_residual, s_residual)
    df_violations_p : pd.DataFrame
        P-wave violations from analyze_monotonicity_violations()
    df_violations_s : pd.DataFrame
        S-wave violations from analyze_monotonicity_violations()
    figsize : tuple, optional
        Figure size (default: (16, 10))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    
    Notes
    -----
    Residuals are defined as: residual = t_detected - t_theo
    Positive residuals indicate late arrivals relative to the 1D velocity model.
    
    All residuals are plotted in seconds for physical interpretation.
    Auto-detects whether input columns use _seconds suffix or legacy naming.
    
    Hypocentral distance is used on the x-axis to reflect the actual wave
    propagation distance from source to station.
    
    Examples
    --------
    >>> violations_p = analyze_monotonicity_violations(df_meta_stations, 'p')
    >>> violations_s = analyze_monotonicity_violations(df_meta_stations, 's')
    >>> fig = analyze_residuals_vs_violations(
    ...     df_meta_stations, 
    ...     violations_p, 
    ...     violations_s
    ... )
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    # ===== AUTO-DETECT RESIDUAL COLUMNS =====
    # P-wave residual
    p_res_col_seconds = 'p_residual_seconds'
    p_res_col_legacy = 'p_residual'
    
    if p_res_col_seconds in df_meta_stations.columns:
        p_res_col = p_res_col_seconds
    elif p_res_col_legacy in df_meta_stations.columns:
        p_res_col = p_res_col_legacy
    else:
        raise ValueError(
            f"No P-wave residual column found. "
            f"Expected '{p_res_col_seconds}' or '{p_res_col_legacy}'"
        )
    
    # S-wave residual
    s_res_col_seconds = 's_residual_seconds'
    s_res_col_legacy = 's_residual'
    
    if s_res_col_seconds in df_meta_stations.columns:
        s_res_col = s_res_col_seconds
    elif s_res_col_legacy in df_meta_stations.columns:
        s_res_col = s_res_col_legacy
    else:
        raise ValueError(
            f"No S-wave residual column found. "
            f"Expected '{s_res_col_seconds}' or '{s_res_col_legacy}'"
        )
    
    # Validate required columns
    required_cols = ['STATION_CODE', 'hypocentral_distance_km', p_res_col, s_res_col]
    missing = [col for col in required_cols if col not in df_meta_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Ensure add_theoretical_arrivals() and phase detection have been run."
        )
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Map phase to residual column
    residual_cols = {'p': p_res_col, 's': s_res_col}
    
    for col_idx, (phase, df_viol) in enumerate([('p', df_violations_p), ('s', df_violations_s)]):
        # Get the appropriate residual column for this phase
        res_col = residual_cols[phase]
        
        # Identify violation stations
        violation_stations = set(df_viol['station'].values) if len(df_viol) > 0 else set()
        
        # Create copy and flag violations
        df_meta_stations_copy = df_meta_stations.copy()
        df_meta_stations_copy['has_violation'] = df_meta_stations_copy['STATION_CODE'].isin(violation_stations)
        
        # Separate residuals by violation status
        residuals_ok = df_meta_stations_copy[~df_meta_stations_copy['has_violation']][res_col]
        residuals_viol = df_meta_stations_copy[df_meta_stations_copy['has_violation']][res_col]
        
        # Top row: Histogram of residuals
        ax_hist = axes[0, col_idx]
        ax_hist.hist(residuals_ok, bins=15, alpha=0.6, label='No violation',
                    edgecolor='black', color='steelblue')
        if len(residuals_viol) > 0:
            ax_hist.hist(residuals_viol, bins=15, alpha=0.6, label='Violation',
                        edgecolor='black', color='red')
        ax_hist.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax_hist.set_xlabel(f'{phase.upper()}-wave residual (s)', fontsize=11)
        ax_hist.set_ylabel('Count', fontsize=11)
        ax_hist.set_title(f'{phase.upper()}-wave: Residual distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\n{phase.upper()}-wave residual statistics:")
        print(f"  No violations: mean={residuals_ok.mean():.3f}s, std={residuals_ok.std():.3f}s, n={len(residuals_ok)}")
        if len(residuals_viol) > 0:
            print(f"  Violations:    mean={residuals_viol.mean():.3f}s, std={residuals_viol.std():.3f}s, n={len(residuals_viol)}")
        
        # Bottom row: Residuals vs distance
        ax_scatter = axes[1, col_idx]
        
        # Plot stations without violations
        mask_ok = ~df_meta_stations_copy['has_violation']
        ax_scatter.scatter(
            df_meta_stations_copy[mask_ok]['hypocentral_distance_km'],
            df_meta_stations_copy[mask_ok][res_col],
            alpha=0.6, s=50, label='No violation', color='steelblue'
        )
        
        # Plot stations with violations
        if violation_stations:
            mask_viol = df_meta_stations_copy['has_violation']
            ax_scatter.scatter(
                df_meta_stations_copy[mask_viol]['hypocentral_distance_km'],
                df_meta_stations_copy[mask_viol][res_col],
                alpha=0.8, s=100, marker='x', linewidth=2, 
                label='Violation', color='red', zorder=5
            )
        
        # Zero residual line
        ax_scatter.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        ax_scatter.set_xlabel('Hypocentral Distance (km)', fontsize=11)
        ax_scatter.set_ylabel(f'{phase.upper()}-wave residual (s)', fontsize=11)
        ax_scatter.set_title(f'{phase.UP()}-wave: Residuals vs Distance', fontsize=12, fontweight='bold')
        ax_scatter.legend(fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig