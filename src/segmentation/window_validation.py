"""
Quality control and validation for phase detection and windowing.

This module provides comprehensive validation functions to identify
problematic onset picks, assess signal quality, and detect systematic
errors in phase detection results.

Functions
---------
Individual checks:
    check_peak_in_s_wave : Validate peak timing (should be in S-wave window)
    check_monotonicity_station : Check arrival time ordering by distance
    check_snr : Signal-to-noise ratio validation

Batch validation:
    quality_control_all_stations : Run all checks for entire dataset
    print_quality_control_summary : Hierarchical table of results
    print_failed_checks : Detailed failure information
    print_detailed_failures : Full diagnostic output

Monotonicity analysis:
    analyze_monotonicity_violations : Identify and analyze violations
    print_violation_summary : Human-readable violation report
    plot_monotonicity_analysis : Visual analysis of distance-time relationship
    analyze_residuals_vs_violations : Correlate residuals with violations

Quality Control Tests
---------------------
1. **Peak Timing Check**
   - Validates that peak ground motion (PGA/PGV/PGD) occurs in S-wave window
   - Failure indicates: misidentified S-onset, or unusual wave propagation
   - Physical basis: S-waves carry most seismic energy

2. **Monotonicity Check**
   - Validates that arrival times increase with hypocentral distance
   - Uses 3D distance (not epicentral) accounting for source depth
   - Failure indicates: picking errors, or strong lateral velocity variations
   - Physical basis: waves propagate outward from source at finite velocity

3. **Signal-to-Noise Ratio (SNR)**
   - Compares RMS amplitude in phase window vs pre-event noise
   - Default threshold: SNR ≥ 3.0 (factor of 3 above noise)
   - Failure indicates: weak signal, high noise, or misidentified arrival
   - Uses first N seconds of phase window (default: 5s)

Typical Validation Results
--------------------------
For good-quality local/regional data (d < 150 km, M > 3.5):
- Peak timing: 90-95% pass rate
- Monotonicity P: 85-95% pass rate
- Monotonicity S: 80-90% pass rate (S harder to pick)
- SNR P: 95-99% pass rate
- SNR S: 85-95% pass rate

Lower pass rates may indicate:
- Poor event location quality
- Complex crustal structure (strong lateral variations)
- High ambient noise levels
- Inadequate search window sizing

Monotonicity Violations
-----------------------
Common causes:
1. **Picking errors**: Most common for S-waves (more emergent)
2. **Velocity anomalies**: Sedimentary basins (slow) vs basement (fast)
3. **Location errors**: Epicenter/depth uncertainties of ±2-5 km
4. **3D effects**: Assuming 1D velocity model in complex geology

Investigation workflow:
1. Run analyze_monotonicity_violations() to identify problem stations
2. Check residuals: large |residual| suggests velocity model mismatch
3. Visually inspect waveforms using plot functions
4. Compare with theoretical arrivals
5. Consider re-picking or excluding from analysis

Examples
--------
>>> # Comprehensive quality control
>>> qc_results = quality_control_all_stations(
...     windowed_signals,
...     df_full,
...     df_meta_stations,
...     peak_column='PGA_CM/S^2',
...     time_peak_column='TIME_PGA_S',
...     snr_threshold=3.0
... )
>>> print_quality_control_summary(qc_results)
>>> 
>>> # Analyze monotonicity violations
>>> violations_p = analyze_monotonicity_violations(df_meta_stations, phase='p')
>>> print_violation_summary(violations_p, phase='p')
>>> 
>>> # Visual analysis
>>> fig = plot_monotonicity_analysis(df_meta_stations, violations_p, violations_s)
>>> plt.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Union, Any, Literal
from pathlib import Path

def check_peak_in_s_wave(
    df_metadata: pd.DataFrame, 
    station: str, 
    component: str, 
    peak_column: str, 
    time_peak_column: str,
    coda_method: str = 'rautian', 
    sampling_rate: float = 200
) -> Dict[str, Any]:
    """
    Check if peak value occurs in S-wave window using only metadata.
    
    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata with onset times and peak info
        Expected columns (dual representation with auto-detection):
        - t_p_detected_seconds, t_p_detected_samples (or legacy t_p_detected)
        - t_s_detected_seconds, t_s_detected_samples (or legacy t_s_detected)
        - t_coda_<method>_seconds, t_coda_<method>_samples (or legacy t_coda_<method>)
        - STATION_CODE, COMPONENT
        - peak_column, time_peak_column (specified by caller)
    station : str
        Station code
    component : str
        Component name
    peak_column : str
        Column name for peak value (e.g., 'PGA_CM/S^2', 'PGV_CM/S', 'PGD_CM')
    time_peak_column : str
        Column name for time of peak (e.g., 'TIME_PGA_S', 'TIME_PGV_S', 'TIME_PGD_S')
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
            'peak_window': str,
            'peak_value': float,
            'peak_time': float
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
            'peak_window': 'UNKNOWN',
            'peak_value': np.nan,
            'peak_time': np.nan
        }
    
    row = df_metadata[mask].iloc[0]
    
    # Peak time and value (time is always in seconds from raw data)
    peak_time = row[time_peak_column]
    peak_value = row[peak_column]
    
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
    if peak_time < t_p:
        peak_window = 'pre_event'
    elif t_p <= peak_time < t_s:
        peak_window = 'p_wave'
    elif t_s <= peak_time < t_coda:
        peak_window = 's_wave'
    else:  # peak_time >= t_coda
        peak_window = 'coda'
    
    return {
        'passed': peak_window == 's_wave',
        'peak_window': peak_window,
        'peak_value': peak_value,
        'peak_time': peak_time
    }

def check_monotonicity_station(
    df_meta_stations: pd.DataFrame, 
    station: str, 
    phase: str = 'p', 
    sampling_rate: float = 200
) -> Dict[str, Any]:
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

def check_snr(
    windowed_signals: Dict[str, Dict[str, Dict[str, Dict]]], 
    station: str, 
    component: str, 
    phase: str = 'p', 
    threshold: float = 3.0, 
    signal_duration: float = 5.0, 
    dt: float = 0.005
) -> Dict[str, Any]:
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


def quality_control_all_stations(
    windowed_signals: Dict[str, Dict[str, Dict[str, Dict]]], 
    df_full: pd.DataFrame, 
    df_meta_stations: pd.DataFrame,
    peak_column: str, 
    time_peak_column: str,
    snr_threshold: float = 3.0, 
    coda_method: str = 'rautian'
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run all quality checks for all stations and components.
    
    Performs comprehensive validation including peak timing, monotonicity
    with distance, and signal-to-noise ratio checks.
    
    Parameters
    ----------
    windowed_signals : dict
        Segmented signals from segment_all_signals()
        Structure: {station: {component: {'pre_arrival': array, 'p_wave': array, ...}}}
    df_full : pd.DataFrame
        Component-level metadata (used for peak check)
        Must contain: 'STATION_CODE', 'COMPONENT',
                      peak_column, time_peak_column,
                      't_p_detected', 't_s_detected', 't_coda_{method}'
    df_meta_stations : pd.DataFrame
        Station-level metadata (used for monotonicity check)
        Must contain: 'STATION_CODE', 'hypocentral_distance_km',
                      't_p_detected', 't_s_detected'
    peak_column : str
        Column name for peak value (e.g., 'PGA_CM/S^2', 'PGV_CM/S', 'PGD_CM')
    time_peak_column : str
        Column name for time of peak (e.g., 'TIME_PGA_S', 'TIME_PGV_S', 'TIME_PGD_S')
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
                    'peak_check': dict,
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
    of a station), while peak and SNR checks are component-specific.
    
    Examples
    --------
    >>> qc_results = quality_control_all_stations(
    ...     windowed_signals, 
    ...     df_full, 
    ...     df_meta_stations,
    ...     peak_column='PGA_CM/S^2',
    ...     time_peak_column='TIME_PGA_S',
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
    
    if peak_column not in df_full.columns:
        raise ValueError(f"Column '{peak_column}' not found in df_full")
    
    if time_peak_column not in df_full.columns:
        raise ValueError(f"Column '{time_peak_column}' not found in df_full")
    
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
            # Peak check (component-specific, uses df_full)
            peak_check = check_peak_in_s_wave(
                df_full, station, component,
                peak_column=peak_column,
                time_peak_column=time_peak_column,
                coda_method=coda_method
            ) if 's_wave' in windowed_signals[station][component] else {
                'passed': False,
                'error': 'Missing s_wave window'
            }
            
            # Monotonicity checks (station-level, cached)
            mono_p = monotonicity_cache[station]['p']
            mono_s = monotonicity_cache[station]['s']
            
            # SNR checks (component-specific, uses windowed_signals)
            snr_p = check_snr(
                windowed_signals, station, component,
                phase='p', threshold=snr_threshold
            ) if 'p_wave' in windowed_signals[station][component] and 'pre_event' in windowed_signals[station][component] else {
                'passed': False, 'snr': np.nan, 'rms_signal': np.nan,
                'rms_noise': np.nan, 'threshold': snr_threshold,
                'error': 'Missing p_wave or pre_event window'
            }

            snr_s = check_snr(
                windowed_signals, station, component,
                phase='s', threshold=snr_threshold
            ) if 's_wave' in windowed_signals[station][component] and 'pre_event' in windowed_signals[station][component] else {
                'passed': False, 'snr': np.nan, 'rms_signal': np.nan,
                'rms_noise': np.nan, 'threshold': snr_threshold,
                'error': 'Missing s_wave or pre_event window'
            }
            
            # Aggregate results
            all_passed = (
                peak_check['passed'] and
                mono_p['passed'] and
                mono_s['passed'] and
                snr_p['passed'] and
                snr_s['passed']
            )
            
            results[station][component] = {
                'peak_check': peak_check,
                'monotonicity_p': mono_p,
                'monotonicity_s': mono_s,
                'snr_p': snr_p,
                'snr_s': snr_s,
                'all_passed': all_passed
            }
    
    return results

def print_quality_control_summary(
    qc_results: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """
    Print quality control results in hierarchical table format.
    
    Shows check results for each station/component:
    - Peak: whether peak value occurs in S-wave window
    - MonoP/MonoS: whether P/S arrival times are monotonic with distance
    - SNRP/SNRS: whether P/S windows meet SNR threshold
    
    Parameters
    ----------
    qc_results : dict
        Output from quality_control_all_stations()
    
    Examples
    --------
    >>> qc_results = quality_control_all_stations(...)
    >>> print_quality_control_summary(qc_results)
    Quality Control Summary
    ========================================
    ACER
      ├─ HGZ: [✓ Peak] [✓ MonoP] [✓ MonoS] [✓ SNRP] [✓ SNRS]
      ├─ HGN: [✗ Peak] [✓ MonoP] [✓ MonoS] [✓ SNRP] [✓ SNRS]
      └─ HGE: [✓ Peak] [✓ MonoP] [✓ MonoS] [✗ SNRP] [✓ SNRS]
    """
    print("\nQuality Control Summary")
    print("=" * 70)
    
    total_components = 0
    check_failures = {
        'peak': 0, 'monotonicity_p': 0, 'monotonicity_s': 0,
        'snr_p': 0, 'snr_s': 0
    }
    check_applicable = {
        'peak': 0, 'monotonicity_p': 0, 'monotonicity_s': 0,
        'snr_p': 0, 'snr_s': 0
    }
    
    for station in sorted(qc_results.keys()):
        print(f"\n{station}")
        
        components = sorted(qc_results[station].keys())
        
        for i, component in enumerate(components):
            total_components += 1
            checks = qc_results[station][component]
            
            # Format check results
            peak_status = "✓" if checks['peak_check']['passed'] else "✗"
            mono_p_status = "✓" if checks['monotonicity_p']['passed'] else "✗"
            mono_s_status = "✓" if checks['monotonicity_s']['passed'] else "✗"
            snr_p_status = "✓" if checks['snr_p']['passed'] else "✗"
            snr_s_status = "✓" if checks['snr_s']['passed'] else "✗"
            
            # Count failures
            # Peak
            if 'error' not in checks['peak_check']:
                check_applicable['peak'] += 1
                if not checks['peak_check']['passed']:
                    check_failures['peak'] += 1

            # Monotonicity P
            if 'error' not in checks['monotonicity_p']:
                check_applicable['monotonicity_p'] += 1
                if not checks['monotonicity_p']['passed']:
                    check_failures['monotonicity_p'] += 1

            # Monotonicity S
            if 'error' not in checks['monotonicity_s']:
                check_applicable['monotonicity_s'] += 1
                if not checks['monotonicity_s']['passed']:
                    check_failures['monotonicity_s'] += 1

            # SNR P
            if 'error' not in checks['snr_p']:
                check_applicable['snr_p'] += 1
                if not checks['snr_p']['passed']:
                    check_failures['snr_p'] += 1

            # SNR S
            if 'error' not in checks['snr_s']:
                check_applicable['snr_s'] += 1
                if not checks['snr_s']['passed']:
                    check_failures['snr_s'] += 1
            
            # Tree structure
            if i == len(components) - 1:
                prefix = "  └─"
            else:
                prefix = "  ├─"
            
            print(
                f"{prefix} {component}: "
                f"[{peak_status} Peak] "
                f"[{mono_p_status} MonoP] "
                f"[{mono_s_status} MonoS] "
                f"[{snr_p_status} SNRP] "
                f"[{snr_s_status} SNRS]"
            )
    
    # Summary statistics
    print("\n" + "=" * 70)
    print(f"Total components: {total_components}")
    print(f"\nCheck failures:")
    print(f"  Peak in S-wave:     {check_failures['peak']}/{check_applicable['peak']} ({100*check_failures['peak']/check_applicable['peak']:.1f}%)")
    print(f"  Monotonicity P:     {check_failures['monotonicity_p']}/{check_applicable['monotonicity_p']} ({100*check_failures['monotonicity_p']/check_applicable['monotonicity_p']:.1f}%)")
    print(f"  Monotonicity S:     {check_failures['monotonicity_s']}/{check_applicable['monotonicity_s']} ({100*check_failures['monotonicity_s']/check_applicable['monotonicity_s']:.1f}%)")
    print(f"  SNR P-wave:         {check_failures['snr_p']}/{check_applicable['snr_p']} ({100*check_failures['snr_p']/check_applicable['snr_p']:.1f}%)")
    print(f"  SNR S-wave:         {check_failures['snr_s']}/{check_applicable['snr_s']} ({100*check_failures['snr_s']/check_applicable['snr_s']:.1f}%)")
    print("=" * 70)


def print_failed_checks(
    qc_results: Dict[str, Dict[str, Dict[str, Any]]], 
    signal_unit: str = 'cm/s²'
) -> None:
    """
    Print detailed information for components that failed quality checks.
    
    Parameters
    ----------
    qc_results : dict
        Output from quality_control_all_stations()
    signal_unit : str, optional
        Unit for peak value display (default: 'cm/s²')
    
    Examples
    --------
    >>> print_failed_checks(qc_results, signal_unit='cm/s')
    """
    print("\nFailed Quality Checks - Details")
    print("=" * 70)
    
    has_failures = False
    
    for station in sorted(qc_results.keys()):
        for component in sorted(qc_results[station].keys()):
            checks = qc_results[station][component]
            
            if checks['all_passed']:
                continue  # Skip if all passed
            
            has_failures = True
            print(f"\n{station}-{component}:")
            
            # Peak failure
            if not checks['peak_check']['passed']:
                peak = checks['peak_check']
                print(f"  ✗ Peak: Found in '{peak['peak_window']}' window (value={peak['peak_value']:.4f} {signal_unit})")
            
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
                print(f"  ✗ SNRP: SNR={snr['snr']:.2f} < threshold={snr['threshold']:.1f}")
            
            # SNR S failure
            if not checks['snr_s']['passed']:
                snr = checks['snr_s']
                print(f"  ✗ SNRS: SNR={snr['snr']:.2f} < threshold={snr['threshold']:.1f}")
    
    if not has_failures:
        print("No failures - all components passed quality checks!")
    
    print("=" * 70)

def print_detailed_failures(
    qc_results: Dict[str, Dict[str, Dict[str, Any]]], 
    signal_unit: str = 'cm/s²'
) -> None:
    """
    Print detailed failure report with specific values.

    Shows RMS values for SNR checks and specific time comparisons
    for monotonicity violations.

    Parameters
    ----------
    qc_results : dict
        Output from quality_control_all_stations()
    signal_unit : str, optional
        Unit for peak value display (default: 'cm/s²')
        
    Examples
    --------
    >>> print_detailed_failures(qc_results, signal_unit='cm/s')
    """
    
    print("\n" + "="*80)
    print("DETAILED FAILURE REPORT")
    print("="*80)
    
    for station in sorted(qc_results.keys()):
        for component in sorted(qc_results[station].keys()):
            checks = qc_results[station][component]
            
            if checks['all_passed']:
                continue
            
            print(f"\n{station}-{component}:")
            
            # Peak failure
            if not checks['peak_check']['passed']:
                peak = checks['peak_check']
                print(f"  ✗ Peak: Found in '{peak['peak_window']}' window "
                      f"(value={peak['peak_value']:.4f} {signal_unit})")
            
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

def analyze_monotonicity_violations(
    df_meta_stations: pd.DataFrame, 
    phase: str = 'p', 
    sampling_rate: float = 200
) -> pd.DataFrame:
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
        - t_theo_seconds: theoretical arrival time (s)
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
        df_violations_sorted = df_violations.copy()
        df_violations_sorted['abs_residual'] = df_violations_sorted['residual'].abs()
        top3 = df_violations_sorted.nlargest(3, 'abs_residual')[
            ['station', 'distance_km', 't_detected', 't_theo', 'residual']
        ]
        print(top3.to_string(index=False))
    else:
        print(f"\nNo monotonicity violations found for {phase.upper()}-wave")
    
    return df_violations

def print_violation_summary(
    df_violations: pd.DataFrame, 
    phase: str = 'p'
) -> None:
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
            print(f"PREV VIOLATION:")
            print(f"       {row['prev_station']} (d={row['prev_distance']:.2f}km) arrived at {row['prev_t_detected']:.3f}s")
            print(f"      Closer station arrived LATER or same time (Δt = {row['prev_t_detected'] - row['t_detected']:+.3f}s)")
            if row['prev_residual'] is not None:
                print(f"       Residuals: prev={row['prev_residual']:+.3f}s, this={row['residual']:+.3f}s")
        
        if 'next' in row['violation_type']:
            print(f"    NEXT VIOLATION:")
            print(f"       {row['next_station']} (d={row['next_distance']:.2f}km) arrived at {row['next_t_detected']:.3f}s")
            print(f"       Farther station arrived EARLIER or same time (Δt = {row['next_t_detected'] - row['t_detected']:+.3f}s)")
            if row['next_residual'] is not None:
                print(f"       Residuals: this={row['residual']:+.3f}s, next={row['next_residual']:+.3f}s")
    
    print("\n" + "="*80)

def plot_monotonicity_analysis(
    df_meta_stations: pd.DataFrame, 
    df_violations_p: Optional[pd.DataFrame] = None, 
    df_violations_s: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (16, 6), 
    output_path: Optional[Union[str, Path]] = None, 
    sampling_rate: float = 200
) -> plt.Figure:
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

def analyze_residuals_vs_violations(
    df_meta_stations: pd.DataFrame, 
    df_violations_p: pd.DataFrame, 
    df_violations_s: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
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
        ax_scatter.set_title(f'{phase.upper()}-wave: Residuals vs Distance', fontsize=12, fontweight='bold')
        ax_scatter.legend(fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig