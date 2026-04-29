# src/segmentation/onset_detection.py

"""
Onset detection using AR-AIC method.

Functions for detecting P and S wave arrivals using the Autoregressive-Akaike
Information Criterion (AR-AIC) method implemented in ObsPy.

References
----------
Leonard, M., & Kennett, B. L. N. (1999). Multi-component autoregressive 
techniques for the analysis of seismograms. Physics of the Earth and 
Planetary Interiors, 113(1-4), 247-263.
"""

import numpy as np
import pandas as pd
from obspy.signal.trigger import ar_pick
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr

def detect_onsets_arpick(signals_dict, df_meta_stations,
                         sampling_rate=200,
                         unit='samples',
                         p_window_before=5, p_window_after=5,
                         s_window_before=7, s_window_after=7):
    """
    Detect P and S onsets using AR-AIC with theoretical search windows.
    
    Applies AR-AIC method to signal subsets extracted around theoretical
    arrival times. Creates dual representation (samples + seconds) for all outputs.
    
    Parameters
    ----------
    signals_dict : dict
        Nested dictionary from convert_signals_to_dict()
        Structure: {station: {component: array, 'time': array}}
    df_meta_stations : pd.DataFrame
        Station metadata with columns:
        - STATION_CODE
        - Theoretical arrivals (one of):
          * t_p_theo_samples, t_s_theo_samples (preferred), OR
          * t_p_theo_seconds, t_s_theo_seconds, OR
          * t_p_theo, t_s_theo (legacy, interpreted as seconds)
        - LOW_CUT_FREQUENCY_HZ, HIGH_CUT_FREQUENCY_HZ
        - Search windows (one of):
          * p_window_start_samples, p_window_end_samples (preferred), OR
          * p_window_start_seconds, p_window_end_seconds, OR
          * p_window_start, p_window_end (legacy)
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
    unit : {'samples', 'seconds'}, optional
        Preferred unit for legacy column names (default: 'samples')
    p_window_before, p_window_after : float, optional
        Fallback window sizes in SECONDS if search windows not in df_meta_stations
    s_window_before, s_window_after : float, optional
        Fallback window sizes in SECONDS if search windows not in df_meta_stations
    
    Returns
    -------
    pd.DataFrame
        df_meta_stations with added columns (DUAL representation):
        
        Detected onsets:
        - t_p_detected_samples, t_s_detected_samples (int)
        - t_p_detected_seconds, t_s_detected_seconds (float)
        - t_p_detected, t_s_detected (legacy alias based on unit)
        
        Residuals (seconds only, as physical interpretation):
        - p_residual_seconds, s_residual_seconds (float)
        - p_residual, s_residual (legacy alias)
        
        Status flags:
        - p_detection_success, s_detection_success (bool)
        - error_message (str)
        - components_used (str)
    
    Notes
    -----
    ar_pick returns sample indices (int), which are then converted to both
    sample and second representations. This avoids rounding artifacts.
    
    Workflow per station:
    1. Extract subset around theoretical arrival (using search windows)
    2. Apply ar_pick() → returns SAMPLE INDEX within subset
    3. Convert to absolute sample index in full signal
    4. Convert to seconds for dual representation
    
    Examples
    --------
    >>> df_meta_stations = calculate_adaptive_windows(df_meta_stations, thresholds)
    >>> df_meta_stations = detect_onsets_arpick(signals_dict, df_meta_stations)
    >>> print(f"P success: {df_meta_stations['p_detection_success'].sum()}")
    >>> print(df_meta_stations[['STATION_CODE', 't_p_detected_samples', 't_p_detected_seconds']])
    """
    
    # Initialize onset columns - DUAL representation
    df_meta_stations['t_p_detected_samples'] = pd.NA
    df_meta_stations['t_s_detected_samples'] = pd.NA
    df_meta_stations['t_p_detected_seconds'] = np.nan
    df_meta_stations['t_s_detected_seconds'] = np.nan
    
    df_meta_stations['p_residual_seconds'] = np.nan
    df_meta_stations['s_residual_seconds'] = np.nan
    
    df_meta_stations['p_detection_success'] = False
    df_meta_stations['s_detection_success'] = False
    df_meta_stations['error_message'] = ''
    df_meta_stations['components_used'] = ''
    
    print(f"Running AR-AIC onset detection with theoretical windows...")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Fallback P window: [{-p_window_before}, +{p_window_after}]s around t_p_theo")
    print(f"  Fallback S window: [{-s_window_before}, +{s_window_after}]s around t_s_theo")
    print("\nProcessing: ", end="", flush=True)
    
    for idx, station_meta in df_meta_stations.iterrows():
        print(".", end="", flush=True)
        
        station = station_meta['STATION_CODE']
        
        # Check if station exists in signals_dict
        if station not in signals_dict:
            df_meta_stations.loc[idx, 'error_message'] = 'Station not found in signals_dict'
            continue
        
        data = signals_dict[station]
        
        # Identify available components
        components = [k for k in data.keys() if k != 'time']
        time = data['time']
        
        # Find component names
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
        
        # Check if we have all 3 components
        if comp_z is None or comp_n is None or comp_e is None:
            df_meta_stations.loc[idx, 'error_message'] = (
                f'Incomplete components: Z={comp_z}, N={comp_n}, E={comp_e}'
            )
            df_meta_stations.loc[idx, 'components_used'] = ','.join(
                [c for c in [comp_z, comp_n, comp_e] if c]
            )
            continue
        
        # Get full signals
        signal_z_full = data[comp_z]
        signal_n_full = data[comp_n]
        signal_e_full = data[comp_e]
        
        # Record components used
        df_meta_stations.loc[idx, 'components_used'] = f'{comp_e},{comp_n},{comp_z}'
        
        # Get filter parameters
        f1 = station_meta['LOW_CUT_FREQUENCY_HZ']
        f2 = station_meta['HIGH_CUT_FREQUENCY_HZ']
        
        # ===== GET THEORETICAL TIMES (prefer samples, fallback seconds) =====
        if 't_p_theo_samples' in station_meta.index:
            t_p_theo_samp = int(station_meta['t_p_theo_samples'])
            t_s_theo_samp = int(station_meta['t_s_theo_samples'])
            has_theo_samples = True
        elif 't_p_theo_seconds' in station_meta.index:
            t_p_theo_sec = float(station_meta['t_p_theo_seconds'])
            t_s_theo_sec = float(station_meta['t_s_theo_seconds'])
            t_p_theo_samp = int(np.round(t_p_theo_sec * sampling_rate))
            t_s_theo_samp = int(np.round(t_s_theo_sec * sampling_rate))
            has_theo_samples = False
        else:
            # Legacy columns
            t_p_theo_sec = float(station_meta['t_p_theo'])
            t_s_theo_sec = float(station_meta['t_s_theo'])
            t_p_theo_samp = int(np.round(t_p_theo_sec * sampling_rate))
            t_s_theo_samp = int(np.round(t_s_theo_sec * sampling_rate))
            has_theo_samples = False
        
        # ===== GET SEARCH WINDOWS (prefer samples, fallback seconds) =====
        # P window
        if 'p_window_start_samples' in station_meta.index and not pd.isna(station_meta['p_window_start_samples']):
            p_win_start_samp = int(station_meta['p_window_start_samples'])
            p_win_end_samp = int(station_meta['p_window_end_samples'])
        elif 'p_window_start_seconds' in station_meta.index and not pd.isna(station_meta['p_window_start_seconds']):
            p_win_start_sec = float(station_meta['p_window_start_seconds'])
            p_win_end_sec = float(station_meta['p_window_end_seconds'])
            p_win_start_samp = int(np.round(p_win_start_sec * sampling_rate))
            p_win_end_samp = int(np.round(p_win_end_sec * sampling_rate))
        elif 'p_window_start' in station_meta.index and not pd.isna(station_meta['p_window_start']):
            # Legacy - assume seconds
            p_win_start_sec = float(station_meta['p_window_start'])
            p_win_end_sec = float(station_meta['p_window_end'])
            p_win_start_samp = int(np.round(p_win_start_sec * sampling_rate))
            p_win_end_samp = int(np.round(p_win_end_sec * sampling_rate))
        else:
            # Fallback: compute from theo time + window size
            p_win_start_samp = max(0, t_p_theo_samp - int(np.round(p_window_before * sampling_rate)))
            p_win_end_samp = t_p_theo_samp + int(np.round(p_window_after * sampling_rate))
        
        # S window
        if 's_window_start_samples' in station_meta.index and not pd.isna(station_meta['s_window_start_samples']):
            s_win_start_samp = int(station_meta['s_window_start_samples'])
            s_win_end_samp = int(station_meta['s_window_end_samples'])
        elif 's_window_start_seconds' in station_meta.index and not pd.isna(station_meta['s_window_start_seconds']):
            s_win_start_sec = float(station_meta['s_window_start_seconds'])
            s_win_end_sec = float(station_meta['s_window_end_seconds'])
            s_win_start_samp = int(np.round(s_win_start_sec * sampling_rate))
            s_win_end_samp = int(np.round(s_win_end_sec * sampling_rate))
        elif 's_window_start' in station_meta.index and not pd.isna(station_meta['s_window_start']):
            # Legacy
            s_win_start_sec = float(station_meta['s_window_start'])
            s_win_end_sec = float(station_meta['s_window_end'])
            s_win_start_samp = int(np.round(s_win_start_sec * sampling_rate))
            s_win_end_samp = int(np.round(s_win_end_sec * sampling_rate))
        else:
            # Fallback
            s_win_start_samp = max(0, t_s_theo_samp - int(np.round(s_window_before * sampling_rate)))
            s_win_end_samp = t_s_theo_samp + int(np.round(s_window_after * sampling_rate))
        
        # Initialize detection results
        t_p_detected_samp = None
        t_s_detected_samp = None
        t_p_detected_sec = np.nan
        t_s_detected_sec = np.nan
        p_success = False
        s_success = False
        error_msg = ''
        
        # ===== P-WAVE DETECTION =====
        try:
            # Extract P window subset (using sample indices)
            signal_z_p = signal_z_full[p_win_start_samp:p_win_end_samp]
            signal_n_p = signal_n_full[p_win_start_samp:p_win_end_samp]
            signal_e_p = signal_e_full[p_win_start_samp:p_win_end_samp]
            
            if len(signal_z_p) < 100:  # Need at least 0.5s at 200Hz
                raise ValueError(f"P window too short: {len(signal_z_p)} samples")
            
            # Apply AR-AIC to P window
            # ar_pick returns SAMPLE INDEX within the subset
            p_pick_relative_samp, s_pick_in_p_window = ar_pick(
                signal_z_p, signal_n_p, signal_e_p,
                samp_rate=sampling_rate,
                f1=f1,
                f2=f2,
                lta_p=1.0,
                sta_p=0.1,
                lta_s=4.0,
                sta_s=1.0,
                m_p=2,
                m_s=8,
                l_p=0.1,
                l_s=0.2
            )
            
            # Convert to absolute sample index in full signal
            if p_pick_relative_samp is not None and not np.isnan(p_pick_relative_samp):
                t_p_detected_samp = p_win_start_samp + int(p_pick_relative_samp)
                t_p_detected_sec = t_p_detected_samp / sampling_rate
                p_success = True
            else:
                error_msg += 'P-pick returned None; '
        
        except Exception as e:
            error_msg += f'P detection failed: {str(e)}; '
        
        # ===== S-WAVE DETECTION =====
        try:
            # Extract S window subset (using sample indices)
            signal_z_s = signal_z_full[s_win_start_samp:s_win_end_samp]
            signal_n_s = signal_n_full[s_win_start_samp:s_win_end_samp]
            signal_e_s = signal_e_full[s_win_start_samp:s_win_end_samp]
            
            if len(signal_z_s) < 100:
                raise ValueError(f"S window too short: {len(signal_z_s)} samples")
            
            # Apply AR-AIC to S window
            p_pick_in_s_window, s_pick_relative_samp = ar_pick(
                signal_z_s, signal_n_s, signal_e_s,
                samp_rate=sampling_rate,
                f1=f1,
                f2=f2,
                lta_p=1.0,
                sta_p=0.1,
                lta_s=4.0,
                sta_s=1.0,
                m_p=2,
                m_s=8,
                l_p=0.1,
                l_s=0.2
            )
            
            # Convert to absolute sample index in full signal
            if s_pick_relative_samp is not None and not np.isnan(s_pick_relative_samp):
                t_s_detected_samp = s_win_start_samp + int(s_pick_relative_samp)
                t_s_detected_sec = t_s_detected_samp / sampling_rate
                s_success = True
            else:
                error_msg += 'S-pick returned None; '
        
        except Exception as e:
            error_msg += f'S detection failed: {str(e)}; '
        
        # ===== POPULATE DATAFRAME (DUAL REPRESENTATION) =====
        df_meta_stations.loc[idx, 't_p_detected_samples'] = t_p_detected_samp
        df_meta_stations.loc[idx, 't_s_detected_samples'] = t_s_detected_samp
        df_meta_stations.loc[idx, 't_p_detected_seconds'] = t_p_detected_sec
        df_meta_stations.loc[idx, 't_s_detected_seconds'] = t_s_detected_sec
        
        df_meta_stations.loc[idx, 'p_detection_success'] = p_success
        df_meta_stations.loc[idx, 's_detection_success'] = s_success
        
        # Calculate residuals (seconds only, for physical interpretation)
        if p_success:
            if has_theo_samples:
                t_p_theo_sec = t_p_theo_samp / sampling_rate
            df_meta_stations.loc[idx, 'p_residual_seconds'] = t_p_detected_sec - t_p_theo_sec
        
        if s_success:
            if has_theo_samples:
                t_s_theo_sec = t_s_theo_samp / sampling_rate
            df_meta_stations.loc[idx, 's_residual_seconds'] = t_s_detected_sec - t_s_theo_sec
        
        # Record errors if any
        if error_msg:
            df_meta_stations.loc[idx, 'error_message'] = error_msg.strip('; ')
    
    # ===== LEGACY COLUMNS (for backward compatibility) =====
    if unit == 'samples':
        df_meta_stations['t_p_detected'] = df_meta_stations['t_p_detected_samples']
        df_meta_stations['t_s_detected'] = df_meta_stations['t_s_detected_samples']
    else:
        df_meta_stations['t_p_detected'] = df_meta_stations['t_p_detected_seconds']
        df_meta_stations['t_s_detected'] = df_meta_stations['t_s_detected_seconds']
    
    df_meta_stations['p_residual'] = df_meta_stations['p_residual_seconds']
    df_meta_stations['s_residual'] = df_meta_stations['s_residual_seconds']
    
    print("\n\nDetection complete!")
    
    # Summary statistics
    n_stations = len(df_meta_stations)
    n_p_success = df_meta_stations['p_detection_success'].sum()
    n_s_success = df_meta_stations['s_detection_success'].sum()
    
    print(f"\nResults:")
    print(f"  P-wave: {n_p_success}/{n_stations} successful ({100*n_p_success/n_stations:.1f}%)")
    print(f"  S-wave: {n_s_success}/{n_stations} successful ({100*n_s_success/n_stations:.1f}%)")
    
    if n_p_success > 0:
        p_res_mean = df_meta_stations['p_residual_seconds'].mean()
        p_res_std = df_meta_stations['p_residual_seconds'].std()
        print(f"  P residuals: {p_res_mean:.2f} ± {p_res_std:.2f} s")
    
    if n_s_success > 0:
        s_res_mean = df_meta_stations['s_residual_seconds'].mean()
        s_res_std = df_meta_stations['s_residual_seconds'].std()
        print(f"  S residuals: {s_res_mean:.2f} ± {s_res_std:.2f} s")
    
    # Report failures
    failures = df_meta_stations[~(df_meta_stations['p_detection_success'] & 
                                   df_meta_stations['s_detection_success'])]
    if len(failures) > 0:
        print(f"\nFailed/partial detections ({len(failures)} stations):")
        for idx, row in failures.iterrows():
            status = []
            if not row['p_detection_success']:
                status.append('P')
            if not row['s_detection_success']:
                status.append('S')
            print(f"  {row['STATION_CODE']}: {', '.join(status)} failed - {row['error_message']}")
    
    return df_meta_stations

def detect_coda_start(signal, t_s_detected, t_p_detected=None, origin_time=None,
                     sampling_rate=200, method='rautian',
                     unit='samples',
                     threshold_arias=0.95, threshold_envelope=0.3):
    """
    Detect coda onset using multiple methods from seismological literature.
    
    Parameters
    ----------
    signal : np.ndarray
        Seismic signal (single component)
    t_s_detected : int or float
        Detected S-wave onset time (samples or seconds, see unit)
    t_p_detected : int or float, optional
        Detected P-wave onset time (samples or seconds, see unit)
        Required for 'arias' method with proper signal windowing
    origin_time : int or float, optional
        Earthquake origin time (samples or seconds, see unit)
        Required for 'rautian' method
    sampling_rate : float
        Sampling rate in Hz (default: 200)
    method : str
        Detection method:
        - 'rautian': Lapse time = 2 × S-wave travel time (Rautian & Khalturin, 1978)
        - 'arias': Cumulative Arias Intensity threshold (Trifunac & Brady, 1975; Lanzano et al., 2019)
        - 'envelope': Amplitude envelope decay (Boore, 2005)
        - 'median': Median of rautian, arias, envelope (robust to outliers)
    unit : {'samples', 'seconds'}, optional
        Unit of input times (default: 'samples')
    threshold_arias : float
        Threshold for 'arias' method (default: 0.95 for D5-95)
        Common values: 0.75 (D5-75), 0.95 (D5-95)
    threshold_envelope : float
        Threshold factor for 'envelope' method (default: 0.3)
        Common range: 0.2-0.3 (20-30% of peak)
    
    Returns
    -------
    dict
        Dictionary with:
        - 't_coda_samples': int, coda start in samples
        - 't_coda_seconds': float, coda start in seconds
        - 't_coda': float, legacy alias for t_coda_seconds
        - 'method': str, method used
        - 'params': dict, method-specific parameters
        - 'diagnostic': dict, diagnostic information for validation
    
    Examples
    --------
    >>> # NEW: samples-based (preferred)
    >>> result = detect_coda_start(signal, t_s=3040, origin_time=1640,
    ...                            method='rautian', unit='samples')
    >>> print(result['t_coda_samples'], result['t_coda_seconds'])
    
    >>> # OLD: seconds-based (backward compatible)
    >>> result = detect_coda_start(signal, t_s=15.2, origin_time=8.2,
    ...                            method='rautian', unit='seconds')
    """
    
    # Convert inputs to both representations
    if unit == 'samples':
        t_s_samp = int(t_s_detected)
        t_s_sec = t_s_samp / sampling_rate
        
        if t_p_detected is not None:
            t_p_samp = int(t_p_detected)
            t_p_sec = t_p_samp / sampling_rate
        else:
            t_p_samp = None
            t_p_sec = None
        
        if origin_time is not None:
            origin_samp = int(origin_time)
            origin_sec = origin_samp / sampling_rate
        else:
            origin_samp = None
            origin_sec = None
            
    elif unit == 'seconds':
        t_s_sec = float(t_s_detected)
        t_s_samp = int(np.round(t_s_sec * sampling_rate))
        
        if t_p_detected is not None:
            t_p_sec = float(t_p_detected)
            t_p_samp = int(np.round(t_p_sec * sampling_rate))
        else:
            t_p_samp = None
            t_p_sec = None
        
        if origin_time is not None:
            origin_sec = float(origin_time)
            origin_samp = int(np.round(origin_sec * sampling_rate))
        else:
            origin_samp = None
            origin_sec = None
    else:
        raise ValueError(f"unit must be 'samples' or 'seconds', got {unit}")
    
    signal_duration_sec = len(signal) / sampling_rate
    signal_duration_samp = len(signal)
    
    if method == 'rautian':
        if origin_sec is None:
            raise ValueError("origin_time required for 'rautian' method")
        
        # Rautian & Khalturin (1978): lapse_time_coda = 2 × lapse_time_S
        s_lapse_time = t_s_sec - origin_sec
        coda_lapse_time = 2.0 * s_lapse_time
        t_coda_sec = origin_sec + coda_lapse_time
        t_coda_samp = int(np.round(t_coda_sec * sampling_rate))
        
        diagnostic = {
            's_lapse_time': s_lapse_time,
            'coda_lapse_time': coda_lapse_time,
            's_duration_seconds': t_coda_sec - t_s_sec,
            's_duration_samples': t_coda_samp - t_s_samp
        }
        params = {'multiplier': 2.0}
    
    elif method == 'arias':
        """
        Coda onset using Arias Intensity D5-95 method.
        
        Arias Intensity: AI(t) = (π/2g) ∫ a²(τ) dτ
        
        Standard approach (Lanzano et al., 2019 - ESM flatfile):
        - Compute AI on entire significant signal (from before P-onset)
        - D5: time when 5% of total energy is reached (~P arrival)
        - D95: time when 95% of total energy is reached (coda onset)
        - Significant duration: D95 - D5
        """
        
        # Define signal window: start 5s before P-onset (or beginning of file)
        if t_p_samp is not None:
            window_start_samp = max(0, t_p_samp - int(5.0 * sampling_rate))
        elif origin_samp is not None:
            window_start_samp = max(0, origin_samp)
        else:
            window_start_samp = 0
        
        window_start_sec = window_start_samp / sampling_rate
        signal_window = signal[window_start_samp:]
        
        # Compute Arias Intensity on full signal window
        dt = 1.0 / sampling_rate
        g = 9.81  # m/s²
        arias_cumsum = (np.pi / (2 * g)) * np.cumsum(signal_window**2) * dt
        
        # Handle pathological case (zero signal)
        if arias_cumsum[-1] == 0 or len(arias_cumsum) == 0:
            t_coda_sec = t_s_sec + 10.0
            t_coda_samp = t_s_samp + int(10.0 * sampling_rate)
            diagnostic = {
                'threshold': threshold_arias,
                'total_arias_intensity': 0,
                's_duration_seconds': 10.0,
                's_duration_samples': int(10.0 * sampling_rate),
                'warning': 'Zero signal - fallback used',
                'window_start_seconds': window_start_sec,
                'window_start_samples': window_start_samp
            }
            params = {
                'threshold': threshold_arias,
                'window_start_seconds': window_start_sec,
                'window_start_samples': window_start_samp,
                'arias_based': True
            }
        else:
            # Normalize to [0, 1]
            arias_norm = arias_cumsum / arias_cumsum[-1]
            
            # Find D5, D75, D95 (indices relative to window start)
            idx_D5 = np.argmax(arias_norm >= 0.05)
            idx_D75 = np.argmax(arias_norm >= 0.75)
            idx_D95 = np.argmax(arias_norm >= 0.95)
            
            # Convert to absolute sample indices and seconds
            t_D5_samp = window_start_samp + idx_D5
            t_D75_samp = window_start_samp + idx_D75
            t_D95_samp = window_start_samp + idx_D95
            
            t_D5_sec = t_D5_samp / sampling_rate
            t_D75_sec = t_D75_samp / sampling_rate
            t_D95_sec = t_D95_samp / sampling_rate
            
            # Coda onset = D95 (95% energy threshold)
            t_coda_samp = t_D95_samp
            t_coda_sec = t_D95_sec
            
            # Check if threshold was never reached
            if idx_D95 == 0 and arias_norm[0] < threshold_arias:
                t_coda_sec = t_s_sec + 10.0
                t_coda_samp = t_s_samp + int(10.0 * sampling_rate)
                warning = f'Threshold {threshold_arias} never reached - fallback used'
            else:
                warning = None
            
            # Diagnostic info
            diagnostic = {
                'threshold': threshold_arias,
                'total_arias_intensity': arias_cumsum[-1],
                't_D5_samples': t_D5_samp,
                't_D75_samples': t_D75_samp,
                't_D95_samples': t_D95_samp,
                't_D5_seconds': t_D5_sec,
                't_D75_seconds': t_D75_sec,
                't_D95_seconds': t_D95_sec,
                'significant_duration_D5_D95_seconds': t_D95_sec - t_D5_sec,
                'significant_duration_D5_D75_seconds': t_D75_sec - t_D5_sec,
                'significant_duration_D5_D95_samples': t_D95_samp - t_D5_samp,
                'significant_duration_D5_D75_samples': t_D75_samp - t_D5_samp,
                's_duration_seconds': t_coda_sec - t_s_sec,
                's_duration_samples': t_coda_samp - t_s_samp,
                'window_start_seconds': window_start_sec,
                'window_start_samples': window_start_samp,
                'reference': f'D5-D{int(threshold_arias*100)} (Lanzano et al., 2019 - ESM flatfile)'
            }
            
            if warning:
                diagnostic['warning'] = warning
            
            params = {
                'threshold': threshold_arias,
                'window_start_seconds': window_start_sec,
                'window_start_samples': window_start_samp,
                'arias_based': True
            }
    
    elif method == 'envelope':
        """
        Envelope decay method.
        
        Following common practice in strong-motion processing (Boore & Bommer, 2005),
        the coda is defined as beginning when the smoothed envelope falls below a
        threshold (typically 20-30%) of its peak value.
        """
        
        # Extract signal after S-onset
        signal_after_s = signal[t_s_samp:]
        
        # Calculate envelope using Hilbert transform
        from scipy.signal import hilbert
        from scipy.ndimage import uniform_filter1d
        
        envelope = np.abs(hilbert(signal_after_s))
        
        # Smooth envelope with 1-second moving average (standard practice)
        smooth_window = int(sampling_rate)
        envelope_smooth = uniform_filter1d(envelope, size=smooth_window)
        
        # Find peak in first 5 seconds after S-onset
        search_window_samp = min(int(5 * sampling_rate), len(envelope_smooth))
        if search_window_samp == 0:
            t_coda_sec = t_s_sec + 10.0
            t_coda_samp = t_s_samp + int(10.0 * sampling_rate)
            peak_envelope = 0
            threshold_abs = 0
        else:
            peak_envelope = np.max(envelope_smooth[:search_window_samp])
            
            # Coda starts when envelope drops below threshold
            threshold_abs = threshold_envelope * peak_envelope
            
            idx_coda_rel = np.argmax(envelope_smooth < threshold_abs)
            if idx_coda_rel == 0:  # Never drops
                t_coda_sec = t_s_sec + 10.0
                t_coda_samp = t_s_samp + int(10.0 * sampling_rate)
            else:
                t_coda_samp = t_s_samp + idx_coda_rel
                t_coda_sec = t_coda_samp / sampling_rate
        
        diagnostic = {
            'peak_envelope': peak_envelope,
            'threshold_factor': threshold_envelope,
            'threshold_absolute': threshold_abs,
            's_duration_seconds': t_coda_sec - t_s_sec,
            's_duration_samples': t_coda_samp - t_s_samp,
            'reference': 'Boore & Bommer (2005)'
        }
        params = {
            'threshold_factor': threshold_envelope,
            'smoothing_window_s': 1.0,
            'smoothing_window_samples': smooth_window
        }
        
    elif method == 'median':
        """
        Robust median-based method.
        Computes coda onset using all three methods and takes the median.
        Requires origin_time for Rautian method.
        """
        if origin_sec is None:
            raise ValueError("origin_time required for 'median' method (needs Rautian)")
        
        # Calculate all 3 methods
        methods_to_combine = ['rautian', 'arias', 'envelope']
        t_codas_samp = []
        t_codas_sec = []
        results_dict = {}
        
        for m in methods_to_combine:
            try:
                res = detect_coda_start(
                    signal, t_s_detected, t_p_detected=t_p_detected,
                    origin_time=origin_time, sampling_rate=sampling_rate,
                    method=m, unit=unit,
                    threshold_arias=threshold_arias,
                    threshold_envelope=threshold_envelope
                )
                t_codas_samp.append(res['t_coda_samples'])
                t_codas_sec.append(res['t_coda_seconds'])
                results_dict[m] = res
            except Exception as e:
                print(f"Warning: {m} method failed: {e}")
                continue
        
        if len(t_codas_samp) < 2:
            raise ValueError("Median method requires at least 2 valid methods")
        
        # Calculate median
        t_coda_samp = int(np.median(t_codas_samp))
        t_coda_sec = np.median(t_codas_sec)
        
        # Diagnostic info
        diagnostic = {
            'method_values_samples': {
                'rautian': results_dict.get('rautian', {}).get('t_coda_samples'),
                'arias': results_dict.get('arias', {}).get('t_coda_samples'),
                'envelope': results_dict.get('envelope', {}).get('t_coda_samples')
            },
            'method_values_seconds': {
                'rautian': results_dict.get('rautian', {}).get('t_coda_seconds'),
                'arias': results_dict.get('arias', {}).get('t_coda_seconds'),
                'envelope': results_dict.get('envelope', {}).get('t_coda_seconds')
            },
            'median_value_samples': t_coda_samp,
            'median_value_seconds': t_coda_sec,
            's_duration_seconds': t_coda_sec - t_s_sec,
            's_duration_samples': t_coda_samp - t_s_samp,
            'std_seconds': np.std(t_codas_sec),
            'std_samples': np.std(t_codas_samp),
            'range_seconds': (np.min(t_codas_sec), np.max(t_codas_sec)),
            'range_samples': (np.min(t_codas_samp), np.max(t_codas_samp)),
            'reference': 'Median of Rautian, Arias, and Envelope methods'
        }
        
        params = {
            'threshold_arias': threshold_arias,
            'threshold_envelope': threshold_envelope,
            'methods_used': methods_to_combine
        }
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'rautian', 'arias', 'envelope', 'median'"
        )
    
    # Ensure coda is within signal bounds
    t_coda_samp = min(t_coda_samp, signal_duration_samp - int(1.0 * sampling_rate))
    t_coda_samp = max(t_coda_samp, t_s_samp + int(1.0 * sampling_rate))  # At least 1s of S-wave
    
    # Recompute seconds from bounded samples (to ensure consistency)
    t_coda_sec = t_coda_samp / sampling_rate
    
    return {
        't_coda_samples': t_coda_samp,
        't_coda_seconds': t_coda_sec,
        't_coda': t_coda_sec,  # Legacy alias
        'method': method,
        'params': params,
        'diagnostic': diagnostic
    }


def detect_coda_start_all_methods(signal, t_s_detected, t_p_detected=None, origin_time=None,
                                  sampling_rate=200, unit='samples',
                                  threshold_arias=0.95,
                                  threshold_envelope=0.3):
    """
    Apply all coda detection methods and return results for comparison.
    
    Parameters
    ----------
    signal : np.ndarray
        Seismic signal
    t_s_detected : int or float
        S-wave onset time (samples or seconds, see unit)
    t_p_detected : int or float, optional
        P-wave onset time (samples or seconds, see unit)
    origin_time : int or float, optional
        Earthquake origin time (samples or seconds, see unit)
        Required for 'rautian' and 'median' methods
    sampling_rate : float
        Sampling rate (Hz)
    unit : {'samples', 'seconds'}, optional
        Unit of input times (default: 'samples')
    threshold_arias : float
        Threshold for Arias Intensity method (default: 0.95 for D5-95)
    threshold_envelope : float
        Threshold factor for envelope method (default: 0.3)
    
    Returns
    -------
    dict
        Dictionary with method names as keys, each containing result dict
        with dual representation (t_coda_samples, t_coda_seconds)
    
    Examples
    --------
    >>> # NEW: samples-based
    >>> results = detect_coda_start_all_methods(
    ...     signal, t_s=3040, origin_time=1640,
    ...     unit='samples'
    ... )
    >>> for method, res in results.items():
    ...     print(f"{method}: t_coda={res['t_coda_samples']} samp "
    ...           f"({res['t_coda_seconds']:.2f}s)")
    
    >>> # OLD: seconds-based
    >>> results = detect_coda_start_all_methods(
    ...     signal, t_s=15.2, origin_time=8.2,
    ...     unit='seconds'
    ... )
    """
    methods = ['rautian', 'arias', 'envelope', 'median']
    results = {}
    
    for method in methods:
        try:
            if method in ['rautian', 'median'] and origin_time is None:
                continue  # Skip if origin_time not available
            
            result = detect_coda_start(
                signal, t_s_detected, t_p_detected=t_p_detected,
                origin_time=origin_time,
                sampling_rate=sampling_rate, method=method, unit=unit,
                threshold_arias=threshold_arias,
                threshold_envelope=threshold_envelope
            )
            results[method] = result
            
        except Exception as e:
            print(f"Warning: Method '{method}' failed: {e}")
            continue
    
    return results

def add_coda_onsets_to_dataframe(df_full, signals_dict, 
                                 threshold_arias=0.75,
                                 threshold_envelope=0.3,
                                 sampling_rate=200):
    """
    Populate coda onset columns in df_full.
    
    Optimized: Rautian method computed once per station (not per component).
    """
    
    methods = ['rautian', 'arias', 'envelope']
    
    print("Pre-computing Rautian coda onset per station...")
    rautian_cache = {}
    
    for station in df_full['STATION_CODE'].unique():
        station_row = df_full[df_full['STATION_CODE'] == station].iloc[0]
        t_s = station_row['t_s_detected']
        origin_time = station_row['origin_time']
        s_lapse_time = t_s - origin_time
        t_coda_rautian = origin_time + 2.0 * s_lapse_time
        
        rautian_cache[station] = {
            't_coda': t_coda_rautian,
            's_duration': t_coda_rautian - t_s
        }
    
    print(f"  Computed Rautian for {len(rautian_cache)} stations")    
    print(f"Calculating Arias and Envelope coda onset for {len(df_full)} components...")
    
    for idx, row in df_full.iterrows():
        station = row['STATION_CODE']
        component = row['COMPONENT']
        
        if station not in signals_dict or component not in signals_dict[station]:
            continue
        
        signal = signals_dict[station][component]
        t_s = row['t_s_detected']
        origin_time = row['origin_time']
        
        try:
            # ===== RAUTIAN: =====
            if station in rautian_cache:
                df_full.loc[idx, 't_coda_rautian'] = rautian_cache[station]['t_coda']
                df_full.loc[idx, 's_duration_rautian'] = rautian_cache[station]['s_duration']
            
            # ===== ARIAS: =====
            result_arias = detect_coda_start(
                signal, t_s, origin_time=origin_time,
                sampling_rate=sampling_rate, method='arias',
                threshold_arias=threshold_arias,
                threshold_envelope=threshold_envelope
            )
            df_full.loc[idx, 't_coda_arias'] = result_arias['t_coda']
            df_full.loc[idx, 's_duration_arias'] = result_arias['diagnostic']['s_duration']
            
            # ===== ENVELOPE: =====
            result_envelope = detect_coda_start(
                signal, t_s, origin_time=origin_time,
                sampling_rate=sampling_rate, method='envelope',
                threshold_arias=threshold_arias,
                threshold_envelope=threshold_envelope
            )
            df_full.loc[idx, 't_coda_envelope'] = result_envelope['t_coda']
            df_full.loc[idx, 's_duration_envelope'] = result_envelope['diagnostic']['s_duration']
            
            # ===== MEDIAN =====
            t_coda_median = np.median([
                rautian_cache[station]['t_coda'],
                result_arias['t_coda'],
                result_envelope['t_coda']
            ])
            df_full.loc[idx, 't_coda_median'] = t_coda_median
            df_full.loc[idx, 's_duration_median'] = t_coda_median - t_s
        
        except Exception as e:
            print(f"Warning: Coda detection failed for {station}-{component}: {e}")
            continue
    
    print("Done!")
    return df_full

def compute_coda_method_statistics(df_onsets_full, distance_bins=None):
    """
    Compute comprehensive statistics for coda onset method comparison.
    
    Calculates pairwise statistics (correlation, bias, agreement) for all
    method pairs, and stratifies results by epicentral distance bins.
    
    Parameters
    ----------
    df_onsets_full : pd.DataFrame
        Full onset detection results (66 rows) with columns:
        - STATION_CODE, COMPONENT
        - EPICENTRAL_DISTANCE_KM
        - t_coda_rautian, t_coda_arias, t_coda_envelope
        - s_duration_rautian, s_duration_arias, s_duration_envelope
    distance_bins : list of tuple, optional
        Distance bin edges as [(min1, max1), (min2, max2), ...]
        Default: [(0, 50), (50, 100), (100, 200)]
    
    Returns
    -------
    dict
        Nested dictionary with structure:
        - 'data': raw arrays (66 elements each)
        - 'pairwise': statistics for each method pair
        - 'by_distance': stratified statistics by distance bin
        - 'summary': dataset summary info
    
    Examples
    --------
    >>> stats = compute_coda_method_statistics(df_onsets_full)
    >>> print(f"Rautian-Arias correlation: {stats['pairwise']['rautian_arias']['correlation']:.3f}")
    >>> print(f"Mean bias: {stats['pairwise']['rautian_arias']['mean_diff']:.2f}s")
    """
    
    # Default distance bins
    if distance_bins is None:
        distance_bins = [(0, 50), (50, 100), (100, 200)]
    
    # Method names
    methods = ['rautian', 'arias', 'envelope']
    
    # Extract data (filter out NaN values)
    valid_mask = (
        df_onsets_full['t_coda_rautian'].notna() &
        df_onsets_full['t_coda_arias'].notna() &
        df_onsets_full['t_coda_envelope'].notna()
    )
    
    df_valid = df_onsets_full[valid_mask].copy()
    
    n_valid = len(df_valid)
    
    print(f"Computing statistics for {n_valid}/{len(df_onsets_full)} valid components")
    
    # Raw data arrays
    data = {
        'rautian': df_valid['t_coda_rautian'].values,
        'arias': df_valid['t_coda_arias'].values,
        'envelope': df_valid['t_coda_envelope'].values,
        'distance': df_valid['EPICENTRAL_DISTANCE_KM'].values,
        'station_codes': df_valid['STATION_CODE'].values,
        'component_codes': df_valid['COMPONENT'].values
    }
    
    # Compute pairwise statistics
    pairwise = {}
    
    method_pairs = [
        ('rautian', 'arias'),
        ('rautian', 'envelope'),
        ('arias', 'envelope')
    ]
    
    for method1, method2 in method_pairs:
        pair_name = f'{method1}_{method2}'
        
        x = data[method1]
        y = data[method2]
        
        # Differences and means
        diff = x - y
        mean = (x + y) / 2
        
        # Summary statistics
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        # Limits of agreement (Bland-Altman)
        loa_lower = mean_diff - 1.96 * std_diff
        loa_upper = mean_diff + 1.96 * std_diff
        
        # Correlation
        r, p_value = pearsonr(x, y)
        
        # Error metrics
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        
        # Linear fit (for plotting)
        coeffs = np.polyfit(x, y, deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        
        pairwise[pair_name] = {
            'diff': diff,
            'mean': mean,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'limits_of_agreement': (loa_lower, loa_upper),
            'correlation': r,
            'p_value': p_value,
            'rmse': rmse,
            'mae': mae,
            'slope': slope,
            'intercept': intercept,
            'n': n_valid
        }
    
    # Stratify by distance bins
    by_distance = {
        'bins': distance_bins
    }
    
    for method1, method2 in method_pairs:
        pair_name = f'{method1}_{method2}'
        bin_stats = []
        
        for bin_min, bin_max in distance_bins:
            # Filter by distance
            mask_bin = (data['distance'] >= bin_min) & (data['distance'] < bin_max)
            
            if mask_bin.sum() == 0:
                # Empty bin
                bin_stats.append({
                    'bin': (bin_min, bin_max),
                    'n': 0,
                    'mean_diff': np.nan,
                    'std_diff': np.nan,
                    'correlation': np.nan,
                    'rmse': np.nan
                })
                continue
            
            x_bin = data[method1][mask_bin]
            y_bin = data[method2][mask_bin]
            diff_bin = x_bin - y_bin
            
            # Statistics for this bin
            n_bin = mask_bin.sum()
            mean_diff_bin = np.mean(diff_bin)
            std_diff_bin = np.std(diff_bin, ddof=1) if n_bin > 1 else np.nan
            rmse_bin = np.sqrt(np.mean(diff_bin**2))
            
            # Correlation (only if enough points)
            if n_bin >= 3:
                r_bin, _ = pearsonr(x_bin, y_bin)
            else:
                r_bin = np.nan
            
            bin_stats.append({
                'bin': (bin_min, bin_max),
                'n': n_bin,
                'mean_diff': mean_diff_bin,
                'std_diff': std_diff_bin,
                'correlation': r_bin,
                'rmse': rmse_bin
            })
        
        by_distance[pair_name] = bin_stats
    
    # Summary info
    summary = {
        'n_components': n_valid,
        'n_stations': df_valid['STATION_CODE'].nunique(),
        'methods': methods,
        'distance_range': (data['distance'].min(), data['distance'].max())
    }
    
    # Print summary
    print("\n" + "="*70)
    print("CODA METHOD COMPARISON STATISTICS")
    print("="*70)
    print(f"\nDataset: {summary['n_components']} components from {summary['n_stations']} stations")
    print(f"Distance range: {summary['distance_range'][0]:.1f} - {summary['distance_range'][1]:.1f} km")
    
    print("\nPairwise correlations:")
    for pair_name, pair_stats in pairwise.items():
        print(f"  {pair_name:20s}: r = {pair_stats['correlation']:.3f} (p < {pair_stats['p_value']:.1e})")
    
    print("\nMean differences (bias):")
    for pair_name, pair_stats in pairwise.items():
        method1, method2 = pair_name.split('_')
        print(f"  {method1} - {method2:10s}: {pair_stats['mean_diff']:+.2f} ± {pair_stats['std_diff']:.2f} s")
    
    print("\nError metrics:")
    for pair_name, pair_stats in pairwise.items():
        print(f"  {pair_name:20s}: RMSE = {pair_stats['rmse']:.2f} s, MAE = {pair_stats['mae']:.2f} s")
    
    print("\nAgreement by distance (Rautian - Arias):")
    for bin_stat in by_distance['rautian_arias']:
        bin_min, bin_max = bin_stat['bin']
        print(f"  {bin_min:3.0f}-{bin_max:3.0f} km: n={bin_stat['n']:2d}, "
              f"bias={bin_stat['mean_diff']:+.2f}s, r={bin_stat['correlation']:.3f}")
    
    print("="*70)
    
    # Return complete statistics dictionary
    return {
        'data': data,
        'pairwise': pairwise,
        'by_distance': by_distance,
        'summary': summary
    }