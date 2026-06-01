"""
AR-AIC phase detection for seismic signals.

This module implements onset detection using the Autoregressive-Akaike
Information Criterion (AR-AIC) method via ObsPy, plus multiple coda onset
detection algorithms.

Functions
---------
P/S wave detection:
    detect_onsets_arpick : Detect P and S arrivals using AR-AIC with search windows

Coda onset detection:
    detect_coda_start : Single-method coda detection
    detect_coda_start_all_methods : Run all methods for comparison
    add_coda_onsets_to_dataframe : Populate DataFrame with coda times

Statistical validation:
    compute_coda_method_statistics : Compare coda detection methods

Coda Detection Methods
----------------------
Rautian (1978): Lapse time = 2 × S-wave travel time
    - Theoretical, assumes homogeneous scattering
    - Independent of signal amplitude
    - Fast, no waveform processing needed

Arias Intensity (D5-95): Cumulative energy threshold
    - Empirical, based on strong-motion engineering
    - Measures significant duration of ground shaking
    - Standard in ESM flatfile (Lanzano et al., 2019)

Envelope decay: Amplitude-based threshold
    - Signal envelope drops below threshold (typically 20-30% of peak)
    - Common in strong-motion processing (Boore & Bommer, 2005)
    - Sensitive to noise and late-arriving scattered waves

Median: Robust combination of all three methods
    - Reduces impact of outliers
    - Requires all methods to succeed

Output Format
-------------
All functions use dual representation (samples + seconds) to avoid
rounding artifacts. Columns created:
- t_p/s_detected_samples, t_p/s_detected_seconds
- t_coda_<method>_samples, t_coda_<method>_seconds
- Legacy aliases without suffix for backward compatibility

References
----------
Leonard, M., & Kennett, B. L. N. (1999). "Multi-component autoregressive 
    techniques for the analysis of seismograms." Physics of the Earth and 
    Planetary Interiors, 113(1-4), 247-263.
Rautian, T. G., & Khalturin, V. I. (1978). "The use of the coda for
    determination of the earthquake source spectrum." Bulletin of the
    Seismological Society of America, 68(4), 923-948.
Boore, D. M., & Bommer, J. J. (2005). "Processing of strong-motion
    accelerograms: needs, options and consequences." Soil Dynamics and
    Earthquake Engineering, 25(2), 93-115.
Lanzano, G., et al. (2019). "The pan-European Engineering Strong Motion
    (ESM) flatfile: compilation criteria and data statistics." Bulletin of
    Earthquake Engineering, 17, 561-582.
    
Examples
--------
>>> # Detect P and S onsets
>>> df_stations = detect_onsets_arpick(signals_dict, df_meta_stations)
>>> 
>>> # Add coda onsets (all methods)
>>> df_full = add_coda_onsets_to_dataframe(df_full, signals_dict)
>>> 
>>> # Compare methods statistically
>>> stats = compute_coda_method_statistics(df_full)
>>> print(f"Rautian-Arias correlation: {stats['pairwise']['rautian_arias']['correlation']:.3f}")
"""

import numpy as np
import pandas as pd
from obspy.signal.trigger import ar_pick
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr
from typing import Dict, Any, Optional, Union, List, Tuple

def detect_onsets_arpick(
    signals_dict: Dict[str, Dict[str, np.ndarray]], 
    df_meta_stations: pd.DataFrame,
    sampling_rate: float = 200,
    unit: str = 'samples',
    p_window_before: float = 5, 
    p_window_after: float = 5,
    s_window_before: float = 7, 
    s_window_after: float = 7
) -> pd.DataFrame:
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
            if t_p_theo_sec >= 0:
                df_meta_stations.loc[idx, 'p_residual_seconds'] = t_p_detected_sec - t_p_theo_sec

        if s_success:
            if has_theo_samples:
                t_s_theo_sec = t_s_theo_samp / sampling_rate
            if t_s_theo_sec >= 0:
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
        p_res = df_meta_stations['p_residual_seconds'].dropna()
        print(f"  P residuals: {p_res.mean():.2f} ± {p_res.std():.2f} s  "
            f"[{p_res.min():.2f}, {p_res.max():.2f}]")

    if n_s_success > 0:
        s_res = df_meta_stations['s_residual_seconds'].dropna()
        print(f"  S residuals: {s_res.mean():.2f} ± {s_res.std():.2f} s  "
            f"[{s_res.min():.2f}, {s_res.max():.2f}]")
    
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

def detect_coda_start(
    signal: np.ndarray, 
    t_s_detected: Union[int, float], 
    t_p_detected: Optional[Union[int, float]] = None, 
    origin_time: Optional[Union[int, float]] = None,
    sampling_rate: float = 200, 
    method: str = 'rautian',
    unit: str = 'samples',
    threshold_arias: float = 0.95, 
    threshold_envelope: float = 0.3
) -> Dict[str, Any]:
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


def detect_coda_start_all_methods(
    signal: np.ndarray, 
    t_s_detected: Union[int, float], 
    t_p_detected: Optional[Union[int, float]] = None, 
    origin_time: Optional[Union[int, float]] = None,
    sampling_rate: float = 200, 
    unit: str = 'samples',
    threshold_arias: float = 0.95,
    threshold_envelope: float = 0.3
) -> Dict[str, Dict[str, Any]]:
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
    >>> results = detect_coda_start_all_methods(
    ...     signal, t_s=3040, origin_time=1640,
    ...     unit='samples'
    ... )
    >>> for method, res in results.items():
    ...     print(f"{method}: t_coda={res['t_coda_samples']} samp "
    ...           f"({res['t_coda_seconds']:.2f}s)")
    
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

def add_coda_onsets_to_dataframe(
    df_full: pd.DataFrame, 
    signals_dict: Dict[str, Dict[str, np.ndarray]],
    threshold_arias: float = 0.95,
    threshold_envelope: float = 0.3,
    sampling_rate: float = 200,
    unit: str = 'samples'
) -> pd.DataFrame:
    """
    Populate coda onset columns in df_full with dual representation.
    
    Optimized: Rautian method computed once per station (not per component).
    Creates both _samples and _seconds columns for all methods.
    
    Parameters
    ----------
    df_full : pd.DataFrame
        DataFrame with columns:
        - STATION_CODE, COMPONENT
        - t_s_detected_samples, t_s_detected_seconds (or legacy t_s_detected)
        - origin_time_samples, origin_time_seconds (or legacy origin_time)
    signals_dict : dict
        Nested dict: {station: {component: array, 'time': array}}
    threshold_arias : float, optional
        Arias Intensity threshold (default: 0.95 for D5-95)
    threshold_envelope : float, optional
        Envelope decay threshold (default: 0.3)
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
    unit : {'samples', 'seconds'}, optional
        Preferred unit for computation (default: 'samples')
    
    Returns
    -------
    pd.DataFrame
        df_full with added columns (dual representation):
        - t_coda_<method>_samples, t_coda_<method>_seconds (int, float)
        - s_duration_<method>_samples, s_duration_<method>_seconds (int, float)
        - t_coda_<method>, s_duration_<method> (legacy aliases based on unit)
        
        Methods: rautian, arias, envelope, median
    
    Notes
    -----
    Rautian method is station-dependent only (same for all components),
    so it's computed once per station and cached.
    
    Arias and Envelope are component-dependent (use actual signal),
    so they're computed per component.
    """
    
    # Auto-detect input columns (prefer samples, fallback seconds)
    if 't_s_detected_samples' in df_full.columns:
        t_s_col_samples = 't_s_detected_samples'
        t_s_col_seconds = 't_s_detected_seconds'
        has_onset_samples = True
    elif 't_s_detected_seconds' in df_full.columns:
        t_s_col_seconds = 't_s_detected_seconds'
        t_s_col_samples = None
        has_onset_samples = False
    elif 't_s_detected' in df_full.columns:
        t_s_col_seconds = 't_s_detected'
        t_s_col_samples = None
        has_onset_samples = False
    else:
        raise ValueError(
            "No t_s_detected columns found. Expected 't_s_detected_samples'/"
            "'t_s_detected_seconds' or 't_s_detected'"
        )
    
    # Check for origin_time columns
    if 'origin_time_samples' in df_full.columns:
        origin_col_samples = 'origin_time_samples'
        origin_col_seconds = 'origin_time_seconds'
        has_origin_samples = True
    elif 'origin_time_seconds' in df_full.columns:
        origin_col_seconds = 'origin_time_seconds'
        origin_col_samples = None
        has_origin_samples = False
    elif 'origin_time' in df_full.columns:
        origin_col_seconds = 'origin_time'
        origin_col_samples = None
        has_origin_samples = False
    else:
        raise ValueError(
            "No origin_time columns found. Expected 'origin_time_samples'/"
            "'origin_time_seconds' or 'origin_time'"
        )
    
    methods = ['rautian', 'arias', 'envelope', 'median']
    
    # Initialize all columns (dual representation)
    for method in methods:
        df_full[f't_coda_{method}_samples'] = pd.NA
        df_full[f't_coda_{method}_seconds'] = np.nan
        df_full[f's_duration_{method}_samples'] = pd.NA
        df_full[f's_duration_{method}_seconds'] = np.nan
    
    print("Pre-computing Rautian coda onset per station (dual representation)...")
    rautian_cache = {}
    
    for station in df_full['STATION_CODE'].unique():
        station_row = df_full[df_full['STATION_CODE'] == station].iloc[0]
        
        # Get t_s and origin_time in both representations
        if has_onset_samples:
            t_s_samp = int(station_row[t_s_col_samples])
            t_s_sec = float(station_row[t_s_col_seconds])
        else:
            t_s_sec = float(station_row[t_s_col_seconds])
            t_s_samp = int(np.round(t_s_sec * sampling_rate))
        
        if has_origin_samples:
            origin_samp = int(station_row[origin_col_samples])
            origin_sec = float(station_row[origin_col_seconds])
        else:
            origin_sec = float(station_row[origin_col_seconds])
            origin_samp = int(np.round(origin_sec * sampling_rate))
        
        # Rautian formula: t_coda = origin + 2*(t_s - origin)
        s_lapse_time_sec = t_s_sec - origin_sec
        s_lapse_time_samp = t_s_samp - origin_samp
        
        t_coda_rautian_sec = origin_sec + 2.0 * s_lapse_time_sec
        t_coda_rautian_samp = origin_samp + 2 * s_lapse_time_samp
        
        s_duration_rautian_sec = t_coda_rautian_sec - t_s_sec
        s_duration_rautian_samp = t_coda_rautian_samp - t_s_samp
        
        rautian_cache[station] = {
            't_coda_samples': t_coda_rautian_samp,
            't_coda_seconds': t_coda_rautian_sec,
            's_duration_samples': s_duration_rautian_samp,
            's_duration_seconds': s_duration_rautian_sec
        }
    
    print(f"Computed Rautian for {len(rautian_cache)} stations")
    
    print(f"Calculating Arias, Envelope, and Median coda onsets for {len(df_full)} components...")
    
    n_processed = 0
    n_failed = 0
    
    for idx, row in df_full.iterrows():
        station = row['STATION_CODE']
        component = row['COMPONENT']
        
        if station not in signals_dict or component not in signals_dict[station]:
            n_failed += 1
            continue
        
        signal = signals_dict[station][component]
        
        # Get t_s and origin_time (prefer samples for passing to detect_coda_start)
        if has_onset_samples:
            t_s_input = int(row[t_s_col_samples])
            input_unit = 'samples'
        else:
            t_s_input = float(row[t_s_col_seconds])
            input_unit = 'seconds'
        
        if has_origin_samples:
            origin_input = int(row[origin_col_samples])
        else:
            origin_input = float(row[origin_col_seconds])
        
        try:
            # ===== RAUTIAN: Use cached values =====
            if station in rautian_cache:
                df_full.loc[idx, 't_coda_rautian_samples'] = rautian_cache[station]['t_coda_samples']
                df_full.loc[idx, 't_coda_rautian_seconds'] = rautian_cache[station]['t_coda_seconds']
                df_full.loc[idx, 's_duration_rautian_samples'] = rautian_cache[station]['s_duration_samples']
                df_full.loc[idx, 's_duration_rautian_seconds'] = rautian_cache[station]['s_duration_seconds']
            
            # ===== ARIAS: Component-specific =====
            result_arias = detect_coda_start(
                signal, t_s_input, origin_time=origin_input,
                sampling_rate=sampling_rate, method='arias',
                unit=input_unit,
                threshold_arias=threshold_arias,
                threshold_envelope=threshold_envelope
            )
            df_full.loc[idx, 't_coda_arias_samples'] = result_arias['t_coda_samples']
            df_full.loc[idx, 't_coda_arias_seconds'] = result_arias['t_coda_seconds']
            df_full.loc[idx, 's_duration_arias_samples'] = result_arias['diagnostic']['s_duration_samples']
            df_full.loc[idx, 's_duration_arias_seconds'] = result_arias['diagnostic']['s_duration_seconds']
            
            # ===== ENVELOPE: Component-specific =====
            result_envelope = detect_coda_start(
                signal, t_s_input, origin_time=origin_input,
                sampling_rate=sampling_rate, method='envelope',
                unit=input_unit,
                threshold_arias=threshold_arias,
                threshold_envelope=threshold_envelope
            )
            df_full.loc[idx, 't_coda_envelope_samples'] = result_envelope['t_coda_samples']
            df_full.loc[idx, 't_coda_envelope_seconds'] = result_envelope['t_coda_seconds']
            df_full.loc[idx, 's_duration_envelope_samples'] = result_envelope['diagnostic']['s_duration_samples']
            df_full.loc[idx, 's_duration_envelope_seconds'] = result_envelope['diagnostic']['s_duration_seconds']
            
            # ===== MEDIAN: Compute from all three methods =====
            # Median in samples
            t_coda_median_samp = int(np.median([
                rautian_cache[station]['t_coda_samples'],
                result_arias['t_coda_samples'],
                result_envelope['t_coda_samples']
            ]))
            
            # Median in seconds
            t_coda_median_sec = np.median([
                rautian_cache[station]['t_coda_seconds'],
                result_arias['t_coda_seconds'],
                result_envelope['t_coda_seconds']
            ])
            
            # Get t_s for duration calculation
            if has_onset_samples:
                t_s_samp = int(row[t_s_col_samples])
                t_s_sec = float(row[t_s_col_seconds])
            else:
                t_s_sec = float(row[t_s_col_seconds])
                t_s_samp = int(np.round(t_s_sec * sampling_rate))
            
            s_duration_median_samp = t_coda_median_samp - t_s_samp
            s_duration_median_sec = t_coda_median_sec - t_s_sec
            
            df_full.loc[idx, 't_coda_median_samples'] = t_coda_median_samp
            df_full.loc[idx, 't_coda_median_seconds'] = t_coda_median_sec
            df_full.loc[idx, 's_duration_median_samples'] = s_duration_median_samp
            df_full.loc[idx, 's_duration_median_seconds'] = s_duration_median_sec
            
            n_processed += 1
            
        except Exception as e:
            print(f"\nWarning: Coda detection failed for {station}-{component}: {e}")
            n_failed += 1
            continue
    
    # ===== CREATE LEGACY COLUMNS (aliases based on unit preference) =====
    for method in methods:
        if unit == 'samples':
            df_full[f't_coda_{method}'] = df_full[f't_coda_{method}_samples']
            df_full[f's_duration_{method}'] = df_full[f's_duration_{method}_samples']
        else:  # unit == 'seconds'
            df_full[f't_coda_{method}'] = df_full[f't_coda_{method}_seconds']
            df_full[f's_duration_{method}'] = df_full[f's_duration_{method}_seconds']
    
    # Sostituisci il blocco finale con:
    print(f"\nCoda onsets computed for {n_processed}/{len(df_full)} components "
      f"({n_failed} failed)")
    print(f"  Created columns with dual representation (_samples + _seconds)")
    print(f"  Legacy columns (no suffix) point to: {unit}")
    
    return df_full

def compute_coda_method_statistics(
    df_onsets_full: pd.DataFrame, 
    distance_bins: Optional[List[Tuple[float, float]]] = None, 
    unit: str = 'seconds',
    sampling_rate: float = 200
) -> Dict[str, Any]:
    """
Compute comprehensive statistics for coda onset method comparison.

Calculates pairwise correlations, Bland-Altman agreement metrics, error
statistics (RMSE, MAE), and distance-stratified analysis for all three
coda detection methods (Rautian, Arias, Envelope).

Parameters
----------
df_onsets_full : pd.DataFrame
    Component-level DataFrame with columns:
    - STATION_CODE, COMPONENT
    - EPICENTRAL_DISTANCE_KM
    - t_coda_<method>_samples, t_coda_<method>_seconds (or t_coda_<method>)
    Methods: rautian, arias, envelope
distance_bins : list of tuple, optional
    Distance bins for stratified analysis (default: [(0, 50), (50, 100), (100, 200)])
    Each tuple: (bin_min_km, bin_max_km)
unit : {'samples', 'seconds'}, optional
    Which representation to use for statistics (default: 'seconds')
    Note: Statistics are always computed in seconds as they represent physical times
sampling_rate : float, optional
    Sampling rate in Hz (used if input columns are in samples, default: 200)

Returns
-------
dict
    Dictionary with keys:
    - 'data': Raw arrays for all methods and metadata
    - 'pairwise': Pairwise comparison statistics for each method pair
        - 'diff', 'mean': Arrays of differences and means
        - 'mean_diff', 'std_diff': Bias and scatter
        - 'limits_of_agreement': Bland-Altman LoA (lower, upper)
        - 'correlation', 'p_value': Pearson correlation
        - 'rmse', 'mae': Error metrics
        - 'slope', 'intercept': Linear fit parameters
        - 'n': Sample size
    - 'by_distance': Stratified statistics per distance bin
        - For each method pair: list of bin statistics
    - 'summary': Dataset-level summary (n_components, n_stations, distance_range)

Notes
-----
Method pairs analyzed:
- Rautian vs Arias
- Rautian vs Envelope  
- Arias vs Envelope

Statistics include:
- Bland-Altman limits of agreement (±1.96 SD)
- Pearson correlation coefficients
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Distance-stratified bias and correlation

References
----------
Bland, J. M., & Altman, D. G. (1986). "Statistical methods for assessing
    agreement between two methods of clinical measurement." The Lancet,
    327(8476), 307-310.

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
    
    # ===== AUTO-DETECT COLUMN SUFFIXES =====
    # Statistics should be in seconds (physical interpretation)
    # but support both column naming schemes
    
    method_cols = {}
    for method in methods:
        col_samples = f't_coda_{method}_samples'
        col_seconds = f't_coda_{method}_seconds'
        col_legacy = f't_coda_{method}'
        
        if col_seconds in df_onsets_full.columns:
            method_cols[method] = col_seconds
        elif col_legacy in df_onsets_full.columns:
            method_cols[method] = col_legacy
        elif col_samples in df_onsets_full.columns:
            # Convert samples to seconds for statistics
            print(f"Warning: Using {col_samples}, converting to seconds for statistics")
            # Create temporary seconds column
            df_onsets_full[col_seconds] = df_onsets_full[col_samples] / sampling_rate
            method_cols[method] = col_seconds
        else:
            raise ValueError(f"No t_coda columns found for method '{method}'")
    
    # Extract data (filter out NaN values)
    valid_mask = (
        df_onsets_full[method_cols['rautian']].notna() &
        df_onsets_full[method_cols['arias']].notna() &
        df_onsets_full[method_cols['envelope']].notna()
    )
    
    df_valid = df_onsets_full[valid_mask].copy()
    
    n_valid = len(df_valid)
    
    print(f"Computing statistics for {n_valid}/{len(df_onsets_full)} valid components")
    print(f"Using columns: {list(method_cols.values())}")
    
    # Raw data arrays
    data = {
        'rautian': df_valid[method_cols['rautian']].values,
        'arias': df_valid[method_cols['arias']].values,
        'envelope': df_valid[method_cols['envelope']].values,
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

def compute_coda_end(
    signal: np.ndarray,
    t_s_samples: int,
    t_coda_samples: int,
    s_window_signal: np.ndarray,
    threshold_factor: float = 0.10,
    stability_duration: float = 2.0,
    sampling_rate: float = 200.0,
    smoothing_window: float = 1
) -> int:
    """
    Determine end of active coda using energy decay threshold.
    
    Finds the point where signal amplitude drops below a threshold relative
    to S-wave peak and remains low for a sustained period, indicating
    transition from seismic coda to ambient noise.
    
    Parameters
    ----------
    signal : np.ndarray
        Full signal time series
    t_s_samples : int
        S-wave onset time in samples
    t_coda_samples : int
        Coda onset time in samples
    s_window_signal : np.ndarray
        S-wave window signal (for reference amplitude)
    threshold_factor : float, optional
        Threshold as fraction of S-wave peak amplitude (default: 0.10 = 10%)
    stability_duration : float, optional
        Duration in seconds that amplitude must remain below threshold (default: 2.0s)
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
    smoothing_window : float, optional
        Window size in seconds for envelope smoothing (default: 0.5s)
    
    Returns
    -------
    t_coda_end_samples : int
        End of active coda in samples (start of post-event window)
        Returns len(signal) if threshold never crossed
    
    Notes
    -----
    Algorithm:
    1. Calculate threshold = threshold_factor × max(|S-wave|)
    2. Compute smoothed envelope of signal after t_coda
    3. Find first point where envelope < threshold for stability_duration
    4. Return that point, or end of signal if not found
    
    Physical interpretation:
    - Coda decays exponentially from S-wave arrival
    - When amplitude drops to ~10% of peak and stays low, scattered energy
      has dissipated and ambient noise dominates
    - Typical values: threshold_factor=0.05-0.15, stability=1-3s
    
    All time parameters use explicit _samples suffix to avoid unit ambiguity.
    This follows the codebase convention established in onset_detection.py.
    
    Examples
    --------
    >>> t_end = compute_coda_end(
    ...     signal, 
    ...     t_s_samples=2960, 
    ...     t_coda_samples=4140, 
    ...     s_window_signal=s_wave,
    ...     threshold_factor=0.10
    ... )
    >>> print(f"Coda ends at sample {t_end} ({t_end/200:.2f}s)")
    Coda ends at sample 8340 (41.70s)
    """
    
     # Extract post-coda region
    post_coda_signal = signal[t_coda_samples:]
    
    if len(post_coda_signal) == 0:
        return len(signal)
    
    # Compute envelope using Hilbert transform
    envelope = np.abs(hilbert(post_coda_signal))
    
    # Smooth envelope with 1-second moving average
    smooth_window_samples = int(smoothing_window * sampling_rate)
    if smooth_window_samples > 1:
        envelope_smooth = uniform_filter1d(envelope, size=smooth_window_samples, mode='nearest')
    else:
        envelope_smooth = envelope
    
    # Find peak envelope in early coda (first 5s after coda onset)
    search_window_samples = min(int(5.0 * sampling_rate), len(envelope_smooth))
    
    if search_window_samples == 0:
        return len(signal)
    
    peak_envelope = np.max(envelope_smooth[:search_window_samples])
    
    if peak_envelope == 0:
        return len(signal)
    
    # Calculate threshold
    threshold = threshold_factor * peak_envelope
    
    # Find points below threshold
    below_threshold = envelope_smooth < threshold
    
    if not np.any(below_threshold):
        return len(signal)
    
    # Stability requirement
    stability_samples = int(stability_duration * sampling_rate)
    
    # ═══════════════════════════════════════════════════════════════
    # FIX: Enforce minimum coda duration (1 second)
    # ═══════════════════════════════════════════════════════════════
    min_coda_duration_samples = int(1.0 * sampling_rate)
    
    # If signal too short for both minimum duration AND stability check
    if len(below_threshold) < (min_coda_duration_samples + stability_samples):
        # Fallback: find first crossing without stability requirement
        first_crossing = np.where(below_threshold)[0]
        if len(first_crossing) > 0:
            # But still enforce minimum duration
            crossing_idx = max(first_crossing[0], min_coda_duration_samples)
            if crossing_idx < len(below_threshold):
                return t_coda_samples + crossing_idx
        return len(signal)
    
    # Start search AFTER minimum duration
    start_idx = min_coda_duration_samples
    
    # Scan for first point where signal stays below threshold for stability_duration
    for i in range(start_idx, len(below_threshold) - stability_samples):
        if np.all(below_threshold[i:i + stability_samples]):
            t_coda_end_samples = t_coda_samples + i
            return t_coda_end_samples
    
    # Threshold crossed but not sustained: return end of signal
    return len(signal)

def compute_coda_end_arias(
    signal: np.ndarray,
    t_s_samples: int,
    t_coda_samples: int,
    threshold_end: float = 0.995,  # 99.5% dell'energia
    sampling_rate: float = 200.0,
    window_start_samples: int = 0  # Stesso dell'onset (5s prima di P o origine)
) -> int:
    """
    Determine end of active coda using Arias Intensity energy accumulation.
    
    Mirrors the Arias D5-D95 method used for coda onset detection, but uses
    a higher energy threshold (e.g., D99.5) to detect the end of significant
    seismic energy and transition to ambient noise.
    
    Parameters
    ----------
    signal : np.ndarray
        Full signal time series
    t_s_samples : int
        S-wave onset time in samples
    t_coda_samples : int
        Coda onset time in samples (typically D95)
    threshold_end : float, optional
        Energy fraction threshold for coda end (default: 0.995 = 99.5%)
        Common values:
        - 0.990 (D99): more conservative, shorter coda
        - 0.995 (D99.5): balanced (recommended)
        - 0.999 (D99.9): very long coda, may include noise tail
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
    window_start_samples : int, optional
        Start sample for Arias integration window (default: 0)
        Should match the value used for coda onset detection
        (typically 5s before P-onset or event origin)
    
    Returns
    -------
    t_coda_end_samples : int
        End of active coda in samples (D99.5 or specified threshold)
        Returns len(signal) if threshold never reached
    
    Notes
    -----
    Algorithm (consistent with coda onset detection):
    1. Compute Arias Intensity on same window as onset detection
    2. Normalize to [0, 1]
    3. Find time when AI reaches threshold_end (e.g., 99.5%)
    4. Return that time as coda end
    
    Physical interpretation:
    - D95 (onset): 95% of total seismic energy released
    - D99.5 (end): 99.5% of energy released, remaining 0.5% is ambient noise
    - The 4.5% energy band [D95, D99.5] represents the coda decay phase
    
    Advantages over envelope threshold:
    - Physically motivated (energy-based)
    - Robust to amplitude fluctuations
    - Consistent with coda onset methodology
    - No arbitrary threshold tuning
    - No stability duration requirement
    
    References
    ----------
    Lanzano, G., et al. (2019). "The pan-European Engineering Strong Motion 
    (ESM) flatfile: compilation criteria and data statistics." 
    Bulletin of Earthquake Engineering, 17(2), 561-582.
    
    Examples
    --------
    >>> t_end = compute_coda_end_arias(
    ...     signal,
    ...     t_s_samples=2960,
    ...     t_coda_samples=4140,
    ...     threshold_end=0.995,
    ...     window_start_samples=1440
    ... )
    >>> coda_duration = (t_end - 4140) / 200.0
    >>> print(f"Coda duration: {coda_duration:.2f}s")
    Coda duration: 8.34s
    """
    
    # Extract signal window (same as coda onset detection)
    signal_window = signal[window_start_samples:]
    
    if len(signal_window) == 0:
        return len(signal)
    
    # Compute Arias Intensity on full signal window
    dt = 1.0 / sampling_rate
    g = 9.81  # m/s²
    arias_cumsum = (np.pi / (2 * g)) * np.cumsum(signal_window**2) * dt
    
    # Handle pathological case (zero signal)
    if arias_cumsum[-1] == 0 or len(arias_cumsum) == 0:
        return len(signal)
    
    # Normalize to [0, 1]
    arias_norm = arias_cumsum / arias_cumsum[-1]
    
    # Find time when threshold is reached
    idx_threshold_rel = np.argmax(arias_norm >= threshold_end)
    
    # Check if threshold was never reached
    if idx_threshold_rel == 0 and arias_norm[0] < threshold_end:
        # Threshold > 1 or signal ends before reaching it
        return len(signal)
    
    # Convert to absolute sample index
    t_coda_end_samples = window_start_samples + idx_threshold_rel
    
    # Sanity check: coda_end must be after coda_onset
    if t_coda_end_samples <= t_coda_samples:
        # This shouldn't happen if threshold_end > threshold_onset (e.g., 0.995 > 0.95)
        # But enforce it as safety
        return len(signal)
    
    # Enforce minimum coda duration (1 second)
    min_coda_duration_samples = int(1.0 * sampling_rate)
    if (t_coda_end_samples - t_coda_samples) < min_coda_duration_samples:
        # Coda too short: extend to minimum duration or signal end
        t_coda_end_samples = min(
            t_coda_samples + min_coda_duration_samples,
            len(signal)
        )
    
    return t_coda_end_samples

def add_coda_end_to_dataframe(
    df_onsets: pd.DataFrame,
    signals_dict: Dict,
    coda_methods: List[str] = ['rautian', 'arias', 'envelope', 'median'],
    threshold_factor: float = 0.10,
    threshold_end_arias: float = 0.995,
    stability_duration: float = 2.0,
    sampling_rate: float = 200.0,
    smoothing_window: float = 0.5
) -> pd.DataFrame:
    """
    Calculate coda end times and add to DataFrame.
    
    For each coda method, computes the end of active coda (transition to
    post-event noise) using energy decay threshold. Follows the same pattern
    as add_coda_onsets_to_dataframe(): iterates over station-component pairs,
    calculates coda end for each method, and populates DataFrame with dual
    representation (samples + seconds).
    
    Parameters
    ----------
    df_onsets : pd.DataFrame
        Component-level DataFrame with columns:
        - STATION_CODE, COMPONENT
        - t_s_detected_samples (or _seconds, or legacy t_s_detected)
        - t_coda_<method>_samples (or _seconds, or legacy) for each method
    signals_dict : dict
        Nested dictionary {station: {component: array, 'time': array}}
        Structure from convert_signals_to_dict()
    coda_methods : list of str, optional
        Which coda methods to process (default: ['rautian', 'arias', 'envelope', 'median'])
    threshold_factor : float, optional
        Threshold as fraction of S-wave peak amplitude (default: 0.10 = 10%)
        Lower values (0.05) → longer coda, shorter post-event
        Higher values (0.15) → shorter coda, longer post-event
    threshold_end_arias : float, optional
        Energy threshold for Arias method (default: 0.995 = 99.5%)
    stability_duration : float, optional
        Duration in seconds that amplitude must remain below threshold (default: 2.0s)
        Prevents false positives from transient noise spikes
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
    smoothing_window : float, optional
        Envelope smoothing window in seconds (default: 0.5s)
        Larger values → smoother envelope, less sensitive to high-frequency noise
    
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns (dual representation):
        - t_coda_end_<method>_samples : int (sample index)
        - t_coda_end_<method>_seconds : float (time in seconds)
        - t_coda_end_<method> : float (legacy alias, points to seconds)
        
        One set of columns per coda method.
    
    Notes
    -----
    Architectural pattern (matches add_coda_onsets_to_dataframe):
    1. Detection happens here (calls compute_coda_end internally)
    2. Results stored in DataFrame with dual representation
    3. segment_all_signals() reads pre-computed values
    
    This maintains separation of concerns:
    - Detection: add_coda_end_to_dataframe()
    - Segmentation: segment_signal_into_windows()
    
    Missing values (NaN) in output indicate:
    - Station-component not in signals_dict
    - t_s or t_coda missing/invalid
    - S-wave window empty or invalid
    - compute_coda_end() raised exception
    
    Physical interpretation:
    - t_coda_end marks transition from seismic coda to ambient noise
    - Coda window: [t_coda, t_coda_end) — scattered seismic energy
    - Post-event window: [t_coda_end, end] — return to background noise
    
    Examples
    --------
    >>> # After detecting P/S and coda onsets
    >>> df_full = add_coda_onsets_to_dataframe(df_full, signals_dict)
    >>> 
    >>> # Add coda end times
    >>> df_full = add_coda_end_to_dataframe(
    ...     df_full, 
    ...     signals_dict,
    ...     threshold_factor=0.10,
    ...     stability_duration=2.0
    ... )
    >>> 
    >>> # Check results
    >>> print(df_full[['STATION_CODE', 'COMPONENT', 
    ...                't_coda_rautian_seconds', 
    ...                't_coda_end_rautian_seconds']])
    
    >>> # Calculate coda durations
    >>> df_full['coda_duration'] = (
    ...     df_full['t_coda_end_rautian_seconds'] - 
    ...     df_full['t_coda_rautian_seconds']
    ... )
    """
    
    # Initialize columns for each method (dual representation)
    for method in coda_methods:
        df_onsets[f't_coda_end_{method}_samples'] = pd.NA
        df_onsets[f't_coda_end_{method}_seconds'] = np.nan
    
    # Auto-detect column naming scheme for t_s
    has_samples_cols = 't_s_detected_samples' in df_onsets.columns
    has_seconds_cols = 't_s_detected_seconds' in df_onsets.columns
    
    if has_samples_cols:
        t_s_col = 't_s_detected_samples'
        t_s_unit = 'samples'
    elif has_seconds_cols:
        t_s_col = 't_s_detected_seconds'
        t_s_unit = 'seconds'
    else:
        t_s_col = 't_s_detected'
        t_s_unit = 'seconds'  # legacy default
    
    print(f"\n{'='*70}")
    print("COMPUTING CODA END TIMES")
    print(f"{'='*70}")
    print(f"Processing {len(df_onsets)} components...")
    print(f"Methods: {coda_methods}")
    print(f"Threshold: {threshold_factor:.0%} of S-wave peak amplitude")
    print(f"Stability: {stability_duration:.1f}s")
    print(f"Smoothing: {smoothing_window:.1f}s")
    print(f"Sampling rate: {sampling_rate:.0f} Hz")
    print(f"Working with: {t_s_col}")
    print("\nProcessing: ", end="", flush=True)
    
    n_computed = {method: 0 for method in coda_methods}
    n_skipped_no_signal = 0
    n_skipped_missing_onset = 0
    n_skipped_error = 0
    
    for idx, row in df_onsets.iterrows():
        print(".", end="", flush=True)
        
        station = row['STATION_CODE']
        component = row['COMPONENT']
        
        # Check if station-component exists in signals_dict
        if station not in signals_dict:
            n_skipped_no_signal += 1
            continue
        
        if component not in signals_dict[station]:
            n_skipped_no_signal += 1
            continue
        
        signal = signals_dict[station][component]
        
        # Extract t_s
        t_s_value = row[t_s_col]
        if pd.isna(t_s_value):
            n_skipped_missing_onset += 1
            continue
        
        # Convert t_s to samples
        if t_s_unit == 'samples':
            t_s_samples = int(t_s_value)
        else:
            t_s_samples = int(np.round(t_s_value * sampling_rate))
        
        # Validate t_s
        if t_s_samples < 0 or t_s_samples >= len(signal):
            n_skipped_missing_onset += 1
            continue
        
        # Process each coda method
        for method in coda_methods:
            # Auto-detect coda column naming
            coda_col_samples = f't_coda_{method}_samples'
            coda_col_seconds = f't_coda_{method}_seconds'
            coda_col_legacy = f't_coda_{method}'
            
            if coda_col_samples in df_onsets.columns:
                coda_col = coda_col_samples
                coda_unit = 'samples'
            elif coda_col_seconds in df_onsets.columns:
                coda_col = coda_col_seconds
                coda_unit = 'seconds'
            elif coda_col_legacy in df_onsets.columns:
                coda_col = coda_col_legacy
                coda_unit = 'seconds'
            else:
                # Method not available, skip
                continue
            
            t_coda_value = row[coda_col]
            if pd.isna(t_coda_value):
                continue
            
            # Convert t_coda to samples
            if coda_unit == 'samples':
                t_coda_samples = int(t_coda_value)
            else:
                t_coda_samples = int(np.round(t_coda_value * sampling_rate))
            
            # Validate t_coda
            if t_coda_samples <= t_s_samples or t_coda_samples >= len(signal):
                continue
            
            # Extract S-wave window
            s_wave_signal = signal[t_s_samples:t_coda_samples]
            
            if len(s_wave_signal) == 0:
                continue
            
            try:
                if method == 'arias':
                    # Use Arias D99.5 method (energy-based)
                    # Need to find window_start used for onset
                    # Default: 5s before P or origin
                    if 't_p_detected_samples' in df_onsets.columns:
                        t_p_samples = int(row['t_p_detected_samples'])
                        window_start = max(0, t_p_samples - int(5.0 * sampling_rate))
                    else:
                        window_start = 0
                    
                    t_coda_end_samples = compute_coda_end_arias(
                        signal=signal,
                        t_s_samples=t_s_samples,
                        t_coda_samples=t_coda_samples,
                        threshold_end=threshold_end_arias,
                        sampling_rate=sampling_rate,
                        window_start_samples=window_start
                    )
                else:
                    # Use envelope decay method (for rautian, envelope, median)
                    t_coda_end_samples = compute_coda_end(
                        signal=signal,
                        t_s_samples=t_s_samples,
                        t_coda_samples=t_coda_samples,
                        s_window_signal=s_wave_signal,
                        threshold_factor=threshold_factor,
                        stability_duration=stability_duration,
                        sampling_rate=sampling_rate,
                        smoothing_window=smoothing_window
                    )
                
                t_coda_end_seconds = t_coda_end_samples / sampling_rate
                
                # Store dual representation
                df_onsets.loc[idx, f't_coda_end_{method}_samples'] = t_coda_end_samples
                df_onsets.loc[idx, f't_coda_end_{method}_seconds'] = t_coda_end_seconds
                
                n_computed[method] += 1
                
            except Exception as e:
                print(f"\nWarning: Failed to compute coda end for {station}-{component}-{method}: {e}")
                n_skipped_error += 1
                continue
    
    print()  # New line after progress dots
    
    # Add legacy aliases (point to seconds)
    for method in coda_methods:
        df_onsets[f't_coda_end_{method}'] = df_onsets[f't_coda_end_{method}_seconds']
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total components processed: {len(df_onsets)}")
    print(f"\nSuccessfully computed:")
    for method in coda_methods:
        print(f"  {method:12s}: {n_computed[method]:3d} ({100*n_computed[method]/len(df_onsets):.1f}%)")
    
    print(f"\nSkipped:")
    print(f"No signal data:    {n_skipped_no_signal:3d}")
    print(f"Missing onsets:    {n_skipped_missing_onset:3d}")
    print(f"Errors:            {n_skipped_error:3d}")
    
    
    # Statistics on coda durations
    print(f"\nCoda duration statistics:")
    print(f"  {'Method':<10}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  {'Range':<22}  {'At signal end':>13}")
    print(f"  {'-'*75}")

    for method in coda_methods:
        col_coda = f't_coda_{method}_seconds'
        col_end = f't_coda_end_{method}_seconds'

        if col_coda not in df_onsets.columns or col_end not in df_onsets.columns:
            continue

        valid_mask = df_onsets[col_coda].notna() & df_onsets[col_end].notna()

        if valid_mask.sum() == 0:
            print(f"  {method:<10}  {'no valid data'}")
            continue

        coda_durations = (
            df_onsets.loc[valid_mask, col_end] -
            df_onsets.loc[valid_mask, col_coda]
        )

        n_full_coda = sum(
            1 for idx in df_onsets[valid_mask].index
            if (df_onsets.loc[idx, 'STATION_CODE'] in signals_dict and
                df_onsets.loc[idx, 'COMPONENT'] in signals_dict[df_onsets.loc[idx, 'STATION_CODE']] and
                abs(len(signals_dict[df_onsets.loc[idx, 'STATION_CODE']][df_onsets.loc[idx, 'COMPONENT']]) / sampling_rate
                    - df_onsets.loc[idx, col_end]) < 0.1)
        )

        range_str = f"[{coda_durations.min():.2f}, {coda_durations.max():.2f}]s"
        print(f"  {method:<10}  {coda_durations.mean():>6.2f}s  {coda_durations.median():>6.2f}s  "
            f"{coda_durations.std():>6.2f}s  {range_str:<22}  {n_full_coda:>5} at signal end")

    return df_onsets