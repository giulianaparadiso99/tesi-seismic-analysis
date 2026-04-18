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

def detect_onsets_ar_windowed(signals_dict, df_meta_stations,
                              p_window_before=5, p_window_after=5,
                              s_window_before=7, s_window_after=7):
    """
    Detect P and S onsets using AR-AIC with theoretical search windows.
    
    Applies AR-AIC method to signal subsets extracted around theoretical
    arrival times. Populates df_meta_stations directly with detected onset times.
    
    Parameters
    ----------
    signals_dict : dict
        Nested dictionary from convert_signals_to_dict()
        Structure: {station: {component: array, 'time': array}}
    df_meta_stations : pd.DataFrame
        Station metadata with columns:
        - STATION_CODE
        - t_p_theo, t_s_theo (theoretical arrival times)
        - LOW_CUT_FREQUENCY_HZ, HIGH_CUT_FREQUENCY_HZ
        - p_window_start, p_window_end (from calculate_search_windows)
        - s_window_start, s_window_end
    p_window_before : float, optional
        Seconds before t_p_theo for search window (default: 15s)
    p_window_after : float, optional
        Seconds after t_p_theo for search window (default: 15s)
    s_window_before : float, optional
        Seconds before t_s_theo for search window (default: 20s)
    s_window_after : float, optional
        Seconds after t_s_theo for search window (default: 20s)
    
    Returns
    -------
    pd.DataFrame
        df_meta_stations with added columns:
        - t_p_detected, t_s_detected (onset times in file coordinates)
        - p_residual, s_residual (detected - theoretical)
        - p_detection_success, s_detection_success (bool flags)
        - detection_method ('ar_windowed')
        - p_window_used, s_window_used (actual windows used)
        - error_message (if failed)
        - components_used
    
    Notes
    -----
    Workflow per station:
    1. Extract subset: time in [t_p_theo - p_window_before, t_p_theo + p_window_after]
    2. Apply ar_pick() to subset for P detection
    3. Convert relative time in subset → absolute time in file
    4. Repeat for S-wave with S window
    
    Stations with incomplete components (missing Z, N, or E) are skipped.
    
    Examples
    --------
    >>> df_meta_stations = calculate_search_windows(df_meta_stations)
    >>> df_meta_stations = detect_onsets_ar_windowed(signals_dict, df_meta_stations)
    >>> print(f"P success: {df_meta_stations['p_detection_success'].sum()}")
    >>> print(f"S success: {df_meta_stations['s_detection_success'].sum()}")
    """
    
    # Initialize onset columns in df_meta_stations
    df_meta_stations['t_p_detected'] = np.nan
    df_meta_stations['t_s_detected'] = np.nan
    df_meta_stations['p_residual'] = np.nan
    df_meta_stations['s_residual'] = np.nan
    df_meta_stations['p_detection_success'] = False
    df_meta_stations['s_detection_success'] = False
    df_meta_stations['error_message'] = ''
    df_meta_stations['components_used'] = ''
    
    print(f"Running AR-AIC onset detection with theoretical windows...")
    print(f"  P window: [{-p_window_before}, +{p_window_after}]s around t_p_theo")
    print(f"  S window: [{-s_window_before}, +{s_window_after}]s around t_s_theo")
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
        
        # Get theoretical times
        t_p_theo = station_meta['t_p_theo']
        t_s_theo = station_meta['t_s_theo']
        
        # Define search windows (use provided windows or calculate from theo times)
        if 'p_window_start' in station_meta.index and not pd.isna(station_meta['p_window_start']):
            p_win_start = station_meta['p_window_start']
            p_win_end = station_meta['p_window_end']
        else:
            p_win_start = max(0, t_p_theo - p_window_before)
            p_win_end = t_p_theo + p_window_after
        
        if 's_window_start' in station_meta.index and not pd.isna(station_meta['s_window_start']):
            s_win_start = station_meta['s_window_start']
            s_win_end = station_meta['s_window_end']
        else:
            s_win_start = max(0, t_s_theo - s_window_before)
            s_win_end = t_s_theo + s_window_after
        
        # Initialize detection results
        t_p_detected = np.nan
        t_s_detected = np.nan
        p_success = False
        s_success = False
        error_msg = ''
        
        # ===== P-WAVE DETECTION =====
        try:
            # Extract P window subset
            mask_p = (time >= p_win_start) & (time <= p_win_end)
            
            if mask_p.sum() < 100:  # Need at least 0.5s of data at 200Hz
                raise ValueError(f"P window too short: {mask_p.sum()} samples")
            
            signal_z_p = signal_z_full[mask_p]
            signal_n_p = signal_n_full[mask_p]
            signal_e_p = signal_e_full[mask_p]
            time_p = time[mask_p]
            
            # Apply AR-AIC to P window
            p_pick_relative, s_pick_in_p_window = ar_pick(
                signal_z_p, signal_n_p, signal_e_p,
                samp_rate=200,
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
            
            # Convert to absolute time in file
            if p_pick_relative is not None and not np.isnan(p_pick_relative):
                t_p_detected = time_p[0] + p_pick_relative
                p_success = True
            else:
                error_msg += 'P-pick returned None; '
        
        except Exception as e:
            error_msg += f'P detection failed: {str(e)}; '
        
        # ===== S-WAVE DETECTION =====
        try:
            # Extract S window subset
            mask_s = (time >= s_win_start) & (time <= s_win_end)
            
            if mask_s.sum() < 100:
                raise ValueError(f"S window too short: {mask_s.sum()} samples")
            
            signal_z_s = signal_z_full[mask_s]
            signal_n_s = signal_n_full[mask_s]
            signal_e_s = signal_e_full[mask_s]
            time_s = time[mask_s]
            
            # Apply AR-AIC to S window
            p_pick_in_s_window, s_pick_relative = ar_pick(
                signal_z_s, signal_n_s, signal_e_s,
                samp_rate=200,
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
            
            # Convert to absolute time in file
            if s_pick_relative is not None and not np.isnan(s_pick_relative):
                t_s_detected = time_s[0] + s_pick_relative
                s_success = True
            else:
                error_msg += 'S-pick returned None; '
        
        except Exception as e:
            error_msg += f'S detection failed: {str(e)}; '
        
        # ===== POPULATE DATAFRAME =====
        df_meta_stations.loc[idx, 't_p_detected'] = t_p_detected
        df_meta_stations.loc[idx, 't_s_detected'] = t_s_detected
        df_meta_stations.loc[idx, 'p_detection_success'] = p_success
        df_meta_stations.loc[idx, 's_detection_success'] = s_success
        
        # Calculate residuals if detection succeeded
        if p_success:
            df_meta_stations.loc[idx, 'p_residual'] = t_p_detected - t_p_theo
        
        if s_success:
            df_meta_stations.loc[idx, 's_residual'] = t_s_detected - t_s_theo
        
        # Record errors if any
        if error_msg:
            df_meta_stations.loc[idx, 'error_message'] = error_msg.strip('; ')
    
    print("\n\nDetection complete!")
    
    # Summary statistics
    n_stations = len(df_meta_stations)
    n_p_success = df_meta_stations['p_detection_success'].sum()
    n_s_success = df_meta_stations['s_detection_success'].sum()
    
    print(f"\nResults:")
    print(f"  P-wave: {n_p_success}/{n_stations} successful ({100*n_p_success/n_stations:.1f}%)")
    print(f"  S-wave: {n_s_success}/{n_stations} successful ({100*n_s_success/n_stations:.1f}%)")
    
    if n_p_success > 0:
        p_res_mean = df_meta_stations['p_residual'].mean()
        p_res_std = df_meta_stations['p_residual'].std()
        print(f"  P residuals: {p_res_mean:.2f} ± {p_res_std:.2f} s")
    
    if n_s_success > 0:
        s_res_mean = df_meta_stations['s_residual'].mean()
        s_res_std = df_meta_stations['s_residual'].std()
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

def detect_coda_start(signal, t_s_detected, origin_time=None,
                     sampling_rate=200, method='rautian',
                     threshold_arias=0.75, threshold_envelope=0.3):
    """
    Detect coda onset using multiple methods from seismological literature.
    
    Parameters
    ----------
    signal : np.ndarray
        Seismic signal (single component)
    t_s_detected : float
        Detected S-wave onset time (s)
    origin_time : float, optional
        Earthquake origin time (s). Required for 'rautian' method.
    sampling_rate : int
        Sampling rate in Hz (default: 200)
    method : str
        Detection method:
        - 'rautian': Lapse time = 2 × S-wave travel time (Rautian & Khalturin, 1978)
        - 'arias': Cumulative Arias Intensity threshold (Trifunac & Brady, 1975; Lanzano et al., 2019)
        - 'envelope': Amplitude envelope decay (Boore, 2005)
    threshold_arias : float
        Threshold for 'arias' method (default: 0.75 for D5-75)
        Common values: 0.75 (D5-75), 0.95 (D5-95)
    threshold_envelope : float
        Threshold factor for 'envelope' method (default: 0.3)
        Common range: 0.2-0.3 (20-30% of peak)
    
    Returns
    -------
    dict
        Dictionary with:
        - 't_coda': float, coda start time (s)
        - 'method': str, method used
        - 'params': dict, method-specific parameters
        - 'diagnostic': dict, diagnostic information for validation
    
    Examples
    --------
    >>> # Rautian method
    >>> result = detect_coda_start(signal, t_s=15.2, origin_time=8.2, 
    ...                            method='rautian')
    >>> 
    >>> # Arias D5-75 (ESM standard)
    >>> result = detect_coda_start(signal, t_s=15.2, method='arias',
    ...                            threshold_arias=0.75)
    >>> 
    >>> # Arias D5-95 (alternative)
    >>> result = detect_coda_start(signal, t_s=15.2, method='arias',
    ...                            threshold_arias=0.95)
    >>> 
    >>> # Envelope with 25% threshold
    >>> result = detect_coda_start(signal, t_s=15.2, method='envelope',
    ...                            threshold_envelope=0.25)
    """
    
    signal_duration = len(signal) / sampling_rate
    
    if method == 'rautian':
        if origin_time is None:
            raise ValueError("origin_time required for 'rautian' method")
        
        # Rautian & Khalturin (1978): lapse_time_coda = 2 × lapse_time_S
        s_lapse_time = t_s_detected - origin_time
        coda_lapse_time = 2.0 * s_lapse_time
        t_coda = origin_time + coda_lapse_time
        
        
        diagnostic = {
            's_lapse_time': s_lapse_time,
            'coda_lapse_time': coda_lapse_time,
            's_duration': t_coda - t_s_detected
        }
        params = {'multiplier': 2.0, 'min_s_duration': min_s_duration}
    
    elif method == 'arias':
        """
        Coda onset when specified percentage of Arias Intensity is reached.
        
        Arias Intensity: AI(t) = (π/2g) ∫ a²(τ) dτ
        
        Standard definitions:
        - D5-75: 75% threshold (Lanzano et al., 2019 - ESM flatfile)
        - D5-95: 95% threshold (Trifunac & Brady, 1975)
        """
        idx_s = int(t_s_detected * sampling_rate)
        signal_after_s = signal[idx_s:]
        
        # Arias Intensity: AI(t) = (π/2g) ∫ a²(τ) dτ
        dt = 1.0 / sampling_rate
        g = 9.81  # m/s²
        arias_cumsum = (np.pi / (2 * g)) * np.cumsum(signal_after_s**2) * dt
        
        if arias_cumsum[-1] == 0:
            t_coda = t_s_detected + 10.0  # Fallback
        else:
            arias_norm = arias_cumsum / arias_cumsum[-1]
            threshold = threshold_arias
            idx_coda_rel = np.argmax(arias_norm >= threshold)
            
            if idx_coda_rel == 0:  # Never reached
                t_coda = t_s_detected + 10.0
            else:
                t_coda = t_s_detected + idx_coda_rel / sampling_rate
        
        # Determine reference based on threshold
        if threshold == 0.75:
            reference = 'D5-75 (Lanzano et al., 2019 - ESM flatfile)'
        elif threshold == 0.95:
            reference = 'D5-95 (Trifunac & Brady, 1975)'
        else:
            reference = f'D5-{int(threshold*100)} (custom threshold)'
        
        diagnostic = {
            'threshold': threshold,
            'total_arias_intensity': arias_cumsum[-1] if arias_cumsum[-1] != 0 else 0,
            's_duration': t_coda - t_s_detected,
            'reference': reference
        }
        params = {
            'threshold': threshold,
            'arias_based': True
        }
    
    elif method == 'envelope':
        """
        Envelope decay method.
        
        Following common practice in strong-motion processing (Boore & Bommer, 2005),
        the coda is defined as beginning when the smoothed envelope falls below a
        threshold (typically 20-30%) of its peak value.
        """
        
        idx_s = int(t_s_detected * sampling_rate)
        signal_after_s = signal[idx_s:]
        
        # Calculate envelope using Hilbert transform
        envelope = np.abs(hilbert(signal_after_s))
        
        # Smooth envelope with 1-second moving average (standard practice)
        envelope_smooth = uniform_filter1d(envelope, size=sampling_rate)
        
        # Find peak in first 5 seconds after S-onset
        search_window = min(int(5 * sampling_rate), len(envelope_smooth))
        if search_window == 0:
            t_coda = t_s_detected + 10.0
        else:
            peak_envelope = np.max(envelope_smooth[:search_window])
            
            # Coda starts when envelope drops below threshold
            threshold = threshold_envelope * peak_envelope
            
            idx_coda_rel = np.argmax(envelope_smooth < threshold)
            if idx_coda_rel == 0:  # Never drops
                t_coda = t_s_detected + 10.0
            else:
                t_coda = t_s_detected + idx_coda_rel / sampling_rate
        
        diagnostic = {
            'peak_envelope': peak_envelope if search_window > 0 else 0,
            'threshold_factor': threshold_envelope,
            'threshold_absolute': threshold if search_window > 0 else 0,
            's_duration': t_coda - t_s_detected,
            'reference': 'Boore & Bommer (2005)'
        }
        params = {
            'threshold_factor': threshold_envelope,
            'smoothing_window_s': 1.0
        }
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'rautian', 'arias', 'envelope'"
        )
    
    # Ensure coda is within signal bounds
    t_coda = min(t_coda, signal_duration - 1.0)
    t_coda = max(t_coda, t_s_detected + 1.0)  # At least 1s of S-wave
    
    return {
        't_coda': t_coda,
        'method': method,
        'params': params,
        'diagnostic': diagnostic
    }

def detect_coda_start_all_methods(signal, t_s_detected, origin_time=None,
                                  sampling_rate=200,
                                  threshold_arias=0.75,
                                  threshold_envelope=0.3):
    """
    Apply all coda detection methods and return results for comparison.
    
    Parameters
    ----------
    signal : np.ndarray
        Seismic signal
    t_s_detected : float
        S-wave onset time (s)
    origin_time : float, optional
        Earthquake origin time (s). Required for 'rautian' method.
    sampling_rate : int
        Sampling rate (Hz)
    threshold_arias : float
        Threshold for Arias Intensity method (default: 0.75)
    threshold_envelope : float
        Threshold factor for envelope method (default: 0.3)
    
    Returns
    -------
    dict
        Dictionary with method names as keys, each containing result dict
    
    Examples
    --------
    >>> # Test all methods with default thresholds
    >>> results = detect_coda_start_all_methods(signal, t_s=15.2, origin_time=8.2)
    >>> for method, res in results.items():
    ...     print(f"{method}: t_coda={res['t_coda']:.2f}s, "
    ...           f"S_duration={res['diagnostic']['s_duration']:.2f}s")
    >>> 
    >>> # Test with custom thresholds
    >>> results = detect_coda_start_all_methods(signal, t_s=15.2, origin_time=8.2,
    ...                                         threshold_arias=0.95,
    ...                                         threshold_envelope=0.25)
    """
    import numpy as np
    
    methods = ['rautian', 'arias', 'envelope']
    results = {}
    
    for method in methods:
        try:
            if method == 'rautian' and origin_time is None:
                continue  # Skip if origin_time not available
            
            result = detect_coda_start(
                signal, t_s_detected, origin_time=origin_time,
                sampling_rate=sampling_rate, method=method,
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
    """Populate coda onset columns in df_full."""
    
    methods = ['rautian', 'arias', 'envelope']
    
    print(f"Calculating coda onset for {len(df_full)} components...")
    
    for idx, row in df_full.iterrows():
        station = row['STATION_CODE']
        component = row['COMPONENT']
        
        if station not in signals_dict or component not in signals_dict[station]:
            continue
        
        signal = signals_dict[station][component]
        t_s = row['t_s_detected']
        origin_time = row['origin_time']
        
        try:
            # Chiama detect_coda_start_all_methods (invariata!)
            results = detect_coda_start_all_methods(
                signal, t_s, origin_time,
                sampling_rate=sampling_rate,
                threshold_arias=threshold_arias,
                threshold_envelope=threshold_envelope
            )
            
            # Popola DataFrame
            for method in methods:
                df_full.loc[idx, f't_coda_{method}'] = results[method]['t_coda']
                df_full.loc[idx, f's_duration_{method}'] = results[method]['diagnostic']['s_duration']
        
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
    import numpy as np
    from scipy.stats import pearsonr
    
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