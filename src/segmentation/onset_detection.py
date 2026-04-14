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

def detect_onsets_ar_windowed(signals_dict, df_meta_stations,
                              p_window_before=15, p_window_after=15,
                              s_window_before=20, s_window_after=20):
    """
    Detect P and S onsets using AR-AIC with theoretical search windows.
    
    Applies AR-AIC method to signal subsets extracted around theoretical
    arrival times. This approach improves detection accuracy by constraining
    the search to physically plausible time windows.
    
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
        Detection results with columns:
        - STATION_CODE
        - t_p_detected, t_s_detected (onset times in file coordinates)
        - t_p_theo, t_s_theo (theoretical times)
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
    >>> df_results = detect_onsets_ar_windowed(signals_dict, df_meta_stations)
    >>> print(f"P success: {df_results['p_detection_success'].sum()}")
    >>> print(f"S success: {df_results['s_detection_success'].sum()}")
    """
    results = []
    
    print(f"Running AR-AIC onset detection with theoretical windows...")
    print(f"  P window: [{-p_window_before}, +{p_window_after}]s around t_p_theo")
    print(f"  S window: [{-s_window_before}, +{s_window_after}]s around t_s_theo")
    print("\nProcessing: ", end="", flush=True)
    
    for station, data in signals_dict.items():
        print(".", end="", flush=True)
        
        # Get metadata for this station
        station_meta = df_meta_stations[df_meta_stations['STATION_CODE'] == station]
        
        if len(station_meta) == 0:
            results.append({
                'STATION_CODE': station,
                't_p_detected': np.nan,
                't_s_detected': np.nan,
                't_p_theo': np.nan,
                't_s_theo': np.nan,
                'p_residual': np.nan,
                's_residual': np.nan,
                'p_detection_success': False,
                's_detection_success': False,
                'detection_method': 'ar_windowed',
                'p_window_used': '',
                's_window_used': '',
                'error_message': 'Station not found in metadata',
                'components_used': ''
            })
            continue
        
        station_meta = station_meta.iloc[0]
        
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
            results.append({
                'STATION_CODE': station,
                't_p_detected': np.nan,
                't_s_detected': np.nan,
                't_p_theo': station_meta['t_p_theo'],
                't_s_theo': station_meta['t_s_theo'],
                'p_residual': np.nan,
                's_residual': np.nan,
                'p_detection_success': False,
                's_detection_success': False,
                'detection_method': 'ar_windowed',
                'p_window_used': '',
                's_window_used': '',
                'error_message': f'Incomplete components: Z={comp_z}, N={comp_n}, E={comp_e}',
                'components_used': ','.join([c for c in [comp_z, comp_n, comp_e] if c])
            })
            continue
        
        # Get full signals
        signal_z_full = data[comp_z]
        signal_n_full = data[comp_n]
        signal_e_full = data[comp_e]
        
        # Get filter parameters
        f1 = station_meta['LOW_CUT_FREQUENCY_HZ']
        f2 = station_meta['HIGH_CUT_FREQUENCY_HZ']
        
        # Get theoretical times
        t_p_theo = station_meta['t_p_theo']
        t_s_theo = station_meta['t_s_theo']
        
        # Define search windows (use provided windows or calculate from theo times)
        if 'p_window_start' in station_meta.index:
            p_win_start = station_meta['p_window_start']
            p_win_end = station_meta['p_window_end']
        else:
            p_win_start = max(0, t_p_theo - p_window_before)
            p_win_end = t_p_theo + p_window_after
        
        if 's_window_start' in station_meta.index:
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
            t_p_detected = p_win_start + p_pick_relative
            p_success = True
            
        except Exception as e:
            error_msg += f"P detection failed: {type(e).__name__}: {str(e)}; "
        
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
            # We only care about the S pick here
            _, s_pick_relative = ar_pick(
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
            t_s_detected = s_win_start + s_pick_relative
            s_success = True
            
        except Exception as e:
            error_msg += f"S detection failed: {type(e).__name__}: {str(e)}; "
        
        # Calculate residuals
        p_residual = t_p_detected - t_p_theo if p_success else np.nan
        s_residual = t_s_detected - t_s_theo if s_success else np.nan
        
        # Store results
        results.append({
            'STATION_CODE': station,
            't_p_detected': t_p_detected,
            't_s_detected': t_s_detected,
            't_p_theo': t_p_theo,
            't_s_theo': t_s_theo,
            'p_residual': p_residual,
            's_residual': s_residual,
            'p_detection_success': p_success,
            's_detection_success': s_success,
            'detection_method': 'ar_windowed',
            'p_window_used': f'[{p_win_start:.1f}, {p_win_end:.1f}]',
            's_window_used': f'[{s_win_start:.1f}, {s_win_end:.1f}]',
            'error_message': error_msg.strip(),
            'components_used': f'{comp_z},{comp_n},{comp_e}'
        })
    
    print(" Done!")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Print summary
    n_total = len(df_results)
    n_p_success = df_results['p_detection_success'].sum()
    n_s_success = df_results['s_detection_success'].sum()
    n_both_success = (df_results['p_detection_success'] & df_results['s_detection_success']).sum()
    
    print(f"\nDetection summary:")
    print(f"  Total stations: {n_total}")
    print(f"  P successful: {n_p_success} ({100*n_p_success/n_total:.1f}%)")
    print(f"  S successful: {n_s_success} ({100*n_s_success/n_total:.1f}%)")
    print(f"  Both P+S successful: {n_both_success} ({100*n_both_success/n_total:.1f}%)")
    
    if n_p_success > 0:
        df_p_success = df_results[df_results['p_detection_success']]
        print(f"\nP-wave residuals:")
        print(f"  Mean: {df_p_success['p_residual'].mean():+.2f} s")
        print(f"  Std:  {df_p_success['p_residual'].std():.2f} s")
        print(f"  Range: [{df_p_success['p_residual'].min():+.2f}, {df_p_success['p_residual'].max():+.2f}] s")
    
    if n_s_success > 0:
        df_s_success = df_results[df_results['s_detection_success']]
        print(f"\nS-wave residuals:")
        print(f"  Mean: {df_s_success['s_residual'].mean():+.2f} s")
        print(f"  Std:  {df_s_success['s_residual'].std():.2f} s")
        print(f"  Range: [{df_s_success['s_residual'].min():+.2f}, {df_s_success['s_residual'].max():+.2f}] s")
    
    # Print failures
    n_failed = n_total - n_both_success
    if n_failed > 0:
        print(f"\nStations with failures ({n_failed}):")
        df_failed = df_results[~(df_results['p_detection_success'] & df_results['s_detection_success'])]
        for _, row in df_failed.iterrows():
            status = []
            if not row['p_detection_success']:
                status.append('P failed')
            if not row['s_detection_success']:
                status.append('S failed')
            print(f"  {row['STATION_CODE']:6s}: {', '.join(status)} - {row['error_message']}")
    
    return df_results