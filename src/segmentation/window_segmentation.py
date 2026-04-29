"""
Signal windowing functions for seismic phase segmentation.

This module provides functions to segment seismic signals into temporal windows
based on detected phase onsets (P-wave, S-wave, coda).
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple, Union


"""
Signal windowing functions for seismic phase segmentation.

This module provides functions to segment seismic signals into temporal windows
based on detected phase onsets (P-wave, S-wave, coda).
"""

def segment_signal_into_windows(
    signal: np.ndarray,
    t_p: int,
    t_s: int,
    t_coda: int,
    unit: Literal['samples', 'seconds'] = 'samples',
    sampling_rate: float = 200,
    time: Optional[np.ndarray] = None,
    pre_p_duration: Union[int, float, Literal['full']] = 5.0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Segment a single signal into four temporal windows.
    
    Windows defined:
    - pre_event: [t_start, t_p)  where t_start depends on pre_p_duration
    - p_wave:    [t_p, t_s)
    - s_wave:    [t_s, t_coda)
    - coda:      [t_coda, end]
    
    Parameters
    ----------
    signal : np.ndarray
        Acceleration time series
    t_p, t_s, t_coda : int or float
        Onset times in units specified by 'unit' parameter
        - If unit='samples': sample indices (int)
        - If unit='seconds': times in seconds (float)
    unit : {'samples', 'seconds'}, optional
        Unit of t_p, t_s, t_coda (default: 'samples')
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
        Required if unit='seconds' for conversion to samples
    time : np.ndarray, optional
        Time array in seconds (deprecated, for backward compatibility)
        Only needed if unit='seconds' and you want time arrays in output
    pre_p_duration : int, float, or 'full', optional
        Duration of pre-event window before P onset
        - If unit='samples': int (number of samples)
        - If unit='seconds': float (seconds)
        - If 'full': use all available signal from start
        Default: 5.0 (interpreted based on unit)
        
    Returns
    -------
    windows : dict
        Dictionary with keys: 'pre_event', 'p_wave', 's_wave', 'coda'
        Each contains:
        - 'signal': array
        - 'start_samples', 'end_samples': int (sample indices)
        - 'start_seconds', 'end_seconds': float (times in seconds)
        - 'duration_samples': int
        - 'duration_seconds': float
        - 'time': array (only if time parameter was provided)
        
    Examples
    --------
    >>> # NEW: samples-based (preferred, no rounding)
    >>> windows = segment_signal_into_windows(
    ...     signal, t_p=2440, t_s=2960, t_coda=4140,
    ...     unit='samples', pre_p_duration=1000
    ... )
    >>> windows['p_wave']['duration_samples']
    520
    
    >>> # OLD: seconds-based (backward compatible)
    >>> windows = segment_signal_into_windows(
    ...     signal, t_p=12.2, t_s=14.8, t_coda=20.7,
    ...     unit='seconds', sampling_rate=200, time=time_array,
    ...     pre_p_duration=5.0
    ... )
    >>> windows['p_wave']['duration_seconds']
    2.6
    
    Notes
    -----
    Samples-based slicing (unit='samples'):
    - More efficient: direct array slicing signal[start:end]
    - No rounding errors
    - Recommended for all new code
    
    Seconds-based slicing (unit='seconds'):
    - Uses time array masking (slower)
    - Kept for backward compatibility
    - Consider migrating to samples-based
    """
    
    # Convert inputs to samples representation
    if unit == 'samples':
        t_p_samp = int(t_p)
        t_s_samp = int(t_s)
        t_coda_samp = int(t_coda)
        
        # Convert pre_p_duration
        if pre_p_duration == 'full':
            pre_p_samp = t_p_samp  # All samples from start to P
        elif isinstance(pre_p_duration, (int, float)):
            pre_p_samp = int(pre_p_duration)
        else:
            raise TypeError(f"pre_p_duration must be int/float or 'full', got {type(pre_p_duration)}")
        
        # Compute seconds representation
        t_p_sec = t_p_samp / sampling_rate
        t_s_sec = t_s_samp / sampling_rate
        t_coda_sec = t_coda_samp / sampling_rate
        
    elif unit == 'seconds':
        t_p_sec = float(t_p)
        t_s_sec = float(t_s)
        t_coda_sec = float(t_coda)
        
        # Convert to samples
        t_p_samp = int(np.round(t_p_sec * sampling_rate))
        t_s_samp = int(np.round(t_s_sec * sampling_rate))
        t_coda_samp = int(np.round(t_coda_sec * sampling_rate))
        
        # Convert pre_p_duration
        if pre_p_duration == 'full':
            pre_p_samp = t_p_samp
        elif isinstance(pre_p_duration, (int, float)):
            pre_p_samp = int(np.round(float(pre_p_duration) * sampling_rate))
        else:
            raise TypeError(f"pre_p_duration must be float or 'full', got {type(pre_p_duration)}")
    else:
        raise ValueError(f"unit must be 'samples' or 'seconds', got {unit}")
    
    # Validate onset times (use samples for integer comparison)
    if not (t_p_samp < t_s_samp < t_coda_samp):
        raise ValueError(
            f"Onset times must satisfy t_p < t_s < t_coda, "
            f"got t_p={t_p_samp} ({t_p_sec:.2f}s), "
            f"t_s={t_s_samp} ({t_s_sec:.2f}s), "
            f"t_coda={t_coda_samp} ({t_coda_sec:.2f}s)"
        )
    
    if pre_p_samp <= 0 and pre_p_duration != 'full':
        raise ValueError(f"pre_p_duration must be positive, got {pre_p_samp} samples")
    
    # Calculate pre-event window start
    t_pre_start_samp = max(0, t_p_samp - pre_p_samp)
    t_pre_start_sec = t_pre_start_samp / sampling_rate
    
    # Signal end
    t_end_samp = len(signal)
    t_end_sec = t_end_samp / sampling_rate
    
    windows = {}
    
    # Pre-event window
    windows['pre_event'] = {
        'signal': signal[t_pre_start_samp:t_p_samp],
        'start_samples': t_pre_start_samp,
        'end_samples': t_p_samp,
        'start_seconds': t_pre_start_sec,
        'end_seconds': t_p_sec,
        'duration_samples': t_p_samp - t_pre_start_samp,
        'duration_seconds': t_p_sec - t_pre_start_sec
    }
    
    # P-wave window
    windows['p_wave'] = {
        'signal': signal[t_p_samp:t_s_samp],
        'start_samples': t_p_samp,
        'end_samples': t_s_samp,
        'start_seconds': t_p_sec,
        'end_seconds': t_s_sec,
        'duration_samples': t_s_samp - t_p_samp,
        'duration_seconds': t_s_sec - t_p_sec
    }
    
    # S-wave window
    windows['s_wave'] = {
        'signal': signal[t_s_samp:t_coda_samp],
        'start_samples': t_s_samp,
        'end_samples': t_coda_samp,
        'start_seconds': t_s_sec,
        'end_seconds': t_coda_sec,
        'duration_samples': t_coda_samp - t_s_samp,
        'duration_seconds': t_coda_sec - t_s_sec
    }
    
    # Coda window
    windows['coda'] = {
        'signal': signal[t_coda_samp:],
        'start_samples': t_coda_samp,
        'end_samples': t_end_samp,
        'start_seconds': t_coda_sec,
        'end_seconds': t_end_sec,
        'duration_samples': t_end_samp - t_coda_samp,
        'duration_seconds': t_end_sec - t_coda_sec
    }
    
    # Add time arrays if provided (backward compatibility)
    if time is not None:
        windows['pre_event']['time'] = time[t_pre_start_samp:t_p_samp]
        windows['p_wave']['time'] = time[t_p_samp:t_s_samp]
        windows['s_wave']['time'] = time[t_s_samp:t_coda_samp]
        windows['coda']['time'] = time[t_coda_samp:]
    
    # Legacy keys for backward compatibility
    for window_name in windows:
        windows[window_name]['t_start'] = windows[window_name]['start_seconds']
        windows[window_name]['t_end'] = windows[window_name]['end_seconds']
        windows[window_name]['duration'] = windows[window_name]['duration_seconds']
    
    return windows


def segment_all_signals(
    signals_dict: Dict[str, Dict[str, np.ndarray]],
    df_onsets: pd.DataFrame,
    sampling_rate: float = 200,
    unit: Literal['samples', 'seconds'] = 'samples',
    coda_method: Literal['rautian', 'arias', 'envelope'] = 'rautian',
    pre_p_duration: Union[int, float, Literal['full']] = 5.0
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Segment all signals in dictionary into temporal windows.
    
    Parameters
    ----------
    signals_dict : dict
        Nested dict from convert_signals_to_dict():
        {station: {component: array, 'time': array}}
    df_onsets : pd.DataFrame
        DataFrame with detected onset columns (dual representation):
        - STATION_CODE, COMPONENT
        - t_p_detected_samples, t_s_detected_samples (preferred), OR
        - t_p_detected_seconds, t_s_detected_seconds, OR
        - t_p_detected, t_s_detected (legacy)
        - t_coda_<method>_samples or t_coda_<method>_seconds or t_coda_<method>
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
    unit : {'samples', 'seconds'}, optional
        Preferred unit for onset times (default: 'samples')
    coda_method : {'rautian', 'arias', 'envelope'}, optional
        Which coda onset method to use (default: 'rautian')
    pre_p_duration : int, float, or 'full', optional
        Pre-event window duration:
        - If unit='samples': int (number of samples), default: 5.0 → 1000 samples @ 200Hz
        - If unit='seconds': float (seconds), default: 5.0
        - If 'full': use all available signal from recording start
        
    Returns
    -------
    windowed_signals : dict
        Nested structure with dual representation:
        {
            station: {
                component: {
                    'pre_event': {
                        'signal': array,
                        'start_samples', 'end_samples': int,
                        'start_seconds', 'end_seconds': float,
                        'duration_samples': int,
                        'duration_seconds': float,
                        'time': array (if available),
                        't_start', 't_end', 'duration': float (legacy)
                    },
                    'p_wave': {...},
                    's_wave': {...},
                    'coda': {...}
                }
            }
        }
        
    Examples
    --------
    >>> # NEW: samples-based (preferred)
    >>> windowed = segment_all_signals(
    ...     signals_dict, df_onsets,
    ...     unit='samples', pre_p_duration=1000
    ... )
    
    >>> # OLD: seconds-based (backward compatible)
    >>> windowed = segment_all_signals(
    ...     signals_dict, df_onsets,
    ...     unit='seconds', pre_p_duration=5.0
    ... )
    """
    
    # Determine coda column name (prefer samples, fallback seconds, then legacy)
    coda_col_samples = f't_coda_{coda_method}_samples'
    coda_col_seconds = f't_coda_{coda_method}_seconds'
    coda_col_legacy = f't_coda_{coda_method}'
    
    if coda_col_samples in df_onsets.columns:
        coda_col = coda_col_samples
        coda_unit = 'samples'
    elif coda_col_seconds in df_onsets.columns:
        coda_col = coda_col_seconds
        coda_unit = 'seconds'
    elif coda_col_legacy in df_onsets.columns:
        coda_col = coda_col_legacy
        coda_unit = 'seconds'  # Assume legacy columns are seconds
    else:
        raise ValueError(
            f"No coda column found for method '{coda_method}'. "
            f"Expected one of: {coda_col_samples}, {coda_col_seconds}, {coda_col_legacy}. "
            f"Available columns: {list(df_onsets.columns)}"
        )
    
    # Determine onset columns (prefer samples, fallback seconds, then legacy)
    if 't_p_detected_samples' in df_onsets.columns:
        t_p_col = 't_p_detected_samples'
        t_s_col = 't_s_detected_samples'
        onset_unit = 'samples'
    elif 't_p_detected_seconds' in df_onsets.columns:
        t_p_col = 't_p_detected_seconds'
        t_s_col = 't_s_detected_seconds'
        onset_unit = 'seconds'
    elif 't_p_detected' in df_onsets.columns:
        t_p_col = 't_p_detected'
        t_s_col = 't_s_detected'
        onset_unit = 'seconds'  # Assume legacy
    else:
        raise ValueError(
            "No detected onset columns found. "
            "Expected 't_p_detected_samples'/'t_s_detected_samples' or "
            "'t_p_detected_seconds'/'t_s_detected_seconds' or "
            "'t_p_detected'/'t_s_detected'"
        )
    
    # Check unit consistency
    if onset_unit != coda_unit:
        print(f"Warning: onset columns in {onset_unit}, coda columns in {coda_unit}. "
              f"Will convert internally.")
    
    # Use onset_unit as working unit (since onsets are more fundamental)
    working_unit = onset_unit
    
    # Convert pre_p_duration to working unit if needed
    if working_unit == 'samples' and isinstance(pre_p_duration, float) and pre_p_duration != 'full':
        # User probably gave seconds, convert to samples
        pre_p_duration = int(np.round(pre_p_duration * sampling_rate))
        print(f"Note: Converted pre_p_duration to {pre_p_duration} samples @ {sampling_rate}Hz")
    
    windowed_signals = {}
    n_skipped_no_data = 0
    n_skipped_missing = 0
    n_skipped_error = 0
    
    for station in signals_dict.keys():
        windowed_signals[station] = {}
        
        # Get components (exclude 'time')
        components = [k for k in signals_dict[station].keys() if k != 'time']
        time_array = signals_dict[station]['time']
        
        for component in components:
            # Find corresponding row in DataFrame
            mask = (df_onsets['STATION_CODE'] == station) & \
                   (df_onsets['COMPONENT'] == component)
            
            if not mask.any():
                n_skipped_no_data += 1
                continue
            
            row = df_onsets[mask].iloc[0]
            
            # Extract onset times
            t_p = row[t_p_col]
            t_s = row[t_s_col]
            t_coda = row[coda_col]
            
            # Check for missing values
            if pd.isna(t_p) or pd.isna(t_s) or pd.isna(t_coda):
                n_skipped_missing += 1
                continue
            
            # Convert coda to working unit if needed
            if coda_unit != working_unit:
                if coda_unit == 'seconds' and working_unit == 'samples':
                    t_coda = int(np.round(float(t_coda) * sampling_rate))
                elif coda_unit == 'samples' and working_unit == 'seconds':
                    t_coda = int(t_coda) / sampling_rate
            
            # Segment signal
            signal = signals_dict[station][component]
            
            try:
                windows = segment_signal_into_windows(
                    signal=signal,
                    t_p=t_p,
                    t_s=t_s,
                    t_coda=t_coda,
                    unit=working_unit,
                    sampling_rate=sampling_rate,
                    time=time_array,
                    pre_p_duration=pre_p_duration
                )
                
                windowed_signals[station][component] = windows
                
            except (ValueError, TypeError) as e:
                print(f"Error segmenting {station}-{component}: {e}")
                n_skipped_error += 1
                continue
    
    # Summary statistics
    n_stations = len(windowed_signals)
    n_total = sum(len(windowed_signals[s]) for s in windowed_signals)
    
    # Calculate pre-event duration statistics
    if n_total > 0:
        pre_durations_sec = []
        pre_durations_samp = []
        for station in windowed_signals:
            for component in windowed_signals[station]:
                dur_sec = windowed_signals[station][component]['pre_event']['duration_seconds']
                dur_samp = windowed_signals[station][component]['pre_event']['duration_samples']
                pre_durations_sec.append(dur_sec)
                pre_durations_samp.append(dur_samp)
        
        pre_durations_sec = np.array(pre_durations_sec)
        pre_durations_samp = np.array(pre_durations_samp)
        
        print(f"\n{'='*70}")
        print("SIGNAL SEGMENTATION SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully segmented: {n_total} signals from {n_stations} stations")
        print(f"Skipped (no onset data): {n_skipped_no_data}")
        print(f"Skipped (missing values): {n_skipped_missing}")
        print(f"Skipped (errors): {n_skipped_error}")
        print(f"\nCoda detection method: {coda_method}")
        print(f"Working unit: {working_unit}")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Pre-event strategy: {pre_p_duration}")
        
        if pre_p_duration == 'full':
            print(f"\nPre-event window durations (variable):")
            print(f"  Min:    {np.min(pre_durations_sec):.2f}s ({np.min(pre_durations_samp)} samp)")
            print(f"  Max:    {np.max(pre_durations_sec):.2f}s ({np.max(pre_durations_samp)} samp)")
            print(f"  Mean:   {np.mean(pre_durations_sec):.2f}s ({np.mean(pre_durations_samp):.0f} samp)")
            print(f"  Median: {np.median(pre_durations_sec):.2f}s ({np.median(pre_durations_samp):.0f} samp)")
            print(f"  Std:    {np.std(pre_durations_sec):.2f}s ({np.std(pre_durations_samp):.0f} samp)")
        else:
            if working_unit == 'samples':
                target_str = f"{pre_p_duration} samples ({pre_p_duration/sampling_rate:.2f}s)"
            else:
                target_str = f"{pre_p_duration:.2f}s ({int(pre_p_duration*sampling_rate)} samp)"
            
            print(f"\nPre-event window durations:")
            print(f"  Target: {target_str}")
            print(f"  Actual mean: {np.mean(pre_durations_sec):.2f}s ({np.mean(pre_durations_samp):.0f} samp)")
            print(f"  Actual range: [{np.min(pre_durations_sec):.2f}, {np.max(pre_durations_sec):.2f}]s")
            if np.min(pre_durations_samp) < (pre_p_duration if working_unit == 'samples' else pre_p_duration * sampling_rate):
                n_short = np.sum(pre_durations_samp < (pre_p_duration if working_unit == 'samples' else pre_p_duration * sampling_rate))
                print(f"  Note: {n_short} signals have shorter pre-event (P arrives early)")
        
        print(f"{'='*70}")
    
    return windowed_signals


def get_window_statistics(
    windowed_signals: Dict,
    window_name: Literal['pre_event', 'p_wave', 's_wave', 'coda']
) -> pd.DataFrame:
    """
    Extract statistics for a specific window across all signals.
    
    Parameters
    ----------
    windowed_signals : dict
        Output from segment_all_signals()
    window_name : str
        Which window: 'pre_event', 'p_wave', 's_wave', 'coda'
        
    Returns
    -------
    df_stats : pd.DataFrame
        Statistics with columns: station, component, duration, 
        n_samples, mean, std, max, min, pga
        
    Examples
    --------
    >>> stats = get_window_statistics(windowed, 's_wave')
    >>> stats.describe()
    """
    
    records = []
    
    for station in windowed_signals:
        for component in windowed_signals[station]:
            window = windowed_signals[station][component][window_name]
            
            signal = window['signal']
            
            records.append({
                'station': station,
                'component': component,
                'duration': window['duration'],
                'duration_seconds': window.get('duration_seconds', window['duration']),
                'duration_samples': window.get('duration_samples', None),
                'n_samples': len(signal),
                't_start': window['t_start'],
                't_end': window['t_end'],
                'mean': np.mean(signal),
                'std': np.std(signal),
                'max': np.max(signal),
                'min': np.min(signal),
                'pga': np.max(np.abs(signal))
            })
    
    return pd.DataFrame(records)