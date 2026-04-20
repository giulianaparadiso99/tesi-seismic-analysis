"""
Signal windowing functions for seismic phase segmentation.

This module provides functions to segment seismic signals into temporal windows
based on detected phase onsets (P-wave, S-wave, coda).
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple


"""
Signal windowing functions for seismic phase segmentation.

This module provides functions to segment seismic signals into temporal windows
based on detected phase onsets (P-wave, S-wave, coda).
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple, Union


def segment_signal_into_windows(
    signal: np.ndarray,
    time: np.ndarray,
    t_p: float,
    t_s: float,
    t_coda: float,
    pre_p_duration: Union[float, Literal['full']] = 5.0
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
    time : np.ndarray
        Time array corresponding to signal
    t_p : float
        P-wave onset time (seconds)
    t_s : float
        S-wave onset time (seconds)
    t_coda : float
        Coda onset time (seconds)
    pre_p_duration : float or 'full', optional
        Duration of pre-event window before P onset.
        - If float (e.g., 5.0): fixed duration window [t_p - duration, t_p)
        - If 'full': use all available signal from start [time[0], t_p)
        Default: 5.0s (standard for consistency across stations)
        
    Returns
    -------
    windows : dict
        Dictionary with keys: 'pre_event', 'p_wave', 's_wave', 'coda'
        Each contains: {'signal': array, 'time': array, 't_start': float, 
                        't_end': float, 'duration': float}
        
    Examples
    --------
    >>> # Fixed 5s pre-event window (standard)
    >>> windows = segment_signal_into_windows(signal, time, t_p=12.2, t_s=14.5, 
    ...                                       t_coda=20.7, pre_p_duration=5.0)
    >>> windows['pre_event']['duration']
    5.0
    
    >>> # Full pre-event window from recording start
    >>> windows = segment_signal_into_windows(signal, time, t_p=12.2, t_s=14.5,
    ...                                       t_coda=20.7, pre_p_duration='full')
    >>> windows['pre_event']['duration']
    12.2  # Uses all available time before P
    
    Notes
    -----
    Using 'full' pre-event window:
    - Pros: Maximum noise characterization samples
    - Cons: Variable duration across stations, possible contamination,
            non-stationary noise, computational overhead
    
    Recommendation: Use fixed duration (5.0s) for consistency and 
    comparability with literature (NGA-West2, ESM, RESORCE standards).
    """
    
    # Validate onset times
    if not (t_p < t_s < t_coda):
        raise ValueError(
            f"Onset times must satisfy t_p < t_s < t_coda, "
            f"got t_p={t_p:.2f}, t_s={t_s:.2f}, t_coda={t_coda:.2f}"
        )
    
    # Define pre-event window start
    if pre_p_duration == 'full':
        # Use all available signal from start
        t_pre_start = time[0]
    elif isinstance(pre_p_duration, (int, float)):
        # Fixed duration window
        if pre_p_duration <= 0:
            raise ValueError(f"pre_p_duration must be positive, got {pre_p_duration}")
        t_pre_start = max(time[0], t_p - pre_p_duration)
    else:
        raise TypeError(
            f"pre_p_duration must be float or 'full', got {type(pre_p_duration)}"
        )
    
    windows = {}
    
    # Pre-event window
    mask_pre = (time >= t_pre_start) & (time < t_p)
    windows['pre_event'] = {
        'signal': signal[mask_pre],
        'time': time[mask_pre],
        't_start': t_pre_start,
        't_end': t_p,
        'duration': t_p - t_pre_start
    }
    
    # P-wave window
    mask_p = (time >= t_p) & (time < t_s)
    windows['p_wave'] = {
        'signal': signal[mask_p],
        'time': time[mask_p],
        't_start': t_p,
        't_end': t_s,
        'duration': t_s - t_p
    }
    
    # S-wave window
    mask_s = (time >= t_s) & (time < t_coda)
    windows['s_wave'] = {
        'signal': signal[mask_s],
        'time': time[mask_s],
        't_start': t_s,
        't_end': t_coda,
        'duration': t_coda - t_s
    }
    
    # Coda window
    mask_coda = time >= t_coda
    windows['coda'] = {
        'signal': signal[mask_coda],
        'time': time[mask_coda],
        't_start': t_coda,
        't_end': time[-1],
        'duration': time[-1] - t_coda
    }
    
    return windows


def segment_all_signals(
    signals_dict: Dict[str, Dict[str, np.ndarray]],
    df_onsets: pd.DataFrame,
    coda_method: Literal['rautian', 'arias', 'envelope'] = 'rautian',
    pre_p_duration: Union[float, Literal['full']] = 5.0
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Segment all signals in dictionary into temporal windows.
    
    Parameters
    ----------
    signals_dict : dict
        Nested dict from convert_signals_to_dict():
        {station: {component: array, 'time': array}}
    df_onsets : pd.DataFrame
        DataFrame with columns: STATION_CODE, COMPONENT, t_p_detected, 
        t_s_detected, t_coda_rautian, t_coda_arias, t_coda_envelope
    coda_method : {'rautian', 'arias', 'envelope'}, optional
        Which coda onset method to use (default: 'rautian')
    pre_p_duration : float or 'full', optional
        Pre-event window duration strategy:
        - float (e.g., 5.0): Fixed duration before P (default, recommended)
        - 'full': Use all available signal from recording start
        
    Returns
    -------
    windowed_signals : dict
        Nested structure:
        {
            station: {
                component: {
                    'pre_event': {'signal': array, 'time': array, ...},
                    'p_wave': {...},
                    's_wave': {...},
                    'coda': {...}
                }
            }
        }
        
    Examples
    --------
    >>> # Standard fixed pre-event window (recommended)
    >>> windowed = segment_all_signals(signals_dict, df_onsets, 
    ...                                coda_method='rautian',
    ...                                pre_p_duration=5.0)
    
    >>> # Full pre-event window (for comparison/experimentation)
    >>> windowed_full = segment_all_signals(signals_dict, df_onsets,
    ...                                     coda_method='rautian', 
    ...                                     pre_p_duration='full')
    
    >>> # Compare pre-event durations
    >>> dur_fixed = windowed['BRZ']['HGE']['pre_event']['duration']
    >>> dur_full = windowed_full['BRZ']['HGE']['pre_event']['duration']
    >>> print(f"Fixed: {dur_fixed:.1f}s, Full: {dur_full:.1f}s")
    
    Notes
    -----
    Pre-event window strategy choice:
    
    Fixed duration (recommended):
    - Ensures consistency across stations
    - Stationary noise characterization
    - Memory efficient
    - Comparable with literature (NGA-West2, ESM, RESORCE)
    - Avoids contamination from other events
    
    Full window (experimental):
    - Maximum noise samples for robust statistics
    - Variable duration may complicate comparisons
    - May include non-stationary early transients
    - Higher memory usage for long recordings
    """
    
    # Map coda method to column name
    coda_col = f't_coda_{coda_method}'
    
    if coda_col not in df_onsets.columns:
        raise ValueError(
            f"Column '{coda_col}' not found in df_onsets. "
            f"Available columns: {list(df_onsets.columns)}"
        )
    
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
            t_p = row['t_p_detected']
            t_s = row['t_s_detected']
            t_coda = row[coda_col]
            
            # Check for missing values
            if pd.isna(t_p) or pd.isna(t_s) or pd.isna(t_coda):
                n_skipped_missing += 1
                continue
            
            # Segment signal
            signal = signals_dict[station][component]
            
            try:
                windows = segment_signal_into_windows(
                    signal=signal,
                    time=time_array,
                    t_p=t_p,
                    t_s=t_s,
                    t_coda=t_coda,
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
        pre_durations = []
        for station in windowed_signals:
            for component in windowed_signals[station]:
                dur = windowed_signals[station][component]['pre_event']['duration']
                pre_durations.append(dur)
        
        pre_durations = np.array(pre_durations)
        
        print(f"\n{'='*70}")
        print("SIGNAL SEGMENTATION SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully segmented: {n_total} signals from {n_stations} stations")
        print(f"Skipped (no onset data): {n_skipped_no_data}")
        print(f"Skipped (missing values): {n_skipped_missing}")
        print(f"Skipped (errors): {n_skipped_error}")
        print(f"\nCoda detection method: {coda_method}")
        print(f"Pre-event strategy: {pre_p_duration}")
        
        if pre_p_duration == 'full':
            print(f"\nPre-event window durations (variable):")
            print(f"  Min:    {np.min(pre_durations):.2f}s")
            print(f"  Max:    {np.max(pre_durations):.2f}s")
            print(f"  Mean:   {np.mean(pre_durations):.2f}s")
            print(f"  Median: {np.median(pre_durations):.2f}s")
            print(f"  Std:    {np.std(pre_durations):.2f}s")
        else:
            print(f"\nPre-event window durations:")
            print(f"  Target: {pre_p_duration:.2f}s")
            print(f"  Actual mean: {np.mean(pre_durations):.2f}s")
            print(f"  Actual range: [{np.min(pre_durations):.2f}, {np.max(pre_durations):.2f}]s")
            if np.min(pre_durations) < pre_p_duration:
                n_short = np.sum(pre_durations < pre_p_duration)
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