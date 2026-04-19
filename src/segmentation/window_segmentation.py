"""
Signal windowing functions for seismic phase segmentation.

This module provides functions to segment seismic signals into temporal windows
based on detected phase onsets (P-wave, S-wave, coda).
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional, Tuple


def segment_signal_into_windows(
    signal: np.ndarray,
    time: np.ndarray,
    t_p: float,
    t_s: float,
    t_coda: float,
    pre_p_duration: float = 5.0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Segment a single signal into four temporal windows.
    
    Windows defined:
    - pre_event: [0, t_p)
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
    pre_p_duration : float, optional
        Duration of pre-event window before P onset (default: 5.0s)
        
    Returns
    -------
    windows : dict
        Dictionary with keys: 'pre_event', 'p_wave', 's_wave', 'coda'
        Each contains: {'signal': array, 'time': array, 't_start': float, 't_end': float}
        
    Examples
    --------
    >>> signal = signals_dict['BRZ']['HGE']
    >>> time = signals_dict['BRZ']['time']
    >>> windows = segment_signal_into_windows(signal, time, t_p=12.2, t_s=14.5, t_coda=20.7)
    >>> windows['s_wave']['signal'].shape
    (1240,)
    """
    
    # Validate onset times
    if not (t_p < t_s < t_coda):
        raise ValueError(f"Onset times must satisfy t_p < t_s < t_coda, got {t_p}, {t_s}, {t_coda}")
    
    # Define window boundaries
    t_pre_start = max(0, t_p - pre_p_duration)
    
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
    pre_p_duration: float = 5.0
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
    pre_p_duration : float, optional
        Duration of pre-event window (default: 5.0s)
        
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
    >>> windowed = segment_all_signals(signals_dict, df_onsets, coda_method='arias')
    >>> windowed['BRZ']['HGE']['s_wave']['signal'].shape
    (1240,)
    """
    
    # Map coda method to column name
    coda_col = f't_coda_{coda_method}'
    
    if coda_col not in df_onsets.columns:
        raise ValueError(f"Column '{coda_col}' not found. Available: {list(df_onsets.columns)}")
    
    windowed_signals = {}
    
    for station in signals_dict.keys():
        windowed_signals[station] = {}
        
        # Get components (exclude 'time')
        components = [k for k in signals_dict[station].keys() if k != 'time']
        time_array = signals_dict[station]['time']
        
        for component in components:
            # Find corresponding row in DataFrame
            mask = (df_onsets['STATION_CODE'] == station) & (df_onsets['COMPONENT'] == component)
            
            if not mask.any():
                print(f"Warning: No onset data for {station}-{component}, skipping")
                continue
            
            row = df_onsets[mask].iloc[0]
            
            # Extract onset times
            t_p = row['t_p_detected']
            t_s = row['t_s_detected']
            t_coda = row[coda_col]
            
            # Check for missing values
            if pd.isna(t_p) or pd.isna(t_s) or pd.isna(t_coda):
                print(f"Warning: Missing onset times for {station}-{component}, skipping")
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
                
            except ValueError as e:
                print(f"Error segmenting {station}-{component}: {e}")
                continue
    
    # Summary
    n_stations = len(windowed_signals)
    n_total = sum(len(windowed_signals[s]) for s in windowed_signals)
    
    print(f"\nSegmented {n_total} signals from {n_stations} stations")
    print(f"Coda method: {coda_method}")
    print(f"Pre-P duration: {pre_p_duration}s")
    
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
        n_samples, mean, std, max, min
        
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