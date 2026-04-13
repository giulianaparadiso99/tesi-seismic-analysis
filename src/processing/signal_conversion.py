"""
Signal conversion utilities for event segmentation pipeline.

Functions to convert long-format signal DataFrame to nested dictionary
structure optimized for onset detection and moment scaling analysis.
"""

import numpy as np
import pandas as pd

def add_time_columns(df_signals, df_metadata, 
                     time_col='DATE_TIME_FIRST_SAMPLE',
                     sampling_interval_col='SAMPLING_INTERVAL_S'):
    """
    Add relative and absolute time columns to signals DataFrame.
    
    Enriches signal data with temporal information by:
    1. Computing relative time from sample index (t=0 at first sample)
    2. Computing absolute time using file start timestamp from metadata
    
    Parameters
    ----------
    df_signals : pd.DataFrame
        Preprocessed signals with columns ['file', 'sample', 'acceleration']
    df_metadata : pd.DataFrame
        Station metadata with time information per file
    time_col : str, optional
        Column name for first sample timestamp (default: 'DATE_TIME_FIRST_SAMPLE')
    sampling_interval_col : str, optional
        Column name for sampling interval (default: 'SAMPLING_INTERVAL_S')
    
    Returns
    -------
    pd.DataFrame
        Signals with added columns:
        - 'time': Relative time from file start (seconds), t=0 at first sample
        - 'time_absolute': Absolute UTC datetime of each sample
    
    Examples
    --------
    >>> df_signals = preprocess_signals(df_raw, baseline_correction=True)
    >>> df_signals = add_time_columns(df_signals, df_metadata)
    >>> # Now df_signals has 'time' and 'time_absolute' columns
    
    Notes
    -----
    Relative time is used for onset detection and moment scaling analysis.
    Absolute time is used for physical validation of detected onsets.
    """
    df = df_signals.copy()
    
    # Get sampling interval (assumed constant across all files)
    sampling_interval = df_metadata[sampling_interval_col].iloc[0]
    
    # Calculate relative time: time = sample * sampling_interval
    df['time'] = df['sample'] * sampling_interval
    
    print(f"Added relative time column (t=0 at first sample)")
    print(f"Sampling interval: {sampling_interval} s ({1/sampling_interval:.1f} Hz)")
    print(f"Time range: {df['time'].min():.3f} - {df['time'].max():.3f} s")
    
    # Merge with metadata to get DATE_TIME_FIRST_SAMPLE per file
    file_times = df_metadata[['file', time_col]].drop_duplicates('file')
    df = df.merge(file_times, on='file', how='left')
    
    # Calculate absolute time
    df['time_absolute'] = (
        pd.to_datetime(df[time_col]) + 
        pd.to_timedelta(df['time'], unit='s')
    )
    
    # Drop temporary merge column
    df = df.drop(columns=[time_col])
    
    print(f"Added absolute time column")
    print(f"Time range: {df['time_absolute'].min()} to {df['time_absolute'].max()}")
    
    # Verify consistency
    expected_duration = df_metadata['DURATION_S'].iloc[0]
    actual_duration = df.groupby('file')['time'].max().mean()
    duration_diff = abs(expected_duration - actual_duration)
    
    if duration_diff < 0.01:
        print(f"Duration check: ({actual_duration:.2f} s matches metadata)")
    else:
        print(f"Duration check: Expected {expected_duration:.2f} s, got {actual_duration:.2f} s")
    
    return df

def get_station_from_filename(filename):
    """
    Extract station code from file name.
    
    Parameters
    ----------
    filename : str
        File name (e.g., 'IT.SURF..HNE')
        
    Returns
    -------
    station : str
        Station code (e.g., 'SURF')
        
    Examples
    --------
    >>> get_station_from_filename('IT.SURF..HNE')
    'SURF'
    >>> get_station_from_filename('FR.OGAG.00.HNZ')
    'OGAG'
    """
    parts = filename.split('.')
    return parts[1] if len(parts) > 1 else filename

def get_component_from_filename(filename):
    """
    Extract component code from file name.
    
    Parameters
    ----------
    filename : str
        File name (e.g., 'IT.SURF..HNE')
        
    Returns
    -------
    component : str
        Component code (e.g., 'HNE', 'HNN', 'HNZ')
        
    Examples
    --------
    >>> get_component_from_filename('IT.SURF..HNE')
    'HNE'
    >>> get_component_from_filename('FR.OGAG.00.HNZ')
    'HNZ'
    """
    parts = filename.split('.')
    return parts[3] if len(parts) > 3 else filename

def convert_signals_to_dict(df_signals):
    """
    Convert long-format signals DataFrame to nested dictionary.
    
    Transforms DataFrame with one row per sample into nested dict
    structure: {station: {component: array, 'time': array}}
    
    Parameters
    ----------
    df_signals : pd.DataFrame
        Long format with columns: file, sample, time, acceleration
        
    Returns
    -------
    signals_dict : dict
        Nested dictionary structure:
        {
            'SURF': {
                'HNE': np.array([...]),  # acceleration
                'HNN': np.array([...]),
                'HNZ': np.array([...]),
                'time': np.array([...])
            },
            'BRZ': {...},
            ...
        }
        
    Examples
    --------
    >>> df_signals = pd.read_parquet('acc_preprocessed_scaling.parquet')
    >>> signals_dict = convert_signals_to_dict(df_signals)
    >>> print(signals_dict.keys())
    dict_keys(['SURF', 'BRZ', 'OGAG', ...])
    >>> print(signals_dict['SURF'].keys())
    dict_keys(['HNE', 'HNN', 'HNZ', 'time'])
    >>> print(signals_dict['SURF']['HNE'].shape)
    (48000,)
    """
    signals_dict = {}
    
    file_list = df_signals['file'].unique()
    print(f"Converting {len(file_list)} files to nested dictionary...")
    
    for file_name in file_list:
        station = get_station_from_filename(file_name)
        component = get_component_from_filename(file_name)
        
        # Get signal for this file, sorted by sample
        df_file = df_signals[df_signals['file'] == file_name].sort_values('sample')
        
        # Initialize station dict if needed
        if station not in signals_dict:
            signals_dict[station] = {}
        
        # Store acceleration
        signals_dict[station][component] = df_file['acceleration'].values
        
        # Store time array (same for all components, store only once)
        if 'time' not in signals_dict[station]:
            signals_dict[station]['time'] = df_file['time'].values
    
    # Summary
    n_stations = len(signals_dict)
    components_per_station = {
        station: len([k for k in signals_dict[station].keys() if k != 'time'])
        for station in signals_dict
    }
    
    print(f"Converted {len(file_list)} files")
    print(f"Stations: {n_stations}")
    print(f"Components per station: {set(components_per_station.values())}")
    
    # Check for incomplete stations (missing components)
    incomplete = [s for s, n in components_per_station.items() if n < 3]
    if incomplete:
        print(f"  Warning: {len(incomplete)} stations with <3 components: {incomplete}")
    
    return signals_dict

def get_signal_for_station(df_signals, station_code, component='HNE'):
    """
    Extract signal arrays for specific station and component.
    
    Alternative to convert_signals_to_dict when you need only one signal.
    
    Parameters
    ----------
    df_signals : pd.DataFrame
        Long format signals DataFrame
    station_code : str
        Station code (e.g., 'SURF')
    component : str, optional
        Component code (default: 'HNE')
        
    Returns
    -------
    time : np.ndarray
        Time array (s)
    acceleration : np.ndarray
        Acceleration array (cm/s²)
        
    Raises
    ------
    ValueError
        If no signal found for specified station/component
        
    Examples
    --------
    >>> time, acc = get_signal_for_station(df_signals, 'SURF', 'HNE')
    >>> print(time.shape, acc.shape)
    (48000,) (48000,)
    """
    # Filter by station and component
    mask = df_signals['file'].str.contains(f'{station_code}.*{component}')
    df_station = df_signals[mask].sort_values('sample')
    
    if len(df_station) == 0:
        raise ValueError(
            f"No signal found for station '{station_code}', "
            f"component '{component}'"
        )
    
    return df_station['time'].values, df_station['acceleration'].values


def validate_signals_dict(signals_dict, expected_components=['HNE', 'HNN', 'HNZ']):
    """
    Validate signals dictionary structure.
    
    Checks for:
    - All expected components present
    - Same length for all components and time
    - No NaN or Inf values
    
    Parameters
    ----------
    signals_dict : dict
        Nested signals dictionary
    expected_components : list, optional
        Expected component codes
        
    Returns
    -------
    report : dict
        Validation report with issues found
        
    Examples
    --------
    >>> report = validate_signals_dict(signals_dict)
    >>> if report['valid']:
    ...     print("All signals valid!")
    """
    issues = []
    
    for station, data in signals_dict.items():
        # Check components present
        missing = [c for c in expected_components if c not in data]
        if missing:
            issues.append(f"{station}: missing components {missing}")
        
        # Check time array
        if 'time' not in data:
            issues.append(f"{station}: missing time array")
            continue
        
        time_len = len(data['time'])
        
        # Check component lengths
        for component in expected_components:
            if component in data:
                acc = data[component]
                
                # Length check
                if len(acc) != time_len:
                    issues.append(
                        f"{station}.{component}: length mismatch "
                        f"(time={time_len}, acc={len(acc)})"
                    )
                
                # NaN check
                if np.isnan(acc).any():
                    n_nan = np.isnan(acc).sum()
                    issues.append(f"{station}.{component}: {n_nan} NaN values")
                
                # Inf check
                if np.isinf(acc).any():
                    n_inf = np.isinf(acc).sum()
                    issues.append(f"{station}.{component}: {n_inf} Inf values")
    
    report = {
        'valid': len(issues) == 0,
        'n_stations': len(signals_dict),
        'issues': issues
    }
    
    if report['valid']:
        print(f"✓ All {report['n_stations']} stations validated")
    else:
        print(f"✗ Validation failed: {len(issues)} issues found")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
    
    return report
