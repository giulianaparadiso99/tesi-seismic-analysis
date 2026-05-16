"""
Utility functions for PhaseNet phase picking.

Structured as:
- Low-level: single-station processing
- Mid-level: validation, coordinate conversion
- High-level: batch processing (called from notebook)
"""
import numpy as np
import pandas as pd
import logging
from scipy.signal import resample
from obspy import Stream, Trace
from typing import Tuple, Optional, Dict, List

def get_station_from_filename(filename):
    """Extract station code from file name."""
    parts = filename.split('.')
    return parts[1] if len(parts) > 1 else filename

def get_component_from_filename(filename):
    """Extract component code from file name."""
    parts = filename.split('.')
    return parts[3] if len(parts) > 3 else filename

def create_obspy_stream_from_dataframe(df_station: pd.DataFrame,
                                       station_code: str,
                                       sampling_rate: float,
                                       signal_column: str = 'acceleration') -> Tuple[Optional[Stream], Optional[Tuple[str, str, str]]]:
    """
    Create ObsPy Stream directly from station DataFrame.
    
    Parameters
    ----------
    df_station : pd.DataFrame
        DataFrame with columns ['file', 'sample', 'acceleration']
        Contains 3 files (components) for one station
    station_code : str
        Station code
    sampling_rate : float
        Sampling rate in Hz
    signal_column : str
        Column name for the signal data (default: 'acceleration')
        
    Returns
    -------
    stream : obspy.Stream or None
        Stream with 3 traces (Z, N, E) or None if incomplete
    component_names : tuple or None
        (Z_name, N_name, E_name) or None if incomplete
    """
    
    # Get available components
    files = df_station['file'].unique()
    components = {get_component_from_filename(f): f for f in files}
    available = set(components.keys())
    
    # Determine vertical component
    if 'HNZ' in available:
        Z_name = 'HNZ'
    elif 'HGZ' in available:
        Z_name = 'HGZ'
    else:
        return None, None
    
    # Determine horizontal components
    if 'HNN' in available and 'HNE' in available:
        N_name, E_name = 'HNN', 'HNE'
    elif 'HGN' in available and 'HGE' in available:
        N_name, E_name = 'HGN', 'HGE'
    elif 'HN1' in available and 'HN2' in available:
        N_name, E_name = 'HN1', 'HN2'
    else:
        return None, None
    
    # Create Stream
    stream = Stream()
    
    for comp_name in [Z_name, N_name, E_name]:
        file_name = components[comp_name]
        df_comp = df_station[df_station['file'] == file_name].sort_values('sample')
        
        trace = Trace(data=df_comp[signal_column].values)
        trace.stats.sampling_rate = sampling_rate
        trace.stats.station = station_code
        trace.stats.channel = comp_name
        
        stream.append(trace)
    
    return stream, (Z_name, N_name, E_name)

def process_single_station_phasenet(
    df_station: pd.DataFrame,
    station_code: str,
    model,
    signal_column: str,
    sampling_rate_original: float,
    sampling_rate_target: float,
    min_p_probability: float = 0.1,
    min_s_probability: float = 0.1
) -> Optional[Dict]:
    """
    Apply PhaseNet to a single station.
    
    Parameters
    ----------
    df_station : pd.DataFrame
        DataFrame with signal data for one station
    station_code : str
        Station code
    model : seisbench.models.PhaseNet
        Loaded PhaseNet model
    signal_column : str
        Name of signal column ('acceleration', 'velocity', 'displacement')
    sampling_rate_original : float
        Original sampling rate (Hz)
    sampling_rate_target : float
        Target sampling rate for PhaseNet (Hz)
    min_p_probability : float
        Minimum P-wave probability to consider a valid pick (default: 0.1)
    min_s_probability : float
        Minimum S-wave probability to consider a valid pick (default: 0.1)
        
    Returns
    -------
    result : dict or None
        Dictionary with onset times and probabilities, or None if processing failed
    """

    logger = logging.getLogger(__name__) 

    # Create ObsPy Stream
    stream, comp_names = create_obspy_stream_from_dataframe(
        df_station, station_code, sampling_rate_original, signal_column
    )
    
    if stream is None:
        return None
    
    original_starttime = stream[0].stats.starttime
    original_npts = stream[0].stats.npts
    original_duration = original_npts / sampling_rate_original
    
    # Check minimum duration
    expected_samples_target = 3000
    expected_duration = expected_samples_target / sampling_rate_target
    
    if original_duration < expected_duration:
        logger.warning(f"Skipping {station_code}: signal too short ({original_duration:.1f}s < {expected_duration:.1f}s)")
        return None
    
    # Resample to target rate
    stream_resampled = stream.copy()
    stream_resampled.resample(sampling_rate_target)
    
    # Apply PhaseNet
    annotations = model.annotate(stream_resampled)
    
    # Extract P and S probability traces
    p_traces = annotations.select(channel="PhaseNet_P")
    s_traces = annotations.select(channel="PhaseNet_S")
    
    if len(p_traces) == 0 or len(s_traces) == 0:
        return None
    
    p_trace = p_traces[0]
    s_trace = s_traces[0]
    
    # Get probability arrays
    p_prob = p_trace.data
    s_prob = s_trace.data
    
    # Find onset as maximum probability
    p_idx_resampled = p_prob.argmax()
    s_idx_resampled = s_prob.argmax()

    p_prob_max = p_prob[p_idx_resampled]
    s_prob_max = s_prob[s_idx_resampled]
    
    if p_prob_max < min_p_probability:
        logger.warning(
            f"Skipping {station_code}: P probability too low "
            f"({p_prob_max:.3f} < {min_p_probability})"
        )
        return None
    
    if s_prob_max < min_s_probability:
        logger.warning(
            f"Skipping {station_code}: S probability too low "
            f"({s_prob_max:.3f} < {min_s_probability})"
        )
        return None
    
    
    # Convert to original sampling rate and time coordinates
    result = convert_onset_coordinates(
        p_idx_resampled=p_idx_resampled,
        s_idx_resampled=s_idx_resampled,
        p_prob=p_prob,
        s_prob=s_prob,
        p_trace=p_trace,
        original_starttime=original_starttime,
        sampling_rate_target=sampling_rate_target,
        sampling_rate_original=sampling_rate_original,
        original_npts=original_npts,
        station_code=station_code,
        comp_names=comp_names,
        original_duration=original_duration
    )
    
    return result


def convert_onset_coordinates(
    p_idx_resampled: int,
    s_idx_resampled: int,
    p_prob: np.ndarray,
    s_prob: np.ndarray,
    p_trace,
    original_starttime,
    sampling_rate_target: float,
    sampling_rate_original: float,
    original_npts: int,
    station_code: str,
    comp_names: Tuple[str, str, str],
    original_duration: float
) -> Dict:
    """
    Convert PhaseNet onset indices to original time coordinates.
    
    Works primarily in the sample domain to minimize floating-point 
    rounding errors. Only converts to seconds at the final step.
    
    Parameters
    ----------
    p_idx_resampled : int
        P onset index in resampled (100 Hz) signal
    s_idx_resampled : int
        S onset index in resampled signal
    p_prob : np.ndarray
        P-wave probability array
    s_prob : np.ndarray
        S-wave probability array
    p_trace : obspy.Trace
        PhaseNet P probability trace (for starttime)
    original_starttime : obspy.UTCDateTime
        Start time of original stream
    sampling_rate_target : float
        PhaseNet target rate (Hz)
    sampling_rate_original : float
        Original signal rate (Hz)
    original_npts : int
        Number of samples in original signal
    station_code : str
        Station code
    comp_names : tuple
        Component names (Z, N, E)
    original_duration : float
        Original signal duration (s)
        
    Returns
    -------
    result : dict
        Dictionary with onset times (samples, seconds) and probabilities
        
    Notes
    -----
    Algorithm works in sample domain:
    1. Calculate temporal offset in samples (at target rate)
    2. Adjust onset indices for offset (still at target rate)
    3. Scale indices to original sampling rate
    4. Convert to seconds only at the end
    
    This approach minimizes rounding errors compared to converting
    through time domain multiple times.
    """
    
    # Calculate time offset between PhaseNet output and original stream
    time_offset_seconds = float(p_trace.stats.starttime - original_starttime)
    
    # Convert offset to samples at target rate (100 Hz)
    time_offset_samples = int(round(time_offset_seconds * sampling_rate_target))
    
    # Adjust onset indices for temporal offset (still at target rate)
    p_idx_adjusted = p_idx_resampled - time_offset_samples
    s_idx_adjusted = s_idx_resampled - time_offset_samples
    
    # Scale to original sampling rate (e.g., 100 Hz → 200 Hz)
    scale_factor = sampling_rate_original / sampling_rate_target
    
    p_idx_original = int(round(p_idx_adjusted * scale_factor))
    s_idx_original = int(round(s_idx_adjusted * scale_factor))
    
    # Convert to seconds ONLY at the end (from original sample indices)
    t_p_seconds = p_idx_original / sampling_rate_original
    t_s_seconds = s_idx_original / sampling_rate_original
    
    return {
        'station_code': station_code,
        'components': ', '.join(comp_names),
        't_p_samples': p_idx_original,
        't_s_samples': s_idx_original,
        't_p_seconds': float(t_p_seconds),
        't_s_seconds': float(t_s_seconds),
        'p_probability_max': float(p_prob[p_idx_resampled]),
        's_probability_max': float(s_prob[s_idx_resampled]),
        'time_offset_seconds': float(time_offset_seconds),
        'original_duration_s': float(original_duration),
        'annotation_npts': len(p_prob)
    }


# ============================================================================
# HIGH-LEVEL: Batch processing (called from notebook)
# ============================================================================

def apply_phasenet_to_signals(
    df_signals: pd.DataFrame,
    model,
    signal_column: str,
    sampling_rate_original: float = 200,
    sampling_rate_target: float = 100,
    min_p_probability: float = 0.1,
    min_s_probability: float = 0.1
) -> pd.DataFrame:
    """
    Apply PhaseNet phase picking to all stations in dataset.
    
    Parameters
    ----------
    df_signals : pd.DataFrame
        DataFrame with signal data
        Must have columns: ['file', 'sample', signal_column]
    model : seisbench.models.PhaseNet
        Loaded PhaseNet model
    signal_column : str
        Name of signal column ('acceleration', 'velocity', 'displacement')
    sampling_rate_original : float
        Original sampling rate (Hz), default 200
    sampling_rate_target : float
        Target sampling rate for PhaseNet (Hz), default 100
    min_p_probability : float
        Minimum P-wave probability to consider a valid pick (default: 0.1)
    min_s_probability : float
        Minimum S-wave probability to consider a valid pick (default: 0.1)
        
    Returns
    -------
    df_picks : pd.DataFrame
        DataFrame with columns:
        - station_code
        - components
        - t_p_samples, t_s_samples
        - t_p_seconds, t_s_seconds
        - p_probability_max, s_probability_max
        - time_offset_seconds
        - original_duration_s
        - annotation_npts
    """
    
    results = []
    
    station_codes = df_signals['file'].apply(get_station_from_filename).unique()
    
    for station_code in station_codes:
        mask = df_signals['file'].apply(lambda f: get_station_from_filename(f) == station_code)
        df_station = df_signals[mask].copy()
        
        result = process_single_station_phasenet(
            df_station=df_station,
            station_code=station_code,
            model=model,
            signal_column=signal_column,
            sampling_rate_original=sampling_rate_original,
            sampling_rate_target=sampling_rate_target,
            min_p_probability=min_p_probability,
            min_s_probability=min_s_probability
        )
        
        if result is not None:
            results.append(result)
    
    return pd.DataFrame(results)

def merge_phasenet_picks_with_metadata(
    df_picks: pd.DataFrame,
    df_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge PhaseNet onset picks with station metadata.
    
    Calculates origin_time directly from event and signal timestamps
    in the metadata (same approach as AR-AIC workflow).
    """
    
    # Rinomina colonne
    df_picks_renamed = df_picks.rename(columns={
        'station_code': 'STATION_CODE',
        't_p_samples': 't_p_detected_samples',
        't_s_samples': 't_s_detected_samples',
        't_p_seconds': 't_p_detected_seconds',
        't_s_seconds': 't_s_detected_seconds'
    })
    
    # Merge
    df_merged = df_meta.merge(df_picks_renamed, on='STATION_CODE', how='inner')
    
    # Calculate origin_time from metadata timestamps
    # (same as add_theoretical_arrivals does internally)
    event_datetime = pd.to_datetime(df_merged['EVENT_DATE'])
    first_sample_datetime = pd.to_datetime(df_merged['DATE_TIME_FIRST_SAMPLE'])
    
    df_merged['origin_time_seconds'] = (event_datetime - first_sample_datetime).dt.total_seconds()
    df_merged['origin_time_samples'] = (df_merged['origin_time_seconds'] * 200).astype(int)
    
    return df_merged