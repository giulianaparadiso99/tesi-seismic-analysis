"""
PhaseNet deep-learning phase picker integration.

This module provides utilities for applying PhaseNet (Zhu & Beroza, 2019)
to seismic signals via the SeisBench framework. It handles data preparation,
model invocation, coordinate conversion, and metadata merging.

Functions are organized by abstraction level:

Low-level (single-station processing):
    get_station_from_filename : Extract station code from filename
    get_component_from_filename : Extract component code from filename
    create_obspy_stream_from_dataframe : Convert DataFrame to ObsPy Stream
    process_single_station_phasenet : Apply PhaseNet to one station

Mid-level (validation and conversion):
    convert_onset_coordinates : Convert PhaseNet output to original time base
    
High-level (batch processing):
    apply_phasenet_to_signals : Process entire dataset
    merge_phasenet_picks_with_metadata : Join picks with station metadata

Technical Details
-----------------
PhaseNet model: Pre-trained INSTANCE model via SeisBench
Target sampling rate: 100 Hz (resampled internally)
Minimum signal duration: 30 seconds
Output format: Dual representation (samples + seconds)

Coordinate Conversion
---------------------
PhaseNet operates on resampled signals (100 Hz) with potential temporal
offset from original recording. The conversion pipeline:
1. Calculate temporal offset between PhaseNet output and original signal
2. Adjust onset indices for offset (at target rate)
3. Scale indices to original sampling rate
4. Convert to seconds only at final step

This sample-domain approach minimizes floating-point rounding errors
compared to multiple time-domain conversions.

Component Naming
----------------
Automatically detects and handles multiple conventions:
- Standard ITACA: HNE, HNN, HNZ
- Alternative: HGE, HGN, HGZ
- Rotated: HN1, HN2, HNZ

References
----------
Zhu, W., & Beroza, G. C. (2019). "PhaseNet: a deep-neural-network-based
    seismic arrival-time picking method." Geophysical Journal International,
    216(1), 261-273.
Woollam, J., et al. (2022). "SeisBench—A Toolbox for Machine Learning in
    Seismology." Seismological Research Letters, 93(3), 1695-1709.

Examples
--------
>>> import seisbench.models as sbm
>>> 
>>> # Load pre-trained model
>>> model = sbm.PhaseNet.from_pretrained("instance")
>>> 
>>> # Process all signals
>>> df_picks = apply_phasenet_to_signals(
...     df_signals, 
...     model, 
...     signal_column='acceleration',
...     sampling_rate_original=200,
...     sampling_rate_target=100
... )
>>> 
>>> # Merge with metadata
>>> df_merged = merge_phasenet_picks_with_metadata(df_picks, df_meta_stations)
>>> print(df_merged[['STATION_CODE', 't_p_detected_seconds', 't_s_detected_seconds']])
"""

import numpy as np
import pandas as pd
import logging
from scipy.signal import resample
from obspy import Stream, Trace
from typing import Tuple, Optional, Dict, List

def get_station_from_filename(filename: str) -> str:
    """Extract station code from file name."""
    parts = filename.split('.')
    return parts[1] if len(parts) > 1 else filename

def get_component_from_filename(filename: str) -> str:
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
    elif 'HLZ' in available:
        Z_name = 'HLZ'
    else:
        return None, None

    # Determine horizontal components
    if 'HNN' in available and 'HNE' in available:
        N_name, E_name = 'HNN', 'HNE'
    elif 'HGN' in available and 'HGE' in available:
        N_name, E_name = 'HGN', 'HGE'
    elif 'HLN' in available and 'HLE' in available:
        N_name, E_name = 'HLN', 'HLE'
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
    overlap_samples = int(28.0 * sampling_rate_target)
    annotations = model.annotate(stream_resampled, overlap=overlap_samples)
    
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
    df_meta: pd.DataFrame,
    sampling_rate: float = 200
) -> pd.DataFrame:
    """
    Merge PhaseNet onset picks with station metadata and calculate origin time.

    Joins PhaseNet picks with station-level metadata and computes the earthquake
    origin time in both sample and second coordinates using event timestamps
    from the metadata (consistent with AR-AIC workflow).

    Parameters
    ----------
    df_picks : pd.DataFrame
        PhaseNet picks from apply_phasenet_to_signals()
        Must have columns:
        - station_code
        - t_p_samples, t_s_samples
        - t_p_seconds, t_s_seconds
        - p_probability_max, s_probability_max
        - components
    df_meta : pd.DataFrame
        Station metadata with columns:
        - STATION_CODE
        - EVENT_DATE (event origin timestamp)
        - DATE_TIME_FIRST_SAMPLE (signal start timestamp)
        - Theoretical arrivals: t_p_theo_*, t_s_theo_*
        - Station coordinates, distances, etc.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with:
        - All metadata columns
        - Renamed PhaseNet picks: t_p_detected_samples, t_p_detected_seconds, etc.
        - Computed origin time: origin_time_samples, origin_time_seconds
        - Assumes 200 Hz sampling rate for origin_time_samples calculation

    Notes
    -----
    Origin time calculation:
        origin_time = EVENT_DATE - DATE_TIME_FIRST_SAMPLE
        
    This matches the approach used internally by add_theoretical_arrivals()
    in the AR-AIC workflow, ensuring consistency between methods.

    Only stations present in both DataFrames (inner join) are included
    in the output.

    Examples
    --------
    >>> df_picks = apply_phasenet_to_signals(df_signals, model, 'acceleration')
    >>> df_merged = merge_phasenet_picks_with_metadata(df_picks, df_meta_stations)
    >>> print(df_merged[['STATION_CODE', 't_p_detected_seconds', 'origin_time_seconds']])
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
    df_merged['origin_time_samples'] = (df_merged['origin_time_seconds'] * sampling_rate).astype(int)
    
    return df_merged

def extract_picks_from_classify_output(
    classify_output,
    original_starttime,
    sampling_rate_original: float,
    station_code: str,
    comp_names: Tuple[str, str, str],
    original_npts: int,
    original_duration: float
) -> Dict:
    """
    Extract P and S onset times from a SeisBench ClassifyOutput object.

    Converts UTCDateTime pick times to sample indices and seconds in the
    original signal coordinate system. If a phase has no picks above
    threshold, the corresponding fields are set to None/NaN.

    Parameters
    ----------
    classify_output : seisbench.util.ClassifyOutput
        Output of model.classify(), must have attribute .picks (PickList)
    original_starttime : obspy.UTCDateTime
        Start time of the original (unresampled) stream
    sampling_rate_original : float
        Sampling rate of the original signal (Hz)
    station_code : str
        Station code
    comp_names : tuple of str
        Component names (Z, N, E)
    original_npts : int
        Number of samples in the original signal
    original_duration : float
        Duration of the original signal in seconds

    Returns
    -------
    dict
        Dictionary with fields:
        - station_code, components
        - t_p_samples, t_s_samples (int or None)
        - t_p_seconds, t_s_seconds (float or NaN)
        - p_probability_max, s_probability_max (float or NaN)
        - original_duration_s

    Notes
    -----
    When multiple picks of the same phase are present, the one with
    the highest peak_value is selected.
    """
    picks = classify_output.picks

    # Separate P and S picks
    p_picks = [p for p in picks if p.phase == 'P']
    s_picks = [p for p in picks if p.phase == 'S']

    def pick_to_sample(pick):
        """Convert a Pick's peak_time to absolute sample index in original signal."""
        offset_seconds = float(pick.peak_time - original_starttime)
        sample_index = int(round(offset_seconds * sampling_rate_original))
        return sample_index

    # Select best pick per phase (highest peak_value)
    t_p_samples = None
    t_p_seconds = np.nan
    p_probability_max = np.nan

    if p_picks:
        best_p = max(p_picks, key=lambda p: p.peak_value)
        t_p_samples = pick_to_sample(best_p)
        t_p_seconds = t_p_samples / sampling_rate_original
        p_probability_max = float(best_p.peak_value)

    t_s_samples = None
    t_s_seconds = np.nan
    s_probability_max = np.nan

    if s_picks:
        best_s = max(s_picks, key=lambda p: p.peak_value)
        t_s_samples = pick_to_sample(best_s)
        t_s_seconds = t_s_samples / sampling_rate_original
        s_probability_max = float(best_s.peak_value)

    return {
        'station_code': station_code,
        'components': ', '.join(comp_names),
        't_p_samples': t_p_samples,
        't_s_samples': t_s_samples,
        't_p_seconds': t_p_seconds,
        't_s_seconds': t_s_seconds,
        'p_probability_max': p_probability_max,
        's_probability_max': s_probability_max,
        'original_duration_s': float(original_duration)
    }


def process_single_station_phasenet_v2(
    df_station: pd.DataFrame,
    station_code: str,
    model,
    signal_column: str,
    sampling_rate_original: float,
    sampling_rate_target: float,
    min_p_probability: float = 0,
    min_s_probability: float = 0
) -> Optional[Dict]:
    """
    Apply PhaseNet to a single station using classify().

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
        Minimum P-wave probability threshold passed to classify() (default: 0.3)
    min_s_probability : float
        Minimum S-wave probability threshold passed to classify() (default: 0.3)

    Returns
    -------
    dict or None
        Dictionary with onset times and probabilities. Fields for missing
        phases are set to None/NaN. Returns None only if stream construction
        fails or signal is too short.

    Notes
    -----
    Thresholds are passed directly to model.classify() via P_threshold and
    S_threshold keyword arguments, which apply picks_from_annotations
    internally. This differs from the previous annotate() approach where
    argmax was used regardless of probability value.
    """
    logger = logging.getLogger(__name__)

    stream, comp_names = create_obspy_stream_from_dataframe(
        df_station, station_code, sampling_rate_original, signal_column
    )

    if stream is None:
        return None

    original_starttime = stream[0].stats.starttime
    original_npts = stream[0].stats.npts
    original_duration = original_npts / sampling_rate_original

    expected_duration = 3000 / sampling_rate_target
    if original_duration < expected_duration:
        logger.warning(
            f"Skipping {station_code}: signal too short "
            f"({original_duration:.1f}s < {expected_duration:.1f}s)"
        )
        return None

    stream_resampled = stream.copy()
    stream_resampled.resample(sampling_rate_target)

    overlap_samples = int(28.0 * sampling_rate_target)

    classify_output = model.classify(
        stream_resampled,
        overlap=overlap_samples,
        P_threshold=min_p_probability,
        S_threshold=min_s_probability
    )

    return extract_picks_from_classify_output(
        classify_output=classify_output,
        original_starttime=original_starttime,
        sampling_rate_original=sampling_rate_original,
        station_code=station_code,
        comp_names=comp_names,
        original_npts=original_npts,
        original_duration=original_duration
    )