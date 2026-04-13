"""
cleaning_signals.py
-------------------
Preprocessing pipelines for the seismic acceleration time series loaded
from the .ASC files. Each file contains a single signal recorded by one
station on one component (HNE, HNN, or HNZ). The raw acceleration values
(in cm/s²) are stored in a long-format DataFrame with one row per sample.
 
Main function:
    preprocess_signals(df_acc, filter_length=False, baseline_correction=True, 
                       normalize=False, min_samples=48000)
    
    Flexible preprocessing with independent control over each step:
        - filter_length: Retain only long signals (for moment scaling)
        - baseline_correction: Subtract per-signal mean (recommended always)
        - normalize: Divide by per-signal std (ONLY for PDF analysis)
 
Usage examples:
    from src.cleaning_signals import preprocess_signals
    
    # For PDF analysis (all 66 files, normalized)
    df_pdf = preprocess_signals(df_acc_raw, 
                                 filter_length=False,
                                 baseline_correction=True,
                                 normalize=True)
    
    # For moment scaling (48 long files, NOT normalized - preserves physical units)
    df_scaling = preprocess_signals(df_acc_raw,
                                     filter_length=True,
                                     baseline_correction=True,
                                     normalize=False,
                                     min_samples=48000)
    
    # Raw data with only baseline correction
    df_baseline = preprocess_signals(df_acc_raw,
                                      filter_length=False,
                                      baseline_correction=True,
                                      normalize=False)
"""

import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


# ===============================================================================================
# ======================================= Private helpers =======================================
# ===============================================================================================

def _baseline_correction(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the mean from each signal to ensure zero baseline.
    Operates per file.
    """
    df = df_acc.copy()
    means = df.groupby('file')['acceleration'].transform('mean')
    df['acceleration'] = df['acceleration'] - means
    
    # Quality check
    max_residual = df.groupby('file')['acceleration'].mean().abs().max()
    print(f"Baseline correction: max residual mean = {max_residual:.2e} cm/s²")
    
    return df


def _normalize(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes each signal by its standard deviation.
    Operates per file.
    """
    df = df_acc.copy()
    stds = df.groupby('file')['acceleration'].transform('std')
    df['acceleration_normalized'] = df['acceleration'] / stds
    
    # Quality check
    mean_std = df.groupby('file')['acceleration_normalized'].std().mean()
    print(f"Normalization: mean std = {mean_std:.10f} (expected: 1.0)")
    
    return df


def _filter_long(df_acc: pd.DataFrame, min_samples: int = 48000) -> pd.DataFrame:
    """
    Retains only files with at least min_samples samples.
    Unlike the old _truncate helper, this function does NOT truncate signals
    to a common length — it simply drops files that are too short.
    This excludes the 6 near-field stations (SURF, BRZ, BHB, CRI, SLZ, SAV)
    whose recordings are shorter due to their proximity to the epicenter.
    """
    signal_lengths = df_acc.groupby('file')['sample'].max() + 1
    long_files = signal_lengths[signal_lengths >= min_samples].index
    df_filtered = df_acc[df_acc['file'].isin(long_files)].copy()
    print(f"Length filtering: retained {len(long_files)}/{len(signal_lengths)} files (>= {min_samples} samples)")
    return df_filtered


# ===============================================================================================
# ======================================= Main pipeline =========================================
# ===============================================================================================
 
def preprocess_signals(df_acc: pd.DataFrame,
                      filter_length: bool = False,
                      baseline_correction: bool = True,
                      normalize: bool = False,
                      min_samples: int = 48000) -> pd.DataFrame:
    """
    Flexible preprocessing pipeline for seismic acceleration signals.
    
    Parameters
    ----------
    df_acc : pd.DataFrame
        Raw acceleration data with columns ['file', 'sample', 'acceleration']
        where 'acceleration' is in cm/s²
    
    filter_length : bool, default=False
        If True, retain only files with >= min_samples samples.
        - True:  For moment scaling analysis (needs long time scales τ)
        - False: For PDF analysis (use all stations)
        Excludes 6 near-field stations: SURF, BRZ, BHB, CRI, SLZ, SAV
    
    baseline_correction : bool, default=True
        If True, subtract per-signal mean to ensure zero baseline.
        RECOMMENDED: Always True, even if already applied in raw data.
        Essential for integration (velocity/displacement computation).
    
    normalize : bool, default=False
        If True, divide per-signal by its standard deviation.
        Creates 'acceleration_normalized' column (adimensional).
        
        **CRITICAL CHOICE:**
        - True:  For PDF analysis, heavy-tail assessment only
        - False: For moment scaling, velocity/displacement, physical units
        
        When False, 'acceleration_normalized' column is NOT created.
    
    min_samples : int, default=48000
        Minimum samples required when filter_length=True.
        Default (48000) excludes 6 near-field stations.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed data with columns:
        - 'file', 'sample': original identifiers
        - 'acceleration': baseline-corrected (if baseline_correction=True),
                         in physical units (cm/s²)
        - 'acceleration_normalized': baseline-corrected and normalized
                                    (only if normalize=True), adimensional
    
    Examples
    --------
    # PDF analysis on all 66 signals with normalization
    >>> df_pdf = preprocess_signals(df_raw, 
    ...                             filter_length=False,
    ...                             baseline_correction=True,
    ...                             normalize=True)
    >>> # Use: df_pdf['acceleration_normalized']
    
    # Moment scaling on 48 long signals WITHOUT normalization
    >>> df_scaling = preprocess_signals(df_raw,
    ...                                 filter_length=True,
    ...                                 baseline_correction=True,
    ...                                 normalize=False,
    ...                                 min_samples=48000)
    >>> # Use: df_scaling['acceleration'] (preserves physical units!)
    
    # Only baseline correction, no other processing
    >>> df_baseline = preprocess_signals(df_raw,
    ...                                  filter_length=False,
    ...                                  baseline_correction=True,
    ...                                  normalize=False)
    """
       
    df = df_acc.copy()
    
    # Step 1: Length filtering (optional)
    if filter_length:
        df = _filter_long(df, min_samples)
    else:
        print(f"Length filtering: DISABLED (using all {df['file'].nunique()} files)")
    
    # Step 2: Baseline correction (optional but recommended)
    if baseline_correction:
        df = _baseline_correction(df)
    else:
        print("Baseline correction: DISABLED")
        print("WARNING: Non-zero baseline will cause drift in velocity/displacement!")
    
    # Step 3: Normalization (optional)
    if normalize:
        df = _normalize(df)
    else:
        print("Normalization: DISABLED (physical units preserved)")
        # Do NOT create the column at all if not normalizing
        # This prevents accidental use of non-existent normalized data
    
    return df

def validate_preprocessing(df: pd.DataFrame,
                          expected_files: int,
                          check_normalized: bool = True,
                          pipeline_name: str = "preprocessing") -> bool:
    """
    Validate preprocessing results with quality checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe to validate
    expected_files : int
        Expected number of files (66 for PDF, 48 for moment scaling)
    check_normalized : bool
        If True, checks 'acceleration_normalized' column exists and std=1
    pipeline_name : str
        Name for logging (e.g., "PDF analysis", "Moment scaling")
    
    Returns
    -------
    bool
        True if all checks pass
    
    Raises
    ------
    AssertionError
        If any check fails
    """
    logger.info(f"Running quality checks — {pipeline_name} pipeline")
    
    # Check 1: Baseline correction
    max_residual = df.groupby('file')['acceleration'].mean().abs().max()
    assert max_residual < 1e-10, f"Baseline not corrected: max residual = {max_residual:.2e}"
    logger.info(f"Baseline corrected: max residual = {max_residual:.2e} cm/s²")
    
    # Check 2: Normalization (if expected)
    if check_normalized:
        assert 'acceleration_normalized' in df.columns, "Missing acceleration_normalized column"
        mean_std = df.groupby('file')['acceleration_normalized'].std().mean()
        assert abs(mean_std - 1.0) < 1e-6, f"Normalization failed: mean std = {mean_std}"
        logger.info(f"Normalized: mean std = {mean_std:.10f}")
    else:
        assert 'acceleration_normalized' not in df.columns, "acceleration_normalized should not exist"
        logger.info("Not normalized (physical units preserved)")
    
    # Check 3: No NaN
    assert df['acceleration'].isna().sum() == 0, "NaN found in acceleration"
    logger.info("No NaN in acceleration")
    
    if check_normalized:
        assert df['acceleration_normalized'].isna().sum() == 0, "NaN found in acceleration_normalized"
        logger.info("No NaN in acceleration_normalized")
    
    # Check 4: No Inf
    assert np.isinf(df['acceleration']).sum() == 0, "Inf found in acceleration"
    logger.info("No Inf in acceleration")
    
    if check_normalized:
        assert np.isinf(df['acceleration_normalized']).sum() == 0, "Inf found in acceleration_normalized"
        logger.info("No Inf in acceleration_normalized")
    
    # Check 5: Files retained
    n_files = df['file'].nunique()
    assert n_files == expected_files, f"Expected {expected_files} files, got {n_files}"
    logger.info(f"All {expected_files} files retained")
    
    logger.info(f"All checks passed. Shape: {df.shape}")
    return True

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




