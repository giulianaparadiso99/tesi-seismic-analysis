"""
Preprocessing pipelines for seismic signal time series loaded from the .ASC files.
Each file contains a single signal recorded by one station on one component (HNE, HNN, or HNZ).
The raw signal values are stored in a long-format DataFrame with one row per sample.
 
Main function:
    preprocess_signals(df_signals, signal_column='acceleration', filter_length=False, 
                       baseline_correction=True, normalize=False, min_samples=48000)
    
    Flexible preprocessing with independent control over each step:
        - filter_length: Retain only long signals (for moment scaling)
        - baseline_correction: Subtract per-signal mean (recommended always)
        - normalize: Divide by per-signal std (ONLY for PDF analysis)
 
Usage examples:
    from src.cleaning_signals import preprocess_signals
    
    # For PDF analysis (all files, normalized)
    df_pdf = preprocess_signals(df_signals_raw,
                                 signal_column='acceleration',
                                 filter_length=False,
                                 baseline_correction=True,
                                 normalize=True)
    
    # For moment scaling (long files, NOT normalized - preserves physical units)
    df_scaling = preprocess_signals(df_signals_raw,
                                     signal_column='velocity',
                                     filter_length=True,
                                     baseline_correction=True,
                                     normalize=False,
                                     min_samples=48000)
"""

import pandas as pd
import numpy as np
import logging
from fractions import Fraction
from scipy.signal import resample_poly
from typing import Tuple
logger = logging.getLogger(__name__)


# ===============================================================================================
# ======================================= Private helpers =======================================
# ===============================================================================================

def _baseline_correction(df: pd.DataFrame, signal_column: str = 'acceleration') -> pd.DataFrame:
    """
    Removes the mean from each signal to ensure zero baseline.
    Operates per file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Signal dataframe with columns ['file', 'sample', signal_column]
    signal_column : str
        Name of the signal column to correct
    """
    df = df.copy()
    means = df.groupby('file')[signal_column].transform('mean')
    df[signal_column] = df[signal_column] - means
    
    # Quality check
    max_residual = df.groupby('file')[signal_column].mean().abs().max()
    print(f"Baseline correction: max residual mean = {max_residual:.2e}")
    
    return df


def _normalize(df: pd.DataFrame, signal_column: str = 'acceleration') -> pd.DataFrame:
    """
    Normalizes each signal by its standard deviation.
    Operates per file. Creates a new column '{signal_column}_normalized'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Signal dataframe with columns ['file', 'sample', signal_column]
    signal_column : str
        Name of the signal column to normalize
    """
    df = df.copy()
    stds = df.groupby('file')[signal_column].transform('std')
    normalized_col = f'{signal_column}_normalized'
    df[normalized_col] = df[signal_column] / stds
    
    # Quality check
    mean_std = df.groupby('file')[normalized_col].std().mean()
    print(f"Normalization: mean std = {mean_std:.10f} (expected: 1.0)")
    
    return df


def _filter_long(df: pd.DataFrame, min_samples: int = 48000) -> pd.DataFrame:
    """
    Retains only files with at least min_samples samples.
    """
    signal_lengths = df.groupby('file')['sample'].max() + 1
    long_files = signal_lengths[signal_lengths >= min_samples].index
    df_filtered = df[df['file'].isin(long_files)].copy()
    print(f"Length filtering: retained {len(long_files)}/{len(signal_lengths)} files (>= {min_samples} samples)")
    return df_filtered


# ===============================================================================================
# ======================================= Main pipeline =========================================
# ===============================================================================================
 
def preprocess_signals(df: pd.DataFrame,
                      signal_column: str = 'acceleration',
                      filter_length: bool = False,
                      baseline_correction: bool = True,
                      normalize: bool = False,
                      min_samples: int = 48000) -> pd.DataFrame:
    """
    Flexible preprocessing pipeline for seismic signals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw signal data with columns ['file', 'sample', signal_column]
    
    signal_column : str, default='acceleration'
        Name of the signal column to process (e.g., 'acceleration', 'velocity', 'displacement')
    
    filter_length : bool, default=False
        If True, retain only files with >= min_samples samples.
        - True:  For moment scaling analysis (needs long time scales τ)
        - False: For PDF analysis (use all stations)
    
    baseline_correction : bool, default=True
        If True, subtract per-signal mean to ensure zero baseline.
        RECOMMENDED: Always True, even if already applied in raw data.
    
    normalize : bool, default=False
        If True, divide per-signal by its standard deviation.
        Creates '{signal_column}_normalized' column (adimensional).
        
        **CRITICAL CHOICE:**
        - True:  For PDF analysis, heavy-tail assessment only
        - False: For moment scaling, preserves physical units
        
        When False, normalized column is NOT created.
    
    min_samples : int, default=48000
        Minimum samples required when filter_length=True.
        Default (48000).
    
    Returns
    -------
    pd.DataFrame
        Preprocessed data with columns:
        - 'file', 'sample': original identifiers
        - signal_column: baseline-corrected (if baseline_correction=True), in physical units
        - f'{signal_column}_normalized': baseline-corrected and normalized
                                         (only if normalize=True), adimensional
    
    Examples
    --------
    # PDF analysis on all signals with normalization
    >>> df_pdf = preprocess_signals(df_raw,
    ...                             signal_column='acceleration',
    ...                             filter_length=False,
    ...                             baseline_correction=True,
    ...                             normalize=True)
    >>> # Use: df_pdf['acceleration_normalized']
    
    # Moment scaling on long signals WITHOUT normalization
    >>> df_scaling = preprocess_signals(df_raw,
    ...                                 signal_column='velocity',
    ...                                 filter_length=True,
    ...                                 baseline_correction=True,
    ...                                 normalize=False,
    ...                                 min_samples=48000)
    >>> # Use: df_scaling['velocity'] (preserves physical units!)
    """
       
    df = df.copy()
    
    # Step 1: Length filtering (optional)
    if filter_length:
        df = _filter_long(df, min_samples)
    else:
        print(f"Length filtering: DISABLED (using all {df['file'].nunique()} files)")
    
    # Step 2: Baseline correction (optional but recommended)
    if baseline_correction:
        df = _baseline_correction(df, signal_column)
    else:
        print("Baseline correction: DISABLED")
        print("WARNING: Non-zero baseline will cause drift in velocity/displacement!")
    
    # Step 3: Normalization (optional)
    if normalize:
        df = _normalize(df, signal_column)
    else:
        print("Normalization: DISABLED (physical units preserved)")
    
    return df


def validate_preprocessing(df: pd.DataFrame,
                          signal_column: str = 'acceleration',
                          expected_files: int = 66,
                          check_normalized: bool = True,
                          pipeline_name: str = "preprocessing") -> bool:
    """
    Validate preprocessing results with quality checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe to validate
    signal_column : str
        Name of the signal column that was processed
    expected_files : int
        Expected number of files 
    check_normalized : bool
        If True, checks '{signal_column}_normalized' column exists and std=1
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
    
    normalized_col = f'{signal_column}_normalized'
    
    # Check 1: Baseline correction
    max_residual = df.groupby('file')[signal_column].mean().abs().max()
    assert max_residual < 1e-10, f"Baseline not corrected: max residual = {max_residual:.2e}"
    logger.info(f"Baseline corrected: max residual = {max_residual:.2e}")
    
    # Check 2: Normalization (if expected)
    if check_normalized:
        assert normalized_col in df.columns, f"Missing {normalized_col} column"
        mean_std = df.groupby('file')[normalized_col].std().mean()
        assert abs(mean_std - 1.0) < 1e-6, f"Normalization failed: mean std = {mean_std}"
        logger.info(f"Normalized: mean std = {mean_std:.10f}")
    else:
        assert normalized_col not in df.columns, f"{normalized_col} should not exist"
        logger.info("Not normalized (physical units preserved)")
    
    # Check 3: No NaN
    assert df[signal_column].isna().sum() == 0, f"NaN found in {signal_column}"
    logger.info(f"No NaN in {signal_column}")
    
    if check_normalized:
        assert df[normalized_col].isna().sum() == 0, f"NaN found in {normalized_col}"
        logger.info(f"No NaN in {normalized_col}")
    
    # Check 4: No Inf
    assert np.isinf(df[signal_column]).sum() == 0, f"Inf found in {signal_column}"
    logger.info(f"No Inf in {signal_column}")
    
    if check_normalized:
        assert np.isinf(df[normalized_col]).sum() == 0, f"Inf found in {normalized_col}"
        logger.info(f"No Inf in {normalized_col}")
    
    # Check 5: Files retained
    n_files = df['file'].nunique()
    assert n_files == expected_files, f"Expected {expected_files} files, got {n_files}"
    logger.info(f"All {expected_files} files retained")
    
    logger.info(f"All checks passed. Shape: {df.shape}")
    return True


def resample_signal_to_target_rate(
    signal: np.ndarray,
    source_rate: float,
    target_rate: float
) -> np.ndarray:
    """
    Resample a single signal to a target sampling rate using polyphase filtering.

    Uses scipy.signal.resample_poly with an exact rational up/down factor
    derived from the ratio of target to source rate, avoiding the spectral
    artifacts of FFT-based resampling for simple integer ratios (e.g. 100 Hz
    to 200 Hz upsampling).

    Parameters
    ----------
    signal : np.ndarray
        Input signal at source_rate.
    source_rate : float
        Original sampling rate in Hz.
    target_rate : float
        Desired sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Resampled signal at target_rate.

    Notes
    -----
    Resampling does not recover frequency content above the original
    Nyquist frequency (source_rate / 2): a signal originally sampled at
    100 Hz was anti-alias filtered below 50 Hz at digitization, and
    upsampling to 200 Hz only makes the sample indexing consistent with
    the rest of the pipeline, without adding genuine high-frequency
    information.
    """
    ratio = Fraction(target_rate).limit_denominator() / Fraction(source_rate).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(signal, up, down)


def resample_mismatched_sampling_rate_signals(
    df_signals: pd.DataFrame,
    df_meta: pd.DataFrame,
    target_rate: float,
    signal_column: str,
    get_station_from_filename,
    sampling_interval_col: str = 'SAMPLING_INTERVAL_S',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample all files recorded at a sampling rate other than target_rate,
    leaving all other files unchanged.

    Detects mismatched files directly from df_meta[sampling_interval_col]
    rather than requiring an explicit station list, so any future dataset
    with a different set of non-conforming stations is handled without
    code changes.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Long-format signal data with columns ['file', 'sample', signal_column].
    df_meta : pd.DataFrame
        Metadata with columns ['file', sampling_interval_col, 'NDATA',
        'DURATION_S'] and, if present, 'INSTRUMENTAL_FREQUENCY_HZ'.
    target_rate : float
        Sampling rate in Hz that all output files must share.
    signal_column : str
        Name of the signal value column in df_signals.
    get_station_from_filename : callable
        Function mapping a filename to a station code, reused from the
        existing pipeline to keep file-to-station logic in one place.
    sampling_interval_col : str, optional
        Column in df_meta holding the sampling interval in seconds
        (default: 'SAMPLING_INTERVAL_S').

    Returns
    -------
    df_signals_resampled : pd.DataFrame
        df_signals with mismatched files resampled to target_rate and
        their 'sample' column rebuilt from 0.
    df_meta_updated : pd.DataFrame
        df_meta with sampling_interval_col, 'NDATA', and
        'INSTRUMENTAL_FREQUENCY_HZ' (if present) corrected for the
        resampled files. 'DURATION_S' is left unchanged, since duration
        is preserved by resampling.

    Raises
    ------
    ValueError
        If a mismatched file listed in df_meta has no corresponding rows
        in df_signals.
    """
    target_interval = 1.0 / target_rate
    mismatched_meta = df_meta[
        ~np.isclose(df_meta[sampling_interval_col], target_interval)
    ]

    if mismatched_meta.empty:
        print(f"No files found with sampling rate other than {target_rate:.1f} Hz.")
        return df_signals.copy(), df_meta.copy()

    mismatched_files = mismatched_meta['file'].unique()
    mismatched_stations = sorted(set(get_station_from_filename(f) for f in mismatched_files))
    print(
        f"Resampling {len(mismatched_files)} file(s) from station(s) "
        f"{mismatched_stations} to {target_rate:.1f} Hz."
    )

    df_signals_unchanged = df_signals[~df_signals['file'].isin(mismatched_files)].copy()

    resampled_blocks = []
    for file_name in mismatched_files:
        file_meta_row = mismatched_meta[mismatched_meta['file'] == file_name].iloc[0]
        source_rate = 1.0 / file_meta_row[sampling_interval_col]

        file_signal_rows = df_signals[df_signals['file'] == file_name].sort_values('sample')
        if file_signal_rows.empty:
            raise ValueError(f"No signal rows found in df_signals for file '{file_name}'.")

        original_signal = file_signal_rows[signal_column].to_numpy()
        resampled_signal = resample_signal_to_target_rate(original_signal, source_rate, target_rate)

        resampled_blocks.append(pd.DataFrame({
            'file': file_name,
            'sample': np.arange(len(resampled_signal)),
            signal_column: resampled_signal,
        }))

    df_signals_resampled = pd.concat(
        [df_signals_unchanged] + resampled_blocks, ignore_index=True
    )

    df_meta_updated = df_meta.copy()
    mismatched_mask = df_meta_updated['file'].isin(mismatched_files)
    df_meta_updated.loc[mismatched_mask, sampling_interval_col] = target_interval
    if 'INSTRUMENTAL_FREQUENCY_HZ' in df_meta_updated.columns:
        df_meta_updated.loc[mismatched_mask, 'INSTRUMENTAL_FREQUENCY_HZ'] = target_rate
    if 'NDATA' in df_meta_updated.columns:
        new_ndata = {
            block['file'].iloc[0]: len(block) for block in resampled_blocks
        }
        for file_name, n_samples in new_ndata.items():
            df_meta_updated.loc[df_meta_updated['file'] == file_name, 'NDATA'] = n_samples

    print(f"Resampling complete: {len(mismatched_files)} file(s) updated.")

    return df_signals_resampled, df_meta_updated

def truncate_components_to_common_length(
    df_signals: pd.DataFrame,
    df_meta: pd.DataFrame,
    get_station_from_filename,
    sampling_interval_col: str = 'SAMPLING_INTERVAL_S',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Truncate all components of each station to their shortest common length.

    Raw (unprocessed) ITACA exports may have slightly different sample
    counts across the three components of the same station (unlike the
    curated processed export, where components are aligned). Since
    convert_signals_to_dict() stores a single shared 'time' array per
    station, mismatched component lengths cause a length mismatch between
    'time' and 'signal' for the longer components. Truncating each
    station's components to their common minimum length upstream avoids
    this without modifying convert_signals_to_dict() itself.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Long-format signal data with columns ['file', 'sample', ...].
    df_meta : pd.DataFrame
        Metadata with columns ['file', 'NDATA', 'DURATION_S',
        sampling_interval_col].
    get_station_from_filename : callable
        Function mapping a filename to a station code.
    sampling_interval_col : str, optional
        Column in df_meta holding the sampling interval in seconds,
        used to recompute DURATION_S for truncated files
        (default: 'SAMPLING_INTERVAL_S').

    Returns
    -------
    df_signals_truncated : pd.DataFrame
        df_signals with each station's components truncated to their
        shortest common length. Samples beyond the truncation point are
        dropped; sample indices are otherwise left unchanged.
    df_meta_updated : pd.DataFrame
        df_meta with 'NDATA' and 'DURATION_S' corrected for the
        truncated files. Files not truncated are left unchanged.
    """
    df = df_signals.copy()
    df['_station'] = df['file'].apply(get_station_from_filename)

    file_lengths = df.groupby('file')['sample'].max() + 1
    file_to_station = df.drop_duplicates('file').set_index('file')['_station']

    min_length_per_station = file_lengths.groupby(file_to_station).min()

    truncated_files = []
    for file_name, length in file_lengths.items():
        station = file_to_station[file_name]
        target_length = min_length_per_station[station]
        if length > target_length:
            truncated_files.append((file_name, length, target_length))

    if truncated_files:
        print(f"Truncating {len(truncated_files)} file(s) to their station's common minimum length:")
        for file_name, original_length, target_length in truncated_files:
            print(f"  {file_name}: {original_length} -> {target_length} samples "
                  f"(-{original_length - target_length})")
    else:
        print("No files require truncation: all stations have components of equal length.")

    target_lengths = df['_station'].map(min_length_per_station)
    df_signals_truncated = df[df['sample'] < target_lengths].copy()
    df_signals_truncated = df_signals_truncated.drop(columns=['_station'])

    df_meta_updated = df_meta.copy()
    for file_name, _, target_length in truncated_files:
        row_mask = df_meta_updated['file'] == file_name
        df_meta_updated.loc[row_mask, 'NDATA'] = target_length
        sampling_interval = df_meta_updated.loc[row_mask, sampling_interval_col].iloc[0]
        df_meta_updated.loc[row_mask, 'DURATION_S'] = target_length * sampling_interval

    print(f"Truncation complete: {len(truncated_files)} file(s) updated in metadata.")

    return df_signals_truncated, df_meta_updated