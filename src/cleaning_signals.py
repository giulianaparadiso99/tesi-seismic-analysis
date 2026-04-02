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
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
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
    
    print("="*60)
    print(f"Output: {len(df)} samples from {df['file'].nunique()} files")
    print(f"Columns: {list(df.columns)}")
    print("="*60 + "\n")
    
    return df