"""
cleaning_signals.py
-------------------
Preprocessing pipelines for the seismic acceleration time series loaded
from the .ASC files. Each file contains a single signal recorded by one
station on one component (HNE, HNN, or HNZ). The raw acceleration values
(in cm/s²) are stored in a long-format DataFrame with one row per sample.

Two separate pipelines are provided depending on the downstream analysis:

    preprocess_signals_single()
        For single-signal analysis (Notebook 3). Applies to all 66 files.
        Steps:
            1. _baseline_correction — subtract per-signal mean (zero baseline)
            2. _normalize           — divide by per-signal standard deviation

    preprocess_signals_aggregated()
        For aggregated analysis (Notebook 4). Requires equal-length signals.
        Steps:
            1. _truncate            — retain only files with >= min_samples
                                      samples and truncate all to min_samples
                                      (default: 48 000); excludes 6 near-field
                                      stations (SURF, BRZ, BHB, CRI, SLZ, SAV)
            2. _baseline_correction — same as above
            3. _normalize           — same as above

Both pipelines return a DataFrame with two acceleration columns:
    'acceleration'            — baseline-corrected (cm/s²)
    'acceleration_normalized' — baseline-corrected and unit-std normalized

Usage:
    from src.cleaning_signals import preprocess_signals_single
    from src.cleaning_signals import preprocess_signals_aggregated

    df_acc_clean  = preprocess_signals_single(df_acc_raw)
    df_acc_agg    = preprocess_signals_aggregated(df_acc_raw, min_samples=48000)
"""

import pandas as pd


def _baseline_correction(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the mean from each signal to ensure zero baseline.
    Operates per file.
    """
    df = df_acc.copy()
    means = df.groupby('file')['acceleration'].transform('mean')
    df['acceleration'] = df['acceleration'] - means
    return df


def _normalize(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes each signal by its standard deviation.
    Operates per file.
    """
    df = df_acc.copy()
    stds = df.groupby('file')['acceleration'].transform('std')
    df['acceleration_normalized'] = df['acceleration'] / stds
    return df


def preprocess_signals_single(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing pipeline for single signal analysis:
        1. Baseline correction
        2. Normalization by standard deviation

    Returns a dataframe with both 'acceleration' (baseline-corrected)
    and 'acceleration_normalized' columns.
    """
    df = _baseline_correction(df_acc)
    df = _normalize(df)
    return df

def _truncate(df_acc: pd.DataFrame, min_samples: int = 48000) -> pd.DataFrame:
    """
    Keeps only files with at least min_samples samples,
    then truncates all signals to min_samples samples.
    """
    signal_lengths = df_acc.groupby('file')['sample'].max() + 1
    long_files = signal_lengths[signal_lengths >= min_samples].index
    df = df_acc[df_acc['file'].isin(long_files)].copy()
    df = df[df['sample'] < min_samples]
    return df


def preprocess_signals_aggregated(df_acc: pd.DataFrame, min_samples: int = 48000) -> pd.DataFrame:
    """
    Preprocessing pipeline for aggregated signal analysis:
        1. Truncation — keep only files with at least min_samples samples
        2. Baseline correction
        3. Normalization by standard deviation

    Returns a dataframe with 'acceleration' (baseline-corrected)
    and 'acceleration_normalized' columns.
    """
    df = _truncate(df_acc, min_samples)
    df = _baseline_correction(df)
    df = _normalize(df)
    return df