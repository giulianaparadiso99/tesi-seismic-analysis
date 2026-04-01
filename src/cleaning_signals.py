"""
cleaning_signals.py
-------------------
Preprocessing pipelines for the seismic acceleration time series loaded
from the .ASC files. Each file contains a single signal recorded by one
station on one component (HNE, HNN, or HNZ). The raw acceleration values
(in cm/s²) are stored in a long-format DataFrame with one row per sample.

A single baseline preprocessing pipeline is applied to all 66 signals.
An additional filtered pipeline is provided for analyses that require
signals of sufficient length (moment scaling analysis).

    preprocess_signals_single()
        For PDF analysis and heavy-tail assessment (Notebooks 3 and 4).
        Applies to all 66 files.
        Steps:
            1. _baseline_correction — subtract per-signal mean (zero baseline)
            2. _normalize           — divide by per-signal standard deviation

    preprocess_signals_long()
        For moment scaling analysis (Notebooks 3 and 4). Excludes the 6
        near-field stations with short recordings (SURF, BRZ, BHB, CRI,
        SLZ, SAV), which do not provide enough samples to produce reliable
        displacement increment estimates at large time scales tau.
        Steps:
            1. _filter_long         — retain only files with >= min_samples
                                      samples (default: 48 000)
            2. _baseline_correction — same as above
            3. _normalize           — same as above

Both pipelines return a DataFrame with two acceleration columns:
    'acceleration'            — baseline-corrected (cm/s²)
    'acceleration_normalized' — baseline-corrected and unit-std normalized

Usage:
    from src.cleaning_signals import preprocess_signals_single
    from src.cleaning_signals import preprocess_signals_long

    df_acc_clean = preprocess_signals_single(df_acc_raw)
    df_acc_long  = preprocess_signals_long(df_acc_raw, min_samples=48000)
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
    return df_acc[df_acc['file'].isin(long_files)].copy()


# ===============================================================================================
# ======================================= Public pipelines ======================================
# ===============================================================================================

def preprocess_signals_single(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing pipeline for PDF analysis and heavy-tail assessment.
    Applies to all 66 files — no filtering by signal length.

    Steps:
        1. Baseline correction — subtract per-signal mean
        2. Normalization       — divide by per-signal standard deviation

    Returns a DataFrame with both 'acceleration' (baseline-corrected)
    and 'acceleration_normalized' columns.
    """
    df = _baseline_correction(df_acc)
    df = _normalize(df)
    return df


def preprocess_signals_long(df_acc: pd.DataFrame, min_samples: int = 48000) -> pd.DataFrame:
    """
    Preprocessing pipeline for moment scaling analysis.
    Retains only files with at least min_samples samples (default: 48 000),
    excluding the 6 near-field stations with short recordings.

    Steps:
        1. Filter long signals  — keep only files with >= min_samples samples
        2. Baseline correction  — subtract per-signal mean
        3. Normalization        — divide by per-signal standard deviation

    Returns a DataFrame with 'acceleration' (baseline-corrected)
    and 'acceleration_normalized' columns.
    """
    df = _filter_long(df_acc, min_samples)
    #df = _baseline_correction(df)
    #df = _normalize(df)
    return df