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