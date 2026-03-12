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