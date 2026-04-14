"""
cleaning_metadata.py
--------------------
Preprocessing pipeline for the seismic event metadata extracted from
the .ASC file headers. Each file contains 64 header rows in the format
FEATURE: VALUE, covering both event-level fields (constant across all
66 files, e.g. EVENT_DATE, EVENT_LATITUDE_DEGREE) and station-level
fields (varying per file, e.g. STATION_CODE, EPICENTRAL_DISTANCE_KM).

The pipeline is organised as a set of private helper functions, each
responsible for a single preprocessing step, composed by the public
entry point clean_metadata():

    1. _replace_missing  — replace empty strings and 'None' with NaN
    2. _drop_columns     — remove uninformative, constant, or irrelevant columns
    3. _convert_types    — cast numeric columns to float64 and date columns
                           to datetime
    4. _normalize_strings — strip leading/trailing whitespace from string columns
    5. _remove_duplicates — drop duplicate rows

Usage:
    from src.cleaning_metadata import clean_metadata
    df_meta_clean = clean_metadata(df_meta_raw)
"""

import pandas as pd
import numpy as np

# ----------------
# PRIVATE HELPERS
# ----------------

def _replace_missing(df):
    """Replace empty strings and 'None' strings with NaN."""
    return df.map(lambda x: np.nan if x == '' or x == 'None' else x)

def _drop_columns(df):
    """Drop uninformative or empty columns."""
    cols_to_drop = [
        'USER1', 'USER2', 'USER3', 'USER4', 'USER5',
        'HEADER_FORMAT', 'DATABASE_VERSION',
        'ORIGINAL_DATA_MEDIATOR_CITATION',
        'ORIGINAL_DATA_CREATOR_CITATION',
        'ORIGINAL_DATA_CREATOR',
        'DATA_CITATION', 'DATA_LICENSE', 'DATA_CREATOR',
        'EVENT_NAME',
        'INSTRUMENTAL_DAMPING',
        'MORPHOLOGIC_CLASSIFICATION',
        'MAGNITUDE_L',
        'MAGNITUDE_L_REFERENCE',
        'FULL_SCALE_G',
        'N_BIT_DIGITAL_CONVERTER',
        'VS30_M/S', 'DATE_TIME_FIRST_SAMPLE_PRECISION',
        'DATA_TYPE', 'UNITS', 'DATA_TIMESTAMP_YYYYMMDD_HHMMSS',
        'ORIGINAL_DATA_MEDIATOR',
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

def _convert_types(df):
    """Convert columns to appropriate numeric and datetime types."""
    numeric_cols = [
        'EVENT_LATITUDE_DEGREE', 'EVENT_LONGITUDE_DEGREE', 'EVENT_DEPTH_KM',
        'MAGNITUDE_W',
        'STATION_LATITUDE_DEGREE', 'STATION_LONGITUDE_DEGREE', 'STATION_ELEVATION_M',
        'SENSOR_DEPTH_M',
        'EPICENTRAL_DISTANCE_KM', 'EARTHQUAKE_BACKAZIMUTH_DEGREE',
        'SAMPLING_INTERVAL_S', 'NDATA', 'DURATION_S',
        'PGA_CM/S^2', 'TIME_PGA_S',
        'FILTER_ORDER', 'LOW_CUT_FREQUENCY_HZ', 'HIGH_CUT_FREQUENCY_HZ',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['EVENT_DATE'] = pd.to_datetime(
        df['EVENT_DATE_YYYYMMDD'].astype(str) + '_' + df['EVENT_TIME_HHMMSS'].astype(str), 
        format='%Y%m%d_%H%M%S', 
        errors='coerce'
        )
    df['DATE_TIME_FIRST_SAMPLE'] = pd.to_datetime(
        df['DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS'], format='%Y%m%d_%H%M%S.%f', errors='coerce'
    )
    df.drop(columns=['EVENT_DATE_YYYYMMDD', 'EVENT_TIME_HHMMSS',
                     'DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS'], inplace=True)
    return df

def _normalize_strings(df):
    """Strip and uppercase string columns for consistency."""
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df

def _remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()

def _calculate_sampling_rate(df):
    """
    Calculate INSTRUMENTAL_FREQUENCY_HZ from NDATA and DURATION_S.
    
    The sampling rate is computed as: frequency = n_samples / duration
    This field is typically empty in the raw data but can be derived
    from the number of samples and total duration.
    """
    # Check if column exists and is empty/missing
    if 'INSTRUMENTAL_FREQUENCY_HZ' in df.columns:
        # Replace empty strings with NaN (if not already done)
        df['INSTRUMENTAL_FREQUENCY_HZ'] = df['INSTRUMENTAL_FREQUENCY_HZ'].replace('', np.nan)
        df['INSTRUMENTAL_FREQUENCY_HZ'] = pd.to_numeric(df['INSTRUMENTAL_FREQUENCY_HZ'], errors='coerce')
    
    # Calculate from NDATA and DURATION_S where missing
    if 'NDATA' in df.columns and 'DURATION_S' in df.columns:
        mask = df['INSTRUMENTAL_FREQUENCY_HZ'].isna()
        if mask.any() or mask.all():
            df.loc[mask, 'INSTRUMENTAL_FREQUENCY_HZ'] = df.loc[mask, 'NDATA'] / df.loc[mask, 'DURATION_S']
    
    return df

# ----------------
# PUBLIC PIPELINE
# ----------------

def clean_metadata(df_meta):
    """
    Full preprocessing pipeline for the metadata dataframe.
    Returns a cleaned copy.
    """
    df = df_meta.copy()
    df = _replace_missing(df)
    df = _drop_columns(df)
    df = _convert_types(df)
    df = _normalize_strings(df)
    df = _remove_duplicates(df)
    df = _calculate_sampling_rate(df)
    return df