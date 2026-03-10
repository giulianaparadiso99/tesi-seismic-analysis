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
        'INSTRUMENTAL_FREQUENCY_HZ',
        'INSTRUMENTAL_DAMPING',
        'MORPHOLOGIC_CLASSIFICATION',
        'MAGNITUDE_L',
        'MAGNITUDE_L_REFERENCE',
        'FULL_SCALE_G',
        'N_BIT_DIGITAL_CONVERTER',
        'VS30_M/S',
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

    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE_YYYYMMDD'], format='%Y%m%d', errors='coerce')
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
    return df