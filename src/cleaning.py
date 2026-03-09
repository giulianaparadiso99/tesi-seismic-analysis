import pandas as pd
import numpy as np

def clean_metadata(df_meta):
    """
    Cleans and types the metadata dataframe.
    Returns a cleaned copy.
    """
    df = df_meta.copy()

    # Replace empty strings and 'None' strings with NaN
    df = df.map(lambda x: np.nan if x == '' or x == 'None' else x)

    # Drop uninformative columns
    cols_to_drop = [
        'USER1', 'USER2', 'USER3', 'USER4', 'USER5',
        'HEADER_FORMAT', 'DATABASE_VERSION',
        'ORIGINAL_DATA_MEDIATOR_CITATION',
        'ORIGINAL_DATA_CREATOR_CITATION',
        'ORIGINAL_DATA_CREATOR',
        'DATA_CITATION', 'DATA_LICENSE', 'DATA_CREATOR',
        'EVENT_NAME', 'INSTRUMENTAL_FREQUENCY_HZ',
        'INSTRUMENTAL_DAMPING', 'MORPHOLOGIC_CLASSIFICATION',
        'MAGNITUDE_L', 'MAGNITUDE_L_REFERENCE',
        'FULL_SCALE_G', 'N_BIT_DIGITAL_CONVERTER', 'VS30_M/S',
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Convert numeric columns
    numeric_cols = [
        'EVENT_LATITUDE_DEGREE', 'EVENT_LONGITUDE_DEGREE', 'EVENT_DEPTH_KM',
        'MAGNITUDE_W', 'MAGNITUDE_L',
        'STATION_LATITUDE_DEGREE', 'STATION_LONGITUDE_DEGREE', 'STATION_ELEVATION_M',
        'SENSOR_DEPTH_M', 'VS30_M/S',
        'EPICENTRAL_DISTANCE_KM', 'EARTHQUAKE_BACKAZIMUTH_DEGREE',
        'SAMPLING_INTERVAL_S', 'NDATA', 'DURATION_S',
        'PGA_CM/S^2', 'TIME_PGA_S',
        'FILTER_ORDER', 'LOW_CUT_FREQUENCY_HZ', 'HIGH_CUT_FREQUENCY_HZ',
        'INSTRUMENTAL_FREQUENCY_HZ', 'INSTRUMENTAL_DAMPING', 'FULL_SCALE_G',
        'N_BIT_DIGITAL_CONVERTER',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert date and time columns
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE_YYYYMMDD'], format='%Y%m%d', errors='coerce')
    df['DATE_TIME_FIRST_SAMPLE'] = pd.to_datetime(
        df['DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS'], format='%Y%m%d_%H%M%S.%f', errors='coerce'
    )
    df.drop(columns=['EVENT_DATE_YYYYMMDD', 'EVENT_TIME_HHMMSS',
                     'DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS'], inplace=True)

    return df