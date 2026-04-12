"""
Crustal velocity estimation for seismic event segmentation.

This module provides functions to estimate P-wave and S-wave velocities
at station locations using the CRUST1.0 global crustal model.

Level 1: Atomic functions (single values)
Level 2: DataFrame applicators (vectorized operations)

References
----------
Laske, G., Masters, G., Ma, Z., & Pasyanos, M. (2013).
Update on CRUST1.0 - A 1-degree global model of Earth's crust.
Geophysical Research Abstracts, 15, EGU2013-2658.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add crust1 directory to path
CRUST1_DIR = Path(__file__).parent.parent / 'data' / 'crust1'
if str(CRUST1_DIR) not in sys.path:
    sys.path.insert(0, str(CRUST1_DIR))

from crust1 import crustModel


# ============================================================================
# LEVEL 1: ATOMIC FUNCTIONS (single station/value)
# ============================================================================

def extract_crustal_velocities(crust_profile):
    """
    Extract average crustal velocities from CRUST1.0 profile.
    
    Averages velocities across upper, middle, and lower crystalline crust.
    
    Parameters
    ----------
    crust_profile : dict
        Output from crustModel.get_point(lat, lon)
        Dictionary with layer names as keys and [vp, vs, rho, thickness, top]
        as values
        
    Returns
    -------
    vp : float
        Average P-wave velocity (km/s)
    vs : float
        Average S-wave velocity (km/s)
        
    Examples
    --------
    >>> from crust1 import crustModel
    >>> model = crustModel()
    >>> profile = model.get_point(44.5127, 6.8533)
    >>> vp, vs = extract_crustal_velocities(profile)
    >>> print(f"v_P = {vp:.2f} km/s, v_S = {vs:.2f} km/s")
    v_P = 6.47 km/s, v_S = 3.73 km/s
    """
    crust_layers = ['upper_crust', 'middle_crust', 'lower_crust']
    
    vp_values = []
    vs_values = []
    
    for layer in crust_layers:
        if layer in crust_profile:
            # crust_profile[layer] = [vp, vs, rho, thickness, top]
            vp_values.append(crust_profile[layer][0])
            vs_values.append(crust_profile[layer][1])
    
    if vp_values and vs_values:
        vp = np.mean(vp_values)
        vs = np.mean(vs_values)
    else:
        # Fallback: standard continental crust
        vp = 6.0
        vs = 3.5
    
    return vp, vs


def calculate_theoretical_arrival(distance, velocity, origin_time=0):
    """
    Calculate theoretical wave arrival time.
    
    Uses simple 1D model: t = t0 + distance / velocity
    
    Parameters
    ----------
    distance : float
        Epicentral distance (km)
    velocity : float
        Wave velocity (km/s)
    origin_time : float, optional
        Event origin time in seconds (default: 0)
        
    Returns
    -------
    arrival_time : float
        Theoretical arrival time (s)
        
    Examples
    --------
    >>> t_p = calculate_theoretical_arrival(distance=50, velocity=6.0)
    >>> print(f"P-wave arrives at {t_p:.2f} s")
    P-wave arrives at 8.33 s
    """
    return origin_time + distance / velocity


# ============================================================================
# LEVEL 2: DATAFRAME APPLICATORS (vectorized)
# ============================================================================

def add_crustal_velocities(df_stations, lat_col='latitude', lon_col='longitude'):
    """
    Add crustal velocity columns to station DataFrame.
    
    Queries CRUST1.0 model for each station and adds:
    - 'vp_crust': P-wave velocity (km/s)
    - 'vs_crust': S-wave velocity (km/s)
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with latitude and longitude columns
    lat_col : str, optional
        Latitude column name (default: 'latitude')
    lon_col : str, optional
        Longitude column name (default: 'longitude')
        
    Returns
    -------
    df_result : pd.DataFrame
        Copy of input DataFrame with added velocity columns
        
    Raises
    ------
    ValueError
        If required columns are missing
        
    Examples
    --------
    >>> df_meta = pd.read_csv('station_metadata.csv')
    >>> df_meta = add_crustal_velocities(df_meta)
    >>> print(df_meta[['station', 'vp_crust', 'vs_crust']])
    """
    # Validate input
    if lat_col not in df_stations.columns:
        raise ValueError(f"Column '{lat_col}' not found in DataFrame")
    if lon_col not in df_stations.columns:
        raise ValueError(f"Column '{lon_col}' not found in DataFrame")
    
    # Initialize CRUST1.0 model
    print("Loading CRUST1.0 model...")
    model = crustModel()
    
    print(f"Querying crustal velocities for {len(df_stations)} stations...")
    
    vp_list = []
    vs_list = []
    
    # Query each station using get_point()
    for idx, row in df_stations.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Get crustal profile directly from CRUST1.0
        profile = model.get_point(lat, lon)
        
        # Extract average crustal velocities
        vp, vs = extract_crustal_velocities(profile)
        
        vp_list.append(vp)
        vs_list.append(vs)
    
    # Add columns to copy of DataFrame
    df_result = df_stations.copy()
    df_result['vp_crust'] = vp_list
    df_result['vs_crust'] = vs_list
    
    # Print summary statistics
    print(f"Added vp_crust and vs_crust columns")
    print(f"v_P: min={min(vp_list):.2f}, max={max(vp_list):.2f}, "
          f"median={np.median(vp_list):.2f} km/s")
    print(f"v_S: min={min(vs_list):.2f}, max={max(vs_list):.2f}, "
          f"median={np.median(vs_list):.2f} km/s")
    
    return df_result


def add_theoretical_arrivals(df_stations, origin_time=0, 
                            distance_col='distance_km'):
    """
    Add theoretical P and S arrival time columns.
    
    Calculates arrival times using: t = t0 + distance / velocity
    
    Adds columns:
    - 't_p_theo': Theoretical P-wave arrival time (s)
    - 't_s_theo': Theoretical S-wave arrival time (s)
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with 'distance_km', 'vp_crust', 'vs_crust' columns
    origin_time : float, optional
        Event origin time in seconds (default: 0)
    distance_col : str, optional
        Distance column name (default: 'distance_km')
        
    Returns
    -------
    df_result : pd.DataFrame
        Copy with added theoretical arrival time columns
        
    Raises
    ------
    ValueError
        If required columns are missing
        
    Examples
    --------
    >>> df_meta = add_crustal_velocities(df_meta)
    >>> df_meta = add_theoretical_arrivals(df_meta, origin_time=0)
    >>> print(df_meta[['station', 't_p_theo', 't_s_theo']])
    """
    # Validate required columns
    required_cols = [distance_col, 'vp_crust', 'vs_crust']
    missing_cols = [col for col in required_cols if col not in df_stations.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Run add_crustal_velocities() first."
        )
    
    # Vectorized calculation (fast!)
    df_result = df_stations.copy()
    df_result['t_p_theo'] = origin_time + df_result[distance_col] / df_result['vp_crust']
    df_result['t_s_theo'] = origin_time + df_result[distance_col] / df_result['vs_crust']
    
    print(f"Added theoretical arrival times")
    print(f"t_P: {df_result['t_p_theo'].min():.2f} - {df_result['t_p_theo'].max():.2f} s")
    print(f"t_S: {df_result['t_s_theo'].min():.2f} - {df_result['t_s_theo'].max():.2f} s")
    
    return df_result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_regional_average_velocity(df_stations, method='median'):
    """
    Calculate regional average crustal velocities.
    
    Useful for Step 1 of segmentation pipeline when you want a single
    representative value for the region rather than station-specific velocities.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        DataFrame with 'vp_crust' and 'vs_crust' columns
    method : str, optional
        Averaging method: 'median' (default) or 'mean'
        
    Returns
    -------
    vp_avg : float
        Regional average P-wave velocity (km/s)
    vs_avg : float
        Regional average S-wave velocity (km/s)
        
    Examples
    --------
    >>> df_meta = add_crustal_velocities(df_meta)
    >>> vp_avg, vs_avg = get_regional_average_velocity(df_meta)
    >>> print(f"Use v_P = {vp_avg:.1f} km/s, v_S = {vs_avg:.1f} km/s")
    Use v_P = 6.5 km/s, v_S = 3.7 km/s
    """
    required_cols = ['vp_crust', 'vs_crust']
    missing_cols = [col for col in required_cols if col not in df_stations.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Run add_crustal_velocities() first."
        )
    
    if method == 'median':
        vp_avg = df_stations['vp_crust'].median()
        vs_avg = df_stations['vs_crust'].median()
    elif method == 'mean':
        vp_avg = df_stations['vp_crust'].mean()
        vs_avg = df_stations['vs_crust'].mean()
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'median' or 'mean'.")
    
    print(f"\nRegional average crustal velocities ({method}):")
    print(f"  v_P = {vp_avg:.2f} km/s")
    print(f"  v_S = {vs_avg:.2f} km/s")
    print(f"\nRounded (for thesis Step 1):")
    print(f"  v_P = {round(vp_avg, 1):.1f} km/s")
    print(f"  v_S = {round(vs_avg, 1):.1f} km/s")
    
    return vp_avg, vs_avg

