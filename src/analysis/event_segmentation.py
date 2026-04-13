"""
Crustal velocity estimation - adapted for Ubaye 2024 dataset.

Column names from preprocessing:
- STATION_CODE: Station identifier
- STATION_LATITUDE_DEGREE: Station latitude
- STATION_LONGITUDE_DEGREE: Station longitude  
- EPICENTRAL_DISTANCE_KM: Epicentral distance (already computed!)

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
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # Go up to project root
CRUST1_DIR = _project_root / 'data' / 'Crust1.0' / 'model'
 
if not CRUST1_DIR.exists():
    raise FileNotFoundError(
        f"CRUST1.0 directory not found at {CRUST1_DIR}\n"
        f"Please ensure data/Crust1.0/model/ exists with required files:\n"
        f"  - crust1.vp\n"
        f"  - crust1.vs\n"
        f"  - crust1.rho\n"
        f"  - crust1.bnds\n"
        f"  - crust1.py"
    )
 
if str(CRUST1_DIR) not in sys.path:
    sys.path.insert(0, str(CRUST1_DIR))
 
try:
    from crust1 import crustModel
except ImportError as e:
    raise ImportError(
        f"Cannot import crustModel from {CRUST1_DIR}\n"
        f"Make sure crust1.py exists in that directory.\n"
        f"Original error: {e}"
    )


# ============================================================================
# LEVEL 1: ATOMIC FUNCTIONS
# ============================================================================

def extract_crustal_velocities(crust_profile, weighted=True):
    """
    Extract average crustal velocities from CRUST1.0 profile.
    
    Computes thickness-weighted average velocities across upper, middle,
    and lower crystalline crust layers.
    
    Parameters
    ----------
    crust_profile : dict
        Output from crustModel.get_point(lat, lon)
        Dictionary with layer names as keys and [vp, vs, rho, thickness, top]
        as values
    weighted : bool, optional
        If True, use thickness-weighted average (default)
        If False, use simple arithmetic mean
        
    Returns
    -------
    vp : float
        Average P-wave velocity (km/s)
    vs : float
        Average S-wave velocity (km/s)
        
    Notes
    -----
    CRUST1.0 layer format: [vp, vs, rho, thickness, top]
    - vp: P-wave velocity (km/s)
    - vs: S-wave velocity (km/s)
    - rho: density (g/cm³)
    - thickness: layer thickness (km)
    - top: depth to top of layer (km)
    
    Examples
    --------
    >>> from crust1 import crustModel
    >>> model = crustModel()
    >>> profile = model.get_point(44.5127, 6.8533)
    >>> vp, vs = extract_crustal_velocities(profile, weighted=True)
    >>> print(f"v_P = {vp:.2f} km/s, v_S = {vs:.2f} km/s")
    v_P = 6.47 km/s, v_S = 3.73 km/s
    """
    crust_layers = ['upper_crust', 'middle_crust', 'lower_crust']
    
    vp_values = []
    vs_values = []
    thicknesses = []
    
    for layer in crust_layers:
        if layer in crust_profile:
            # crust_profile[layer] = [vp, vs, rho, thickness, top]
            layer_data = crust_profile[layer]
            vp_values.append(layer_data[0])
            vs_values.append(layer_data[1])
            thicknesses.append(layer_data[3])  # thickness in km
    
    if vp_values and vs_values:
        if weighted and thicknesses and sum(thicknesses) > 0:
            # Thickness-weighted average
            total_thickness = sum(thicknesses)
            vp = sum(v * t for v, t in zip(vp_values, thicknesses)) / total_thickness
            vs = sum(v * t for v, t in zip(vs_values, thicknesses)) / total_thickness
        else:
            # Simple arithmetic mean
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
    
    Parameters
    ----------
    distance : float
        Epicentral distance (km)
    velocity : float
        Wave velocity (km/s)
    origin_time : float, optional
        Event origin time (s)
        
    Returns
    -------
    arrival_time : float
        Theoretical arrival time (s)
    """
    return origin_time + distance / velocity


# ============================================================================
# LEVEL 2: DATAFRAME APPLICATORS
# ============================================================================

def add_crustal_velocities(df_stations, 
                          lat_col='STATION_LATITUDE_DEGREE',
                          lon_col='STATION_LONGITUDE_DEGREE'):
    """
    Add crustal velocity columns to station DataFrame.
    
    Queries CRUST1.0 for each station and adds:
    - 'vp_crust': P-wave velocity (km/s)
    - 'vs_crust': S-wave velocity (km/s)
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata from prepare_station_metadata()
    lat_col : str, optional
        Latitude column name (default: 'STATION_LATITUDE_DEGREE')
    lon_col : str, optional
        Longitude column name (default: 'STATION_LONGITUDE_DEGREE')
        
    Returns
    -------
    df_result : pd.DataFrame
        Copy with added velocity columns
        
    Examples
    --------
    >>> df_stations = prepare_station_metadata(df_meta_clean)
    >>> df_stations = add_crustal_velocities(df_stations)
    >>> print(df_stations[['STATION_CODE', 'vp_crust', 'vs_crust']])
    """
    if lat_col not in df_stations.columns:
        raise ValueError(f"Column '{lat_col}' not found in DataFrame")
    if lon_col not in df_stations.columns:
        raise ValueError(f"Column '{lon_col}' not found in DataFrame")
    
    print("Loading CRUST1.0 model...")
    model = crustModel()
    
    print(f"Querying {len(df_stations)} stations...")
    
    vp_list = []
    vs_list = []
    
    for idx, row in df_stations.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        profile = model.get_point(lat, lon)
        vp, vs = extract_crustal_velocities(profile)
        
        vp_list.append(vp)
        vs_list.append(vs)
    
    df_result = df_stations.copy()
    df_result['vp_crust'] = vp_list
    df_result['vs_crust'] = vs_list
    
    print(f"Added vp_crust and vs_crust columns")
    print(f"v_P: min={min(vp_list):.2f}, max={max(vp_list):.2f}, "
          f"median={np.median(vp_list):.2f} km/s")
    print(f"v_S: min={min(vs_list):.2f}, max={max(vs_list):.2f}, "
          f"median={np.median(vs_list):.2f} km/s")
    
    return df_result


def add_theoretical_arrivals(df_stations, origin_time=0, 
                            distance_col='EPICENTRAL_DISTANCE_KM'):
    """
    Add theoretical P and S arrival time columns.
    
    Adds columns:
    - 't_p_theo': Theoretical P-wave arrival time (s)
    - 't_s_theo': Theoretical S-wave arrival time (s)
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with distance and velocity columns
    origin_time : float, optional
        Event origin time (s)
    distance_col : str, optional
        Distance column name (default: 'EPICENTRAL_DISTANCE_KM')
        
    Returns
    -------
    df_result : pd.DataFrame
        Copy with added theoretical arrival time columns
        
    Examples
    --------
    >>> df_stations = add_crustal_velocities(df_stations)
    >>> df_stations = add_theoretical_arrivals(df_stations)
    >>> print(df_stations[['STATION_CODE', 't_p_theo', 't_s_theo']])
    """
    required_cols = [distance_col, 'vp_crust', 'vs_crust']
    missing_cols = [col for col in required_cols if col not in df_stations.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Run add_crustal_velocities() first."
        )
    
    df_result = df_stations.copy()
    df_result['t_p_theo'] = origin_time + df_result[distance_col] / df_result['vp_crust']
    df_result['t_s_theo'] = origin_time + df_result[distance_col] / df_result['vs_crust']
    
    print(f"Added theoretical arrival times")
    print(f"t_P: {df_result['t_p_theo'].min():.2f} - {df_result['t_p_theo'].max():.2f} s")
    print(f"t_S: {df_result['t_s_theo'].min():.2f} - {df_result['t_s_theo'].max():.2f} s")
    
    return df_result
