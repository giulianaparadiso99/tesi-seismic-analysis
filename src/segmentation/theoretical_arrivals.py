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

def add_theoretical_arrivals(df_stations, distance_col='EPICENTRAL_DISTANCE_KM'):
    """
    Add theoretical P and S arrival time columns.
    
    Calculates origin_time from EVENT_DATE and DATE_TIME_FIRST_SAMPLE,
    then adds travel time using simple kinematics: t = origin_time + distance / velocity
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with crustal velocities
    distance_col : str, optional
        Column name for epicentral distance (default: 'EPICENTRAL_DISTANCE_KM')
    
    Returns
    -------
    pd.DataFrame
        df_stations with added columns:
        - origin_time: Event time in file coordinates (s)
        - t_p_theo: Theoretical P-wave arrival (s)
        - t_s_theo: Theoretical S-wave arrival (s)
    """
    required_cols = [distance_col, 'vp_crust', 'vs_crust', 
                     'EVENT_DATE', 'DATE_TIME_FIRST_SAMPLE']
    missing = [col for col in required_cols if col not in df_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run add_crustal_velocities() first."
        )
    
    df_result = df_stations.copy()
    
    # Calculate origin_time: seconds from file start to event
    event_datetime = pd.to_datetime(df_result['EVENT_DATE'])
    first_sample_datetime = pd.to_datetime(df_result['DATE_TIME_FIRST_SAMPLE'])
    origin_time = (event_datetime - first_sample_datetime).dt.total_seconds()
    
    # Save origin_time as column
    df_result['origin_time'] = origin_time
    
    # Calculate theoretical arrivals: t = origin_time + distance / velocity
    df_result['t_p_theo'] = origin_time + df_result[distance_col] / df_result['vp_crust']
    df_result['t_s_theo'] = origin_time + df_result[distance_col] / df_result['vs_crust']
    
    print(f"Added theoretical arrival times:")
    print(f"  Origin time range: {origin_time.min():.2f} - {origin_time.max():.2f} s")
    print(f"  t_P range: {df_result['t_p_theo'].min():.2f} - {df_result['t_p_theo'].max():.2f} s")
    print(f"  t_S range: {df_result['t_s_theo'].min():.2f} - {df_result['t_s_theo'].max():.2f} s")
    
    return df_result

def calculate_search_windows(df_stations, 
                            p_window_before=5, 
                            p_window_after=5,
                            s_window_before=7, 
                            s_window_after=7):
    """
    Calculate search windows for P and S onset detection around theoretical arrival times.
    
    Windows are defined as symmetric intervals around theoretical arrivals,
    accounting for uncertainties in velocity model (CRUST1.0) and event location.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with columns:
        - 't_p_theo': Theoretical P-wave arrival time (s)
        - 't_s_theo': Theoretical S-wave arrival time (s)
    p_window_before : float, optional
        Time before t_p_theo to start search window (default: 15s)
    p_window_after : float, optional
        Time after t_p_theo to end search window (default: 15s)
    s_window_before : float, optional
        Time before t_s_theo to start search window (default: 20s)
    s_window_after : float, optional
        Time after t_s_theo to end search window (default: 20s)
    
    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - 'p_window_start': Start of P-wave search window (s)
        - 'p_window_end': End of P-wave search window (s)
        - 's_window_start': Start of S-wave search window (s)
        - 's_window_end': End of S-wave search window (s)
    
    Notes
    -----
    Search windows account for:
    - CRUST1.0 velocity model uncertainties (~5-10%)
    - Event location uncertainties (typically ±2 km)
    - Lateral crustal heterogeneities
    
    Conservative window sizes (±15s for P, ±20s for S) are recommended
    for regional distances 5-110 km with purely theoretical arrival times.
    
    Examples
    --------
    >>> df_stations = add_theoretical_arrivals(df_stations)
    >>> df_stations = calculate_search_windows(df_stations)
    >>> print(df_stations[['STATION_CODE', 'p_window_start', 'p_window_end']])
    """
    required_cols = ['t_p_theo', 't_s_theo']
    missing = [col for col in required_cols if col not in df_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run add_theoretical_arrivals() first."
        )
    
    df_result = df_stations.copy()
    
    # P-wave search windows
    df_result['p_window_start'] = df_result['t_p_theo'] - p_window_before
    df_result['p_window_end'] = df_result['t_p_theo'] + p_window_after
    
    # S-wave search windows
    df_result['s_window_start'] = df_result['t_s_theo'] - s_window_before
    df_result['s_window_end'] = df_result['t_s_theo'] + s_window_after
    
    # Ensure non-negative start times
    df_result['p_window_start'] = df_result['p_window_start'].clip(lower=0)
    df_result['s_window_start'] = df_result['s_window_start'].clip(lower=0)
    
    print(f"Search windows calculated:")
    print(f"  P-wave: [{-p_window_before}, +{p_window_after}]s around t_p_theo "
          f"(total width: {p_window_before + p_window_after}s)")
    print(f"  S-wave: [{-s_window_before}, +{s_window_after}]s around t_s_theo "
          f"(total width: {s_window_before + s_window_after}s)")
    
    # Summary statistics
    print(f"\nP-wave windows:")
    print(f"  Start: {df_result['p_window_start'].min():.2f} - {df_result['p_window_start'].max():.2f} s")
    print(f"  End: {df_result['p_window_end'].min():.2f} - {df_result['p_window_end'].max():.2f} s")
    
    print(f"\nS-wave windows:")
    print(f"  Start: {df_result['s_window_start'].min():.2f} - {df_result['s_window_start'].max():.2f} s")
    print(f"  End: {df_result['s_window_end'].min():.2f} - {df_result['s_window_end'].max():.2f} s")
    
    return df_result

def calculate_adaptive_windows(df_stations):
    """
    Calculate adaptive search windows that don't overlap and scale with distance.
    
    Windows adapt to epicentral distance (closer stations = narrower windows).
    S-window always starts after P-window ends to prevent overlap.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with columns:
        - 'EPICENTRAL_DISTANCE_KM': Distance from event to station (km)
        - 't_p_theo': Theoretical P-wave arrival time (s)
        - 't_s_theo': Theoretical S-wave arrival time (s)
    
    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - 'p_window_start': Start of P-wave search window (s)
        - 'p_window_end': End of P-wave search window (s)
        - 's_window_start': Start of S-wave search window (s)
        - 's_window_end': End of S-wave search window (s)
    
    Notes
    -----
    Window sizing strategy:
    - Distance < 50 km: ±3s for P, ±6s for S
    - Distance 50-150 km: ±5s for P, ±10s for S
    - Distance > 150 km: ±7s for P, ±14s for S
    
    S-window constraint: always starts at least 0.5s after P-window ends
    to ensure P and S detection regions are temporally separated.
    
    Examples
    --------
    >>> df_stations = add_theoretical_arrivals(df_stations)
    >>> df_stations = calculate_adaptive_windows(df_stations)
    >>> print(df_stations[['STATION_CODE', 'p_window_start', 's_window_start']])
    """
    required_cols = ['EPICENTRAL_DISTANCE_KM', 't_p_theo', 't_s_theo']
    missing = [col for col in required_cols if col not in df_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run add_theoretical_arrivals() first."
        )
    
    df_result = df_stations.copy()
    
    p_starts = []
    p_ends = []
    s_starts = []
    s_ends = []
    
    for idx, row in df_result.iterrows():
        dist = row['EPICENTRAL_DISTANCE_KM']
        t_p = row['t_p_theo']
        t_s = row['t_s_theo']
        
        # P window half-width based on distance
        if dist < 50:
            p_hw = 3.0
        elif dist < 150:
            p_hw = 5.0
        else:
            p_hw = 7.0
        
        # P window
        p_start = t_p - p_hw
        p_end = t_p + p_hw
        
        # S window: wider than P, starts AFTER P ends
        gap = 0.5  # separation between P and S windows
        s_hw = 2 * p_hw  # S window is twice as wide as P
        
        s_start_min = p_end + gap  # earliest S can start (after P ends)
        s_start_theo = t_s - s_hw  # theoretical S window start
        s_start = max(s_start_min, s_start_theo)  # take the later of the two
        s_end = t_s + s_hw
        
        p_starts.append(p_start)
        p_ends.append(p_end)
        s_starts.append(s_start)
        s_ends.append(s_end)
    
    df_result['p_window_start'] = p_starts
    df_result['p_window_end'] = p_ends
    df_result['s_window_start'] = s_starts
    df_result['s_window_end'] = s_ends
    
    # Ensure non-negative start times
    df_result['p_window_start'] = df_result['p_window_start'].clip(lower=0)
    df_result['s_window_start'] = df_result['s_window_start'].clip(lower=0)
    
    # Print summary
    print("Adaptive search windows calculated:")
    print(f"  Distance < 50 km: P ±3s, S ±6s")
    print(f"  Distance 50-150 km: P ±5s, S ±10s")
    print(f"  Distance > 150 km: P ±7s, S ±14s")
    print(f"  S-windows guaranteed to start ≥0.5s after P-windows end")
    
    # Summary statistics
    print(f"\nP-wave windows:")
    print(f"  Start: {df_result['p_window_start'].min():.2f} - {df_result['p_window_start'].max():.2f} s")
    print(f"  End: {df_result['p_window_end'].min():.2f} - {df_result['p_window_end'].max():.2f} s")
    
    print(f"\nS-wave windows:")
    print(f"  Start: {df_result['s_window_start'].min():.2f} - {df_result['s_window_start'].max():.2f} s")
    print(f"  End: {df_result['s_window_end'].min():.2f} - {df_result['s_window_end'].max():.2f} s")
    
    # Check for overlap (should be 0)
    overlaps = (df_result['s_window_start'] < df_result['p_window_end']).sum()
    print(f"\nP-S window overlaps: {overlaps}/22 stations")
    
    return df_result