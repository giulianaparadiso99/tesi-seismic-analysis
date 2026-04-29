"""
Crustal velocity estimation - adapted for Ubaye 2024 dataset.

Column names from preprocessing:
- STATION_CODE: Station identifier
- STATION_LATITUDE_DEGREE: Station latitude
- STATION_LONGITUDE_DEGREE: Station longitude  
- EVENT_DEPTH_KM: Hypocentral distance

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

def extract_crustal_velocities(crust_profile, hypo_depth_km, weighted=True):
    """
    Extract average crustal velocities from CRUST1.0 profile.
    Automatically selects layers based on hypocenter depth to compute
    thickness-weighted average velocities for layers actually traversed
    by seismic waves traveling from hypocenter to surface.
    
    Parameters
    ----------
    crust_profile : dict
        Output from crustModel.get_point(lat, lon)
        Dictionary with layer names as keys and [vp, vs, rho, thickness, top]
        as values
    hypo_depth_km : float
        Hypocenter depth in kilometers (positive downward from surface)
    weighted : bool, optional
        If True, use thickness-weighted average (default)
        If False, use simple arithmetic mean
    
    Returns
    -------
    vp : float
        Average P-wave velocity (km/s)
    vs : float
        Average S-wave velocity (km/s)
    traversed_layers : list of str
        Names of layers actually traversed (for validation/logging)
    
    Notes
    -----
    CRUST1.0 layer format: [vp, vs, rho, thickness, top]
    - vp: P-wave velocity (km/s)
    - vs: S-wave velocity (km/s)
    - rho: density (g/cm³)
    - thickness: layer thickness (km)
    - top: depth to top of layer (km, negative below sea level)
    
    Layer ordering (surface to depth):
    upper_sediments, middle_sediments, lower_sediments,
    upper_crust, middle_crust, lower_crust
    
    The function computes which layers are traversed by a vertical ray
    from hypocenter depth to surface, then averages velocities only
    for those layers.
    
    Examples
    --------
    >>> from crust1 import crustModel
    >>> model = crustModel()
    >>> profile = model.get_point(44.5127, 6.8533)
    >>> vp, vs, layers = extract_crustal_velocities(profile, hypo_depth_km=10.4)
    >>> print(f"v_P = {vp:.2f} km/s, v_S = {vs:.2f} km/s")
    v_P = 5.83 km/s, v_S = 3.21 km/s
    >>> print(f"Traversed layers: {layers}")
    Traversed layers: ['upper_sediments', 'middle_sediments', 'lower_sediments', 'upper_crust']
    """
    # Layer names in order from surface to depth
    # Note: CRUST1.0 may use different naming conventions
    layer_order = [
        'upper_sediments', 'middle_sediments', 'lower_sediments',
        'upper_crust', 'middle_crust', 'lower_crust'
    ]
    
    # Alternative naming conventions in CRUST1.0
    layer_aliases = {
        'upper_sediments': ['upper_sediments', 'upper_seds.', 'soft_sed'],
        'middle_sediments': ['middle_sediments', 'middle_seds.', 'hard_sed'],
        'lower_sediments': ['lower_sediments', 'lower_seds.'],
        'upper_crust': ['upper_crust', 'upper.crust'],
        'middle_crust': ['middle_crust', 'middle.crust'],
        'lower_crust': ['lower_crust', 'lower.crust']
    }
    
    # Convert hypocenter depth to CRUST1.0 convention (negative below surface)
    hypo_depth_crust = -abs(hypo_depth_km)
    
    vp_values = []
    vs_values = []
    thicknesses = []
    traversed_layers = []
    
    for layer_name in layer_order:
        # Try to find layer with possible aliases
        layer_data = None
        actual_key = None
        
        for possible_name in layer_aliases.get(layer_name, [layer_name]):
            if possible_name in crust_profile:
                layer_data = crust_profile[possible_name]
                actual_key = possible_name
                break
        
        if layer_data is None:
            continue
        
        # Extract layer properties
        # layer_data = [vp, vs, rho, thickness, top]
        vp_layer = layer_data[0]
        vs_layer = layer_data[1]
        thickness = layer_data[3]
        top_depth = layer_data[4]  # depth to top of layer (km, negative)
        
        # Skip layers with zero thickness
        if thickness <= 0:
            continue
        
        # Calculate bottom depth of layer
        bottom_depth = top_depth - thickness
        
        # Check if this layer is traversed by vertical ray from hypocenter to surface
        # Layer is traversed if hypocenter is below layer bottom OR within layer
        if hypo_depth_crust <= top_depth:
            # Hypocenter is above this layer (closer to surface)
            # Ray traverses entire layer
            vp_values.append(vp_layer)
            vs_values.append(vs_layer)
            thicknesses.append(thickness)
            traversed_layers.append(actual_key)
        elif hypo_depth_crust > top_depth and hypo_depth_crust >= bottom_depth:
            # Hypocenter is within this layer
            # Ray traverses only portion from hypocenter to top of layer
            partial_thickness = abs(hypo_depth_crust - top_depth)
            vp_values.append(vp_layer)
            vs_values.append(vs_layer)
            thicknesses.append(partial_thickness)
            traversed_layers.append(f"{actual_key} (partial)")
            # Stop here - layers below are not traversed
            break
        else:
            # Hypocenter is below this layer - stop traversing
            break
    
    # Compute average velocities
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
        # This should rarely happen if CRUST1.0 data is complete
        vp = 6.0
        vs = 3.5
        traversed_layers = ['fallback_continental_crust']
    
    return vp, vs, traversed_layers


def add_crustal_velocities(df_stations, 
                          hypo_depth_km,
                          lat_col='STATION_LATITUDE_DEGREE',
                          lon_col='STATION_LONGITUDE_DEGREE'):
    """
    Add crustal velocity columns to station DataFrame.
    Queries CRUST1.0 for each station and adds:
    - 'vp_crust': P-wave velocity (km/s)
    - 'vs_crust': S-wave velocity (km/s)
    - 'traversed_layers': list of layer names used in calculation
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata from prepare_station_metadata()
    hypo_depth_km : float
        Hypocenter depth in kilometers (positive downward from surface)
    lat_col : str, optional
        Latitude column name (default: 'STATION_LATITUDE_DEGREE')
    lon_col : str, optional
        Longitude column name (default: 'STATION_LONGITUDE_DEGREE')
    
    Returns
    -------
    df_result : pd.DataFrame
        Copy with added velocity columns and traversed layers
    
    Examples
    --------
    >>> df_stations = prepare_station_metadata(df_meta_clean)
    >>> df_stations = add_crustal_velocities(df_stations, hypo_depth_km=10.4)
    >>> print(df_stations[['STATION_CODE', 'vp_crust', 'vs_crust']])
    """
    if lat_col not in df_stations.columns:
        raise ValueError(f"Column '{lat_col}' not found in DataFrame")
    if lon_col not in df_stations.columns:
        raise ValueError(f"Column '{lon_col}' not found in DataFrame")
    
    if hypo_depth_km <= 0:
        raise ValueError(f"hypo_depth_km must be positive, got {hypo_depth_km}")
    
    print("Loading CRUST1.0 model...")
    model = crustModel()
    
    print(f"Querying {len(df_stations)} stations...")
    print(f"Using hypocenter depth: {hypo_depth_km} km")
    
    vp_list = []
    vs_list = []
    layers_list = []
    
    for idx, row in df_stations.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        profile = model.get_point(lat, lon)
        vp, vs, traversed_layers = extract_crustal_velocities(
            profile, 
            hypo_depth_km=hypo_depth_km
        )
        
        vp_list.append(vp)
        vs_list.append(vs)
        layers_list.append(traversed_layers)
    
    df_result = df_stations.copy()
    df_result['vp_crust'] = vp_list
    df_result['vs_crust'] = vs_list
    df_result['traversed_layers'] = layers_list
    
    print(f"Added vp_crust, vs_crust, and traversed_layers columns")
    print(f"v_P: min={min(vp_list):.2f}, max={max(vp_list):.2f}, "
          f"median={np.median(vp_list):.2f} km/s")
    print(f"v_S: min={min(vs_list):.2f}, max={max(vs_list):.2f}, "
          f"median={np.median(vs_list):.2f} km/s")
    
    # Summary of layers used
    unique_layers_sets = set(tuple(layers) for layers in layers_list)
    print(f"\nUnique layer combinations used: {len(unique_layers_sets)}")
    for layer_set in unique_layers_sets:
        count = sum(1 for layers in layers_list if tuple(layers) == layer_set)
        print(f"  {count} stations: {list(layer_set)}")
    
    return df_result

def add_theoretical_arrivals(df_stations, 
                            hypo_depth_km, sampling_rate=200,
                            distance_col='EPICENTRAL_DISTANCE_KM'):
    """
    Add theoretical P and S arrival time columns.
    Calculates origin_time from EVENT_DATE and DATE_TIME_FIRST_SAMPLE,
    then adds travel time using simple kinematics with hypocentral distance:
    t = origin_time + distance_hypo / velocity
    
    where distance_hypo = sqrt(distance_epi^2 + hypo_depth^2)
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with crustal velocities
    hypo_depth_km : float
        Hypocenter depth in kilometers (positive downward from surface)
    distance_col : str, optional
        Column name for epicentral distance (default: 'EPICENTRAL_DISTANCE_KM')
    
    Returns
    -------
    pd.DataFrame
        df_stations with added columns:
        - origin_time: Event time in file coordinates (s)
        - hypocentral_distance_km: 3D distance from hypocenter to station (km)
        - t_p_theo_seconds: Theoretical P-wave arrival (s)
        - t_s_theo_seconds: Theoretical S-wave arrival (s)
        - t_p_theo_samples: Theoretical P-wave arrival (int)
        - t_s_theo_samples: Theoretical S-wave arrival (int)
    
    Notes
    -----
    Uses straight-ray approximation (no refraction).
    Hypocentral distance accounts for source depth:
    d_hypo = sqrt(d_epicentral^2 + depth^2)
    """
    required_cols = [distance_col, 'vp_crust', 'vs_crust', 
                     'EVENT_DATE', 'DATE_TIME_FIRST_SAMPLE']
    missing = [col for col in required_cols if col not in df_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run add_crustal_velocities() first."
        )
    
    if hypo_depth_km <= 0:
        raise ValueError(f"hypo_depth_km must be positive, got {hypo_depth_km}")
    
    df_result = df_stations.copy()
    
    # Calculate hypocentral distance (3D distance from hypocenter to station)
    epicentral_dist = df_result[distance_col]
    hypocentral_dist = np.sqrt(epicentral_dist**2 + hypo_depth_km**2)
    df_result['hypocentral_distance_km'] = hypocentral_dist
    
    # Calculate origin_time: seconds from file start to event
    event_datetime = pd.to_datetime(df_result['EVENT_DATE'])
    first_sample_datetime = pd.to_datetime(df_result['DATE_TIME_FIRST_SAMPLE'])
    origin_time = (event_datetime - first_sample_datetime).dt.total_seconds()
    
    # Save origin_time as column
    df_result['origin_time'] = origin_time
    
    # Calculate theoretical arrivals: t = origin_time + distance_hypo / velocity
    df_result['t_p_theo_seconds'] = origin_time + hypocentral_dist / df_result['vp_crust']
    df_result['t_s_theo_seconds'] = origin_time + hypocentral_dist / df_result['vs_crust']
    
    # Convert to samples
    df_result['t_p_theo_samples'] = np.round(df_result['t_p_theo_seconds'] * sampling_rate).astype(int)
    df_result['t_s_theo_samples'] = np.round(df_result['t_s_theo_seconds'] * sampling_rate).astype(int)

    print(f"Added theoretical arrival times:")
    print(f"Hypocenter depth: {hypo_depth_km} km")
    print(f"Epicentral distance range: {epicentral_dist.min():.2f} - {epicentral_dist.max():.2f} km")
    print(f"Hypocentral distance range: {hypocentral_dist.min():.2f} - {hypocentral_dist.max():.2f} km")
    print(f"Origin time range: {origin_time.min():.2f} - {origin_time.max():.2f} s")
    print(f"t_P range: {df_result['t_p_theo'].min():.2f} - {df_result['t_p_theo'].max():.2f} s")
    print(f"t_S range: {df_result['t_s_theo'].min():.2f} - {df_result['t_s_theo'].max():.2f} s")
    
    # Quantify the difference caused by depth correction
    distance_diff = hypocentral_dist - epicentral_dist
    print(f"\nDepth correction impact:")
    print(f"  Distance increase: {distance_diff.min():.2f} - {distance_diff.max():.2f} km")
    print(f"  Median distance increase: {distance_diff.median():.2f} km")
    
    return df_result

def calculate_search_windows(df_stations, 
                            p_window_before=5.0, 
                            p_window_after=5.0,
                            s_window_before=7.0, 
                            s_window_after=7.0):
    """
    Calculate fixed-width search windows for P and S onset detection.
    
    This is a baseline implementation using constant window sizes for all stations.
    For distance-adaptive windows, use calculate_adaptive_search_windows() instead.
    
    Windows are defined as symmetric intervals around theoretical arrival times,
    accounting for uncertainties in velocity model (CRUST1.0) and event location.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with columns:
        - 't_p_theo': Theoretical P-wave arrival time (s)
        - 't_s_theo': Theoretical S-wave arrival time (s)
    p_window_before : float, optional
        Time before t_p_theo to start search window (default: 5.0s)
    p_window_after : float, optional
        Time after t_p_theo to end search window (default: 5.0s)
    s_window_before : float, optional
        Time before t_s_theo to start search window (default: 7.0s)
    s_window_after : float, optional
        Time after t_s_theo to end search window (default: 7.0s)
    
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
    Fixed-width windows do not account for distance-dependent uncertainties.
    For regional distances (5-110 km), adaptive windows that scale with
    epicentral distance are recommended.
    
    Search window uncertainties arise from:
    - CRUST1.0 velocity model uncertainties (~5-10%)
    - Event location uncertainties (typically ±2 km horizontal, ±1 km vertical)
    - Lateral crustal heterogeneities
    - 1D velocity model assumption (ignores 3D structure)
    
    Default values (±5s for P, ±7s for S) are conservative estimates suitable
    for near-field stations (<30 km) with good event location quality.
    Larger windows may be needed for:
    - Distant stations (>50 km)
    - Poor event location quality
    - Complex crustal structure
    
    Examples
    --------
    >>> df_stations = add_theoretical_arrivals(df_stations, hypo_depth_km=10.4)
    >>> df_stations = calculate_search_windows(df_stations)
    >>> print(df_stations[['STATION_CODE', 'p_window_start', 'p_window_end']])
    
    >>> # Custom window sizes for distant stations
    >>> df_stations = calculate_search_windows(
    ...     df_stations, 
    ...     p_window_before=15.0, 
    ...     p_window_after=15.0
    ... )
    
    See Also
    --------
    calculate_adaptive_search_windows : Distance-dependent window sizing (recommended)
    """
    required_cols = ['t_p_theo', 't_s_theo']
    missing = [col for col in required_cols if col not in df_stations.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run add_theoretical_arrivals() first."
        )
    
    # Validate window parameters
    if any(param <= 0 for param in [p_window_before, p_window_after, 
                                     s_window_before, s_window_after]):
        raise ValueError("All window parameters must be positive")
    
    df_result = df_stations.copy()
    
    # P-wave search windows
    df_result['p_window_start'] = df_result['t_p_theo'] - p_window_before
    df_result['p_window_end'] = df_result['t_p_theo'] + p_window_after
    
    # S-wave search windows
    df_result['s_window_start'] = df_result['t_s_theo'] - s_window_before
    df_result['s_window_end'] = df_result['t_s_theo'] + s_window_after
    
    # Ensure non-negative start times (cannot search before recording starts)
    df_result['p_window_start'] = df_result['p_window_start'].clip(lower=0)
    df_result['s_window_start'] = df_result['s_window_start'].clip(lower=0)
    
    print(f"Fixed-width search windows calculated:")
    print(f"  P-wave: [-{p_window_before}, +{p_window_after}]s around t_p_theo "
          f"(total width: {p_window_before + p_window_after}s)")
    print(f"  S-wave: [-{s_window_before}, +{s_window_after}]s around t_s_theo "
          f"(total width: {s_window_before + s_window_after}s)")
    
    # Summary statistics
    print(f"\nP-wave windows:")
    print(f"  Start: {df_result['p_window_start'].min():.2f} - "
          f"{df_result['p_window_start'].max():.2f} s")
    print(f"  End: {df_result['p_window_end'].min():.2f} - "
          f"{df_result['p_window_end'].max():.2f} s")
    
    print(f"\nS-wave windows:")
    print(f"  Start: {df_result['s_window_start'].min():.2f} - "
          f"{df_result['s_window_start'].max():.2f} s")
    print(f"  End: {df_result['s_window_end'].min():.2f} - "
          f"{df_result['s_window_end'].max():.2f} s")
    
    # Warning if using fixed windows for large distance range
    distance_range = df_result['EPICENTRAL_DISTANCE_KM'].max() - df_result['EPICENTRAL_DISTANCE_KM'].min()
    if distance_range > 30:
        print(f"\nWarning: Large distance range ({distance_range:.1f} km). "
              f"Consider using adaptive windows instead.")
    
    return df_result

def calculate_distance_thresholds(df_stations, 
                                  distance_col='hypocentral_distance_km',
                                  method='tertiles',
                                  custom_quantiles=None):
    """
    Calculate distance thresholds for adaptive window sizing.
    
    Computes empirical distance cutoffs based on the distribution of 
    station distances, enabling data-driven adaptation of search windows.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with distance column
    distance_col : str, optional
        Column name for distance metric (default: 'hypocentral_distance_km')
        Can also be 'EPICENTRAL_DISTANCE_KM' for comparison
    method : str, optional
        Threshold calculation method (default: 'tertiles')
        - 'tertiles': Split into 3 equal groups (33rd, 67th percentiles)
        - 'quartiles': Split into 4 equal groups (25th, 50th, 75th percentiles)
        - 'custom': Use custom_quantiles parameter
    custom_quantiles : list of float, optional
        Custom quantile values between 0 and 1 (e.g., [0.25, 0.75])
        Only used when method='custom'
    
    Returns
    -------
    thresholds : list of float
        Distance thresholds in km, sorted in ascending order
        Length depends on method: tertiles→2 values, quartiles→3 values
    
    Notes
    -----
    Thresholds define bin edges for distance-based window adaptation:
    - For tertiles [t1, t2]: distances split into (0, t1], (t1, t2], (t2, ∞)
    - This creates 3 distance bins with equal number of stations
    
    The empirical approach ensures balanced representation across bins,
    avoiding empty or over-populated distance ranges.
    
    Examples
    --------
    >>> # Calculate tertile thresholds from hypocentral distances
    >>> thresholds = calculate_distance_thresholds(df_stations)
    >>> print(f"Tertile thresholds: {thresholds}")
    Tertile thresholds: [42.3, 78.5]
    
    >>> # Use quartiles instead
    >>> thresholds = calculate_distance_thresholds(df_stations, method='quartiles')
    >>> print(f"Quartile thresholds: {thresholds}")
    Quartile thresholds: [35.2, 58.7, 89.4]
    
    >>> # Compare epicentral vs hypocentral
    >>> epi_thresh = calculate_distance_thresholds(
    ...     df_stations, 
    ...     distance_col='EPICENTRAL_DISTANCE_KM'
    ... )
    >>> hypo_thresh = calculate_distance_thresholds(
    ...     df_stations,
    ...     distance_col='hypocentral_distance_km'
    ... )
    >>> print(f"Epicentral tertiles: {epi_thresh}")
    >>> print(f"Hypocentral tertiles: {hypo_thresh}")
    
    >>> # Custom quantiles (e.g., 30th and 70th percentiles)
    >>> thresholds = calculate_distance_thresholds(
    ...     df_stations,
    ...     method='custom',
    ...     custom_quantiles=[0.3, 0.7]
    ... )
    
    See Also
    --------
    calculate_adaptive_windows : Apply thresholds to compute search windows
    """
    if distance_col not in df_stations.columns:
        raise ValueError(
            f"Column '{distance_col}' not found in DataFrame. "
            f"Available columns: {list(df_stations.columns)}"
        )
    
    distances = df_stations[distance_col].values
    
    if len(distances) == 0:
        raise ValueError("DataFrame contains no stations")
    
    # Determine quantiles based on method
    if method == 'tertiles':
        quantiles = [1/3, 2/3]
    elif method == 'quartiles':
        quantiles = [0.25, 0.50, 0.75]
    elif method == 'custom':
        if custom_quantiles is None:
            raise ValueError("custom_quantiles must be provided when method='custom'")
        if not all(0 < q < 1 for q in custom_quantiles):
            raise ValueError("custom_quantiles must be between 0 and 1 (exclusive)")
        quantiles = sorted(custom_quantiles)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid options: 'tertiles', 'quartiles', 'custom'"
        )
    
    # Calculate thresholds
    thresholds = [np.percentile(distances, q * 100) for q in quantiles]
    
    # Validation: ensure thresholds are strictly increasing
    if not all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1)):
        raise ValueError(
            f"Thresholds are not strictly increasing: {thresholds}. "
            f"This may indicate insufficient distance variation in the dataset."
        )
    
    # Print summary
    print(f"Distance thresholds calculated using {method}:")
    print(f"  Distance column: {distance_col}")
    print(f"  Number of stations: {len(distances)}")
    print(f"  Distance range: {distances.min():.2f} - {distances.max():.2f} km")
    print(f"  Thresholds: {[f'{t:.2f}' for t in thresholds]} km")
    
    # Show bin populations
    bins = [0] + thresholds + [np.inf]
    print(f"\nDistance bins:")
    for i in range(len(bins) - 1):
        count = np.sum((distances > bins[i]) & (distances <= bins[i+1]))
        if bins[i+1] == np.inf:
            print(f"  Bin {i+1}: ({bins[i]:.2f}, ∞) km → {count} stations")
        else:
            print(f"  Bin {i+1}: ({bins[i]:.2f}, {bins[i+1]:.2f}] km → {count} stations")
    
    return thresholds


def calculate_adaptive_windows(df_stations,
                               distance_thresholds,
                               sampling_rate=200,
                               distance_col='hypocentral_distance_km',
                               unit='samples',
                               window_widths=None,
                               gap=0.5):
    """
    Calculate adaptive search windows that scale with distance.
    
    Windows adapt to station distance using empirical thresholds, with
    S-window guaranteed to start after P-window ends to prevent overlap.
    
    Parameters
    ----------
    df_stations : pd.DataFrame
        Station metadata with columns:
        - distance_col: Distance from hypocenter/epicenter to station (km)
        - 't_p_theo_samples', 't_s_theo_samples': Theoretical arrivals (int), OR
        - 't_p_theo_seconds', 't_s_theo_seconds': Theoretical arrivals (float), OR
        - 't_p_theo', 't_s_theo': Legacy columns (float, interpreted as seconds)
    distance_thresholds : list of float
        Distance cutoffs in km, in ascending order
        Example: [50, 150] creates 3 bins: (0,50], (50,150], (150,∞)
        Use calculate_distance_thresholds() to compute empirically
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
        Used for conversions between samples and seconds
    distance_col : str, optional
        Column name for distance metric (default: 'hypocentral_distance_km')
        Can also be 'EPICENTRAL_DISTANCE_KM'
    unit : {'samples', 'seconds'}, optional
        Default unit for computations (default: 'samples')
        Note: Both representations are ALWAYS computed and stored
    window_widths : list of tuple, optional
        (p_halfwidth, s_halfwidth) for each distance bin, in SECONDS
        Length must be len(distance_thresholds) + 1
        Default: [(3, 6), (5, 10), (7, 14)] for 2 thresholds
    gap : float, optional
        Minimum time separation between P and S windows in SECONDS (default: 0.5s)
        Ensures temporal separation of phase arrivals
    
    Returns
    -------
    df_stations : pd.DataFrame
        Copy with added columns (BOTH representations always present):
        - p_window_start_samples, p_window_end_samples (int)
        - s_window_start_samples, s_window_end_samples (int)
        - p_window_start_seconds, p_window_end_seconds (float)
        - s_window_start_seconds, s_window_end_seconds (float)
        
        Legacy columns (for backward compatibility):
        - p_window_start, p_window_end (alias for _samples if unit='samples', else _seconds)
        - s_window_start, s_window_end (alias for _samples if unit='samples', else _seconds)
    
    Notes
    -----
    Window sizing strategy:
    - Narrower windows for closer stations (lower model uncertainty)
    - Wider windows for distant stations (accumulated velocity errors)
    - S-windows are 2× wider than P-windows (S arrivals more emergent)
    
    Internal computation uses samples by default to avoid rounding artifacts.
    Seconds representation is always available for interpretability.
    
    S-window constraint:
    s_window_start = max(t_s_theo - s_halfwidth, p_window_end + gap)
    This ensures P and S detection regions never overlap.
    
    Examples
    --------
    >>> # Standard usage (samples-based, seconds also computed)
    >>> thresholds = calculate_distance_thresholds(df_stations)
    >>> df_stations = calculate_adaptive_windows(df_stations, thresholds)
    >>> print(df_stations[['STATION_CODE', 'p_window_start_samples', 'p_window_start_seconds']])
    
    >>> # Backward compatible (seconds-based)
    >>> df_stations = calculate_adaptive_windows(df_stations, thresholds, unit='seconds')
    >>> # Both _samples and _seconds columns are still created
    
    See Also
    --------
    calculate_distance_thresholds : Compute empirical distance thresholds
    add_theoretical_arrivals : Create theoretical arrival columns with dual representation
    """
    # Validate inputs
    required_base_cols = [distance_col]
    missing = [col for col in required_base_cols if col not in df_stations.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Determine which theo columns are available (samples preferred, seconds fallback)
    if 't_p_theo_samples' in df_stations.columns and 't_s_theo_samples' in df_stations.columns:
        t_p_col_samples = 't_p_theo_samples'
        t_s_col_samples = 't_s_theo_samples'
        has_samples = True
    else:
        has_samples = False
    
    if 't_p_theo_seconds' in df_stations.columns and 't_s_theo_seconds' in df_stations.columns:
        t_p_col_seconds = 't_p_theo_seconds'
        t_s_col_seconds = 't_s_theo_seconds'
        has_seconds = True
    elif 't_p_theo' in df_stations.columns and 't_s_theo' in df_stations.columns:
        # Legacy columns (assume seconds)
        t_p_col_seconds = 't_p_theo'
        t_s_col_seconds = 't_s_theo'
        has_seconds = True
    else:
        has_seconds = False
    
    if not has_samples and not has_seconds:
        raise ValueError(
            "No theoretical arrival columns found. "
            "Expected 't_p_theo_samples'/'t_s_theo_samples' or "
            "'t_p_theo_seconds'/'t_s_theo_seconds' or "
            "'t_p_theo'/'t_s_theo'. "
            "Run add_theoretical_arrivals() first."
        )
    
    # Validate distance thresholds
    if not isinstance(distance_thresholds, (list, tuple, np.ndarray)):
        raise TypeError("distance_thresholds must be a list, tuple, or array")
    
    distance_thresholds = list(distance_thresholds)
    
    if len(distance_thresholds) == 0:
        raise ValueError("distance_thresholds cannot be empty")
    
    if not all(distance_thresholds[i] < distance_thresholds[i+1] 
               for i in range(len(distance_thresholds)-1)):
        raise ValueError(
            f"distance_thresholds must be in strictly increasing order, "
            f"got {distance_thresholds}"
        )
    
    if any(t <= 0 for t in distance_thresholds):
        raise ValueError("All distance thresholds must be positive")
    
    # Number of bins = number of thresholds + 1
    n_bins = len(distance_thresholds) + 1
    
    # Set default window widths if not provided (in seconds)
    if window_widths is None:
        if n_bins == 3:
            window_widths = [(3.0, 6.0), (5.0, 10.0), (7.0, 14.0)]
        elif n_bins == 4:
            window_widths = [(3.0, 6.0), (4.0, 8.0), (6.0, 12.0), (8.0, 16.0)]
        else:
            window_widths = [(3.0 + 2*i, 6.0 + 4*i) for i in range(n_bins)]
            print(f"Warning: Using generic window widths for {n_bins} bins: {window_widths}")
    
    if len(window_widths) != n_bins:
        raise ValueError(
            f"window_widths length ({len(window_widths)}) must equal "
            f"number of distance bins ({n_bins} = len(thresholds) + 1)"
        )
    
    # Validate window widths
    for i, (p_hw, s_hw) in enumerate(window_widths):
        if p_hw <= 0 or s_hw <= 0:
            raise ValueError(f"All window halfwidths must be positive, got {window_widths[i]}")
        if s_hw < p_hw:
            raise ValueError(
                f"S halfwidth must be >= P halfwidth, got {window_widths[i]}. "
                f"S arrivals are typically more emergent and require wider windows."
            )
    
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    
    # Create result DataFrame
    df_result = df_stations.copy()
    
    # Assign stations to distance bins
    distances = df_result[distance_col].values
    bins = [0] + distance_thresholds + [np.inf]
    distance_bins = np.digitize(distances, bins[1:-1]) + 1
    df_result['distance_bin'] = distance_bins
    
    # Prepare lists for both representations
    p_starts_samples = []
    p_ends_samples = []
    s_starts_samples = []
    s_ends_samples = []
    
    p_starts_seconds = []
    p_ends_seconds = []
    s_starts_seconds = []
    s_ends_seconds = []
    
    # Calculate windows for each station
    for idx, row in df_result.iterrows():
        bin_idx = row['distance_bin'] - 1
        p_hw_sec, s_hw_sec = window_widths[bin_idx]
        
        # Convert halfwidths to samples
        p_hw_samp = int(np.round(p_hw_sec * sampling_rate))
        s_hw_samp = int(np.round(s_hw_sec * sampling_rate))
        gap_samp = int(np.round(gap * sampling_rate))
        
        # Get theoretical arrivals (prefer samples, fallback to seconds)
        if has_samples:
            t_p_samp = int(row[t_p_col_samples])
            t_s_samp = int(row[t_s_col_samples])
            
            # Compute in samples (PRIMARY)
            p_start_samp = max(0, t_p_samp - p_hw_samp)
            p_end_samp = t_p_samp + p_hw_samp
            
            s_start_min_samp = p_end_samp + gap_samp
            s_start_theo_samp = t_s_samp - s_hw_samp
            s_start_samp = max(s_start_min_samp, s_start_theo_samp, 0)
            s_end_samp = t_s_samp + s_hw_samp
            
            # Convert to seconds (DERIVED)
            p_start_sec = p_start_samp / sampling_rate
            p_end_sec = p_end_samp / sampling_rate
            s_start_sec = s_start_samp / sampling_rate
            s_end_sec = s_end_samp / sampling_rate
            
        else:
            # has_seconds is True
            t_p_sec = float(row[t_p_col_seconds])
            t_s_sec = float(row[t_s_col_seconds])
            
            # Compute in seconds (PRIMARY)
            p_start_sec = max(0.0, t_p_sec - p_hw_sec)
            p_end_sec = t_p_sec + p_hw_sec
            
            s_start_min_sec = p_end_sec + gap
            s_start_theo_sec = t_s_sec - s_hw_sec
            s_start_sec = max(s_start_min_sec, s_start_theo_sec, 0.0)
            s_end_sec = t_s_sec + s_hw_sec
            
            # Convert to samples (DERIVED)
            p_start_samp = int(np.round(p_start_sec * sampling_rate))
            p_end_samp = int(np.round(p_end_sec * sampling_rate))
            s_start_samp = int(np.round(s_start_sec * sampling_rate))
            s_end_samp = int(np.round(s_end_sec * sampling_rate))
        
        # Store BOTH representations
        p_starts_samples.append(p_start_samp)
        p_ends_samples.append(p_end_samp)
        s_starts_samples.append(s_start_samp)
        s_ends_samples.append(s_end_samp)
        
        p_starts_seconds.append(p_start_sec)
        p_ends_seconds.append(p_end_sec)
        s_starts_seconds.append(s_start_sec)
        s_ends_seconds.append(s_end_sec)
    
    # Assign all columns (both representations)
    df_result['p_window_start_samples'] = p_starts_samples
    df_result['p_window_end_samples'] = p_ends_samples
    df_result['s_window_start_samples'] = s_starts_samples
    df_result['s_window_end_samples'] = s_ends_samples
    
    df_result['p_window_start_seconds'] = p_starts_seconds
    df_result['p_window_end_seconds'] = p_ends_seconds
    df_result['s_window_start_seconds'] = s_starts_seconds
    df_result['s_window_end_seconds'] = s_ends_seconds
    
    # Legacy columns (for backward compatibility)
    if unit == 'samples':
        df_result['p_window_start'] = df_result['p_window_start_samples']
        df_result['p_window_end'] = df_result['p_window_end_samples']
        df_result['s_window_start'] = df_result['s_window_start_samples']
        df_result['s_window_end'] = df_result['s_window_end_samples']
    else:  # unit == 'seconds'
        df_result['p_window_start'] = df_result['p_window_start_seconds']
        df_result['p_window_end'] = df_result['p_window_end_seconds']
        df_result['s_window_start'] = df_result['s_window_start_seconds']
        df_result['s_window_end'] = df_result['s_window_end_seconds']
    
    # Print summary
    print(f"Adaptive search windows calculated:")
    print(f"  Distance metric: {distance_col}")
    print(f"  Number of bins: {n_bins}")
    print(f"  S-P gap: {gap}s ({int(np.round(gap * sampling_rate))} samples)")
    print(f"  Primary unit: {unit}")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"\nWindow sizing by distance bin:")
    
    for i in range(n_bins):
        p_hw_sec, s_hw_sec = window_widths[i]
        p_hw_samp = int(np.round(p_hw_sec * sampling_rate))
        s_hw_samp = int(np.round(s_hw_sec * sampling_rate))
        
        if i == 0:
            range_str = f"(0, {distance_thresholds[0]:.2f}]"
        elif i == n_bins - 1:
            range_str = f"({distance_thresholds[-1]:.2f}, ∞)"
        else:
            range_str = f"({distance_thresholds[i-1]:.2f}, {distance_thresholds[i]:.2f}]"
        
        n_stations = (df_result['distance_bin'] == i+1).sum()
        print(f"  Bin {i+1}: {range_str} km → "
              f"P ±{p_hw_sec}s ({p_hw_samp} samp), "
              f"S ±{s_hw_sec}s ({s_hw_samp} samp) "
              f"({n_stations} stations)")
    
    # Summary statistics (use seconds for display)
    print(f"\nP-wave windows:")
    print(f"  Start: {df_result['p_window_start_seconds'].min():.2f} - "
          f"{df_result['p_window_start_seconds'].max():.2f} s")
    print(f"  End: {df_result['p_window_end_seconds'].min():.2f} - "
          f"{df_result['p_window_end_seconds'].max():.2f} s")
    print(f"  Width: {(df_result['p_window_end_seconds'] - df_result['p_window_start_seconds']).min():.2f} - "
          f"{(df_result['p_window_end_seconds'] - df_result['p_window_start_seconds']).max():.2f} s")
    
    print(f"\nS-wave windows:")
    print(f"  Start: {df_result['s_window_start_seconds'].min():.2f} - "
          f"{df_result['s_window_start_seconds'].max():.2f} s")
    print(f"  End: {df_result['s_window_end_seconds'].min():.2f} - "
          f"{df_result['s_window_end_seconds'].max():.2f} s")
    print(f"  Width: {(df_result['s_window_end_seconds'] - df_result['s_window_start_seconds']).min():.2f} - "
          f"{(df_result['s_window_end_seconds'] - df_result['s_window_start_seconds']).max():.2f} s")
    
    # Check for P-S overlap (should be 0 by design)
    overlaps = (df_result['s_window_start_samples'] < df_result['p_window_end_samples']).sum()
    if overlaps > 0:
        print(f"\nWarning: {overlaps}/{len(df_result)} stations have P-S window overlap")
    else:
        print(f"\nP-S window separation verified: 0 overlaps")
    
    return df_result