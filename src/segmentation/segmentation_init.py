"""
Seismic signal segmentation and phase detection module.

This module provides a comprehensive pipeline for detecting seismic phases
(P-wave, S-wave, coda) and segmenting signals into temporal windows for
subsequent analysis.

Submodules
----------
search_windows : Crustal velocity models and search window calculation
    - CRUST1.0 velocity extraction
    - Theoretical arrival time prediction
    - Adaptive search window sizing based on distance
    
onset_detection : AR-AIC phase detection
    - P and S onset detection using Autoregressive-AIC method
    - Coda onset detection (Rautian, Arias, Envelope methods)
    - Method comparison and validation statistics
    
phasenet_utils : PhaseNet deep-learning phase picker
    - SeisBench/PhaseNet integration
    - Batch processing utilities
    - Coordinate conversion and metadata merging
    
window_segmentation : Signal windowing
    - Four-window segmentation (pre-event, P-wave, S-wave, coda)
    - Dual representation (samples and seconds)
    - Batch processing for multiple stations
    
window_validation : Quality control
    - Peak timing validation (should occur in S-wave window)
    - Monotonicity checks (arrivals ordered by distance)
    - Signal-to-noise ratio (SNR) validation
    - Comprehensive reporting and visualization

Typical Workflow
----------------
1. Calculate crustal velocities and theoretical arrivals:
   >>> df_stations = add_crustal_velocities(df_stations, hypo_depth_km=10.4)
   >>> df_stations = add_theoretical_arrivals(df_stations, hypo_depth_km=10.4)

2. Define adaptive search windows:
   >>> thresholds = calculate_distance_thresholds(df_stations)
   >>> df_stations = calculate_adaptive_windows(df_stations, thresholds)

3. Detect onsets (AR-AIC or PhaseNet):
   >>> df_stations = detect_onsets_arpick(signals_dict, df_stations)
   >>> # OR
   >>> df_picks = apply_phasenet_to_signals(df_signals, model, 'acceleration')

4. Detect coda onset:
   >>> df_full = add_coda_onsets_to_dataframe(df_full, signals_dict)

5. Segment signals into windows:
   >>> windowed = segment_all_signals(signals_dict, df_full, coda_method='rautian')

6. Quality control:
   >>> qc_results = quality_control_all_stations(
   ...     windowed, df_full, df_stations,
   ...     peak_column='PGA_CM/S^2',
   ...     time_peak_column='TIME_PGA_S'
   ... )
   >>> print_quality_control_summary(qc_results)

References
----------
Rautian, T. G., & Khalturin, V. I. (1978). "The use of the coda for
    determination of the earthquake source spectrum." BSSA, 68(4), 923-948.
Zhu, W., & Beroza, G. C. (2019). "PhaseNet: a deep-neural-network-based
    seismic arrival-time picking method." GJI, 216(1), 261-273.
Laske, G., et al. (2013). "Update on CRUST1.0 - A 1-degree global model
    of Earth's crust." Geophysical Research Abstracts, 15, EGU2013-2658.
"""

# Search windows and theoretical arrivals
from .search_windows import (
    extract_crustal_velocities,
    add_crustal_velocities,
    add_hypocentral_distance,
    add_theoretical_arrivals,
    calculate_distance_thresholds,
    calculate_search_windows,
    calculate_adaptive_windows
)

# AR-AIC phase detection
from .onset_detection import (
    detect_onsets_arpick,
    detect_coda_start,
    detect_coda_start_all_methods,
    add_coda_onsets_to_dataframe,
    compute_coda_method_statistics,
    find_coda_end
)

# PhaseNet phase detection
from .phasenet_utils import (
    get_station_from_filename, 
    get_component_from_filename,
    create_obspy_stream_from_dataframe,
    process_single_station_phasenet,
    convert_onset_coordinates,
    apply_phasenet_to_signals,
    merge_phasenet_picks_with_metadata
)

# Signal windowing
from .window_segmentation import (
    segment_signal_into_windows,
    segment_all_signals,
    get_window_statistics
)

# Quality control
from .window_validation import (
    check_peak_in_s_wave,
    check_monotonicity_station,
    check_snr,
    quality_control_all_stations,
    print_quality_control_summary,
    print_failed_checks,
    print_detailed_failures,
    analyze_monotonicity_violations,
    print_violation_summary,
    plot_monotonicity_analysis,
    analyze_residuals_vs_violations
)

__all__ = [
    # Search windows
    'extract_crustal_velocities',
    'add_crustal_velocities',
    'add_hypocentral_distance',
    'add_theoretical_arrivals',
    'calculate_distance_thresholds',
    'calculate_search_windows',
    'calculate_adaptive_windows',
    
    # AR-AIC detection
    'detect_onsets_arpick',
    'detect_coda_start',
    'detect_coda_start_all_methods',
    'add_coda_onsets_to_dataframe',
    'compute_coda_method_statistics',
    
    # PhaseNet detection
    'get_station_from_filename',
    'get_component_from_filename',
    'create_obspy_stream_from_dataframe',
    'process_single_station_phasenet',
    'convert_onset_coordinates',
    'apply_phasenet_to_signals',
    'merge_phasenet_picks_with_metadata',
    
    # Windowing
    'segment_signal_into_windows',
    'segment_all_signals',
    'get_window_statistics',
    
    # Quality control
    'check_peak_in_s_wave',
    'check_monotonicity_station',
    'check_snr',
    'quality_control_all_stations',
    'print_quality_control_summary',
    'print_failed_checks',
    'print_detailed_failures',
    'analyze_monotonicity_violations',
    'print_violation_summary',
    'plot_monotonicity_analysis',
    'analyze_residuals_vs_violations',
]