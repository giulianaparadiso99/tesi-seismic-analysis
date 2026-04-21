"""
Analysis module for seismic signals.

Includes window detection (temporal phase separation), PDF analysis,
and moment scaling (both time-averaged and ensemble-averaged).

Modules:
    signals_pdf - Probability density function analysis
    window_detection - Temporal window detection (STA/LTA, ObsPy-compatible)
    signals_scaling - Moment scaling (time-averaged, sliding window)
    signals_scaling_ensemble - Moment scaling (ensemble-averaged, windowed)
"""

# PDF Analysis
from .signals_pdf import (
    gaussian_fit_analysis,
    heavy_tail_assessment
)

# Moment Scaling (Time-Averaged)
from .signals_scaling_temporal import (
    compute_temporal_ensemble_moments,
    compute_scaling_exponents_temporal,
    compute_ensemble_single_window_temporal
)

# Moment Scaling (Ensemble-Averaged)
from .signals_scaling_spatial import (
    analyze_all_windows,
    save_results_parquet
)

__all__ = [
    # PDF Analysis
    'gaussian_fit_analysis',
    'heavy_tail_assessment',
    
    # Window Detection
    'plot_signal_for_visual_inspection',
    'identify_windows_pga_based',
    'compute_sta_lta',
    'detect_onset_sta_lta',
    'identify_windows_sta_lta',
    'identify_windows_combined',
    'identify_windows_all_files',
    
    # Moment Scaling (Time)
    'compute_increments',
    'compute_moments_from_increments',
    'compute_moment_scaling',
    'compute_scaling_exponents',
    'test_scaling_linearity',
    'fit_piecewise_scaling',
    'build_scaling_summary',
    'trim_to_event_window',
    
    # Moment Scaling (Ensemble)
    'compute_increments_ensemble_windowed',
    'compute_increments_all_windows',
    'compute_moments_from_increments_ensemble',
    'compute_moments_all_windows',
    'compute_exponents_all_windows',
    'save_windowed_results',
    'load_windowed_results',
    'compute_and_save_all_windowed',
    'validate_moments_ensemble',
    'analyze_increments_ensemble',
]
