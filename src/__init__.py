"""
Seismic ground motion analysis package.

This package provides tools for preprocessing, analyzing, and visualizing
seismic acceleration signals, with focus on moment scaling and probability
density function analysis.

Modules:
    preprocessing - Data loading, cleaning, and integration
    analysis - Window detection, PDF analysis, and moment scaling
    visualization - Plotting utilities and settings
    utils - Export and utility functions

Usage:
    from src import build_dataframes, clean_metadata, preprocess_signals
    
    df_meta, df_acc = build_dataframes('../data/raw/query.zip')
    df_meta_clean = clean_metadata(df_meta)
    df_acc_clean = preprocess_signals(df_acc, normalize=True)
"""

__version__ = "1.0.0"
__author__ = "Giuliana Paradiso"

# ===============================================================================
# Data I/O
# ===============================================================================

from .io.io import (
    build_metadata,
    build_accelerations,
    build_dataframes,
)

# ===============================================================================
# Preprocessing
# ===============================================================================

from .preprocessing.cleaning_metadata import clean_metadata

from .preprocessing.cleaning_signals import (
    preprocess_signals,
    validate_preprocessing
)

# ===============================================================================
# Processing
# ===============================================================================


from .processing.signals_integration import (
    integrate_to_velocity,
    integrate_to_displacement,
    validate_integration
)

from .processing.signal_conversion import (
    add_time_columns,
    get_station_from_filename,
    get_component_from_filename,
    convert_signals_to_dict,
    get_signal_for_station,
    validate_signals_dict,
    expand_to_component_level
)

# ===============================================================================
# Analysis - PDF
# ===============================================================================

from .analysis.signals_pdf import (
    gaussian_fit_analysis,
    heavy_tail_assessment
)

# ===============================================================================
# Analysis - Window Detection
# ===============================================================================

from .segmentation.theoretical_arrivals import (
    extract_crustal_velocities,
    calculate_theoretical_arrival,
    add_crustal_velocities,
    add_theoretical_arrivals,
    calculate_search_windows,
    calculate_adaptive_windows
)

from .segmentation.onset_detection import (
    detect_onsets_ar_windowed,
    detect_coda_start,
    detect_coda_start_all_methods
)

# ===============================================================================
# Analysis - Moment Scaling (Time-Averaged)
# ===============================================================================

from .analysis.signals_scaling import (
    compute_increments,
    compute_moments_from_increments,
    compute_moment_scaling,
    compute_scaling_exponents,
    test_scaling_linearity,
    fit_piecewise_scaling,
    build_scaling_summary,
    trim_to_event_window
)

# ===============================================================================
# Analysis - Moment Scaling (Ensemble-Averaged)
# ===============================================================================

from .analysis.signals_scaling_ensemble import (
    compute_increments_ensemble_windowed,
    compute_increments_all_windows,
    compute_moments_from_increments_ensemble,
    compute_moments_all_windows,
    compute_exponents_all_windows,
    save_windowed_results,
    load_windowed_results,
    compute_and_save_all_windowed,
    validate_moments_ensemble,
    analyze_increments_ensemble
)

# ===============================================================================
# Visualization
# ===============================================================================

from .visualization.plot_settings import set_plot_style

from .visualization.plots_metadata import (
    plot_column_types_pie,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    plot_significant_corr_diff,
    plot_station_map,
    plot_pga_and_duration_by_component,
    plot_pga_correlation_by_group
)

from .visualization.plots_signals import (
    plot_signal_length_distribution,
    plot_three_components,
    plot_acceleration_distributions,
    plot_postcheck_pdf,
    plot_postcheck_moment_scaling,
    plot_empirical_pdfs,
    plot_onset_diagnostic,
    plot_onset_distribution
)

from .visualization.plots_segmentation import (
    display_theoretical_arrivals_table,
    plot_theoretical_arrivals,
    plot_onset_detection_results,
    plot_coda_method_comparison
)


# ===============================================================================
# Export Utilities
# ===============================================================================

from .utils.latex_export import (
    corr_diff_to_latex,
    preprocess_checks_to_latex,
    heavy_tail_to_latex,
    metadata_table_to_latex,
    onset_detection_to_latex,
    coda_onset_comparison_to_latex
)

# ===============================================================================
# Exports
# ===============================================================================

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # I/O
    'build_metadata',
    'build_accelerations',
    'build_dataframes',
    
    # Preprocessing
    'clean_metadata',
    'preprocess_signals',
    'validate_preprocessing',
    'integrate_to_velocity',
    'integrate_to_displacement',
    'validate_integration',
    
    # Analysis - PDF
    'gaussian_fit_analysis',
    'heavy_tail_assessment',
    
    # Analysis - Window Detection
    'plot_signal_for_visual_inspection',
    'identify_windows_pga_based',
    'compute_sta_lta',
    'detect_onset_sta_lta',
    'identify_windows_sta_lta',
    'identify_windows_combined',
    'identify_windows_all_files',
    
    # Analysis - Moment Scaling (Time)
    'compute_increments',
    'compute_moments_from_increments',
    'compute_moment_scaling',
    'compute_scaling_exponents',
    'test_scaling_linearity',
    'fit_piecewise_scaling',
    'build_scaling_summary',
    'trim_to_event_window',
    
    # Analysis - Moment Scaling (Ensemble)
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
    
    # Visualization
    'set_plot_style',
    'plot_column_types_pie',
    'plot_numerical_distributions',
    'plot_categorical_distributions',
    'plot_correlation_matrix',
    'plot_significant_corr_diff',
    'plot_station_map',
    'plot_pga_and_duration_by_component',
    'plot_pga_correlation_by_group',
    'plot_signal_length_distribution',
    'plot_example_signals',
    'plot_acceleration_distributions',
    'plot_postcheck_pdf',
    'plot_postcheck_moment_scaling',
    'plot_empirical_pdfs',
    'plot_onset_diagnostic',
    'plot_onset_distribution',
    'plot_increments_histograms_dual_view',
    'plot_ergodicity_test',
    
    # Export
    'corr_diff_to_latex',
    'preprocess_checks_to_latex',
    'heavy_tail_to_latex',
]
