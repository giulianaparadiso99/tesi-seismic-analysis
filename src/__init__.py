"""
Seismic ground motion analysis package.

This package provides tools for preprocessing, analyzing, and visualizing
seismic acceleration signals, with focus on moment scaling and probability
density function analysis.

Modules:
    io - Data loading from .ASC archive
    cleaning_metadata - Metadata preprocessing pipeline
    cleaning_signals - Signal preprocessing pipeline
    signals_pdf - Probability density function analysis
    signals_scaling - Moment scaling and multifractal analysis
    signals_autocorrelation - Temporal correlation analysis
    plots_metadata - Metadata visualization
    plots_signals - Signal visualization
    plot_settings - Global matplotlib style configuration
    latex_export - Export results to LaTeX tables

Usage:
    from src import build_dataframes, clean_metadata, preprocess_signals
    
    df_meta, df_acc = build_dataframes('../data/raw/query.zip')
    df_meta_clean = clean_metadata(df_meta)
    df_acc_clean = preprocess_signals(df_acc, normalize=True)
"""

__version__ = "1.0.0"
__author__ = "Giuliana Paradiso"

# Data I/O
from .io import (
    build_metadata,
    build_accelerations,
    build_dataframes,
)

# Preprocessing
from .cleaning_metadata import clean_metadata
from .cleaning_signals import preprocess_signals, validate_preprocessing

# Analysis - PDF
from .signals_pdf import (
    gaussian_fit_analysis,
    heavy_tail_assessment
)

# Analysis - Moment Scaling
from .window_detection import (
    plot_signal_for_visual_inspection,
    identify_windows_pga_based,
    compute_sta_lta,
    detect_onset_sta_lta,
    identify_windows_sta_lta,
    identify_windows_combined,
    identify_windows_all_files,
    compute_and_save_all_windowed
)

from .signals_scaling import (
    integrate_to_velocity,
    integrate_to_displacement,
    compute_increments,
    compute_moments_from_increments,
    compute_moment_scaling,
    compute_scaling_exponents,
    test_scaling_linearity,
    fit_piecewise_scaling,
    build_scaling_summary,
    trim_to_event_window
)

from .signals_scaling_ensemble import (
    compute_increments_ensemble,
    compute_moments_from_increments_ensemble,
    validate_moments_ensemble,
    analyze_increments_ensemble
)


# Visualization
from .plot_settings import set_plot_style

from .plots_metadata import (
    plot_column_types_pie,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    plot_significant_corr_diff,
    plot_station_map,
    plot_pga_and_duration_by_component,
    plot_pga_correlation_by_group
)

from .plots_signals import (
    plot_signal_length_distribution,
    plot_example_signals,
    plot_acceleration_distributions,
    plot_postcheck_pdf,
    plot_postcheck_moment_scaling,
    plot_empirical_pdfs,
    plot_onset_diagnostic,
    plot_onset_distribution,
    plot_increments_histograms_dual_view,
    plot_ergodicity_test
    )

# Export
from .latex_export import (
    corr_diff_to_latex,
    preprocess_checks_to_latex,
    heavy_tail_to_latex,
)

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
    
    # Analysis - PDF
    'gaussian_fit_analysis',
    'heavy_tail_analysis',
    
    # Analysis - Moment Scaling
    'integrate_to_velocity',
    'integrate_to_displacement',
    'compute_increments',
    'compute_moments_from_increments',
    'compute_scaling_exponents',
    'test_scaling_linearity',
    'fit_piecewise_scaling',
    'build_scaling_summary',
    
    # Visualization
    'set_plot_style',
    
    # Export
    'corr_diff_to_latex',
    'preprocess_checks_to_latex',
    'heavy_tail_to_latex',
]