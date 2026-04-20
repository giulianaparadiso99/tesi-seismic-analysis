"""
Visualization module for seismic data.

Includes plotting utilities for metadata, signals, and analysis results.

Modules:
    plot_settings - Global matplotlib style configuration
    plots_metadata - Metadata visualization (maps, distributions, correlations)
    plots_signals - Signal visualization (waveforms, PDFs, spectrograms)
"""

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
    plot_three_components,
    plot_acceleration_distributions,
    plot_postcheck_pdf,
    plot_postcheck_moment_scaling,
    plot_empirical_pdfs,
    plot_onset_diagnostic,
    plot_onset_distribution
)

from .plots_segmentation import (
    display_theoretical_arrivals_table,
    plot_theoretical_arrivals,
    plot_onset_detection_results,
    plot_onset_detection_results,
    plot_coda_scatter_comparison,
    plot_bland_altman_comparison,
    plot_residuals_vs_distance,
    plot_pairwise_difference_histograms,
    plot_correlation_matrix_heatmap,
    plot_station_windows,
    plot_multiple_stations,
    plot_window_comparison
)

__all__ = [
    # Settings
    'set_plot_style',
    
    # Metadata plots
    'plot_column_types_pie',
    'plot_numerical_distributions',
    'plot_categorical_distributions',
    'plot_correlation_matrix',
    'plot_significant_corr_diff',
    'plot_station_map',
    'plot_pga_and_duration_by_component',
    'plot_pga_correlation_by_group',
    
    # Signal plots
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
]
