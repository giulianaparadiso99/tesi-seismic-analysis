"""
Visualization module for seismic signal analysis.

Provides plotting utilities organized by analysis stage:
    - plot_settings: Global matplotlib style configuration
    - plots_metadata: Metadata exploration and correlation analysis
    - plots_signals: Signal waveforms, distributions, and validation
    - plots_segmentation: Onset detection and window segmentation results
    - plots_moment_scaling: Moment scaling analysis and exponent plots

All plotting functions support optional output paths for saving figures
and use consistent styling via plot_settings.
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
    plot_pga_correlation_by_group,
)

from .plots_signals import (
    plot_signal_length_distribution,
    plot_station_waveforms,
    plot_signals_distributions,
    plot_postcheck_pdf,
    plot_postcheck_moment_scaling,
    plot_empirical_pdfs,
)

from .plots_segmentation import (
    display_theoretical_arrivals_table,
    plot_apparent_vs_crustal_velocities,
    plot_crustal_velocities_vs_distance,
    plot_theoretical_arrivals,
    plot_onset_detection_results,
    plot_onset_detection_results_v2,
    plot_coda_onset_results,
    plot_coda_scatter_comparison,
    plot_bland_altman_comparison,
    plot_residuals_vs_distance,
    plot_pairwise_difference_histograms,
    plot_correlation_matrix_heatmap,
    get_station_components,
    plot_station_windows,
    plot_multiple_stations,
    plot_window_comparison,
)

from .plots_moment_scaling import (
    plot_scaling_curves,
    plot_scaling_exponents,
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
    'plot_station_waveforms',
    'plot_signals_distributions',
    'plot_postcheck_pdf',
    'plot_postcheck_moment_scaling',
    'plot_empirical_pdfs',
    
    # Segmentation plots
    'display_theoretical_arrivals_table',
    'plot_apparent_vs_crustal_velocities',
    'plot_crustal_velocities_vs_distance',
    'plot_theoretical_arrivals',
    'plot_onset_detection_results',
    'plot_onset_detection_results_v2',
    'plot_coda_onset_results',
    'plot_coda_scatter_comparison',
    'plot_bland_altman_comparison',
    'plot_residuals_vs_distance',
    'plot_pairwise_difference_histograms',
    'plot_correlation_matrix_heatmap',
    'get_station_components',
    'plot_station_windows',
    'plot_multiple_stations',
    'plot_window_comparison',
    
    # Moment scaling plots
    'plot_scaling_curves',
    'plot_scaling_exponents',
]