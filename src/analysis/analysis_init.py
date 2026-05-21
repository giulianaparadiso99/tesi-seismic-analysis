"""
Analysis module for seismic signals.

This module provides statistical analysis tools for seismic acceleration,
velocity, and displacement signals, including probability density function
characterization, moment scaling analysis (both temporal and spatial ensemble
averaging), and sensitivity analysis for phase detection uncertainty.

Submodules
----------
signals_pdf : Probability density function analysis
    - Gaussian fit analysis with Anderson-Darling test
    - Heavy-tail assessment (Laplace, Student-t, Lévy stable)
    - Power-law tail exponent estimation (Hill estimator)
    
signals_scaling_temporal : Time-averaged moment scaling
    - Sliding-window increments within single signals
    - Temporal ensemble averaging across time windows
    - Scaling exponent extraction with linear/piecewise fits
    
signals_scaling_spatial : Spatial ensemble-averaged moment scaling
    - Fixed reference time t₀, varying lag τ
    - Ensemble averaging across multiple stations/components
    - Four-window analysis (pre-event, P-wave, S-wave, coda)
    - Scaling spectrum ζ(q) for anomalous diffusion characterization
    
sensitivity : Monte Carlo sensitivity analysis
    - Gaussian and systematic perturbations of phase picks
    - Impact on windowing and derived metrics
    - Statistical quantification of uncertainty propagation

Typical Workflow
----------------
1. PDF Analysis (single signals):
   >>> from src.analysis import gaussian_fit_analysis, heavy_tail_assessment
   >>> df_gauss = gaussian_fit_analysis(df_clean, signal_column='acceleration')
   >>> df_tails = heavy_tail_assessment(df_clean, signal_column='acceleration')

2. Temporal Scaling (single signal, sliding window):
   >>> from src.analysis import compute_temporal_ensemble_moments
   >>> moments = compute_temporal_ensemble_moments(signal, tau_min=0.01, ...)

3. Spatial Scaling (ensemble across stations):
   >>> from src.analysis import analyze_all_windows, save_results_parquet
   >>> results = analyze_all_windows(windowed_signals, tau_min=0.01, ...)
   >>> save_results_parquet(results, output_dir='../data/processed/ensemble')

4. Sensitivity Analysis:
   >>> from src.analysis import run_sensitivity_analysis
   >>> sensitivity_results = run_sensitivity_analysis(
   ...     signals_dict, df_meta, n_iterations=100, perturbation_std=0.1
   ... )

References
----------
Vollmer, M., et al. (2024). "Moment scaling spectra reveal uniform
    liquefaction potential of tailing dams." Communications Earth & Environment.
Beck, C., & Cohen, E. G. D. (2003). "Superstatistics." Physica A, 322, 267-275.
"""

# PDF Analysis
from .signals_pdf import (
    gaussian_fit_analysis,
    heavy_tail_assessment
)

# Moment Scaling (Temporal - Time-Averaged)
from .signals_scaling_temporal import (
    compute_temporal_ensemble_moments,
    compute_scaling_exponents_temporal,
    compute_ensemble_single_window_temporal
)

# Moment Scaling (Spatial - Ensemble-Averaged)
from .signals_scaling_spatial import (
    analyze_all_windows,
    save_results_parquet,
    prepare_window_data,
    compute_moments_single_signal,
    compute_spatial_ensemble,
    extract_scaling_exponents,
    analyze_single_signal
)

# Sensitivity Analysis
from .sensitivity import (
    perturb_picks_gaussian,
    perturb_picks_bias,
    compute_sensitivity_metrics,
    compute_monte_carlo_statistics,
    create_sensitivity_summary,
    save_intermediate_results,
    run_sensitivity_analysis,
    create_summary
)

__all__ = [
    # PDF Analysis
    'gaussian_fit_analysis',
    'heavy_tail_assessment',
    
    # Moment Scaling (Temporal)
    'compute_temporal_ensemble_moments',
    'compute_scaling_exponents_temporal',
    'compute_ensemble_single_window_temporal',
    
    # Moment Scaling (Spatial Ensemble)
    'analyze_all_windows',
    'save_results_parquet',
    'prepare_window_data',
    'compute_moments_single_signal',
    'compute_spatial_ensemble',
    'extract_scaling_exponents',
    'analyze_single_signal',
    
    # Sensitivity Analysis
    'perturb_picks_gaussian',
    'perturb_picks_bias',
    'compute_sensitivity_metrics',
    'compute_monte_carlo_statistics',
    'create_sensitivity_summary',
    'save_intermediate_results',
    'run_sensitivity_analysis',
    'create_summary',
]

