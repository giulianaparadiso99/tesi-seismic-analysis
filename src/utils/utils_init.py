"""
Utility module for seismic analysis.

Provides LaTeX table export functions for analysis results, including
metadata tables, preprocessing checks, correlation analysis, phase detection,
and heavy-tail assessment.

Submodules:
    latex_export - LaTeX table generation and export functions
"""

from .latex_export import (
    corr_diff_to_latex,
    preprocess_checks_to_latex,
    heavy_tail_to_latex,
    metadata_table_to_latex,
    constant_fields_to_latex,
    onset_detection_to_latex,
    coda_onset_comparison_to_latex
)

__all__ = [
    'corr_diff_to_latex',
    'preprocess_checks_to_latex',
    'metadata_table_to_latex',
    'constant_fields_to_latex',
    'heavy_tail_to_latex',
    'onset_detection_to_latex',
    'coda_onset_comparison_to_latex',
]