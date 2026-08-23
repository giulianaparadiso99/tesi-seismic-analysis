"""
Preprocessing module for seismic signals.

Provides metadata cleaning and signal preprocessing functions for seismic
acceleration, velocity, and displacement data.

Submodules:
    cleaning_metadata   - Metadata preprocessing pipeline
    cleaning_signals    - Signal preprocessing (baseline, filtering, normalization)
    signals_integration - Integration (acceleration → velocity → displacement)
"""

from .cleaning_metadata import (
    clean_metadata,
    set_filter_band
)
from .cleaning_signals import (
    preprocess_signals,
    validate_preprocessing,
    resample_signal_to_target_rate,
    resample_mismatched_sampling_rate_signals,
    truncate_components_to_common_length,
    apply_filter_band_to_signals,
    apply_bandpass_filter
)

__all__ = [
    'clean_metadata',
    'preprocess_signals',
    'validate_preprocessing',
]