"""
Preprocessing module for seismic signals.

Provides metadata cleaning and signal preprocessing functions for seismic
acceleration, velocity, and displacement data.

Submodules:
    cleaning_metadata   - Metadata preprocessing pipeline
    cleaning_signals    - Signal preprocessing (baseline, filtering, normalization)
    signals_integration - Integration (acceleration → velocity → displacement)

Currently exposed at module level:
    clean_metadata - Full metadata cleaning pipeline
"""

from .cleaning_metadata import clean_metadata

from .cleaning_signals import (
    preprocess_signals,
    validate_preprocessing
)

__all__ = [
    # Metadata
    'clean_metadata',
    # Signals
    'preprocess_signals',
    'validate_preprocessing'
]
